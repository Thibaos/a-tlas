//! The Region pipeline's GPU half (renderer-impl ticket 02): per-Region voxel
//! pool buffers + procedural AABB BLASes + one TLAS over the Region
//! instances, the shared ray tracing pipeline, and the per-frame render task
//! that ray-passes the swapchain storage images with the capture raygen
//! (color + t-channel for the validator).

use std::sync::Arc;

use dot_vox::DotVoxData;
use vulkano::{
    DeviceSize,
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureGeometries,
        AccelerationStructureGeometryAabbsData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        Pipeline,
        ray_tracing::{RayTracingPipeline, ShaderBindingTable},
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
    resource::HostAccessType,
};

use crate::{
    app::GpuStack,
    region::{REGION_COUNT, pack::pack_regions, snapshot::emit_snapshots},
    rt::acceleration_structure,
    tasks::render::build_ray_tracing_pipeline,
    world::{chunk::Chunks, voxel::get_palette},
};

pub(crate) mod capture_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/region/capture.rgen",
        vulkan_version: "1.3"
    }
}

pub(crate) mod intersect {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "intersection",
        path: "shaders/region/intersect.rint",
        vulkan_version: "1.3"
    }
}

pub(crate) mod miss {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "miss",
        path: "shaders/region/miss.rmiss",
        vulkan_version: "1.3"
    }
}

pub(crate) mod closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/region/closest_hit.rchit",
        vulkan_version: "1.3"
    }
}

/// The static Region path, built once per world from the snapshot batch:
/// snapshots → per-Region pools → procedural AABB BLASes → TLAS (ADR
/// 0001/0004), plus the camera, palette and Region-table buffers.
pub struct RegionPipeline {
    pub camera_buffer_id: Id<Buffer>,
    /// The TLAS instance buffer (one instance per Region).
    pub instance_buffer_id: Id<Buffer>,
    /// Kept alive here: the instances reference the BLASes by device address,
    /// which carries no lifetime.
    pub blases: Vec<Arc<AccelerationStructure>>,
    pub region_table_storage_id: StorageBufferId,
    pub camera_storage_id: StorageBufferId,
    pub palette_storage_id: StorageBufferId,
    pub acceleration_structure_id: AccelerationStructureId,
}

impl RegionPipeline {
    pub fn new(gpu: &GpuStack, world: &Arc<Chunks>, voxel_data: &DotVoxData) -> Self {
        let snapshots = emit_snapshots(world);
        let regions = pack_regions(&snapshots);

        // --- per-Region pool buffers + AABB buffers ---------------------
        let mut pool_buffer_ids = Vec::with_capacity(regions.len());
        let mut aabb_buffers = Vec::with_capacity(regions.len());

        for region in &regions {
            let pool_id = gpu
                .resources
                .create_buffer(
                    &BufferCreateInfo {
                        usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    DeviceLayout::new_unsized::<[u8]>(region.blocks.len() as u64).unwrap(),
                )
                .unwrap();
            pool_buffer_ids.push(pool_id);

            let aabb_buffer = Buffer::from_iter(
                &gpu.memory_allocator,
                &BufferCreateInfo {
                    usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                region.aabbs.iter().copied(),
            )
            .expect("AABB buffer creation failed");
            aabb_buffers.push(aabb_buffer);
        }

        // --- Region table: Region id → pool device address --------------
        let region_table_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::RegionTable>(),
            )
            .unwrap();

        let mut table_bdas = vec![0u64; REGION_COUNT];
        for (region, pool_id) in regions.iter().zip(&pool_buffer_ids) {
            let address = gpu
                .resources
                .buffer(*pool_id)
                .buffer()
                .device_address()
                .get();
            table_bdas[region.region_id() as usize] = address;
        }
        let region_table = capture_raygen::RegionTable {
            bdas: table_bdas
                .try_into()
                .expect("region table length must be REGION_COUNT"),
        };

        // --- camera + palette buffers -----------------------------------
        let camera_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::Camera>(),
            )
            .unwrap();

        let palette = get_palette(voxel_data).map(|color| [color.x, color.y, color.z, 1.0]);
        let palette_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::Palette>(),
            )
            .unwrap();

        // --- one-shot upload: table + pools + palette --------------------
        let mut upload_access = vec![
            (region_table_buffer_id, HostAccessType::Write),
            (palette_buffer_id, HostAccessType::Write),
        ];
        upload_access.extend(
            pool_buffer_ids
                .iter()
                .copied()
                .map(|id| (id, HostAccessType::Write)),
        );

        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<capture_raygen::RegionTable>(region_table_buffer_id, ..) =
                        region_table;
                    *tcx.write_buffer::<capture_raygen::Palette>(palette_buffer_id, ..) =
                        capture_raygen::Palette { colors: palette };

                    for (region, pool_id) in regions.iter().zip(&pool_buffer_ids) {
                        tcx.write_buffer::<[u8]>(*pool_id, ..)
                            .copy_from_slice(&region.blocks);
                    }

                    Ok(())
                },
                upload_access,
                [],
                [],
            )
        }
        .unwrap();

        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        // --- one procedural AABB BLAS per Region -------------------------
        let mut blases = Vec::with_capacity(regions.len());
        for aabb_buffer in &aabb_buffers {
            let aabb_data = AccelerationStructureGeometryAabbsData {
                data: Some(aabb_buffer.clone().into_bytes()),
                stride: size_of::<AabbPositions>() as u32,
                ..Default::default()
            };

            let blas = acceleration_structure::build_acceleration_structure_common(
                AccelerationStructureGeometries::Aabbs(vec![aabb_data]),
                BuildAccelerationStructureMode::Build,
                aabb_buffer.len() as u32,
                AccelerationStructureType::BottomLevel,
                gpu.memory_allocator.clone(),
                gpu.device.clone(),
                gpu.compute_queue.clone(),
                &gpu.resources,
                gpu.compute_flight_id,
            );
            blases.push(blas);
        }

        // --- TLAS: one lattice-static instance per Region -----------------
        debug_assert!(!regions.is_empty(), "the harness worlds always have voxels");
        let instances: Vec<AccelerationStructureInstance> = regions
            .iter()
            .zip(&blases)
            .map(|(region, blas)| {
                let origin = region.origin().as_vec3().to_array();
                AccelerationStructureInstance {
                    transform: [
                        [1.0, 0.0, 0.0, origin[0]],
                        [0.0, 1.0, 0.0, origin[1]],
                        [0.0, 0.0, 1.0, origin[2]],
                    ],
                    instance_custom_index_and_mask: vulkano::Packed24_8::new(
                        region.region_id(),
                        0xFF,
                    ),
                    acceleration_structure_reference: blas.device_address().into(),
                    ..Default::default()
                }
            })
            .collect();

        let instance_buffer_id =
            gpu.resources
                .create_buffer(
                    &BufferCreateInfo {
                        usage: BufferUsage::SHADER_DEVICE_ADDRESS
                            | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    DeviceLayout::new_unsized::<[AccelerationStructureInstance]>(
                        instances.len() as u64
                    )
                    .unwrap(),
                )
                .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    let dst =
                        tcx.write_buffer::<[AccelerationStructureInstance]>(instance_buffer_id, ..);
                    for (dst, src) in dst.iter_mut().zip(instances.iter().copied()) {
                        *dst = src;
                    }
                    Ok(())
                },
                [(instance_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        let instance_subbuffer =
            Subbuffer::new(gpu.resources.buffer(instance_buffer_id).buffer().clone())
                .cast_aligned::<AccelerationStructureInstance>();
        let tlas = acceleration_structure::build_tlas(
            instance_subbuffer,
            instances.len() as u32,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
            gpu.compute_queue.clone(),
            &gpu.resources,
            gpu.compute_flight_id,
        );

        // --- bindless registrations ---------------------------------------
        let bcx = gpu.resources.bindless_context().unwrap();

        let region_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                region_table_buffer_id,
                0,
                Some(size_of::<capture_raygen::RegionTable>() as DeviceSize),
            )
            .unwrap();

        let camera_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                camera_buffer_id,
                0,
                Some(size_of::<capture_raygen::Camera>() as DeviceSize),
            )
            .unwrap();

        let palette_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                palette_buffer_id,
                0,
                Some(size_of::<capture_raygen::Palette>() as DeviceSize),
            )
            .unwrap();

        let acceleration_structure_id = bcx.global_set().add_acceleration_structure(tlas.clone());

        Self {
            camera_buffer_id,
            instance_buffer_id,
            blases,
            region_table_storage_id,
            camera_storage_id,
            palette_storage_id,
            acceleration_structure_id,
        }
    }
}

/// The per-frame world the Region render task reads.
pub struct RegionRenderContext {
    pub camera: capture_raygen::Camera,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    /// Validation only: the capture raygen additionally writes payload.t
    /// here; the production raygen passes INVALID and never dereferences it.
    pub t_image_storage_id: StorageImageId,
}

pub struct RegionRenderTask {
    swapchain_id: Id<Swapchain>,
    camera_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    camera_storage_id: StorageBufferId,
    palette_storage_id: StorageBufferId,
    region_table_storage_id: StorageBufferId,
    acceleration_structure_id: AccelerationStructureId,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    /// Keeps the BLAS memory alive: the TLAS instances reference the BLASes
    /// by device address only.
    #[allow(dead_code)]
    blases: Vec<Arc<AccelerationStructure>>,
}

impl RegionRenderTask {
    pub fn new(
        gpu: &GpuStack,
        world: &Arc<Chunks>,
        voxel_data: &DotVoxData,
        virtual_swapchain_id: Id<Swapchain>,
    ) -> Self {
        let pipeline_resources = RegionPipeline::new(gpu, world, voxel_data);

        let pipeline = {
            let raygen = unsafe {
                capture_raygen::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };
            let miss = unsafe {
                miss::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };
            let intersection = unsafe {
                intersect::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };
            let closest_hit = unsafe {
                closest_hit::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };

            build_ray_tracing_pipeline(gpu, &raygen, &miss, &intersection, &closest_hit)
        };

        let shader_binding_table =
            ShaderBindingTable::new(&gpu.memory_allocator, &pipeline).unwrap();

        Self {
            swapchain_id: virtual_swapchain_id,
            camera_buffer_id: pipeline_resources.camera_buffer_id,
            instance_buffer_id: pipeline_resources.instance_buffer_id,
            camera_storage_id: pipeline_resources.camera_storage_id,
            palette_storage_id: pipeline_resources.palette_storage_id,
            region_table_storage_id: pipeline_resources.region_table_storage_id,
            acceleration_structure_id: pipeline_resources.acceleration_structure_id,
            shader_binding_table,
            pipeline,
            blases: pipeline_resources.blases,
        }
    }

    /// The instance buffer the TLAS reads (declared in the task graph).
    pub fn instance_buffer_id(&self) -> Id<Buffer> {
        self.instance_buffer_id
    }
}

impl Task for RegionRenderTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        unsafe { cbf.update_buffer(self.camera_buffer_id, 0, &rcx.camera) };

        // vkCmdUpdateBuffer's writes are not tracked by the taskgraph, so
        // make them visible to the ray pass explicitly.
        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                    dst_access: vulkano::sync::AccessFlags::SHADER_READ
                        | vulkano::sync::AccessFlags::SHADER_STORAGE_READ,
                    src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,
                    dst_stages: vulkano::sync::PipelineStages::RAY_TRACING_SHADER,
                    ..Default::default()
                }],
                ..Default::default()
            })
        };

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &capture_raygen::RegionPushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    t_image_id: rcx.t_image_storage_id,
                    acceleration_structure_id: self.acceleration_structure_id,
                    camera_buffer_id: self.camera_storage_id,
                    palette_buffer_id: self.palette_storage_id,
                    region_table_buffer_id: self.region_table_storage_id,
                },
            )
        };

        unsafe {
            cbf.bind_pipeline_ray_tracing(&self.pipeline);
        }

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) };

        Ok(())
    }
}
