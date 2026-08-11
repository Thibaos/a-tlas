use crate::{
    app::{GpuStack, RenderContext},
    rt::{acceleration_structure, closest_hit, intersection, miss, raygen},
    world::{
        chunk::Chunks,
        voxel::{get_palette, triangles_from_box},
    },
};
use dot_vox::DotVoxData;
use glam::{IVec3, Vec3};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use vulkano::{
    acceleration_structure::{AccelerationStructure, AccelerationStructureInstance},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
        Pipeline, PipelineShaderStageCreateInfo,
    },
    shader::EntryPoint,
    swapchain::Swapchain,
    sync::{AccessFlags, PipelineStages},
    DeviceSize,
};
use vulkano_taskgraph::{
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
    resource::HostAccessType,
    Id, Task, TaskContext, TaskResult,
};

/// Everything the ray pass needs that is independent of which raygen shader is
/// used. Built once per world; the production task and the validator's capture
/// task both construct from it, differing only in their raygen stage.
pub struct RenderResources {
    pub camera_buffer_id: Id<Buffer>,
    pub sunlight_buffer_id: Id<Buffer>,
    pub instance_buffer_id: Id<Buffer>,
    pub acceleration_structure_ids: [AccelerationStructureId; 2],
    pub camera_storage_buffer_id: StorageBufferId,
    pub palette_storage_buffer_id: StorageBufferId,
    pub sunlight_storage_buffer_id: StorageBufferId,
    pub blas: Arc<AccelerationStructure>,
    pub acceleration_structures: [Arc<AccelerationStructure>; 2],
    pub current_as_index: Arc<AtomicBool>,
}

pub fn create_render_resources(
    gpu: &GpuStack,
    world: &Arc<Chunks>,
    voxel_data: &DotVoxData,
    max_instance_count: u64,
) -> RenderResources {
    let vertices = triangles_from_box(Vec3::ZERO);
    let vertex_buffer = Buffer::from_iter(
        &gpu.memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .expect("Vertex buffer creation failed");

    let blas = acceleration_structure::build_blas(
        vertex_buffer,
        gpu.memory_allocator.clone(),
        gpu.device.clone(),
        gpu.compute_queue.clone(),
        &gpu.resources,
        gpu.compute_flight_id,
    );

    let render_instances: Vec<AccelerationStructureInstance> = world.to_instances(
        &IVec3::ZERO,
        blas.device_address().into(),
        max_instance_count,
        None,
    );
    let instance_buffer_id = gpu
        .resources
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
            DeviceLayout::new_unsized::<[AccelerationStructureInstance]>(max_instance_count)
                .unwrap(),
        )
        .unwrap();

    let tlas_count = render_instances.len().max(1) as u32;
    let instance_buffer =
        Subbuffer::new(gpu.resources.buffer(instance_buffer_id).buffer().clone())
            .cast_aligned::<AccelerationStructureInstance>();

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
            DeviceLayout::new_sized::<raygen::Palette>(),
        )
        .unwrap();

    unsafe {
        vulkano_taskgraph::execute(
            &gpu.transfer_queue,
            &gpu.resources,
            gpu.graphics_flight_id,
            |_cbf, tcx| {
                *tcx.write_buffer(palette_buffer_id, ..) = raygen::Palette { colors: palette };

                let write_instance_buffer =
                    tcx.write_buffer::<[AccelerationStructureInstance]>(instance_buffer_id, ..);

                for (dst, src) in write_instance_buffer.iter_mut().zip(render_instances) {
                    *dst = src;
                }

                Ok(())
            },
            [
                (palette_buffer_id, HostAccessType::Write),
                (instance_buffer_id, HostAccessType::Write),
            ],
            [],
            [],
        )
    }
    .unwrap();

    gpu.resources
        .flight(gpu.graphics_flight_id)
        .wait_idle()
        .unwrap();

    let acceleration_structures = [
        acceleration_structure::build_tlas(
            instance_buffer.clone(),
            tlas_count,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
            gpu.compute_queue.clone(),
            &gpu.resources,
            gpu.compute_flight_id,
        ),
        acceleration_structure::build_tlas(
            instance_buffer,
            tlas_count,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
            gpu.compute_queue.clone(),
            &gpu.resources,
            gpu.compute_flight_id,
        ),
    ];

    let bcx = gpu.resources.bindless_context().unwrap();

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
            DeviceLayout::new_sized::<raygen::Camera>(),
        )
        .unwrap();

    let sunlight_buffer_id = gpu
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
            DeviceLayout::new_sized::<raygen::Sunlight>(),
        )
        .unwrap();

    let acceleration_structure_ids = [
        bcx.global_set()
            .add_acceleration_structure(acceleration_structures[0].clone()),
        bcx.global_set()
            .add_acceleration_structure(acceleration_structures[1].clone()),
    ];

    let camera_storage_buffer_id = bcx
        .global_set()
        .create_storage_buffer(
            camera_buffer_id,
            0,
            Some(size_of::<raygen::Camera>() as DeviceSize),
        )
        .unwrap();


    let palette_storage_buffer_id = bcx
        .global_set()
        .create_storage_buffer(
            palette_buffer_id,
            0,
            Some(size_of::<raygen::Palette>() as DeviceSize),
        )
        .unwrap();

    let sunlight_storage_buffer_id = bcx
        .global_set()
        .create_storage_buffer(
            sunlight_buffer_id,
            0,
            Some(size_of::<raygen::Sunlight>() as DeviceSize),
        )
        .unwrap();

    RenderResources {
        camera_buffer_id,
        sunlight_buffer_id,
        instance_buffer_id,
        acceleration_structure_ids,
        camera_storage_buffer_id,
        palette_storage_buffer_id,
        sunlight_storage_buffer_id,
        blas,
        acceleration_structures,
        current_as_index: Arc::new(AtomicBool::new(false)),
    }
}

/// Builds the shared ray tracing pipeline (raygen + miss + procedural hit
/// group) around the given raygen entry point. The production task and the
/// validator's capture task differ only in which raygen they pass.
#[allow(clippy::too_many_arguments)]
pub fn build_ray_tracing_pipeline(
    gpu: &GpuStack,
    raygen: &EntryPoint,
    miss: &EntryPoint,
    intersection: &EntryPoint,
    closest_hit: &EntryPoint,
) -> Arc<RayTracingPipeline> {
    let bcx = gpu.resources.bindless_context().unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(raygen),
        PipelineShaderStageCreateInfo::new(miss),
        PipelineShaderStageCreateInfo::new(intersection),
        PipelineShaderStageCreateInfo::new(closest_hit),
    ];

    let groups = [
        RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
        RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
        RayTracingShaderGroupCreateInfo::ProceduralHit {
            closest_hit_shader: Some(3),
            any_hit_shader: None,
            intersection_shader: 2,
        },
    ];

    let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

    let base_info = RayTracingPipelineCreateInfo::new(&layout);

    RayTracingPipeline::new(
        &gpu.device,
        None,
        &RayTracingPipelineCreateInfo {
            stages: &stages,
            groups: &groups,
            max_pipeline_ray_recursion_depth: 1,
            ..base_info
        },
    )
    .unwrap()
}

pub struct RayTracingRenderTask {
    swapchain_id: Id<Swapchain>,
    pub acceleration_structure_ids: [AccelerationStructureId; 2],
    pub camera_buffer_id: Id<Buffer>,
    pub sunlight_buffer_id: Id<Buffer>,
    pub instance_buffer_id: Id<Buffer>,
    camera_storage_buffer_id: StorageBufferId,
    palette_storage_buffer_id: StorageBufferId,
    sunlight_storage_buffer_id: StorageBufferId,
    shader_binding_table: ShaderBindingTable,
    pub blas: Arc<AccelerationStructure>,
    pub acceleration_structures: [Arc<AccelerationStructure>; 2],
    pub current_as_index: Arc<AtomicBool>,
    pipeline: Arc<RayTracingPipeline>,
}

impl RayTracingRenderTask {
    pub fn new(
        gpu: &GpuStack,
        world: &Arc<Chunks>,
        voxel_data: &DotVoxData,
        virtual_swapchain_id: Id<Swapchain>,
        max_instance_count: u64,
    ) -> Self {
        let resources = create_render_resources(gpu, world, voxel_data, max_instance_count);

        let pipeline = {
            let raygen = unsafe {
                raygen::load(&gpu.device)
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
                intersection::load(&gpu.device)
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

        RayTracingRenderTask {
            swapchain_id: virtual_swapchain_id,
            camera_buffer_id: resources.camera_buffer_id,
            sunlight_buffer_id: resources.sunlight_buffer_id,
            instance_buffer_id: resources.instance_buffer_id,
            acceleration_structure_ids: resources.acceleration_structure_ids,
            camera_storage_buffer_id: resources.camera_storage_buffer_id,
            palette_storage_buffer_id: resources.palette_storage_buffer_id,
            sunlight_storage_buffer_id: resources.sunlight_storage_buffer_id,
            shader_binding_table,
            blas: resources.blas,
            acceleration_structures: resources.acceleration_structures,
            current_as_index: resources.current_as_index,
            pipeline,
        }
    }
}

impl Task for RayTracingRenderTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        unsafe { cbf.update_buffer(self.camera_buffer_id, 0, &rcx.rt_camera_data) };
        unsafe { cbf.update_buffer(self.sunlight_buffer_id, 0, &rcx.rt_sunlight_data) };

        // vkCmdUpdateBuffer's writes are not tracked by the taskgraph, so make
        // them visible to the ray pass explicitly. Without this barrier the
        // camera reads stale memory on early frames (the validator exposed it:
        // a one-shot frame read all zeros).
        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_access: AccessFlags::TRANSFER_WRITE,
                    dst_access: AccessFlags::SHADER_READ
                        | AccessFlags::SHADER_STORAGE_READ,
                    src_stages: PipelineStages::ALL_TRANSFER,
                    dst_stages: PipelineStages::RAY_TRACING_SHADER,
                    ..Default::default()
                }],
                ..Default::default()
            })
        };

        let front_index = self.current_as_index.load(Ordering::Acquire);

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &raygen::PushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    acceleration_structure_id: self.acceleration_structure_ids
                        [front_index as usize],
                    camera_buffer_id: self.camera_storage_buffer_id,
                    palette_buffer_id: self.palette_storage_buffer_id,
                    sunlight_buffer_id: self.sunlight_storage_buffer_id,
                    t_image_id: StorageImageId::INVALID,
                },
            )
        };

        unsafe {
            cbf.bind_pipeline_ray_tracing(&self.pipeline);
        }

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) };

        let dependency_info = DependencyInfo {
            memory_barriers: &[MemoryBarrier {
                src_stages: PipelineStages::LATE_FRAGMENT_TESTS,
                dst_stages: PipelineStages::EARLY_FRAGMENT_TESTS,
                src_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                ..Default::default()
            }],
            ..Default::default()
        };

        unsafe { cbf.pipeline_barrier(&dependency_info) };

        Ok(())
    }
}
