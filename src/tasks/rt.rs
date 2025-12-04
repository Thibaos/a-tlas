use crate::{
    App, RenderContext,
    rt::{acceleration_structure, closest_hit, intersection, miss, raygen},
    world::voxel::{get_palette, triangles_from_box},
};
use glam::{IVec3, Vec3};
use std::sync::Arc;
use vulkano::{
    DeviceSize, Packed24_8,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildType, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureInstance,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        Pipeline, PipelineShaderStageCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    descriptor_set::{AccelerationStructureId, StorageBufferId},
    resource::HostAccessType,
};

pub struct RayTracingPass {
    swapchain_id: Id<Swapchain>,
    pub acceleration_structure_id: AccelerationStructureId,
    pub camera_buffer_id: Id<Buffer>,
    pub sunlight_buffer_id: Id<Buffer>,
    pub instance_buffer_id: Id<Buffer>,
    pub scratch_buffer_id: Id<Buffer>,
    camera_storage_buffer_id: StorageBufferId,
    palette_storage_buffer_id: StorageBufferId,
    sunlight_storage_buffer_id: StorageBufferId,
    shader_binding_table: ShaderBindingTable,
    pub blas: Arc<AccelerationStructure>,
    pub tlas: Arc<AccelerationStructure>,
    pipeline: Arc<RayTracingPipeline>,
}

impl RayTracingPass {
    pub fn new(app: &App, virtual_swapchain_id: Id<Swapchain>, max_instance_count: u64) -> Self {
        let vertices = triangles_from_box(Vec3::ZERO);
        let vertex_buffer = Buffer::from_iter(
            &app.memory_allocator,
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
            app.memory_allocator.clone(),
            app.device.clone(),
            app.compute_queue.clone(),
            &app.resources,
            app.compute_flight_id,
        );

        // let render_instances: Vec<AccelerationStructureInstance> = app.world.to_instances(
        //     0,
        //     &IVec3::ZERO,
        //     blas.device_address().into(),
        //     max_instance_count,
        // );

        let render_instances = (0..max_instance_count)
            .map(|_| {
                const RANGE: i32 = 256;
                let x = rand::random_range(-RANGE..=RANGE) as f32;
                let y = rand::random_range(-RANGE..=RANGE) as f32;
                let z = rand::random_range(-RANGE..=RANGE) as f32;

                AccelerationStructureInstance {
                    acceleration_structure_reference: blas.device_address().into(),
                    instance_custom_index_and_mask: Packed24_8::new(
                        rand::random::<u8>() as u32,
                        0xFF,
                    ),
                    transform: [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z]],
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        let build_geometry_info = AccelerationStructureBuildGeometryInfo {
            ..AccelerationStructureBuildGeometryInfo::new(
                AccelerationStructureGeometries::Instances(
                    AccelerationStructureGeometryInstancesData::new(
                        AccelerationStructureGeometryInstancesDataType::Values(None),
                    ),
                ),
            )
        };

        let build_sizes_info = app
            .device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &build_geometry_info,
                &[max_instance_count as u32],
            )
            .unwrap();

        let scratch_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
                DeviceLayout::new_unsized::<[u8]>(build_sizes_info.build_scratch_size).unwrap(),
            )
            .unwrap();

        let instance_buffer_id = app
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

        let tlas = acceleration_structure::build_tlas(
            render_instances.clone(),
            app.memory_allocator.clone(),
            app.device.clone(),
            app.compute_queue.clone(),
            &app.resources,
            app.compute_flight_id,
        );

        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let raygen = raygen::load(&app.device)
                .unwrap()
                .entry_point("main")
                .unwrap();
            let miss = miss::load(&app.device)
                .unwrap()
                .entry_point("main")
                .unwrap();
            let intersection = intersection::load(&app.device)
                .unwrap()
                .entry_point("main")
                .unwrap();
            let closest_hit = closest_hit::load(&app.device)
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(&raygen),
                PipelineShaderStageCreateInfo::new(&miss),
                PipelineShaderStageCreateInfo::new(&intersection),
                PipelineShaderStageCreateInfo::new(&closest_hit),
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
                &app.device,
                None,
                &RayTracingPipelineCreateInfo {
                    stages: &stages,
                    groups: &groups,
                    max_pipeline_ray_recursion_depth: 1,
                    ..base_info
                },
            )
            .unwrap()
        };

        let camera_buffer_id = app
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

        let palette = get_palette(&app.voxel_data).map(|color| [color.x, color.y, color.z, 1.0]);

        let palette_buffer_id = app
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

        let sunlight_buffer_id = app
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

        unsafe {
            vulkano_taskgraph::execute(
                &app.transfer_queue,
                &app.resources,
                app.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer(palette_buffer_id, ..)? = raygen::Palette { colors: palette };

                    let write_instance_buffer = tcx
                        .write_buffer::<[AccelerationStructureInstance]>(instance_buffer_id, ..)?;

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

        app.resources
            .flight(app.graphics_flight_id)
            .unwrap()
            .wait_idle()
            .unwrap();

        let shader_binding_table =
            ShaderBindingTable::new(&app.memory_allocator, &pipeline).unwrap();

        let acceleration_structure_id = bcx.global_set().add_acceleration_structure(tlas.clone());

        let camera_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                camera_buffer_id,
                0,
                size_of::<raygen::Camera>() as DeviceSize,
            )
            .unwrap();

        let palette_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                palette_buffer_id,
                0,
                size_of::<raygen::Palette>() as DeviceSize,
            )
            .unwrap();

        let sunlight_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                sunlight_buffer_id,
                0,
                size_of::<raygen::Sunlight>() as DeviceSize,
            )
            .unwrap();

        RayTracingPass {
            swapchain_id: virtual_swapchain_id,
            acceleration_structure_id,
            camera_buffer_id,
            sunlight_buffer_id,
            instance_buffer_id,
            scratch_buffer_id,
            camera_storage_buffer_id,
            palette_storage_buffer_id,
            sunlight_storage_buffer_id,
            shader_binding_table,
            blas,
            tlas,
            pipeline,
        }
    }
}

impl Task for RayTracingPass {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id)?;
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        unsafe { cbf.update_buffer(self.camera_buffer_id, 0, &rcx.rt_camera_data) }?;
        unsafe { cbf.update_buffer(self.sunlight_buffer_id, 0, &rcx.rt_sunlight_data) }?;

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &raygen::PushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    acceleration_structure_id: self.acceleration_structure_id,
                    camera_buffer_id: self.camera_storage_buffer_id,
                    palette_buffer_id: self.palette_storage_buffer_id,
                    sunlight_buffer_id: self.sunlight_storage_buffer_id,
                },
            )
        }?;

        unsafe {
            cbf.bind_pipeline_ray_tracing(&self.pipeline)?;
        }

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) }?;

        Ok(())
    }
}
