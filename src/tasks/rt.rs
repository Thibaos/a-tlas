use crate::{
    App, RenderContext,
    rt::{acceleration_structure, closest_hit, intersection, miss, raygen},
    world::voxel::{get_palette, triangles_from_box},
};
use glam::{IVec3, Vec3};
use std::sync::Arc;
use vulkano::{
    DeviceSize,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildType, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureInstance, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
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

pub const MAX_INSTANCE_COUNT: u64 = 2u64.pow(20);

pub struct RayTracingPass {
    swapchain_id: Id<Swapchain>,
    pub acceleration_structure_id: AccelerationStructureId,
    pub camera_buffer_id: Id<Buffer>,
    pub sunlight_buffer_id: Id<Buffer>,
    pub instance_buffer_id: Id<Buffer>,
    scratch_buffer_id: Id<Buffer>,
    camera_storage_buffer_id: StorageBufferId,
    palette_storage_buffer_id: StorageBufferId,
    sunlight_storage_buffer_id: StorageBufferId,
    shader_binding_table: ShaderBindingTable,
    pub blas: Arc<AccelerationStructure>,
    tlas: Arc<AccelerationStructure>,
    pipeline: Arc<RayTracingPipeline>,
    pub render_instances: Vec<AccelerationStructureInstance>,
}

impl RayTracingPass {
    pub fn new(app: &App, virtual_swapchain_id: Id<Swapchain>) -> Self {
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
            app.graphics_queue.clone(),
            &app.resources,
            app.graphics_flight_id,
        );

        let render_instances =
            app.world
                .to_instances(0, &IVec3::ZERO, blas.device_address().into());

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
                &[MAX_INSTANCE_COUNT as u32],
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

        let tlas = acceleration_structure::build_tlas(
            render_instances.clone(),
            app.memory_allocator.clone(),
            app.device.clone(),
            app.graphics_queue.clone(),
            &app.resources,
            app.graphics_flight_id,
        );

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
                DeviceLayout::new_unsized::<[AccelerationStructureInstance]>(
                    MAX_INSTANCE_COUNT as u64,
                )
                .unwrap(),
            )
            .unwrap();

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

                    Ok(())
                },
                [(palette_buffer_id, HostAccessType::Write)],
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
            render_instances,
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

        let write_instance_buffer = tcx
            .write_buffer::<[AccelerationStructureInstance]>(self.instance_buffer_id, ..)
            .unwrap();

        for (o, render_instance) in write_instance_buffer.iter_mut().zip(&self.render_instances) {
            *o = *render_instance;
        }

        let instance_buffer = Subbuffer::new(
            tcx.buffer(self.instance_buffer_id)
                .unwrap()
                .buffer()
                .clone(),
        )
        .cast_aligned::<AccelerationStructureInstance>();

        let geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer.clone())),
        );

        let geometries = AccelerationStructureGeometries::Instances(geometry_instances_data);
        let primitive_count = self.render_instances.len() as u32;

        let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
            mode: BuildAccelerationStructureMode::Update(self.tlas.clone()),
            flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
            ..AccelerationStructureBuildGeometryInfo::new(geometries)
        };

        let scratch_buffer =
            Subbuffer::new(tcx.buffer(self.scratch_buffer_id).unwrap().buffer().clone());

        build_geometry_info.dst_acceleration_structure = Some(self.tlas.clone());
        build_geometry_info.scratch_data = Some(scratch_buffer);

        // unsafe {
        //     cbf.as_raw().build_acceleration_structure(
        //         &build_geometry_info,
        //         &[AccelerationStructureBuildRangeInfo {
        //             primitive_count,
        //             ..Default::default()
        //         }],
        //     )
        // }?;

        // unsafe {
        //     cbf.pipeline_barrier(&DependencyInfo {
        //         memory_barriers: &[MemoryBarrier {
        //             src_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        //             dst_stages: PipelineStages::ALL_COMMANDS,
        //             src_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE,
        //             dst_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE,
        //             ..Default::default()
        //         }],
        //         ..Default::default()
        //     })
        // }?;

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
