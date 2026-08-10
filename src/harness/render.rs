//! The harness's ray pass: the same render path as the app (same
//! acceleration structures, instance buffer, palette, camera buffer), but
//! with a raygen that additionally writes the committed hit distance to a
//! t-channel image. The color store is byte-identical to the production
//! raygen (both share shaders/rt/raygen_common.glsl), so the captured color
//! frame IS the raw renderer output.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use dot_vox::DotVoxData;
use vulkano::{
    buffer::Buffer,
    pipeline::{
        ray_tracing::{RayTracingPipeline, ShaderBindingTable},
        Pipeline,
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer,
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
    Id, Task, TaskContext, TaskResult,
};

use crate::{
    app::GpuStack,
    rt::raygen,
    tasks::render::{build_ray_tracing_pipeline, create_render_resources},
    world::chunk::Chunks,
};

pub(crate) mod capture_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/harness/capture.rgen",
        vulkan_version: "1.3"
    }
}

/// The per-frame context the harness's tasks read.
pub struct HarnessRenderContext {
    pub camera: raygen::Camera,
    pub sunlight: raygen::Sunlight,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    pub t_image_storage_id: StorageImageId,
}

pub struct HarnessRenderTask {
    swapchain_id: Id<Swapchain>,
    camera_buffer_id: Id<Buffer>,
    sunlight_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    camera_storage_buffer_id: StorageBufferId,
    palette_storage_buffer_id: StorageBufferId,
    sunlight_storage_buffer_id: StorageBufferId,
    acceleration_structure_ids: [AccelerationStructureId; 2],
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    current_as_index: Arc<AtomicBool>,
}

impl HarnessRenderTask {
    /// The instance buffer the TLAS reads (declared in the task graph).
    pub fn instance_buffer_id(&self) -> Id<Buffer> {
        self.instance_buffer_id
    }

    /// The camera buffer the ray pass reads (diagnostics/readback).
    pub fn camera_buffer_id(&self) -> Id<Buffer> {
        self.camera_buffer_id
    }

    pub fn new(
        gpu: &GpuStack,
        world: &Arc<Chunks>,
        voxel_data: &DotVoxData,
        virtual_swapchain_id: Id<Swapchain>,
        max_instance_count: u64,
    ) -> Self {
        let resources = create_render_resources(gpu, world, voxel_data, max_instance_count);

        let raygen_entry = unsafe {
            capture_raygen::load(&gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };
        let miss_entry = unsafe {
            crate::rt::miss::load(&gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };
        let intersection_entry = unsafe {
            crate::rt::intersection::load(&gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };
        let closest_hit_entry = unsafe {
            crate::rt::closest_hit::load(&gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

        let pipeline = build_ray_tracing_pipeline(
            gpu,
            &raygen_entry,
            &miss_entry,
            &intersection_entry,
            &closest_hit_entry,
        );

        let shader_binding_table =
            ShaderBindingTable::new(&gpu.memory_allocator, &pipeline).unwrap();

        Self {
            swapchain_id: virtual_swapchain_id,
            camera_buffer_id: resources.camera_buffer_id,
            sunlight_buffer_id: resources.sunlight_buffer_id,
            instance_buffer_id: resources.instance_buffer_id,
            camera_storage_buffer_id: resources.camera_storage_buffer_id,
            palette_storage_buffer_id: resources.palette_storage_buffer_id,
            sunlight_storage_buffer_id: resources.sunlight_storage_buffer_id,
            acceleration_structure_ids: resources.acceleration_structure_ids,
            shader_binding_table,
            pipeline,
            current_as_index: resources.current_as_index,
        }
    }
}

impl Task for HarnessRenderTask {
    type World = HarnessRenderContext;

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
        unsafe { cbf.update_buffer(self.sunlight_buffer_id, 0, &rcx.sunlight) };

        // The taskgraph does not track update_buffer's writes, so make the
        // camera/palette writes visible to the ray pass explicitly.
        unsafe {
            cbf.pipeline_barrier(&vulkano_taskgraph::command_buffer::DependencyInfo {
                memory_barriers: &[vulkano_taskgraph::command_buffer::MemoryBarrier {
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

        let front_index = self.current_as_index.load(Ordering::Acquire);

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &capture_raygen::PushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    t_image_id: rcx.t_image_storage_id,
                    acceleration_structure_id: self.acceleration_structure_ids
                        [front_index as usize],
                    camera_buffer_id: self.camera_storage_buffer_id,
                    palette_buffer_id: self.palette_storage_buffer_id,
                    sunlight_buffer_id: self.sunlight_storage_buffer_id,
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
