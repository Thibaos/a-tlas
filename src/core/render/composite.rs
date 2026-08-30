use core::slice;
use std::sync::Arc;
use vulkano::{
    pipeline::{
        ComputePipeline, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
};

use crate::core::render::{
    gpu::GpuDesc,
    region::task::{RegionRenderContext, RenderMode},
};

pub mod composite {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "compute",
        path: "shaders/composite.comp",
        vulkan_version: "1.3"
    }
}

pub fn create_composite_pipeline(gpu: &GpuDesc) -> Arc<ComputePipeline> {
    let shader = unsafe {
        composite::load(&gpu.device)
            .unwrap()
            .entry_point("main")
            .unwrap()
    };
    let stage = PipelineShaderStageCreateInfo::new(&shader);
    let bcx = gpu.resources.bindless_context().unwrap();
    let layout = bcx
        .pipeline_layout_from_stages(slice::from_ref(&stage))
        .unwrap();
    ComputePipeline::new(
        &gpu.device,
        None,
        &ComputePipelineCreateInfo::new(stage, &layout),
    )
    .unwrap()
}

pub struct CompositeTask {
    pub swapchain_id: Id<Swapchain>,
    pub pipeline: Option<Arc<ComputePipeline>>,
}

impl CompositeTask {
    pub fn new(swapchain_id: Id<Swapchain>) -> Self {
        Self {
            swapchain_id,
            pipeline: None,
        }
    }
}

impl Task for CompositeTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        if rcx.mode != RenderMode::Voxel && !rcx.mode.is_nrd_validation() {
            return Ok(());
        }

        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        let pipeline = self.pipeline.as_ref().unwrap();

        unsafe { cbf.bind_pipeline(pipeline) };
        unsafe {
            cbf.push_constants(
                pipeline.layout(),
                0,
                &composite::PushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    radiance_id: rcx.diff_radiance_image_id,
                    spec_radiance_id: rcx.spec_radiance_image_id,
                    diff_denoised_id: rcx.denoised_diff_image_id,
                    spec_denoised_id: rcx.denoised_spec_image_id,
                    viewz_id: rcx.viewz_image_id,
                    albedo_metal_id: rcx.albedo_metal_image_id,
                    validation_id: rcx.validation_image_id,
                    ev: rcx.ev,
                    mode: rcx.mode as u32,
                    denoiser: u32::from(rcx.denoiser_active),
                    width: extent[0],
                    height: extent[1],
                },
            )
        };
        unsafe { cbf.dispatch([extent[0].div_ceil(16), extent[1].div_ceil(16), 1]) };

        Ok(())
    }
}
