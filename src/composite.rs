//! The composite node: the taskgraph's last pass before present, exposing
//! the trace pass's radiance to the swapchain with manual EV exposure and
//! the ACES tonemap. Part of the path-tracing output contract (ADR 0007);
//! from ticket 08 the composite consumes the Denoise pass's output instead
//! of the noisy radiance directly.

use core::slice;
use std::sync::Arc;
use vulkano::{
    pipeline::{ComputePipeline, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo},
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
};

use crate::{app::App, region::render::RenderMode};

pub mod composite {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "compute",
        path: "shaders/region/composite.comp",
        vulkan_version: "1.3"
    }
}

/// Builds the composite compute pipeline: reads the radiance source and
/// paints the swapchain storage image (exposure + ACES tonemap + gamma).
pub fn create_composite_pipeline(app: &App) -> Arc<ComputePipeline> {
    let shader = unsafe {
        composite::load(&app.gpu.device)
            .unwrap()
            .entry_point("main")
            .unwrap()
    };
    let stage = PipelineShaderStageCreateInfo::new(&shader);
    let bcx = app.gpu.resources.bindless_context().unwrap();
    let layout = bcx
        .pipeline_layout_from_stages(slice::from_ref(&stage))
        .unwrap();
    ComputePipeline::new(
        &app.gpu.device,
        None,
        &ComputePipelineCreateInfo::new(stage, &layout),
    )
    .unwrap()
}

/// The composite node: a full-screen compute pass exposing the trace pass's
/// radiance to the swapchain. It runs only when the Render mode is Voxel
/// (the path-tracing mode); the debug modes paint the swapchain directly
/// from the raygen, so the composite no-ops for them.
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
    type World = crate::region::render::RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        if rcx.mode != RenderMode::Voxel {
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
                    ev: rcx.ev,
                    mode: rcx.mode as u32,
                    width: extent[0],
                    height: extent[1],
                },
            )
        };
        unsafe { cbf.dispatch([extent[0].div_ceil(16), extent[1].div_ceil(16), 1]) };

        Ok(())
    }
}
