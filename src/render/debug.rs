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
    descriptor_set::StorageBufferId,
};

use crate::{core::gpu::GpuStack, render::region::task::RenderMode};

pub mod heatmap {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "compute",
        path: "shaders/debug/heatmap.comp",
        vulkan_version: "1.3"
    }
}

pub fn create_heatmap_pipeline(gpu: &GpuStack) -> Arc<ComputePipeline> {
    let shader = unsafe {
        heatmap::load(&gpu.device)
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

pub struct DrawHeatmapTask {
    pub swapchain_id: Id<Swapchain>,
    pub hull_count_storage_id: StorageBufferId,
    pub pipeline: Option<Arc<ComputePipeline>>,
}

impl DrawHeatmapTask {
    pub fn new(swapchain_id: Id<Swapchain>, hull_count_storage_id: StorageBufferId) -> Self {
        Self {
            swapchain_id,
            hull_count_storage_id,
            pipeline: None,
        }
    }
}

impl Task for DrawHeatmapTask {
    type World = crate::render::region::task::RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        if rcx.mode != RenderMode::HullCrossed {
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
                &heatmap::PushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    hull_count_buffer_id: self.hull_count_storage_id,
                    width: extent[0],
                    height: extent[1],
                },
            )
        };
        unsafe { cbf.dispatch([extent[0].div_ceil(16), extent[1].div_ceil(16), 1]) };

        Ok(())
    }
}
