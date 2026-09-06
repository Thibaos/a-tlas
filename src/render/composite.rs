use anyhow::Context;
use core::slice;
use std::sync::Arc;
use vulkano::{
    pipeline::{
        ComputePipeline, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
    },
    swapchain::Swapchain,
    sync::{AccessFlags, PipelineStages},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
};

use crate::render::{
    context::RenderContext,
    region::task::{RegionRenderContext, RenderMode},
};

pub mod composite_shader {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "compute",
        path: "shaders/composite.comp",
        vulkan_version: "1.3"
    }
}

pub fn create_composite_pipeline(gpu: &RenderContext) -> anyhow::Result<Arc<ComputePipeline>> {
    let shader = unsafe {
        composite_shader::load(&gpu.device)?
            .entry_point("main")
            .context("main entry point not found for composite shader")?
    };

    let stage = PipelineShaderStageCreateInfo::new(&shader);
    let bcx = gpu
        .resources
        .bindless_context()
        .context("bindless context not found")?;
    let layout = bcx.pipeline_layout_from_stages(slice::from_ref(&stage))?;

    Ok(ComputePipeline::new(
        &gpu.device,
        None,
        &ComputePipelineCreateInfo::new(stage, &layout),
    )?)
}

pub struct CompositeTask {
    pub swapchain_id: Id<Swapchain>,
    pub pipeline: Option<Arc<ComputePipeline>>,
}

impl CompositeTask {
    pub const fn new(swapchain_id: Id<Swapchain>) -> Self {
        Self {
            swapchain_id,
            pipeline: None,
        }
    }
}

impl Task for CompositeTask {
    type World = RegionRenderContext;

    #[allow(clippy::as_conversions)]
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

        let Some(image_index) = swapchain_state.current_image_index() else {
            eprintln!("swapchain current image index not found");
            return Ok(());
        };

        let Some(swapchain_first_image) = swapchain_state.images().first() else {
            eprintln!("swapchain image not found");
            return Ok(());
        };

        let Some(image_id) = rcx.swapchain_storage_image_ids.get(image_index as usize) else {
            eprintln!("swapchain storage image {image_index} not found");
            return Ok(());
        };

        let Some(pipeline) = self.pipeline.as_ref() else {
            eprintln!("pipeline not found");
            return Ok(());
        };

        let extent = swapchain_first_image.extent();

        unsafe { cbf.bind_pipeline(pipeline) };

        unsafe {
            cbf.push_constants(
                pipeline.layout(),
                0,
                &composite_shader::PushConstants {
                    image_id: *image_id,
                    color_id: rcx.color_image_id,
                    mode: rcx.mode as u32,
                    width: extent[0],
                    height: extent[1],
                },
            )
        };

        unsafe { cbf.dispatch([extent[0].div_ceil(16), extent[1].div_ceil(16), 1]) };

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_access: AccessFlags::SHADER_STORAGE_WRITE,
                    dst_access: AccessFlags::SHADER_STORAGE_READ
                        | AccessFlags::SHADER_STORAGE_WRITE,
                    src_stages: PipelineStages::COMPUTE_SHADER,
                    dst_stages: PipelineStages::COMPUTE_SHADER,
                    ..MemoryBarrier::default()
                }],
                ..DependencyInfo::default()
            })
        };

        Ok(())
    }
}
