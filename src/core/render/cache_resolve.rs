use core::slice;
use std::sync::Arc;
use vulkano::{
    pipeline::{
        ComputePipeline, PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
    },
};
use vulkano_taskgraph::{
    Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
};
use crate::core::render::{
    gpu::GpuDesc,
    region::{residency::RegionBindings, task::RegionRenderContext},
};

pub mod cache_resolve {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "compute",
        path: "shaders/region/cache_resolve.comp",
        vulkan_version: "1.3"
    }
}

pub fn create_cache_resolve_pipeline(gpu: &GpuDesc) -> Arc<ComputePipeline> {
    let shader = unsafe {
        cache_resolve::load(&gpu.device)
            .unwrap()
    };
    let entry_point = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(&entry_point);
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

// 02's resolve half: SHaRC's Resolve at one thread per table entry — the
// dirty-Region sweep, stale-aging eviction, and the blend of each touched
// face's deposits against its stored state (the 02 ladder and impulse
// clamp) in one pass, resetting accumulators for the next frame's deposits.
pub struct CacheResolveTask {
    pub bindings: RegionBindings,
    pub pipeline: Option<Arc<ComputePipeline>>,
}

impl CacheResolveTask {
    pub fn new(bindings: RegionBindings) -> Self {
        Self {
            bindings,
            pipeline: None,
        }
    }
}

impl Task for CacheResolveTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        if rcx.cache_resolve_dispatch == 0 {
            return Ok(());
        }

        unsafe { cbf.bind_pipeline(self.pipeline.as_ref().unwrap()) };
        unsafe {
            cbf.push_constants(
                self.pipeline.as_ref().unwrap().layout(),
                0,
                &cache_resolve::ResolvePushConstants {
                    cache_state_buffer_id: self.bindings.cache_state_storage_id,
                },
            )
        };
        unsafe { cbf.dispatch([rcx.cache_resolve_dispatch.div_ceil(64), 1, 1]) };

        Ok(())
    }
}
