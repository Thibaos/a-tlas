use core::slice;
use std::sync::Arc;
use vulkano::{
    buffer::Buffer,
    pipeline::{GraphicsPipeline, Pipeline},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
};

use crate::app::RenderContext;

pub mod shader {
    pub(crate) mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "shaders/debug/lines/vert.glsl",
            vulkan_version: "1.3"
        }
    }

    pub(crate) mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/debug/lines/frag.glsl",
            vulkan_version: "1.3"
        }
    }
}

pub struct DrawDebugTask {
    pub vertex_count: u32,
    pub vertex_buffer_id: Id<Buffer>,
    pub pipeline: Option<Arc<GraphicsPipeline>>,
}

impl Task for DrawDebugTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let push_constants = rcx.debug_constant_data;

        let pipeline = self.pipeline.as_ref().unwrap();

        unsafe { cbf.set_viewport(0, slice::from_ref(&rcx.viewport)) }?;
        unsafe { cbf.bind_pipeline_graphics(pipeline) }?;
        unsafe { cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]) }?;
        unsafe { cbf.push_constants(pipeline.layout(), 0, &push_constants) }?;
        unsafe { cbf.draw(self.vertex_count, 1, 0, 0) }?;

        Ok(())
    }
}
