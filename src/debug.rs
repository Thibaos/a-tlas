use core::slice;
use std::sync::Arc;
use vulkano::{
    buffer::Buffer,
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
        },
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    descriptor_set::StorageBufferId,
    graph::TaskNode,
};

use crate::{
    app::App,
    region::render::{RegionRenderContext, RenderMode},
    world::Vertex3DColor,
};

pub mod shader {
    pub(crate) mod vert {
        vulkano_shaders::shader! {
            root_path_env: "CARGO_MANIFEST_DIR",
            ty: "vertex",
            path: "shaders/debug/lines/vert.glsl",
            vulkan_version: "1.3"
        }
    }

    pub(crate) mod frag {
        vulkano_shaders::shader! {
            root_path_env: "CARGO_MANIFEST_DIR",
            ty: "fragment",
            path: "shaders/debug/lines/frag.glsl",
            vulkan_version: "1.3"
        }
    }
}

pub fn create_debug_pipeline(
    app: &App,
    node: &TaskNode<RegionRenderContext>,
) -> Arc<GraphicsPipeline> {
    let subpass = node.subpass().unwrap().clone();

    let vertex = unsafe {
        shader::vert::load(&app.gpu.device)
            .unwrap()
            .entry_point("main")
            .unwrap()
    };
    let fragment = unsafe {
        shader::frag::load(&app.gpu.device)
            .unwrap()
            .entry_point("main")
            .unwrap()
    };

    let stages = [
        PipelineShaderStageCreateInfo::new(&vertex),
        PipelineShaderStageCreateInfo::new(&fragment),
    ];

    let bcx = app.gpu.resources.bindless_context().unwrap();

    let debug_pipeline_layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

    let vertex_input_state = Vertex3DColor::per_vertex().definition(&vertex).unwrap();

    GraphicsPipeline::new(
        &app.gpu.device,
        None,
        &GraphicsPipelineCreateInfo {
            stages: &stages,
            vertex_input_state: Some(&vertex_input_state),
            input_assembly_state: Some(&InputAssemblyState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            }),
            viewport_state: Some(&ViewportState::default()),
            rasterization_state: Some(&RasterizationState::default()),
            multisample_state: Some(&MultisampleState::default()),
            color_blend_state: Some(&ColorBlendState {
                attachments: &[ColorBlendAttachmentState::default()],
                ..Default::default()
            }),
            dynamic_state: &[DynamicState::Viewport],
            subpass: Some((&subpass).into()),
            ..GraphicsPipelineCreateInfo::new(&debug_pipeline_layout)
        },
    )
    .expect("Failed to create debug pipeline")
}

pub struct DrawDebugTask {
    pub vertex_buffer_id: Id<Buffer>,
    pub pipeline: Option<Arc<GraphicsPipeline>>,
}

impl DrawDebugTask {
    pub fn new(vertex_buffer_id: Id<Buffer>) -> Self {
        Self {
            vertex_buffer_id,
            pipeline: None,
        }
    }
}

impl Task for DrawDebugTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let push_constants = rcx.debug_constant_data;

        let pipeline = self.pipeline.as_ref().unwrap();

        let debug_lines_count = rcx.debug_lines.len() as u32;

        if debug_lines_count == 0 {
            return Ok(());
        }

        tcx.write_buffer::<[Vertex3DColor]>(
            self.vertex_buffer_id,
            0u64..(debug_lines_count as u64 * size_of::<Vertex3DColor>() as u64),
        )
        .copy_from_slice(&rcx.debug_lines);

        unsafe { cbf.set_viewport(0, slice::from_ref(&rcx.viewport)) };
        unsafe { cbf.bind_pipeline(pipeline) };
        unsafe { cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]) };
        unsafe { cbf.push_constants(pipeline.layout(), 0, &push_constants) };
        unsafe { cbf.draw(debug_lines_count, 1, 0, 0) };

        Ok(())
    }
}

pub mod heatmap {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "compute",
        path: "shaders/debug/heatmap.comp",
        vulkan_version: "1.3"
    }
}

/// Builds the hull-crossed heatmap compute pipeline: reads the per-pixel count
/// buffer (bindless) and paints the swapchain storage image (debug builds
/// only — the hull-crossed mode has no surface in release).
pub fn create_heatmap_pipeline(app: &App) -> Arc<ComputePipeline> {
    let shader = unsafe {
        heatmap::load(&app.gpu.device)
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

/// The hull-crossed heatmap overlay node: a full-screen compute pass that reads
/// the per-pixel count buffer and repaints the swapchain image. It runs only
/// when the Render mode is [`RenderMode::HullCrossed`]; otherwise it is a no-op
/// and leaves the ray pass's output untouched.
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
    type World = RegionRenderContext;

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
