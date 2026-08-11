//! The capture task: copies the raw ray-pass output into host-readable
//! buffers before anything else (the debug overlay, when present, runs after
//! this node on the same image). This is the seam that satisfies "capture
//! happens before the debug overlay draws" — the validation graph orders
//! Render → Capture and never lets an overlay touch the copied bytes.

use vulkano::{
    buffer::Buffer,
    format::Format,
    image::{Image, ImageAspects, ImageSubresourceLayers},
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    command_buffer::{BufferImageCopy as TgBufferImageCopy, CopyImageToBufferInfo as TgCopyInfo, RecordingCommandBuffer},
    resource::ImageLayoutType,
    Id, Task, TaskContext, TaskResult,
};


use crate::region::render::RegionRenderContext;

pub struct CaptureTask {
    swapchain_id: Id<Swapchain>,
    /// The t-channel image (written by the validation raygen).
    t_image_id: Id<Image>,
    t_format: Format,
    pub color_readback_buffer_id: Id<Buffer>,
    pub t_readback_buffer_id: Id<Buffer>,
}

impl CaptureTask {
    pub fn new(
        swapchain_id: Id<Swapchain>,
        t_image_id: Id<Image>,
        t_format: Format,
        color_readback_buffer_id: Id<Buffer>,
        t_readback_buffer_id: Id<Buffer>,
    ) -> Self {
        Self {
            swapchain_id,
            t_image_id,
            t_format,
            color_readback_buffer_id,
            t_readback_buffer_id,
        }
    }

    pub fn t_format(&self) -> Format {
        self.t_format
    }
}

impl Task for CaptureTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let extent = swapchain_state.images()[0].extent();

        let region = TgBufferImageCopy {
            image_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: Some(1),
            },
            image_extent: extent,
            ..Default::default()
        };

        unsafe {
            cbf.copy_image_to_buffer(&TgCopyInfo {
                src_image: self.swapchain_id.current_image_id(),
                src_image_layout: ImageLayoutType::General,
                dst_buffer: self.color_readback_buffer_id,
                regions: std::slice::from_ref(&region),
                ..Default::default()
            })
        };

        unsafe {
            cbf.copy_image_to_buffer(&TgCopyInfo {
                src_image: self.t_image_id,
                src_image_layout: ImageLayoutType::General,
                dst_buffer: self.t_readback_buffer_id,
                regions: std::slice::from_ref(&region),
                ..Default::default()
            })
        };

        Ok(())
    }
}
