//! The capture task: copies the raw ray-pass output into host-readable
//! buffers before anything else (the debug overlay, when present, runs after
//! this node on the same image). This is the seam that satisfies "capture
//! happens before the debug overlay draws." The validation graph orders
//! Render → Capture and never lets an overlay touch the copied bytes.

use vulkano::{
    buffer::Buffer,
    format::Format,
    image::{Image, ImageAspects, ImageSubresourceLayers},
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{
        BufferImageCopy as TgBufferImageCopy, CopyImageToBufferInfo as TgCopyInfo,
        RecordingCommandBuffer,
    },
    resource::ImageLayoutType,
};

use crate::render::region::task::RegionRenderContext;

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

/// The shading-half capture task (ticket 07): copies the production ray
/// pass's radiance pair (diffuse + specular RGBA16F) and the albedo+metalness
/// aux (RGBA8) into host-readable buffers after each path-traced frame, so
/// the validator can accumulate the per-pixel means the CPU mirror is diffed
/// against. The geometry half's [`CaptureTask`] is unchanged. The byte-exact
/// {color, t} comparison keeps the capture raygen's output.
pub struct PathCaptureTask {
    diff_image_id: Id<Image>,
    spec_image_id: Id<Image>,
    albedo_image_id: Id<Image>,
    normal_roughness_image_id: Id<Image>,
    pub diff_readback_buffer_id: Id<Buffer>,
    pub spec_readback_buffer_id: Id<Buffer>,
    pub albedo_readback_buffer_id: Id<Buffer>,
    pub normal_roughness_readback_buffer_id: Id<Buffer>,
    width: u32,
    height: u32,
}

impl PathCaptureTask {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        diff_image_id: Id<Image>,
        spec_image_id: Id<Image>,
        albedo_image_id: Id<Image>,
        normal_roughness_image_id: Id<Image>,
        diff_readback_buffer_id: Id<Buffer>,
        spec_readback_buffer_id: Id<Buffer>,
        albedo_readback_buffer_id: Id<Buffer>,
        normal_roughness_readback_buffer_id: Id<Buffer>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            diff_image_id,
            spec_image_id,
            albedo_image_id,
            normal_roughness_image_id,
            diff_readback_buffer_id,
            spec_readback_buffer_id,
            albedo_readback_buffer_id,
            normal_roughness_readback_buffer_id,
            width,
            height,
        }
    }
}

impl Task for PathCaptureTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        _rcx: &Self::World,
    ) -> TaskResult {
        let region = TgBufferImageCopy {
            image_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: Some(1),
            },
            image_extent: [self.width, self.height, 1],
            ..Default::default()
        };

        for (src, dst) in [
            (self.diff_image_id, self.diff_readback_buffer_id),
            (self.spec_image_id, self.spec_readback_buffer_id),
            (self.albedo_image_id, self.albedo_readback_buffer_id),
            (
                self.normal_roughness_image_id,
                self.normal_roughness_readback_buffer_id,
            ),
        ] {
            unsafe {
                cbf.copy_image_to_buffer(&TgCopyInfo {
                    src_image: src,
                    src_image_layout: ImageLayoutType::General,
                    dst_buffer: dst,
                    regions: std::slice::from_ref(&region),
                    ..Default::default()
                })
            };
        }

        Ok(())
    }
}
