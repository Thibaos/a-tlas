//! GPU readback helpers for the validator: host-visible capture buffers,
//! byte decoding, and the hidden swapchain / t-image setup.

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    format::Format,
    image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use vulkano_taskgraph::{
    Id,
    descriptor_set::StorageImageId,
    resource::Resources,
};

use crate::core::gpu::{GpuStack, MIN_SWAPCHAIN_IMAGES};

pub fn read_host_bytes(gpu: &GpuStack, id: Id<Buffer>) -> Vec<u8> {
    let buffer = gpu.resources.buffer(id).buffer().clone();
    let subbuffer = Subbuffer::new(buffer).cast_aligned::<u8>();
    subbuffer
        .read()
        .expect("host read of capture buffer")
        .to_vec()
}

/// Reads a host-visible buffer as f32s.
pub fn read_host_floats(gpu: &GpuStack, id: Id<Buffer>) -> Vec<f32> {
    let buffer = gpu.resources.buffer(id).buffer().clone();
    let subbuffer = Subbuffer::new(buffer).cast_aligned::<f32>();
    subbuffer
        .read()
        .expect("host read of capture buffer")
        .to_vec()
}


pub fn create_host_readback(gpu: &GpuStack, bytes: u64) -> Id<Buffer> {
    gpu.resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(bytes).unwrap(),
        )
        .unwrap()
}

/// Decodes RGBA16F (R16G16B16A16_SFLOAT) bytes, 8 per pixel, to f32 RGBA.
pub fn decode_rgba16f(bytes: &[u8]) -> Vec<glam::Vec4> {
    bytes
        .chunks_exact(8)
        .map(|px| {
            glam::Vec4::new(
                half_to_f32(u16::from_le_bytes([px[0], px[1]])),
                half_to_f32(u16::from_le_bytes([px[2], px[3]])),
                half_to_f32(u16::from_le_bytes([px[4], px[5]])),
                half_to_f32(u16::from_le_bytes([px[6], px[7]])),
            )
        })
        .collect()
}

/// IEEE-754 binary16 → f32 (the radiance pair's storage format; the CPU
/// mirror computes f32 and the tolerance absorbs the quantization).
pub fn half_to_f32(h: u16) -> f32 {
    let negative = h & 0x8000 != 0;
    let exp = (h >> 10) & 0x1F;
    let mant = u32::from(h & 0x3FF);
    let magnitude = if exp == 0 {
        (mant as f32) * 2.0_f32.powi(-24)
    } else if exp == 31 {
        // Inf/NaN. The radiance pair never contains them; map to +inf.
        f32::INFINITY
    } else {
        ((mant as f32) + 1024.0) * 2.0_f32.powi(i32::from(exp) - 25)
    };
    if negative { -magnitude } else { magnitude }
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

pub fn create_validate_swapchain(
    gpu: &GpuStack,
    surface: &Arc<Surface>,
    extent: [u32; 2],
) -> Result<(Id<Swapchain>, Format), vulkano::VulkanError> {
    let surface_capabilities = gpu
        .device
        .physical_device()
        .surface_capabilities(surface, &Default::default())?;
    let (image_format, image_color_space) = gpu
        .device
        .physical_device()
        .surface_formats(surface, &Default::default())?
        .into_iter()
        // Only UNORM formats: an sRGB swapchain would sRGB-encode the ray
        // pass's linear writes while the reference quantizes raw palette
        // bytes, causing systematic color mismatches.
        .filter(|(format, _)| {
            matches!(
                format,
                Format::R8G8B8A8_UNORM
                    | Format::B8G8R8A8_UNORM
                    | Format::R8G8B8A8_SNORM
                    | Format::B8G8R8A8_SNORM
            )
        })
        .find(|(format, _)| {
            gpu.device
                .physical_device()
                .image_format_properties(&vulkano::image::ImageFormatInfo {
                    format: *format,
                    usage: ImageUsage::STORAGE
                        | ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                })
                .unwrap()
                .is_some()
        })
        .ok_or(vulkano::VulkanError::Unknown)?;

    let swapchain_id = gpu.resources.create_swapchain(
        surface,
        &SwapchainCreateInfo {
            present_mode: PresentMode::Immediate,
            min_image_count: surface_capabilities
                .min_image_count
                .max(MIN_SWAPCHAIN_IMAGES),
            image_format,
            image_extent: extent,
            image_usage: ImageUsage::STORAGE
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::TRANSFER_SRC,
            image_color_space,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )?;

    Ok((swapchain_id, image_format))
}

/// Creates the rgba32f t-channel image and registers it in the bindless set.
pub fn create_t_image(
    resources: &Arc<Resources>,
    width: u32,
    height: u32,
) -> Result<(Id<Image>, StorageImageId, Format), String> {
    let format = Format::R32G32B32A32_SFLOAT;
    let image_id = resources
        .create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [width, height, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .map_err(|e| format!("{e}"))?;

    let image = resources.image(image_id).image().clone();
    let image_view = ImageView::new_default(&image).unwrap();

    let storage_id = resources
        .bindless_context()
        .unwrap()
        .global_set()
        .add_storage_image(image_view, ImageLayout::General);

    Ok((image_id, storage_id, format))
}

/// Decodes the captured bytes (in the swapchain's channel order) to RGBA8.
pub fn decode_rgba(format: Format, bytes: &[u8]) -> Vec<u8> {
    match format {
        Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SNORM => bytes.to_vec(),
        Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SNORM => bytes
            .chunks_exact(4)
            .flat_map(|p| [p[2], p[1], p[0], p[3]])
            .collect(),
        other => panic!("unsupported swapchain format for capture: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

#[cfg(test)]
mod path_readback_tests {
    use super::*;

    #[test]
    fn half_to_f32_decodes_known_values() {
        assert_eq!(half_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(half_to_f32(0x4b77), 1911.0 * 2.0f32.powi(-7)); // 14.9296875
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3e00), 1.5); // 1.5
        assert_eq!(half_to_f32(0x3800), 0.5); // 0.5
    }

    #[test]
    fn decode_rgba16f_reads_pixel_alignment() {
        // One pixel: R=1.0 (0x3c00), G=1.5 (0x3e00), B=0.25 (0x3400), A=14.93 (0x4b77)
        let bytes = [0x00, 0x3c, 0x00, 0x3e, 0x00, 0x34, 0x77, 0x4b];
        let decoded = decode_rgba16f(&bytes);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].x, 1.0);
        assert_eq!(decoded[0].y, 1.5);
        assert_eq!(decoded[0].z, 0.25);
        assert_eq!(decoded[0].w, 1911.0 * 2.0f32.powi(-7));
    }
}
