#[cfg(debug_assertions)]
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

pub mod chunk;
pub mod loader;
pub mod voxel;

// The debug overlay's vertex type (app-only, cfg'd out in release).
#[cfg(debug_assertions)]
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
pub struct Vertex3DColor {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

#[derive(Debug, Default)]
pub struct HostVoxel {
    material_index: u32,
}

impl HostVoxel {
    pub fn new(material_index: u32) -> Self {
        Self { material_index }
    }

    /// The palette index of this voxel (0 is a real color).
    pub(crate) fn material_index(&self) -> u32 {
        self.material_index
    }
}
