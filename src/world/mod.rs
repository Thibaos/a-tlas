use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

pub mod chunk;
pub mod loader;
pub mod voxel;

#[derive(BufferContents)]
#[repr(C)]
pub struct Vertex3D {
    position: [f32; 3],
}

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
    scale: f32,
    material_index: u32,
}
