use vulkano::buffer::BufferContents;

pub mod chunk;
pub mod loader;
pub mod voxel;

#[derive(BufferContents)]
#[repr(C)]
pub struct Vertex3D {
    position: [f32; 3],
}

#[derive(BufferContents)]
#[repr(C)]
pub struct Vertex3DColor {
    position: [f32; 3],
    color: [f32; 4],
}

#[derive(Debug, Default)]
pub struct HostVoxel {
    scale: f32,
    material_index: u32,
}
