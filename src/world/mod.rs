pub mod loader;
pub mod voxel;
pub mod world;

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
