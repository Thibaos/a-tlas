//! The world's sparse voxel storage and the world side of the renderer input
//! contract boundary.
//!
//! A [`World`] is a flat sparse set of occupied voxels (global `IVec3` →
//! palette index). The world stays region-agnostic: it knows only its voxel
//! extent (imported from [`crate::grid`]) and never a Region/Micro-chunk
//! constant. Its extent derives from the renderer lattice, so it can never
//! hold a voxel the renderer cannot represent.

use std::{collections::HashMap, fmt::Display};

use dot_vox::DotVoxData;
use glam::{IVec3, UVec3, Vec4, Vec4Swizzles};

use crate::grid;
use crate::world::{HostVoxel, loader::SceneGraphTraverser};

/// The voxel length in meters (CONTEXT.md "Voxel Scale": 1 voxel = 1/16 m).
/// Referenced by the domain definition but not yet consumed by physics/player.
#[allow(dead_code)]
pub const VOXEL_PHYSICAL_LENGTH: f32 = 1.0 / 16.0;

/// How out-of-lattice voxels are handled at load time.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BoundsPolicy {
    /// Reject: an out-of-lattice voxel panics (the production default).
    Panic,
    /// Clip: an out-of-lattice voxel is dropped whole.
    Clip,
}

/// The sparse world: one `HostVoxel` per occupied global coordinate.
#[derive(Debug, Default)]
pub struct World {
    inner: HashMap<IVec3, HostVoxel>,
}

impl World {
    /// Panics with the lattice extent in the message.
    fn assert_in_lattice(position: &IVec3) {
        if !grid::in_lattice(*position) {
            panic!(
                "voxel {position} outside the ±{} lattice",
                grid::LATTICE_HALF_EXTENT
            );
        }
    }

    /// Inserts a voxel, returning `true` iff it was clipped (dropped) rather
    /// than inserted.
    pub(crate) fn insert(&mut self, position: IVec3, voxel: HostVoxel, policy: BoundsPolicy) -> bool {
        if !grid::in_lattice(position) {
            match policy {
                BoundsPolicy::Panic => Self::assert_in_lattice(&position),
                BoundsPolicy::Clip => return true,
            }
        }
        self.inner.insert(position, voxel);
        false
    }

    /// Loads the world, panicking on any voxel outside the renderer lattice.
    pub fn new(voxel_data: &DotVoxData) -> Self {
        let (world, clipped) = Self::load(voxel_data, BoundsPolicy::Panic);
        debug_assert_eq!(clipped, 0);
        world
    }

    /// Loads the world, dropping (clipping) voxels outside the renderer
    /// lattice. Returns the world and the number of voxels clipped. Clipping
    /// is voxel-atomic: a voxel with any coordinate outside the lattice is
    /// dropped whole.
    pub fn new_clipped(voxel_data: &DotVoxData) -> (Self, usize) {
        Self::load(voxel_data, BoundsPolicy::Clip)
    }

    fn load(voxel_data: &DotVoxData, policy: BoundsPolicy) -> (Self, usize) {
        let mut world = World::default();

        let mut loader = SceneGraphTraverser {
            world: &mut world,
            policy,
            scene: voxel_data,
            models: vec![],
        };

        // The traverser inserts flat-model voxels directly when the .vox has
        // no scene graph; otherwise it collects the scene-graph's transformed
        // models for the loop below.
        let mut clipped = loader.traverse();

        for (translation, rotation, size, voxels) in loader.models {
            let transform = SceneGraphTraverser::to_transform(translation, rotation, size);

            for voxel in voxels {
                let local_position =
                    UVec3::new(voxel.x as u32, voxel.z as u32, size.y - voxel.y as u32 - 1)
                        .as_ivec3();

                let position = (transform
                    * Vec4::new(
                        local_position.x as f32,
                        local_position.y as f32,
                        local_position.z as f32,
                        1.0,
                    ))
                .xyz()
                .as_ivec3();

                let position = IVec3::new(position.x, position.y, -position.z);

                if world.insert(position, HostVoxel::new(voxel.i.into()), policy) {
                    clipped += 1;
                }
            }
        }

        (world, clipped)
    }

    pub fn contains(&self, position: &IVec3) -> bool {
        Self::assert_in_lattice(position);
        self.inner.contains_key(position)
    }

    pub fn get_voxel(&self, position: &IVec3) -> Option<&HostVoxel> {
        Self::assert_in_lattice(position);
        self.inner.get(position)
    }

    /// Like [`World::get_voxel`] but returns `None` instead of panicking for
    /// positions outside the world bounds (used by the reference tracer's grid
    /// walk, which may probe one cell past the occupied set).
    pub(crate) fn try_get_voxel(&self, position: &IVec3) -> Option<&HostVoxel> {
        if !grid::in_lattice(*position) {
            return None;
        }
        self.inner.get(position)
    }

    /// Inserts a voxel with the given palette index (test/validate helper).
    pub(crate) fn insert_voxel_at(&mut self, position: IVec3, material_index: u32) {
        Self::assert_in_lattice(&position);
        self.inner.insert(position, HostVoxel::new(material_index));
    }

    /// Removes the voxel at `position` (world-side edit helper used by the
    /// validator's edit-at-the-seam flow). Returns whether a voxel was present.
    pub(crate) fn remove_voxel_at(&mut self, position: IVec3) -> bool {
        Self::assert_in_lattice(&position);
        self.inner.remove(&position).is_some()
    }

    /// Iterates every occupied voxel as (global position, voxel).
    ///
    /// The validator's reference tracer reads the world side of the renderer
    /// input contract through this iterator (plus `get_voxel` and the palette)
    /// — it never touches renderer state.
    pub fn iter_voxels(&self) -> impl Iterator<Item = (IVec3, &HostVoxel)> + '_ {
        self.inner.iter().map(|(position, voxel)| (*position, voxel))
    }

    /// The inclusive axis-aligned bounds of the occupied voxel set, or `None`
    /// for an empty world.
    pub fn voxel_bounds(&self) -> Option<(IVec3, IVec3)> {
        let mut min: Option<IVec3> = None;
        let mut max: Option<IVec3> = None;

        for (position, _) in self.iter_voxels() {
            min = Some(match min {
                None => position,
                Some(m) => m.min(position),
            });
            max = Some(match max {
                None => position,
                Some(m) => m.max(position),
            });
        }

        min.zip(max)
    }

    /// The number of occupied voxels in the world.
    pub fn voxel_count(&self) -> usize {
        self.inner.len()
    }
}

impl Display for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "World {{ voxels: {} }}", self.inner.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::HostVoxel;

    #[test]
    fn insert_and_contains() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(1, 1, 1), 1);
        world.insert_voxel_at(IVec3::new(-8, 19, -15), 2);
        assert!(world.contains(&IVec3::new(1, 1, 1)));
        assert!(world.contains(&IVec3::new(-8, 19, -15)));
        assert!(!world.contains(&IVec3::new(0, 0, 0)));
    }

    #[test]
    fn voxel_count_and_bounds() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(0, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(5, -3, 2), 2);
        assert_eq!(world.voxel_count(), 2);
        assert_eq!(
            world.voxel_bounds(),
            Some((IVec3::new(0, -3, 0), IVec3::new(5, 0, 2)))
        );
    }

    /// The world extent is half-open: -2048 is in, 2048 is out.
    #[test]
    fn world_extent_is_half_open() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(-2048, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(2047, 0, 0), 2);
        assert!(world.contains(&IVec3::new(-2048, 0, 0)));
        assert!(world.contains(&IVec3::new(2047, 0, 0)));
    }

    /// An out-of-lattice voxel panics at insert (the production boundary).
    #[test]
    #[should_panic(expected = "outside the ±2048 lattice")]
    fn insert_rejects_beyond_lattice() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(2048, 0, 0), 1);
    }

    /// The clip path drops out-of-lattice voxels whole, keeping the in-lattice
    /// subset.
    #[test]
    fn clip_drops_out_of_lattice_voxels() {
        let mut world = World::default();
        assert!(!world.insert(IVec3::new(0, 0, 0), HostVoxel::new(1), BoundsPolicy::Clip));
        assert!(world.insert(IVec3::new(3000, 0, 0), HostVoxel::new(2), BoundsPolicy::Clip));
        assert!(world.insert(IVec3::new(0, -3000, 0), HostVoxel::new(3), BoundsPolicy::Clip));
        assert_eq!(world.voxel_count(), 1);
        assert_eq!(
            world.iter_voxels().next().map(|(p, _)| p),
            Some(IVec3::new(0, 0, 0))
        );
    }
}
