use std::{collections::HashMap, fmt::Display};

use dot_vox::DotVoxData;
use glam::{IVec3, UVec3, Vec4, Vec4Swizzles};

use crate::core::world::scene_graph::SceneGraphTraverser;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BoundsPolicy {
    Panic,
    Clip,
}

pub mod format;
pub mod grid;
pub mod scene_graph;
pub mod snapshot;

#[derive(Debug, Default)]
pub struct World {
    inner: HashMap<IVec3, u32>,
}

impl World {
    fn assert_in_lattice(position: &IVec3) {
        if !grid::in_lattice(*position) {
            panic!(
                "voxel {position} outside the ±{} lattice",
                grid::LATTICE_HALF_EXTENT
            );
        }
    }

    pub(crate) fn insert(&mut self, position: IVec3, voxel: u32, policy: BoundsPolicy) -> bool {
        if !grid::in_lattice(position) {
            match policy {
                BoundsPolicy::Panic => Self::assert_in_lattice(&position),
                BoundsPolicy::Clip => return true,
            }
        }
        self.inner.insert(position, voxel);
        false
    }

    pub fn new(voxel_data: &DotVoxData) -> Self {
        let (world, clipped) = Self::load(voxel_data, BoundsPolicy::Panic);
        debug_assert_eq!(clipped, 0);
        world
    }

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

                if world.insert(position, voxel.i.into(), policy) {
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

    pub fn get_voxel(&self, position: &IVec3) -> Option<&u32> {
        Self::assert_in_lattice(position);
        self.inner.get(position)
    }

    #[allow(dead_code)]
    pub(crate) fn try_get_voxel(&self, position: &IVec3) -> Option<&u32> {
        if !grid::in_lattice(*position) {
            return None;
        }
        self.inner.get(position)
    }

    #[allow(dead_code)]
    pub(crate) fn insert_voxel_at(&mut self, position: IVec3, material_index: u32) {
        Self::assert_in_lattice(&position);
        self.inner.insert(position, material_index);
    }

    #[allow(dead_code)]
    pub(crate) fn remove_voxel_at(&mut self, position: IVec3) -> bool {
        Self::assert_in_lattice(&position);
        self.inner.remove(&position).is_some()
    }

    pub fn iter_voxels(&self) -> impl Iterator<Item = (IVec3, &u32)> + '_ {
        self.inner
            .iter()
            .map(|(position, voxel)| (*position, voxel))
    }

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

    #[test]
    fn world_extent_is_half_open() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(-2048, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(2047, 0, 0), 2);
        assert!(world.contains(&IVec3::new(-2048, 0, 0)));
        assert!(world.contains(&IVec3::new(2047, 0, 0)));
    }

    #[test]
    #[should_panic(expected = "outside the ±2048 lattice")]
    fn insert_rejects_beyond_lattice() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(2048, 0, 0), 1);
    }

    #[test]
    fn clip_drops_out_of_lattice_voxels() {
        let mut world = World::default();
        assert!(!world.insert(IVec3::new(0, 0, 0), 1, BoundsPolicy::Clip));
        assert!(world.insert(IVec3::new(3000, 0, 0), 2, BoundsPolicy::Clip));
        assert!(world.insert(IVec3::new(0, -3000, 0), 3, BoundsPolicy::Clip));
        assert_eq!(world.voxel_count(), 1);
        assert_eq!(
            world.iter_voxels().next().map(|(p, _)| p),
            Some(IVec3::new(0, 0, 0))
        );
    }
}
