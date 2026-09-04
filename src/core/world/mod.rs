use std::{collections::HashMap, fmt::Display, ops::Neg};

use dot_vox::DotVoxData;
use rustc_hash::FxBuildHasher;
use glam::{IVec3, IVec4, UVec3, Vec4Swizzles};

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

#[cfg(test)]
mod bench;

#[derive(Debug, Default)]
pub struct World {
    inner: HashMap<IVec3, u32, FxBuildHasher>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum InsertResult {
    Ok,
    Clipped,
    Existing,
}

impl World {
    fn assert_in_lattice(position: &IVec3) {
        assert!(
            grid::in_lattice(*position),
            "voxel {position} outside the ±{} lattice",
            grid::LATTICE_HALF_EXTENT
        );
    }

    pub(crate) fn insert(
        &mut self,
        position: IVec3,
        voxel: u32,
        policy: BoundsPolicy,
    ) -> InsertResult {
        if !grid::in_lattice(position) {
            match policy {
                BoundsPolicy::Panic => Self::assert_in_lattice(&position),
                BoundsPolicy::Clip => return InsertResult::Clipped,
            }
        }

        if self.inner.insert(position, voxel).is_some() {
            InsertResult::Existing
        } else {
            InsertResult::Ok
        }
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
        let mut world = Self::default();

        if voxel_data.scenes.is_empty() {
            let direct = voxel_data
                .models
                .iter()
                .map(|model| model.voxels.len())
                .sum();
            world.inner.reserve(direct);
        }

        let mut loader = SceneGraphTraverser {
            world: &mut world,
            policy,
            scene: voxel_data,
            models: vec![],
        };

        let mut clipped = loader.traverse();

        let models = std::mem::take(&mut loader.models);

        world.inner
            .reserve(models.iter().map(|(.., voxels)| voxels.len()).sum());

        for (translation, rotation, size, voxels) in models {
            let transform = SceneGraphTraverser::to_transform(translation, rotation, size);

            for voxel in voxels {
                let local_position = UVec3::new(
                    u32::from(voxel.x),
                    u32::from(voxel.z),
                    size.y.strict_sub(u32::from(voxel.y)).strict_sub(1),
                )
                .as_ivec3();

                let position =
                    IVec4::new(local_position.x, local_position.y, local_position.z, 1).as_vec4();

                let position = (transform.mul_vec4(position)).xyz().as_ivec3();

                let position = IVec3::new(position.x, position.y, position.z.neg());

                if world.insert(position, voxel.i.into(), policy) == InsertResult::Clipped {
                    clipped = clipped.saturating_add(1);
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

    #[cfg(test)]
    pub(crate) fn insert_voxel_at(&mut self, position: IVec3, material_index: u32) {
        Self::assert_in_lattice(&position);
        self.inner.insert(position, material_index);
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
            min = Some(min.map_or(position, |m| m.min(position)));
            max = Some(max.map_or(position, |m| m.max(position)));
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
        assert_eq!(
            world.insert(IVec3::new(0, 0, 0), 1, BoundsPolicy::Clip),
            InsertResult::Ok
        );
        assert_eq!(
            world.insert(IVec3::new(3000, 0, 0), 2, BoundsPolicy::Clip),
            InsertResult::Clipped
        );
        assert_eq!(
            world.insert(IVec3::new(0, -3000, 0), 3, BoundsPolicy::Clip),
            InsertResult::Clipped
        );
        assert_eq!(world.voxel_count(), 1);
        assert_eq!(
            world.iter_voxels().next().map(|(p, _)| p),
            Some(IVec3::new(0, 0, 0))
        );
    }

    fn scene_fixture() -> DotVoxData {
        use dot_vox::{Dict, Frame, Model, SceneNode, ShapeModel, Size, Voxel};

        DotVoxData {
            version: 200,
            index_map: Vec::new(),
            models: vec![Model {
                size: Size { x: 2, y: 2, z: 2 },
                voxels: vec![
                    Voxel {
                        x: 0,
                        y: 0,
                        z: 0,
                        i: 0,
                    },
                    Voxel {
                        x: 0,
                        y: 1,
                        z: 0,
                        i: 0,
                    },
                ],
            }],
            palette: Vec::new(),
            materials: Vec::new(),
            scenes: vec![
                SceneNode::Transform {
                    attributes: Dict::default(),
                    frames: vec![Frame::default()],
                    child: 1,
                    layer_id: 0,
                },
                SceneNode::Shape {
                    attributes: Dict::default(),
                    models: vec![ShapeModel {
                        model_id: 0,
                        attributes: Dict::default(),
                    }],
                },
            ],
            layers: Vec::new(),
        }
    }

    #[test]
    fn scene_path_flips_rows_without_underflow() {
        let world = World::new(&scene_fixture());

        assert_eq!(world.voxel_count(), 2);
        assert!(world.contains(&IVec3::new(-1, -1, 0)));
        assert!(world.contains(&IVec3::new(-1, -1, 1)));
    }
}
