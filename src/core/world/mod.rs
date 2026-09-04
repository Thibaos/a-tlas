use std::{collections::HashMap, fmt::Display};

use dot_vox::DotVoxData;
use rustc_hash::FxBuildHasher;
use glam::IVec3;

use crate::core::world::scene_graph::{SceneGraphTraverser, VoxelPlacement};

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

        let mut placements = Vec::with_capacity(models.len());
        let mut live = 0usize;

        for (translation, rotation, size, voxels) in models {
            let placement = VoxelPlacement::new(translation, rotation, size);

            if policy == BoundsPolicy::Clip && placement.misses_lattice() {
                clipped = clipped.saturating_add(voxels.len());
            } else {
                live = live.saturating_add(voxels.len());
                placements.push((placement, voxels));
            }
        }

        world.inner.reserve(live);

        for (placement, voxels) in placements {
            for voxel in voxels {
                let position = placement.place(voxel);

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
#[derive(Debug)]
struct ModelSpec {
    size: (u32, u32, u32),
    voxels: Vec<dot_vox::Voxel>,
    rotation: u8,
    translation: [i32; 3],
}

#[cfg(test)]
fn scene_fixture(specs: &[ModelSpec]) -> DotVoxData {
    use dot_vox::{Dict, Frame, Model, SceneNode, ShapeModel, Size};

    let children: Vec<u32> = (0..specs.len()).map(|k| (2 * k + 1) as u32).collect();
    let mut scenes = vec![SceneNode::Group {
        attributes: Dict::new(),
        children,
    }];

    let mut models = Vec::new();
    for (k, spec) in specs.iter().enumerate() {
        let mut frame = Dict::new();
        frame.insert("_r".to_owned(), spec.rotation.to_string());
        frame.insert(
            "_t".to_owned(),
            format!("{} {} {}", spec.translation[0], spec.translation[1], spec.translation[2]),
        );

        scenes.push(SceneNode::Transform {
            attributes: Dict::new(),
            frames: vec![Frame::new(frame)],
            child: (2 * k + 2) as u32,
            layer_id: 0,
        });
        scenes.push(SceneNode::Shape {
            attributes: Dict::new(),
            models: vec![ShapeModel {
                model_id: k as u32,
                attributes: Dict::new(),
            }],
        });
        models.push(Model {
            size: Size {
                x: spec.size.0,
                y: spec.size.1,
                z: spec.size.2,
            },
            voxels: spec.voxels.clone(),
        });
    }

    DotVoxData {
        version: 200,
        index_map: Vec::new(),
        models,
        palette: Vec::new(),
        materials: Vec::new(),
        scenes,
        layers: Vec::new(),
    }
}

#[cfg(test)]
mod placement_differential {
    use std::collections::HashMap;
    use std::ops::Neg;

    use dot_vox::{DotVoxData, Rotation, Voxel};
    use glam::{DMat4, DQuat, DVec3, DVec4, IVec3, IVec4, UVec3, Vec4Swizzles};

    use super::grid;
    use super::scene_graph::SceneGraphTraverser;
    use super::{BoundsPolicy, ModelSpec, World, scene_fixture};

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    const TRANSLATIONS: &[i32] = &[
        0, 1, -1, 3, -3, 2047, -2047, 2048, -2048, 2049, -2049, 100_000, -100_000, 1_000_000,
        -1_000_000,
    ];

    fn rotation_bytes() -> Vec<u8> {
        (0u8..128)
            .filter(|byte| {
                let first = byte & 0b11;
                let second = (byte >> 2) & 0b11;
                first != 0b11 && second != 0b11 && first != second
            })
            .collect()
    }

    fn random_size(rng: &mut Rng) -> (u32, u32, u32) {
        let mut axis = || {
            if rng.below(8) == 0 {
                250 + rng.below(7) as u32
            } else {
                1 + rng.below(8) as u32
            }
        };
        (axis(), axis(), axis())
    }

    fn random_voxels(rng: &mut Rng, size: (u32, u32, u32)) -> Vec<Voxel> {
        let count = 1 + rng.below(12) as usize;
        (0..count)
            .map(|_| Voxel {
                x: rng.below(size.0 as u64) as u8,
                y: rng.below(size.1 as u64) as u8,
                z: rng.below(size.2 as u64) as u8,
                i: rng.below(256) as u8,
            })
            .collect()
    }

    fn random_translation(rng: &mut Rng) -> [i32; 3] {
        let mut pick = || {
            if rng.below(2) == 0 {
                TRANSLATIONS[rng.below(TRANSLATIONS.len() as u64) as usize]
            } else {
                (rng.below(4_000_001) as i64 - 2_000_000) as i32
            }
        };
        [pick(), pick(), pick()]
    }

    fn collected_models(data: &DotVoxData) -> Vec<(IVec3, Rotation, UVec3, Vec<Voxel>)> {
        let mut world = World::default();
        let mut loader = SceneGraphTraverser {
            world: &mut world,
            policy: BoundsPolicy::Clip,
            scene: data,
            models: Vec::new(),
        };
        loader.traverse();
        std::mem::take(&mut loader.models)
    }

    fn rounded(value: f64) -> i32 {
        (value + 0.5).floor() as i32
    }

    fn exact_oracle_transform(translation: IVec3, rotation: Rotation, size: UVec3) -> DMat4 {
        let (quat, scale) = rotation.to_quat_scale();
        let quat = DQuat::from_xyzw(
            f64::from(quat[0]),
            f64::from(quat[1]),
            f64::from(quat[2]),
            f64::from(quat[3]),
        );
        let quat = DQuat::from_xyzw(quat.x, quat.z, -quat.y, quat.w);
        let scale = DVec3::new(f64::from(scale[0]), f64::from(scale[2]), f64::from(scale[1]));

        let translation = DVec3::new(
            f64::from(translation.x),
            f64::from(translation.z),
            -f64::from(translation.y),
        );

        let mut offset = DVec3::new(
            if size.x & 1 == 1 { 0.5 } else { 0.0 },
            if size.z & 1 == 1 { 0.5 } else { 0.0 },
            if size.y & 1 == 1 { -0.5 } else { 0.0 },
        );
        offset = quat.mul_vec3(offset);

        let center =
            quat.mul_vec3(DVec3::new(f64::from(size.x), f64::from(size.z), f64::from(size.y)) * 0.5);

        DMat4::from_scale_rotation_translation(scale, quat, translation - center * scale + offset)
    }

    fn exact_oracle_map(data: &DotVoxData) -> HashMap<IVec3, u32> {
        let mut map = HashMap::new();
        for (translation, rotation, size, voxels) in collected_models(data) {
            let transform = exact_oracle_transform(translation, rotation, size);
            for voxel in voxels {
                let local = DVec3::new(
                    f64::from(voxel.x),
                    f64::from(voxel.z),
                    f64::from(size.y) - f64::from(voxel.y) - 1.0,
                );
                let placed = transform.mul_vec4(DVec4::new(local.x, local.y, local.z, 1.0));
                let position = IVec3::new(rounded(placed.x), rounded(placed.y), -rounded(placed.z));
                if grid::in_lattice(position) {
                    map.insert(position, u32::from(voxel.i));
                }
            }
        }
        map
    }

    fn legacy_float_map(data: &DotVoxData) -> HashMap<IVec3, u32> {
        let mut map = HashMap::new();
        for (translation, rotation, size, voxels) in collected_models(data) {
            let transform = SceneGraphTraverser::legacy_float_transform(translation, rotation, size);
            for voxel in voxels.iter().copied() {
                let local = UVec3::new(
                    u32::from(voxel.x),
                    u32::from(voxel.z),
                    size.y.strict_sub(u32::from(voxel.y)).strict_sub(1),
                )
                .as_ivec3();
                let position = IVec4::new(local.x, local.y, local.z, 1).as_vec4();
                let position = (transform.mul_vec4(position)).xyz().as_ivec3();
                let position = IVec3::new(position.x, position.y, position.z.neg());
                if grid::in_lattice(position) {
                    map.insert(position, u32::from(voxel.i));
                }
            }
        }
        map
    }

    fn production_map(data: &DotVoxData) -> HashMap<IVec3, u32> {
        let (world, _) = World::new_clipped(data);
        world
            .iter_voxels()
            .map(|(position, voxel)| (position, *voxel))
            .collect()
    }

    fn random_specs(rng: &mut Rng, rotation: u8) -> Vec<ModelSpec> {
        let mut specs = Vec::new();
        for _ in 0..(1 + rng.below(4)) {
            let size = random_size(rng);
            specs.push(ModelSpec {
                size,
                voxels: random_voxels(rng, size),
                rotation,
                translation: random_translation(rng),
            });
        }
        specs
    }

    #[test]
    fn integer_placement_matches_exact_oracle_on_randomized_scenes() {
        let mut rng = Rng::new(0x5EED_2024);
        let valid_rotations = rotation_bytes();

        for rotation in &valid_rotations {
            for _ in 0..8 {
                let specs = random_specs(&mut rng, *rotation);
                let data = scene_fixture(&specs);
                assert_eq!(
                    production_map(&data),
                    exact_oracle_map(&data),
                    "rotation {rotation:#010b}, specs {specs:?}"
                );
            }
        }

        for _ in 0..32 {
            let specs: Vec<ModelSpec> = (0..(1 + rng.below(4)))
                .map(|_| {
                    let size = random_size(&mut rng);
                    ModelSpec {
                        size,
                        voxels: random_voxels(&mut rng, size),
                        rotation: valid_rotations[rng.below(valid_rotations.len() as u64) as usize],
                        translation: random_translation(&mut rng),
                    }
                })
                .collect();
            let data = scene_fixture(&specs);
            assert_eq!(production_map(&data), exact_oracle_map(&data), "specs {specs:?}");
        }
    }

    #[test]
    fn integer_placement_matches_legacy_float_on_exact_quaternion_classes() {
        let exact_pairs = [(0u8, 1u8), (1, 2), (2, 0)];
        let mut rng = Rng::new(0xC0FFEE);

        for rotation in rotation_bytes() {
            let pair = (rotation & 0b11, (rotation >> 2) & 0b11);
            if !exact_pairs.contains(&pair) {
                continue;
            }

            for _ in 0..8 {
                let specs = random_specs(&mut rng, rotation);
                let data = scene_fixture(&specs);
                assert_eq!(
                    production_map(&data),
                    legacy_float_map(&data),
                    "rotation {rotation:#010b}, specs {specs:?}"
                );
            }
        }
    }

    #[test]
    #[ignore = "asset: cargo test --release church_matches_legacy_float_path -- --ignored --nocapture"]
    fn church_matches_legacy_float_path() {
        let data = dot_vox::load("assets/church.vox").unwrap();
        assert_eq!(production_map(&data), legacy_float_map(&data));
    }

    #[test]
    #[ignore = "asset: cargo test --release bistro_matches_legacy_float_path -- --ignored --nocapture"]
    fn bistro_matches_legacy_float_path() {
        let data = dot_vox::load("assets/bistro.vox").unwrap();
        assert_eq!(production_map(&data), legacy_float_map(&data));
    }

    #[test]
    fn placement_known_example_from_legacy_pipeline() {
        let specs = [ModelSpec {
            size: (1, 1, 1),
            voxels: vec![Voxel { x: 0, y: 0, z: 0, i: 7 }],
            rotation: 0b0001,
            translation: [5, 7, 9],
        }];
        let world = World::new(&scene_fixture(&specs));

        assert_eq!(world.voxel_count(), 1);
        assert_eq!(
            world.get_voxel(&IVec3::new(5, 8, 6)),
            Some(&7),
            "the legacy pipeline evaluates this origin voxel to m = (5, 8, -6), so the world position after the final z negation is (5, 8, 6)"
        );
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

    #[test]
    fn scene_path_flips_rows_without_underflow() {
        let data = scene_fixture(&[ModelSpec {
            size: (2, 2, 2),
            voxels: vec![
                dot_vox::Voxel { x: 0, y: 0, z: 0, i: 0 },
                dot_vox::Voxel { x: 0, y: 1, z: 0, i: 0 },
            ],
            rotation: 0b0000100,
            translation: [0, 0, 0],
        }]);
        let world = World::new(&data);

        assert_eq!(world.voxel_count(), 2);
        assert!(world.contains(&IVec3::new(-1, -1, 0)));
        assert!(world.contains(&IVec3::new(-1, -1, 1)));
    }
}
