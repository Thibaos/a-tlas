use dot_vox::{DotVoxData, Rotation, SceneNode, Voxel};
use glam::{IVec3, UVec3};

#[cfg(test)]
use glam::{Mat4, Vec3A, Vec3Swizzles};

use crate::core::world::{InsertResult, grid};

use super::{BoundsPolicy, World};

pub struct SceneGraphTraverser<'a> {
    pub world: &'a mut World,
    pub policy: BoundsPolicy,
    pub scene: &'a DotVoxData,
    pub models: Vec<(IVec3, Rotation, UVec3, Vec<Voxel>)>,
}

impl SceneGraphTraverser<'_> {
    pub fn traverse(&mut self) -> usize {
        if self.scene.scenes.is_empty() {
            let mut clipped = 0usize;
            for voxel in self.scene.models.iter().flat_map(|model| &model.voxels) {
                if self.world.insert(
                    IVec3::new(i32::from(voxel.x), i32::from(voxel.z), i32::from(voxel.y)),
                    u32::from(voxel.i),
                    self.policy,
                ) == InsertResult::Clipped
                {
                    clipped = clipped.saturating_add(1);
                }
            }
            clipped
        } else {
            self.traverse_recursive(0, IVec3::ZERO, Rotation::IDENTITY);
            0
        }
    }

    fn traverse_recursive(&mut self, node: u32, translation: IVec3, rotation: Rotation) {
        let Some(node) = self
            .scene
            .scenes
            .get(usize::try_from(node).unwrap_or(usize::MAX))
        else {
            panic!("scene node {node} out of range");
        };

        match node {
            SceneNode::Transform { frames, child, .. } => {
                let [frame] = &frames[..] else {
                    panic!(
                        "transform node must have exactly one frame, got {}",
                        frames.len()
                    );
                };

                let this_translation = frame.position().map_or(IVec3::ZERO, |position| IVec3 {
                    x: position.x,
                    y: position.y,
                    z: position.z,
                });

                let this_rotation = frame.orientation().unwrap_or(Rotation::IDENTITY);

                let translation = translation.saturating_add(this_translation);

                self.traverse_recursive(*child, translation, compose(rotation, this_rotation));
            }
            SceneNode::Group { children, .. } => {
                for child in children {
                    self.traverse_recursive(*child, translation, rotation);
                }
            }
            SceneNode::Shape { models, .. } => {
                let [shape_model] = models.as_slice() else {
                    panic!(
                        "shape node must have exactly one model, got {}",
                        models.len()
                    );
                };

                let model = self
                    .scene
                    .models
                    .get(usize::try_from(shape_model.model_id).unwrap_or(usize::MAX))
                    .unwrap_or_else(|| panic!("shape model {} out of range", shape_model.model_id));

                if model.voxels.is_empty() {
                    return;
                }

                let size = model.size;

                self.models.push((
                    translation,
                    rotation,
                    UVec3::new(size.x, size.y, size.z),
                    model.voxels.clone(),
                ));
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn legacy_float_transform(
        translation: IVec3,
        rotation: Rotation,
        size: UVec3,
    ) -> Mat4 {
        use std::ops::{Add, Mul, Sub};

        use glam::Quat;

        let mut translation = translation.as_vec3a().xzy();
        translation.z *= -1.0;

        let (quat, scale) = rotation.to_quat_scale();
        let quat = Quat::from_array(quat);
        let quat = Quat::from_xyzw(quat.x, quat.z, -quat.y, quat.w);
        let scale = Vec3A::from_array(scale).xzy();

        let mut offset = Vec3A::new(
            if size.x.is_multiple_of(2) { 0.0 } else { 0.5 },
            if size.z.is_multiple_of(2) { 0.0 } else { 0.5 },
            if size.y.is_multiple_of(2) { 0.0 } else { -0.5 },
        );
        offset = quat.mul_vec3a(offset);

        let center = quat.mul_vec3a(size.xzy().as_vec3a().mul(0.5));

        Mat4::from_scale_rotation_translation(
            scale.into(),
            quat,
            translation.sub(center.mul(scale)).add(offset).into(),
        )
    }
}

pub struct VoxelPlacement {
    rows: [IVec3; 3],
    translation: IVec3,
    center: IVec3,
    size: UVec3,
}

impl VoxelPlacement {
    pub fn new(translation: IVec3, rotation: Rotation, size: UVec3) -> Self {
        let columns = rotation.to_cols_array_2d();
        let one = 1.0_f32.to_bits();
        let minus_one = (-1.0_f32).to_bits();

        let unit_row = |row: &[f32; 3]| -> IVec3 {
            let mut picked: Option<(usize, i32)> = None;
            for (column, value) in row.iter().enumerate() {
                let sign = match value.to_bits() {
                    bits if bits == one => 1,
                    bits if bits == minus_one => -1,
                    _ => 0,
                };

                if sign == 0 {
                    continue;
                }

                if picked.is_some() {
                    panic!("rotation row {row:?} is not a signed permutation");
                }

                picked = Some((column, sign));
            }

            match picked
                .unwrap_or_else(|| panic!("rotation row {row:?} is not a signed permutation"))
            {
                (0, sign) => IVec3::new(sign, 0, 0),
                (1, sign) => IVec3::new(0, sign, 0),
                (_, sign) => IVec3::new(0, 0, sign),
            }
        };

        let model_rows = [
            unit_row(&[columns[0][0], columns[1][0], columns[2][0]]),
            unit_row(&[columns[0][1], columns[1][1], columns[2][1]]),
            unit_row(&[columns[0][2], columns[1][2], columns[2][2]]),
        ];

        let determinant = model_rows[0].dot(model_rows[1].cross(model_rows[2]));

        Self {
            rows: [model_rows[0], model_rows[2], model_rows[1]],
            translation: IVec3::new(translation.x, translation.z, translation.y),
            center: IVec3::new(
                half_center(size.x, determinant),
                half_center(size.y, determinant),
                half_center(size.z, determinant),
            ),
            size,
        }
    }

    pub fn misses_lattice(&self) -> bool {
        let half = grid::LATTICE_HALF_EXTENT.cast_signed();
        let neg_half = half.wrapping_neg();

        let corner = |extent: u32| {
            u8::try_from(extent.min(256).saturating_sub(1))
                .unwrap_or_else(|_| panic!("model extent {extent} out of voxel range"))
        };
        let low = self.project(0, 0, 0);
        let high = self.project(
            i32::from(corner(self.size.x)),
            i32::from(corner(self.size.y)),
            i32::from(corner(self.size.z)),
        );
        let min = low.min(high);
        let max = low.max(high);

        min.x >= half
            || min.y >= half
            || min.z >= half
            || max.x < neg_half
            || max.y < neg_half
            || max.z < neg_half
    }

    pub fn place(&self, voxel: Voxel) -> IVec3 {
        self.project(i32::from(voxel.x), i32::from(voxel.y), i32::from(voxel.z))
    }

    fn project(&self, x: i32, y: i32, z: i32) -> IVec3 {
        let shifted = IVec3::new(
            x.wrapping_sub(self.center.x),
            y.wrapping_add(1).wrapping_sub(self.center.y),
            z.wrapping_sub(self.center.z),
        );

        IVec3::new(
            self.translation.x.saturating_add(self.rows[0].dot(shifted)),
            self.translation.y.saturating_add(self.rows[1].dot(shifted)),
            self.translation.z.saturating_add(self.rows[2].dot(shifted)),
        )
    }
}

fn half_center(size: u32, determinant: i32) -> i32 {
    let half = if determinant < 0 {
        size.div_euclid(2).saturating_add(size & 1)
    } else {
        size.div_euclid(2)
    };

    i32::try_from(half)
        .unwrap_or_else(|_| panic!("model size {size} too large for integer placement"))
}

#[allow(clippy::arithmetic_side_effects)] // dot_vox composes packed rotation bitfields
fn compose(rotation: Rotation, next: Rotation) -> Rotation {
    rotation * next
}
