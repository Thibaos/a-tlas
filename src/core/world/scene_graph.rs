use dot_vox::{DotVoxData, Rotation, SceneNode, Voxel};
use glam::{IVec3, Mat4, UVec3, Vec3A, Vec3Swizzles};

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
                ) {
                    clipped = clipped.saturating_add(1);
                }
            }
            clipped
        } else {
            self.traverse_recursive(0, IVec3::ZERO, Rotation::IDENTITY);
            0
        }
    }

    fn traverse_recursive(&mut self, node: u32, translation: glam::IVec3, rotation: Rotation) {
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
                    panic!("transform node must have exactly one frame, got {}", frames.len());
                };

                let this_translation = frame
                    .position()
                    .map_or(IVec3::ZERO, |position| IVec3 {
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
                    panic!("shape node must have exactly one model, got {}", models.len());
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

    pub fn to_transform(translation: glam::IVec3, rotation: Rotation, size: glam::UVec3) -> Mat4 {
        use std::ops::{Add, Mul, Sub};

        let mut translation = translation.as_vec3a().xzy();
        translation.z *= -1.0;

        let (quat, scale) = rotation.to_quat_scale();
        let quat = glam::Quat::from_array(quat);
        let quat = glam::Quat::from_xyzw(quat.x, quat.z, -quat.y, quat.w);
        let scale = glam::Vec3A::from_array(scale).xzy();

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

#[allow(clippy::arithmetic_side_effects)] // dot_vox composes packed rotation bitfields
fn compose(rotation: Rotation, next: Rotation) -> Rotation {
    rotation * next
}
