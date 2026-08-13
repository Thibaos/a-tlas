use dot_vox::{DotVoxData, Rotation, SceneNode, Voxel};
use glam::{IVec3, Mat4, UVec3, Vec3A, Vec3Swizzles};

use crate::world::HostVoxel;

use super::world::{BoundsPolicy, World};

pub struct SceneGraphTraverser<'a> {
    pub world: &'a mut World,
    pub policy: BoundsPolicy,
    pub scene: &'a DotVoxData,
    pub models: Vec<(IVec3, Rotation, UVec3, Vec<Voxel>)>,
}

impl SceneGraphTraverser<'_> {
    /// Walks the scene graph. Scene-less .vox files have their flat model
    /// voxels inserted directly (returning the number clipped); otherwise the
    /// transformed models are collected into `models` for the caller to
    /// insert. Returns the number of voxels clipped by the direct path.
    pub fn traverse(&mut self) -> usize {
        if self.scene.scenes.is_empty() {
            let mut clipped = 0usize;
            for voxel in self.scene.models.iter().flat_map(|model| &model.voxels) {
                if self.world.insert(
                    IVec3::new(voxel.x as i32, voxel.z as i32, voxel.y as i32),
                    HostVoxel::new(voxel.i as u32),
                    self.policy,
                ) {
                    clipped += 1;
                }
            }
            clipped
        } else {
            self.traverse_recursive(0, IVec3::ZERO, Rotation::IDENTITY);
            0
        }
    }

    pub fn traverse_recursive(&mut self, node: u32, translation: glam::IVec3, rotation: Rotation) {
        let node = &self.scene.scenes[node as usize];
        match node {
            SceneNode::Transform { frames, child, .. } => {
                if frames.len() != 1 {
                    unimplemented!("Multiple frames in transform node");
                }
                let frame = &frames[0];
                let this_translation = frame
                    .position()
                    .map(|position| IVec3 {
                        x: position.x,
                        y: position.y,
                        z: position.z,
                    })
                    .unwrap_or(IVec3::ZERO);

                let this_rotation = frame.orientation().unwrap_or(Rotation::IDENTITY);

                let translation = translation + this_translation;

                self.traverse_recursive(*child, translation, rotation * this_rotation);
            }
            SceneNode::Group { children, .. } => {
                for child in children {
                    self.traverse_recursive(*child, translation, rotation);
                }
            }
            SceneNode::Shape { models, .. } => {
                if models.len() != 1 {
                    unimplemented!("Multiple shape models in Shape node");
                }
                let shape_model = &models[0];
                let model = &self.scene.models[shape_model.model_id as usize];
                if model.voxels.is_empty() {
                    return;
                }

                let size = self.scene.models[shape_model.model_id as usize].size;

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

        let center = quat * (size.xzy().as_vec3a() / 2.0);

        Mat4::from_scale_rotation_translation(
            scale.into(),
            quat,
            (translation - center * scale + offset).into(),
        )
    }
}
