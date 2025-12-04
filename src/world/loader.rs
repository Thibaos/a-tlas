use dot_vox::{DotVoxData, Rotation, SceneNode, Voxel};
use glam::{IVec3, Mat4, UVec3, Vec3A, Vec3Swizzles};

use crate::world::HostVoxel;

use super::chunk::{Chunks, ChunksInner};

pub struct SceneGraphTraverser<'a> {
    pub chunks: &'a mut ChunksInner,
    pub scene: &'a DotVoxData,
    pub models: Vec<(IVec3, Rotation, UVec3, Vec<Voxel>)>,
}

impl SceneGraphTraverser<'_> {
    pub fn traverse(&mut self) {
        if self.scene.scenes.is_empty() {
            for voxel in self.scene.models.iter().flat_map(|model| &model.voxels) {
                Chunks::insert_voxel(
                    self.chunks,
                    IVec3::new(voxel.x as i32, voxel.z as i32, voxel.y as i32),
                    HostVoxel {
                        scale: 1.0,
                        material_index: voxel.i as u32,
                    },
                );
            }
        } else {
            self.traverse_recursive(0, IVec3::ZERO, Rotation::IDENTITY);
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

                self.traverse_recursive(*child, translation, this_rotation);
            }
            SceneNode::Group { children, .. } => {
                for child in children {
                    self.traverse_recursive(*child, IVec3::ZERO, Rotation::IDENTITY);
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
        let scale = glam::Vec3A::from_array(scale).xzy(); // no need to negate scale.y because scale is not a coordinate

        let mut offset = Vec3A::new(
            if size.x % 2 == 0 { 0.0 } else { 0.5 },
            if size.z % 2 == 0 { 0.0 } else { 0.5 },
            if size.y % 2 == 0 { 0.0 } else { -0.5 },
        );
        offset = quat.mul_vec3a(offset); // If another seam shows up in the future, try multiplying this with `scale`

        let center = quat * (size.xzy().as_vec3a() / 2.0);

        Mat4::from_scale_rotation_translation(
            scale.into(),
            quat,
            (translation - center * scale + offset).into(),
        )
    }
}
