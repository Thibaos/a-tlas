use glam::{Mat4, Vec3, Vec4Swizzles};

/// Six frustum planes: left, right, bottom, top, near, far.
/// Each plane is stored as (normal, distance) where the inside half-space
/// satisfies `normal · point + distance >= 0`.
#[derive(Clone)]
pub struct Frustum {
    pub planes: [(Vec3, f32); 6],
}

impl Frustum {
    /// Extract frustum planes from a world→clip matrix `P * V`.
    /// Assumes Vulkan's [0, 1] depth range.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        // Vulkan [0,1] clip Z:
        //   left   = row3 + row0
        //   right  = row3 - row0
        //   bottom = row3 + row1
        //   top    = row3 - row1
        //   near   = row2          (z >= 0 in clip space)
        //   far    = row3 - row2    (z <= w in clip space)
        let raw = [
            row3 + row0, // left
            row3 - row0, // right
            row3 + row1, // bottom
            row3 - row1, // top
            row2,        // near
            row3 - row2, // far
        ];

        let mut planes = [(Vec3::ZERO, 0.0f32); 6];
        for (i, plane) in raw.iter().enumerate() {
            let normal = plane.xyz();
            let len = normal.length();
            planes[i] = (normal / len, plane.w / len);
        }

        Frustum { planes }
    }

    /// Returns `true` if the AABB is at least partially inside the frustum.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for (normal, distance) in &self.planes {
            // p-vertex: the AABB corner that maximizes dot(normal, point)
            let p = Vec3::new(
                if normal.x > 0.0 { max.x } else { min.x },
                if normal.y > 0.0 { max.y } else { min.y },
                if normal.z > 0.0 { max.z } else { min.z },
            );
            if normal.dot(p) + *distance < 0.0 {
                return false; // AABB completely outside this plane
            }
        }
        true // AABB intersects all planes
    }
}
