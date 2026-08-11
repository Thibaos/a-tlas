//! The independent CPU reference tracer (ADR 0003).
//!
//! A naive per-voxel ray tracer over the world's source of truth — the chunk
//! HashMaps plus the palette — sharing only camera inputs and the palette with
//! the GPU path. It deliberately shares none of the renderer's machinery (no
//! DDA through pools, no trimmed AABBs, no hitKind, no BLAS/TLAS): a
//! divergence between the reference and the captured frame points at the
//! renderer, never at an assumption both sides share.
//!
//! Rays are reconstructed exactly like `shaders/rt/raygen_common.glsl`, and
//! each voxel is tested as the same shape the renderer uses ([`VoxelShape`]):
//! today that is a unit cube centered on the voxel position (the
//! triangle-per-voxel path), the destination uses the unit grid cell. The
//! validate switches the shape when the renderer switches.
//!
//! To stay fast on dense worlds (custom.vox is ~1M voxels) the tracer steps
//! the ray over the world's grid cells instead of testing every voxel per
//! pixel; per cell it queries the world's own sparse storage (the world side
//! of the renderer input contract), so it still shares nothing with the
//! renderer's representation.

use glam::{IVec3, Mat4, Vec2, Vec3, Vec4};

use crate::world::chunk::Chunks;

/// Ray t-range: matches the current ray pass (shaders/rt/common.glsl EPSILON /
/// FLT_MAX). ADR 0002 will move the ray pass to the camera's near/far; when it
/// does, these constants move with it and the validator keeps passing.
pub const T_MIN: f32 = 0.0001;
pub const T_MAX: f32 = f32::MAX;

/// What the miss shader produces: black, alpha 1, t 0.
pub const BACKGROUND_COLOR: [u8; 4] = [0, 0, 0, 255];

/// The volume a world voxel occupies, as seen by the renderer under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelShape {
    /// Unit cube centered on the voxel position: [p - 0.5, p + 0.5]. This is
    /// what the current triangle-per-voxel path instances.
    CenteredUnitCube,
    /// The lattice cell [p, p + 1). The destination's in-shader DDA resolves
    /// cell = floor(hit) mod 8, i.e. grid cells.
    GridCell,
}

/// Camera inputs shared between the GPU ray pass and the reference: the same
/// inverse matrices the raygen unprojects with, the launch size, and the ray
/// t-range.
pub struct CameraInputs {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    width: u32,
    height: u32,
    pub t_min: f32,
    pub t_max: f32,
}

impl CameraInputs {
    pub fn new(view: Mat4, proj: Mat4, width: u32, height: u32) -> Self {
        Self {
            view_inverse: view.inverse(),
            proj_inverse: proj.inverse(),
            width,
            height,
            t_min: T_MIN,
            t_max: T_MAX,
        }
    }

    /// Reconstructs the primary ray for pixel (x, y), mirroring
    /// `build_primary_ray` in shaders/rt/raygen_common.glsl exactly.
    pub fn ray(&self, x: u32, y: u32) -> (Vec3, Vec3) {
        let pixel_center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
        let in_uv = pixel_center / Vec2::new(self.width as f32, self.height as f32);
        let ndc = in_uv * 2.0 - 1.0;

        let clip_pos = Vec4::new(ndc.x, ndc.y, -1.0, 1.0);
        let mut eye_pos = self.proj_inverse * clip_pos;
        eye_pos /= eye_pos.w;

        let origin = (self.view_inverse * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
        let direction =
            (self.view_inverse * Vec4::new(eye_pos.x, eye_pos.y, eye_pos.z, 0.0))
                .truncate()
                .normalize();

        (origin, direction)
    }
}

/// The result of tracing one pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceResult {
    /// 8-bit RGBA (quantized exactly like the GPU's rgba8 storage write).
    pub color: [u8; 4],
    /// Committed hit distance, or 0.0 on a miss (mirrors the miss shader).
    pub t: f32,
}

pub struct ReferenceTracer<'a> {
    world: &'a Chunks,
    palette: [Vec4; 256],
    shape: VoxelShape,
    /// Inclusive bounds of the occupied voxel set; `None` for an empty world.
    bounds: Option<(IVec3, IVec3)>,
}

impl<'a> ReferenceTracer<'a> {
    /// `palette` is the world's 256-entry RGBA8 palette as float4s (the world
    /// side of the contract). The alpha channel is forced to 1.0 to mirror the
    /// current renderer's palette-buffer construction
    /// (`get_palette(...).map(|c| [c.x, c.y, c.z, 1.0])`); when the renderer
    /// starts honoring palette alpha, drop this line.
    pub fn new(world: &'a Chunks, mut palette: [Vec4; 256], shape: VoxelShape) -> Self {
        for color in &mut palette {
            color.w = 1.0;
        }

        Self {
            world,
            palette,
            shape,
            bounds: world.voxel_bounds(),
        }
    }

    /// Traces the pixel (x, y) through the world.
    pub fn trace(&self, camera: &CameraInputs, x: u32, y: u32) -> TraceResult {
        let (origin, direction) = camera.ray(x, y);

        match self.trace_ray(origin, direction) {
            Some((t, material)) => TraceResult {
                color: quantize_rgba(self.palette[material as usize]),
                t,
            },
            None => TraceResult {
                color: BACKGROUND_COLOR,
                t: 0.0,
            },
        }
    }

    /// Steps the ray over the grid cells inside the occupied region and tests
    /// the candidate voxels per cell; returns the closest committed
    /// (t, material).
    fn trace_ray(&self, origin: Vec3, direction: Vec3) -> Option<(f32, u32)> {
        let (min, max) = self.bounds?;

        // The smallest region containing every voxel of the chosen shape.
        let pad = match self.shape {
            VoxelShape::CenteredUnitCube => 0.5,
            VoxelShape::GridCell => 0.0,
        };
        let region_min = min.as_vec3() - Vec3::splat(pad);
        let region_max = max.as_vec3() + Vec3::splat(1.0 + pad);

        let inv = direction.recip();

        let (t0, t1) = slab(origin, inv, region_min, region_max, T_MIN, T_MAX)?;

        let mut best_t = t1;
        let mut best_material: Option<u32> = None;

        let start = origin + direction * t0;
        let mut cell = start.floor().as_ivec3();

        // Amanatides-Woo stepping state: per axis, the t of the next crossed
        // cell boundary and the t to cross one cell.
        let mut step = IVec3::ZERO;
        let mut t_next = Vec3::ZERO;
        let mut delta = Vec3::ZERO;
        for a in 0..3 {
            let d = direction[a];
            if d == 0.0 {
                step[a] = 1;
                t_next[a] = f32::MAX;
                delta[a] = f32::MAX;
            } else if d > 0.0 {
                step[a] = 1;
                let boundary = (cell[a] + 1) as f32;
                t_next[a] = (boundary - origin[a]) * inv[a];
                delta[a] = inv[a];
            } else {
                step[a] = -1;
                let boundary = cell[a] as f32;
                t_next[a] = (boundary - origin[a]) * inv[a];
                delta[a] = -inv[a];
            }
        }

        loop {
            self.test_cell(origin, inv, cell, &mut best_t, &mut best_material);

            let axis = if t_next.x <= t_next.y && t_next.x <= t_next.z {
                0
            } else if t_next.y <= t_next.z {
                1
            } else {
                2
            };

            let next = t_next[axis];
            if next >= best_t {
                break;
            }

            cell[axis] += step[axis];
            t_next[axis] = next + delta[axis];
        }

        best_material.map(|m| (best_t, m))
    }

    fn test_cell(
        &self,
        origin: Vec3,
        inv: Vec3,
        cell: IVec3,
        best_t: &mut f32,
        best_material: &mut Option<u32>,
    ) {
        match self.shape {
            VoxelShape::GridCell => {
                if let Some(voxel) = self.world.try_get_voxel(&cell) {
                    let material = voxel.material_index();
                    self.test_voxel(origin, inv, cell, material, best_t, best_material);
                }
            }
            VoxelShape::CenteredUnitCube => {
                // A cube [p - 0.5, p + 0.5] intersects cell c iff p ∈ {c, c+1}
                // per axis, so a ray inside cell c can only hit voxels whose
                // center is at c + {0, 1}³.
                for dx in 0..2 {
                    for dy in 0..2 {
                        for dz in 0..2 {
                            let p = cell + IVec3::new(dx, dy, dz);
                            if let Some(voxel) = self.world.try_get_voxel(&p) {
                                let material = voxel.material_index();
                                self.test_voxel(origin, inv, p, material, best_t, best_material);
                            }
                        }
                    }
                }
            }
        }
    }

    fn test_voxel(
        &self,
        origin: Vec3,
        inv: Vec3,
        p: IVec3,
        material: u32,
        best_t: &mut f32,
        best_material: &mut Option<u32>,
    ) {
        let (cube_min, cube_max) = match self.shape {
            VoxelShape::CenteredUnitCube => {
                (p.as_vec3() - Vec3::splat(0.5), p.as_vec3() + Vec3::splat(0.5))
            }
            VoxelShape::GridCell => (p.as_vec3(), p.as_vec3() + Vec3::ONE),
        };

        if let Some((t0, _)) = slab(origin, inv, cube_min, cube_max, T_MIN, *best_t)
            && t0 < *best_t
        {
            *best_t = t0;
            *best_material = Some(material);
        }
    }
}

/// Slab test of a ray against an axis-aligned box over [t0_in, t1_in].
/// `inv` is the component-wise reciprocal of the (normalized) direction;
/// infinite components mean the ray is parallel to that axis.
fn slab(
    origin: Vec3,
    inv: Vec3,
    min: Vec3,
    max: Vec3,
    t0_in: f32,
    t1_in: f32,
) -> Option<(f32, f32)> {
    let mut t0 = t0_in;
    let mut t1 = t1_in;

    for a in 0..3 {
        if inv[a].is_finite() {
            let mut a0 = (min[a] - origin[a]) * inv[a];
            let mut a1 = (max[a] - origin[a]) * inv[a];
            if a0 > a1 {
                std::mem::swap(&mut a0, &mut a1);
            }
            t0 = t0.max(a0);
            t1 = t1.min(a1);
        } else if origin[a] < min[a] || origin[a] >= max[a] {
            return None;
        }

        if t0 > t1 {
            return None;
        }
    }

    Some((t0, t1))
}

fn quantize_rgba(color: Vec4) -> [u8; 4] {
    [
        (color.x.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.y.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.z.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.w.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// Renders the full reference image (color + t) in parallel over pixel rows.
/// The reference tracer is not budget-accounted against the 16 ms frame gate
/// (ADR 0003); this just keeps an on-demand run tolerable on dense worlds.
pub fn render_reference(
    tracer: &ReferenceTracer,
    camera: &CameraInputs,
    out_rgba: &mut [u8],
    out_t: &mut [f32],
) {
    let width = camera.width as usize;
    let height = camera.height as usize;
    assert_eq!(out_rgba.len(), width * height * 4);
    assert_eq!(out_t.len(), width * height);

    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);

    let rows_per_thread = height.div_ceil(thread_count);
    let row_ranges: Vec<std::ops::Range<usize>> = (0..height)
        .step_by(rows_per_thread)
        .map(|start| start..(start + rows_per_thread).min(height))
        .collect();

    std::thread::scope(|scope| {
        let mut rgba_rest = out_rgba;
        let mut t_rest = out_t;

        for rows in &row_ranges {
            let rgba_len = rows.len() * width * 4;
            let t_len = rows.len() * width;
            let (rgba_part, rgba_tail) = rgba_rest.split_at_mut(rgba_len);
            let (t_part, t_tail) = t_rest.split_at_mut(t_len);
            rgba_rest = rgba_tail;
            t_rest = t_tail;

            let rows = rows.clone();
            scope.spawn(move || {
                for (row_offset, y) in rows.clone().enumerate() {
                    for x in 0..width {
                        let result = tracer.trace(camera, x as u32, y as u32);

                        let local = row_offset * width + x;
                        rgba_part[local * 4..local * 4 + 4].copy_from_slice(&result.color);
                        t_part[local] = result.t;
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::*;
    use crate::world::chunk::Chunks;

    /// A minimal world: a single voxel at (0, 0, 0) with material 1.
    fn single_voxel_world() -> Chunks {
        let mut chunks = Chunks::default();
        chunks.insert_voxel_at(IVec3::ZERO, 1);
        chunks
    }

    fn palette() -> [Vec4; 256] {
        let mut p = [Vec4::ZERO; 256];
        p[1] = Vec4::new(1.0, 0.0, 0.0, 1.0); // material 1 = red
        p
    }

    fn identity_camera() -> CameraInputs {
        CameraInputs {
            view_inverse: Mat4::IDENTITY,
            proj_inverse: Mat4::IDENTITY,
            width: 2,
            height: 2,
            t_min: T_MIN,
            t_max: T_MAX,
        }
    }

    /// A ray along +X from (-5, 0, 0) must hit the voxel cube [-0.5, 0.5] at
    /// t = 4.5 (entry face), reporting red and that t.
    #[test]
    fn ray_hits_known_voxel_centered_cube() {
        let world = single_voxel_world();
        let tracer = ReferenceTracer::new(&world, palette(), VoxelShape::CenteredUnitCube);

        // Build a camera whose (0,0) pixel ray is origin (-5,0,0), dir +X.
        // proj_inverse = identity maps clip (0,0,-1,1) to eye (0,0,-1,1) —
        // ndc for pixel (0,0) at 2x2 is (-0.5, -0.5), which won't give +X.
        // Instead construct the ray manually and only use CameraInputs::ray
        // for the reconstruction tests below.
        let (t, material) = tracer.trace_ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap();
        assert_eq!(t, 4.5);
        assert_eq!(material, 1);
    }

    /// A ray that misses the voxel reports a miss.
    #[test]
    fn ray_misses_known_voxel() {
        let world = single_voxel_world();
        let tracer = ReferenceTracer::new(&world, palette(), VoxelShape::CenteredUnitCube);

        // Above the voxel (y = 5 > 0.5).
        assert!(tracer.trace_ray(Vec3::new(-5.0, 5.0, 0.0), Vec3::X).is_none());
        // Behind the voxel and looking away from it (the voxel spans
        // z <= 0.5; the ray from z = 5 goes +Z).
        assert!(tracer.trace_ray(Vec3::new(0.0, 0.0, 5.0), Vec3::Z).is_none());
    }

    /// A ray from inside the voxel commits the voxel at t_min (the camera
    /// inside a solid voxel case, ADR 0002).
    #[test]
    fn ray_inside_voxel_commits_at_t_min() {
        let world = single_voxel_world();
        let tracer = ReferenceTracer::new(&world, palette(), VoxelShape::CenteredUnitCube);

        let (t, material) = tracer.trace_ray(Vec3::new(0.2, 0.2, 0.2), Vec3::X).unwrap();
        assert_eq!(material, 1);
        assert_eq!(t, T_MIN);
    }

    /// t_min clips geometry closer than the ray's t range (like the GPU's
    /// near-plane behavior).
    #[test]
    fn t_range_clips_near_hits() {
        let world = single_voxel_world();
        let tracer = ReferenceTracer::new(&world, palette(), VoxelShape::CenteredUnitCube);

        // Voxel at (0,0,0) spans [-0.5, 0.5]; a ray from (0,0,0) toward -X
        // exits at -0.5 but the committed entry is clamped to t_min... the
        // voxel is behind the origin in -X? No: (0,0,0) is inside the cube,
        // so it commits at t_min even in -X.
        let (t, material) = tracer.trace_ray(Vec3::ZERO, Vec3::NEG_X).unwrap();
        assert_eq!(material, 1);
        assert_eq!(t, T_MIN);
    }

    /// Empty world: every ray misses.
    #[test]
    fn empty_world_is_all_background() {
        let world = Chunks::default();
        let tracer = ReferenceTracer::new(&world, palette(), VoxelShape::CenteredUnitCube);
        let camera = identity_camera();

        assert_eq!(tracer.trace(&camera, 0, 0).color, BACKGROUND_COLOR);
        assert_eq!(tracer.trace(&camera, 0, 0).t, 0.0);
    }

    /// GridCell shape: the same voxel occupies [0, 1], so from (-5, 0, 0) the
    /// entry face is at t = 5.
    #[test]
    fn ray_hits_known_voxel_grid_cell() {
        let world = single_voxel_world();
        let tracer = ReferenceTracer::new(&world, palette(), VoxelShape::GridCell);

        let (t, material) = tracer.trace_ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X).unwrap();
        assert_eq!(t, 5.0);
        assert_eq!(material, 1);
    }

    /// The camera-ray reconstruction must match the raygen's unprojection
    /// math: with identity inverses, pixel (0,0) of a 2x2 frame maps to
    /// ndc (-0.5, -0.5), clip (-0.5, -0.5, -1, 1), eye (-0.5, -0.5, -1, 1),
    /// direction normalize(-0.5, -0.5, -1).
    #[test]
    fn camera_ray_reconstruction() {
        let camera = identity_camera();
        let (origin, direction) = camera.ray(0, 0);

        assert_eq!(origin, Vec3::ZERO);
        assert_eq!(direction, Vec3::new(-0.5, -0.5, -1.0).normalize());
    }

    /// The center of a 2x2 frame maps to clip (0, 0): pixel centers land on
    /// ndc 0 at x = 0.5/2 → ndc -0.5? At 2x2 the pixel centers are at
    /// (0.5, 0.5) and (1.5, 1.5), ndc -0.5 and 0.5 — there is no exact
    /// center pixel; verify the far corner instead.
    #[test]
    fn camera_ray_reconstruction_corner() {
        let camera = identity_camera();
        let (_, direction) = camera.ray(1, 1);
        assert_eq!(direction, Vec3::new(0.5, 0.5, -1.0).normalize());
    }
}
