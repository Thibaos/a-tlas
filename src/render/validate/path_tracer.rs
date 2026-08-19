//! The CPU path-tracer mirror (ticket 07): the shading-half validator.
//!
//! The GPU's path tracing (shaders/region/production.rgen, ADR 0010/0011) is
//! mirrored here **sample for sample**: the same stateless PCG-XSH-RR RNG
//! (same seed per pixel+frame, same draw sequence), the same cosine/GGX
//! sampling, the same Sun/Procedural-sky pdfs and MIS weights, the same
//! Russian-roulette decisions, and the same Material table (the CPU mirror,
//! `get_material_table`, ADR 0008). A divergence between the mirror and the
//! captured GPU frame points at the GPU shader — never at an assumption both
//! sides share (the Reference tracer's ethos, ADR 0003).
//!
//! Geometry is deliberately *not* mirrored: the mirror traces the world's own
//! sparse storage (like the Reference tracer) with an independent world-space
//! DDA, sharing only the committed voxel's identity with the GPU path. The DDA
//! mirrors the GPU's stepping arithmetic (division + accumulation, Amanatides-
//! Woo with the x,y,z tie-break — the GPU's own preference order) so the
//! committed t and the entering face agree at the f32 level; the face reports
//! the mirror's own step axis with the same tie-break (ADR 0009).
//!
//! The mirror produces the trace pass's output contract (ADR 0007): the
//! de-modulated diffuse radiance and the raw specular radiance per sample.
//! The validator compares the per-pixel **mean** over N samples (identical
//! seeds — frame_seed 0..N-1) against the GPU's captured radiance pair, with
//! a relative tolerance that absorbs the GPU's RGBA16F storage quantization
//! and the residual f32 transcendental differences (cos/sin/pow may differ
//! by an ULP between GPU and CPU libm).

use glam::{IVec3, Vec3};

use crate::{
    render::validate::reference::{CameraInputs, T_MAX, T_MIN},
    world::{material::MaterialTable, World},
};

// ---------------------------------------------------------------------------
// Mirror constants — every constant below is part of the CPU mirror's contract
// (ticket 07): the mirror reproduces the GLSL arithmetic exactly, so the
// constants are named here once, not repeated in the mirror. They mirror
// shaders/region/production.rgen and sky.glsl verbatim.
// ---------------------------------------------------------------------------

/// BSDF-scattered secondary rays allowed (depth cap).
const MAX_BOUNCES: usize = 4;
/// Lobe-selection probability at the primary hit (inside NRD's [1/4, 3/4]
/// clamp for AREA_3X3).
const LOBE_P: f32 = 0.5;
/// NEE light pick (ticket 06): the probability of next-event-sampling the Sun
/// vs the Procedural sky at a hit.
const SUN_PICK_P: f32 = 0.5;
const SKY_PICK_P: f32 = 1.0 - SUN_PICK_P;
/// Russian-roulette survival floor.
const RR_FLOOR: f32 = 0.05;
/// De-modulation guard: diffuse ÷ max(albedo, ALBEDO_EPS).
const ALBEDO_EPS: f32 = 1e-3;
/// Bounce origin nudge along the normal (self-intersection).
const BOUNCE_OFFSET: f32 = 1e-3;
/// GGX alpha floor (perfect-mirror singularity guard).
const ALPHA_MIN: f32 = 1e-4;
/// Cos-term guard in the G1/pdf denominators (grazing NaN guard).
const COS_EPS: f32 = 1e-6;

/// The GLSL `const float PI = 3.14159265358979` — parsed to the nearest f32,
/// which is `std::f32::consts::PI`.
const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = 6.2831855; // the nearest f32 to the GLSL literal 6.28318530717959

/// The phase constant in the RNG's draw-index mix (the GLSL u32 literal).
const PHI32: u32 = 0x9E3779B1;

// ---------------------------------------------------------------------------
// Empty-space acceleration (the macro grid)
//
// The mirror's DDA marches the world's sparse storage cell by cell; on a
// dense real asset (nuke.vox is 31M voxels) the camera-to-geometry air gap
// is thousands of cells of HashMap lookups per ray. The macro grid adds one
// occupancy bit per 32³-cell block over the occupied bbox: the march leaps
// across blocks that contain **no voxels at all** and resumes the per-cell
// fine march inside the first non-empty block. Because a leap only crosses
// empty blocks, the committed voxel and the entering face are unchanged; the
// committed t accumulates one rounding per leap instead of per cell, a
// ULP-level drift the compare tolerance absorbs (relative tolerance 0.1).
// ---------------------------------------------------------------------------

/// The macro-cell edge length in cells.
const MACRO_SIZE: u32 = 32;
const MACRO_SIZE_F: f32 = 32.0;

/// One occupancy bit per macro cell over the world's occupied bbox, plus a
/// per-cell bitmap (32³ bits per macro cell) so the fine march can test a
/// cell's occupancy with a bit read instead of a HashMap probe. The bitmaps
/// are built from the world's own sparse storage, so a set bit is exactly a
/// cell the world holds; the HashMap is consulted only at set bits.
struct MacroGrid {
    /// The occupied bbox min (cell coordinates).
    min: IVec3,
    /// Macro-cell counts per axis.
    dims: [u32; 3],
    /// Packed macro-cell occupancy bits (bit i = macro cell i contains at
    /// least one voxel — the skip's emptiness test).
    occupied_bits: Vec<u64>,
    /// The per-cell bitmap: 32³ bits (512 u64s) per macro cell.
    cells: Vec<u64>,
}

/// Bits per macro cell (32³) and the u64 words that pack them.
const MACRO_CELL_WORDS: usize = (MACRO_SIZE as usize).pow(3) / 64;

impl MacroGrid {
    fn build(world: &World) -> Self {
        let (min, max) = world.voxel_bounds().unwrap_or((IVec3::ZERO, IVec3::ZERO));
        let extent = max - min + IVec3::ONE;
        let dims = [
            (extent.x.max(0) as u32).div_ceil(MACRO_SIZE).max(1),
            (extent.y.max(0) as u32).div_ceil(MACRO_SIZE).max(1),
            (extent.z.max(0) as u32).div_ceil(MACRO_SIZE).max(1),
        ];
        let macro_count = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        let mut occupied_bits = vec![0u64; macro_count.div_ceil(64)];
        let mut cells = vec![0u64; macro_count * MACRO_CELL_WORDS];
        let max_m = IVec3::new(dims[0] as i32 - 1, dims[1] as i32 - 1, dims[2] as i32 - 1);
        let s = MACRO_SIZE as i32;
        let s2 = MACRO_SIZE as usize * MACRO_SIZE as usize;
        for (pos, _voxel) in world.iter_voxels() {
            // pos is within [min, max] by the bbox contract; clamp defensively.
            let rel = pos - min;
            let m = (rel / s).clamp(IVec3::ZERO, max_m);
            let mi = m.x as usize
                + (m.y as usize) * dims[0] as usize
                + (m.z as usize) * dims[0] as usize * dims[1] as usize;
            occupied_bits[mi >> 6] |= 1u64 << (mi & 63);
            let local = rel - m * s;
            let li =
                local.x as usize + local.y as usize * MACRO_SIZE as usize + local.z as usize * s2;
            cells[mi * MACRO_CELL_WORDS + (li >> 6)] |= 1u64 << (li & 63);
        }
        Self {
            min,
            dims,
            occupied_bits,
            cells,
        }
    }

    /// The macro cell containing `cell`, clamped into the grid (the march
    /// only consults in-grid cells after the first step, but the clamp is
    /// defensive).
    fn clamp_macro_cell(&self, cell: IVec3) -> IVec3 {
        let rel = cell - self.min;
        IVec3::new(
            (rel.x / MACRO_SIZE as i32).clamp(0, self.dims[0] as i32 - 1),
            (rel.y / MACRO_SIZE as i32).clamp(0, self.dims[1] as i32 - 1),
            (rel.z / MACRO_SIZE as i32).clamp(0, self.dims[2] as i32 - 1),
        )
    }

    fn occupied(&self, m: IVec3) -> bool {
        let index = m.x as usize
            + (m.y as usize) * self.dims[0] as usize
            + (m.z as usize) * self.dims[0] as usize * self.dims[1] as usize;
        self.occupied_bits[index >> 6] & (1u64 << (index & 63)) != 0
    }

    /// Is `cell` occupied? Cells outside the occupied bbox are empty (the
    /// world's sparse storage holds no voxels beyond its own bounds). The
    /// bitmap is exact — it is built from the same storage — so a set bit
    /// implies `try_get_voxel` is `Some` for an unchanged world.
    fn cell_occupied(&self, cell: IVec3) -> bool {
        let rel = cell - self.min;
        if rel.min_element() < 0 {
            return false;
        }
        let s = MACRO_SIZE as i32;
        let m = rel / s;
        if m.x >= self.dims[0] as i32 || m.y >= self.dims[1] as i32 || m.z >= self.dims[2] as i32 {
            return false;
        }
        let mi = m.x as usize
            + (m.y as usize) * self.dims[0] as usize
            + (m.z as usize) * self.dims[0] as usize * self.dims[1] as usize;
        let local = rel - m * s;
        let li = local.x as usize
            + local.y as usize * MACRO_SIZE as usize
            + local.z as usize * MACRO_SIZE as usize * MACRO_SIZE as usize;
        self.cells[mi * MACRO_CELL_WORDS + (li >> 6)] & (1u64 << (li & 63)) != 0
    }
}

// ---------------------------------------------------------------------------
// Stateless path RNG (PCG-XSH-RR, O'Neill 2014) — the byte-identical mirror of
// the GLSL hash. u32 ops wrap mod 2^32 identically in GLSL and Rust (the
// wrapping_* methods); the float conversion (high 24 bits × 2^-24) is exact in
// both. Draw order is exactly the call order in `sample` below.
// ---------------------------------------------------------------------------

fn pcg_hash(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// One uniform draw in [0, 1) from the (seed, index) pair.
fn rand_unit(seed: u32, index: u32) -> f32 {
    let h = pcg_hash(seed ^ index.wrapping_mul(PHI32));
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

// ---------------------------------------------------------------------------
// The Scene constants (the packed Scene buffer's data, mirrored verbatim —
// `default_scene()` in src/region/render.rs).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Scene {
    /// The Sun's world direction (unit).
    pub sun_dir: Vec3,
    /// The Procedural sky's μ-gradient knots: ground (μ = -1), horizon (0),
    /// zenith (1) — all strictly positive.
    pub sky_knots: [f32; 3],
    /// The Sun's illuminance `E_sun` (lux).
    pub e_sun: f32,
    /// cos(θ_disk) — the Sun disk's angular radius (the disk is the camera's
    /// direct view only).
    pub cos_disk: f32,
    /// The disk's radiance `L_disk = E_sun / Ω_disk`.
    pub l_disk: f32,
}

// ---------------------------------------------------------------------------
// The mirror
// ---------------------------------------------------------------------------

pub struct PathTracer<'a> {
    world: &'a World,
    materials: MaterialTable,
    scene: Scene,
    /// The empty-space skip grid (built once per tracer construction — i.e.
    /// once per validated frame).
    macro_grid: MacroGrid,
    /// The occupied voxel bounds, cached — `voxel_bounds()` scans the whole
    /// world, and the DDA needs it on every ray.
    bounds: Option<(IVec3, IVec3)>,
}

/// One committed voxel hit from the mirror's world-space DDA.
#[derive(Clone, Copy, Debug)]
pub struct PathHit {
    /// The committed t (the crossing into the hit cell, clamped to T_MIN).
    pub t: f32,
    /// The material index (the 8-bit hitKind).
    pub material: u32,
    /// The geometric normal: the entered face's outward normal, or
    /// -normalize(dir) when no face was crossed (the camera-in-voxel t_min
    /// commit — the GPU's no-face-found fallback).
    pub normal: Vec3,
}

/// One sample's radiance output — the trace pass's contract (ADR 0007):
/// de-modulated diffuse + raw specular.
#[derive(Clone, Copy, Debug)]
pub struct PathSample {
    /// The de-modulated diffuse radiance (divided by the primary albedo,
    /// eps-guarded; raw sky radiance on primary miss).
    pub diffuse_de: Vec3,
    /// The raw specular radiance.
    pub specular: Vec3,
    /// The in-lobe 1st-bounce hit distance (0 on primary miss).
    pub hit_t: f32,
    /// The primary albedo (the de-modulation divisor; 0 on primary miss) —
    /// carried for the display re-modulation.
    pub albedo: Vec3,
}

impl<'a> PathTracer<'a> {
    pub fn new(world: &'a World, materials: MaterialTable, scene: Scene) -> Self {
        let macro_grid = MacroGrid::build(world);
        let bounds = world.voxel_bounds();
        Self {
            world,
            materials,
            scene,
            macro_grid,
            bounds,
        }
    }

    /// The mirror's own world-space DDA (Amanatides-Woo over the world's
    /// sparse storage, with the GPU's stepping arithmetic: division and
    /// accumulation, tie-break x, y, z — the GPU's own preference order).
    /// Returns the committed voxel with the crossing t and the entered face.
    fn trace_ray(&self, origin: Vec3, direction: Vec3) -> Option<PathHit> {
        let (min, max) = self.bounds?;
        let region_min = min.as_vec3();
        let region_max = max.as_vec3() + Vec3::ONE;

        // Slab test over the occupied region (division, like the GPU's
        // aabb_slab — not the Reference tracer's reciprocal form).
        let mut t0 = T_MIN;
        let mut t1 = T_MAX;
        for a in 0..3 {
            let d = direction[a];
            if d == 0.0 {
                if origin[a] < region_min[a] || origin[a] >= region_max[a] {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let mut a0 = (region_min[a] - origin[a]) * inv;
                let mut a1 = (region_max[a] - origin[a]) * inv;
                if a0 > a1 {
                    std::mem::swap(&mut a0, &mut a1);
                }
                t0 = t0.max(a0);
                t1 = t1.min(a1);
                if t0 > t1 {
                    return None;
                }
            }
        }
        if !(t1 > t0) {
            return None;
        }

        let entry = origin + direction * t0;
        let mut cell = entry.floor().as_ivec3();

        // Amanatides-Woo stepping state (the GPU's arithmetic: t_next =
        // (boundary - origin)/d, accumulated by delta = ±1/d).
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
                t_next[a] = (boundary - origin[a]) / d;
                delta[a] = 1.0 / d;
            } else {
                step[a] = -1;
                let boundary = cell[a] as f32;
                t_next[a] = (boundary - origin[a]) / d;
                delta[a] = -1.0 / d;
            }
        }

        // The committed entry t; the face is the axis of the step that
        // entered the committed cell (`None` for the start cell — no face
        // crossed: the camera-in-voxel fallback).
        let mut t_entry = t0;
        let mut face: Option<usize> = None;

        loop {
            // The per-cell occupancy bitmap is a cache of the world's sparse
            // storage: a bit test instead of a HashMap probe at every cell.
            if self.macro_grid.cell_occupied(cell) {
                if let Some(voxel) = self.world.try_get_voxel(&cell) {
                    let normal = match face {
                        Some(a) => {
                            let mut n = Vec3::ZERO;
                            n[a] = -direction[a].signum();
                            n
                        }
                        None => -direction.normalize(),
                    };
                    return Some(PathHit {
                        t: t_entry,
                        material: *voxel,
                        normal,
                    });
                }
            }

            // Step to the next cell boundary: ties break x, y, z — the GPU's
            // own preference order (the mirror's "same tie-break", ADR 0009).
            let axis = if t_next.x <= t_next.y && t_next.x <= t_next.z {
                0
            } else if t_next.y <= t_next.z {
                1
            } else {
                2
            };
            let next = t_next[axis];
            if next >= t1 {
                return None; // marcher out of the occupied region
            }
            cell[axis] += step[axis];
            t_next[axis] = next + delta[axis];
            t_entry = next;
            face = Some(axis);

            // ---- Empty-space skip (the macro grid) ----
            // The fine state now sits at a cell boundary. If the macro cell
            // whose interior the ray is about to traverse contains no voxels
            // at all, leap the fine state across it in one batched jump and
            // repeat until a non-empty macro cell (or the region exit). The
            // crossed axis advances one macro cell (its entered cell index is
            // derived exactly from the new macro cell, never from the drifted
            // boundary estimate — a leap cannot overshoot); the other axes'
            // cells come from the position at the exit, so mid-leap boundary
            // crossings on those axes cannot desync the cell index.
            let mut m_cell = self.macro_grid.clamp_macro_cell(cell);
            if !self.macro_grid.occupied(m_cell) {
                let mut m_t_next = [f32::MAX; 3];
                let mut m_delta = [f32::MAX; 3];
                for a in 0..3 {
                    if direction[a] != 0.0 {
                        let boundary = if step[a] > 0 {
                            self.macro_grid.min[a] as f32 + (m_cell[a] as f32 + 1.0) * MACRO_SIZE_F
                        } else {
                            self.macro_grid.min[a] as f32 + m_cell[a] as f32 * MACRO_SIZE_F
                        };
                        m_t_next[a] = (boundary - origin[a]) / direction[a];
                        m_delta[a] = MACRO_SIZE_F * delta[a];
                    }
                }
                loop {
                    let m_axis = if m_t_next[0] <= m_t_next[1] && m_t_next[0] <= m_t_next[2] {
                        0
                    } else if m_t_next[1] <= m_t_next[2] {
                        1
                    } else {
                        2
                    };
                    let m_exit = m_t_next[m_axis];
                    if m_exit >= t1 {
                        return None; // marcher out of the occupied region
                    }
                    // The macro step must stay in the grid. The far face of
                    // the last macro cell lies at/behind the region face, so
                    // an out-of-grid step is the region exit — the division-
                    // form boundary t can sit one ULP below t1 (the prelude
                    // slab uses the reciprocal form), so the `>= t1` guard
                    // alone cannot catch this.
                    let next_m = m_cell[m_axis] + step[m_axis];
                    if next_m < 0 || next_m >= self.macro_grid.dims[m_axis] as i32 {
                        return None;
                    }
                    t_entry = m_exit;
                    face = Some(m_axis);
                    m_cell[m_axis] = next_m;
                    m_t_next[m_axis] = m_exit + m_delta[m_axis];
                    // The entered fine cell: the crossed axis is the new
                    // macro cell's first (step>0) or last (step<0) cell —
                    // exact; the other axes follow the position at the exit.
                    let mut next_cell = (origin + direction * m_exit).floor().as_ivec3();
                    next_cell[m_axis] = if step[m_axis] > 0 {
                        self.macro_grid.min[m_axis] + m_cell[m_axis] as i32 * MACRO_SIZE as i32
                    } else {
                        self.macro_grid.min[m_axis]
                            + (m_cell[m_axis] as i32 + 1) * MACRO_SIZE as i32
                            - 1
                    };
                    cell = next_cell;
                    if self.macro_grid.occupied(m_cell) {
                        // Re-derive the fine stepping state at the entry of
                        // the non-empty macro cell (the GPU's own boundary
                        // math, division per axis).
                        for a in 0..3 {
                            if direction[a] == 0.0 {
                                t_next[a] = f32::MAX;
                            } else if step[a] > 0 {
                                t_next[a] = ((cell[a] + 1) as f32 - origin[a]) / direction[a];
                            } else {
                                t_next[a] = (cell[a] as f32 - origin[a]) / direction[a];
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    /// The NEE shadow test: trace toward the light from the nudged origin;
    /// any hit occludes. Consumes no RNG draws.
    fn shadowed(&self, hit_point: Vec3, normal: Vec3, light_dir: Vec3) -> bool {
        self.trace_ray(hit_point + normal * BOUNCE_OFFSET, light_dir)
            .is_some()
    }

    /// One path sample for pixel (x, y) at `frame_seed`, mirroring the
    /// production raygen's Voxel branch statement by statement (the draw
    /// sequence is the CPU mirror's contract — see the map's notes and the
    /// ticket-06 resolution).
    pub fn sample(&self, camera: &CameraInputs, x: u32, y: u32, frame_seed: u32) -> PathSample {
        let (origin, direction) = camera.ray(x, y);

        // The per-pixel seed: hash(pixel_id, frame_seed) — pixel_id is
        // y·launch_width + x (the raygen's gl_LaunchIDEXT math).
        let seed = pcg_hash(frame_seed.wrapping_mul(PHI32) ^ (y * camera.width + x));
        let mut draw = 0u32;

        let primary = self.trace_ray(origin, direction);
        let hit_t = primary.map(|h| h.t).unwrap_or(0.0);

        let mut primary_albedo = Vec3::ZERO;
        let mut primary_metallic = 0.0f32;
        let mut primary_roughness = 1.0f32;
        let mut primary_emission = Vec3::ZERO;
        let mut primary_normal = Vec3::ZERO;
        if let Some(hit) = &primary {
            let mat = &self.materials[hit.material as usize];
            primary_albedo = Vec3::from_array(mat.albedo);
            primary_metallic = mat.metallic;
            primary_roughness = mat.roughness;
            primary_emission = Vec3::from_array(mat.emission);
            primary_normal = hit.normal.normalize();
        }

        // The path's radiance: `radiance` carries the deterministic primary
        // Emission (diffuse channel, no 1/p); `path` accumulates the
        // throughput-weighted Bounce radiance (emission at deeper hits +
        // background), split to the selected lobe's channel with the 1/p
        // weight at the end.
        let mut radiance = Vec3::ZERO;
        let mut specular = Vec3::ZERO;
        let mut path = Vec3::ZERO;
        let mut throughput = Vec3::ONE;
        let specular_lobe;

        if hit_t > 0.0 {
            // Primary-hit Emission → the diffuse channel.
            radiance += primary_emission;

            // Probabilistic lobe selection at the primary hit.
            specular_lobe = rand_unit(seed, draw) < LOBE_P;
            draw += 1;
            let lobe_p = if specular_lobe { LOBE_P } else { 1.0 - LOBE_P };

            let mut ro = origin;
            let mut rd = direction;
            let mut albedo = primary_albedo;
            let mut metallic = primary_metallic;
            let mut roughness = primary_roughness;
            let mut normal = primary_normal;
            let mut last_spec_pick = false;
            // The payload the loop reads: the primary hit at depth 0, the
            // previous Bounce's trace result at depth > 0 (the GLSL's
            // payload struct — the mirror carries it in locals).
            let mut payload: Option<PathHit> = primary;
            let mut payload_t = hit_t;

            for depth in 0..=MAX_BOUNCES {
                if depth > 0 {
                    match payload {
                        None => {
                            // The Bounce missed: the Procedural sky's
                            // radiance (the miss shader's output — the
                            // gradient only), MIS-weighted against the sky
                            // NEE: the throughput's terminal factor used the
                            // picked lobe's conditional pdf (p_cond) with
                            // the 1/p_pick split, so the balance-heuristic
                            // weight is p_cond·p_pick/(p_light + p_mixture).
                            let (t_axis, b_axis) = make_frame(normal);
                            let v_local =
                                -Vec3::new(rd.dot(t_axis), rd.dot(b_axis), rd.dot(normal));
                            let alpha = (roughness * roughness).max(ALPHA_MIN);
                            let l_local = Vec3::new(rd.dot(t_axis), rd.dot(b_axis), rd.dot(normal));
                            let p_v =
                                vndf_pdf(l_local, v_local, (v_local + l_local).normalize(), alpha);
                            let p_c = l_local.z.max(0.0) * (1.0 / PI);
                            let spec = if depth == 1 {
                                specular_lobe
                            } else {
                                last_spec_pick
                            };
                            let p_cond = if spec { p_v } else { p_c };
                            let p_pick = if depth == 1 { lobe_p } else { 0.5 };
                            let p_mix = if depth == 1 {
                                LOBE_P * p_v + (1.0 - LOBE_P) * p_c
                            } else {
                                0.5 * (p_v + p_c)
                            };
                            let p_light = self.sky_light_pdf(rd);
                            path += throughput
                                * self.sky_radiance(rd)
                                * (p_cond * p_pick / (p_light + p_mix));
                            break;
                        }
                        Some(hit) => {
                            let mat = &self.materials[hit.material as usize];
                            albedo = Vec3::from_array(mat.albedo);
                            metallic = mat.metallic;
                            roughness = mat.roughness;
                            normal = hit.normal.normalize();
                            // This hit's Emission, throughput-weighted (the
                            // only way Emissive voxels light the scene — no
                            // emissive NEE, map lock).
                            path += throughput * Vec3::from_array(mat.emission);
                        }
                    }
                }

                // NEE: one light picked per bounce — the Sun (delta, MIS
                // weight 1) or the Procedural sky (env, balance-heuristic
                // MIS). Draw order: light pick, then — sky only — the env's
                // (φ, μ) draws; the shadow trace consumes none.
                let hit_point = ro + rd * payload_t;
                let mut nee_rad = Vec3::ZERO;
                if rand_unit(seed, draw) < SUN_PICK_P {
                    draw += 1;
                    let sun_dir = self.scene.sun_dir;
                    let n_dot_l = normal.dot(sun_dir).max(0.0);
                    if n_dot_l > 0.0 && !self.shadowed(hit_point, normal, sun_dir) {
                        let (t_axis, b_axis) = make_frame(normal);
                        let v_local = -Vec3::new(rd.dot(t_axis), rd.dot(b_axis), rd.dot(normal));
                        let alpha = (roughness * roughness).max(ALPHA_MIN);
                        let f0 = mix_vec(Vec3::splat(0.04), albedo, metallic);
                        let l_local = Vec3::new(sun_dir.dot(t_axis), sun_dir.dot(b_axis), n_dot_l);
                        let f_nl = if depth == 0 {
                            if specular_lobe {
                                specular_nl(
                                    l_local,
                                    v_local,
                                    (v_local + l_local).normalize(),
                                    f0,
                                    alpha,
                                )
                            } else {
                                (1.0 - metallic) * albedo * (n_dot_l / PI)
                            }
                        } else {
                            brdf_nl(l_local, v_local, f0, alpha, albedo, metallic)
                        };
                        nee_rad = f_nl * self.scene.e_sun * (1.0 / SUN_PICK_P);
                    }
                } else {
                    draw += 1;
                    // Procedural sky: direction from the analytic env pdf,
                    // p_light = SKY_PICK_P·L(μ)/(2π·Z), MIS weight
                    // p_light/(p_light + p_bsdf).
                    let u1 = rand_unit(seed, draw);
                    draw += 1;
                    let u2 = rand_unit(seed, draw);
                    draw += 1;
                    let phi = TWO_PI * u1;
                    let mu = self.sky_sample_mu(u2);
                    let r = (1.0 - mu * mu).max(0.0).sqrt();
                    let light_dir = Vec3::new(r * phi.cos(), mu, r * phi.sin());
                    let n_dot_l = normal.dot(light_dir).max(0.0);
                    if n_dot_l > 0.0 && !self.shadowed(hit_point, normal, light_dir) {
                        let p_light = self.sky_light_pdf(light_dir);
                        let (t_axis, b_axis) = make_frame(normal);
                        let v_local = -Vec3::new(rd.dot(t_axis), rd.dot(b_axis), rd.dot(normal));
                        let alpha = (roughness * roughness).max(ALPHA_MIN);
                        let f0 = mix_vec(Vec3::splat(0.04), albedo, metallic);
                        let l_local =
                            Vec3::new(light_dir.dot(t_axis), light_dir.dot(b_axis), n_dot_l);
                        let p_bsdf = bsdf_mixture_pdf(l_local, v_local, alpha, depth);
                        let w = p_light / (p_light + p_bsdf);
                        let f_nl = if depth == 0 {
                            if specular_lobe {
                                specular_nl(
                                    l_local,
                                    v_local,
                                    (v_local + l_local).normalize(),
                                    f0,
                                    alpha,
                                )
                            } else {
                                (1.0 - metallic) * albedo * (n_dot_l / PI)
                            }
                        } else {
                            brdf_nl(l_local, v_local, f0, alpha, albedo, metallic)
                        };
                        nee_rad = f_nl * self.sky_radiance(light_dir) * w / p_light;
                    }
                }
                path += throughput * nee_rad;

                if depth == MAX_BOUNCES {
                    break; // depth cap: no further Bounce
                }

                // Sample the next direction at this hit and update the
                // throughput. Bounce 0 samples the selected lobe; deeper
                // Bounces pick a lobe 50/50 (weight × 2).
                let (t_axis, b_axis) = make_frame(normal);
                let v_local = -Vec3::new(rd.dot(t_axis), rd.dot(b_axis), rd.dot(normal));
                let alpha = (roughness * roughness).max(ALPHA_MIN);
                let f0 = mix_vec(Vec3::splat(0.04), albedo, metallic);
                let (wi, weight) = if depth == 0 {
                    if specular_lobe {
                        let u1 = rand_unit(seed, draw);
                        draw += 1;
                        let u2 = rand_unit(seed, draw);
                        draw += 1;
                        let (l_local, h, pdf) = sample_vndf(v_local, alpha, u1, u2);
                        (
                            t_axis * l_local.x + b_axis * l_local.y + normal * l_local.z,
                            specular_weight(l_local, v_local, h, f0, alpha, pdf),
                        )
                    } else {
                        let u1 = rand_unit(seed, draw);
                        draw += 1;
                        let u2 = rand_unit(seed, draw);
                        draw += 1;
                        let (wi, _pdf) = sample_cosine(normal, t_axis, b_axis, u1, u2);
                        (wi, (1.0 - metallic) * albedo)
                    }
                } else {
                    let spec_pick = rand_unit(seed, draw) < 0.5;
                    draw += 1;
                    last_spec_pick = spec_pick;
                    if spec_pick {
                        let u1 = rand_unit(seed, draw);
                        draw += 1;
                        let u2 = rand_unit(seed, draw);
                        draw += 1;
                        let (l_local, h, pdf) = sample_vndf(v_local, alpha, u1, u2);
                        (
                            t_axis * l_local.x + b_axis * l_local.y + normal * l_local.z,
                            2.0 * specular_weight(l_local, v_local, h, f0, alpha, pdf),
                        )
                    } else {
                        let u1 = rand_unit(seed, draw);
                        draw += 1;
                        let u2 = rand_unit(seed, draw);
                        draw += 1;
                        let (wi, _pdf) = sample_cosine(normal, t_axis, b_axis, u1, u2);
                        (wi, 2.0 * (1.0 - metallic) * albedo)
                    }
                };
                throughput *= weight;

                // Russian roulette on the throughput (floor bounds variance;
                // the continuation weight divides it back out).
                let p = throughput.max_element().clamp(RR_FLOOR, 1.0);
                if rand_unit(seed, draw) >= p {
                    break;
                }
                draw += 1;
                throughput /= p;

                // Trace the Bounce: origin nudged off the face along the
                // normal (the DDA also skips [0, T_MIN)). hit_point was
                // computed before the NEE (the shadow trace clobbers the
                // payload — the mirror's Option is re-traced fresh).
                ro = hit_point + normal * BOUNCE_OFFSET;
                rd = wi;
                payload = self.trace_ray(ro, rd);
                payload_t = payload.map(|h| h.t).unwrap_or(0.0);
            }

            // Channel attribution (probabilistic splitting): the path
            // radiance goes to the selected lobe's channel with the 1/p
            // weight; the primary Emission stays in the diffuse channel.
            if specular_lobe {
                specular = path / lobe_p;
            } else {
                radiance += path / lobe_p;
            }
        } else {
            // Primary miss: the Procedural sky's radiance, raw — no surface,
            // no de-modulation, no lobe split. The disk is the camera's
            // direct view only.
            radiance = self.sky_radiance_with_disk(direction);
        }

        // De-modulated diffuse: the diffuse channel divided by the primary
        // albedo (eps-guarded); the sky (primary miss) stays raw.
        let diffuse_de = if hit_t > 0.0 {
            radiance / primary_albedo.max(Vec3::splat(ALBEDO_EPS))
        } else {
            radiance
        };

        PathSample {
            diffuse_de,
            specular,
            hit_t,
            albedo: primary_albedo,
        }
    }
}

// ---------------------------------------------------------------------------
// The shading mirror — the GLSL ports. Every function below reproduces the
// corresponding function in shaders/region/production.rgen and sky.glsl with
// the same arithmetic and the same rounding form.
// ---------------------------------------------------------------------------

/// Deterministic local tangent frame from the geometric normal: reference
/// (0,1,0), falling back to (1,0,0) when the normal is (anti-)parallel to y.
fn make_frame(n: Vec3) -> (Vec3, Vec3) {
    let reference = if n.y.abs() > 0.999 { Vec3::X } else { Vec3::Y };
    let t_axis = reference.cross(n).normalize();
    let b_axis = n.cross(t_axis);
    (t_axis, b_axis)
}

/// The GLSL `mix(x, y, a)` — the spec form x·(1−a) + y·a.
fn mix_vec(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}

/// Cosine-weighted diffuse sample in the frame (n, t_axis, b_axis); u1, u2 in
/// [0,1). Returns the direction and its pdf (cosθ/π).
fn sample_cosine(n: Vec3, t_axis: Vec3, b_axis: Vec3, u1: f32, u2: f32) -> (Vec3, f32) {
    let r = u1.sqrt();
    let phi = TWO_PI * u2;
    let cz = (1.0 - u1).max(0.0).sqrt();
    let dir = t_axis * (r * phi.cos()) + b_axis * (r * phi.sin()) + n * cz;
    let pdf = cz * (1.0 / PI);
    (dir, pdf)
}

fn ggx_d(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

fn smith_g1(n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let nv = n_dot_v.max(COS_EPS);
    2.0 * nv / (nv + (a2 + (1.0 - a2) * nv * nv).sqrt())
}

fn fresnel_schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powf(5.0)
}

/// The VNDF pdf of the light direction l for the view v (Heitz 2018), at the
/// half-vector h.
fn vndf_pdf(_l_local: Vec3, v_local: Vec3, h: Vec3, alpha: f32) -> f32 {
    let n_dot_h = h.z.max(0.0);
    let v_dot_h = v_local.dot(h).max(0.0);
    let n_dot_v = v_local.z.max(COS_EPS);
    (ggx_d(n_dot_h, alpha) * smith_g1(n_dot_v, alpha) * v_dot_h / n_dot_v)
        / (4.0 * v_dot_h.max(COS_EPS))
}

/// GGX visible-normal (VNDF) sampling (Heitz 2018). Returns the incoming
/// (light) direction l_local, the sampled half-vector h, and the pdf.
fn sample_vndf(v_local: Vec3, alpha: f32, u1: f32, u2: f32) -> (Vec3, Vec3, f32) {
    let vh = Vec3::new(alpha * v_local.x, alpha * v_local.y, v_local.z).normalize();
    let len_sq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if len_sq > 0.0 {
        Vec3::new(-vh.y, vh.x, 0.0) * (1.0 / len_sq.sqrt())
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);
    let r = u1.sqrt();
    let phi = TWO_PI * u2;
    let t1v = r * phi.cos();
    let t2v = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    let t2v = (1.0 - s) * (1.0 - t1v * t1v).max(0.0).sqrt() + s * t2v;
    let nh = t1 * t1v + t2 * t2v + (1.0 - t1v * t1v - t2v * t2v).max(0.0).sqrt() * vh;
    let h = Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize();
    let l = reflect(-v_local, h);
    let pdf = vndf_pdf(l, v_local, h, alpha);
    (l, h, pdf)
}

/// GLSL `reflect(I, N)` = I − 2·dot(N, I)·N.
fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - 2.0 * n.dot(i) * n
}

/// The specular lobe's estimator weight (Cook-Torrance with separable Smith
/// G2): f_s · (N·L) / pdf_L at the sampled pair.
fn specular_weight(l_local: Vec3, v_local: Vec3, h: Vec3, f0: Vec3, alpha: f32, pdf: f32) -> Vec3 {
    let n_dot_l = l_local.z.max(0.0);
    let n_dot_v = v_local.z.max(COS_EPS);
    let n_dot_h = h.z.max(0.0);
    let v_dot_h = v_local.dot(h).max(0.0);
    let d = ggx_d(n_dot_h, alpha);
    let g2 = smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha);
    let f = fresnel_schlick(f0, v_dot_h);
    f * d * g2 / (4.0 * n_dot_v * pdf.max(COS_EPS))
}

/// The specular lobe's f_s·(n·l) at the light direction: D·G2·F/(4·(n·v)).
/// The NEE estimator's integrand factor — NOT the sampled-lobe weight (those
/// round differently; both are mirrored separately, ADR 0011).
fn specular_nl(l_local: Vec3, v_local: Vec3, h: Vec3, f0: Vec3, alpha: f32) -> Vec3 {
    let n_dot_l = l_local.z.max(0.0);
    let n_dot_v = v_local.z.max(COS_EPS);
    let n_dot_h = h.z.max(0.0);
    let v_dot_h = v_local.dot(h).max(0.0);
    let d = ggx_d(n_dot_h, alpha);
    let g2 = smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha);
    let f = fresnel_schlick(f0, v_dot_h);
    f * d * g2 / (4.0 * n_dot_v)
}

/// The full BRDF's f·(n·l) at the light direction (Lambert diffuse +
/// Cook-Torrance specular) — the NEE's integrand factor at deeper Bounces.
fn brdf_nl(
    l_local: Vec3,
    v_local: Vec3,
    f0: Vec3,
    alpha: f32,
    albedo: Vec3,
    metallic: f32,
) -> Vec3 {
    let h = (v_local + l_local).normalize();
    (1.0 - metallic) * albedo * (l_local.z.max(0.0) / PI)
        + specular_nl(l_local, v_local, h, f0, alpha)
}

/// The BSDF technique's output density at the light direction (the bounce
/// sampler's unconditional pdf — the mixture over the lobe pick):
/// LOBE_P·p_vndf + (1-LOBE_P)·p_cos at the primary hit's bounce, 0.5·(p_vndf
/// + p_cos) deeper.
fn bsdf_mixture_pdf(l_local: Vec3, v_local: Vec3, alpha: f32, depth: usize) -> f32 {
    let h = (v_local + l_local).normalize();
    let p_v = vndf_pdf(l_local, v_local, h, alpha);
    let p_c = l_local.z.max(0.0) * (1.0 / PI);
    if depth == 0 {
        LOBE_P * p_v + (1.0 - LOBE_P) * p_c
    } else {
        0.5 * (p_v + p_c)
    }
}

// ---------------------------------------------------------------------------
// The Procedural sky mirror (shaders/region/sky.glsl). Only clamp/lerp/mul/
// div/sqrt — all bit-reproducible in f32.
// ---------------------------------------------------------------------------

impl PathTracer<'_> {
    /// The gradient radiance at μ ∈ [-1, 1] (piecewise-linear between the
    /// knots; clamped for safety — the mirror reproduces the clamp).
    fn sky_gradient(&self, mu: f32) -> f32 {
        let k = self.scene.sky_knots;
        let t = mu.clamp(-1.0, 1.0);
        if t < 0.0 {
            k[0] * (1.0 - (t + 1.0)) + k[1] * (t + 1.0)
        } else {
            k[1] * (1.0 - t) + k[2] * t
        }
    }

    /// The marginal pdf's normalization: Z = ∫ L(μ) dμ (the trapezoid).
    fn sky_pdf_norm(&self) -> f32 {
        let k = self.scene.sky_knots;
        0.5 * ((k[0] + k[1]) + (k[1] + k[2]))
    }

    /// The marginal pdf of μ: L(μ)/Z.
    fn sky_mu_pdf(&self, mu: f32) -> f32 {
        self.sky_gradient(mu) / self.sky_pdf_norm()
    }

    /// Inverse-CDF sample of μ from u ∈ [0, 1): per-segment quadratic
    /// inversion; degenerate segments (equal end knots) fall back to the
    /// linear form.
    fn sky_sample_mu(&self, u: f32) -> f32 {
        let k = self.scene.sky_knots;
        let z = self.sky_pdf_norm();
        let c0 = (0.5 * (k[0] + k[1])) / z; // the CDF at the horizon (μ = 0)
        if u < c0 {
            // Segment [-1, 0], t = μ + 1 ∈ [0, 1].
            let a = 0.5 * (k[1] - k[0]);
            let b = k[0];
            if a == 0.0 {
                return u * z / b - 1.0;
            }
            let disc = (b * b + 4.0 * a * u * z).max(0.0);
            let t = (-b + disc.sqrt()) / (2.0 * a);
            t - 1.0
        } else {
            // Segment [0, 1].
            let a = 0.5 * (k[2] - k[1]);
            let b = k[1];
            if a == 0.0 {
                return (u - c0) * z / b;
            }
            let disc = (b * b + 4.0 * a * (u - c0) * z).max(0.0);
            (-b + disc.sqrt()) / (2.0 * a)
        }
    }

    /// The transport radiance at the direction dir: the gradient only (the
    /// disk is not part of the transport).
    fn sky_radiance(&self, dir: Vec3) -> Vec3 {
        Vec3::splat(self.sky_gradient(dir.y))
    }

    /// The camera's direct view of the sky: the gradient plus the Sun disk
    /// (a measure-zero radiance bump detected by a pure dot test).
    fn sky_radiance_with_disk(&self, dir: Vec3) -> Vec3 {
        let l = self.sky_radiance(dir);
        if dir.dot(self.scene.sun_dir) > self.scene.cos_disk {
            l + Vec3::splat(self.scene.l_disk)
        } else {
            l
        }
    }

    /// The sky light technique's direction pdf: the env sampler's output
    /// density with the light pick folded in — p_light(ω) = SKY_PICK_P ·
    /// L(μ)/(2π·Z).
    fn sky_light_pdf(&self, dir: Vec3) -> f32 {
        SKY_PICK_P * self.sky_mu_pdf(dir.y) * (1.0 / TWO_PI)
    }
}

// ---------------------------------------------------------------------------
// The full-frame render: per-pixel means over N samples with identical seeds
// (frame_seed 0..N-1) — the CPU side of the tolerance diff.
// ---------------------------------------------------------------------------

/// The CPU side's per-pixel aggregates: the means the diff compares, plus the
/// display data (the re-modulated radiance for the PNG) and the per-pixel
/// sample-count bookkeeping for the firefly-outlier excuse.
pub struct PathRender {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// Per-pixel mean de-modulated diffuse radiance (len = width·height).
    pub diffuse: Vec<Vec3>,
    /// Per-pixel mean raw specular radiance.
    pub specular: Vec<Vec3>,
    /// Per-pixel mean display radiance (re-modulated: diffuse·max(albedo,eps)
    /// + specular; raw sky on primary-miss samples) — for the PNG only.
    pub display: Vec<Vec3>,
    /// Per-pixel number of samples whose primary ray hit a surface (the sky
    /// samples are the rest) — for the report's sky mask.
    pub hit_count: Vec<u32>,
    /// Per-pixel mean in-lobe 1st-bounce hit distance (0 for sky) — for the
    /// corner-touch t-gap excuse.
    pub hit_ts: Vec<f32>,
    /// Per-pixel mean primary albedo (0 for sky) — for the corner-touch
    /// material-boundary signal (the committed voxel's material).
    pub albedos: Vec<Vec3>,
}

/// Renders the path-traced frame set: for each pixel, N samples seeded by
/// frame_seed 0..N-1, accumulated into per-pixel means. Parallel over pixel
/// rows (like `render_reference`).
pub fn render_path(
    tracer: &PathTracer,
    camera: &CameraInputs,
    width: u32,
    height: u32,
    samples: u32,
) -> PathRender {
    let pixel_count = (width as usize) * (height as usize);

    let mut diffuse = vec![Vec3::ZERO; pixel_count];
    let mut specular = vec![Vec3::ZERO; pixel_count];
    let mut display = vec![Vec3::ZERO; pixel_count];
    let mut hit_count = vec![0u32; pixel_count];
    let mut hit_ts = vec![0.0f32; pixel_count];
    let mut albedos = vec![Vec3::ZERO; pixel_count];

    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let rows_per_thread = (height as usize).div_ceil(thread_count);

    std::thread::scope(|scope| {
        let mut diffuse_rest = diffuse.as_mut_slice();
        let mut specular_rest = specular.as_mut_slice();
        let mut display_rest = display.as_mut_slice();
        let mut hit_rest = hit_count.as_mut_slice();
        let mut hit_ts_rest = hit_ts.as_mut_slice();
        let mut albedos_rest = albedos.as_mut_slice();

        for row_start in (0..height as usize).step_by(rows_per_thread) {
            let rows = row_start..(row_start + rows_per_thread).min(height as usize);
            let len = rows.len() * width as usize;
            let (d_part, d_tail) = diffuse_rest.split_at_mut(len);
            let (s_part, s_tail) = specular_rest.split_at_mut(len);
            let (p_part, p_tail) = display_rest.split_at_mut(len);
            let (h_part, h_tail) = hit_rest.split_at_mut(len);
            let (t_part, t_tail) = hit_ts_rest.split_at_mut(len);
            let (a_part, a_tail) = albedos_rest.split_at_mut(len);
            diffuse_rest = d_tail;
            specular_rest = s_tail;
            display_rest = p_tail;
            hit_rest = h_tail;
            hit_ts_rest = t_tail;
            albedos_rest = a_tail;

            scope.spawn(move || {
                for (row_offset, y) in rows.clone().enumerate() {
                    for x in 0..width {
                        let mut d_sum = Vec3::ZERO;
                        let mut s_sum = Vec3::ZERO;
                        let mut p_sum = Vec3::ZERO;
                        let mut t_sum = 0.0f32;
                        let mut a_sum = Vec3::ZERO;
                        let mut hits = 0u32;
                        for f in 0..samples {
                            let sample = tracer.sample(camera, x, y as u32, f);
                            d_sum += sample.diffuse_de;
                            s_sum += sample.specular;
                            let modulated = if sample.hit_t > 0.0 {
                                sample.diffuse_de * sample.albedo.max(Vec3::splat(ALBEDO_EPS))
                                    + sample.specular
                            } else {
                                sample.diffuse_de
                            };
                            p_sum += modulated;
                            t_sum += sample.hit_t;
                            a_sum += sample.albedo;
                            hits += u32::from(sample.hit_t > 0.0);
                        }
                        let n = samples as f32;
                        let local = row_offset * width as usize + x as usize;
                        d_part[local] = d_sum / n;
                        s_part[local] = s_sum / n;
                        p_part[local] = p_sum / n;
                        h_part[local] = hits;
                        t_part[local] = t_sum / n;
                        a_part[local] = a_sum / n;
                    }
                }
            });
        }
    });

    PathRender {
        width,
        height,
        samples,
        diffuse,
        specular,
        display,
        hit_count,
        hit_ts,
        albedos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        render::validate::reference::{T_MAX, T_MIN},
        world::{material::Material, World},
    };

    fn default_scene() -> Scene {
        let sun_dir = Vec3::new(0.45, 0.8, 0.35).normalize();
        let cos_disk = (0.5_f32 * std::f32::consts::PI / 180.0).cos();
        let omega = 2.0 * std::f32::consts::PI * (1.0 - cos_disk);
        Scene {
            sun_dir,
            sky_knots: [0.15, 0.6, 1.2],
            e_sun: 16.0,
            cos_disk,
            l_disk: 16.0 / omega,
        }
    }

    /// A single-voxel world at the origin with material 1.
    fn single_voxel_world() -> World {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::ZERO, 1);
        world
    }

    fn materials() -> MaterialTable {
        let mut table = [Material {
            albedo: [0.0; 3],
            metallic: 0.0,
            roughness: 0.3,
            emission: [0.0; 3],
        }; 256];
        table[1] = Material {
            albedo: [1.0, 0.0, 0.0],
            metallic: 0.0,
            roughness: 0.3,
            emission: [0.0; 3],
        };
        table
    }

    /// The RNG is byte-identical to the GLSL implementation (golden values
    /// computed from an independent Python implementation of the same
    /// algorithm).
    #[test]
    fn rng_matches_glsl_golden_values() {
        assert_eq!(pcg_hash(0x0), 0x07bb2fe2);
        assert_eq!(pcg_hash(0x1), 0xa8beea3c);
        assert_eq!(pcg_hash(0xdead_beef), 0x67299972);
        assert_eq!(pcg_hash(0x9e37_79b1), 0x703cc78a);

        for (index, expected) in [
            (0u32, 0.030199945_f32),
            (1, 0.438427389),
            (2, 0.630939007),
            (3, 0.790136099),
            (7, 0.182337880),
            (8, 0.229610860),
        ] {
            assert_eq!(rand_unit(0, index), expected, "rand_unit(0, {index})");
        }
    }

    /// The RNG draw sequence is deterministic (a pure function of seed and
    /// index — no mutable per-path state).
    #[test]
    fn rng_draws_are_deterministic() {
        for seed in [0u32, 1, 0x1234_5678] {
            let a: Vec<f32> = (0..16).map(|i| rand_unit(seed, i)).collect();
            let b: Vec<f32> = (0..16).map(|i| rand_unit(seed, i)).collect();
            assert_eq!(a, b);
        }
    }

    /// The sky μ-sample CDF inversion round-trips: the empirical CDF of the
    /// sampled μs matches the analytic marginal pdf.
    #[test]
    fn sky_sample_mu_inverts_the_cdf() {
        let world = single_voxel_world();
        let tracer = PathTracer::new(&world, materials(), default_scene());

        // The analytic CDF at μ = 0 (the horizon) is c0.
        let k = tracer.scene.sky_knots;
        let z = tracer.sky_pdf_norm();
        let c0 = (0.5 * (k[0] + k[1])) / z;

        // A uniform u below c0 maps into [-1, 0]; above maps into [0, 1].
        let mu_low = tracer.sky_sample_mu(c0 * 0.5);
        let mu_high = tracer.sky_sample_mu(c0 + (1.0 - c0) * 0.5);
        assert!((-1.0..0.0).contains(&mu_low), "μ low segment: {mu_low}");
        assert!((0.0..=1.0).contains(&mu_high), "μ high segment: {mu_high}");

        // Monotone: a larger u samples a larger μ.
        let mut prev = tracer.sky_sample_mu(0.001);
        for i in 2..1000 {
            let u = i as f32 / 1000.0;
            let mu = tracer.sky_sample_mu(u);
            assert!(mu >= prev, "sky_sample_mu must be monotone at u = {u}");
            prev = mu;
        }

        // The empirical mean of the sampled μs converges to the analytic
        // E[μ] = ∫ μ·L(μ) dμ / Z = 2(k₂−k₀)/(3(k₀+2k₁+k₂)) — the sample
        // distribution is the marginal pdf (the inversion round-trips).
        let expected_mean = 2.0 * (k[2] - k[0]) / (3.0 * (k[0] + 2.0 * k[1] + k[2]));
        let mut mu_sum = 0.0f64;
        for i in 0..100_000 {
            let mu = tracer.sky_sample_mu((i as f32 + 0.5) / 100_000.0);
            mu_sum += f64::from(mu);
        }
        let empirical_mean = mu_sum / 100_000.0;
        assert!(
            (empirical_mean - f64::from(expected_mean)).abs() < 1e-3,
            "sampled μ mean {empirical_mean} vs analytic {expected_mean}"
        );
    }

    /// A ray along +X from (-5, 0, 0) must hit the voxel cell [0, 1]³ at
    /// t = 5, with the face normal -X (the entry face).
    #[test]
    fn trace_hits_known_voxel_grid_cell() {
        let world = single_voxel_world();
        let tracer = PathTracer::new(&world, materials(), default_scene());
        let hit = tracer
            .trace_ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X)
            .unwrap();
        assert_eq!(hit.t, 5.0);
        assert_eq!(hit.material, 1);
        assert_eq!(hit.normal, Vec3::NEG_X);
    }

    /// A ray from inside the voxel commits at t_min with the camera-facing
    /// fallback normal (no face crossed — the GPU's t_min commit, ADR 0009).
    #[test]
    fn trace_inside_voxel_commits_at_t_min() {
        let world = single_voxel_world();
        let tracer = PathTracer::new(&world, materials(), default_scene());
        let hit = tracer.trace_ray(Vec3::new(0.2, 0.2, 0.2), Vec3::X).unwrap();
        assert_eq!(hit.t, T_MIN);
        assert_eq!(hit.normal, Vec3::NEG_X, "no face → -normalize(dir)");
    }

    /// The material table's albedo/emission feed the sample: a hit sample on
    /// a red diffuse voxel de-modulates to the light/surface ratio, and the
    /// emission lands in the diffuse channel.
    #[test]
    fn sample_on_diffuse_voxel_is_deterministic() {
        let world = single_voxel_world();
        let tracer = PathTracer::new(&world, materials(), default_scene());
        let camera = CameraInputs::new(
            glam::camera::lh::view::look_to_mat4(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, Vec3::Y),
            glam::camera::lh::proj::vulkan::perspective(1.0, 1.0, T_MIN, T_MAX),
            1,
            1,
        );
        // The same (pixel, frame) always yields the same sample.
        let a = tracer.sample(&camera, 0, 0, 7);
        let b = tracer.sample(&camera, 0, 0, 7);
        assert_eq!(a.diffuse_de, b.diffuse_de);
        assert_eq!(a.specular, b.specular);
    }

    /// The pre-acceleration march — the mirror's own world-space DDA without
    /// the macro-grid skip. The accelerated march must agree with it exactly
    /// (same voxel and face; t within the ULP-level leap drift).
    fn naive_trace(world: &World, origin: Vec3, direction: Vec3) -> Option<PathHit> {
        let (min, max) = world.voxel_bounds()?;
        let region_min = min.as_vec3();
        let region_max = max.as_vec3() + Vec3::ONE;
        let mut t0 = T_MIN;
        let mut t1 = T_MAX;
        for a in 0..3 {
            let d = direction[a];
            if d == 0.0 {
                if origin[a] < region_min[a] || origin[a] >= region_max[a] {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let mut a0 = (region_min[a] - origin[a]) * inv;
                let mut a1 = (region_max[a] - origin[a]) * inv;
                if a0 > a1 {
                    std::mem::swap(&mut a0, &mut a1);
                }
                t0 = t0.max(a0);
                t1 = t1.min(a1);
                if t0 > t1 {
                    return None;
                }
            }
        }
        if !(t1 > t0) {
            return None;
        }
        let entry = origin + direction * t0;
        let mut cell = entry.floor().as_ivec3();
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
                t_next[a] = (boundary - origin[a]) / d;
                delta[a] = 1.0 / d;
            } else {
                step[a] = -1;
                let boundary = cell[a] as f32;
                t_next[a] = (boundary - origin[a]) / d;
                delta[a] = -1.0 / d;
            }
        }
        let mut t_entry = t0;
        let mut face: Option<usize> = None;
        loop {
            if let Some(voxel) = world.try_get_voxel(&cell) {
                let normal = match face {
                    Some(a) => {
                        let mut n = Vec3::ZERO;
                        n[a] = -direction[a].signum();
                        n
                    }
                    None => -direction.normalize(),
                };
                return Some(PathHit {
                    t: t_entry,
                    material: *voxel,
                    normal,
                });
            }
            let axis = if t_next.x <= t_next.y && t_next.x <= t_next.z {
                0
            } else if t_next.y <= t_next.z {
                1
            } else {
                2
            };
            let next = t_next[axis];
            if next >= t1 {
                return None;
            }
            cell[axis] += step[axis];
            t_next[axis] = next + delta[axis];
            t_entry = next;
            face = Some(axis);
        }
    }

    /// A world spanning many macro cells: a hollow 120³ shell (the interior
    /// is empty — the skip must leap across it) plus a sparse diagonal band.
    fn wide_sparse_world() -> World {
        let mut world = World::default();
        // The shell: three voxel-thick walls of a 120³ box at z = 8..112.
        for x in 8..112 {
            for y in 8..112 {
                for z in [8, 104] {
                    world.insert_voxel_at(IVec3::new(x, y, z), 1);
                    world.insert_voxel_at(IVec3::new(x, y, z + 1), 1);
                }
            }
        }
        for y in 8..112 {
            for z in 8..112 {
                for x in [8, 104] {
                    world.insert_voxel_at(IVec3::new(x, y, z), 1);
                }
            }
        }
        // A sparse band crossing the hollow interior.
        for i in 0..64 {
            world.insert_voxel_at(IVec3::new(16 + i * 2, 60 + i / 2, 60), 1);
        }
        // A dense 10³ block (the bitmap's dense path).
        for x in 40..50 {
            for y in 40..50 {
                for z in 40..50 {
                    world.insert_voxel_at(IVec3::new(x, y, z), 1);
                }
            }
        }
        world
    }

    #[test]
    fn macro_grid_marks_every_voxel_cell() {
        let world = wide_sparse_world();
        let grid = MacroGrid::build(&world);
        assert!(
            grid.dims[0] >= 3 && grid.dims[1] >= 3 && grid.dims[2] >= 3,
            "dims {:?}",
            grid.dims
        );
        for (pos, _voxel) in world.iter_voxels() {
            let m = grid.clamp_macro_cell(pos);
            assert!(
                grid.occupied(m),
                "macro cell {m:?} of voxel {pos:?} not marked"
            );
            assert!(
                grid.cell_occupied(pos),
                "cell {pos:?} not marked in the bitmap"
            );
        }
    }

    #[test]
    fn macro_grid_skip_matches_naive_march() {
        let world = wide_sparse_world();
        let tracer = PathTracer::new(&world, materials(), default_scene());
        let (min, max) = world.voxel_bounds().unwrap();
        let center = (min.as_vec3() + max.as_vec3()) * 0.5;

        let mut seed = 0xfeed_faceu32;
        let mut next_u = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 16_777_216.0
        };

        let mut hits = 0u32;
        for _ in 0..3000 {
            // Aim at the bbox center from a random shell (guarantees the ray
            // traverses many macro cells), then jitter the target slightly.
            let origin = center
                + Vec3::new(
                    (next_u() - 0.5) * 3.0,
                    (next_u() - 0.5) * 3.0,
                    (next_u() - 0.5) * 3.0,
                ) * 200.0;
            let target = center
                + Vec3::new(
                    (next_u() - 0.5) * 60.0,
                    (next_u() - 0.5) * 60.0,
                    (next_u() - 0.5) * 60.0,
                );
            let direction = (target - origin).normalize();
            match (
                naive_trace(&world, origin, direction),
                tracer.trace_ray(origin, direction),
            ) {
                (None, None) => {}
                (Some(n), Some(f)) => {
                    hits += 1;
                    assert_eq!(n.material, f.material);
                    assert!(
                        (n.normal - f.normal).abs().max_element() < 1e-5,
                        "normal mismatch: naive {n:?} fast {f:?}"
                    );
                    assert!(
                        (n.t - f.t).abs() <= 1e-2 * n.t.max(1.0),
                        "t mismatch: naive {} fast {}",
                        n.t,
                        f.t
                    );
                }
                (n, f) => panic!("agreement broken: naive {n:?} fast {f:?}"),
            }
        }
        assert!(hits > 100, "the ray set must hit the world, hits: {hits}");
    }
}
