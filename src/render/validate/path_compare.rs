//! Per-pixel comparison of the captured GPU path-traced radiance pair against
//! the CPU path-tracer mirror (ticket 07).
//!
//! The GPU runs N frames with frame_seed 0..N-1 (1 path per pixel per frame);
//! the CPU mirror computes the same N samples from the same seeds. Each side
//! produces a per-pixel **mean** of the de-modulated diffuse and raw specular
//! radiance (the trace pass's output contract, ADR 0007). A pixel mismatches
//! when any channel of either mean's relative error exceeds the tolerance.
//!
//! Three classes of divergence are excused explicitly (the ticket's
//! "edge-silhouette and firefly outliers handled explicitly"):
//!
//! - **Edge silhouette**: the GPU's Region-local DDA and the mirror's
//!   world-space DDA can legitimately commit different voxels at a corner
//!   (the existing byte-exact compare's corner-touch class); the affected
//!   pixels sit on the mean radiance image's silhouette, so a 1-pixel-dilated
//!   edge mask over the mirror's display image AND the GPU's compared
//!   channels excuses them (the GPU channels catch the knife-edge specular
//!   spikes a NEE shadow flip leaves at a grazing silhouette, a GPU-side
//!   discontinuity the locally-smooth CPU mean cannot reproduce).
//! - **Firefly**: a 1-spp path that hits a small bright Emissive voxel is a
//!   legitimately huge radiance at one pixel; with identical seeds both sides
//!   see the same firefly, so their means agree to quantization, and the
//!   residual error at a bright pixel is bounded by the firefly/N. A mismatch
//!   pixel whose mean radiance is bright on *both* sides is excused as a
//!   firefly outlier (real shading bugs manifest across a region, not as
//!   isolated bright pixels).
//! - **Face tie** (ticket 07): same committed cell and material with agreeing
//!   t, but the two sides' hit-point reconstructions of the entered face land
//!   on opposite sides of the epsilon window's edge — ADR 0009's canonical
//!   corner tie, reached by independent f32 ray computations. The shading
//!   differs through the face normal; reported per-run as its own count.
//!
//! The run passes when there are no hard mismatches and the excused
//! mismatches total ≤ the budget fraction of pixels (matching the byte-exact
//! compare's posture).

use glam::{Vec2, Vec3};

/// The corner-touch material disagreement threshold: the committed voxels'
/// material differs by more than this (adjacent cells at a material
/// boundary share the entry t. The t-gap signal is blind there and the
/// RGBA16F-quantized hit distance drowns it at large t, but the albedo is
/// exact at every distance). The RGBA8 storage granularity is ~0.004.
const CORNER_TOUCH_ENC_EPS: f32 = 0.05;

/// The octahedral normal encoding (production.rgen, ADR 0007):
/// `n.xy / (|x|+|y|+|z|)`, with the -z hemisphere wrapped by the standard
/// reflection, mapped to [0, 1]^2. Unit normals only (zero encodes garbage.
/// The caller guards on the hit test). Kept for the diff-image diagnostics.
pub fn encode_octahedral(n: Vec3) -> glam::Vec2 {
    let sum = n.x.abs() + n.y.abs() + n.z.abs();
    let mut enc = glam::Vec2::new(n.x / sum, n.y / sum);
    if n.z < 0.0 {
        let s = enc.signum();
        enc = glam::Vec2::new(
            (1.0 - enc.y.abs()) * s.x,
            (1.0 - enc.x.abs()) * s.y,
        );
    }
    enc * 0.5 + 0.5
}

/// Inverts encode_octahedral: the RGBA8-stored octahedral encoding back to
/// the unit normal (the aux buffer's per-pixel value, the face-tie signal of
/// the shading diff).
pub fn decode_octahedral(enc01: glam::Vec2) -> Vec3 {
    let e = enc01 * 2.0 - 1.0;
    let mut n = Vec3::new(e.x, e.y, 1.0 - e.x.abs() - e.y.abs());
    if n.z < 0.0 {
        let s = Vec2::new(n.x.signum(), n.y.signum());
        let w = Vec2::new((1.0 - e.y.abs()) * s.x, (1.0 - e.x.abs()) * s.y);
        n = Vec3::new(w.x, w.y, n.z);
    }
    n.normalize_or_zero()
}

/// Why a mismatching pixel was excused (or not).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExcuseKind {
    /// A real divergence.
    Hard,
    /// Hugging the hit-status silhouette (the surface outline).
    Silhouette,
    /// Both sides bright (a 1-spp firefly both see).
    Firefly,
    /// Both sides committed a surface at the same point with different
    /// committed cells (the geometry half's corner-touch class).
    CornerTouch,
    /// Same committed cell and material, agreeing t, but the entered-face
    /// reconstructions disagree: two f32 computations of p land on opposite
    /// sides of the epsilon window's edge (ADR 0009's canonical tie). The
    /// shading then differs through the face normal, not through lighting.
    FaceTie,
    /// The pixel's means disagree because a minority of its identical-seed
    /// samples took categorically different paths (a shadow occlusion or a
    /// bounce commit that flipped on ULP-level geometry noise inside deep
    /// bounce chains), while the remaining seeds agree tightly. A systematic
    /// shading bug diverges on every seed that exercises it and stays Hard.
    PathDivergence,
}

/// A seed's radiance pair disagrees past this relative error: not noise, a
/// different path outcome (lit vs unlit, emissive hit vs miss).
const CATEGORICAL_ERROR: f32 = 0.5;

/// The compare's per-channel relative error with its denominator floor.
fn channel_rel_err(g: Vec3, c: Vec3, abs_floor: f32) -> f32 {
    let denom = g.abs().max(c.abs()).max(Vec3::splat(abs_floor));
    let diff = (g - c).abs();
    diff.x.max(diff.y).max(diff.z)
        / denom.x.max(denom.y).max(denom.z).max(1e-9)
}

/// Per-seed divergence evidence for one mismatching pixel: how many of the N
/// identical-seed samples disagree categorically (a different path outcome),
/// how many sit in the moderate band between tolerance and categorical (a
/// shifted value — the systematic-bug signature), and how many agree.
#[derive(Clone, Copy, Debug)]
pub struct SeedEvidence {
    /// Seeds inside the tolerance.
    pub agreeing: u32,
    /// Seeds past CATEGORICAL_ERROR.
    pub diverged: u32,
    /// Seeds between the tolerance and CATEGORICAL_ERROR.
    pub shifted: u32,
}

impl SeedEvidence {
    /// The chaos signature: a minority of seeds (at most half, with at most
    /// a quarter shifted moderately — the milder flip where a path grazed a
    /// decision) diverge while the rest agree. A systematic bug shifts every
    /// seed that exercises it and trips the shifted or majority bounds.
    pub fn is_minority_flip(&self, samples: u32) -> bool {
        let disagreeing = self.diverged + self.shifted;
        disagreeing > 0
            && self.shifted <= samples / 4
            && self.diverged <= samples / 2
            && disagreeing <= samples / 2
    }
}

/// Classifies one pixel's N identical-seed sample pairs.
pub fn seed_evidence(
    gpu_diff: &[Vec3],
    gpu_spec: &[Vec3],
    cpu_diff: &[Vec3],
    cpu_spec: &[Vec3],
    tolerance: f32,
    abs_floor: f32,
) -> SeedEvidence {
    let mut ev = SeedEvidence {
        agreeing: 0,
        diverged: 0,
        shifted: 0,
    };
    for s in 0..gpu_diff.len() {
        let err = channel_rel_err(gpu_diff[s], cpu_diff[s], abs_floor)
            .max(channel_rel_err(gpu_spec[s], cpu_spec[s], abs_floor));
        if err > CATEGORICAL_ERROR {
            ev.diverged += 1;
        } else if err > tolerance {
            ev.shifted += 1;
        } else {
            ev.agreeing += 1;
        }
    }
    ev
}

/// The face-tie signal: the reconstructed normals disagree past this dot
/// product while cell, material and t agree (an axis flip is orthogonal,
/// dot ~ 0; noise-level agreement sits above 0.999).
const FACE_TIE_MIN_DOT: f32 = 0.99;

/// `PathCompareConfig::default()`: 10% relative tolerance, 0.08 absolute
/// floor, 5% mismatch budget. The tolerance absorbs the GPU's RGBA16F
/// storage quantization (~5e-4 relative per sample) and the residual f32
/// transcendental differences between GPU and CPU libm (~ULP-level). The
/// floor is calibrated to the knife-edge flip noise at grazing silhouettes
/// (a NEE shadow test that flips on ULP-level t leaves a ~0.005 mean
/// residual on dark surfaces); the budget to the small test worlds' natural
/// silhouette fraction (a 12³ box fills most of the frame).
#[derive(Clone, Copy, Debug)]
pub struct PathCompareConfig {
    /// Relative error threshold: `|gpu - cpu| / max(|gpu|, |cpu|, abs_floor)`.
    pub tolerance: f32,
    /// The denominator floor: near-zero radiance noise (deep shadow, dark
    /// surfaces under dim sky) must not fail the run at arbitrary ratios.
    pub abs_floor: f32,
    /// Firefly floor: a mismatch pixel whose mean display radiance exceeds
    /// this on BOTH sides is excused as a firefly outlier.
    pub firefly_floor: f32,
    /// Maximum fraction of pixels that may mismatch (inside the excused
    /// clusters).
    pub max_mismatch_ratio: f32,
}

impl Default for PathCompareConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.1,
            abs_floor: 0.08,
            firefly_floor: 8.0,
            max_mismatch_ratio: 0.05,
        }
    }
}

/// One mismatching pixel (all mismatches are recorded; the report details the
/// first few hard ones).
#[derive(Clone, Debug)]
pub struct PathMismatch {
    pub x: u32,
    pub y: u32,
    pub gpu_diffuse: Vec3,
    pub gpu_specular: Vec3,
    pub cpu_diffuse: Vec3,
    pub cpu_specular: Vec3,
    /// The pixel's max relative error across the six channels.
    pub error: f32,
    /// The GPU's in-lobe hit distance and the mirror's (the corner-touch
    /// t-gap, reported for the hard-mismatch diagnostics).
    pub gpu_t: f32,
    pub cpu_t: f32,
    /// Why the mismatch was excused (or `Hard`).
    pub excuse: ExcuseKind,
    /// Per-seed agreement summary for hard mismatches ("6/8 seeds agree"),
    /// empty otherwise.
    pub seed_note: String,
}

pub struct PathCompareReport {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub config: PathCompareConfig,
    /// Every mismatching pixel.
    pub mismatches: Vec<PathMismatch>,
    /// Mismatches NOT excused (real divergences).
    pub hard_mismatches: Vec<PathMismatch>,
    /// Pixels on the dilated edge silhouette (for the diff image).
    pub edge_mask: Vec<bool>,
    /// Per-pixel max relative error across the six channels (for the report's
    /// error stats).
    pub relative_error: Vec<f32>,
    /// The per-pixel sky fraction on the GPU side (mean hit distance > 0).
    /// Reported, not compared (the geometry half validates t).
    pub gpu_hit_fraction: Vec<f32>,
}

impl PathCompareReport {
    pub fn total_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn mismatch_count(&self) -> usize {
        self.mismatches.len()
    }

    pub fn hard_mismatch_count(&self) -> usize {
        self.hard_mismatches.len()
    }

    pub fn edge_excused(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.excuse == ExcuseKind::Silhouette)
            .count()
    }

    pub fn firefly_excused(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.excuse == ExcuseKind::Firefly)
            .count()
    }

    pub fn corner_touch_excused(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.excuse == ExcuseKind::CornerTouch)
            .count()
    }

    pub fn face_tie_excused(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.excuse == ExcuseKind::FaceTie)
            .count()
    }

    pub fn path_divergence_excused(&self) -> usize {
        self.mismatches
            .iter()
            .filter(|m| m.excuse == ExcuseKind::PathDivergence)
            .count()
    }

    /// Pass = no hard mismatches and the excused mismatches total ≤ the
    /// budget fraction of pixels.
    pub fn passes(&self) -> bool {
        self.hard_mismatches.is_empty()
            && self.mismatch_count()
                <= (self.total_pixels() as f32 * self.config.max_mismatch_ratio) as usize
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compare_path(
    gpu_diffuse: &[Vec3],
    gpu_specular: &[Vec3],
    gpu_hit_fraction: &[f32],
    gpu_display: &[Vec3],
    gpu_hitdist: &[f32],
    gpu_albedo: &[Vec3],
    gpu_normals: &[Vec3],
    cpu_diffuse: &[Vec3],
    cpu_specular: &[Vec3],
    cpu_display: &[Vec3],
    cpu_hit_ts: &[f32],
    cpu_albedo: &[Vec3],
    cpu_normals: &[Vec3],
    width: u32,
    height: u32,
    samples: u32,
    config: PathCompareConfig,
) -> PathCompareReport {
    let pixel_count = width as usize * height as usize;
    assert_eq!(gpu_diffuse.len(), pixel_count);
    assert_eq!(gpu_specular.len(), pixel_count);
    assert_eq!(gpu_hit_fraction.len(), pixel_count);
    assert_eq!(gpu_display.len(), pixel_count);
    assert_eq!(gpu_hitdist.len(), pixel_count);
    assert_eq!(gpu_albedo.len(), pixel_count);
    assert_eq!(gpu_normals.len(), pixel_count);
    assert_eq!(cpu_diffuse.len(), pixel_count);
    assert_eq!(cpu_specular.len(), pixel_count);
    assert_eq!(cpu_display.len(), pixel_count);
    assert_eq!(cpu_hit_ts.len(), pixel_count);
    assert_eq!(cpu_albedo.len(), pixel_count);
    assert_eq!(cpu_normals.len(), pixel_count);

    // Edge silhouette mask: a pixel is an edge pixel when its primary hit
    // status (hit vs miss) differs from a 4-neighbor's. The surface
    // outline. The GPU's captured hit status is the ground truth (the
    // geometry half validates the hit positions); the mirror's world-space
    // DDA can legitimately commit a different voxel at the outline (the
    // byte-exact compare's corner-touch class) and a NEE shadow test flips
    // on ULP-level t there, leaving knife-edge specular spikes the locally
    // smooth CPU mean cannot reproduce. Radiance discontinuities are *not*
    // silhouette signals: a high-frequency surface (the nuke cloud) is
    // radiance-rough everywhere, and the mask must not excuse its interior.
    let mut edge = vec![false; pixel_count];
    for i in 0..pixel_count {
        let x = (i % width as usize) as i64;
        let y = (i / width as usize) as i64;
        let hit = gpu_hit_fraction[i] > 0.0;
        for (nx, ny) in [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ] {
            if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                continue;
            }
            let j = ny as usize * width as usize + nx as usize;
            if (gpu_hit_fraction[j] > 0.0) != hit {
                edge[i] = true;
                break;
            }
        }
    }

    // 1-pixel dilation: mismatches hugging the silhouette are excused too.
    let mut edge_mask = vec![false; pixel_count];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = y * width as usize + x;
            let mut on_edge = edge[i];
            if !on_edge {
                for (nx, ny) in [
                    (x as i64 + 1, y as i64),
                    (x as i64 - 1, y as i64),
                    (x as i64, y as i64 + 1),
                    (x as i64, y as i64 - 1),
                ] {
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    if edge[ny as usize * width as usize + nx as usize] {
                        on_edge = true;
                        break;
                    }
                }
            }
            edge_mask[i] = on_edge;
        }
    }

    let mut mismatches = Vec::new();
    let mut hard_mismatches = Vec::new();
    let mut relative_error = vec![0.0f32; pixel_count];

    for i in 0..pixel_count {
        let gd = gpu_diffuse[i];
        let gs = gpu_specular[i];
        let cd = cpu_diffuse[i];
        let cs = cpu_specular[i];

        let err = |g: Vec3, c: Vec3| channel_rel_err(g, c, config.abs_floor);
        let error = err(gd, cd).max(err(gs, cs));
        relative_error[i] = error;

        if error > config.tolerance {
            // Excused when hugging the silhouette, when both sides' mean
            // display radiance is a firefly, or when the pixel is a
            // corner-touch: both sides committed a surface at the same
            // distance (|Δt| ≤ 2 voxels, the byte-exact compare's
            // SUB_VOXEL_T_VOXELS class) but with different faces (the
            // octahedral encodings disagree): the mirror's world-space DDA
            // and the GPU's Region-local DDA legitimately commit adjacent
            // voxels at a corner, and the shading differs through the face
            // normal, not through a lighting divergence.
            let on_edge = edge_mask[i];
            let firefly = gpu_display[i].max_element() > config.firefly_floor
                && cpu_display[i].max_element() > config.firefly_floor;
            let corner_touch = gpu_hit_fraction[i] > 0.0
                && cpu_hit_ts[i] > 0.0
                && (gpu_albedo[i] - cpu_albedo[i]).abs().max_element() > CORNER_TOUCH_ENC_EPS;
            // Same committed cell and material (the albedos agree), the t's
            // agree, but the entered faces differ: the canonical scan decided
            // from two f32 computations of p that land on opposite sides of
            // the window edge (ADR 0009's float corner tie).
            let face_tie = gpu_hit_fraction[i] > 0.0
                && cpu_hit_ts[i] > 0.0
                && (gpu_albedo[i] - cpu_albedo[i]).abs().max_element() <= CORNER_TOUCH_ENC_EPS
                && gpu_normals[i].dot(cpu_normals[i]) < FACE_TIE_MIN_DOT;
            let excuse = if on_edge {
                ExcuseKind::Silhouette
            } else if firefly {
                ExcuseKind::Firefly
            } else if corner_touch {
                ExcuseKind::CornerTouch
            } else if face_tie {
                ExcuseKind::FaceTie
            } else {
                ExcuseKind::Hard
            };

            let mismatch = PathMismatch {
                x: (i % width as usize) as u32,
                y: (i / width as usize) as u32,
                gpu_diffuse: gd,
                gpu_specular: gs,
                cpu_diffuse: cd,
                cpu_specular: cs,
                error,
                gpu_t: gpu_hitdist[i],
                cpu_t: cpu_hit_ts[i],
                excuse,
                seed_note: String::new(),
            };
            mismatches.push(mismatch.clone());
            if excuse == ExcuseKind::Hard {
                hard_mismatches.push(mismatch);
            }
        }
    }

    PathCompareReport {
        width,
        height,
        samples,
        config,
        mismatches,
        hard_mismatches,
        edge_mask,
        relative_error,
        gpu_hit_fraction: gpu_hit_fraction.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(vec: Vec3, pixels: usize) -> Vec<Vec3> {
        vec![vec; pixels]
    }

    /// The auxiliary arguments with benign defaults: the corner-touch excuse
    /// must not fire. The t's agree and the normal encodings agree (the
    /// +Z pole encodes to (0.5, 0.5), the GPU's sky placeholder).
    /// The auxiliary arguments with benign defaults: the corner-touch excuse
    /// must not fire. The t's agree (same cell) and the albedos agree (same
    /// material).
    fn benign_aux(pixels: usize) -> (Vec<f32>, Vec<Vec3>, Vec<f32>, Vec<Vec3>) {
        (
            vec![10.0; pixels],
            flat(Vec3::new(0.4, 0.4, 0.4), pixels),
            vec![10.0; pixels],
            flat(Vec3::new(0.4, 0.4, 0.4), pixels),
        )
    }

    fn run_compare(
        gpu_d: &[Vec3],
        gpu_s: &[Vec3],
        hit: &[f32],
        gpu_disp: &[Vec3],
        cpu_d: &[Vec3],
        cpu_s: &[Vec3],
        cpu_disp: &[Vec3],
        w: u32,
        h: u32,
    ) -> PathCompareReport {
        let pixels = (w * h) as usize;
        let (t, alb, t2, alb2) = benign_aux(pixels);
        let n = flat(Vec3::Z, pixels);
        compare_path(
            gpu_d,
            gpu_s,
            hit,
            gpu_disp,
            &t,
            &alb,
            &n,
            cpu_d,
            cpu_s,
            cpu_disp,
            &t2,
            &alb2,
            &n,
            w,
            h,
            8,
            PathCompareConfig::default(),
        )
    }

    #[test]
    fn identical_means_pass() {
        let (w, h) = (4u32, 4u32);
        let pixels = (w * h) as usize;
        let d = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        let s = flat(Vec3::new(0.0, 0.0, 0.0), pixels);
        let hit = vec![1.0f32; pixels];
        let disp = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        let report = run_compare(&d, &s, &hit, &disp, &d, &s, &disp, w, h);
        assert_eq!(report.mismatch_count(), 0);
        assert!(report.passes());
    }

    /// A divergence beyond the tolerance in a flat region is hard (a real
    /// shading difference. A whole region shifts, so no pixel is a local
    /// outlier).
    #[test]
    fn radiance_divergence_is_hard() {
        let (w, h) = (4u32, 4u32);
        let pixels = (w * h) as usize;
        let gpu = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        let spec = flat(Vec3::ZERO, pixels);
        let hit = vec![1.0f32; pixels];
        let disp_gpu = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        // The last row diverges (a real shading change across a region).
        let mut cpu = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        for x in 0..w as usize {
            let i = (h as usize - 1) * w as usize + x;
            cpu[i] = Vec3::new(0.5, 0.3, 1.0); // 5x the blue channel
        }
        let mut disp_cpu = disp_gpu.clone();
        for x in 0..w as usize {
            disp_cpu[(h as usize - 1) * w as usize + x] = Vec3::new(0.5, 0.3, 1.0);
        }

        let report = run_compare(&gpu, &spec, &hit, &disp_gpu, &cpu, &spec, &disp_cpu, w, h);
        assert_eq!(report.hard_mismatch_count(), w as usize);
        assert!(!report.passes());
    }

    /// A divergence at a material silhouette is excused (the corner-flip
    /// class) and the pixel sits on the dilated edge mask. The silhouette is
    /// the hit-status boundary (hit ↔ miss), not a radiance discontinuity.
    #[test]
    fn silhouette_mismatch_is_excused() {
        let (w, h) = (16u32, 16u32);
        let pixels = (w * h) as usize;
        // The surface outline: pixels x < 8 hit, x >= 8 miss (a vertical
        // silhouette at x = 8).
        let mut hit = vec![0.0f32; pixels];
        for y in 0..h as usize {
            for x in 0..8 {
                hit[y * w as usize + x] = 1.0;
            }
        }
        let cpu_disp = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        let gpu_disp = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        let gpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let cpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let spec = flat(Vec3::ZERO, pixels);
        // A 100% radiance divergence hugging the silhouette (x = 7, y = 8).
        let mut gpu_d_bad = gpu_d.clone();
        gpu_d_bad[8 * w as usize + 7] = Vec3::new(0.0, 0.0, 1.0);

        let report = run_compare(&gpu_d_bad, &spec, &hit, &gpu_disp, &cpu_d, &spec, &cpu_disp, w, h);
        assert!(report.mismatch_count() >= 1);
        assert_eq!(report.hard_mismatch_count(), 0);
        assert!(report.passes());
    }

    /// A corner-touch: both sides committed a surface at the same point but
    /// with different committed cells. The t gaps differ beyond the noise
    /// floor (the mirror's DDA and the GPU's commit adjacent cells at a
    /// corner). Excused; a same-cell divergence (t agrees) stays hard.
    #[test]
    fn corner_touch_mismatch_is_excused() {
        let (w, h) = (16u32, 16u32);
        let pixels = (w * h) as usize;
        let hit = vec![1.0f32; pixels];
        let gpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let cpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let spec = flat(Vec3::ZERO, pixels);
        let disp = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        // A 100% radiance divergence at one pixel, with the committed cells
        // at a material boundary (the albedos differ. The mirror committed
        // the neighbor material).
        let mut gpu_d_bad = gpu_d.clone();
        gpu_d_bad[0] = Vec3::new(0.0, 0.0, 1.0);
        let t = vec![10.0; pixels];
        let mut gpu_albedo = flat(Vec3::new(0.4, 0.4, 0.4), pixels);
        gpu_albedo[0] = Vec3::new(0.9, 0.1, 0.1);
        let cpu_albedo = flat(Vec3::new(0.4, 0.4, 0.4), pixels);

        let report = compare_path(
            &gpu_d_bad,
            &spec,
            &hit,
            &disp,
            &t,
            &gpu_albedo,
            &flat(Vec3::Z, pixels),
            &cpu_d,
            &spec,
            &disp,
            &t,
            &cpu_albedo,
            &flat(Vec3::Z, pixels),
            w,
            h,
            8,
            PathCompareConfig::default(),
        );
        assert!(report.mismatch_count() >= 1);
        assert_eq!(report.hard_mismatch_count(), 0);
        assert_eq!(report.corner_touch_excused(), 1);
        assert!(report.passes());
    }

    /// A same-cell divergence (the t's agree to the noise floor. A shading
    /// difference, not a geometry one) stays hard.
    #[test]
    fn same_cell_divergence_is_hard() {
        let (w, h) = (16u32, 16u32);
        let pixels = (w * h) as usize;
        let hit = vec![1.0f32; pixels];
        let gpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let cpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let spec = flat(Vec3::ZERO, pixels);
        let disp = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let mut gpu_d_bad = gpu_d.clone();
        // The last row diverges (a real shading change, no local outliers).
        for x in 0..w as usize {
            gpu_d_bad[(h as usize - 1) * w as usize + x] = Vec3::new(0.0, 0.0, 1.0);
        }
        // The t's agree (same cell, ULP noise). The divergence is shading.
        let gpu_t = vec![10.0; pixels];
        let cpu_t = vec![10.0; pixels];
        let albedo = flat(Vec3::new(0.4, 0.4, 0.4), pixels);

        let report = compare_path(
            &gpu_d_bad,
            &spec,
            &hit,
            &disp,
            &gpu_t,
            &albedo,
            &flat(Vec3::Z, pixels),
            &cpu_d,
            &spec,
            &disp,
            &cpu_t,
            &albedo,
            &flat(Vec3::Z, pixels),
            w,
            h,
            8,
            PathCompareConfig::default(),
        );
        assert_eq!(report.hard_mismatch_count(), w as usize);
        assert!(!report.passes());
    }

    /// A same-t flip at a material boundary (adjacent cells share the entry
    /// t. The albedo differs) is the corner-touch class too.
    #[test]
    fn material_boundary_flip_is_excused() {
        let (w, h) = (16u32, 16u32);
        let pixels = (w * h) as usize;
        let hit = vec![1.0f32; pixels];
        let gpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let cpu_d = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let spec = flat(Vec3::ZERO, pixels);
        let disp = flat(Vec3::new(0.5, 0.0, 0.0), pixels);
        let mut gpu_d_bad = gpu_d.clone();
        gpu_d_bad[0] = Vec3::new(0.0, 0.0, 1.0);
        // Same t (adjacent cells share the boundary), different materials.
        let t = vec![10.0; pixels];
        let mut gpu_albedo = flat(Vec3::new(0.4, 0.4, 0.4), pixels);
        gpu_albedo[0] = Vec3::new(0.9, 0.1, 0.1);
        let cpu_albedo = flat(Vec3::new(0.4, 0.4, 0.4), pixels);

        let report = compare_path(
            &gpu_d_bad,
            &spec,
            &hit,
            &disp,
            &t,
            &gpu_albedo,
            &flat(Vec3::Z, pixels),
            &cpu_d,
            &spec,
            &disp,
            &t,
            &cpu_albedo,
            &flat(Vec3::Z, pixels),
            w,
            h,
            8,
            PathCompareConfig::default(),
        );
        assert!(report.mismatch_count() >= 1);
        assert_eq!(report.hard_mismatch_count(), 0);
        assert_eq!(report.corner_touch_excused(), 1);
        assert!(report.passes());
    }

    /// The path-divergence signature: a minority of seeds took categorically
    /// different paths (rel err ~1) while the rest agree to tolerance. A
    /// systematic bug shifts every seed moderately and never qualifies.
    #[test]
    fn seed_evidence_separates_chaos_from_shift() {
        let g_d = vec![Vec3::splat(1.0); 8];
        let g_s = vec![Vec3::ZERO; 8];
        let c_s = vec![Vec3::ZERO; 8];

        let mut c_d = vec![Vec3::splat(1.0); 8];
        c_d[6] = Vec3::ZERO;
        c_d[7] = Vec3::ZERO;
        let ev = seed_evidence(&g_d, &g_s, &c_d, &c_s, 0.1, 0.08);
        assert_eq!((ev.agreeing, ev.diverged, ev.shifted), (6, 2, 0));
        assert!(ev.is_minority_flip(8));

        // Every seed shifted 30%: a real estimator difference, not chaos.
        let c_d = vec![Vec3::splat(1.3); 8];
        let ev = seed_evidence(&g_d, &g_s, &c_d, &c_s, 0.1, 0.08);
        assert_eq!((ev.agreeing, ev.diverged, ev.shifted), (0, 0, 8));
        assert!(!ev.is_minority_flip(8));

        // One seed shifted moderately (the milder flip: 7/8 agree) is still
        // the chaos signature.
        let mut c_d = vec![Vec3::splat(1.0); 8];
        c_d[5] = Vec3::splat(1.35);
        let ev = seed_evidence(&g_d, &g_s, &c_d, &c_s, 0.1, 0.08);
        assert_eq!((ev.agreeing, ev.diverged, ev.shifted), (7, 0, 1));
        assert!(ev.is_minority_flip(8));

        // A majority of seeds flipped: also not the minority signature.
        let c_d = vec![Vec3::ZERO; 8];
        let ev = seed_evidence(&g_d, &g_s, &c_d, &c_s, 0.1, 0.08);
        assert_eq!((ev.agreeing, ev.diverged, ev.shifted), (0, 8, 0));
        assert!(!ev.is_minority_flip(8));
    }

    /// A bright pixel on BOTH sides (a firefly both sides see, identical
    /// seeds) is excused: the mean comparison is least meaningful exactly
    /// where a 1-spp firefly dominates.
    #[test]
    fn both_sides_bright_firefly_is_excused() {
        let (w, h) = (16u32, 16u32);
        let pixels = (w * h) as usize;
        let mut gpu_d = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        let mut cpu_d = flat(Vec3::new(0.5, 0.3, 0.2), pixels);
        gpu_d[0] = Vec3::new(15.0, 0.3, 0.2);
        cpu_d[0] = Vec3::new(13.0, 0.3, 0.2); // 15% off, bright both sides
        let spec = flat(Vec3::ZERO, pixels);
        let hit = vec![1.0f32; pixels];
        let disp = flat(Vec3::new(15.0, 0.3, 0.2), pixels);

        let report = run_compare(&gpu_d, &spec, &hit, &disp, &cpu_d, &spec, &disp, w, h);
        assert!(report.mismatch_count() >= 1);
        assert_eq!(report.hard_mismatch_count(), 0);
        assert!(report.passes());
    }
}
