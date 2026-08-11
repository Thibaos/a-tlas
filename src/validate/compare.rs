//! Per-pixel comparison of the captured GPU frame against the reference
//! tracer (rendering-core ticket 06 / ADR 0003).
//!
//! A pixel mismatches when the 8-bit colors differ (exact — both sides use
//! the same palette) or when t differs beyond the relative tolerance
//! `1e-3 * max(t, 1)`. Mismatches that sit on the reference image's
//! edge-silhouette (a 1-pixel-dilated color discontinuity) are excused as
//! sub-voxel boundary effects; the run passes when there are no mismatches
//! outside those clusters and the clusters total ≤ 1% of pixels.

/// `CompareConfig::default()` — 1e-3 relative t tolerance, 1% mismatch budget.
#[derive(Clone, Copy, Debug)]
pub struct CompareConfig {
    /// Relative t tolerance: `|t_gpu - t_ref| <= t_tolerance * max(t, 1)`.
    pub t_tolerance: f32,
    /// Maximum fraction of pixels that may mismatch (inside edge clusters).
    pub max_mismatch_ratio: f32,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            t_tolerance: 1e-3,
            max_mismatch_ratio: 0.01,
        }
    }
}

/// One pixel's {color, t} on one side of the comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelSample {
    pub color: [u8; 4],
    pub t: f32,
}

/// A mismatching pixel (all mismatches are recorded; the report details the
/// first few hard ones).
#[derive(Clone, Copy, Debug)]
pub struct Mismatch {
    pub x: u32,
    pub y: u32,
    pub gpu: PixelSample,
    pub reference: PixelSample,
}

pub struct CompareReport {
    pub width: u32,
    pub height: u32,
    pub config: CompareConfig,
    /// Every mismatching pixel.
    pub mismatches: Vec<Mismatch>,
    /// Mismatches NOT on the dilated edge silhouette (real divergences).
    pub hard_mismatches: Vec<Mismatch>,
    /// Pixels on the dilated edge silhouette (for the diff image).
    pub edge_mask: Vec<bool>,
}

impl CompareReport {
    pub fn total_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn mismatch_count(&self) -> usize {
        self.mismatches.len()
    }

    pub fn hard_mismatch_count(&self) -> usize {
        self.hard_mismatches.len()
    }

    /// Pass = zero mismatches outside the ≤1% edge-silhouette clusters.
    pub fn passes(&self) -> bool {
        self.hard_mismatches.is_empty()
            && self.mismatch_count()
                <= (self.total_pixels() as f32 * self.config.max_mismatch_ratio) as usize
    }
}

pub fn compare(
    gpu_rgba: &[u8],
    gpu_t: &[f32],
    reference_rgba: &[u8],
    reference_t: &[f32],
    width: u32,
    height: u32,
    config: CompareConfig,
) -> CompareReport {
    let pixel_count = width as usize * height as usize;
    assert_eq!(gpu_rgba.len(), pixel_count * 4);
    assert_eq!(gpu_t.len(), pixel_count);
    assert_eq!(reference_rgba.len(), pixel_count * 4);
    assert_eq!(reference_t.len(), pixel_count);

    // Edge silhouette mask over the REFERENCE image: a pixel is an edge pixel
    // when any 4-neighbor has a different color.
    let mut edge = vec![false; pixel_count];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = y * width as usize + x;
            let color = &reference_rgba[i * 4..i * 4 + 4];

            let mut is_edge = false;
            for (nx, ny) in [
                (x as i64 + 1, y as i64),
                (x as i64 - 1, y as i64),
                (x as i64, y as i64 + 1),
                (x as i64, y as i64 - 1),
            ] {
                if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                    continue;
                }
                let j = (ny as usize) * width as usize + nx as usize;
                if reference_rgba[j * 4..j * 4 + 4] != *color {
                    is_edge = true;
                    break;
                }
            }
            edge[i] = is_edge;
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
                    if edge[(ny as usize) * width as usize + nx as usize] {
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

    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = y * width as usize + x;

            let gpu = PixelSample {
                color: gpu_rgba[i * 4..i * 4 + 4].try_into().expect("4-byte color"),
                t: gpu_t[i],
            };
            let reference = PixelSample {
                color: reference_rgba[i * 4..i * 4 + 4]
                    .try_into()
                    .expect("4-byte color"),
                t: reference_t[i],
            };

            let color_match = gpu.color == reference.color;
            let t_match = (gpu.t - reference.t).abs()
                <= config.t_tolerance * gpu.t.abs().max(reference.t.abs()).max(1.0);

            if !color_match || !t_match {
                let mismatch = Mismatch {
                    x: x as u32,
                    y: y as u32,
                    gpu,
                    reference,
                };
                mismatches.push(mismatch);

                if !edge_mask[i] {
                    hard_mismatches.push(mismatch);
                }
            }
        }
    }

    CompareReport {
        width,
        height,
        config,
        mismatches,
        hard_mismatches,
        edge_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_color(color: [u8; 4], pixels: usize) -> Vec<u8> {
        vec![color; pixels].into_iter().flatten().collect()
    }

    fn flat_t(t: f32, pixels: usize) -> Vec<f32> {
        vec![t; pixels]
    }

    #[test]
    fn identical_images_pass() {
        let rgba = flat_color([10, 20, 30, 255], 16);
        let t = flat_t(4.5, 16);
        let report = compare(&rgba, &t, &rgba, &t, 4, 4, CompareConfig::default());
        assert_eq!(report.mismatch_count(), 0);
        assert_eq!(report.hard_mismatch_count(), 0);
        assert!(report.passes());
    }

    #[test]
    fn color_divergence_in_flat_region_is_hard() {
        let reference = flat_color([10, 20, 30, 255], 16);
        let mut gpu = flat_color([10, 20, 30, 255], 16);
        // corrupt one interior pixel
        gpu[(1 * 4 + 1) * 4..(1 * 4 + 1) * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);
        let t = flat_t(4.5, 16);

        let report = compare(&gpu, &t, &reference, &t, 4, 4, CompareConfig::default());
        assert_eq!(report.mismatch_count(), 1);
        assert_eq!(report.hard_mismatch_count(), 1);
        assert!(!report.passes());
    }

    #[test]
    fn silhouette_mismatch_is_excused() {
        // 16x16 (256 px; the 1% budget allows 2). Reference: left half red,
        // right half blue → a vertical silhouette at x = 8. Corrupt a pixel
        // adjacent to it; it must be excused (and stay within the budget).
        let (w, h) = (16usize, 16usize);
        let mut reference = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let color = if x < 8 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                reference[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&color);
            }
        }
        let mut gpu = reference.clone();
        // Corrupt a pixel AT the silhouette (x = 8, adjacent to the red half)
        // to a color that differs from the reference there (green).
        gpu[(8 * w + 8) * 4..(8 * w + 8) * 4 + 4].copy_from_slice(&[0, 255, 0, 255]);
        let t = flat_t(4.5, w * h);

        let report = compare(
            &gpu,
            &t,
            &reference,
            &t,
            w as u32,
            h as u32,
            CompareConfig::default(),
        );
        assert_eq!(report.mismatch_count(), 1);
        assert_eq!(report.hard_mismatch_count(), 0);
        assert!(report.passes());
    }

    #[test]
    fn t_divergence_beyond_tolerance_is_hard() {
        let rgba = flat_color([10, 20, 30, 255], 16);
        let reference_t = flat_t(10.0, 16);
        let mut gpu_t = flat_t(10.0, 16);
        gpu_t[0] = 10.5; // 5% off → beyond 1e-3 * 10

        let report = compare(
            &rgba,
            &gpu_t,
            &rgba,
            &reference_t,
            4,
            4,
            CompareConfig::default(),
        );
        assert_eq!(report.hard_mismatch_count(), 1);
        assert!(!report.passes());
    }

    #[test]
    fn small_t_divergence_passes() {
        let rgba = flat_color([10, 20, 30, 255], 16);
        let reference_t = flat_t(10.0, 16);
        let mut gpu_t = flat_t(10.0, 16);
        gpu_t[0] = 10.0001; // well inside 1e-3 * 10

        let report = compare(
            &rgba,
            &gpu_t,
            &rgba,
            &reference_t,
            4,
            4,
            CompareConfig::default(),
        );
        assert_eq!(report.hard_mismatch_count(), 0);
        assert!(report.passes());
    }

    #[test]
    fn background_miss_matches() {
        // Both sides: black, t = 0 — no mismatch.
        let rgba = flat_color([0, 0, 0, 255], 16);
        let t = flat_t(0.0, 16);
        let report = compare(&rgba, &t, &rgba, &t, 4, 4, CompareConfig::default());
        assert_eq!(report.mismatch_count(), 0);
        assert!(report.passes());
    }
}
