//! Report artifacts for a validation run: the captured GPU frame, the reference
//! frame, a diff image, and a text report identifying differing pixels.

use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use super::compare::CompareReport;
use super::runner::PassSpec;
use super::test_worlds::Camera;

/// Writes an 8-bit RGBA image as a PNG.
pub fn write_png(path: &Path, rgba: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}

/// Builds the diff image: the reference frame with mismatch pixels overlaid.
/// Red for hard mismatches (real divergences), yellow for edge-silhouette
/// mismatches (excused).
pub fn build_diff_image(reference_rgba: &[u8], report: &CompareReport) -> Vec<u8> {
    let mut diff = reference_rgba.to_vec();
    let width = report.width as usize;

    // Hard mismatches (red) first so excused ones (yellow) never overwrite
    // them; mark all in one pass.
    let mut hard = std::collections::HashSet::new();
    for mismatch in &report.hard_mismatches {
        hard.insert((mismatch.x, mismatch.y));
    }

    for mismatch in &report.mismatches {
        let color = if hard.contains(&(mismatch.x, mismatch.y)) {
            [255, 0, 0, 255]
        } else {
            [255, 255, 0, 255]
        };
        let i = (mismatch.y as usize * width + mismatch.x as usize) * 4;
        diff[i..i + 4].copy_from_slice(&color);
    }

    diff
}

/// Writes the text report with the pass/fail verdict and the differing pixels.
pub fn write_text_report(
    path: &Path,
    world_name: &str,
    world_path: &str,
    camera_description: &str,
    width: u32,
    height: u32,
    reference_seconds: f64,
    report: &CompareReport,
) -> std::io::Result<()> {
    let mut out = String::new();

    out.push_str(&format!("atlas-rt correctness validate report\n"));
    out.push_str(&format!("world: {world_name} ({world_path})\n"));
    out.push_str(&format!("camera: {camera_description}\n"));
    out.push_str(&format!(
        "frame: {width}x{height} ({} pixels)\n",
        width as usize * height as usize
    ));
    out.push_str(&format!("reference trace time: {reference_seconds:.2}s\n"));
    out.push_str(&format!(
        "t tolerance: {} * max(t, 1); mismatch budget: {:.0}% of pixels\n",
        report.config.t_tolerance,
        report.config.max_mismatch_ratio * 100.0
    ));
    out.push_str("\n");

    out.push_str(&format!(
        "mismatches: {} total, {} excused (silhouette/corner-touch), {} hard\n",
        report.mismatch_count(),
        report.mismatch_count() - report.hard_mismatch_count(),
        report.hard_mismatch_count()
    ));
    out.push_str(&format!(
        "verdict: {}\n",
        if report.passes() { "PASS" } else { "FAIL" }
    ));
    out.push_str("\n");

    if !report.hard_mismatches.is_empty() {
        out.push_str("hard mismatches (differing pixels, first 64):\n");
        out.push_str("  (x, y)  gpu color (r,g,b,a)  ref color (r,g,b,a)  gpu t  ref t\n");
        for mismatch in report.hard_mismatches.iter().take(64) {
            let gpu = mismatch.gpu;
            let reference = mismatch.reference;
            out.push_str(&format!(
                "  ({:>4}, {:>4})  ({:>3},{:>3},{:>3},{:>3})    ({:>3},{:>3},{:>3},{:>3})  {:10.6}  {:10.6}\n",
                mismatch.x,
                mismatch.y,
                gpu.color[0],
                gpu.color[1],
                gpu.color[2],
                gpu.color[3],
                reference.color[0],
                reference.color[1],
                reference.color[2],
                reference.color[3],
                gpu.t,
                reference.t,
            ));
        }
        if report.hard_mismatches.len() > 64 {
            out.push_str(&format!(
                "  ... and {} more\n",
                report.hard_mismatches.len() - 64
            ));
        }
        out.push_str("\n");
    }

    if report.passes() {
        out.push_str("The GPU frame matches the independent CPU reference tracer.\n");
    } else {
        out.push_str(
            "See gpu.png (captured frame), reference.png, and diff.png.\n\
             diff.png is the reference frame with mismatch pixels overlaid: red = hard\n\
             mismatch (real divergence), yellow = excused silhouette/corner-touch mismatch\n\
             (sub-voxel boundary effects: a 1-pixel-dilated color silhouette, or both sides\n\
             committing the same color within a voxel-ish t distance).\n\
             When there are no mismatches it is identical to reference.png.\n",
        );
    }

    let mut file = File::create(path)?;
    file.write_all(out.as_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The shading-half (CPU path-tracer diff) report artifacts (ticket 07)
// ---------------------------------------------------------------------------

/// The ACES filmic fit (Narkowicz 2015), the composite node's tonemap
/// (shaders/region/composite.comp), mirrored for the display PNGs.
fn aces_fit(x: glam::Vec3) -> glam::Vec3 {
    let (a, b, c, d, e) = (2.51f32, 0.03, 2.43, 0.59, 0.14);
    (x * (a * x + b)) / (x * (c * x + d) + e)
}

/// Tone-maps linear radiance to an RGBA8 display (ACES + gamma, like the
/// composite). The path-traced frames are linear HDR, so the PNGs are
/// display captures, not the compared quantity.
pub fn tone_map_rgba(radiance: &[glam::Vec3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(radiance.len() * 4);
    for &l in radiance {
        let color = aces_fit(l).powf(1.0 / 2.2);
        out.extend([
            (color.x.clamp(0.0, 1.0) * 255.0).round() as u8,
            (color.y.clamp(0.0, 1.0) * 255.0).round() as u8,
            (color.z.clamp(0.0, 1.0) * 255.0).round() as u8,
            255,
        ]);
    }
    out
}

/// Builds the path diff image: the tone-mapped CPU display frame with
/// mismatch pixels overlaid. Red for hard mismatches (real divergences),
/// yellow for excused ones (silhouette/firefly).
pub fn build_path_diff_image(
    cpu_display: &[glam::Vec3],
    report: &super::path_compare::PathCompareReport,
) -> Vec<u8> {
    let mut diff = tone_map_rgba(cpu_display);
    let width = report.width as usize;

    let mut hard = std::collections::HashSet::new();
    for mismatch in &report.hard_mismatches {
        hard.insert((mismatch.x, mismatch.y));
    }

    for mismatch in &report.mismatches {
        let color = if hard.contains(&(mismatch.x, mismatch.y)) {
            [255, 0, 0, 255]
        } else {
            [255, 255, 0, 255]
        };
        let i = (mismatch.y as usize * width + mismatch.x as usize) * 4;
        diff[i..i + 4].copy_from_slice(&color);
    }

    diff
}

/// Writes the shading-half text report: the pass/fail verdict, the error
/// stats, and the differing pixels.
#[allow(clippy::too_many_arguments)]
pub fn write_path_report(
    path: &std::path::Path,
    world_name: &str,
    world_path: &str,
    camera_description: &str,
    width: u32,
    height: u32,
    cpu_seconds: f64,
    report: &super::path_compare::PathCompareReport,
) -> std::io::Result<()> {
    let mut out = String::new();

    out.push_str(&format!("atlas-rt path-trace validate report\n"));
    out.push_str(&format!("world: {world_name} ({world_path})\n"));
    out.push_str(&format!("camera: {camera_description}\n"));
    out.push_str(&format!(
        "frame: {width}x{height} ({} pixels)\n",
        width as usize * height as usize
    ));
    out.push_str(&format!(
        "samples: {} per pixel (frame_seed 0..{}), identical seeds on both sides\n",
        report.samples, report.samples - 1
    ));
    out.push_str(&format!("cpu trace time: {cpu_seconds:.2}s\n"));
    out.push_str(&format!(
        "relative tolerance: {} (denominator max(|gpu|, |cpu|, {})); mismatch budget: {:.0}% of pixels\n",
        report.config.tolerance,
        report.config.abs_floor,
        report.config.max_mismatch_ratio * 100.0
    ));
    out.push_str(&format!(
        "excuses: hit-status silhouette (1px dilated), firefly (both sides > {}), corner-touch (same t, different face)\n",
        report.config.firefly_floor
    ));
    out.push_str("\n");

    out.push_str(&format!(
        "mismatches: {} total, {} excused (silhouette: {}, firefly: {}, corner-touch: {}), {} hard\n",
        report.mismatch_count(),
        report.mismatch_count() - report.hard_mismatch_count(),
        report.edge_excused(),
        report.firefly_excused(),
        report.corner_touch_excused(),
        report.hard_mismatch_count()
    ));
    out.push_str(&format!(
        "verdict: {}\n",
        if report.passes() { "PASS" } else { "FAIL" }
    ));
    out.push_str("\n");

    // Error stats over the whole frame (the compared quantity is the
    // de-modulated diffuse + raw specular radiance means).
    let mut errors: Vec<f32> = report.relative_error.iter().copied().collect();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if let Some(p) = errors.get(errors.len() / 2) {
        out.push_str(&format!(
            "relative error: p50 {:.4}, p99 {:.4}, max {:.4}\n",
            p,
            errors[(errors.len() * 99) / 100],
            errors[errors.len() - 1]
        ));
    }
    out.push_str(&format!(
        "sky fraction (gpu mean hit distance == 0): {:.1}% of pixels\n",
        report
            .gpu_hit_fraction
            .iter()
            .filter(|&&f| f <= 0.0)
            .count() as f64
            / report.total_pixels() as f64
            * 100.0
    ));
    out.push_str("\n");

    if !report.hard_mismatches.is_empty() {
        out.push_str("hard mismatches (differing pixels, first 64):\n");
        out.push_str("  (x, y)  gpu diffuse          cpu diffuse           gpu spec             cpu spec              err\n");
        for mismatch in report.hard_mismatches.iter().take(64) {
            let d = mismatch.gpu_diffuse;
            let c = mismatch.cpu_diffuse;
            let s = mismatch.gpu_specular;
            let cs = mismatch.cpu_specular;
            out.push_str(&format!(
                "  ({:>4}, {:>4})  ({:>6.3},{:>6.3},{:>6.3})  ({:>6.3},{:>6.3},{:>6.3})  ({:>6.3},{:>6.3},{:>6.3})  ({:>6.3},{:>6.3},{:>6.3})  {:5.2}\n",
                mismatch.x, mismatch.y, d.x, d.y, d.z, c.x, c.y, c.z, s.x, s.y, s.z, cs.x, cs.y,
                cs.z, mismatch.error
            ));
        }
        if report.hard_mismatches.len() > 64 {
            out.push_str(&format!(
                "  ... and {} more\n",
                report.hard_mismatches.len() - 64
            ));
        }
        out.push_str("\n");
    }

    if report.passes() {
        out.push_str(
            "The GPU path-traced radiance pair matches the independent CPU\n\
             path-tracer mirror (identical RNG seeds, per-pixel mean tolerance).\n",
        );
    } else {
        out.push_str(
            "See path-gpu.png (tonemapped GPU radiance mean), path-cpu.png (tonemapped CPU\n\
             mean), and path-diff.png (CPU frame with mismatch pixels overlaid: red = hard\n\
             mismatch, yellow = excused silhouette/firefly). The compared quantity is the\n\
             de-modulated diffuse + raw specular radiance means (ADR 0007); the PNGs are\n\
             display captures (ACES + gamma), not the compared bytes.\n",
        );
    }

    let mut file = File::create(path)?;
    file.write_all(out.as_bytes())?;
    Ok(())
}

pub const ALBEDO_EPS: f32 = 1e-3;

/// Writes the report artifacts (PNGs + text) for one frame. `label` suffixes
/// the artifact names ("" for the first/only frame, "-after-edit" for the
/// edit-at-the-seam second frame).
#[allow(clippy::too_many_arguments)]
pub fn write_report(
    out_dir: &Path,
    pass: &PassSpec,
    width: u32,
    height: u32,
    reference_seconds: f64,
    gpu_rgba: &[u8],
    reference_rgba: &[u8],
    report: &CompareReport,
    label: &str,
) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;

    write_png(
        &out_dir.join(format!("gpu{label}.png")),
        gpu_rgba,
        width,
        height,
    )?;
    write_png(
        &out_dir.join(format!("reference{label}.png")),
        reference_rgba,
        width,
        height,
    )?;

    let diff = build_diff_image(reference_rgba, report);
    write_png(
        &out_dir.join(format!("diff{label}.png")),
        &diff,
        width,
        height,
    )?;

    let camera_description = pass
        .camera
        .map(camera_description)
        .unwrap_or_else(|| "framed world bounding box".to_string());

    write_text_report(
        &out_dir.join(format!("report{label}.txt")),
        &pass.name,
        &pass.path,
        &camera_description,
        width,
        height,
        reference_seconds,
        report,
    )
}

pub fn camera_description(camera: Camera) -> String {
    format!(
        "eye {:?} target {:?} up {:?}",
        camera.eye, camera.target, camera.up
    )
}
