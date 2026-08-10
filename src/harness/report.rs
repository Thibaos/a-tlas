//! Report artifacts for a harness run: the captured GPU frame, the reference
//! frame, a diff image, and a text report identifying differing pixels.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::harness::compare::CompareReport;

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

/// Builds the diff image: the reference frame with mismatch pixels overlaid —
/// red for hard mismatches (real divergences), yellow for edge-silhouette
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

    out.push_str(&format!("a-tlas correctness harness report\n"));
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
        "mismatches: {} total, {} excused (edge silhouette), {} hard\n",
        report.mismatch_count(),
        report.mismatch_count() - report.hard_mismatch_count(),
        report.hard_mismatch_count()
    ));
    out.push_str(&format!(
        "verdict: {}\n",
        if report.passes() {
            "PASS"
        } else {
            "FAIL"
        }
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
        out.push_str("See gpu.png (captured frame), reference.png, and diff.png.\n");
    }

    let mut file = File::create(path)?;
    file.write_all(out.as_bytes())?;
    Ok(())
}
