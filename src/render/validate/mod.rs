use std::path::{Path, PathBuf};

use winit::event_loop::EventLoop;

use crate::{
    core::gpu::GpuStack,
    render::validate::{
        cli::{parse_args, print_help},
        runner::ValidateRunner,
        test_worlds::{WorldSpec, all_worlds, generate_all},
    },
};

pub mod capture;
pub mod cli;
pub mod compare;
pub mod path_compare;
pub mod path_tracer;
pub mod readback;
pub mod reference;
pub mod report;
pub mod runner;
pub mod test_worlds;

/// One compared frame's outcome. The edit-at-the-seam world
/// runs two frames (label "" then "-after-edit"); every other world runs
/// one.
pub struct FrameSummary {
    pub label: String,
    pub pass: bool,
    pub mismatches: usize,
    pub hard_mismatches: usize,
}

/// Result of one world's run (aggregated over its frames).
pub struct PassSummary {
    pub name: String,
    pub path: String,
    pub pass: bool,
    pub mismatches: usize,
    pub hard_mismatches: usize,
    pub out_dir: PathBuf,
    pub frames: Vec<FrameSummary>,
}

/// Result of one world's shading-half diff (ticket 07): the CPU path-tracer
/// mirror vs the captured GPU radiance pair, per-pixel means over N
/// identical-seed samples.
pub struct PathPassSummary {
    pub name: String,
    pub pass: bool,
    pub mismatches: usize,
    pub hard_mismatches: usize,
    pub samples: u32,
}

pub fn run(args: &[String]) -> Result<(), String> {
    let opts = parse_args(args)?;

    if opts.help {
        print_help();
        return Ok(());
    }

    if opts.gen_test_worlds {
        let paths = generate_all(Path::new(test_worlds::TEST_WORLDS_DIR))
            .map_err(|e| format!("failed to write test worlds: {e}"))?;
        for path in &paths {
            println!("wrote {}", path.display());
        }
        return Ok(());
    }

    if opts.list {
        for world in all_worlds() {
            println!(
                "{:>14}  {:<28} {}",
                world.name, world.path, world.description
            );
        }
        return Ok(());
    }

    let suite = all_worlds();
    let missing: Vec<_> = suite
        .iter()
        .filter(|w| w.path.starts_with("assets/test/") && !Path::new(&w.path).exists())
        .map(|w| w.name.clone())
        .collect();
    if !missing.is_empty() {
        println!("generating missing test worlds: {}", missing.join(", "));
        generate_all(Path::new(test_worlds::TEST_WORLDS_DIR))
            .map_err(|e| format!("failed to write test worlds: {e}"))?;
    }

    let worlds: Vec<WorldSpec> = match &opts.world {
        Some(path) => vec![WorldSpec {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("world")
                .to_string(),
            path: path.to_str().unwrap_or_default().to_string(),
            description: String::new(),
            camera: None,
            edit: None,
        }],
        None => suite,
    };

    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    let gpu = GpuStack::new(&event_loop);

    let mut app = ValidateRunner {
        gpu,
        opts: opts.clone(),
        worlds,
        results: Vec::new(),
        path_results: Vec::new(),
        done: false,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop: {e}"))?;

    let mut failures = 0;

    for (result, path) in app.results.iter().zip(&app.path_results) {
        match result {
            Ok(summary) => {
                println!(
                    "[{:>15}] {}  (mismatches: {}, hard: {})",
                    summary.name,
                    if summary.pass { "PASS" } else { "FAIL" },
                    summary.mismatches,
                    summary.hard_mismatches,
                );

                for frame in &summary.frames {
                    if frame.label.is_empty() {
                        continue;
                    }
                    println!(
                        "              {}: {} (mismatches: {}, hard: {}), report: {}",
                        frame.label,
                        if frame.pass { "PASS" } else { "FAIL" },
                        frame.mismatches,
                        frame.hard_mismatches,
                        summary.out_dir.join("report.txt").display()
                    );
                }

                if !summary.pass {
                    failures += 1;
                }
            }
            Err(error) => {
                eprintln!("ERROR: {error}");
                failures += 1;
            }
        }

        if let Some(path) = path {
            println!(
                "[{:>15}] PATH {}  (N={}, mismatches: {}, hard: {})",
                path.name,
                if path.pass { "PASS" } else { "FAIL" },
                path.samples,
                path.mismatches,
                path.hard_mismatches,
            );
            if !path.pass {
                failures += 1;
            }
        }
    }

    if failures > 0 {
        Err(format!("{failures} of {} worlds failed", app.results.len()))
    } else {
        Ok(())
    }
}
