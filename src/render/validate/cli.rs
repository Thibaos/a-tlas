//! CLI parsing for the `atlas-rt validate` subcommand.

use std::path::PathBuf;

const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 480;
const DEFAULT_OUT_DIR: &str = "target/validate";

#[derive(Clone, Debug)]
pub struct ValidateOptions {
    pub world: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    pub gen_test_worlds: bool,
    pub list: bool,
    pub help: bool,
    /// The shading-half diff's sample count (ticket 07): the GPU runs N
    /// frames at frame_seed 0..N-1 (1 path per pixel per frame) and the CPU
    /// mirror computes the same N samples from the same seeds; the per-pixel
    /// means are compared with tolerance.
    pub path_trace_samples: u32,
    /// Disable the shading-half diff (geometry-only run).
    pub no_path_trace: bool,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            world: None,
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            gen_test_worlds: false,
            list: false,
            help: false,
            path_trace_samples: 8,
            no_path_trace: false,
        }
    }
}

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
    /// Per-frame outcomes (see [`FrameSummary`]).
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

pub fn print_help() {
    println!(
        "atlas-rt correctness validator\n\
         \n\
         Renders a test .vox world through the real renderer, captures the raw frame\n\
         before any overlay, traces the same frame with an independent CPU reference\n\
         tracer, and reports per-pixel {{color, t}} mismatches.\n\
         \n\
         USAGE:\n\
         \x20 atlas-rt validate [OPTIONS]\n\
         \n\
         OPTIONS:\n\
         \x20 --world <path>       Run a single world (default: the whole suite:\n\
         \x20                      assets/test/* + assets/custom.vox smoke)\n\
         \x20 --out <dir>          Report output directory (default: {DEFAULT_OUT_DIR})\n\
         \x20 --width <n>          Frame width  (default {DEFAULT_WIDTH})\n\
         \x20 --height <n>         Frame height (default {DEFAULT_HEIGHT})\n\
         \x20 --path-trace-samples <n>  Shading-half diff sample count\n\
         \x20                      (default 8: the GPU runs N frames at\n\
         \x20                      frame_seed 0..N-1, the CPU mirror N\n\
         \x20                      identical-seed samples; means compared)\n\
         \x20 --no-path-trace      Skip the shading-half diff (geometry only)\n\
         \x20 --gen-test-worlds    (Re)write the hand-authored test worlds to\n\
         \x20                      assets/test/ and exit\n\
         \x20 --list               List the suite and exit\n\
         \x20 --help               Show this help\n"
    );
}

pub fn parse_args(args: &[String]) -> Result<ValidateOptions, String> {
    let mut opts = ValidateOptions::default();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--world" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--world requires a path".to_string())?;
                opts.world = Some(PathBuf::from(value));
            }
            "--out" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--out requires a directory".to_string())?;
                opts.out_dir = PathBuf::from(value);
            }
            "--width" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--width requires a number".to_string())?;
                opts.width = value
                    .parse()
                    .map_err(|_| format!("invalid --width: {value}"))?;
            }
            "--height" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--height requires a number".to_string())?;
                opts.height = value
                    .parse()
                    .map_err(|_| format!("invalid --height: {value}"))?;
            }
            "--gen-test-worlds" => opts.gen_test_worlds = true,
            "--list" => opts.list = true,
            "--path-trace-samples" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--path-trace-samples requires a number".to_string())?;
                opts.path_trace_samples = value
                    .parse()
                    .map_err(|_| format!("invalid --path-trace-samples: {value}"))?;
            }
            "--no-path-trace" => opts.no_path_trace = true,
            "--help" | "-h" => opts.help = true,
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if opts.width == 0 || opts.height == 0 {
        return Err("--width and --height must be nonzero".to_string());
    }

    if opts.path_trace_samples == 0 {
        return Err("--path-trace-samples must be nonzero".to_string());
    }

    Ok(opts)
}

