//! The correctness validator: the single test seam
//! for the whole rendering effort.
//!
//! For each world it loads the .vox through the real world loader, renders
//! one frame through the real renderer into the (hidden) swapchain, captures
//! the raw color frame plus a t-channel before anything else draws, traces
//! the same frame with the independent CPU reference tracer, and writes a
//! comparison report (PNGs + text) under `--out`.
//!
//! Usage (see `atlas-rt validate --help`):
//!   cargo run -- validate                      # run the whole suite
//!   cargo run -- validate --world assets/test/single.vox
//!   cargo run -- validate --gen-test-worlds    # (re)write assets/test/*.vox

use std::{
    f32::consts::PI,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use glam::{IVec3, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    format::Format,
    image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::StorageImageId,
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{AccessTypes, HostAccessType, ImageLayoutType, Resources},
    resource_map,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

use crate::{
    app::{GpuStack, MIN_SWAPCHAIN_IMAGES},
    grid::{MICRO_CHUNK_EDGE, grid_origin},
    region::{
        input::RendererInput,
        pack::pack_regions,
        rebuild::RebuildLogEntry,
        render::{RegionRenderContext, RegionRenderTask, RenderMode, default_scene},
        residency::RegionStore,
        snapshot::{MicroChunkSnapshot, emit_snapshots},
    },
    validate::{
        capture::{CaptureTask, PathCaptureTask},
        compare::{CompareConfig, CompareReport, compare},
        path_compare::{PathCompareConfig, PathCompareReport, compare_path},
        path_tracer::{PathRender, PathTracer, render_path},
        reference::{CameraInputs, ReferenceTracer, VoxelShape, render_reference},
        report::{
            build_diff_image, build_path_diff_image, tone_map_rgba, write_path_report, write_png,
            write_text_report,
        },
        test_worlds::{Camera, WorldSpec, all_worlds, generate_all},
    },
    world::{
        voxel::{get_material_table, get_palette, open_file},
        world::World,
    },
};

pub mod capture;
pub mod compare;
pub mod path_compare;
pub mod path_tracer;
pub mod reference;
pub mod report;
pub mod test_worlds;

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

/// Entry point for `atlas-rt validate ...`.
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

    // The test worlds are committed assets; regenerate only if missing.
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

    // winit can only create one event loop per process (Windows), so all
    // worlds run through the same loop + GPU stack.
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    let gpu = GpuStack::new(&event_loop);

    let mut app = ValidateApp {
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

        // The shading-half diff (ticket 07): the CPU path-tracer mirror vs
        // the captured GPU radiance pair (identical RNG seeds, per-pixel
        // means, tolerance).
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

/// Static per-pass data (the world's identity + camera).
struct PassSpec {
    name: String,
    path: String,
    camera: Option<Camera>,
}

/// Everything shared between a world's frames: the hidden window, the
/// swapchain with its bindless storage images, the t-image and the readback
/// buffers, and the shading-half resources (ticket 07): the production ray
/// pass's six output images (the radiance pair + albedo aux captured) and
/// their readback buffers. Built once per world; each frame gets its own task
/// graph over the same resources (the edit-at-the-seam world runs two
/// frames).
struct FrameSetup {
    #[allow(dead_code)]
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    swapchain_format: Format,
    swapchain_storage_image_ids: Vec<StorageImageId>,
    t_image_id: Id<Image>,
    t_image_storage_id: StorageImageId,
    t_format: Format,
    color_readback_buffer_id: Id<Buffer>,
    t_readback_buffer_id: Id<Buffer>,
    camera: crate::region::render::capture_raygen::Camera,
    camera_inputs: CameraInputs,
    /// The shading-half output images (ticket 07): the production raygen's
    /// six-buffer set (ADR 0007); diff/spec/albedo are captured per frame.
    path_images: PathTraceImages,
    path_diff_readback_buffer_id: Id<Buffer>,
    path_spec_readback_buffer_id: Id<Buffer>,
    path_albedo_readback_buffer_id: Id<Buffer>,
}

struct ValidateFrame {
    virtual_swapchain_id: Id<Swapchain>,
    swapchain_id: Id<Swapchain>,
    swapchain_format: Format,
    color_readback_buffer_id: Id<Buffer>,
    t_readback_buffer_id: Id<Buffer>,
    task_graph: ExecutableTaskGraph<RegionRenderContext>,
    rcx: RegionRenderContext,
    camera_inputs: CameraInputs,
    /// The shading-half graph (ticket 07): the production ray pass + the
    /// radiance-pair capture. Executed once per path-traced frame (the
    /// frame_seed is set on the shared `rcx`).
    path_virtual_swapchain_id: Id<Swapchain>,
    path_task_graph: ExecutableTaskGraph<RegionRenderContext>,
    /// The radiance-pair + albedo + normal readback buffers (copied per path
    /// frame).
    path_diff_readback_buffer_id: Id<Buffer>,
    path_spec_readback_buffer_id: Id<Buffer>,
    path_albedo_readback_buffer_id: Id<Buffer>,
}

struct ValidateApp {
    gpu: GpuStack,
    opts: ValidateOptions,
    worlds: Vec<WorldSpec>,
    results: Vec<Result<PassSummary, String>>,
    /// The shading-half diff's outcome per world (parallel to `results`;
    /// `None` when the run was geometry-only).
    path_results: Vec<Option<PathPassSummary>>,
    done: bool,
}

impl ApplicationHandler for ValidateApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }

        let worlds = std::mem::take(&mut self.worlds);
        let mut results = Vec::new();
        let mut path_results = Vec::new();
        for world in worlds {
            match self.run_world(&world, event_loop) {
                Ok((summary, path)) => {
                    results.push(Ok(summary));
                    path_results.push(path);
                }
                Err(error) => {
                    results.push(Err(error));
                    path_results.push(None);
                }
            }
        }
        self.results = results;
        self.path_results = path_results;

        self.done = true;
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            event_loop.exit();
        }
    }
}

impl ValidateApp {
    /// Runs one full pass over a single world: hidden window + swapchain +
    /// ray pass + capture + reference trace + comparison + report files, and
    /// (unless disabled) the shading-half diff: the production ray pass at
    /// frame_seed 0..N-1 captured against the CPU path-tracer mirror.
    fn run_world(
        &mut self,
        world: &WorldSpec,
        event_loop: &ActiveEventLoop,
    ) -> Result<(PassSummary, Option<PathPassSummary>), String> {
        let width = self.opts.width;
        let height = self.opts.height;

        let voxel_data = open_file(&world.path);
        let mut world_data = Arc::new(World::new(&voxel_data));
        let voxel_count = world_data.voxel_count();
        // The destination path's in-shader DDA resolves grid cells,
        // so the reference traces the same shape:
        // [p, p + 1) per voxel.
        let shape = VoxelShape::GridCell;

        println!(
            "[{:>15}] {} ({} voxels) — rendering…",
            world.name, world.path, voxel_count
        );

        let window_attributes = WindowAttributes::default()
            .with_inner_size(PhysicalSize::new(width, height))
            .with_visible(false);

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let surface = Surface::from_window(&self.gpu.instance, &window).unwrap();

        let (swapchain_id, swapchain_format) =
            create_validate_swapchain(&self.gpu, &surface, [width, height])
                .map_err(|e| format!("swapchain: {e}"))?;

        let swapchain_storage_image_ids =
            crate::app::window_size_dependent_setup(&self.gpu.resources, swapchain_id);

        let (t_image_id, t_image_storage_id, t_format) =
            create_t_image(&self.gpu.resources, width, height)
                .map_err(|e| format!("t image: {e}"))?;

        let color_readback_buffer_id = self
            .gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[u8]>((width * height * 4) as u64).unwrap(),
            )
            .unwrap();

        let t_readback_buffer_id = self
            .gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[f32]>((width * height * 4) as u64).unwrap(),
            )
            .unwrap();

        let (camera, camera_inputs) = build_camera(&world_data, width, height, world.camera);

        // The shading-half resources (ticket 07): the production ray pass's
        // six output images (diff/spec RGBA16F, albedo RGBA8 captured) and
        // the readback buffers for the radiance pair + albedo aux.
        let path_images = create_path_trace_images(&self.gpu.resources, width, height);
        let path_diff_readback_buffer_id =
            create_host_readback(&self.gpu, (u64::from(width) * u64::from(height) * 8) as u64);
        let path_spec_readback_buffer_id =
            create_host_readback(&self.gpu, (u64::from(width) * u64::from(height) * 8) as u64);
        let path_albedo_readback_buffer_id =
            create_host_readback(&self.gpu, (u64::from(width) * u64::from(height) * 4) as u64);

        let setup = FrameSetup {
            window,
            swapchain_id,
            swapchain_format,
            swapchain_storage_image_ids,
            t_image_id,
            t_image_storage_id,
            t_format,
            color_readback_buffer_id,
            t_readback_buffer_id,
            camera,
            camera_inputs,
            path_images,
            path_diff_readback_buffer_id,
            path_spec_readback_buffer_id,
            path_albedo_readback_buffer_id,
        };

        // Startup: the world voices its initial state as one submit_batch;
        // the worker drains it into per-Region mirrors; the residency
        // manager builds the lattice from the packed mirrors
        // (never the world directly).
        let input = RendererInput::new();
        input.submit_batch(emit_snapshots(&world_data));
        input.wait_until_idle();

        let mut store = RegionStore::new(&self.gpu, &voxel_data, input.packed_regions());

        // The task graph is built once per world: the store's buffers are
        // stable across frames (residency rebuilds rewrite them in place),
        // so every frame executes the same graph.
        let mut frame = self.build_validate_frame(&setup, &store)?;

        // Frame 1: startup via submit_batch.
        let (first, first_path) =
            self.run_frame(&world_data, &voxel_data, shape, world, &mut frame, "")?;

        let mut frames = vec![first];
        let mut path_frames = vec![first_path];

        // Frames 2..N: each step mutates
        // the world, voices the change through the contract (zero-mask
        // snapshots for emptied Micro-chunks, fresh snapshots for added
        // voxels), the residency manager consumes the change cycle, and the
        // next frame must match the reference over the edited world.
        if let Some(script) = world.edit.clone() {
            for step in script.steps {
                // --- apply the step to the world -------------------------
                for mc in &step.remove_microchunks {
                    let cells: Vec<IVec3> = {
                        let world =
                            Arc::get_mut(&mut world_data).expect("the validator owns the world");
                        world
                            .iter_voxels()
                            .filter(|(p, _)| {
                                p.cmpge(*mc).all()
                                    && p.cmplt(*mc + IVec3::splat(MICRO_CHUNK_EDGE)).all()
                            })
                            .map(|(p, _)| p)
                            .collect()
                    };
                    assert!(
                        !cells.is_empty(),
                        "edit step {:?}: no voxels inside Micro-chunk {mc:?}",
                        step.label
                    );
                    {
                        let world =
                            Arc::get_mut(&mut world_data).expect("the validator owns the world");
                        for position in cells {
                            assert!(
                                world.remove_voxel_at(position),
                                "edit step {:?}: voxel {position:?} already absent",
                                step.label
                            );
                        }
                    }
                }
                {
                    let world =
                        Arc::get_mut(&mut world_data).expect("the validator owns the world");
                    for &(position, material) in &step.add_voxels {
                        world.insert_voxel_at(position, material.into());
                    }
                }
                assert!(
                    world_data.voxel_count() > 0,
                    "edit step {:?}: the world must survive the edit",
                    step.label
                );

                // --- voice the change through the contract ----------------
                for mc in &step.remove_microchunks {
                    input.submit_microchunk(MicroChunkSnapshot {
                        global_coords: *mc,
                        mask: [0u8; 64],
                        materials: Vec::new(),
                    });
                }
                if !step.add_voxels.is_empty() {
                    let affected: std::collections::HashSet<IVec3> = step
                        .add_voxels
                        .iter()
                        .map(|(p, _)| grid_origin(*p, MICRO_CHUNK_EDGE))
                        .collect();
                    let voiced: Vec<_> = emit_snapshots(&world_data)
                        .into_iter()
                        .filter(|s| affected.contains(&s.global_coords))
                        .collect();
                    assert!(
                        !voiced.is_empty(),
                        "edit step {:?}: no snapshots voiced for the added voxels",
                        step.label
                    );
                    input.submit_batch(voiced);
                }
                input.wait_until_idle();

                // The seam: the change-path pack equals the direct pack of
                // the edited world — the exact bytes the pipeline consumes.
                let expected = pack_regions(&emit_snapshots(&world_data));
                let actual = input.packed_regions();
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.region_index, b.region_index);
                    assert_eq!(a.blocks, b.blocks);
                    assert_eq!(a.aabbs, b.aabbs);
                }

                // --- the residency manager consumes the change cycle -----
                let report = store.apply(&self.gpu, &input);

                // Invariants (every step): the TLAS rebuilds iff a residency
                // transition or a BLAS capacity replacement happened (the
                // only instance-set/instance-data changes), and the store's
                // resident set matches the input contract's mirrors.
                let transitioned =
                    !report.became_resident.is_empty() || !report.left_resident.is_empty();
                assert_eq!(
                    report.tlas_rebuilt,
                    transitioned || !report.blas_replaced.is_empty(),
                    "TLAS instance set/data must change only on residency transitions or BLAS replacements"
                );
                // The instance set equals the mirrors' set (sorted by id).
                let mirror_ids: Vec<u32> = input
                    .packed_regions()
                    .iter()
                    .map(|region| region.region_id())
                    .collect();
                assert_eq!(
                    store.resident_ids(),
                    mirror_ids.as_slice(),
                    "the TLAS instance set must match the input contract's mirrors"
                );
                assert_eq!(
                    store.resident_count(),
                    input.region_count(),
                    "the store's resident count must match the input contract's mirrors"
                );

                // The rebuild logs/counters: a content edit rebuilds the
                // Region's BLAS **in place** (device address stable → TLAS
                // untouched — no `BuildTlas` entry); a residency transition
                // rebuilds the TLAS, adding/removing exactly one instance
                // (the scripts transition one Region per step).
                let log = &report.rebuild_log;
                assert_eq!(
                    log.iter()
                        .any(|e| matches!(e, RebuildLogEntry::BuildTlas { .. })),
                    report.tlas_rebuilt,
                    "the rebuild log must record a TLAS build iff the TLAS rebuilt"
                );
                if report.tlas_rebuilt {
                    let net =
                        report.became_resident.len() as isize - report.left_resident.len() as isize;
                    assert_eq!(
                        report.instance_count as isize,
                        report.instance_count_before as isize + net,
                        "the TLAS instance set changes only by the residency transitions ({} became, {} left)",
                        report.became_resident.len(),
                        report.left_resident.len(),
                    );
                    assert!(
                        log.iter()
                            .any(|e| matches!(e, RebuildLogEntry::RewriteInstances { .. })),
                        "a TLAS rebuild must be preceded by an instance rewrite"
                    );
                }
                if !transitioned && report.blas_replaced.is_empty() {
                    // A pure content edit: every dirty Region's BLAS rebuilt
                    // in place, TLAS untouched.
                    for region in &report.dirty {
                        assert!(
                            log.iter().any(|e| matches!(
                                e,
                                RebuildLogEntry::BuildBlas {
                                    region_index: r,
                                    fresh: false,
                                    ..
                                } if r == region
                            )),
                            "content edit of region {region:?} must rebuild its BLAS in place (TLAS untouched)"
                        );
                    }
                }
                // Per-node GPU attribution ):
                // real builds report nonzero times when timestamps exist.
                if report.timings.supported {
                    if report.tlas_rebuilt {
                        assert!(
                            report.timings.tlas_ns > 0,
                            "the TLAS rebuild's GPU time must be attributable (tlas_ns > 0)"
                        );
                    }
                    if log
                        .iter()
                        .any(|e| matches!(e, RebuildLogEntry::BuildBlas { .. }))
                    {
                        assert!(
                            report.timings.blas_ns > 0,
                            "the BLAS rebuilds' GPU time must be attributable (blas_ns > 0)"
                        );
                    }
                }
                if !report.rebuild_log.is_empty() {
                    println!(
                        "              rebuild {}: upload {:>6} ns, blas {:>6} ns, tlas {:>6} ns{}",
                        step.label,
                        report.timings.upload_ns,
                        report.timings.blas_ns,
                        report.timings.tlas_ns,
                        if report.timings.supported {
                            ""
                        } else {
                            " (timestamps unsupported)"
                        }
                    );
                }

                let (summary, path) = self.run_frame(
                    &world_data,
                    &voxel_data,
                    shape,
                    world,
                    &mut frame,
                    &step.label,
                )?;
                frames.push(summary);
                path_frames.push(path);
            }
        }

        // The residency world's scripted transitions: Region (1,0,0) leaves
        // residency when emptied, and its re-population re-creates it — from
        // the free lists, not fresh allocations (the ordering invariant's
        // probe: freed AS memory is reused only after the rebuild that
        // dropped the referencing instance executed). Gated on the world
        // name because only this world's script exercises the empty →
        // re-populate cycle.
        if world.name == "residency" && world.edit.is_some() {
            assert_eq!(
                store.alloc_stats.pool_allocations, 3,
                "no fresh pool beyond the initial 3 regions"
            );
            assert_eq!(
                store.alloc_stats.blas_allocations, 3,
                "no fresh BLAS beyond the initial 3 regions"
            );
            assert_eq!(
                store.alloc_stats.pool_reuses, 1,
                "the re-populated Region reuses its freed pool"
            );
            assert_eq!(
                store.alloc_stats.blas_reuses, 1,
                "the re-populated Region reuses its freed BLAS"
            );
        }

        // The shading-half diff aggregates over the world's frames (the
        // path-tracer mirror runs against every frame's world state).
        let path_summary = if self.opts.no_path_trace {
            None
        } else {
            let pass = path_frames
                .iter()
                .all(|p| p.as_ref().is_some_and(|p| p.pass));
            Some(PathPassSummary {
                name: world.name.clone(),
                pass,
                mismatches: path_frames
                    .iter()
                    .filter_map(|p| p.as_ref())
                    .map(|p| p.mismatches)
                    .sum(),
                hard_mismatches: path_frames
                    .iter()
                    .filter_map(|p| p.as_ref())
                    .map(|p| p.hard_mismatches)
                    .sum(),
                samples: self.opts.path_trace_samples,
            })
        };

        Ok((
            PassSummary {
                name: world.name.clone(),
                path: world.path.clone(),
                pass: frames.iter().all(|frame| frame.pass),
                mismatches: frames.iter().map(|frame| frame.mismatches).sum(),
                hard_mismatches: frames.iter().map(|frame| frame.hard_mismatches).sum(),
                out_dir: self.opts.out_dir.join(&world.name),
                frames,
            },
            path_summary,
        ))
    }

    /// Builds the compiled task graph over the store's stable buffers.
    /// The setup's swapchain/images/buffers are shared; the graph is built **once per
    /// world** and every frame executes it — residency rebuilds rewrite the
    /// store's buffers in place, so the ids never move.
    fn build_validate_frame(
        &self,
        setup: &FrameSetup,
        store: &RegionStore,
    ) -> Result<ValidateFrame, String> {
        let mut task_graph = TaskGraph::new(&self.gpu.resources);
        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        // The validation render task: the same Region render task the app
        // runs, but with the capture raygen (color + t-channel for the
        // per-pixel {color, t} comparison).
        let raygen = unsafe {
            crate::region::render::capture_raygen::load(&self.gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };
        let rt_task = RegionRenderTask::new(
            &self.gpu,
            store,
            virtual_swapchain_id,
            &raygen,
            // No measurement pool: the validator
            // never measures — no timestamp commands are recorded, so the
            // captured frames are bit-identical with or without measurement
            // (which runs on demand in the app only).
            None,
            // No counter: the validator's intersection shader is specialized
            // with COUNTER_ENABLED = false (no atomicAdd), so the captured
            // frames stay byte-identical.
            None,
            // No hull-crossed counter: the validator's capture raygen hardcodes
            // Voxel, so the hull-crossed hit group is never selected and the
            // captured frames stay byte-identical.
            None,
            // The capture pipeline's miss shader stays black: the byte-exact
            // Reference comparison is against a constant background, so the
            // sky (ticket 06) is production-only.
            false,
        );
        let instance_buffer_id = rt_task.instance_buffer_id();

        let rt_node_id = task_graph
            .create_task_node("ValidateRender", QueueFamilyType::Graphics, rt_task)
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .image_access(
                setup.t_image_id,
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .buffer_access(
                instance_buffer_id,
                AccessTypes::RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
            )
            .build();

        // The capture node reads both images in the GENERAL layout.
        //
        // NOTE (vulkano-taskgraph, pinned eae054666): declaring the swapchain
        // image access here as `ImageLayoutType::Optimal` (which transitions
        // General -> TransferSrcOptimal between the nodes) makes the ray pass
        // miss everything — the trace in the previous node reports no hits.
        // Reading in General (no layout transition) is required; verified by
        // isolating the capture node's accesses. The copy command itself
        // accepts a General-layout source, so nothing is lost.
        let capture_task = CaptureTask::new(
            virtual_swapchain_id,
            setup.t_image_id,
            setup.t_format,
            setup.color_readback_buffer_id,
            setup.t_readback_buffer_id,
        );

        let capture_node_id = task_graph
            .create_task_node("Capture", QueueFamilyType::Graphics, capture_task)
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COPY_TRANSFER_READ,
                ImageLayoutType::General,
            )
            .image_access(
                setup.t_image_id,
                AccessTypes::COPY_TRANSFER_READ,
                ImageLayoutType::General,
            )
            .build();

        // The capture is ordered strictly after the ray pass (and before any
        // overlay node would run), so the copied bytes are the raw renderer
        // output — "capture before the debug overlay draws".
        task_graph.add_edge(rt_node_id, capture_node_id).unwrap();

        task_graph.add_host_buffer_access(setup.color_readback_buffer_id, HostAccessType::Read);
        task_graph.add_host_buffer_access(setup.t_readback_buffer_id, HostAccessType::Read);

        let task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.gpu.graphics_queue],
                present_queue: Some(&self.gpu.graphics_queue),
                flight_id: self.gpu.graphics_flight_id,
                ..Default::default()
            })
        }
        .map_err(|e| format!("compile: {e}"))?;

        // The shading-half graph (ticket 07): the production ray pass (the
        // app's raygen with the sky miss shader and the real Scene
        // constants) writing the six output images, then a capture node
        // copying the radiance pair + albedo aux to host-readable buffers.
        // The graph is rebuilt per world (no resize) and executed once per
        // path-traced frame with the shared `rcx`'s frame_seed set.
        let path_raygen = unsafe {
            crate::region::render::production_raygen::load(&self.gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };
        let path_rt_task = RegionRenderTask::new(
            &self.gpu,
            store,
            virtual_swapchain_id,
            &path_raygen,
            None,
            None,
            None,
            // The production pipeline's miss shader returns the Procedural
            // sky (ticket 06) — the shading half compares the real output.
            true,
        );
        let path_instance_buffer_id = path_rt_task.instance_buffer_id();

        let mut path_graph = TaskGraph::new(&self.gpu.resources);
        let path_virtual_swapchain_id = path_graph.add_swapchain(&SwapchainCreateInfo::default());

        let mut path_rt_node =
            path_graph.create_task_node("PathRender", QueueFamilyType::Graphics, path_rt_task);
        path_rt_node.image_access(
            path_virtual_swapchain_id.current_image_id(),
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        // The production raygen's six-buffer set (ADR 0007) — physical ids
        // referenced directly (the validator never resizes, so the app's
        // virtual-id indirection is unnecessary).
        for image in [
            &setup.path_images.diff,
            &setup.path_images.spec,
            &setup.path_images.normal_roughness,
            &setup.path_images.viewz,
            &setup.path_images.mv,
            &setup.path_images.albedo,
        ] {
            path_rt_node.image_access(
                image.physical_id,
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            );
        }
        path_rt_node.buffer_access(
            path_instance_buffer_id,
            AccessTypes::RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        );
        let path_rt_node_id = path_rt_node.build();

        let path_capture_task = PathCaptureTask::new(
            setup.path_images.diff.physical_id,
            setup.path_images.spec.physical_id,
            setup.path_images.albedo.physical_id,
            setup.path_diff_readback_buffer_id,
            setup.path_spec_readback_buffer_id,
            setup.path_albedo_readback_buffer_id,
            self.opts.width,
            self.opts.height,
        );
        let path_capture_node_id = path_graph
            .create_task_node("PathCapture", QueueFamilyType::Graphics, path_capture_task)
            .image_access(
                setup.path_images.diff.physical_id,
                AccessTypes::COPY_TRANSFER_READ,
                ImageLayoutType::General,
            )
            .image_access(
                setup.path_images.spec.physical_id,
                AccessTypes::COPY_TRANSFER_READ,
                ImageLayoutType::General,
            )
            .image_access(
                setup.path_images.albedo.physical_id,
                AccessTypes::COPY_TRANSFER_READ,
                ImageLayoutType::General,
            )
            .build();
        path_graph
            .add_edge(path_rt_node_id, path_capture_node_id)
            .unwrap();
        path_graph.add_host_buffer_access(setup.path_diff_readback_buffer_id, HostAccessType::Read);
        path_graph.add_host_buffer_access(setup.path_spec_readback_buffer_id, HostAccessType::Read);
        path_graph
            .add_host_buffer_access(setup.path_albedo_readback_buffer_id, HostAccessType::Read);

        let path_task_graph = unsafe {
            path_graph.compile(&CompileInfo {
                queues: &[&self.gpu.graphics_queue],
                present_queue: Some(&self.gpu.graphics_queue),
                flight_id: self.gpu.graphics_flight_id,
                ..Default::default()
            })
        }
        .map_err(|e| format!("path compile: {e}"))?;

        let rcx = RegionRenderContext {
            camera: setup.camera,
            // The analytic lights' constants (ticket 06): written into the
            // Scene buffer every frame (the production sky miss + NEE read
            // them). The capture path never dereferences the buffer, so the
            // byte-exact geometry validator is unchanged.
            scene: default_scene(),
            swapchain_storage_image_ids: setup.swapchain_storage_image_ids.clone(),
            t_image_storage_id: setup.t_image_storage_id,
            // The shading-half output set (ADR 0007): the production raygen
            // writes all six in Voxel mode. The capture raygen never
            // dereferences them, so the shared rcx serves both graphs.
            diff_radiance_image_id: setup.path_images.diff.storage_id,
            spec_radiance_image_id: setup.path_images.spec.storage_id,
            normal_roughness_image_id: setup.path_images.normal_roughness.storage_id,
            viewz_image_id: setup.path_images.viewz.storage_id,
            mv_image_id: setup.path_images.mv.storage_id,
            albedo_metal_image_id: setup.path_images.albedo.storage_id,
            ev: 0.0,
            // The validator is Voxel-only: the capture raygen hardcodes
            // sbtRecordOffset = 0 and never toggles to Hull; the production
            // raygen path-traces (mode 0) and writes the radiance pair.
            mode: RenderMode::default(),
            // The path-tracing RNG's per-frame seed: set by the shading
            // loop (0..N-1) before each path frame; the capture raygen never
            // reads it.
            frame_seed: 0,
        };

        Ok(ValidateFrame {
            virtual_swapchain_id,
            swapchain_id: setup.swapchain_id,
            swapchain_format: setup.swapchain_format,
            color_readback_buffer_id: setup.color_readback_buffer_id,
            t_readback_buffer_id: setup.t_readback_buffer_id,
            task_graph,
            rcx,
            camera_inputs: setup.camera_inputs,
            path_virtual_swapchain_id,
            path_task_graph,
            path_diff_readback_buffer_id: setup.path_diff_readback_buffer_id,
            path_spec_readback_buffer_id: setup.path_spec_readback_buffer_id,
            path_albedo_readback_buffer_id: setup.path_albedo_readback_buffer_id,
        })
    }

    /// Executes one frame, captures color + t, traces the reference,
    /// compares, and writes the report artifacts (suffixed by `label`: ""
    /// for the first/only frame, the step label for edit-at-the-seam
    /// frames). Also runs the shading-half diff (ticket 07) unless disabled:
    /// the CPU path-tracer mirror vs the production ray pass captured at
    /// frame_seed 0..N-1.
    fn run_frame(
        &mut self,
        world_data: &Arc<World>,
        voxel_data: &dot_vox::DotVoxData,
        shape: VoxelShape,
        world: &WorldSpec,
        frame: &mut ValidateFrame,
        label: &str,
    ) -> Result<(FrameSummary, Option<PathPassSummary>), String> {
        let width = self.opts.width;
        let height = self.opts.height;
        let out_dir = self.opts.out_dir.join(&world.name);

        let gpu = &self.gpu;

        // Wait for any prior flight work, execute the graph, then wait again
        // so the host reads below see completed copies.
        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        let resource_map =
            resource_map!(&frame.task_graph, frame.virtual_swapchain_id => frame.swapchain_id)
                .unwrap();

        let execute_result = unsafe { frame.task_graph.execute(resource_map, &frame.rcx, || {}) };

        if let Err(error) = execute_result {
            return Err(format!("frame execution failed: {error:?}"));
        }

        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        // --- read back the captured color frame and t channel ------------
        let color_bytes = read_host_bytes(gpu, frame.color_readback_buffer_id);
        let t_floats = read_host_floats(gpu, frame.t_readback_buffer_id);

        let gpu_rgba = decode_rgba(frame.swapchain_format, &color_bytes);
        let gpu_t: Vec<f32> = t_floats.iter().step_by(4).copied().collect();

        // --- reference trace ---------------------------------------------
        let palette = get_palette(voxel_data);
        let tracer = ReferenceTracer::new(world_data, palette, shape);

        let mut reference_rgba = vec![0u8; (width * height * 4) as usize];
        let mut reference_t = vec![0f32; (width * height) as usize];

        let reference_start = Instant::now();
        render_reference(
            &tracer,
            &frame.camera_inputs,
            &mut reference_rgba,
            &mut reference_t,
        );
        let reference_seconds = reference_start.elapsed().as_secs_f64();

        // --- compare + write the report ----------------------------------
        let report = compare(
            &gpu_rgba,
            &gpu_t,
            &reference_rgba,
            &reference_t,
            width,
            height,
            CompareConfig::default(),
        );

        let pass = PassSpec {
            name: world.name.clone(),
            path: world.path.clone(),
            camera: world.camera,
        };

        if let Err(error) = write_report(
            &out_dir,
            &pass,
            width,
            height,
            reference_seconds,
            &gpu_rgba,
            &reference_rgba,
            &report,
            label,
        ) {
            return Err(format!("failed to write report: {error}"));
        }

        // The shading half (ticket 07): the CPU path-tracer diff over the
        // same world state — N GPU frames (frame_seed 0..N-1) vs N
        // identical-seed CPU samples, per-pixel means with tolerance.
        let path = if self.opts.no_path_trace {
            None
        } else {
            Some(self.run_path_validation(world_data, voxel_data, world, frame, label)?)
        };

        Ok((
            FrameSummary {
                label: label.to_string(),
                pass: report.passes(),
                mismatches: report.mismatch_count(),
                hard_mismatches: report.hard_mismatch_count(),
            },
            path,
        ))
    }

    /// The shading-half diff (ticket 07): runs the production ray pass at
    /// frame_seed 0..N-1, captures the radiance pair + albedo aux per frame,
    /// accumulates the GPU's per-pixel means; computes the CPU mirror's
    /// identical-seed N samples; compares the means with tolerance; writes
    /// the path artifacts (PNGs + text). Returns the frame's path verdict.
    fn run_path_validation(
        &mut self,
        world_data: &Arc<World>,
        voxel_data: &dot_vox::DotVoxData,
        world: &WorldSpec,
        frame: &mut ValidateFrame,
        label: &str,
    ) -> Result<PathPassSummary, String> {
        let width = self.opts.width;
        let height = self.opts.height;
        let samples = self.opts.path_trace_samples;
        let out_dir = self.opts.out_dir.join(&world.name);
        let gpu = &self.gpu;

        // The CPU mirror (ticket 07): the same world, the same Material
        // table (ADR 0008), and the same Scene constants the GPU reads
        // (default_scene — the packed values are data, mirrored verbatim).
        let materials = get_material_table(voxel_data);
        let packed = default_scene();
        let scene = path_tracer::Scene {
            sun_dir: glam::Vec3::new(packed.sun_dir[0], packed.sun_dir[1], packed.sun_dir[2]),
            sky_knots: [
                packed.sky_knots[0],
                packed.sky_knots[1],
                packed.sky_knots[2],
            ],
            e_sun: packed.sun_disk[0],
            cos_disk: packed.sun_disk[1],
            l_disk: packed.sun_disk[2],
        };
        let tracer = PathTracer::new(world_data, materials, scene);

        // --- GPU side: N frames at frame_seed 0..N-1, capture the radiance
        // pair + albedo aux, accumulate the per-pixel means.
        let pixel_count = (width as usize) * (height as usize);
        let mut diff_sum = vec![glam::Vec3::ZERO; pixel_count];
        let mut spec_sum = vec![glam::Vec3::ZERO; pixel_count];
        let mut albedo_sum = vec![glam::Vec3::ZERO; pixel_count];
        let mut hit_sum = vec![0u32; pixel_count];
        // The octahedral normal encodings (RGBA8) and the in-lobe hit
        // distance (the diffuse alpha) — for the corner-touch excuse.
        let mut hitdist_sum = vec![0.0f32; pixel_count];

        for f in 0..samples {
            frame.rcx.frame_seed = f;

            gpu.resources
                .flight(gpu.graphics_flight_id)
                .wait_idle()
                .unwrap();

            let resource_map = resource_map!(
                &frame.path_task_graph,
                frame.path_virtual_swapchain_id => frame.swapchain_id
            )
            .unwrap();
            let execute_result = unsafe {
                frame
                    .path_task_graph
                    .execute(resource_map, &frame.rcx, || {})
            };
            if let Err(error) = execute_result {
                return Err(format!("path frame execution failed: {error:?}"));
            }

            gpu.resources
                .flight(gpu.graphics_flight_id)
                .wait_idle()
                .unwrap();

            let diff_bytes = read_host_bytes(gpu, frame.path_diff_readback_buffer_id);
            let spec_bytes = read_host_bytes(gpu, frame.path_spec_readback_buffer_id);
            let albedo_bytes = read_host_bytes(gpu, frame.path_albedo_readback_buffer_id);

            let diff_rgba = decode_rgba16f(&diff_bytes);
            let spec_rgba = decode_rgba16f(&spec_bytes);
            for i in 0..pixel_count {
                diff_sum[i] += diff_rgba[i].truncate();
                spec_sum[i] += spec_rgba[i].truncate();
                let a = &albedo_bytes[i * 4..i * 4 + 4];
                albedo_sum[i] += glam::Vec3::new(
                    f32::from(a[0]) / 255.0,
                    f32::from(a[1]) / 255.0,
                    f32::from(a[2]) / 255.0,
                );
                // The in-lobe hit distance rides the diffuse alpha: 0 is
                // the sky sentinel (primary miss).
                hitdist_sum[i] += diff_rgba[i].w;
                hit_sum[i] += u32::from(diff_rgba[i].w > 0.0);
            }
        }

        let n = samples as f32;
        let gpu_diffuse: Vec<glam::Vec3> = diff_sum.iter().map(|v| *v / n).collect();
        let gpu_specular: Vec<glam::Vec3> = spec_sum.iter().map(|v| *v / n).collect();
        let gpu_albedo: Vec<glam::Vec3> = albedo_sum.iter().map(|v| *v / n).collect();
        let gpu_hit_fraction: Vec<f32> = hit_sum.iter().map(|h| *h as f32 / n).collect();
        let gpu_hitdist: Vec<f32> = hitdist_sum.iter().map(|h| *h / n).collect();
        // The display radiance (re-modulated like the composite): the sky
        // pixels stay raw (alpha 0 → no re-modulation).
        let gpu_display: Vec<glam::Vec3> = (0..pixel_count)
            .map(|i| {
                let diff_de = gpu_diffuse[i];
                let spec = gpu_specular[i];
                if gpu_hit_fraction[i] > 0.0 {
                    diff_de * gpu_albedo[i].max(glam::Vec3::splat(ALBEDO_EPS)) + spec
                } else {
                    diff_de
                }
            })
            .collect();

        // --- CPU side: the identical-seed mirror, parallel over rows.
        let cpu_start = Instant::now();
        let cpu: PathRender = render_path(&tracer, &frame.camera_inputs, width, height, samples);
        let cpu_seconds = cpu_start.elapsed().as_secs_f64();

        // --- compare + write the report ----------------------------------
        let path_report: PathCompareReport = compare_path(
            &gpu_diffuse,
            &gpu_specular,
            &gpu_hit_fraction,
            &gpu_display,
            &gpu_hitdist,
            &gpu_albedo,
            &cpu.diffuse,
            &cpu.specular,
            &cpu.display,
            &cpu.hit_ts,
            &cpu.albedos,
            width,
            height,
            samples,
            PathCompareConfig::default(),
        );

        fs::create_dir_all(&out_dir)
            .map_err(|e| format!("failed to create path report dir: {e}"))?;
        write_png(
            &out_dir.join(format!("path-gpu{label}.png")),
            &tone_map_rgba(&gpu_display),
            width,
            height,
        )
        .map_err(|e| format!("failed to write path-gpu.png: {e}"))?;
        write_png(
            &out_dir.join(format!("path-cpu{label}.png")),
            &tone_map_rgba(&cpu.display),
            width,
            height,
        )
        .map_err(|e| format!("failed to write path-cpu.png: {e}"))?;
        write_png(
            &out_dir.join(format!("path-diff{label}.png")),
            &build_path_diff_image(&cpu.display, &path_report),
            width,
            height,
        )
        .map_err(|e| format!("failed to write path-diff.png: {e}"))?;
        write_path_report(
            &out_dir.join(format!("path-report{label}.txt")),
            &world.name,
            &world.path,
            &world
                .camera
                .map(camera_description)
                .unwrap_or_else(|| "framed world bounding box".to_string()),
            width,
            height,
            cpu_seconds,
            &path_report,
        )
        .map_err(|e| format!("failed to write path report: {e}"))?;

        Ok(PathPassSummary {
            name: world.name.clone(),
            pass: path_report.passes(),
            mismatches: path_report.mismatch_count(),
            hard_mismatches: path_report.hard_mismatch_count(),
            samples,
        })
    }
}

/// The de-modulation guard (the composite's re-modulation divisor) — the
/// display PNG re-modulates with the same max(albedo, eps) the trace pass
/// divided by (ADR 0010).
const ALBEDO_EPS: f32 = 1e-3;

/// Writes the report artifacts (PNGs + text) for one frame. `label` suffixes
/// the artifact names ("" for the first/only frame, "-after-edit" for the
/// edit-at-the-seam second frame).
#[allow(clippy::too_many_arguments)]
fn write_report(
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

fn camera_description(camera: Camera) -> String {
    format!(
        "eye {:?} target {:?} up {:?}",
        camera.eye, camera.target, camera.up
    )
}

/// Reads a host-visible buffer as raw bytes.
fn read_host_bytes(gpu: &GpuStack, id: Id<Buffer>) -> Vec<u8> {
    let buffer = gpu.resources.buffer(id).buffer().clone();
    let subbuffer = Subbuffer::new(buffer).cast_aligned::<u8>();
    subbuffer
        .read()
        .expect("host read of capture buffer")
        .to_vec()
}

/// Reads a host-visible buffer as f32s.
fn read_host_floats(gpu: &GpuStack, id: Id<Buffer>) -> Vec<f32> {
    let buffer = gpu.resources.buffer(id).buffer().clone();
    let subbuffer = Subbuffer::new(buffer).cast_aligned::<f32>();
    subbuffer
        .read()
        .expect("host read of capture buffer")
        .to_vec()
}

// ---------------------------------------------------------------------------
// The shading-half resources (ticket 07)
// ---------------------------------------------------------------------------

/// One of the shading-half's six output images: the physical image + its
/// bindless storage id. The validator never resizes, so the app's
/// virtual-id indirection is unnecessary — the graph references the physical
/// ids directly.
struct PathTraceImage {
    physical_id: Id<Image>,
    storage_id: StorageImageId,
}

struct PathTraceImages {
    diff: PathTraceImage,
    spec: PathTraceImage,
    normal_roughness: PathTraceImage,
    viewz: PathTraceImage,
    mv: PathTraceImage,
    albedo: PathTraceImage,
}

/// Creates the production ray pass's six output images (ADR 0007), sized to
/// the frame, registered in the bindless set. All carry TRANSFER_SRC (the
/// radiance pair + albedo are copied to host-readable buffers per frame).
fn create_path_trace_images(
    resources: &Arc<Resources>,
    width: u32,
    height: u32,
) -> PathTraceImages {
    let bcx = resources.bindless_context().unwrap();
    let create = |format: Format| {
        let physical_id = resources
            .create_image(
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [width, height, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            )
            .unwrap();
        let image = resources.image(physical_id).image().clone();
        let image_view = ImageView::new_default(&image).unwrap();
        let storage_id = bcx
            .global_set()
            .add_storage_image(image_view, ImageLayout::General);
        PathTraceImage {
            physical_id,
            storage_id,
        }
    };
    PathTraceImages {
        diff: create(Format::R16G16B16A16_SFLOAT),
        spec: create(Format::R16G16B16A16_SFLOAT),
        normal_roughness: create(Format::R8G8B8A8_UNORM),
        viewz: create(Format::R32_SFLOAT),
        mv: create(Format::R16G16B16A16_SFLOAT),
        albedo: create(Format::R8G8B8A8_UNORM),
    }
}

/// Creates a host-readable TRANSFER_DST buffer of `bytes` (the capture
/// readbacks: the copy target the capture node writes, read back per frame).
fn create_host_readback(gpu: &GpuStack, bytes: u64) -> Id<Buffer> {
    gpu.resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(bytes).unwrap(),
        )
        .unwrap()
}

/// Decodes RGBA16F (R16G16B16A16_SFLOAT) bytes — 8 per pixel — to f32 RGBA.
fn decode_rgba16f(bytes: &[u8]) -> Vec<glam::Vec4> {
    bytes
        .chunks_exact(8)
        .map(|px| {
            glam::Vec4::new(
                half_to_f32(u16::from_le_bytes([px[0], px[1]])),
                half_to_f32(u16::from_le_bytes([px[2], px[3]])),
                half_to_f32(u16::from_le_bytes([px[4], px[5]])),
                half_to_f32(u16::from_le_bytes([px[6], px[7]])),
            )
        })
        .collect()
}

/// IEEE-754 binary16 → f32 (the radiance pair's storage format; the CPU
/// mirror computes f32 and the tolerance absorbs the quantization).
fn half_to_f32(h: u16) -> f32 {
    let negative = h & 0x8000 != 0;
    let exp = (h >> 10) & 0x1F;
    let mant = u32::from(h & 0x3FF);
    let magnitude = if exp == 0 {
        (mant as f32) * 2.0_f32.powi(-24)
    } else if exp == 31 {
        // Inf/NaN — the radiance pair never contains them; map to +inf.
        f32::INFINITY
    } else {
        ((mant as f32) + 1024.0) * 2.0_f32.powi(i32::from(exp) - 25)
    };
    if negative { -magnitude } else { magnitude }
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

fn create_validate_swapchain(
    gpu: &GpuStack,
    surface: &Arc<Surface>,
    extent: [u32; 2],
) -> Result<(Id<Swapchain>, Format), vulkano::VulkanError> {
    let surface_capabilities = gpu
        .device
        .physical_device()
        .surface_capabilities(surface, &Default::default())?;
    let (image_format, image_color_space) = gpu
        .device
        .physical_device()
        .surface_formats(surface, &Default::default())?
        .into_iter()
        // Only UNORM formats: an sRGB swapchain would sRGB-encode the ray
        // pass's linear writes while the reference quantizes raw palette
        // bytes, causing systematic color mismatches.
        .filter(|(format, _)| {
            matches!(
                format,
                Format::R8G8B8A8_UNORM
                    | Format::B8G8R8A8_UNORM
                    | Format::R8G8B8A8_SNORM
                    | Format::B8G8R8A8_SNORM
            )
        })
        .find(|(format, _)| {
            gpu.device
                .physical_device()
                .image_format_properties(&vulkano::image::ImageFormatInfo {
                    format: *format,
                    usage: ImageUsage::STORAGE
                        | ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                })
                .unwrap()
                .is_some()
        })
        .ok_or(vulkano::VulkanError::Unknown)?;

    let swapchain_id = gpu.resources.create_swapchain(
        surface,
        &SwapchainCreateInfo {
            present_mode: PresentMode::Immediate,
            min_image_count: surface_capabilities
                .min_image_count
                .max(MIN_SWAPCHAIN_IMAGES),
            image_format,
            image_extent: extent,
            image_usage: ImageUsage::STORAGE
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::TRANSFER_SRC,
            image_color_space,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )?;

    Ok((swapchain_id, image_format))
}

/// Creates the rgba32f t-channel image and registers it in the bindless set.
fn create_t_image(
    resources: &Arc<Resources>,
    width: u32,
    height: u32,
) -> Result<(Id<Image>, StorageImageId, Format), String> {
    let format = Format::R32G32B32A32_SFLOAT;
    let image_id = resources
        .create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [width, height, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .map_err(|e| format!("{e}"))?;

    let image = resources.image(image_id).image().clone();
    let image_view = ImageView::new_default(&image).unwrap();

    let storage_id = resources
        .bindless_context()
        .unwrap()
        .global_set()
        .add_storage_image(image_view, ImageLayout::General);

    Ok((image_id, storage_id, format))
}

/// Decodes the captured bytes (in the swapchain's channel order) to RGBA8.
fn decode_rgba(format: Format, bytes: &[u8]) -> Vec<u8> {
    match format {
        Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SNORM => bytes.to_vec(),
        Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SNORM => bytes
            .chunks_exact(4)
            .flat_map(|p| [p[2], p[1], p[0], p[3]])
            .collect(),
        other => panic!("unsupported swapchain format for capture: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Builds the view/proj matrices and the shared camera inputs for the world.
fn build_camera(
    world: &World,
    width: u32,
    height: u32,
    camera: Option<Camera>,
) -> (crate::region::render::capture_raygen::Camera, CameraInputs) {
    let (eye, target, up) = match camera {
        Some(spec) => (
            Vec3::from(spec.eye),
            Vec3::from(spec.target),
            Vec3::from(spec.up),
        ),
        None => frame_world_camera(world),
    };

    let view = glam::camera::lh::view::look_to_mat4(eye, (target - eye).normalize(), up);
    let proj = glam::camera::lh::proj::vulkan::perspective(
        PI / 2.0,
        width as f32 / height as f32,
        0.01,
        10000.0,
    );

    let gpu_camera = crate::region::render::capture_raygen::Camera {
        proj_inverse: proj.inverse().to_cols_array_2d(),
        view_inverse: view.inverse().to_cols_array_2d(),
    };

    let inputs = CameraInputs::new(view, proj, width, height);

    (gpu_camera, inputs)
}

/// Places a camera that frames the world's occupied bounding box.
fn frame_world_camera(world: &World) -> (Vec3, Vec3, Vec3) {
    let (min, max) = world.voxel_bounds().unwrap_or((IVec3::ZERO, IVec3::ZERO));
    let center = (min.as_vec3() + max.as_vec3()) * 0.5;
    let extent = (max - min).max_element().max(1) as f32;
    let distance = extent * 1.8 + 4.0;
    let direction = Vec3::new(1.0, 0.65, 0.85).normalize();

    (center + direction * distance, center, Vec3::Y)
}

#[cfg(test)]
mod path_readback_tests {
    use super::*;

    #[test]
    fn half_to_f32_decodes_known_values() {
        assert_eq!(half_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(half_to_f32(0x4b77), 1911.0 * 2.0f32.powi(-7)); // 14.9296875
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3e00), 1.5); // 1.5
        assert_eq!(half_to_f32(0x3800), 0.5); // 0.5
    }

    #[test]
    fn decode_rgba16f_reads_pixel_alignment() {
        // One pixel: R=1.0 (0x3c00), G=1.5 (0x3e00), B=0.25 (0x3400), A=14.93 (0x4b77)
        let bytes = [0x00, 0x3c, 0x00, 0x3e, 0x00, 0x34, 0x77, 0x4b];
        let decoded = decode_rgba16f(&bytes);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].x, 1.0);
        assert_eq!(decoded[0].y, 1.5);
        assert_eq!(decoded[0].z, 0.25);
        assert_eq!(decoded[0].w, 1911.0 * 2.0f32.powi(-7));
    }
}
