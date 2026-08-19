//! The validator's driver: one hidden window + swapchain per world, the
//! captured frames (geometry and shading halves), the CPU mirrors, and the
//! report artifacts.

use std::{f32::consts::PI, fs, sync::Arc, time::Instant};

use glam::{IVec3, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    format::Format,
    image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
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
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};

use crate::{
    core::gpu::GpuStack,
    core::grid::{MICRO_CHUNK_EDGE, grid_origin},
    render::{
        region::{
            feed::RendererInput,
            pack::pack_regions,
            rebuild::RebuildLogEntry,
            residency::RegionStore,
            task::{
                RegionRenderContext, RegionRenderTask, RenderMode, default_scene,
            },
        },
        swapchain::window_size_dependent_setup,
        validate::{
            capture::{CaptureTask, PathCaptureTask},
            cli::ValidateOptions,
            compare::{CompareConfig, compare},
            path_compare::{PathCompareConfig, PathCompareReport, compare_path},
            path_tracer,
            path_tracer::{PathRender, PathTracer, render_path},
            readback::{
                create_host_readback, create_t_image, create_validate_swapchain, decode_rgba,
                decode_rgba16f, read_host_bytes, read_host_floats,
            },
            reference::{CameraInputs, ReferenceTracer, VoxelShape, render_reference},
            report::{
                ALBEDO_EPS, build_path_diff_image, camera_description, tone_map_rgba,
                write_path_report, write_png, write_report,
            },
            test_worlds::{Camera, WorldSpec},
            FrameSummary, PassSummary, PathPassSummary,
        },
    },
    world::{
        format::{get_palette, open_file},
        material::get_material_table,
        snapshot::{MicroChunkSnapshot, emit_snapshots},
        World,
    },
};

/// Static per-pass data (the world's identity + camera).
pub struct PassSpec {
    pub name: String,
    pub path: String,
    pub camera: Option<Camera>,
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
    camera: crate::render::region::task::capture_raygen::Camera,
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

pub struct ValidateRunner {
    pub gpu: GpuStack,
    pub opts: ValidateOptions,
    pub worlds: Vec<WorldSpec>,
    pub results: Vec<Result<PassSummary, String>>,
    /// The shading-half diff's outcome per world (parallel to `results`;
    /// `None` when the run was geometry-only).
    pub path_results: Vec<Option<PathPassSummary>>,
    pub done: bool,
}

impl ApplicationHandler for ValidateRunner {
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

impl ValidateRunner {
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
            window_size_dependent_setup(&self.gpu.resources, swapchain_id);

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
                // The rebuild log line (step observability): what the cycle's
                // nodes did, in node order.
                if !report.rebuild_log.is_empty() {
                    println!("              rebuild {}: {:?}", step.label, report.rebuild_log);
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
            crate::render::region::task::capture_raygen::load(&self.gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };
        let rt_task = RegionRenderTask::new(
            &self.gpu,
            store,
            virtual_swapchain_id,
            &raygen,
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
            crate::render::region::task::production_raygen::load(&self.gpu.device)
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

fn build_camera(
    world: &World,
    width: u32,
    height: u32,
    camera: Option<Camera>,
) -> (crate::render::region::task::capture_raygen::Camera, CameraInputs) {
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

    let gpu_camera = crate::render::region::task::capture_raygen::Camera {
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
