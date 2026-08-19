mod input;
mod player;
mod schedule;
mod stats;

use std::{f32::consts::PI, sync::Arc, time::Duration};
use vulkano::{
    VulkanError,
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageFormatInfo, ImageLayout, ImageType, ImageUsage,
        view::ImageView,
    },
    memory::allocator::AllocationCreateInfo,
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};

#[cfg(debug_assertions)]
use vulkano::buffer::{BufferCreateInfo, BufferUsage};
#[cfg(debug_assertions)]
use vulkano::memory::allocator::{DeviceLayout, MemoryTypeFilter};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::StorageImageId,
    graph::{CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, ImageLayoutType, Resources},
    resource_map,
};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

use crate::{
    app::{
        input::{Input, InputButton, InputKey},
        player::PlayerController,
        schedule::ScheduleController,
        stats::Measurement,
    },
    core::gpu::{GpuStack, MIN_SWAPCHAIN_IMAGES},
    core::grid::LATTICE_HALF_EXTENT,
    render::{
        composite::{CompositeTask, create_composite_pipeline},
        region::{
            feed::RendererInput,
            residency::RegionStore,
            task::{
                HullCrossedCounter, RegionRenderContext, RegionRenderTask, RenderMode,
                capture_raygen, default_scene, production_raygen,
            },
        },
        swapchain::window_size_dependent_setup,
    },
    world::{format::open_file, snapshot::emit_snapshots, World},
};

#[cfg(debug_assertions)]
use crate::render::debug::{DrawHeatmapTask, create_heatmap_pipeline};

/// The per-pixel hull-crossed count buffer's fixed pixel ceiling (4K): debug
/// builds only. Resizing the window beyond it is unsupported in the heatmap.
#[cfg(debug_assertions)]
const HEATMAP_MAX_PIXELS: u64 = 3840 * 2160;

pub struct App {
    close_requested: bool,

    pub gpu: GpuStack,

    delta_time: Duration,
    focused: bool,
    /// The path-tracing RNG's per-frame seed (ticket 05, ADR 0010):
    /// incremented every rendered frame and written into the render
    /// context's `frame_seed`, so consecutive frames decorrelate for the
    /// Denoise pass's temporal accumulation.
    frame_seed: u32,

    pub voxel_data: dot_vox::DotVoxData,
    pub world: Arc<World>,

    player_controller: PlayerController,
    player_input: Input,
    schedule_controller: ScheduleController,

    /// The renderer input contract: the world voices its
    /// initial state as one `submit_batch`; the worker drains it into
    /// per-Region mirrors. Kept so future world edits flow through the
    /// same contract (the minimal snapshot emitter stays as the world
    /// side's seed).
    input: RendererInput,
    /// The full static lattice's GPU half: residency,
    /// free lists, the stable TLAS. Built once from the initial batch (the
    /// one-shot pre-loop build) and rebuilt through the ordered rebuild
    /// nodes on change cycles.
    store: RegionStore,

    /// GPU measurement: per-stage timestamps
    /// (trace_rays / AS rebuild / flight), min/avg/p95 in the FPS log, the
    /// 16 ms gate as the GPU timestamp sum with wall-clock beside it. Only
    /// created with `atlas-rt --measure` (on demand); the validator never
    /// constructs one, so the harness's captured frames are unaffected.
    measurement: Option<Measurement>,

    rcx: Option<RenderContext>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    virtual_swapchain_id: Id<Swapchain>,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<RegionRenderContext>,
    /// The trace pass's output images (ADR 0007): virtual graph resources
    /// plus the physical images recreated on resize.
    trace_pass_images: TracePassImages,
    /// The per-frame world the graph executes with (camera, storage
    /// images, the app's debug overlay fields — see
    /// [`RegionRenderContext`]).
    region: RegionRenderContext,
}

impl App {
    pub fn new(
        event_loop: &EventLoop<()>,
        measure: bool,
        world_path: &str,
        clip_oob: bool,
    ) -> Self {
        let gpu = GpuStack::new(event_loop);

        let voxel_data = open_file(world_path);
        let (world, clipped) = if clip_oob {
            World::new_clipped(&voxel_data)
        } else {
            (World::new(&voxel_data), 0)
        };
        if clipped > 0 {
            println!(
                "clipped {clipped} voxels outside the ±{} lattice",
                LATTICE_HALF_EXTENT
            );
        }
        let world = Arc::new(world);

        // The input contract: the world voices its initial
        // state as one `submit_batch`; the worker drains it into per-Region
        // mirrors. The minimal snapshot emitter stays as the world side's
        // seed for feeding the renderer.
        let input = RendererInput::new();
        input.submit_batch(emit_snapshots(&world));
        input.wait_until_idle();

        // The one-shot pre-loop build: every initial Region
        // becomes resident through the ordered rebuild graph (pool upload →
        // BLAS build → TLAS build). The startup batch's published dirty set
        // is consumed here — the frame loop applies only post-startup
        // change cycles.
        let store = RegionStore::new(&gpu, &voxel_data, input.packed_regions());
        input.take_dirty_regions();

        // Measurement: on demand only — the pool is attached
        // to the render task in `resumed`, and the frame loop feeds it the
        // rebuild timings and per-frame readbacks.
        let measurement = measure.then(|| Measurement::new(&gpu));

        let mut schedule_controller = ScheduleController::new();
        schedule_controller.add_schedule_frames("delta", 1);
        schedule_controller.add_schedule_duration("log", Duration::from_secs(1));

        App {
            close_requested: false,

            gpu,

            delta_time: Duration::ZERO,
            focused: false,
            frame_seed: 0,

            player_controller: PlayerController::default(),
            player_input: Input::default(),
            schedule_controller,

            voxel_data,
            world,

            input,
            store,

            measurement,

            rcx: None,
        }
    }

    pub fn toggle_capture_mouse(&mut self) {
        let window = &self.rcx.as_mut().unwrap().window;

        if self.focused {
            self.focused = false;
            window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            window.set_cursor_visible(true);
        } else {
            self.focused = true;
            window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .unwrap();
            window.set_cursor_visible(false);
        }
    }

    /// TAB (debug builds) cycles the Render mode Voxel -> Hull -> Ray
    /// latency -> hull-crossed -> normal heatmap. Reads the just-pressed
    /// edge from the shared input layer; the mode lives in the render
    /// context and is written into the push constants every frame, so
    /// toggling is a per-frame flag, never a pipeline rebuild.
    #[cfg(debug_assertions)]
    fn toggle_render_mode(&mut self) {
        if self
            .player_input
            .just_pressed
            .contains(&InputKey::ToggleRenderMode)
        {
            let rcx = self.rcx.as_mut().unwrap();
            rcx.region.mode = match rcx.region.mode {
                RenderMode::Voxel => RenderMode::Hull,
                RenderMode::Hull => RenderMode::RayLatency,
                RenderMode::RayLatency => RenderMode::HullCrossed,
                RenderMode::HullCrossed => RenderMode::Normal,
                RenderMode::Normal => RenderMode::Voxel,
            };
        }
    }

    fn update_delta_time(&mut self) {
        self.delta_time = self
            .schedule_controller
            .check("delta")
            .expect("Delta time calculation returned None!");
    }

    fn request_log(&mut self) {
        if self.schedule_controller.check("log").is_some() {
            println!("{:.2} fps", 1.0 / self.delta_time.as_secs_f32());
            // Measurement surface: min/avg/p95 per stage over
            // the ~60-frame window, the 16 ms gate as the GPU timestamp
            // sum (trace + as rebuild) and the wall-clock beside it.
            if let Some(measurement) = &self.measurement {
                measurement.print_log();
            }
        }
    }

    fn update_camera(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();

        // Look: this frame's mouse-motion delta (only while cursor-captured).
        if self.focused {
            self.player_controller
                .rotate(self.player_input.mouse_motion);
        }
        self.player_controller
            .fly_movement(self.delta_time, &self.player_input);
        let view = self.player_controller.view();

        let size = rcx.window.inner_size();

        let proj = glam::camera::lh::proj::vulkan::perspective(
            PI / 2.0,
            (size.width as f32) / (size.height as f32),
            0.01,
            10000.0,
        );

        rcx.region.camera = capture_raygen::Camera {
            proj_inverse: proj.inverse().to_cols_array_2d(),
            view_inverse: view.inverse().to_cols_array_2d(),
        };
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes =
            WindowAttributes::default().with_inner_size(PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let window_size = window.inner_size();
        let surface = Surface::from_window(&self.gpu.instance, &window).unwrap();

        let swapchain = {
            let surface_capabilities = self
                .gpu
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();
            let (image_format, image_color_space) = self
                .gpu
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()
                .into_iter()
                .find(|(format, _)| {
                    self.gpu
                        .device
                        .physical_device()
                        .image_format_properties(&ImageFormatInfo {
                            format: *format,
                            usage: ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT,
                            ..Default::default()
                        })
                        .unwrap()
                        .is_some()
                })
                .unwrap();

            let present_mode = PresentMode::Immediate;

            (
                self.gpu
                    .resources
                    .create_swapchain(
                        &surface,
                        &SwapchainCreateInfo {
                            present_mode,
                            min_image_count: surface_capabilities
                                .min_image_count
                                .max(MIN_SWAPCHAIN_IMAGES),
                            image_format,
                            image_extent: window_size.into(),
                            image_usage: ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT,
                            image_color_space,
                            composite_alpha: surface_capabilities
                                .supported_composite_alpha
                                .into_iter()
                                .next()
                                .unwrap(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                image_format,
            )
        };

        let swapchain_id = swapchain.0;

        let swapchain_storage_image_ids =
            window_size_dependent_setup(&self.gpu.resources, swapchain_id);

        let mut task_graph = TaskGraph::new(&self.gpu.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        // The trace pass's output images (ADR 0007): virtual graph resources
        // first (the nodes' accesses reference them below), physical images
        // + bindless ids attached right after (sized to the window and
        // recreated on resize — the per-frame resource map binds the
        // virtual ids, so recreation needs no graph rebuild).
        let mut trace_pass_images = TracePassImages::add_virtual(&mut task_graph);
        let swapchain_state = self.gpu.resources.swapchain(swapchain_id);
        let extent = swapchain_state.images()[0].extent();
        let physical_trace_pass_images =
            create_trace_pass_images(&self.gpu.resources, extent[0], extent[1]);
        trace_pass_images.attach_physical(physical_trace_pass_images);

        // The Region render task: ray-passes
        // the store's stable TLAS with the production raygen (color only;
        // `t_image_id` stays INVALID and is never dereferenced).
        let raygen = unsafe {
            production_raygen::load(&self.gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

        // The per-pixel hull-crossed count buffer (debug builds): the heatmap
        // overlay reads it and the render task resets it each frame. Fixed-size
        // for the 4K window ceiling. Release and the validator pass None.
        #[cfg(debug_assertions)]
        let hull_crossed = {
            let buffer_id = self
                .gpu
                .resources
                .create_buffer(
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    DeviceLayout::new_unsized::<[u32]>(HEATMAP_MAX_PIXELS).unwrap(),
                )
                .unwrap();
            let storage_id = self
                .gpu
                .resources
                .bindless_context()
                .unwrap()
                .global_set()
                .create_storage_buffer(buffer_id, 0, Some(HEATMAP_MAX_PIXELS * 4))
                .unwrap();
            Some(HullCrossedCounter { buffer_id, storage_id })
        };
        #[cfg(not(debug_assertions))]
        let hull_crossed: Option<HullCrossedCounter> = None;

        let rt_pass = RegionRenderTask::new(
            &self.gpu,
            &self.store,
            virtual_swapchain_id,
            &raygen,
            // The measurement pool: attached only with
            // `--measure`; `None` records no timestamps.
            self.measurement.as_ref().and_then(Measurement::pool),
            // The march-and-miss counter: same on-demand gate; `None` (the
            // default app) pushes INVALID and specializes COUNTER_ENABLED off.
            self.measurement.as_ref().and_then(Measurement::counter),
            hull_crossed.as_ref(),
            // The production pipeline's miss shader returns the Procedural
            // sky (ticket 06).
            true,
        );
        let instance_buffer_id = rt_pass.instance_buffer_id();

        // The render node id is needed only by the heatmap edge below.
        let mut rt_node = task_graph.create_task_node("Render", QueueFamilyType::Graphics, rt_pass);
        rt_node.image_access(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        rt_node.buffer_access(
            instance_buffer_id,
            AccessTypes::RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
        );
        // The count buffer is written by the intersection shader's atomicAdd in
        // hull-crossed mode (debug only); declaring it here makes the
        // rt -> heatmap edge insert the memory barrier that makes those
        // atomics visible to the heatmap's read.
        #[cfg(debug_assertions)]
        {
            rt_node.buffer_access(
                hull_crossed.as_ref().unwrap().buffer_id,
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            );
        }
        // The trace pass's output images (ADR 0007): written by the
        // production raygen in Voxel mode, consumed by the composite (and,
        // from ticket 08, the Denoise pass). Accessed through the virtual
        // ids, which the per-frame resource map binds to the physical images.
        rt_node.image_access(
            trace_pass_images.diff_radiance.virtual_id,
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        rt_node.image_access(
            trace_pass_images.spec_radiance.virtual_id,
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        rt_node.image_access(
            trace_pass_images.normal_roughness.virtual_id,
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        rt_node.image_access(
            trace_pass_images.viewz.virtual_id,
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        rt_node.image_access(
            trace_pass_images.mv.virtual_id,
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        rt_node.image_access(
            trace_pass_images.albedo_metal.virtual_id,
            AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        #[cfg_attr(not(debug_assertions), allow(unused_variables))]
        let rt_node_id = rt_node.build();

        // The hull-crossed heatmap overlay node (debug builds): a compute pass
        // that reads the per-pixel count buffer and repaints the swapchain
        // image when the mode is hull-crossed. Ordered after the ray pass.
        #[cfg(debug_assertions)]
        let heatmap_node_id = task_graph
            .create_task_node(
                "Heatmap",
                QueueFamilyType::Graphics,
                DrawHeatmapTask::new(
                    virtual_swapchain_id,
                    hull_crossed.as_ref().unwrap().storage_id,
                ),
            )
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .buffer_access(
                hull_crossed.as_ref().unwrap().buffer_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ,
            )
            .build();

        #[cfg(debug_assertions)]
        task_graph.add_edge(rt_node_id, heatmap_node_id).unwrap();

        // The composite node (ADR 0007): exposes the trace pass's radiance to
        // the swapchain (manual EV + ACES tonemap) in Voxel mode; a no-op for
        // the debug modes, which paint the swapchain directly from the raygen.
        let composite_node_id = task_graph
            .create_task_node(
                "Composite",
                QueueFamilyType::Graphics,
                CompositeTask::new(virtual_swapchain_id),
            )
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .image_access(
                trace_pass_images.diff_radiance.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ,
                ImageLayoutType::General,
            )
            .build();

        // The composite follows the heatmap overlay (debug builds) or the
        // render node directly (release); either way it is the last node
        // before present.
        #[cfg(debug_assertions)]
        task_graph.add_edge(heatmap_node_id, composite_node_id).unwrap();
        #[cfg(not(debug_assertions))]
        task_graph.add_edge(rt_node_id, composite_node_id).unwrap();

        let task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.gpu.graphics_queue],
                present_queue: Some(&self.gpu.graphics_queue),
                flight_id: self.gpu.graphics_flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        // Pipeline injection needs mutable access to the compiled graph.
        let mut task_graph = task_graph;

        // The heatmap compute pipeline is injected into the compiled graph
        // only in debug builds (the overlay is app-only).
        #[cfg(debug_assertions)]
        {
            let heatmap_pipeline = create_heatmap_pipeline(&self.gpu);
            task_graph
                .task_node_mut(heatmap_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DrawHeatmapTask>()
                .unwrap()
                .pipeline = Some(heatmap_pipeline);
        }

        // The composite compute pipeline has no subpass, so it is created
        // from the app only (no task-node reference), like the heatmap's.
        {
            let composite_pipeline = create_composite_pipeline(&self.gpu);
            task_graph
                .task_node_mut(composite_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<CompositeTask>()
                .unwrap()
                .pipeline = Some(composite_pipeline);
        }

        let region = RegionRenderContext {
            camera: capture_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
            },
            // The analytic lights' constants (ticket 06): the defaults —
            // tunable later (the Scene buffer is written every frame).
            scene: default_scene(),
            swapchain_storage_image_ids,
            // The production raygen never dereferences `t_image_id`
            // (shaders/region/production.rgen) — it stays INVALID.
            t_image_storage_id: StorageImageId::INVALID,
            // The trace pass's output set (ADR 0007): the storage ids the
            // production raygen writes in Voxel mode.
            diff_radiance_image_id: trace_pass_images.diff_radiance.storage_id,
            spec_radiance_image_id: trace_pass_images.spec_radiance.storage_id,
            normal_roughness_image_id: trace_pass_images.normal_roughness.storage_id,
            viewz_image_id: trace_pass_images.viewz.storage_id,
            mv_image_id: trace_pass_images.mv.storage_id,
            albedo_metal_image_id: trace_pass_images.albedo_metal.storage_id,
            // Manual exposure: 0 EV (radiance unchanged) until adjusted with
            // [ / ] (ADR 0007).
            ev: 0.0,
            // Voxel is the default; TAB (debug builds) toggles this in the
            // render context before each frame.
            mode: RenderMode::default(),
            // The path-tracing RNG seed starts at 0; the frame loop
            // increments it before each frame (ticket 05).
            frame_seed: 0,
        };

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            trace_pass_images,
            region,
        });
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.close_requested = true;
            }
            WindowEvent::Resized(_) => {
                self.rcx.as_mut().unwrap().recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                self.update_delta_time();
                self.update_camera();
                // The path-tracing RNG's per-frame seed (ticket 05):
                // increment before this frame renders so consecutive frames
                // decorrelate for the Denoise pass's temporal accumulation.
                self.frame_seed = self.frame_seed.wrapping_add(1);
                self.rcx.as_mut().unwrap().region.frame_seed = self.frame_seed;
                self.request_log();

                // The change cycle: anything the world voiced since the last
                // frame is applied through the ordered rebuild nodes (pool
                // upload → BLAS build → TLAS build on residency transitions)
                // before the consuming trace.
                let report = if !self.input.take_dirty_regions().is_empty() {
                    Some(self.store.apply(&self.gpu, &self.input))
                } else {
                    None
                };
                // The cycle's per-node rebuild time is attributed
                // to the frame being assembled (a rebuild spike shows up in
                // the AS-rebuild line, not in trace_rays).
                if let (Some(measurement), Some(report)) = (&mut self.measurement, &report) {
                    measurement.record_rebuild(&report.timings);
                }

                {
                    let rcx = self.rcx.as_mut().unwrap();

                    let window_size = rcx.window.inner_size();

                    if window_size.width == 0 || window_size.height == 0 {
                        return;
                    }

                    if rcx.recreate_swapchain {
                        rcx.swapchain_id = self
                            .gpu
                            .resources
                            .recreate_swapchain(rcx.swapchain_id, |create_info| {
                                SwapchainCreateInfo {
                                    image_extent: window_size.into(),
                                    ..create_info.clone()
                                }
                            })
                            .expect("failed to recreate swapchain");

                        let mut batch = self.gpu.resources.create_deferred_batch();

                        for &id in &rcx.region.swapchain_storage_image_ids {
                            batch.destroy_storage_image(id);
                        }

                        // The trace pass's output images are window-sized
                        // too: destroy the physical images and their bindless
                        // ids, then recreate them. The graph references the
                        // virtual ids (mapped per frame), so no graph rebuild
                        // is needed.
                        let t = &rcx.trace_pass_images;
                        for image in [
                            &t.diff_radiance,
                            &t.spec_radiance,
                            &t.normal_roughness,
                            &t.viewz,
                            &t.mv,
                            &t.albedo_metal,
                        ] {
                            batch.destroy_image(image.physical_id);
                            batch.destroy_storage_image(image.storage_id);
                        }

                        batch.enqueue();

                        rcx.region.swapchain_storage_image_ids =
                            window_size_dependent_setup(&self.gpu.resources, rcx.swapchain_id);

                        let swapchain_state = self.gpu.resources.swapchain(rcx.swapchain_id);
                        let extent = swapchain_state.images()[0].extent();
                        let physical = create_trace_pass_images(
                            &self.gpu.resources,
                            extent[0],
                            extent[1],
                        );
                        rcx.trace_pass_images.attach_physical(physical);
                        rcx.region.diff_radiance_image_id =
                            rcx.trace_pass_images.diff_radiance.storage_id;
                        rcx.region.spec_radiance_image_id =
                            rcx.trace_pass_images.spec_radiance.storage_id;
                        rcx.region.normal_roughness_image_id =
                            rcx.trace_pass_images.normal_roughness.storage_id;
                        rcx.region.viewz_image_id = rcx.trace_pass_images.viewz.storage_id;
                        rcx.region.mv_image_id = rcx.trace_pass_images.mv.storage_id;
                        rcx.region.albedo_metal_image_id =
                            rcx.trace_pass_images.albedo_metal.storage_id;

                        rcx.recreate_swapchain = false;
                    }
                }

                self.gpu
                    .resources
                    .flight(self.gpu.graphics_flight_id)
                    .wait_idle()
                    .unwrap();

                // complete the previous frame's sample — the
                // flight idle above makes its timestamps available; the wall
                // interval is this frame's `delta_time` (the previous
                // frame's interval, aligned with the readback).
                if let Some(measurement) = &mut self.measurement {
                    measurement.record_frame(&self.gpu, self.delta_time.as_nanos() as u64);
                }

                let rcx = self.rcx.as_mut().unwrap();

                let resource_map = resource_map!(
                    &rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                    rcx.trace_pass_images.diff_radiance.virtual_id =>
                        rcx.trace_pass_images.diff_radiance.physical_id,
                    rcx.trace_pass_images.spec_radiance.virtual_id =>
                        rcx.trace_pass_images.spec_radiance.physical_id,
                    rcx.trace_pass_images.normal_roughness.virtual_id =>
                        rcx.trace_pass_images.normal_roughness.physical_id,
                    rcx.trace_pass_images.viewz.virtual_id =>
                        rcx.trace_pass_images.viewz.physical_id,
                    rcx.trace_pass_images.mv.virtual_id =>
                        rcx.trace_pass_images.mv.physical_id,
                    rcx.trace_pass_images.albedo_metal.virtual_id =>
                        rcx.trace_pass_images.albedo_metal.physical_id,
                )
                .unwrap();

                let execute_result = unsafe {
                    rcx.task_graph.execute(resource_map, &rcx.region, || {
                        rcx.window.pre_present_notify()
                    })
                };

                match execute_result {
                    Ok(()) => {}
                    Err(ExecuteError::Swapchain {
                        error: VulkanError::OutOfDate,
                        ..
                    }) => {
                        rcx.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to execute next frame: {e:?}");
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(mapped) = input::map_button(button) {
                    match state {
                        ElementState::Pressed => {
                            // Cursor capture toggles on the right-button
                            // press edge; the held button is recorded too.
                            if mapped == InputButton::Right {
                                self.toggle_capture_mouse();
                            }
                            self.player_input.buttons_down.insert(mapped);
                        }
                        ElementState::Released => {
                            self.player_input.buttons_down.remove(&mapped);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => {
                self.player_input.scroll_delta += y;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(key) = input::map_key(&event.logical_key) {
                    match event.state {
                        ElementState::Pressed => {
                            self.player_input.down.insert(key);
                            self.player_input.just_pressed.insert(key);
                        }
                        ElementState::Released => {
                            self.player_input.down.remove(&key);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Close (Escape) and the Render-mode toggle (TAB, debug builds) are
        // edge reads from the input layer's just-pressed set.
        if self.player_input.just_pressed.contains(&InputKey::Close) {
            self.close_requested = true;
        }
        // Manual exposure (ADR 0007): [ / ] step the composite's EV by half
        // a stop per press, app-wide (not debug-gated — exposure is a
        // rendering control, not a diagnostic).
        if let Some(rcx) = &mut self.rcx {
            if self.player_input.just_pressed.contains(&InputKey::ExposureUp) {
                rcx.region.ev += 0.5;
            }
            if self.player_input.just_pressed.contains(&InputKey::ExposureDown) {
                rcx.region.ev -= 0.5;
            }
        }
        #[cfg(debug_assertions)]
        self.toggle_render_mode();
        self.player_input.end_frame();

        if self.close_requested {
            event_loop.exit();
        } else {
            self.rcx.as_mut().unwrap().window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.player_input.mouse_motion.0 += delta.0;
            self.player_input.mouse_motion.1 += delta.1;
        };
    }
}

/// One of the trace pass's six output images (ADR 0007): the virtual graph
/// resource the nodes' accesses reference, the physical image, and its
/// bindless storage id. The graph maps the virtual id to the physical id
/// per frame, so a window resize can destroy and recreate the physical
/// image without rebuilding the graph.
pub(crate) struct TracePassImage {
    pub virtual_id: Id<Image>,
    pub physical_id: Id<Image>,
    pub storage_id: StorageImageId,
}

/// The trace pass's output images (ADR 0007): the noisy radiance pair and
/// the auxiliary guide buffers the production raygen writes in Voxel mode
/// (diffuse+specular radiance with in-lobe hit distance in alpha,
/// normal+roughness, linear viewZ, backward motion vectors,
/// albedo+metalness). The composite node exposes them (and, from ticket 08,
/// the Denoise pass consumes them).
pub(crate) struct TracePassImages {
    pub diff_radiance: TracePassImage,
    pub spec_radiance: TracePassImage,
    pub normal_roughness: TracePassImage,
    pub viewz: TracePassImage,
    pub mv: TracePassImage,
    pub albedo_metal: TracePassImage,
}

impl TracePassImages {
    /// Adds the six virtual image resources to the task graph (the nodes'
    /// accesses reference these; the per-frame resource map binds them to
    /// the physical images). The create infos must match the physical
    /// images' format and sharing mode (the map asserts on it). Physical
    /// ids and storage ids are filled in by [`Self::attach_physical`].
    fn add_virtual(task_graph: &mut TaskGraph<RegionRenderContext>) -> Self {
        let mut add = |format: Format| {
            task_graph.add_image(&ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                usage: ImageUsage::STORAGE,
                ..Default::default()
            })
        };
        let image = |virtual_id| TracePassImage {
            virtual_id,
            physical_id: Id::INVALID,
            storage_id: StorageImageId::INVALID,
        };
        Self {
            diff_radiance: image(add(Format::R16G16B16A16_SFLOAT)),
            spec_radiance: image(add(Format::R16G16B16A16_SFLOAT)),
            normal_roughness: image(add(Format::R8G8B8A8_UNORM)),
            viewz: image(add(Format::R32_SFLOAT)),
            mv: image(add(Format::R16G16B16A16_SFLOAT)),
            albedo_metal: image(add(Format::R8G8B8A8_UNORM)),
        }
    }

    /// Replaces the physical images and bindless storage ids with the given
    /// freshly created ones (startup and resize), keeping the virtual ids.
    /// The task graph is untouched — the per-frame resource map binds the
    /// virtual ids to the new physical images.
    fn attach_physical(&mut self, physical: TracePassImages) {
        self.diff_radiance.physical_id = physical.diff_radiance.physical_id;
        self.diff_radiance.storage_id = physical.diff_radiance.storage_id;
        self.spec_radiance.physical_id = physical.spec_radiance.physical_id;
        self.spec_radiance.storage_id = physical.spec_radiance.storage_id;
        self.normal_roughness.physical_id = physical.normal_roughness.physical_id;
        self.normal_roughness.storage_id = physical.normal_roughness.storage_id;
        self.viewz.physical_id = physical.viewz.physical_id;
        self.viewz.storage_id = physical.viewz.storage_id;
        self.mv.physical_id = physical.mv.physical_id;
        self.mv.storage_id = physical.mv.storage_id;
        self.albedo_metal.physical_id = physical.albedo_metal.physical_id;
        self.albedo_metal.storage_id = physical.albedo_metal.storage_id;
    }
}

/// Creates the trace pass's six physical output images (ADR 0007), sized to
/// the window, and registers them in the bindless set. Called at startup
/// and again on resize (the old images are destroyed first); the virtual
/// ids in the returned struct are INVALID — they ride on the render
/// context's [`TracePassImages`].
pub(crate) fn create_trace_pass_images(
    resources: &Resources,
    width: u32,
    height: u32,
) -> TracePassImages {
    let bcx = resources.bindless_context().unwrap();
    let create = |format: Format| {
        let physical_id = resources
            .create_image(
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [width, height, 1],
                    usage: ImageUsage::STORAGE,
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
        TracePassImage {
            virtual_id: Id::INVALID,
            physical_id,
            storage_id,
        }
    };
    TracePassImages {
        diff_radiance: create(Format::R16G16B16A16_SFLOAT),
        spec_radiance: create(Format::R16G16B16A16_SFLOAT),
        normal_roughness: create(Format::R8G8B8A8_UNORM),
        viewz: create(Format::R32_SFLOAT),
        mv: create(Format::R16G16B16A16_SFLOAT),
        albedo_metal: create(Format::R8G8B8A8_UNORM),
    }
}
