use std::{f32::consts::PI, sync::Arc, time::Duration};
use vulkano::{
    VulkanError, VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    image::{ImageFormatInfo, ImageLayout, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};

#[cfg(debug_assertions)]
use vulkano::buffer::{BufferCreateInfo, BufferUsage};
#[cfg(debug_assertions)]
use vulkano::memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::{BindlessContext, StorageImageId},
    graph::{CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, Flight, ImageLayoutType, Resources, ResourcesCreateInfo},
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
    grid::LATTICE_HALF_EXTENT,
    input::{self, Input, InputButton, InputKey},
    measure::Measurement,
    physics::PhysicsController,
    player::PlayerController,
    region::{
        input::RendererInput,
        render::{
            HullCrossedCounter, RegionRenderContext, RegionRenderTask, RenderMode, capture_raygen,
            production_raygen,
        },
        residency::RegionStore,
        snapshot::emit_snapshots,
    },
    schedule::ScheduleController,
    world::{voxel::open_file, world::World},
};

#[cfg(debug_assertions)]
use crate::debug::{DrawHeatmapTask, create_heatmap_pipeline};

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;
pub const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;
pub const TICKS_PER_SECOND: u32 = 1;
/// The per-pixel hull-crossed count buffer's fixed pixel ceiling (4K): debug
/// builds only. Resizing the window beyond it is unsupported in the heatmap.
#[cfg(debug_assertions)]
const HEATMAP_MAX_PIXELS: u64 = 3840 * 2160;

/// The shared GPU stack (instance, device, queues, allocator, taskgraph
/// resources and flights). Constructed once per event loop by [`App::new`] and
/// by the offline validator; the validator renders through the same stack so the
/// device/queue/extension surface it validates is the app's.
pub struct GpuStack {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,

    pub graphics_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
    pub transfer_queue: Arc<Queue>,

    pub memory_allocator: Arc<dyn MemoryAllocator>,

    pub resources: Arc<Resources>,
    pub graphics_flight_id: Id<Flight>,
    pub compute_flight_id: Id<Flight>,
}

impl GpuStack {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let required_extensions = Surface::required_extensions(event_loop);

        let library = unsafe { VulkanLibrary::new() }.unwrap();
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &InstanceExtensions {
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_acceleration_structure: true,
            khr_deferred_host_operations: true,
            khr_ray_tracing_maintenance1: true,
            khr_ray_tracing_pipeline: true,
            khr_synchronization2: true,
            khr_shader_clock: true,
            khr_swapchain: true,
            ..BindlessContext::required_extensions(&instance)
        };
        let device_features = DeviceFeatures {
            acceleration_structure: true,
            descriptor_binding_acceleration_structure_update_after_bind: true,
            ray_tracing_pipeline: true,
            buffer_device_address: true,
            storage_push_constant8: true,
            synchronization2: true,
            shader_float64: true,
            shader_int64: true,
            shader_int8: true,
            shader_subgroup_clock: true,
            // The device-scope clock for the debug Ray-latency mode (a
            // clockRealtime delta). Debug-only: the release raygen omits the
            // clock (ATLAS_RT_RAY_LATENCY undefined), so release needs no
            // shader_device_clock feature.
            shader_device_clock: cfg!(debug_assertions),
            storage_buffer8_bit_access: true,
            ..BindlessContext::required_features(&instance)
        };

        let (physical_device, graphics_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().contains(&device_features)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let compute_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
            .min_by_key(|(_, q)| q.queue_flags.count())
            .unwrap()
            .0 as u32;

        let transfer_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_flags.intersects(QueueFlags::TRANSFER))
            .min_by_key(|(_, q)| q.queue_flags.count())
            .unwrap()
            .0 as u32;

        let (device, mut queues) = {
            let mut queue_create_infos = vec![QueueCreateInfo {
                queue_family_index: graphics_family_index,
                ..Default::default()
            }];

            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: compute_family_index,
                ..Default::default()
            });

            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: transfer_family_index,
                ..Default::default()
            });

            Device::new(
                &physical_device,
                &DeviceCreateInfo {
                    enabled_extensions: &device_extensions,
                    enabled_features: &device_features,
                    queue_create_infos: &queue_create_infos,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let graphics_queue = queues.next().unwrap();
        let compute_queue = queues.next().unwrap();
        let transfer_queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let resources = Resources::new(
            &device,
            &ResourcesCreateInfo {
                bindless_context: Some(&Default::default()),
                ..Default::default()
            },
        )
        .unwrap();

        let graphics_flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();
        let compute_flight_id = resources.create_flight(1).unwrap();

        Self {
            instance,
            device,
            graphics_queue,
            compute_queue,
            transfer_queue,
            memory_allocator,
            resources,
            graphics_flight_id,
            compute_flight_id,
        }
    }
}

pub struct App {
    close_requested: bool,

    pub gpu: GpuStack,

    delta_time: Duration,
    focused: bool,

    pub voxel_data: dot_vox::DotVoxData,
    pub world: Arc<World>,

    player_controller: PlayerController,
    player_input: Input,
    physics_controller: PhysicsController,
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

            player_controller: PlayerController::default(),
            player_input: Input::default(),
            physics_controller: PhysicsController::new(),
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
    /// latency -> hull-crossed. Reads the just-pressed edge from the shared
    /// input layer; the mode lives in the render context and is written into
    /// the push constants every frame, so toggling is a per-frame flag, never a
    /// pipeline rebuild.
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
                RenderMode::HullCrossed => RenderMode::Voxel,
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

        let task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.gpu.graphics_queue],
                present_queue: Some(&self.gpu.graphics_queue),
                flight_id: self.gpu.graphics_flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        // The heatmap compute pipeline is injected into the compiled graph
        // only in debug builds (the overlay is app-only).
        #[cfg(debug_assertions)]
        let mut task_graph = task_graph;

        #[cfg(debug_assertions)]
        {
            // The heatmap compute pipeline has no subpass, so it is created
            // from the app only (no task-node reference).
            let heatmap_pipeline = create_heatmap_pipeline(self);
            task_graph
                .task_node_mut(heatmap_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DrawHeatmapTask>()
                .unwrap()
                .pipeline = Some(heatmap_pipeline);
        }

        let region = RegionRenderContext {
            camera: capture_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
            },
            swapchain_storage_image_ids,
            // The production raygen never dereferences `t_image_id`
            // (shaders/region/production.rgen) — it stays INVALID.
            t_image_storage_id: StorageImageId::INVALID,
            // Voxel is the default; TAB (debug builds) toggles this in the
            // render context before each frame.
            mode: RenderMode::default(),
        };

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
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
                self.physics_controller.request_update();
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

                        batch.enqueue();

                        rcx.region.swapchain_storage_image_ids =
                            window_size_dependent_setup(&self.gpu.resources, rcx.swapchain_id);

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

                let resource_map =
                    resource_map!(&rcx.task_graph, rcx.virtual_swapchain_id => rcx.swapchain_id)
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

pub(crate) fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
) -> Vec<StorageImageId> {
    let bcx = resources.bindless_context().unwrap();
    let swapchain_state = resources.swapchain(swapchain_id);
    let images = swapchain_state.images();

    images
        .iter()
        .map(|image| {
            let image_view = ImageView::new_default(image).unwrap();

            bcx.global_set()
                .add_storage_image(image_view, ImageLayout::General)
        })
        .collect()
}
