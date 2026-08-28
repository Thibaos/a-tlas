mod input;
mod player;
mod schedule;

use std::{f32::consts::PI, sync::Arc, time::Duration};
use vulkano::{
    VulkanError,
    image::{ImageFormatInfo, ImageUsage},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};

use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::StorageImageId,
    graph::{CompileInfo, ExecutableTaskGraph, ExecuteError, ResourceMap, TaskGraph},
    resource::{AccessTypes, ImageLayoutType},
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
    },
    core::{
        render::{
            composite::{CompositeTask, create_composite_pipeline},
            frame_images::FrameImages,
            gpu::{GpuDesc, MIN_SWAPCHAIN_IMAGES},
            nrd::{DenoiseTask, NrdInstance},
            region::{
                feed::RendererInput,
                residency::RegionStore,
                task::{
                    NrdFrame, RegionRenderContext, RegionRenderTask, RenderMode, capture_raygen,
                    default_ev, default_scene, production_raygen,
                },
            },
        },
        world::{World, format::open_file, grid::LATTICE_HALF_EXTENT, snapshot::emit_snapshots},
    },
};

pub struct App {
    close_requested: bool,

    pub gpu: GpuDesc,

    delta_time: Duration,
    focused: bool,
    frame_seed: u32,

    pub voxel_data: dot_vox::DotVoxData,
    pub world: Arc<World>,

    player_controller: PlayerController,
    player_input: Input,
    schedule_controller: ScheduleController,

    input: RendererInput,
    store: RegionStore,

    nrd: Option<Arc<NrdInstance>>,
    prev_view: glam::Mat4,
    prev_proj: glam::Mat4,
    camera_valid: bool,
    nrd_frame_index: u32,
    nrd_clear_pending: bool,

    rcx: Option<RenderContext>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    virtual_swapchain_id: Id<Swapchain>,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<RegionRenderContext>,
    frame_images: FrameImages,
    denoise_node_id: vulkano_taskgraph::graph::NodeId,
    region: RegionRenderContext,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>, world_path: &str, clip_oob: bool) -> Self {
        let gpu = GpuDesc::new(event_loop);

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

        let input = RendererInput::new();
        input.submit_batch(emit_snapshots(&world));

        let store = RegionStore::new(&gpu, &voxel_data, &input);

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

            nrd: None,
            prev_view: glam::Mat4::IDENTITY,
            prev_proj: glam::Mat4::IDENTITY,
            camera_valid: false,
            nrd_frame_index: 0,
            nrd_clear_pending: true,

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

    #[cfg(debug_assertions)]
    fn handle_toggle_render_mode(&mut self) {
        if self
            .player_input
            .just_pressed
            .contains(&InputKey::ToggleRenderMode)
        {
            let rcx = self.rcx.as_mut().unwrap();
            rcx.region.mode = match rcx.region.mode {
                RenderMode::Voxel => RenderMode::Hull,
                RenderMode::Hull => RenderMode::Normal,
                RenderMode::Normal => {
                    if self.nrd.is_some() {
                        RenderMode::NrdValidation
                    } else {
                        eprintln!("render mode: NRD validation unavailable, denoiser inactive");
                        RenderMode::Voxel
                    }
                }
                RenderMode::NrdValidation => RenderMode::Voxel,
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
        }
    }

    fn update_camera(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();

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
            view_prev: self.prev_view.to_cols_array_2d(),
            proj_prev: self.prev_proj.to_cols_array_2d(),
        };

        rcx.region.nrd.view_to_clip = proj.to_cols_array();
        rcx.region.nrd.world_to_view = view.to_cols_array();
        if self.camera_valid {
            rcx.region.nrd.view_to_clip_prev = self.prev_proj.to_cols_array();
            rcx.region.nrd.world_to_view_prev = self.prev_view.to_cols_array();
        } else {
            rcx.region.nrd.view_to_clip_prev = rcx.region.nrd.view_to_clip;
            rcx.region.nrd.world_to_view_prev = rcx.region.nrd.world_to_view;
        }

        self.prev_view = view;
        self.prev_proj = proj;
        self.camera_valid = true;
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

        let mut task_graph = TaskGraph::new(&self.gpu.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        let mut frame_images = FrameImages::declare(&mut task_graph);
        let swapchain_state = self.gpu.resources.swapchain(swapchain_id);
        let extent = swapchain_state.images()[0].extent();
        frame_images.recreate(&self.gpu.resources, swapchain_id, extent);
        self.nrd = NrdInstance::recreate(self.nrd.take(), &self.gpu, extent);

        let raygen = unsafe {
            production_raygen::load(&self.gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

        let rt_pass =
            RegionRenderTask::new(&self.gpu, &self.store, virtual_swapchain_id, &raygen, true);
        let instance_buffer_id = rt_pass.instance_buffer_id();

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
        frame_images.declare_trace_outputs(&mut rt_node);
        let rt_node_id = rt_node.build();

        let mut composite_node = task_graph.create_task_node(
            "Composite",
            QueueFamilyType::Graphics,
            CompositeTask::new(virtual_swapchain_id),
        );
        composite_node.image_access(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
            ImageLayoutType::General,
        );
        frame_images.declare_composite_reads(&mut composite_node);
        let composite_node_id = composite_node.build();

        let mut denoise_node =
            task_graph.create_task_node("Denoise", QueueFamilyType::Graphics, DenoiseTask::new());
        frame_images.declare_denoise_io(&mut denoise_node);
        let denoise_node_id = denoise_node.build();

        task_graph.add_edge(rt_node_id, denoise_node_id).unwrap();
        task_graph
            .add_edge(denoise_node_id, composite_node_id)
            .unwrap();

        let task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.gpu.graphics_queue],
                present_queue: Some(&self.gpu.graphics_queue),
                flight_id: self.gpu.graphics_flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        let mut task_graph = task_graph;

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

        if let Some(nrd) = self.nrd.clone() {
            let task = task_graph
                .task_node_mut(denoise_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DenoiseTask>()
                .unwrap();
            task.instance = Some(nrd.clone());
            task.inputs = Some(frame_images.nrd_inputs(&self.gpu.resources));
        }

        let mut region = RegionRenderContext {
            camera: capture_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
                view_prev: glam::Mat4::IDENTITY.to_cols_array_2d(),
                proj_prev: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
            scene: default_scene(),
            swapchain_storage_image_ids: Vec::new(),
            t_image_storage_id: StorageImageId::INVALID,
            diff_radiance_image_id: StorageImageId::INVALID,
            spec_radiance_image_id: StorageImageId::INVALID,
            normal_roughness_image_id: StorageImageId::INVALID,
            viewz_image_id: StorageImageId::INVALID,
            mv_image_id: StorageImageId::INVALID,
            denoised_diff_image_id: StorageImageId::INVALID,
            denoised_spec_image_id: StorageImageId::INVALID,
            validation_image_id: StorageImageId::INVALID,
            denoiser_enabled: self.nrd.is_some(),
            nrd: NrdFrame::default(),
            albedo_metal_image_id: StorageImageId::INVALID,
            ev: default_ev(),
            mode: RenderMode::default(),
            frame_seed: 0,
        };

        frame_images.bind_into(&mut region);

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            frame_images,
            denoise_node_id,
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
                self.frame_seed = self.frame_seed.wrapping_add(1);
                self.rcx.as_mut().unwrap().region.frame_seed = self.frame_seed;
                self.request_log();

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

                        let extent =
                            self.gpu.resources.swapchain(rcx.swapchain_id).images()[0].extent();

                        self.nrd = NrdInstance::recreate(self.nrd.take(), &self.gpu, extent);

                        rcx.frame_images
                            .recreate(&self.gpu.resources, rcx.swapchain_id, extent);
                        rcx.frame_images.bind_into(&mut rcx.region);
                        rcx.region.denoiser_enabled = self.nrd.is_some();

                        if let Some(nrd) = &self.nrd {
                            self.nrd_clear_pending = true;

                            if let Some(task) = rcx
                                .task_graph
                                .task_node_mut(rcx.denoise_node_id)
                                .ok()
                                .and_then(|node| node.task_mut().downcast_mut::<DenoiseTask>())
                            {
                                task.instance = Some(nrd.clone());
                                task.inputs =
                                    Some(rcx.frame_images.nrd_inputs(&self.gpu.resources));
                            }
                        }

                        rcx.recreate_swapchain = false;
                    }
                }

                self.gpu
                    .resources
                    .flight(self.gpu.graphics_flight_id)
                    .wait_idle()
                    .unwrap();

                let apply_report = self.store.apply(&self.gpu, &self.input);
                let edited = !apply_report.dirty.is_empty();

                let clear = self.nrd_clear_pending && self.nrd.is_some();
                let reset = edited && !clear;

                self.nrd_frame_index = if clear || reset {
                    1
                } else {
                    self.nrd_frame_index.wrapping_add(1)
                };
                self.nrd_clear_pending = false;

                let rcx = self.rcx.as_mut().unwrap();
                rcx.region.nrd.clear = clear;
                rcx.region.nrd.reset = reset;
                rcx.region.nrd.frame_index = if clear || reset {
                    0
                } else {
                    self.nrd_frame_index
                };

                let resource_map = {
                    let mut map = ResourceMap::new(&rcx.task_graph).unwrap();
                    map.insert(rcx.virtual_swapchain_id, rcx.swapchain_id)
                        .unwrap();

                    for (virtual_id, physical_id) in rcx.frame_images.resource_pairs() {
                        map.insert(virtual_id, physical_id).unwrap();
                    }

                    map
                };

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
                if let Some(mapped) = input::map_mouse_button(button) {
                    match state {
                        ElementState::Pressed => {
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
        if self.player_input.just_pressed.contains(&InputKey::Close) {
            self.close_requested = true;
        }

        #[cfg(debug_assertions)]
        self.handle_toggle_render_mode();
        self.player_input.clear();

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
