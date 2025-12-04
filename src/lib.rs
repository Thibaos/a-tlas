use glam::{Mat4, vec3};
use physics::PhysicsController;
use player_controller::PlayerController;
use std::{
    f32::consts::PI,
    sync::Arc,
    time::{Duration, Instant},
};
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
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::{BindlessContext, StorageImageId},
    graph::{CompileInfo, ExecutableTaskGraph, ExecuteError, NodeId, TaskGraph},
    resource::{
        AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources, ResourcesCreateInfo,
    },
    resource_map,
};

#[cfg(debug_assertions)]
use vulkano::pipeline::graphics::viewport::Viewport;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use crate::{
    rt::raygen,
    tasks::{as_update::UpdateAccelerationStructureTask, debug, rt::RayTracingPass},
    world::{chunk::Chunks, voxel::open_file},
};

mod physics;
mod player_controller;
mod rt;
mod tasks;
mod world;

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;
pub const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;
pub const TICKS_PER_SECOND: u32 = 1;

pub struct App {
    close_requested: bool,

    instance: Arc<Instance>,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    #[cfg(debug_assertions)]
    transfer_queue: Arc<Queue>,

    memory_allocator: Arc<dyn MemoryAllocator>,

    resources: Arc<Resources>,
    graphics_flight_id: Id<Flight>,
    compute_flight_id: Id<Flight>,
    rcx: Option<RenderContext>,

    last_frame_update: Instant,
    next_log_update: Instant,
    delta_time: Duration,
    focused: bool,

    max_instance_count: u64,
    voxel_data: dot_vox::DotVoxData,
    world: Chunks,

    player_controller: PlayerController,
    physics_controller: PhysicsController,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    virtual_swapchain_id: Id<Swapchain>,
    swapchain_storage_image_ids: Vec<StorageImageId>,
    // scene_params: tree64::SceneParams,
    rt_camera_data: raygen::Camera,
    rt_sunlight_data: raygen::Sunlight,
    #[cfg(debug_assertions)]
    debug_constant_data: debug::shader::vert::PushConstants,
    #[cfg(debug_assertions)]
    viewport: Viewport,
    recreate_swapchain: bool,
    renderer_node_id: NodeId,
    task_graph: ExecutableTaskGraph<Self>,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

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
                            && p.presentation_support(i as u32, event_loop).unwrap()
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
        #[cfg(debug_assertions)]
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

        let max_instance_count = device
            .physical_device()
            .properties()
            .max_instance_count
            .expect("Max instance count not found");

        let voxel_data = open_file("assets/nuke.vox");
        let world = Chunks::new(&voxel_data);

        App {
            close_requested: false,

            instance,
            device,

            graphics_queue,
            compute_queue,
            #[cfg(debug_assertions)]
            transfer_queue,

            memory_allocator,

            resources,
            graphics_flight_id,
            compute_flight_id,

            last_frame_update: Instant::now(),
            next_log_update: Instant::now().checked_add(Duration::from_secs(1)).unwrap(),
            delta_time: Duration::ZERO,
            focused: false,

            player_controller: PlayerController::default(),
            physics_controller: PhysicsController::new(),

            max_instance_count,
            voxel_data,
            world,

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

    fn update_log_instant(&mut self) {
        let now = Instant::now();
        if !now.duration_since(self.next_log_update).is_zero() {
            self.next_log_update = Instant::now().checked_add(Duration::from_secs(1)).unwrap();
        }
    }

    pub fn update_delta_time(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_update);
        if !now.duration_since(self.next_log_update).is_zero() {
            #[cfg(debug_assertions)]
            println!("{:.2} fps", 1.0 / delta.as_secs_f32());
        }
        self.last_frame_update = now;
        self.delta_time = delta;
    }

    pub fn update_camera(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();

        self.player_controller.fly_movement(self.delta_time);
        let translation = self.player_controller.translation;
        let view = self.player_controller.view();

        let size = rcx.window.inner_size();

        let proj = Mat4::perspective_lh(
            PI / 2.0,
            (size.width as f32) / (size.height as f32),
            0.01,
            10000.0,
        );

        let inv_proj_mat = (proj * view).inverse().to_cols_array_2d();
        // let camera_position = Padded(translation.to_array());

        // rcx.scene_params = tree64::SceneParams {
        //     inv_proj_mat,
        //     camera_position,
        //     resolution: size.into(),
        // };

        rcx.rt_camera_data = raygen::Camera {
            proj_inverse: proj.inverse().to_cols_array_2d(),
            view_inverse: view.inverse().to_cols_array_2d(),
            view_proj: (view * proj).to_cols_array_2d(),
        };

        #[cfg(debug_assertions)]
        {
            rcx.debug_constant_data = debug::shader::vert::PushConstants {
                world: Mat4::default().to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
            };
        }
    }

    pub fn update_look_position(&mut self, delta: (f64, f64)) {
        if self.focused {
            self.player_controller.rotate(delta);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes =
            WindowAttributes::default().with_inner_size(PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let window_size = window.inner_size();
        let surface = Surface::from_window(&self.instance, &window).unwrap();

        let swapchain_id = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();
            let (image_format, image_color_space) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()
                .into_iter()
                .find(|(format, _)| {
                    self.device
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

            self.resources
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
                .unwrap()
        };

        let swapchain_storage_image_ids =
            window_size_dependent_setup(&self.resources, swapchain_id);

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        let rt_pass = RayTracingPass::new(&self, virtual_swapchain_id, self.max_instance_count);
        let tlas_instance_buffer_id = rt_pass.instance_buffer_id;

        let update_as_task = UpdateAccelerationStructureTask {
            blas_reference: rt_pass.blas.device_address().into(),
            instance_buffer_id: rt_pass.instance_buffer_id,
            scratch_buffer_id: rt_pass.scratch_buffer_id,
            tlas: rt_pass.tlas.clone(),
            max_instance_count: self.max_instance_count,
        };

        let renderer_node_id = task_graph
            .create_task_node("rt", QueueFamilyType::Graphics, rt_pass)
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .build();

        let update_as_node_id = task_graph
            .create_task_node("update AS", QueueFamilyType::Compute, update_as_task)
            .build();

        task_graph
            .add_edge(renderer_node_id, update_as_node_id)
            .unwrap();

        task_graph.add_host_buffer_access(tlas_instance_buffer_id, HostAccessType::Write);

        let task_graph = unsafe {
            task_graph
                .compile(&CompileInfo {
                    queues: &[&self.graphics_queue],
                    present_queue: Some(&self.graphics_queue),
                    flight_id: self.graphics_flight_id,
                    ..Default::default()
                })
                .unwrap()
        };
        // let scene_params = tree64::SceneParams {
        //     inv_proj_mat: Mat4::perspective_lh(PI / 2.0, 16.0 / 9.0, 0.01, 10000.0)
        //         .to_cols_array_2d(),
        //     camera_position: Padded([0.0, 0.0, 0.0]),
        //     resolution: window_size.into(),
        // };

        #[cfg(debug_assertions)]
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let rt_camera_data = raygen::Camera {
            proj_inverse: [[0.0; 4]; 4],
            view_inverse: [[0.0; 4]; 4],
            view_proj: [[0.0; 4]; 4],
        };

        let rt_sunlight_data = raygen::Sunlight {
            direction: vec3(0.5, -0.5, 0.5).to_array(),
        };

        #[cfg(debug_assertions)]
        let debug_constant_data = debug::shader::vert::PushConstants {
            world: Mat4::default().to_cols_array_2d(),
            view: Mat4::default().to_cols_array_2d(),
            proj: Mat4::default().to_cols_array_2d(),
        };

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            // scene_params,
            rt_camera_data,
            rt_sunlight_data,
            #[cfg(debug_assertions)]
            debug_constant_data,
            #[cfg(debug_assertions)]
            viewport,
            swapchain_storage_image_ids,
            renderer_node_id,
        });
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                self.close_requested = true;
            }
            WindowEvent::Resized(_) => {
                self.rcx.as_mut().unwrap().recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                self.update_delta_time();
                self.update_camera();
                self.physics_controller.request_update();
                self.update_log_instant();

                {
                    let rcx = self.rcx.as_mut().unwrap();

                    let window_size = rcx.window.inner_size();

                    if window_size.width == 0 || window_size.height == 0 {
                        return;
                    }

                    if rcx.recreate_swapchain {
                        rcx.swapchain_id = self
                            .resources
                            .recreate_swapchain(rcx.swapchain_id, |create_info| {
                                SwapchainCreateInfo {
                                    image_extent: window_size.into(),
                                    ..create_info.clone()
                                }
                            })
                            .expect("failed to recreate swapchain");

                        #[cfg(debug_assertions)]
                        {
                            rcx.viewport = Viewport {
                                offset: [0.0, 0.0],
                                extent: window_size.into(),
                                min_depth: 0.0,
                                max_depth: 1.0,
                            };
                        }

                        let mut batch = self.resources.create_deferred_batch();

                        for &id in &rcx.swapchain_storage_image_ids {
                            batch.destroy_storage_image(id);
                        }

                        batch.enqueue();

                        rcx.swapchain_storage_image_ids =
                            window_size_dependent_setup(&self.resources, rcx.swapchain_id);

                        // let renderer = rcx
                        //     .task_graph
                        //     .task_node_mut(rcx.renderer_node_id)
                        //     .unwrap()
                        //     .task_mut()
                        //     .downcast_mut::<RayTracingPass>()
                        //     .unwrap();

                        // renderer.swapchain_id = rcx.swapchain_id;

                        rcx.recreate_swapchain = false;
                    }
                }

                self.resources
                    .flight(self.graphics_flight_id)
                    .unwrap()
                    .wait_idle()
                    .unwrap();

                let rcx = self.rcx.as_mut().unwrap();

                let resource_map =
                    resource_map!(&rcx.task_graph, rcx.virtual_swapchain_id => rcx.swapchain_id)
                        .unwrap();

                let execute_result = unsafe {
                    rcx.task_graph
                        .execute(resource_map, rcx, || rcx.window.pre_present_notify())
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
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => self.toggle_capture_mouse(),
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => self.player_controller.handle_speed_change(y),
            WindowEvent::KeyboardInput { event, .. } => {
                self.player_controller.handle_keyboard_event(event)
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
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
            self.update_look_position(delta)
        };
    }
}

fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
) -> Vec<StorageImageId> {
    let bcx = resources.bindless_context().unwrap();
    let swapchain_state = resources.swapchain(swapchain_id).unwrap();
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
