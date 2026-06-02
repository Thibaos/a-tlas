use glam::{IVec3, Mat4, vec3};
use std::{
    f32::consts::PI,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use vulkano::{
    VulkanError, VulkanLibrary,
    acceleration_structure::AccelerationStructure,
    buffer::{BufferCreateInfo, BufferUsage},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    image::{ImageFormatInfo, ImageLayout, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{
        AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::{BindlessContext, StorageImageId},
    graph::{AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
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
    async_tlas::run_worker,
    physics::PhysicsController,
    player_controller::PlayerController,
    rt::raygen,
    tasks::{render::RayTracingRenderTask, update_as::UpdateAccelerationStructureTask},
    world::{Vertex3DColor, chunk::Chunks, voxel::open_file},
};

#[cfg(debug_assertions)]
use crate::tasks::debug::{self, DrawDebugTask, create_debug_pipeline};

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;
pub const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;
pub const TICKS_PER_SECOND: u32 = 1;
pub const MAX_DEBUG_LINES: u32 = 4096;
pub const MAX_INSTANCE_COUNT: u64 = 2u64.pow(20);

pub struct App {
    close_requested: bool,

    instance: Arc<Instance>,
    pub device: Arc<Device>,

    pub graphics_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
    pub transfer_queue: Arc<Queue>,

    pub memory_allocator: Arc<dyn MemoryAllocator>,

    pub resources: Arc<Resources>,
    pub graphics_flight_id: Id<Flight>,
    pub compute_flight_id: Id<Flight>,

    last_frame_update: Instant,
    next_log_update: Instant,
    delta_time: Duration,
    focused: bool,

    pub voxel_data: dot_vox::DotVoxData,
    pub world: Arc<Chunks>,

    player_controller: PlayerController,
    physics_controller: PhysicsController,

    worker_available: Arc<AtomicBool>,

    rcx: Option<RenderContext>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    virtual_swapchain_id: Id<Swapchain>,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    // scene_params: tree64::SceneParams,
    pub rt_camera_data: raygen::Camera,
    pub rt_sunlight_data: raygen::Sunlight,
    // pub tlas: Arc<AccelerationStructure>,
    pub acceleration_structures: [Arc<AccelerationStructure>; 2],
    pub current_as_index: Arc<AtomicBool>,
    #[cfg(debug_assertions)]
    pub debug_constant_data: debug::shader::vert::PushConstants,
    #[cfg(debug_assertions)]
    pub debug_lines: Vec<Vertex3DColor>,
    #[cfg(debug_assertions)]
    pub viewport: Viewport,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<Self>,

    channel: mpsc::Sender<IVec3>,
}

pub struct AsyncRenderContext {
    pub acceleration_structures: [Arc<AccelerationStructure>; 2],
    pub current_as_index: Arc<AtomicBool>,
    pub world: Arc<Chunks>,
    pub position: glam::IVec3,
}

impl App {
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

        let voxel_data = open_file("assets/castle.vox");
        let world = Arc::new(Chunks::new(&voxel_data));

        App {
            close_requested: false,

            instance,
            device,

            graphics_queue,
            compute_queue,
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

            voxel_data,
            world,

            worker_available: Arc::new(AtomicBool::new(true)),

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
        let view = self.player_controller.view();

        let size = rcx.window.inner_size();

        let proj = Mat4::perspective_lh(
            PI / 2.0,
            (size.width as f32) / (size.height as f32),
            0.01,
            10000.0,
        );

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

    #[cfg(debug_assertions)]
    pub fn update_debug_lines(&mut self) {
        self.rcx.as_mut().unwrap().debug_lines = self.world.debug_lines();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes =
            WindowAttributes::default().with_inner_size(PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let window_size = window.inner_size();
        let surface = Surface::from_window(&self.instance, &window).unwrap();

        let (swapchain_id, swapchain_format) = {
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

            (
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
                    .unwrap(),
                image_format,
            )
        };

        let swapchain_storage_image_ids =
            window_size_dependent_setup(&self.resources, swapchain_id);

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        let rt_pass = RayTracingRenderTask::new(self, virtual_swapchain_id, MAX_INSTANCE_COUNT);

        let update_as_task = UpdateAccelerationStructureTask::new(
            self,
            MAX_INSTANCE_COUNT as u32,
            rt_pass.instance_buffer_id,
            rt_pass.blas.device_address().into(),
        );

        let acceleration_structures = rt_pass.acceleration_structures.clone();
        let current_as_index = rt_pass.current_as_index.clone();

        let instance_buffer_id = rt_pass.instance_buffer_id;

        let (channel, receiver) = mpsc::channel();

        run_worker(
            receiver,
            update_as_task,
            self.compute_queue.clone(),
            self.resources.clone(),
            self.graphics_flight_id,
            self.compute_flight_id,
            rt_pass.acceleration_structures.clone(),
            rt_pass.current_as_index.clone(),
            self.world.clone(),
            self.worker_available.clone(),
        );

        let rt_node_id = task_graph
            .create_task_node("Render", QueueFamilyType::Graphics, rt_pass)
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .buffer_access(
                instance_buffer_id,
                AccessTypes::RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ,
            )
            .build();

        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let debug_vertex_buffer_id = self
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                        | MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[Vertex3DColor]>(MAX_DEBUG_LINES.into()).unwrap(),
            )
            .unwrap();

        #[cfg(debug_assertions)]
        let debug_pass = DrawDebugTask::new(debug_vertex_buffer_id);

        #[cfg(debug_assertions)]
        let debug_node_id = task_graph
            .create_task_node("Debug", QueueFamilyType::Graphics, debug_pass)
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    format: swapchain_format,
                    ..Default::default()
                },
            )
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
            )
            .buffer_access(debug_vertex_buffer_id, AccessTypes::VERTEX_ATTRIBUTE_READ)
            .build();

        // task_graph.add_host_buffer_access(instance_buffer_id, HostAccessType::Write);

        #[cfg(debug_assertions)]
        task_graph.add_host_buffer_access(debug_vertex_buffer_id, HostAccessType::Write);

        // task_graph.add_edge(update_as_node_id, rt_node_id).unwrap();
        #[cfg(debug_assertions)]
        task_graph.add_edge(rt_node_id, debug_node_id).unwrap();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.graphics_queue],
                present_queue: Some(&self.graphics_queue),
                flight_id: self.graphics_flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        #[cfg(debug_assertions)]
        {
            let node = task_graph.task_node(debug_node_id).unwrap();

            let debug_pipeline = create_debug_pipeline(self, node);

            task_graph
                .task_node_mut(debug_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DrawDebugTask>()
                .unwrap()
                .pipeline = Some(debug_pipeline);
        }

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
            acceleration_structures,
            current_as_index,
            #[cfg(debug_assertions)]
            debug_constant_data,
            #[cfg(debug_assertions)]
            debug_lines: vec![],
            #[cfg(debug_assertions)]
            viewport,
            swapchain_storage_image_ids,

            channel,
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
                // self.update_debug_lines();
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

                        rcx.recreate_swapchain = false;
                    }
                }

                self.resources
                    .flight(self.graphics_flight_id)
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
                if event.state == ElementState::Pressed
                    && let Some(txt) = event.logical_key.to_text()
                    && txt == "r"
                    && self.worker_available.load(Ordering::Acquire)
                {
                    self.rcx
                        .as_mut()
                        .unwrap()
                        .channel
                        .send(self.player_controller.translation.as_ivec3())
                        .unwrap();
                }
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
