mod input;
mod player;
mod schedule;

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
    },
    core::gpu::{GpuStack, MIN_SWAPCHAIN_IMAGES},
    core::grid::LATTICE_HALF_EXTENT,
    render::{
        composite::{CompositeTask, create_composite_pipeline},
        nrd::{DenoiseTask, NrdInputs, NrdInstance},
        region::{
            feed::RendererInput,
            residency::RegionStore,
            task::{
                HullCrossedCounter, NrdFrame, RegionRenderContext, RegionRenderTask, RenderMode,
                capture_raygen, default_scene, production_raygen,
            },
        },
        swapchain::window_size_dependent_setup,
    },
    world::{World, format::open_file, snapshot::emit_snapshots},
};

#[cfg(debug_assertions)]
use crate::render::debug::{DrawHeatmapTask, create_heatmap_pipeline};

#[cfg(debug_assertions)]
const HEATMAP_MAX_PIXELS: u64 = 3840 * 2160;

pub struct App {
    close_requested: bool,

    pub gpu: GpuStack,

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
    trace_pass_images: TracePassImages,
    denoise_node_id: vulkano_taskgraph::graph::NodeId,
    region: RegionRenderContext,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>, world_path: &str, clip_oob: bool) -> Self {
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

        let swapchain_storage_image_ids =
            window_size_dependent_setup(&self.gpu.resources, swapchain_id);

        let mut task_graph = TaskGraph::new(&self.gpu.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        let mut trace_pass_images = TracePassImages::add_virtual(&mut task_graph);
        let swapchain_state = self.gpu.resources.swapchain(swapchain_id);
        let extent = swapchain_state.images()[0].extent();
        let physical_trace_pass_images =
            create_trace_pass_images(&self.gpu.resources, extent[0], extent[1]);
        trace_pass_images.attach_physical(physical_trace_pass_images);

        let nrd = match NrdInstance::new(&self.gpu, extent[0], extent[1]) {
            Ok(instance) => {
                println!(
                    "denoiser: NVIDIA NRD v{}.{}.{} ReBLUR (REBLUR_DIFFUSE_SPECULAR)",
                    crate::render::nrd::sys::NRD_VERSION_MAJOR,
                    crate::render::nrd::sys::NRD_VERSION_MINOR,
                    crate::render::nrd::sys::NRD_VERSION_BUILD,
                );
                Some(Arc::new(instance))
            }
            Err(error) => {
                eprintln!("denoiser disabled: {error}");
                None
            }
        };
        self.nrd = nrd;

        let raygen = unsafe {
            production_raygen::load(&self.gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

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
            Some(HullCrossedCounter {
                buffer_id,
                storage_id,
            })
        };
        #[cfg(not(debug_assertions))]
        let hull_crossed: Option<HullCrossedCounter> = None;

        let rt_pass = RegionRenderTask::new(
            &self.gpu,
            &self.store,
            virtual_swapchain_id,
            &raygen,
            hull_crossed.as_ref(),
            true,
        );
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
        #[cfg(debug_assertions)]
        {
            rt_node.buffer_access(
                hull_crossed.as_ref().unwrap().buffer_id,
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
            );
        }
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
            .image_access(
                trace_pass_images.denoised_diff.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ,
                ImageLayoutType::General,
            )
            .image_access(
                trace_pass_images.denoised_spec.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ,
                ImageLayoutType::General,
            )
            .image_access(
                trace_pass_images.viewz.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ,
                ImageLayoutType::General,
            )
            .build();

        let mut denoise_node =
            task_graph.create_task_node("Denoise", QueueFamilyType::Graphics, DenoiseTask::new());
        for image in [
            &trace_pass_images.diff_radiance,
            &trace_pass_images.spec_radiance,
            &trace_pass_images.normal_roughness,
            &trace_pass_images.viewz,
            &trace_pass_images.mv,
        ] {
            denoise_node.image_access(
                image.virtual_id,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            );
        }
        for image in [
            &trace_pass_images.denoised_diff,
            &trace_pass_images.denoised_spec,
        ] {
            denoise_node.image_access(
                image.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            );
        }
        // The constants buffer stays undeclared on purpose: declaring the
        // physical id here trips ResourceMap validation (InvalidSlotError).
        // Its hazards are covered anyway: update_buffer lands in the same
        // recording as the dispatches behind an explicit TRANSFER_WRITE
        // barrier, and frames are serialized by the per-frame wait_idle.
        let denoise_node_id = denoise_node.build();

        #[cfg(debug_assertions)]
        task_graph
            .add_edge(heatmap_node_id, denoise_node_id)
            .unwrap();
        #[cfg(not(debug_assertions))]
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
            let input_view = |image: &TracePassImage| {
                ImageView::new_default(&self.gpu.resources.image(image.physical_id).image().clone())
                    .unwrap()
            };
            let inputs = NrdInputs {
                diff_radiance: input_view(&trace_pass_images.diff_radiance),
                spec_radiance: input_view(&trace_pass_images.spec_radiance),
                normal_roughness: input_view(&trace_pass_images.normal_roughness),
                viewz: input_view(&trace_pass_images.viewz),
                mv: input_view(&trace_pass_images.mv),
                diff_out: input_view(&trace_pass_images.denoised_diff),
                spec_out: input_view(&trace_pass_images.denoised_spec),
            };

            task_graph
                .task_node_mut(denoise_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DenoiseTask>()
                .unwrap()
                .instance = Some(nrd.clone());
            task_graph
                .task_node_mut(denoise_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DenoiseTask>()
                .unwrap()
                .inputs = Some(inputs);
        }

        let region = RegionRenderContext {
            camera: capture_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
                view_prev: glam::Mat4::IDENTITY.to_cols_array_2d(),
                proj_prev: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
            scene: default_scene(),
            swapchain_storage_image_ids,
            t_image_storage_id: StorageImageId::INVALID,
            diff_radiance_image_id: trace_pass_images.diff_radiance.storage_id,
            spec_radiance_image_id: trace_pass_images.spec_radiance.storage_id,
            normal_roughness_image_id: trace_pass_images.normal_roughness.storage_id,
            viewz_image_id: trace_pass_images.viewz.storage_id,
            mv_image_id: trace_pass_images.mv.storage_id,
            albedo_metal_image_id: trace_pass_images.albedo_metal.storage_id,
            denoised_diff_image_id: trace_pass_images.denoised_diff.storage_id,
            denoised_spec_image_id: trace_pass_images.denoised_spec.storage_id,
            denoiser_enabled: self.nrd.is_some(),
            nrd: NrdFrame::default(),
            ev: 0.0,
            mode: RenderMode::default(),
            frame_seed: 0,
        };

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            trace_pass_images,
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

                        let mut batch = self.gpu.resources.create_deferred_batch();

                        for &id in &rcx.region.swapchain_storage_image_ids {
                            batch.destroy_storage_image(id);
                        }

                        let t = &rcx.trace_pass_images;
                        for image in [
                            &t.diff_radiance,
                            &t.spec_radiance,
                            &t.normal_roughness,
                            &t.viewz,
                            &t.mv,
                            &t.albedo_metal,
                            &t.denoised_diff,
                            &t.denoised_spec,
                        ] {
                            batch.destroy_image(image.physical_id);
                            batch.destroy_storage_image(image.storage_id);
                        }

                        if let Some(old) = &self.nrd {
                            let (images, constants) = old.resource_ids();
                            for id in images {
                                batch.destroy_image(id);
                            }
                            batch.destroy_buffer(constants);
                        }

                        batch.enqueue();

                        rcx.region.swapchain_storage_image_ids =
                            window_size_dependent_setup(&self.gpu.resources, rcx.swapchain_id);

                        let swapchain_state = self.gpu.resources.swapchain(rcx.swapchain_id);
                        let extent = swapchain_state.images()[0].extent();
                        let physical =
                            create_trace_pass_images(&self.gpu.resources, extent[0], extent[1]);
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
                        rcx.region.denoised_diff_image_id =
                            rcx.trace_pass_images.denoised_diff.storage_id;
                        rcx.region.denoised_spec_image_id =
                            rcx.trace_pass_images.denoised_spec.storage_id;

                        self.nrd = match NrdInstance::new(&self.gpu, extent[0], extent[1]) {
                            Ok(instance) => {
                                self.nrd_clear_pending = true;
                                Some(Arc::new(instance))
                            }
                            Err(error) => {
                                eprintln!("denoiser disabled: {error}");
                                None
                            }
                        };
                        rcx.region.denoiser_enabled = self.nrd.is_some();

                        if let Some(nrd) = &self.nrd {
                            let input_view = |image: &TracePassImage| {
                                ImageView::new_default(
                                    &self.gpu.resources.image(image.physical_id).image().clone(),
                                )
                                .unwrap()
                            };
                            let inputs = NrdInputs {
                                diff_radiance: input_view(&rcx.trace_pass_images.diff_radiance),
                                spec_radiance: input_view(&rcx.trace_pass_images.spec_radiance),
                                normal_roughness: input_view(
                                    &rcx.trace_pass_images.normal_roughness,
                                ),
                                viewz: input_view(&rcx.trace_pass_images.viewz),
                                mv: input_view(&rcx.trace_pass_images.mv),
                                diff_out: input_view(&rcx.trace_pass_images.denoised_diff),
                                spec_out: input_view(&rcx.trace_pass_images.denoised_spec),
                            };

                            if let Some(task) = rcx
                                .task_graph
                                .task_node_mut(rcx.denoise_node_id)
                                .ok()
                                .and_then(|node| node.task_mut().downcast_mut::<DenoiseTask>())
                            {
                                task.instance = Some(nrd.clone());
                                task.inputs = Some(inputs);
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
                    rcx.trace_pass_images.denoised_diff.virtual_id =>
                        rcx.trace_pass_images.denoised_diff.physical_id,
                    rcx.trace_pass_images.denoised_spec.virtual_id =>
                        rcx.trace_pass_images.denoised_spec.physical_id,
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
        self.player_input.drain();

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

pub(crate) struct TracePassImage {
    pub virtual_id: Id<Image>,
    pub physical_id: Id<Image>,
    pub storage_id: StorageImageId,
}

pub(crate) struct TracePassImages {
    pub diff_radiance: TracePassImage,
    pub spec_radiance: TracePassImage,
    pub normal_roughness: TracePassImage,
    pub viewz: TracePassImage,
    pub mv: TracePassImage,
    pub albedo_metal: TracePassImage,
    pub denoised_diff: TracePassImage,
    pub denoised_spec: TracePassImage,
}

impl TracePassImages {
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
            denoised_diff: image(add(Format::R16G16B16A16_SFLOAT)),
            denoised_spec: image(add(Format::R16G16B16A16_SFLOAT)),
        }
    }

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
        self.denoised_diff.physical_id = physical.denoised_diff.physical_id;
        self.denoised_diff.storage_id = physical.denoised_diff.storage_id;
        self.denoised_spec.physical_id = physical.denoised_spec.physical_id;
        self.denoised_spec.storage_id = physical.denoised_spec.storage_id;
    }
}

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
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
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

    let denoised = |format: Format| {
        let physical_id = resources
            .create_image(
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [width, height, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
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
        denoised_diff: denoised(Format::R16G16B16A16_SFLOAT),
        denoised_spec: denoised(Format::R16G16B16A16_SFLOAT),
    }
}
