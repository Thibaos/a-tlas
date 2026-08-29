//! The frame pipeline: swapchain, task graph, and NRD lifecycle as one owner
//! of the frame sequence — construction, resize recreation, and per-frame
//! execution. The app reports events and per-frame inputs; the sequence
//! itself lives here.

use std::sync::Arc;

use vulkano::{
    VulkanError,
    image::{ImageFormatInfo, ImageUsage},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    descriptor_set::StorageImageId,
    graph::{CompileInfo, ExecutableTaskGraph, ExecuteError, NodeId, ResourceMap, TaskGraph},
    resource::{AccessTypes, ImageLayoutType},
};
use winit::window::Window;

use crate::core::render::{
    composite::{CompositeTask, create_composite_pipeline},
    frame_images::FrameImages,
    gpu::{GpuDesc, MIN_SWAPCHAIN_IMAGES},
    nrd::{DenoiseTask, NrdInstance},
    region::{
        residency::RegionStore,
        task::{
            NrdFrame, RegionRenderContext, RegionRenderTask, RenderMode, default_ev,
            default_scene, production_raygen,
        },
    },
};

pub struct FramePipeline {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    virtual_swapchain_id: Id<Swapchain>,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<RegionRenderContext>,
    frame_images: FrameImages,
    denoise_node_id: NodeId,
    region: RegionRenderContext,
    nrd: Option<Arc<NrdInstance>>,
}

impl FramePipeline {
    pub fn new(gpu: &GpuDesc, window: Arc<Window>, store: &RegionStore) -> Self {
        let surface = Surface::from_window(&gpu.instance, &window).unwrap();

        let window_size = window.inner_size();

        let swapchain_id = {
            let surface_capabilities = gpu
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            let (image_format, image_color_space) = gpu
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()
                .into_iter()
                .find(|(format, _)| {
                    gpu.device
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

            gpu.resources
                .create_swapchain(
                    &surface,
                    &SwapchainCreateInfo {
                        present_mode: PresentMode::Immediate,
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

        let mut task_graph = TaskGraph::new(&gpu.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        let mut frame_images = FrameImages::declare(&mut task_graph);
        let extent = gpu.resources.swapchain(swapchain_id).images()[0].extent();
        frame_images.recreate(&gpu.resources, swapchain_id, extent);
        let nrd = NrdInstance::recreate(None, gpu, extent);

        let raygen = unsafe {
            production_raygen::load(&gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

        let rt_pass = RegionRenderTask::new(gpu, store, virtual_swapchain_id, &raygen, true);
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
        task_graph.add_edge(denoise_node_id, composite_node_id).unwrap();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&gpu.graphics_queue],
                present_queue: Some(&gpu.graphics_queue),
                flight_id: gpu.graphics_flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        {
            let composite_pipeline = create_composite_pipeline(gpu);
            task_graph
                .task_node_mut(composite_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<CompositeTask>()
                .unwrap()
                .pipeline = Some(composite_pipeline);
        }

        if let Some(nrd) = nrd.clone() {
            let task = task_graph
                .task_node_mut(denoise_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<DenoiseTask>()
                .unwrap();
            task.instance = Some(nrd.clone());
            task.inputs = Some(frame_images.nrd_inputs(&gpu.resources));
        }

        let mut region = RegionRenderContext {
            camera: production_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
                view_prev: glam::Mat4::IDENTITY.to_cols_array_2d(),
                proj_prev: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
            scene: default_scene(),
            swapchain_storage_image_ids: Vec::new(),
            diff_radiance_image_id: StorageImageId::INVALID,
            spec_radiance_image_id: StorageImageId::INVALID,
            normal_roughness_image_id: StorageImageId::INVALID,
            viewz_image_id: StorageImageId::INVALID,
            mv_image_id: StorageImageId::INVALID,
            denoised_diff_image_id: StorageImageId::INVALID,
            denoised_spec_image_id: StorageImageId::INVALID,
            validation_image_id: StorageImageId::INVALID,
            denoiser_enabled: nrd.is_some(),
            nrd: NrdFrame::default(),
            albedo_metal_image_id: StorageImageId::INVALID,
            ev: default_ev(),
            mode: RenderMode::default(),
            frame_seed: 0,
        };

        frame_images.bind_into(&mut region);

        Self {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            frame_images,
            denoise_node_id,
            region,
            nrd,
        }
    }

    pub const fn window(&self) -> &Arc<Window> {
        &self.window
    }

    pub const fn region_mut(&mut self) -> &mut RegionRenderContext {
        &mut self.region
    }

    pub const fn denoiser_enabled(&self) -> bool {
        self.nrd.is_some()
    }

    pub const fn request_recreate(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn recreate_if_needed(&mut self, gpu: &GpuDesc) -> bool {
        if !self.recreate_swapchain {
            return false;
        }

        let window = self.window.clone();

        self.swapchain_id = gpu
            .resources
            .recreate_swapchain(self.swapchain_id, |create_info| SwapchainCreateInfo {
                image_extent: window.inner_size().into(),
                ..create_info.clone()
            })
            .expect("failed to recreate swapchain");

        let extent = gpu.resources.swapchain(self.swapchain_id).images()[0].extent();

        self.nrd = NrdInstance::recreate(self.nrd.take(), gpu, extent);

        self.frame_images
            .recreate(&gpu.resources, self.swapchain_id, extent);
        self.frame_images.bind_into(&mut self.region);
        self.region.denoiser_enabled = self.nrd.is_some();

        if let Some(nrd) = &self.nrd {
            if let Some(task) = self
                .task_graph
                .task_node_mut(self.denoise_node_id)
                .ok()
                .and_then(|node| node.task_mut().downcast_mut::<DenoiseTask>())
            {
                task.instance = Some(nrd.clone());
                task.inputs = Some(self.frame_images.nrd_inputs(&gpu.resources));
            }
        }

        self.recreate_swapchain = false;

        true
    }

    pub fn execute(&mut self) {
        let mut map = ResourceMap::new(&self.task_graph).unwrap();
        map.insert(self.virtual_swapchain_id, self.swapchain_id).unwrap();

        for (virtual_id, physical_id) in self.frame_images.resource_pairs() {
            map.insert(virtual_id, physical_id).unwrap();
        }

        let window = self.window.clone();

        let result = unsafe {
            self.task_graph
                .execute(map, &self.region, move || window.pre_present_notify())
        };

        match result {
            Ok(()) => {}
            Err(ExecuteError::Swapchain {
                error: VulkanError::OutOfDate,
                ..
            }) => self.recreate_swapchain = true,
            Err(error) => panic!("failed to execute next frame: {error:?}"),
        }
    }
}
