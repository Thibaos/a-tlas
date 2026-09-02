use dot_vox::DotVoxData;
use glam::{Mat4, camera::lh::proj::vulkan::perspective};
use std::sync::Arc;

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
use winit::window::Window;

use crate::core::render::{
    composite::{CompositeTask, create_composite_pipeline},
    frame_images::FrameImages,
    gpu::{GpuDesc, MIN_SWAPCHAIN_IMAGES},
    region::{
        feed::RendererInput,
        residency::RegionStore,
        task::{
            RegionRenderContext, RegionRenderTask, RenderMode, default_scene, production_raygen,
        },
    },
};
use crate::core::world::{World, snapshot::emit_snapshots};

const PROJ_FOV: f32 = std::f32::consts::FRAC_PI_2;
const PROJ_NEAR: f32 = 0.01;
const PROJ_FAR: f32 = 10000.0;

pub struct FrameInput {
    pub view: Mat4,
    pub resized: bool,
    pub next_mode: bool,
    pub delta_time: f32,
}

pub struct FramePipeline {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    virtual_swapchain_id: Id<Swapchain>,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<RegionRenderContext>,
    frame_images: FrameImages,
    region: RegionRenderContext,
    input: RendererInput,
    store: RegionStore,
}

impl FramePipeline {
    pub fn new(
        gpu: &GpuDesc,
        window: Arc<Window>,
        voxel_data: &DotVoxData,
        world: &World,
    ) -> anyhow::Result<Self> {
        let input = RendererInput::new();
        input.submit_batch(emit_snapshots(world));

        let store = RegionStore::new(gpu, voxel_data, &input)?;

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

        let raygen = unsafe {
            production_raygen::load(&gpu.device)
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

        let rt_pass = RegionRenderTask::new(gpu, &store, virtual_swapchain_id, &raygen);
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

        task_graph.add_edge(rt_node_id, composite_node_id).unwrap();

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

            let task = task_graph
                .task_node_mut(composite_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<CompositeTask>()
                .unwrap();

            task.pipeline = Some(composite_pipeline);
        }

        let mut region = RegionRenderContext {
            camera: production_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
            },
            scene: default_scene(),
            swapchain_storage_image_ids: Vec::new(),
            color_image_id: StorageImageId::INVALID,
            delta_time: 0.0,
            mode: RenderMode::default(),
        };

        frame_images.bind_into(&mut region);

        Ok(Self {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            frame_images,
            region,
            input,
            store,
        })
    }

    fn recreate_if_needed(&mut self, gpu: &GpuDesc) -> bool {
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

        self.frame_images
            .recreate(&gpu.resources, self.swapchain_id, extent);
        self.frame_images.bind_into(&mut self.region);

        self.recreate_swapchain = false;

        true
    }

    pub fn run_frame(&mut self, gpu: &GpuDesc, input: FrameInput) -> anyhow::Result<()> {
        self.recreate_swapchain |= input.resized;

        let extent = self.window.inner_size();
        let plan = frame_plan(self.recreate_swapchain, extent.width, extent.height);

        if plan.recreate {
            self.recreate_if_needed(gpu);
        }

        if !plan.execute {
            return Ok(());
        }

        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        self.store.apply(gpu, &self.input)?;

        #[cfg(debug_assertions)]
        if input.next_mode {
            self.region.mode = next_render_mode(self.region.mode);
        }

        let aspect = extent.width as f32 / extent.height as f32;
        let proj = perspective(PROJ_FOV, aspect, PROJ_NEAR, PROJ_FAR);
        self.region.camera = production_raygen::Camera {
            proj_inverse: proj.inverse().to_cols_array_2d(),
            view_inverse: input.view.inverse().to_cols_array_2d(),
        };

        self.region.delta_time = input.delta_time;

        self.execute();

        Ok(())
    }

    fn execute(&mut self) {
        let mut map = ResourceMap::new(&self.task_graph).unwrap();
        map.insert(self.virtual_swapchain_id, self.swapchain_id)
            .unwrap();

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

struct FramePlan {
    recreate: bool,
    execute: bool,
}

fn frame_plan(recreate_requested: bool, width: u32, height: u32) -> FramePlan {
    let drawable = width > 0 && height > 0;

    FramePlan {
        recreate: drawable && recreate_requested,
        execute: drawable,
    }
}

#[cfg(debug_assertions)]
fn next_render_mode(mode: RenderMode) -> RenderMode {
    match mode {
        RenderMode::Voxel => RenderMode::Hull,
        RenderMode::Hull => RenderMode::Normal,
        RenderMode::Normal => RenderMode::Voxel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recreate_plans_a_recreate() {
        let plan = frame_plan(true, 1920, 1080);

        assert!(plan.recreate);
        assert!(plan.execute);
    }

    #[test]
    fn zero_extent_skips_the_frame_including_recreate() {
        let plan = frame_plan(true, 0, 1080);

        assert!(!plan.recreate);
        assert!(!plan.execute);
    }

    #[test]
    fn ordinary_frame_executes_without_recreate() {
        let plan = frame_plan(false, 1920, 1080);

        assert!(!plan.recreate);
        assert!(plan.execute);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn mode_cycle_returns_to_voxel() {
        assert_eq!(next_render_mode(RenderMode::Voxel), RenderMode::Hull);
        assert_eq!(next_render_mode(RenderMode::Hull), RenderMode::Normal);
        assert_eq!(next_render_mode(RenderMode::Normal), RenderMode::Voxel);
    }
}
