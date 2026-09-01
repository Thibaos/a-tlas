//! The frame's one owner: the Change queue, RegionStore, and Frame history
//! behind the swapchain, task graph, and NRD lifecycle. The app reports
//! events and per-frame inputs; the whole frame sequence — recreate,
//! resize, flight wait, store drain, history advance, execute — lives here.

use dot_vox::DotVoxData;
use glam::{IVec3, Mat4, camera::lh::proj::vulkan::perspective};
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
    cache_resolve::{CacheResolveTask, create_cache_resolve_pipeline},
    composite::{CompositeTask, create_composite_pipeline},
    frame_images::FrameImages,
    gpu::{GpuDesc, MIN_SWAPCHAIN_IMAGES},
    nrd::{DenoiseTask, NrdInstance, history::NrdHistory},
    region::{
        feed::RendererInput,
        residency::{CACHE_DIRTY_WORDS, RegionStore},
        task::{
            NrdFrame, RegionRenderContext, RegionRenderTask, RenderMode, default_ev, default_scene,
            production_raygen,
        },
    },
};
use crate::core::world::grid::region_id;
use crate::core::world::{World, snapshot::emit_snapshots};

// the camera's near/far are the ray pass's t-range (contract.glsl's RAY_T_MIN/MAX)
const PROJ_FOV: f32 = std::f32::consts::FRAC_PI_2;
const PROJ_NEAR: f32 = 0.01;
const PROJ_FAR: f32 = 10000.0;

// contract.glsl's CACHE_TABLE_ENTRIES, cross-checked by the pack contract test.
pub(crate) const CACHE_TABLE_ENTRIES: u32 = 1 << 23;

// contract.glsl's CACHE_EVENT_FRAMES, cross-checked by the pack contract test.
pub(crate) const CACHE_EVENT_FRAMES: u32 = 10;

fn scene_changed(a: &production_raygen::Scene, b: &production_raygen::Scene) -> bool {
    a.sun_dir != b.sun_dir || a.sky_knots != b.sky_knots || a.sun_disk != b.sun_disk
}

// The edit-frame bitset the resolve sweeps the table with: one bit per
// Region, region bits ride the key (06).
fn cache_dirty_bits(dirty: &[IVec3]) -> [u32; CACHE_DIRTY_WORDS] {
    let mut words = [0u32; CACHE_DIRTY_WORDS];

    for region in dirty {
        let id = region_id(*region) as usize;

        words[id / 32] |= 1 << (id % 32);
    }

    words
}

pub struct FrameInput {
    pub view: Mat4,
    pub resized: bool,
    pub next_mode: bool,
    pub toggle_denoiser: bool,
}

#[derive(Default)]
#[allow(dead_code)]
pub struct FrameReport {
    pub edited: bool,
    pub resized: bool,
}

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
    input: RendererInput,
    store: RegionStore,
    history: NrdHistory,
    prev_scene: production_raygen::Scene,
    cache_event_frames: u32,
}

impl FramePipeline {
    pub fn new(gpu: &GpuDesc, window: Arc<Window>, voxel_data: &DotVoxData, world: &World) -> Self {
        let input = RendererInput::new();
        input.submit_batch(emit_snapshots(world));

        let store = RegionStore::new(gpu, voxel_data, &input);
        let history = NrdHistory::new();

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

        let rt_pass = RegionRenderTask::new(gpu, &store, virtual_swapchain_id, &raygen, true);
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

        let mut resolve_node = task_graph.create_task_node(
            "CacheResolve",
            QueueFamilyType::Graphics,
            CacheResolveTask::new(store.bindings),
        );
        let resolve_node_id = resolve_node.build();

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

        task_graph.add_edge(rt_node_id, resolve_node_id).unwrap();
        task_graph
            .add_edge(resolve_node_id, denoise_node_id)
            .unwrap();
        task_graph
            .add_edge(denoise_node_id, composite_node_id)
            .unwrap();

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

        {
            let resolve_pipeline = create_cache_resolve_pipeline(gpu);
            task_graph
                .task_node_mut(resolve_node_id)
                .unwrap()
                .task_mut()
                .downcast_mut::<CacheResolveTask>()
                .unwrap()
                .pipeline = Some(resolve_pipeline);
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
            cache_state: store.initial_cache_state(gpu),
            cache_dirty: [0; CACHE_DIRTY_WORDS],
            swapchain_storage_image_ids: Vec::new(),
            diff_radiance_image_id: StorageImageId::INVALID,
            spec_radiance_image_id: StorageImageId::INVALID,
            normal_roughness_image_id: StorageImageId::INVALID,
            viewz_image_id: StorageImageId::INVALID,
            mv_image_id: StorageImageId::INVALID,
            denoised_diff_image_id: StorageImageId::INVALID,
            denoised_spec_image_id: StorageImageId::INVALID,
            validation_image_id: StorageImageId::INVALID,
            denoiser_active: nrd.is_some(),
            nrd: NrdFrame::default(),
            albedo_metal_image_id: StorageImageId::INVALID,
            disocclusion_mix_image_id: StorageImageId::INVALID,
            ev: default_ev(),
            mode: RenderMode::default(),
            frame_seed: 0,
            cache_resolve_dispatch: 0,
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
            input,
            store,
            history,
            prev_scene: default_scene(),
            cache_event_frames: 0,
        }
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

        self.nrd = NrdInstance::recreate(self.nrd.take(), gpu, extent);

        self.frame_images
            .recreate(&gpu.resources, self.swapchain_id, extent);
        self.frame_images.bind_into(&mut self.region);
        self.region.denoiser_active = self.nrd.is_some();

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

    pub fn run_frame(&mut self, gpu: &GpuDesc, input: FrameInput) -> FrameReport {
        self.recreate_swapchain |= input.resized;

        let extent = self.window.inner_size();
        let plan = frame_plan(self.recreate_swapchain, extent.width, extent.height);

        if plan.recreate {
            self.recreate_if_needed(gpu);
        }

        if plan.resized {
            self.history.resized();
        }

        if !plan.execute {
            return FrameReport::default();
        }

        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        if self.store.cache_stats_enabled() && self.region.cache_state.frame_index % 60 == 0 {
            let (lookups, fallbacks, touched, deposits, landed, live, young) =
                self.store.cache_stats_tick(gpu);

            println!(
                "cache stats: {lookups} lookups, {fallbacks} fallbacks, {touched} faces touched, \
                 {deposits} deposits, {landed} landed, {live} live, {young} young"
            );
        }

        let apply_report = self.store.apply(gpu, &self.input);
        let edited = !apply_report.dirty.is_empty();
        self.region.cache_dirty = cache_dirty_bits(&apply_report.dirty);

        #[cfg(debug_assertions)]
        if input.next_mode {
            self.region.mode = next_render_mode(self.region.mode, self.nrd.is_some());
        }

        if input.toggle_denoiser && self.nrd.is_some() {
            self.region.denoiser_active = !self.region.denoiser_active;

            if self.region.denoiser_active {
                self.history.request_clear();
            }

            println!(
                "denoiser: {}",
                if self.region.denoiser_active {
                    "on"
                } else {
                    "raw"
                }
            );
        }

        let aspect = extent.width as f32 / extent.height as f32;
        let proj = perspective(PROJ_FOV, aspect, PROJ_NEAR, PROJ_FAR);
        let prev = self.history.observe_camera(input.view, proj);

        self.region.camera = production_raygen::Camera {
            proj_inverse: proj.inverse().to_cols_array_2d(),
            view_inverse: input.view.inverse().to_cols_array_2d(),
            view_prev: prev.view.to_cols_array_2d(),
            proj_prev: prev.proj.to_cols_array_2d(),
        };

        self.region.frame_seed = self.region.frame_seed.wrapping_add(1);

        self.region.nrd = self.history.advance(edited, self.nrd.is_some());

        if plan.resized || self.region.nrd.clear {
            self.store.clear_cache_table(gpu);
        }

        if scene_changed(&self.region.scene, &self.prev_scene) {
            self.cache_event_frames = CACHE_EVENT_FRAMES;
        } else if self.cache_event_frames > 0 {
            self.cache_event_frames -= 1;
        }

        self.prev_scene = self.region.scene;
        self.region.cache_state.frame_index = self
            .region
            .cache_state
            .frame_index
            .wrapping_add(1);
        self.region.cache_state.event_frames = self.cache_event_frames;
        self.region.cache_resolve_dispatch = CACHE_TABLE_ENTRIES;

        self.execute();

        FrameReport {
            edited,
            resized: plan.recreate,
        }
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
    resized: bool,
    execute: bool,
}

fn frame_plan(recreate_requested: bool, width: u32, height: u32) -> FramePlan {
    let drawable = width > 0 && height > 0;

    FramePlan {
        recreate: drawable && recreate_requested,
        resized: drawable && recreate_requested,
        execute: drawable,
    }
}

#[cfg(debug_assertions)]
fn next_render_mode(mode: RenderMode, denoiser_present: bool) -> RenderMode {
    match mode {
        RenderMode::Voxel => RenderMode::Hull,
        RenderMode::Hull => RenderMode::Normal,
        RenderMode::Normal => {
            if denoiser_present {
                RenderMode::NrdValidation
            } else {
                eprintln!("render mode: NRD validation unavailable, denoiser inactive");
                RenderMode::Voxel
            }
        }
        RenderMode::NrdValidation => RenderMode::Voxel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recreate_plans_exactly_one_history_resize() {
        let plan = frame_plan(true, 1920, 1080);

        assert!(plan.recreate);
        assert!(plan.resized);
        assert!(plan.execute);
    }

    #[test]
    fn zero_extent_skips_the_frame_including_recreate() {
        let plan = frame_plan(true, 0, 1080);

        assert!(!plan.recreate);
        assert!(!plan.resized);
        assert!(!plan.execute);
    }

    #[test]
    fn ordinary_frame_executes_without_recreate() {
        let plan = frame_plan(false, 1920, 1080);

        assert!(!plan.recreate);
        assert!(!plan.resized);
        assert!(plan.execute);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn mode_cycle_gates_nrd_validation_on_denoiser_presence() {
        assert_eq!(next_render_mode(RenderMode::Voxel, true), RenderMode::Hull);
        assert_eq!(next_render_mode(RenderMode::Hull, true), RenderMode::Normal);
        assert_eq!(
            next_render_mode(RenderMode::Normal, true),
            RenderMode::NrdValidation
        );
        assert_eq!(
            next_render_mode(RenderMode::NrdValidation, true),
            RenderMode::Voxel
        );
        assert_eq!(
            next_render_mode(RenderMode::Normal, false),
            RenderMode::Voxel
        );
    }

    #[test]
    fn dirty_bits_flag_exactly_the_dirty_regions() {
        let dirty = vec![IVec3::new(0, 0, 0), IVec3::new(-8, 7, 3), IVec3::new(7, -8, -8)];
        let words = cache_dirty_bits(&dirty);

        let mut flagged = 0usize;

        for (id, word) in words.iter().enumerate() {
            for bit in 0..32 {
                if ((word >> bit) & 1) == 1 {
                    flagged += 1;

                    let region = (id * 32 + bit) as u32;

                    assert!(dirty.iter().any(|r| region_id(*r) == region));
                }
            }
        }

        assert_eq!(flagged, dirty.len());
    }
}
