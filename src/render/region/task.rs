use std::sync::Arc;

use anyhow::Context;
use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::Buffer,
    pipeline::{
        PipelineShaderStageCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
    },
    shader::EntryPoint,
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    descriptor_set::StorageImageId,
};

use crate::{
    render::context::RenderContext,
    render::region::residency::{RegionBindingsIds, RegionStore},
};

pub mod production_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/production.rgen",
        vulkan_version: "1.3"
    }
}

pub mod intersect {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "intersection",
        path: "shaders/voxel/intersect.rint",
        vulkan_version: "1.3"
    }
}

pub mod miss {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "miss",
        path: "shaders/miss.rmiss",
        vulkan_version: "1.3"
    }
}

pub mod closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/voxel/closest_hit.rchit",
        vulkan_version: "1.3"
    }
}

pub mod shadow_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/voxel/shadow_hit.rchit",
        vulkan_version: "1.3"
    }
}

pub mod hull_intersect {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "intersection",
        path: "shaders/hull/intersect.rint",
        vulkan_version: "1.3"
    }
}

#[cfg(debug_assertions)]
pub mod hull_closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/hull/closest_hit.rchit",
        vulkan_version: "1.3"
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderMode {
    #[default]
    Voxel = 0,
    #[cfg(debug_assertions)]
    Hull = 1,
    #[cfg(debug_assertions)]
    Normal = 2,
}

pub struct RegionRenderContext {
    pub camera: production_raygen::Camera,
    pub scene: production_raygen::Scene,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    pub color_image_id: StorageImageId,
    pub delta_time: f32,
    pub mode: RenderMode,
}

pub struct RegionRenderTask {
    swapchain_id: Id<Swapchain>,
    bindings: RegionBindingsIds,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    _blases: Vec<Arc<AccelerationStructure>>,
}

impl RegionRenderTask {
    pub fn new(
        gpu: &RenderContext,
        store: &RegionStore,
        virtual_swapchain_id: Id<Swapchain>,
        raygen: &EntryPoint,
    ) -> anyhow::Result<Self> {
        let pipeline = {
            let miss = unsafe {
                miss::load(&gpu.device)?
                    .entry_point("main")
                    .context("main entry point not found for miss shader")?
            };

            let intersection = unsafe {
                intersect::load(&gpu.device)?
                    .entry_point("main")
                    .context("main entry point not found for intersect shader")?
            };

            let closest_hit = unsafe {
                closest_hit::load(&gpu.device)?
                    .entry_point("main")
                    .context("main entry point not found for closest hit shader")?
            };

            let shadow_hit = unsafe {
                shadow_hit::load(&gpu.device)?
                    .entry_point("main")
                    .context("main entry point not found for shadow hit shader")?
            };

            build_ray_tracing_pipeline(gpu, raygen, &miss, &intersection, &closest_hit, &shadow_hit)
        }?;

        let shader_binding_table = ShaderBindingTable::new(&gpu.memory_allocator, &pipeline)?;

        Ok(Self {
            swapchain_id: virtual_swapchain_id,
            bindings: store.bindings,
            shader_binding_table,
            pipeline,
            _blases: store.blases(),
        })
    }

    pub const fn instance_buffer_id(&self) -> Id<Buffer> {
        self.bindings.instance_buffer
    }
}

pub const fn default_scene() -> production_raygen::Scene {
    production_raygen::Scene {
        sky_knots: [0.15, 0.6, 1.2, 0.0],
    }
}

fn build_ray_tracing_pipeline(
    gpu: &RenderContext,
    raygen: &EntryPoint,
    miss: &EntryPoint,
    intersection: &EntryPoint,
    closest_hit: &EntryPoint,
    shadow_hit: &EntryPoint,
) -> anyhow::Result<Arc<RayTracingPipeline>> {
    let bcx = gpu
        .resources
        .bindless_context()
        .context("bindless context not found")?;

    let hull_intersection = unsafe {
        hull_intersect::load(&gpu.device)?
            .entry_point("main")
            .context("main entry point not found for hull intersection shader")?
    };

    #[cfg(debug_assertions)]
    let hull_closest_hit = unsafe {
        hull_closest_hit::load(&gpu.device)?
            .entry_point("main")
            .context("main entry point not found for hull closest hit shader")?
    };

    let (stages, groups) = {
        const RAYGEN_INDEX: u32 = 0;
        const MISS_INDEX: u32 = 1;
        const DEFAULT_INTERSECTION_INDEX: u32 = 2;
        const COARSE_INTERSECTION_INDEX: u32 = 3;
        const DEFAULT_CHIT_INDEX: u32 = 4;
        const SHADOW_CHIT_INDEX: u32 = 5;
        #[cfg(debug_assertions)]
        const HULL_INTERSECTION_INDEX: u32 = 6;
        #[cfg(debug_assertions)]
        const HULL_CHIT_INDEX: u32 = 7;

        let raygen = PipelineShaderStageCreateInfo::new(raygen);
        let miss = PipelineShaderStageCreateInfo::new(miss);
        let default_intersection = PipelineShaderStageCreateInfo::new(intersection);
        let coarse_intersection = PipelineShaderStageCreateInfo::new(&hull_intersection);
        let default_closest_hit = PipelineShaderStageCreateInfo::new(closest_hit);
        let shadow_closest_hit = PipelineShaderStageCreateInfo::new(shadow_hit);

        #[cfg(not(debug_assertions))]
        let stages = vec![
            raygen,
            miss,
            default_intersection,
            coarse_intersection,
            default_closest_hit,
            shadow_closest_hit,
        ];

        #[cfg(debug_assertions)]
        let stages = vec![
            raygen,
            miss,
            default_intersection,
            coarse_intersection,
            default_closest_hit,
            shadow_closest_hit,
            PipelineShaderStageCreateInfo::new(&hull_intersection),
            PipelineShaderStageCreateInfo::new(&hull_closest_hit),
        ];

        let groups = vec![
            RayTracingShaderGroupCreateInfo::General {
                general_shader: RAYGEN_INDEX,
            },
            RayTracingShaderGroupCreateInfo::General {
                general_shader: MISS_INDEX,
            },
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                closest_hit_shader: Some(DEFAULT_CHIT_INDEX),
                any_hit_shader: None,
                intersection_shader: DEFAULT_INTERSECTION_INDEX,
            },
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                closest_hit_shader: Some(SHADOW_CHIT_INDEX),
                any_hit_shader: None,
                intersection_shader: DEFAULT_INTERSECTION_INDEX,
            },
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                closest_hit_shader: Some(SHADOW_CHIT_INDEX),
                any_hit_shader: None,
                intersection_shader: COARSE_INTERSECTION_INDEX,
            },
            #[cfg(debug_assertions)]
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                closest_hit_shader: Some(HULL_CHIT_INDEX),
                any_hit_shader: None,
                intersection_shader: HULL_INTERSECTION_INDEX,
            },
        ];

        (stages, groups)
    };

    let layout = bcx.pipeline_layout_from_stages(&stages)?;

    let base_info = RayTracingPipelineCreateInfo::new(&layout);

    let pipeline = RayTracingPipeline::new(
        &gpu.device,
        None,
        &RayTracingPipelineCreateInfo {
            stages: &stages,
            groups: &groups,
            max_pipeline_ray_recursion_depth: 1,
            ..base_info
        },
    )?;

    Ok(pipeline)
}

impl Task for RegionRenderTask {
    type World = RegionRenderContext;

    #[allow(clippy::as_conversions)]
    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id);

        let Some(image_index) = swapchain_state.current_image_index() else {
            eprintln!("swapchain has no current image");
            return Ok(());
        };

        let Ok(image_index) = usize::try_from(image_index) else {
            eprintln!("swapchain image index does not fit usize");
            return Ok(());
        };

        let Some(swapchain_first_image) = swapchain_state.images().first() else {
            eprintln!("swapchain has no images");
            return Ok(());
        };

        let extent = swapchain_first_image.extent();

        let Some(image_id) = rcx.swapchain_storage_image_ids.get(image_index) else {
            eprintln!("no storage image bound for the current swapchain image");
            return Ok(());
        };

        unsafe { cbf.update_buffer(self.bindings.camera_buffer, 0, &rcx.camera) };
        unsafe { cbf.update_buffer(self.bindings.scene_buffer, 0, &rcx.scene) };

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                    dst_access: vulkano::sync::AccessFlags::SHADER_READ
                        | vulkano::sync::AccessFlags::SHADER_STORAGE_READ,
                    src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,
                    dst_stages: vulkano::sync::PipelineStages::RAY_TRACING_SHADER,
                    ..MemoryBarrier::default()
                }],
                ..DependencyInfo::default()
            })
        };

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &production_raygen::RegionPushConstants {
                    image_id: *image_id,
                    acceleration_structure_id: self.bindings.acceleration_structure,
                    camera_buffer_id: self.bindings.camera_storage,
                    palette_buffer_id: self.bindings.palette_storage,
                    scene_buffer_id: self.bindings.scene_storage,
                    region_table_buffer_id: self.bindings.region_table_storage,
                    aabb_table_buffer_id: self.bindings.aabb_table_storage,
                    mode: rcx.mode as u32,
                    color_image_id: rcx.color_image_id,
                },
            )
        };

        unsafe {
            cbf.bind_pipeline(&self.pipeline);
        }

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) };

        Ok(())
    }
}
