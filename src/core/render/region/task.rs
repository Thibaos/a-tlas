//! The Region pipeline's GPU half: the
//! shared ray tracing pipeline and the per-frame render task that
//! ray-passes the swapchain storage images, with the capture raygen
//! (color + t-channel for the validator) or the production raygen (color
//! only; `t_image_id` pushed as INVALID and never dereferenced).
//! The pipeline builder is shared by both raygen stages;
//! the miss/intersection/closest-hit stages are the Region path's own
//! (shaders/region).
//!
//! All per-Region GPU state: voxel pools, procedural AABB BLASes, the
//! lattice-static instance set and the stable TLAS. Lives in
//! [`RegionStore`](crate::render::region::residency::RegionStore). The render task
//! only holds the ids the push constants and the task graph need; the store
//! keeps the buffers alive across rebuilds.

use std::sync::Arc;

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
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
};

use crate::core::{render::gpu::GpuDesc, render::region::residency::RegionStore};

pub(crate) mod capture_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/region/capture.rgen",
        vulkan_version: "1.3"
    }
}

pub(crate) mod production_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/region/production.rgen",
        vulkan_version: "1.3"
    }
}

pub(crate) mod intersect {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "intersection",
        path: "shaders/region/intersect.rint",
        vulkan_version: "1.3"
    }
}

pub(crate) mod miss {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "miss",
        path: "shaders/region/miss.rmiss",
        vulkan_version: "1.3"
    }
}

pub(crate) mod miss_sky {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "miss",
        path: "shaders/region/miss_sky.rmiss",
        vulkan_version: "1.3"
    }
}

pub(crate) mod closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/region/closest_hit.rchit",
        vulkan_version: "1.3"
    }
}

#[cfg(debug_assertions)]
pub(crate) mod hull_intersect {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "intersection",
        path: "shaders/region/hull.rint",
        vulkan_version: "1.3"
    }
}

#[cfg(debug_assertions)]
pub(crate) mod hull_closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/region/hull.rchit",
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
    Normal = 4,
}

#[derive(Clone, Copy)]
pub struct NrdFrame {
    pub view_to_clip: [f32; 16],
    pub view_to_clip_prev: [f32; 16],
    pub world_to_view: [f32; 16],
    pub world_to_view_prev: [f32; 16],
    pub frame_index: u32,
    pub reset: bool,
    pub clear: bool,
}

impl Default for NrdFrame {
    fn default() -> Self {
        Self {
            view_to_clip: [0.0; 16],
            view_to_clip_prev: [0.0; 16],
            world_to_view: identity_cols(),
            world_to_view_prev: identity_cols(),
            frame_index: 0,
            reset: false,
            clear: true,
        }
    }
}

fn identity_cols() -> [f32; 16] {
    glam::Mat4::IDENTITY.to_cols_array()
}

pub struct RegionRenderContext {
    pub camera: capture_raygen::Camera,
    pub scene: capture_raygen::Scene,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    pub t_image_storage_id: StorageImageId,
    pub diff_radiance_image_id: StorageImageId,
    pub spec_radiance_image_id: StorageImageId,
    pub normal_roughness_image_id: StorageImageId,
    pub viewz_image_id: StorageImageId,
    pub mv_image_id: StorageImageId,
    pub denoised_diff_image_id: StorageImageId,
    pub denoised_spec_image_id: StorageImageId,
    pub denoiser_enabled: bool,
    pub nrd: NrdFrame,
    pub albedo_metal_image_id: StorageImageId,
    pub ev: f32,
    pub mode: RenderMode,
    pub frame_seed: u32,
}

pub struct RegionRenderTask {
    swapchain_id: Id<Swapchain>,
    camera_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    camera_storage_id: StorageBufferId,
    palette_storage_id: StorageBufferId,
    scene_buffer_id: Id<Buffer>,
    scene_storage_id: StorageBufferId,
    material_table_storage_id: StorageBufferId,
    region_table_storage_id: StorageBufferId,
    aabb_table_storage_id: StorageBufferId,
    acceleration_structure_id: AccelerationStructureId,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    #[allow(dead_code)]
    blases: Vec<Arc<AccelerationStructure>>,
}

impl RegionRenderTask {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuDesc,
        store: &RegionStore,
        virtual_swapchain_id: Id<Swapchain>,
        raygen: &EntryPoint,
        sky_background: bool,
    ) -> Self {
        let pipeline = {
            let miss = if sky_background {
                unsafe {
                    miss_sky::load(&gpu.device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap()
                }
            } else {
                unsafe {
                    miss::load(&gpu.device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap()
                }
            };

            let intersection = unsafe {
                intersect::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };
            let closest_hit = unsafe {
                closest_hit::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };

            build_ray_tracing_pipeline(gpu, raygen, &miss, &intersection, &closest_hit)
        };

        let shader_binding_table =
            ShaderBindingTable::new(&gpu.memory_allocator, &pipeline).unwrap();

        Self {
            swapchain_id: virtual_swapchain_id,
            camera_buffer_id: store.camera_buffer_id,
            instance_buffer_id: store.instance_buffer_id,
            camera_storage_id: store.camera_storage_id,
            scene_buffer_id: store.scene_buffer_id,
            scene_storage_id: store.scene_storage_id,
            palette_storage_id: store.palette_storage_id,
            material_table_storage_id: store.material_table_storage_id,
            region_table_storage_id: store.region_table_storage_id,
            aabb_table_storage_id: store.aabb_table_storage_id,
            acceleration_structure_id: store.acceleration_structure_id,
            shader_binding_table,
            pipeline,
            blases: store.blases(),
        }
    }

    pub fn instance_buffer_id(&self) -> Id<Buffer> {
        self.instance_buffer_id
    }

    fn aabb_table_storage_id(&self) -> StorageBufferId {
        self.aabb_table_storage_id
    }
}

pub const E_SUN: f32 = 16.0;

pub fn default_ev() -> f32 {
    (std::f32::consts::PI / E_SUN).log2()
}

pub fn default_scene() -> capture_raygen::Scene {
    let sun_dir = glam::Vec3::new(0.45, 0.8, 0.35).normalize();
    let knots = [0.15, 0.6, 1.2];
    let cos_disk = (0.5_f32 * std::f32::consts::PI / 180.0).cos();
    let omega = 2.0 * std::f32::consts::PI * (1.0 - cos_disk);
    let l_disk = E_SUN / omega;

    capture_raygen::Scene {
        sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
        sky_knots: [knots[0], knots[1], knots[2], 0.0],
        sun_disk: [E_SUN, cos_disk, l_disk, 0.0],
    }
}

pub(crate) fn build_ray_tracing_pipeline(
    gpu: &GpuDesc,
    raygen: &EntryPoint,
    miss: &EntryPoint,
    intersection: &EntryPoint,
    closest_hit: &EntryPoint,
) -> Arc<RayTracingPipeline> {
    let bcx = gpu.resources.bindless_context().unwrap();

    #[cfg(debug_assertions)]
    let hull_intersection = unsafe {
        hull_intersect::load(&gpu.device)
            .unwrap()
            .entry_point("main")
            .unwrap()
    };

    #[cfg(debug_assertions)]
    let hull_closest_hit = unsafe {
        hull_closest_hit::load(&gpu.device)
            .unwrap()
            .entry_point("main")
            .unwrap()
    };

    #[cfg_attr(not(debug_assertions), allow(unused_mut))]
    let mut stages = vec![
        PipelineShaderStageCreateInfo::new(raygen),
        PipelineShaderStageCreateInfo::new(miss),
        PipelineShaderStageCreateInfo::new(intersection),
        PipelineShaderStageCreateInfo::new(closest_hit),
    ];

    #[cfg_attr(not(debug_assertions), allow(unused_mut))]
    let mut groups = vec![
        RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
        RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
        RayTracingShaderGroupCreateInfo::ProceduralHit {
            closest_hit_shader: Some(3),
            any_hit_shader: None,
            intersection_shader: 2,
        },
    ];

    #[cfg(debug_assertions)]
    {
        let hull_intersection_idx = stages.len() as u32;
        let hull_closest_hit_idx = hull_intersection_idx + 1;
        stages.push(PipelineShaderStageCreateInfo::new(&hull_intersection));
        stages.push(PipelineShaderStageCreateInfo::new(&hull_closest_hit));
        groups.push(RayTracingShaderGroupCreateInfo::ProceduralHit {
            closest_hit_shader: Some(hull_closest_hit_idx),
            any_hit_shader: None,
            intersection_shader: hull_intersection_idx,
        });
    }

    let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

    let base_info = RayTracingPipelineCreateInfo::new(&layout);

    RayTracingPipeline::new(
        &gpu.device,
        None,
        &RayTracingPipelineCreateInfo {
            stages: &stages,
            groups: &groups,
            max_pipeline_ray_recursion_depth: 1,
            ..base_info
        },
    )
    .unwrap()
}

impl Task for RegionRenderTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        unsafe { cbf.update_buffer(self.camera_buffer_id, 0, &rcx.camera) };
        unsafe { cbf.update_buffer(self.scene_buffer_id, 0, &rcx.scene) };

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                    dst_access: vulkano::sync::AccessFlags::SHADER_READ
                        | vulkano::sync::AccessFlags::SHADER_STORAGE_READ,
                    src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,
                    dst_stages: vulkano::sync::PipelineStages::RAY_TRACING_SHADER,
                    ..Default::default()
                }],
                ..Default::default()
            })
        };

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &capture_raygen::RegionPushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    t_image_id: rcx.t_image_storage_id,
                    acceleration_structure_id: self.acceleration_structure_id,
                    camera_buffer_id: self.camera_storage_id,
                    palette_buffer_id: self.palette_storage_id,
                    material_table_buffer_id: self.material_table_storage_id,
                    scene_buffer_id: self.scene_storage_id,
                    region_table_buffer_id: self.region_table_storage_id,
                    aabb_table_buffer_id: self.aabb_table_storage_id(),
                    mode: rcx.mode as u32,
                    frame_seed: rcx.frame_seed,
                    diff_radiance_image_id: rcx.diff_radiance_image_id,
                    spec_radiance_image_id: rcx.spec_radiance_image_id,
                    normal_roughness_image_id: rcx.normal_roughness_image_id,
                    viewz_image_id: rcx.viewz_image_id,
                    mv_image_id: rcx.mv_image_id,
                    albedo_metal_image_id: rcx.albedo_metal_image_id,
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
