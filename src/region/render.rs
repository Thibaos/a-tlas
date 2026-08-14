//! The Region pipeline's GPU half: the
//! shared ray tracing pipeline and the per-frame render task that
//! ray-passes the swapchain storage images — with the capture raygen
//! (color + t-channel for the validator) or the production raygen (color
//! only; `t_image_id` pushed as INVALID and never dereferenced).
//! The pipeline builder is shared by both raygen stages;
//! the miss/intersection/closest-hit stages are the Region path's own
//! (shaders/region).
//!
//! All per-Region GPU state — voxel pools, procedural AABB BLASes, the
//! lattice-static instance set and the stable TLAS — lives in
//! [`RegionStore`](crate::region::residency::RegionStore). The render task
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
    query::QueryPool,
    shader::EntryPoint,
    swapchain::Swapchain,
    sync::PipelineStage,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
};

#[cfg(debug_assertions)]
use crate::debug;
#[cfg(debug_assertions)]
use crate::world::Vertex3DColor;
use crate::{
    app::GpuStack,
    measure::{
        FLIGHT_BEGIN_SLOT, FLIGHT_END_SLOT, TIMESTAMP_SLOT_COUNT, TRACE_BEGIN_SLOT, TRACE_END_SLOT,
    },
    region::residency::RegionStore,
};
#[cfg(debug_assertions)]
use vulkano::pipeline::graphics::viewport::Viewport;

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

pub(crate) mod closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/region/closest_hit.rchit",
        vulkan_version: "1.3"
    }
}

/// The Hull hit group's AABB intersection shader (debug builds only — the
/// Hull mode has no surface in release).
#[cfg(debug_assertions)]
pub(crate) mod hull_intersect {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "intersection",
        path: "shaders/region/hull.rint",
        vulkan_version: "1.3"
    }
}

/// The Hull hit group's AABB closest-hit shader (debug builds only).
#[cfg(debug_assertions)]
pub(crate) mod hull_closest_hit {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "closesthit",
        path: "shaders/region/hull.rchit",
        vulkan_version: "1.3"
    }
}

/// The Render mode: what a primary ray resolves into (CONTEXT.md).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderMode {
    /// The DDA commits the surface voxel, shaded from the Palette. The
    /// default, and the only mode in release builds.
    #[default]
    Voxel = 0,
    /// Each Micro-chunk's trimmed AABB is the surface, colored by a
    /// coordinate hash, with no DDA. Debug builds only.
    #[cfg(debug_assertions)]
    Hull = 1,
}

/// The per-frame world the Region render task reads (shared by the
/// validator's graph and the app's graph).
///
/// The debug-overlay fields are app-only (the validator's graph never draws
/// the overlay, but builds the same world type): the app's per-frame debug
/// lines, push constants, and viewport, behind `debug_assertions`.
pub struct RegionRenderContext {
    pub camera: capture_raygen::Camera,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    /// Validation only: the capture raygen additionally writes payload.t
    /// here; the production raygen passes INVALID and never dereferences it
    pub t_image_storage_id: StorageImageId,
    /// The Render mode written into the push constants every frame: the
    /// production raygen's shader-binding-table record offset (0 = Voxel,
    /// 1 = Hull). Always Voxel in release and in the validator; the app
    /// toggles it on TAB in debug builds.
    pub mode: RenderMode,
    #[cfg(debug_assertions)]
    pub debug_lines: Vec<Vertex3DColor>,
    #[cfg(debug_assertions)]
    pub debug_constant_data: debug::shader::vert::PushConstants,
    #[cfg(debug_assertions)]
    pub viewport: Viewport,
}

pub struct RegionRenderTask {
    swapchain_id: Id<Swapchain>,
    camera_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    camera_storage_id: StorageBufferId,
    palette_storage_id: StorageBufferId,
    region_table_storage_id: StorageBufferId,
    /// Debug-only: the bindless id of the region -> AABB-buffer table (the
    /// Hull intersection shader's lookup). Absent in release, where the push
    /// constant carries INVALID and no Hull shader dereferences it.
    #[cfg(debug_assertions)]
    aabb_table_storage_id: StorageBufferId,
    acceleration_structure_id: AccelerationStructureId,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    /// Lifetime anchors: the TLAS instances reference the resident BLASes by
    /// device address only, so the task keeps them alive for the pass. (The
    /// store keeps every resident and free-listed BLAS alive regardless; this
    /// is a build-time snapshot, so a BLAS replaced by capacity growth stays
    /// alive for the whole pass — harmless retention.)
    #[allow(dead_code)]
    blases: Vec<Arc<AccelerationStructure>>,
    /// The measurement pool: when attached, the
    /// frame's GPU time is attributed per stage — the pool is reset at the
    /// top of the frame, then flight begin / trace begin-end / flight end
    /// timestamps are written around the node and the `trace_rays` call
    /// (slot layout in [`crate::measure`]). `None` (the validator's path)
    /// records nothing — the harness's captured frames are bit-identical.
    timestamps: Option<Arc<QueryPool>>,
}

impl RegionRenderTask {
    /// Builds the shared ray tracing pipeline around the given raygen stage
    /// (capture for the validator, production for the app) and binds the
    /// store's buffers. The store outlives the task, so its ids stay valid
    /// for every frame of the pass (residency rebuilds rewrite the buffers
    /// in place).
    ///
    /// `timestamps`: the measurement pool, when
    /// the app measures — the task records per-stage timestamps (flight /
    /// trace_rays) around the node. The validator passes `None`.
    pub fn new(
        gpu: &GpuStack,
        store: &RegionStore,
        virtual_swapchain_id: Id<Swapchain>,
        raygen: &EntryPoint,
        timestamps: Option<Arc<QueryPool>>,
    ) -> Self {
        let pipeline = {
            let miss = unsafe {
                miss::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
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
            palette_storage_id: store.palette_storage_id,
            region_table_storage_id: store.region_table_storage_id,
            #[cfg(debug_assertions)]
            aabb_table_storage_id: store.aabb_table_storage_id,
            acceleration_structure_id: store.acceleration_structure_id,
            shader_binding_table,
            pipeline,
            blases: store.blases(),
            timestamps,
        }
    }

    /// The instance buffer the TLAS reads (declared in the task graph).
    pub fn instance_buffer_id(&self) -> Id<Buffer> {
        self.instance_buffer_id
    }

    /// The bindless id of the debug-only region -> AABB-buffer table (the
    /// Hull intersection shader's lookup). INVALID in release, where no Hull
    /// shader exists and the push-constant field is never dereferenced.
    fn aabb_table_storage_id(&self) -> StorageBufferId {
        #[cfg(debug_assertions)]
        {
            self.aabb_table_storage_id
        }
        #[cfg(not(debug_assertions))]
        {
            StorageBufferId::INVALID
        }
    }
}

/// Builds the shared ray tracing pipeline (raygen + miss + procedural hit
/// group) around the given raygen entry point. The production task and the
/// validator's capture task differ only in which raygen they pass — the
/// miss/intersection/closest-hit stages are the Region path's own, so this
/// is the one pipeline builder for the whole renderer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_ray_tracing_pipeline(
    gpu: &GpuStack,
    raygen: &EntryPoint,
    miss: &EntryPoint,
    intersection: &EntryPoint,
    closest_hit: &EntryPoint,
) -> Arc<RayTracingPipeline> {
    let bcx = gpu.resources.bindless_context().unwrap();

    // Debug builds load the Hull hit group's shaders here (function scope, so
    // the entry points outlive the pipeline create below). Release builds
    // have no Hull surface: the stages/groups stay the DDA's three groups.
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

    // `mut` only because debug builds push the Hull stages/groups; release
    // builds leave them as the DDA's three groups.
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

    // The second procedural hit group (index 1; the DDA's is 0): the Hull
    // AABB intersection + closest-hit shaders, siblings of the DDA's in the
    // same pipeline. The production raygen selects it via sbtRecordOffset =
    // mode, so a TAB toggle never rebuilds the pipeline.
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
        // Measurement: reset the pool (queries
        // must be reset between uses — Vulkan spec, queries.adoc) and write
        // the flight begin as the first command of the node.
        if let Some(pool) = &self.timestamps {
            unsafe { cbf.as_raw().reset_query_pool(pool, 0, TIMESTAMP_SLOT_COUNT) };
            unsafe {
                cbf.as_raw()
                    .write_timestamp(pool, FLIGHT_BEGIN_SLOT, PipelineStage::AllCommands)
            };
        }

        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        unsafe { cbf.update_buffer(self.camera_buffer_id, 0, &rcx.camera) };

        // vkCmdUpdateBuffer's writes are not tracked by the taskgraph, so
        // make them visible to the ray pass explicitly.
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
                    region_table_buffer_id: self.region_table_storage_id,
                    aabb_table_buffer_id: self.aabb_table_storage_id(),
                    mode: rcx.mode as u32,
                },
            )
        };

        unsafe {
            cbf.bind_pipeline(&self.pipeline);
        }

        if let Some(pool) = &self.timestamps {
            unsafe {
                cbf.as_raw().write_timestamp(
                    pool,
                    TRACE_BEGIN_SLOT,
                    PipelineStage::RayTracingShader,
                )
            };
        }

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) };

        if let Some(pool) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(pool, TRACE_END_SLOT, PipelineStage::RayTracingShader)
            };
            // The flight end: the last command of the production path's
            // frame work (the app-only debug overlay, when present, draws
            // after — excluded from "flight").
            unsafe {
                cbf.as_raw()
                    .write_timestamp(pool, FLIGHT_END_SLOT, PipelineStage::AllCommands)
            };
        }

        Ok(())
    }
}
