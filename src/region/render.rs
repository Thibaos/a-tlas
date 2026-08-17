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
    shader::{EntryPoint, SpecializationConstant},
    swapchain::Swapchain,
    sync::PipelineStage,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, FillBufferInfo, MemoryBarrier, RecordingCommandBuffer},
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
};

use crate::{
    app::GpuStack,
    measure::{
        CounterBuffer, FLIGHT_BEGIN_SLOT, FLIGHT_END_SLOT, TIMESTAMP_SLOT_COUNT, TRACE_BEGIN_SLOT,
        TRACE_END_SLOT,
    },
    region::residency::RegionStore,
};

pub(crate) mod capture_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/region/capture.rgen",
        vulkan_version: "1.3"
    }
}

pub(crate) mod production_raygen {
    // Debug builds compile the Ray latency branch (clockRealtime); release
    // builds omit it so the raygen carries no ShaderClock capability and the
    // device needs no shader_device_clock feature.
    #[cfg(debug_assertions)]
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/region/production.rgen",
        define: [("ATLAS_RT_RAY_LATENCY", "1")],
        vulkan_version: "1.3"
    }
    #[cfg(not(debug_assertions))]
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

/// The production miss shader (ticket 06): the Procedural sky's radiance —
/// the gradient at the ray's world direction (the disk is the camera's
/// direct view, added by the raygen's primary-miss branch). The capture
/// pipeline keeps the black `miss` module, so the byte-exact validator is
/// unchanged.
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

/// The Render mode: what the ray pass paints each pixel with (CONTEXT.md) —
/// surface identity (Voxel, Hull) or a diagnostic quantity (Ray latency,
/// hull-crossed). The diagnostic modes are debug-build-only.
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
    /// Each pixel is colored by its ray's wall-clock lifetime (a
    /// `clockRealtime` delta around `traceRayEXT`). Latency, not cost. Debug
    /// builds only.
    #[cfg(debug_assertions)]
    RayLatency = 2,
    /// Each pixel is colored by how many Micro-chunk hulls its ray entered
    /// (the spatial form of the march-and-miss counter's `hull_crossed` word).
    /// Debug builds only.
    #[cfg(debug_assertions)]
    HullCrossed = 3,
    /// Each pixel is colored by its hit's geometric surface normal (ticket
    /// 04, ADR 0009): -1..1 mapped to 0..1 per channel, so voxel faces paint
    /// by their axis (x red, y green, z blue; + side bright, - side dark),
    /// background gray. Traces the DDA hit group like Voxel (the normal
    /// rides the payload); Debug builds only.
    #[cfg(debug_assertions)]
    Normal = 4,
}

/// The per-pixel hull-crossed count buffer (debug builds): one uint per ray
/// pass pixel, incremented by the DDA intersection shader's atomicAdd at
/// slab-pass when the hull-crossed mode selects its hit group. Reset with a
/// fill each frame; read on the GPU by the heatmap overlay node. The count is
/// the same quantity as the `--measure` counter's `hull_crossed` word: a lower
/// bound on traversal work (hulls entered, not rejected AABB tests) and an
/// upper bound per-pixel (Vulkan may invoke intersection shaders redundantly).
/// The validator and release paths pass `None` and never attach it.
#[derive(Clone, Copy)]
pub struct HullCrossedCounter {
    /// The buffer id the render task fills (reset) each frame.
    pub buffer_id: Id<Buffer>,
    /// The bindless id pushed into the shader's push constants.
    pub storage_id: StorageBufferId,
}

/// The per-frame world the Region render task reads (shared by the
/// validator's graph and the app's graph).
pub struct RegionRenderContext {
    pub camera: capture_raygen::Camera,
    /// The analytic lights' constants (ticket 06): the Sun (direction +
    /// illuminance), the Procedural sky's μ-gradient knots, and the disk.
    /// Written into the Scene buffer every frame (tunable); the capture
    /// path never reads them (its miss shader is black), so the byte-exact
    /// validator is unchanged.
    pub scene: capture_raygen::Scene,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    /// Validation only: the capture raygen additionally writes payload.t
    /// here; the production raygen passes INVALID and never dereferences it
    pub t_image_storage_id: StorageImageId,
    /// Path-tracing output contract (ADR 0007): the trace pass's noisy
    /// radiance pair and auxiliary guide buffers, written by the production
    /// raygen in Voxel mode (diffuse+specular radiance with in-lobe hit
    /// distance in alpha, normal+roughness, linear viewZ, backward motion
    /// vectors, albedo+metalness). The composite node exposes them (and, from
    /// ticket 08, the Denoise pass consumes them). The validator pushes
    /// INVALID for all six — the capture raygen never writes them.
    pub diff_radiance_image_id: StorageImageId,
    pub spec_radiance_image_id: StorageImageId,
    pub normal_roughness_image_id: StorageImageId,
    pub viewz_image_id: StorageImageId,
    pub mv_image_id: StorageImageId,
    pub albedo_metal_image_id: StorageImageId,
    /// Manual exposure in EV stops, applied by the composite node
    /// (ADR 0007). App-only; the validator leaves it at 0.
    pub ev: f32,
    /// The Render mode written into the push constants every frame: the
    /// production raygen's shader-binding-table record offset (0 = Voxel,
    /// 1 = Hull; Voxel, Ray latency and the normal heatmap all trace the
    /// DDA hit group 0). Always Voxel in release and in the validator; the
    /// app toggles it on TAB in debug builds.
    pub mode: RenderMode,
    /// The path-tracing RNG's per-frame seed (ticket 05, ADR 0010): the app
    /// increments it every frame so consecutive frames decorrelate for the
    /// Denoise pass's temporal accumulation. The validator's capture raygen
    /// never reads it and pushes 0.
    pub frame_seed: u32,
}

pub struct RegionRenderTask {
    swapchain_id: Id<Swapchain>,
    camera_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    camera_storage_id: StorageBufferId,
    palette_storage_id: StorageBufferId,
    /// The Scene buffer (ticket 06): the analytic lights' constants —
    /// updated every frame from the context like the camera; pushed into
    /// the bindless Scene binding (the production sky miss shader and the
    /// production raygen read it; the capture path never dereferences it).
    scene_buffer_id: Id<Buffer>,
    scene_storage_id: StorageBufferId,
    /// The bindless Material table (ADR 0008): pushed every frame; the DDA
    /// closest-hit reads it for the surface color (albedo == palette, so the
    /// capture path stays byte-identical) and the production raygen reads it
    /// through the payload's hit_kind in Voxel mode (real metalness +
    /// emission-as-albedo-light).
    material_table_storage_id: StorageBufferId,
    region_table_storage_id: StorageBufferId,
    /// The bindless id of the region -> AABB-buffer table (the DDA's and the
    /// debug Hull shader's lookup). The push constant carries it every frame.
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
    /// The march-and-miss counter buffer (reset with a fill + pushed each
    /// frame only when measuring). `None` (the validator's path) records no
    /// fill and pushes an INVALID id — and the intersection shader is
    /// specialized with COUNTER_ENABLED = false, folding the atomicAdds away so
    /// the validator/default pipelines render byte-identical output.
    counter: Option<CounterBuffer>,
    /// The per-pixel hull-crossed count buffer (debug builds only): reset with
    /// a fill, pushed into the intersection shader's push constants, and read
    /// by the heatmap overlay node. `None` (the validator's and release paths)
    /// pushes INVALID and never attaches the hull-crossed hit group.
    hull_crossed: Option<HullCrossedCounter>,
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
    /// `hull_crossed`: the per-pixel hull-crossed count buffer (debug
    /// builds only). `sky_background` (ticket 06): whether the miss shader
    /// returns the Procedural sky (the app's production pipeline) or stays
    /// black (the validator's capture pipeline — its byte-exact Reference
    /// comparison is against a constant background).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuStack,
        store: &RegionStore,
        virtual_swapchain_id: Id<Swapchain>,
        raygen: &EntryPoint,
        timestamps: Option<Arc<QueryPool>>,
        counter: Option<&CounterBuffer>,
        hull_crossed: Option<&HullCrossedCounter>,
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
                    .specialize(&[(0, SpecializationConstant::Bool(counter.is_some()))])
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
            timestamps,
            counter: counter.copied(),
            hull_crossed: hull_crossed.copied(),
        }
    }

    /// The instance buffer the TLAS reads (declared in the task graph).
    pub fn instance_buffer_id(&self) -> Id<Buffer> {
        self.instance_buffer_id
    }

    /// The bindless id of the region -> AABB-buffer table (the DDA's and the
    /// debug Hull shader's lookup).
    fn aabb_table_storage_id(&self) -> StorageBufferId {
        self.aabb_table_storage_id
    }
}

/// The analytic lights' default constants (ticket 06): the Sun's world
/// direction and illuminance, the Procedural sky's μ-gradient knots, and
/// the Sun disk. Tunable (the context writes them every frame; the app can
/// expose them later); the CPU path tracer (07) mirrors the same packed
/// values as data.
pub fn default_scene() -> capture_raygen::Scene {
    // Sun: normalize(0.45, 0.8, 0.35) — ~52° elevation, world-fixed. The
    // packed direction is data; the shader never renormalizes.
    let sun_dir = glam::Vec3::new(0.45, 0.8, 0.35).normalize();
    // Sky: the piecewise-linear radiance gradient in μ = cos(elevation),
    // knots at ground (μ = -1) / horizon (0) / zenith (1) — all strictly
    // positive (the marginal pdf stays positive everywhere).
    let knots = [0.15, 0.6, 1.2];
    // Sun disk: 0.5° angular radius (the real Sun); the disk radiance is
    // E_sun / Ω_disk, so the disk's integrated radiance equals the Sun's
    // illuminance — the same source: the disk is the visual (the camera's
    // direct view), the delta is the transport.
    let e_sun: f32 = 16.0;
    let cos_disk = (0.5_f32 * std::f32::consts::PI / 180.0).cos();
    let omega = 2.0 * std::f32::consts::PI * (1.0 - cos_disk);
    let l_disk = e_sun / omega;
    capture_raygen::Scene {
        sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
        sky_knots: [knots[0], knots[1], knots[2], 0.0],
        sun_disk: [e_sun, cos_disk, l_disk, 0.0],
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

    // The hull-crossed hit group's intersection stage (debug builds): the DDA
    // intersection re-specialized with PER_PIXEL_COUNTER = true. Declared at
    // function scope like the Hull stages so it outlives the pipeline create.
    #[cfg(debug_assertions)]
    let hull_crossed_intersection = unsafe {
        intersect::load(&gpu.device)
            .unwrap()
            .specialize(&[
                (0, SpecializationConstant::Bool(false)),
                (1, SpecializationConstant::Bool(true)),
            ])
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

        // The third procedural hit group (index 2): the DDA intersection shader
        // re-specialized with PER_PIXEL_COUNTER = true, sharing the DDA
        // closest-hit (stage index 3). It runs the same real DDA march, but
        // also increments the per-pixel hull-crossed count at slab-pass. The
        // production raygen selects it via sbtRecordOffset = 2 (the third
        // hit-region index; HullCrossed = 3 is the mode value the raygen maps
        // to hit-region 2).
        let hull_crossed_idx = stages.len() as u32;
        stages.push(PipelineShaderStageCreateInfo::new(&hull_crossed_intersection));
        groups.push(RayTracingShaderGroupCreateInfo::ProceduralHit {
            closest_hit_shader: Some(3),
            any_hit_shader: None,
            intersection_shader: hull_crossed_idx,
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

        // The Scene buffer (ticket 06): the analytic lights' constants,
        // updated per frame like the camera (tunable).
        unsafe { cbf.update_buffer(self.scene_buffer_id, 0, &rcx.scene) };

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

        // Reset the march-and-miss counter (fill 0) before the ray pass —
        // the app reads the words back after the flight idle. A fill runs on
        // the transfer stage, so make it visible to the shader's atomicAdd.
        if let Some(counter) = self.counter {
            unsafe {
                cbf.fill_buffer(&FillBufferInfo {
                    dst_buffer: counter.buffer_id,
                    data: 0,
                    ..Default::default()
                });
                cbf.pipeline_barrier(&DependencyInfo {
                    memory_barriers: &[MemoryBarrier {
                        src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                        dst_access: vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE
                            | vulkano::sync::AccessFlags::SHADER_STORAGE_READ,
                        src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,
                        dst_stages: vulkano::sync::PipelineStages::RAY_TRACING_SHADER,
                        ..Default::default()
                    }],
                    ..Default::default()
                });
            }
        }

        // Reset the per-pixel hull-crossed count buffer (fill 0) before the
        // ray pass — the heatmap overlay reads it after. Same transfer-stage
        // visibility discipline as the counter fill above.
        if let Some(hull_crossed) = self.hull_crossed {
            unsafe {
                cbf.fill_buffer(&FillBufferInfo {
                    dst_buffer: hull_crossed.buffer_id,
                    data: 0,
                    ..Default::default()
                });
                cbf.pipeline_barrier(&DependencyInfo {
                    memory_barriers: &[MemoryBarrier {
                        src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                        dst_access: vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE
                            | vulkano::sync::AccessFlags::SHADER_STORAGE_READ,
                        src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,
                        dst_stages: vulkano::sync::PipelineStages::RAY_TRACING_SHADER,
                        ..Default::default()
                    }],
                    ..Default::default()
                });
            }
        }

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
                    counter_buffer_id: self.counter.map(|counter| counter.storage_id).unwrap_or(StorageBufferId::INVALID),
                    hull_count_buffer_id: self.hull_crossed.map(|hull_crossed| hull_crossed.storage_id).unwrap_or(StorageBufferId::INVALID),
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
