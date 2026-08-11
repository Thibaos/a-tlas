//! The Region pipeline's GPU half (renderer-impl tickets 02/04): the shared
//! ray tracing pipeline and the per-frame render task that ray-passes the
//! swapchain storage images with the capture raygen (color + t-channel for
//! the validator).
//!
//! All per-Region GPU state — voxel pools, procedural AABB BLASes, the
//! lattice-static instance set and the stable TLAS — lives in
//! [`RegionStore`](crate::region::residency::RegionStore) (ticket 04: the
//! full static lattice, residency transitions, free lists). The render task
//! only holds the ids the push constants and the task graph need; the store
//! keeps the buffers alive across rebuilds.

use std::sync::Arc;

use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::Buffer,
    pipeline::{
        Pipeline,
        ray_tracing::{RayTracingPipeline, ShaderBindingTable},
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    descriptor_set::{AccelerationStructureId, StorageBufferId, StorageImageId},
};

use crate::{
    app::GpuStack,
    region::residency::RegionStore,
    tasks::render::build_ray_tracing_pipeline,
};

pub(crate) mod capture_raygen {
    vulkano_shaders::shader! {
        root_path_env: "CARGO_MANIFEST_DIR",
        ty: "raygen",
        path: "shaders/region/capture.rgen",
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

/// The per-frame world the Region render task reads.
pub struct RegionRenderContext {
    pub camera: capture_raygen::Camera,
    pub swapchain_storage_image_ids: Vec<StorageImageId>,
    /// Validation only: the capture raygen additionally writes payload.t
    /// here; the production raygen passes INVALID and never dereferences it.
    pub t_image_storage_id: StorageImageId,
}

pub struct RegionRenderTask {
    swapchain_id: Id<Swapchain>,
    camera_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    camera_storage_id: StorageBufferId,
    palette_storage_id: StorageBufferId,
    region_table_storage_id: StorageBufferId,
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
}

impl RegionRenderTask {
    /// Builds the shared ray tracing pipeline and binds the store's buffers.
    /// The store outlives the task, so its ids stay valid for every frame of
    /// the pass (residency rebuilds rewrite the buffers in place).
    pub fn new(
        gpu: &GpuStack,
        store: &RegionStore,
        virtual_swapchain_id: Id<Swapchain>,
    ) -> Self {
        let pipeline = {
            let raygen = unsafe {
                capture_raygen::load(&gpu.device)
                    .unwrap()
                    .entry_point("main")
                    .unwrap()
            };
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

            build_ray_tracing_pipeline(gpu, &raygen, &miss, &intersection, &closest_hit)
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
            acceleration_structure_id: store.acceleration_structure_id,
            shader_binding_table,
            pipeline,
            blases: store.blases(),
        }
    }

    /// The instance buffer the TLAS reads (declared in the task graph).
    pub fn instance_buffer_id(&self) -> Id<Buffer> {
        self.instance_buffer_id
    }
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
                },
            )
        };

        unsafe {
            cbf.bind_pipeline_ray_tracing(&self.pipeline);
        }

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) };

        Ok(())
    }
}
