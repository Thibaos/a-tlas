//! Ordered rebuild nodes.
//!
//! Rebuilds execute as **ordered taskgraph nodes** between the consuming
//! trace and the next frame: pool upload → BLAS build → TLAS build (on
//! residency transitions), so in-place rebuilds are race-free by ordering.
//! [`RegionStore`](crate::render::region::residency::RegionStore) computes a
//! CPU-side [`RebuildPlan`] (allocations, tables, packed instances — the
//! worker keeps only the CPU-side drain/pack) and hands it to
//! [`RebuildGraph`], which compiles one ordered graph per change cycle over
//! exactly the buffers involved:
//!
//! - an **upload node** writes the pool bytes, the trimmed AABBs, the region
//!   table and the packed instance prefix through the taskgraph's host-write
//!   access (at record time);
//! - a **BLAS-build node** rebuilds every dirty Region's procedural AABB
//!   BLAS **in place** (device address stable → the TLAS instance stays
//!   valid, the TLAS untouched); become-resident/replacement BLASes get
//!   their storage in the plan phase and are built into it in place, so no
//!   AS object ever moves;
//! - a **TLAS-build node** rebuilds the stable TLAS **in place** (the
//!   bindless acceleration-structure id never moves) iff a residency
//!   transition happened.
//!
//! There is **no back-AS double buffer and no flip atomic** in this path:
//! the store owns exactly one stable TLAS (rebuilt in place), the nodes
//! only record ordered commands, and the plan is plain data. Startup is a
//! one-shot pre-loop build from the initial batch through the same graph.
//! Each node records begin/end GPU timestamps (when the device supports
//! them), so rebuild time is attributable per node.

use std::sync::Arc;

use glam::IVec3;
use vulkano::{
    DeviceSize,
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureGeometry,
        AccelerationStructureGeometryAabbsData, AccelerationStructureGeometryData,
        AccelerationStructureGeometryInstancesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::AllocationCreateInfo,
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    sync::{AccessFlags, PipelineStage, PipelineStages},
};
use vulkano_taskgraph::{
    Id, QueueFamilyType, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{AccessTypes, HostAccessType},
    resource_map,
};

use crate::{
    core::gpu::GpuStack,
    render::{
        accel,
        region::{pack::REGION_COUNT, residency::RegionStore, task::capture_raygen},
    },
};

/// One Region's GPU upload: the pool bytes and the trimmed AABBs the upload
/// node copies into the Region's buffers at record time. The plan phase
/// already made every allocation/table decision; the node only records.
pub struct RegionUpload {
    pub region_index: IVec3,
    pub pool_buffer_id: Id<Buffer>,
    pub pool_bytes: Vec<u8>,
    pub aabb_buffer_id: Id<Buffer>,
    pub aabbs: Vec<AabbPositions>,
}

/// One BLAS build. Every build is recorded as an in-place build into known
/// storage: a become-resident/replacement BLAS gets its storage created by
/// the plan phase (CPU) and the node builds into it — the AS object and its
/// device address never move after creation.
pub struct BlasBuild {
    pub region_index: IVec3,
    pub aabb_buffer_id: Id<Buffer>,
    pub aabb_count: u32,
    /// The destination AS storage (the Region's BLAS, stable while
    /// resident). The plan phase asserted the build fits the storage.
    pub blas: Arc<AccelerationStructure>,
    /// The build's scratch (kept alive by the plan).
    pub scratch: Arc<Buffer>,
    /// `true` for become-resident and capacity-replacement builds (fresh
    /// storage); `false` for in-place content-edit rebuilds.
    pub fresh: bool,
}

/// The TLAS rebuild: rebuild the store's stable TLAS in place over the
/// packed instance prefix (one instance per resident Region).
pub struct TlasBuild {
    pub instance_count: u32,
    /// The build's scratch (kept alive by the plan).
    pub scratch: Arc<Buffer>,
}

/// The CPU-side plan for one change cycle: what the ordered rebuild nodes
/// will do. Built by the store's plan phase (CPU-only) and consumed by
/// [`RebuildGraph`]. Plain data — no double buffer, no flip atomic.
#[derive(Default)]
pub struct RebuildPlan {
    pub uploads: Vec<RegionUpload>,
    pub blas_builds: Vec<BlasBuild>,
    /// The region table rewrite (Region id → pool device address), when
    /// residency or pool addresses changed.
    pub table: Option<[u64; REGION_COUNT]>,
    /// The packed instance prefix (one per resident Region; the never-hit
    /// dummy instance when nothing is resident), when residency changed.
    pub instances: Option<Vec<AccelerationStructureInstance>>,
    /// The TLAS rebuild, iff a residency transition happened.
    pub tlas: Option<TlasBuild>,
}

impl RebuildPlan {
    /// The plan does no GPU work at all (an idle change cycle: dirty regions
    /// whose mirrors emptied before the cycle — the renderer costs zero, no
    /// rebuild graph is compiled or submitted).
    pub fn is_empty(&self) -> bool {
        self.uploads.is_empty()
            && self.blas_builds.is_empty()
            && self.table.is_none()
            && self.instances.is_none()
            && self.tlas.is_none()
    }

    /// The ordered rebuild log for the harness invariants: what
    /// the nodes will do, in node order (upload → BLAS → TLAS). The
    /// counters the acceptance criteria check: a content edit logs an
    /// in-place `BuildBlas` and **no** `BuildTlas`; a residency transition
    /// logs `RewriteInstances` + `BuildTlas`.
    pub fn log(&self) -> Vec<RebuildLogEntry> {
        let mut log = Vec::new();
        for upload in &self.uploads {
            log.push(RebuildLogEntry::Upload {
                region_index: upload.region_index,
                pool_bytes: upload.pool_bytes.len() as u64,
                aabbs: upload.aabbs.len() as u32,
            });
        }
        if self.table.is_some() {
            log.push(RebuildLogEntry::WriteRegionTable);
        }
        if let Some(instances) = &self.instances {
            log.push(RebuildLogEntry::RewriteInstances {
                instance_count: instances.len() as u32,
            });
        }
        for build in &self.blas_builds {
            log.push(RebuildLogEntry::BuildBlas {
                region_index: build.region_index,
                aabb_count: build.aabb_count,
                fresh: build.fresh,
            });
        }
        if let Some(tlas) = &self.tlas {
            log.push(RebuildLogEntry::BuildTlas {
                instance_count: tlas.instance_count,
            });
        }
        log
    }
}

/// One entry of the ordered rebuild log (the harness's counters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildLogEntry {
    /// The upload node copied a Region's pool bytes + trimmed AABBs.
    Upload {
        region_index: IVec3,
        pool_bytes: u64,
        aabbs: u32,
    },
    /// The BLAS node rebuilt a Region's BLAS in place (`fresh: false` —
    /// content edit, device address stable, TLAS untouched) or built into
    /// fresh storage (`fresh: true` — become-resident/replacement).
    BuildBlas {
        region_index: IVec3,
        aabb_count: u32,
        fresh: bool,
    },
    /// The upload node rewrote the region table (Region id → pool BDA).
    WriteRegionTable,
    /// The upload node rewrote the packed instance prefix (residency
    /// changed).
    RewriteInstances { instance_count: u32 },
    /// The TLAS node rebuilt the stable TLAS in place.
    BuildTlas { instance_count: u32 },
}

/// Per-node GPU timings for one rebuild cycle: each node records begin/end timestamps around its work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeTimings {
    /// The upload node's GPU time (host writes — expected ~0).
    pub upload_ns: u64,
    /// The BLAS-build node's GPU time.
    pub blas_ns: u64,
    /// The TLAS-build node's GPU time.
    pub tlas_ns: u64,
    /// Whether timestamp queries are available on this device (`false` → the
    /// fields stay 0; the GPU time is still attributable by node *shape*).
    pub supported: bool,
}

/// One node's begin/end timestamp queries inside the shared pool.
struct QueryRange {
    pool: Arc<QueryPool>,
    begin: u32,
    end: u32,
}

// ---------------------------------------------------------------------------
// The ordered nodes
// ---------------------------------------------------------------------------

/// The first node: host writes. Copies the plan's pool bytes, trimmed AABBs,
/// region table and packed instance prefix into the store's buffers at
/// record time (the taskgraph flushes the host writes before submit, so the
/// subsequent nodes' declared accesses see them). No GPU commands — its
/// begin/end timestamps bracket the (empty) node, keeping the per-node
/// attribution uniform.
struct UploadRegionsTask {
    uploads: Vec<RegionUpload>,
    table: Option<[u64; REGION_COUNT]>,
    instances: Option<Vec<AccelerationStructureInstance>>,
    instance_buffer_id: Id<Buffer>,
    region_table_buffer_id: Id<Buffer>,
    timestamps: Option<QueryRange>,
}

impl Task for UploadRegionsTask {
    type World = ();

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        if let Some(range) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(&range.pool, range.begin, PipelineStage::AllCommands)
            };
        }

        for upload in &self.uploads {
            tcx.write_buffer::<[u8]>(
                upload.pool_buffer_id,
                0..upload.pool_bytes.len() as DeviceSize,
            )
            .copy_from_slice(&upload.pool_bytes);

            let dst = tcx.write_buffer::<[AabbPositions]>(
                upload.aabb_buffer_id,
                0..(upload.aabbs.len() as DeviceSize * size_of::<AabbPositions>() as DeviceSize),
            );
            for (slot, aabb) in dst.iter_mut().zip(upload.aabbs.iter().copied()) {
                *slot = aabb;
            }
        }

        if let Some(table) = &self.table {
            *tcx.write_buffer::<capture_raygen::RegionTable>(self.region_table_buffer_id, ..) =
                capture_raygen::RegionTable { bdas: *table };
        }

        if let Some(instances) = &self.instances {
            let dst = tcx.write_buffer::<[AccelerationStructureInstance]>(
                self.instance_buffer_id,
                0..(instances.len() as DeviceSize
                    * size_of::<AccelerationStructureInstance>() as DeviceSize),
            );
            for (slot, instance) in dst.iter_mut().zip(instances) {
                *slot = *instance;
            }
        }

        if let Some(range) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(&range.pool, range.end, PipelineStage::AllCommands)
            };
        }

        Ok(())
    }
}

/// The second node: rebuild every dirty Region's BLAS **in place** (device
/// address stable → TLAS untouched). Reads the upload node's host-written
/// AABB buffers (declared access + the pre-barrier's HOST_WRITE src), writes
/// the AS storage.
struct BuildBlasTask {
    builds: Vec<BlasBuild>,
    timestamps: Option<QueryRange>,
}

impl Task for BuildBlasTask {
    type World = ();

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        if let Some(range) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(&range.pool, range.begin, PipelineStage::AllCommands)
            };
        }

        for build in &self.builds {
            let aabb_buffer = Subbuffer::new(tcx.buffer(build.aabb_buffer_id).buffer().clone())
                .cast_aligned::<AabbPositions>();
            let aabb_data = AccelerationStructureGeometryAabbsData {
                data: aabb_buffer.device_address().unwrap().get(),
                stride: size_of::<AabbPositions>() as u32,
                ..Default::default()
            };
            let geometries = vec![AccelerationStructureGeometry::new(
                AccelerationStructureGeometryData::Aabbs(aabb_data),
            )];

            let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
                ty: AccelerationStructureType::BottomLevel,
                mode: BuildAccelerationStructureMode::Build,
                flags: accel::build_flags(AccelerationStructureType::BottomLevel),
                geometries: &geometries,
                ..AccelerationStructureBuildGeometryInfo::new()
            };
            build_geometry_info.dst_acceleration_structure = Some(&build.blas);
            build_geometry_info.scratch_data = build.scratch.device_address().get();

            // The pre-barrier: the upload node's host writes (flush + this
            // dependency) and any prior build's writes are visible to this
            // build; the post-barrier makes the built BLAS visible to later
            // rebuilds and traces.
            unsafe {
                cbf.pipeline_barrier(&DependencyInfo {
                    memory_barriers: &[build_pre_barrier()],
                    ..Default::default()
                })
            };

            unsafe {
                cbf.as_raw().build_acceleration_structure(
                    &build_geometry_info,
                    &[AccelerationStructureBuildRangeInfo {
                        primitive_count: build.aabb_count,
                        ..Default::default()
                    }],
                )
            };

            unsafe {
                cbf.pipeline_barrier(&DependencyInfo {
                    memory_barriers: &[build_post_barrier()],
                    ..Default::default()
                })
            };
        }

        if let Some(range) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(&range.pool, range.end, PipelineStage::AllCommands)
            };
        }

        Ok(())
    }
}

/// The third node: rebuild the stable TLAS **in place** over the packed
/// instance prefix — only on residency transitions. Reads the upload node's
/// host-written instance buffer (declared access), writes the TLAS storage.
///
/// The node declares the instance buffer with
/// `ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE` — the build
/// *reads* it, but the declared write is the ordering proxy (inherited from
/// the retired async TLAS worker): the render task declares
/// `RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ` on the same buffer, so
/// the trace waits for this node's submission — the rebuild lands between
/// the consuming trace and the next frame, and in-place rebuilds are
/// race-free by ordering.
struct BuildTlasTask {
    instance_count: u32,
    instance_buffer_id: Id<Buffer>,
    tlas: Arc<AccelerationStructure>,
    scratch: Arc<Buffer>,
    timestamps: Option<QueryRange>,
}

impl Task for BuildTlasTask {
    type World = ();

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        if let Some(range) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(&range.pool, range.begin, PipelineStage::AllCommands)
            };
        }

        let instance_buffer = Subbuffer::new(tcx.buffer(self.instance_buffer_id).buffer().clone())
            .cast_aligned::<AccelerationStructureInstance>();
        let instances_data = AccelerationStructureGeometryInstancesData {
            data: instance_buffer.device_address().unwrap().get(),
            ..Default::default()
        };
        let geometries = vec![AccelerationStructureGeometry::new(
            AccelerationStructureGeometryData::Instances(instances_data),
        )];

        let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
            ty: AccelerationStructureType::TopLevel,
            mode: BuildAccelerationStructureMode::Build,
            flags: accel::build_flags(AccelerationStructureType::TopLevel),
            geometries: &geometries,
            ..AccelerationStructureBuildGeometryInfo::new()
        };
        build_geometry_info.dst_acceleration_structure = Some(&self.tlas);
        build_geometry_info.scratch_data = self.scratch.device_address().get();

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[build_pre_barrier()],
                ..Default::default()
            })
        };

        unsafe {
            cbf.as_raw().build_acceleration_structure(
                &build_geometry_info,
                &[AccelerationStructureBuildRangeInfo {
                    primitive_count: self.instance_count,
                    ..Default::default()
                }],
            )
        };

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[build_post_barrier()],
                ..Default::default()
            })
        };

        if let Some(range) = &self.timestamps {
            unsafe {
                cbf.as_raw()
                    .write_timestamp(&range.pool, range.end, PipelineStage::AllCommands)
            };
        }

        Ok(())
    }
}

/// The pre-build memory barrier: the build input (host-written uploads,
/// prior transfer/shader writes) and any prior build of the same AS are
/// visible before this build writes it.
fn build_pre_barrier() -> MemoryBarrier<'static> {
    MemoryBarrier {
        src_access: AccessFlags::TRANSFER_WRITE
            | AccessFlags::SHADER_WRITE
            | AccessFlags::HOST_WRITE
            | AccessFlags::ACCELERATION_STRUCTURE_WRITE,
        dst_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE
            | AccessFlags::ACCELERATION_STRUCTURE_READ,
        src_stages: PipelineStages::ALL_TRANSFER
            | PipelineStages::COMPUTE_SHADER
            | PipelineStages::HOST
            | PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        dst_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        ..Default::default()
    }
}

/// The post-build memory barrier: the built AS is visible to later rebuilds
/// and traces.
fn build_post_barrier() -> MemoryBarrier<'static> {
    MemoryBarrier {
        src_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE,
        dst_access: AccessFlags::ACCELERATION_STRUCTURE_READ | AccessFlags::SHADER_READ,
        src_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        dst_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD
            | PipelineStages::RAY_TRACING_SHADER,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// The per-cycle ordered graph
// ---------------------------------------------------------------------------

/// The per-cycle rebuild graph: upload → BLAS build → TLAS build (on
/// residency), compiled once per change cycle over exactly the buffers
/// involved, executed on the compute queue and waited before returning — so
/// the freed memory release in the store is safe (the dropping rebuild
/// executed) and the per-node timestamps can be read back. An idle renderer
/// never builds one (zero cost: no pending rebuilds, no wakeups).
pub struct RebuildGraph {
    executable: ExecutableTaskGraph<()>,
    query_pool: Option<Arc<QueryPool>>,
    slots: NodeTimingSlots,
    /// The timestamp pool's query count (0 when unsupported).
    query_count: u32,
}

/// Begin/end query slots per node kind (nodes that don't run have `None`).
struct NodeTimingSlots {
    upload: (u32, u32),
    blas: Option<(u32, u32)>,
    tlas: Option<(u32, u32)>,
}

impl RebuildGraph {
    /// Compiles the ordered graph for one change cycle. The plan's data is
    /// moved into the nodes (the graph records from it); the store's stable
    /// buffers and the stable TLAS are referenced by id/Arc.
    pub fn new(gpu: &GpuStack, store: &RegionStore, plan: RebuildPlan) -> Self {
        // The BLAS node's declared build-input accesses (collected before
        // the plan's data moves into the nodes): the taskgraph inserts the
        // initial barrier making the upload node's host writes visible to
        // the build stage, and tracks the buffers across graphs.
        let blas_buffers = blas_buffer_ids(&plan);

        let mut task_graph = TaskGraph::new(&gpu.resources);

        // The upload node's host writes (recorded at execute time, flushed
        // before submit; the declared device accesses of the later nodes see
        // them through the taskgraph's initial barrier).
        for upload in &plan.uploads {
            task_graph.add_host_buffer_access(upload.pool_buffer_id, HostAccessType::Write);
            task_graph.add_host_buffer_access(upload.aabb_buffer_id, HostAccessType::Write);
        }
        if plan.table.is_some() {
            task_graph
                .add_host_buffer_access(store.region_table_buffer_id(), HostAccessType::Write);
        }
        if plan.instances.is_some() {
            task_graph.add_host_buffer_access(store.instance_buffer_id, HostAccessType::Write);
        }

        let timestamps_supported = timestamp_supported(gpu);
        let mut slots = NodeTimingSlots {
            upload: (0, 1),
            blas: None,
            tlas: None,
        };
        let mut next_slot = 2u32;
        let mut node_count = 1;
        if !plan.blas_builds.is_empty() {
            slots.blas = Some((next_slot, next_slot + 1));
            next_slot += 2;
            node_count += 1;
        }
        if plan.tlas.is_some() {
            slots.tlas = Some((next_slot, next_slot + 1));
            node_count += 1;
        }
        let query_count = node_count * 2;
        let query_pool = timestamps_supported.then(|| {
            QueryPool::new(
                &gpu.device,
                &QueryPoolCreateInfo {
                    query_count,
                    ..QueryPoolCreateInfo::new(QueryType::Timestamp)
                },
            )
            .unwrap()
        });
        let query_range = |pool: &Option<Arc<QueryPool>>, slot: (u32, u32)| {
            pool.as_ref().map(|pool| QueryRange {
                pool: pool.clone(),
                begin: slot.0,
                end: slot.1,
            })
        };

        let upload_node = task_graph
            .create_task_node(
                "Rebuild upload",
                QueueFamilyType::Compute,
                UploadRegionsTask {
                    uploads: plan.uploads,
                    table: plan.table,
                    instances: plan.instances,
                    instance_buffer_id: store.instance_buffer_id,
                    region_table_buffer_id: store.region_table_buffer_id(),
                    timestamps: query_range(&query_pool, slots.upload),
                },
            )
            .build();

        let mut blas_node = None;
        if !plan.blas_builds.is_empty() {
            let mut builder = task_graph.create_task_node(
                "Rebuild BLAS",
                QueueFamilyType::Compute,
                BuildBlasTask {
                    builds: plan.blas_builds,
                    timestamps: query_range(&query_pool, slots.blas.unwrap()),
                },
            );
            for build in blas_buffers {
                builder.buffer_access(
                    build,
                    AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_READ,
                );
            }
            blas_node = Some(builder.build());
        }

        let mut tlas_node = None;
        if let Some(tlas) = plan.tlas {
            tlas_node = Some(
                task_graph
                    .create_task_node(
                        "Rebuild TLAS",
                        QueueFamilyType::Compute,
                        BuildTlasTask {
                            instance_count: tlas.instance_count,
                            instance_buffer_id: store.instance_buffer_id,
                            tlas: store.tlas(),
                            scratch: tlas.scratch,
                            timestamps: query_range(&query_pool, slots.tlas.unwrap()),
                        },
                    )
                    .buffer_access(
                        store.instance_buffer_id,
                        AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                    )
                    .build(),
            );
        }

        // The ordered chain: upload → BLAS → TLAS (skipping missing nodes).
        let mut previous = Some(upload_node);
        if let Some(blas_node) = blas_node {
            task_graph.add_edge(previous.unwrap(), blas_node).unwrap();
            previous = Some(blas_node);
        }
        if let Some(tlas_node) = tlas_node {
            task_graph.add_edge(previous.unwrap(), tlas_node).unwrap();
        }

        let executable = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&gpu.compute_queue],
                present_queue: None,
                flight_id: gpu.compute_flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        Self {
            executable,
            query_pool,
            slots,
            query_count,
        }
    }

    /// Executes the ordered rebuild nodes, waits for the compute flight, and
    /// reads back the per-node GPU timestamps.
    pub fn execute(self, gpu: &GpuStack) -> NodeTimings {
        let resource_map = resource_map!(&self.executable).unwrap();

        unsafe { self.executable.execute(resource_map, &(), || {}) }.unwrap();

        // The rebuild sequence must complete before the caller reads the
        // timestamps or releases the pending frees (the dropping rebuild
        // executed).
        gpu.resources
            .flight(gpu.compute_flight_id)
            .wait_idle()
            .unwrap();

        self.read_timings(gpu)
    }

    /// Reads the per-node begin/end timestamp pairs (elapsed ×
    /// `timestamp_period`; 0 when unsupported or unavailable).
    fn read_timings(&self, gpu: &GpuStack) -> NodeTimings {
        let mut timings = NodeTimings {
            supported: self.query_pool.is_some(),
            ..Default::default()
        };
        let Some(pool) = &self.query_pool else {
            return timings;
        };

        let mut values = vec![0u64; self.query_count as usize];
        let available = pool
            .get_results::<u64>(0, self.query_count, &mut values, QueryResultFlags::empty())
            .unwrap();
        if !available {
            return timings;
        }

        let period = gpu.device.physical_device().properties().timestamp_period as f64;
        let elapsed = |slot: (u32, u32)| {
            let begin = values[slot.0 as usize];
            let end = values[slot.1 as usize];
            (end.wrapping_sub(begin) as f64 * period) as u64
        };

        timings.upload_ns = elapsed(self.slots.upload);
        if let Some(slot) = self.slots.blas {
            timings.blas_ns = elapsed(slot);
        }
        if let Some(slot) = self.slots.tlas {
            timings.tlas_ns = elapsed(slot);
        }
        timings
    }
}

/// The AABB buffer ids the BLAS node builds from (its declared device
/// accesses).
fn blas_buffer_ids(plan: &RebuildPlan) -> Vec<Id<Buffer>> {
    plan.blas_builds
        .iter()
        .map(|build| build.aabb_buffer_id)
        .collect()
}

/// Whether the compute queue family supports timestamp queries.
fn timestamp_supported(gpu: &GpuStack) -> bool {
    let index = gpu.compute_queue.queue_family_index() as usize;
    gpu.device.physical_device().queue_family_properties()[index]
        .timestamp_valid_bits
        .is_some()
}

/// Allocates a build scratch buffer of `size` bytes (the plan phase; kept
/// alive by the plan until the rebuild completes).
pub(crate) fn allocate_scratch(gpu: &GpuStack, size: DeviceSize) -> Arc<Buffer> {
    Buffer::new_slice::<u8>(
        &gpu.memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        size.max(1),
    )
    .unwrap()
    .buffer()
    .clone()
}

/// The BLAS build sizes (AS storage + scratch) over `aabb_buffer` with
/// `aabb_count` AABBs — the plan phase sizes the scratch and asserts the
/// build fits the Region's storage.
pub(crate) fn blas_build_sizes(
    gpu: &GpuStack,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
) -> vulkano::acceleration_structure::AccelerationStructureBuildSizesInfo {
    let geometries = aabb_geometries(aabb_buffer);
    accel::acceleration_structure_build_sizes(
        &gpu.device,
        &geometries,
        AccelerationStructureType::BottomLevel,
        aabb_count,
    )
}

/// The procedural-AABB geometry list for a BLAS build over `aabb_buffer`.
pub(crate) fn aabb_geometries(
    aabb_buffer: &Subbuffer<[AabbPositions]>,
) -> Vec<AccelerationStructureGeometry<'static>> {
    let aabb_data = AccelerationStructureGeometryAabbsData {
        data: aabb_buffer.device_address().unwrap().get(),
        stride: size_of::<AabbPositions>() as u32,
        ..Default::default()
    };
    vec![AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Aabbs(aabb_data),
    )]
}

/// The TLAS build sizes (AS storage + scratch) over the packed instance
/// prefix.
pub(crate) fn tlas_build_sizes(
    gpu: &GpuStack,
    instance_buffer: &Subbuffer<[AccelerationStructureInstance]>,
    instance_count: u32,
) -> vulkano::acceleration_structure::AccelerationStructureBuildSizesInfo {
    let instances_data = AccelerationStructureGeometryInstancesData {
        data: instance_buffer.device_address().unwrap().get(),
        ..Default::default()
    };
    let geometries = vec![AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Instances(instances_data),
    )];
    accel::acceleration_structure_build_sizes(
        &gpu.device,
        &geometries,
        AccelerationStructureType::TopLevel,
        instance_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty plan is plain data with no GPU work — the idle invariant's
    /// shape: no rebuild graph is compiled when nothing changed. (The new
    /// path carries no back-AS double buffer and no flip atomic: the plan
    /// has no AS array and no index state, and the store owns exactly one
    /// stable TLAS — see [`RegionStore::tlas`].)
    #[test]
    fn empty_plan_means_no_rebuild_work() {
        let plan = RebuildPlan::default();
        assert!(plan.is_empty());
        assert!(plan.log().is_empty());
    }

    /// The log derives from the plan in node order: uploads, table,
    /// instances, BLAS builds, TLAS build — the counters the harness checks.
    /// Content edits (in-place BLAS, no TLAS) and transitions (instances +
    /// TLAS) are distinguishable by shape.
    #[test]
    fn plan_log_orders_entries_by_node() {
        let mut plan = RebuildPlan {
            uploads: vec![RegionUpload {
                region_index: IVec3::new(0, 0, 0),
                pool_buffer_id: Id::INVALID,
                pool_bytes: vec![0u8; 64],
                aabb_buffer_id: Id::INVALID,
                aabbs: vec![AabbPositions {
                    min: [0.0; 3],
                    max: [1.0; 3],
                }],
            }],
            blas_builds: vec![],
            table: None,
            instances: None,
            tlas: None,
        };
        assert_eq!(
            plan.log(),
            vec![RebuildLogEntry::Upload {
                region_index: IVec3::new(0, 0, 0),
                pool_bytes: 64,
                aabbs: 1,
            }]
        );

        // A transition plan: instances + TLAS (the build entries need GPU
        // objects, so the shape is asserted with the data-only parts).
        plan.instances = Some(vec![AccelerationStructureInstance::default()]);
        plan.tlas = None;
        let log = plan.log();
        assert!(
            log.iter()
                .any(|e| matches!(e, RebuildLogEntry::RewriteInstances { instance_count: 1 }))
        );
        assert!(
            !log.iter()
                .any(|e| matches!(e, RebuildLogEntry::BuildTlas { .. }))
        );
    }
}
