//! Multi-region residency: the full static lattice.
//!
//! The renderer owns the lattice: Region = 256^3 voxels,
//! origin-aligned, v1 extent ±2048/axis → 16^3 = 4096 Regions — exactly the
//! 12-bit region-id budget. [`RegionStore`] is the GPU half of that lattice:
//! per-Region voxel pools and trimmed-AABB BLASes exist across the lattice;
//! a Region becomes **Resident** on its first non-empty Micro-chunk (a pool
//! buffer + a procedural AABB BLAS, allocated from free lists) and leaves
//! residency on its last (memory returned to the free lists; the CPU mirror
//! is freed with the Region by the input contract). The TLAS holds one
//! instance per Resident region — lattice-static transform, custom index =
//! region id, mask 0xFF — added on residency, removed on region-empty, and
//! rebuilt **in place** so the bindless acceleration-structure id never
//! moves.

use std::sync::Arc;

use dot_vox::DotVoxData;
use glam::IVec3;
use vulkano::{
    DeviceSize, Packed24_8,
    acceleration_structure::{AabbPositions, AccelerationStructure, AccelerationStructureInstance},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
};
use vulkano_taskgraph::{
    Id,
    descriptor_set::{AccelerationStructureId, StorageBufferId},
    resource::HostAccessType,
};

use crate::{
    app::GpuStack,
    grid::{REGION_EDGE, region_id},
    region::{
        input::RendererInput,
        pack::{REGION_COUNT, RegionData},
        rebuild::{
            BlasBuild, NodeTimings, RebuildGraph, RebuildLogEntry, RebuildPlan, RegionUpload,
            TlasBuild, allocate_scratch, blas_build_sizes, tlas_build_sizes,
        },
        render::capture_raygen,
    },
    rt::acceleration_structure,
    world::voxel::{get_material_table, get_palette},
};

/// One resident Region's GPU resources plus its allocation capacities.
struct ResidentRegion {
    /// The Region's voxel pool buffer (offset table + compact blocks).
    pool_buffer_id: Id<Buffer>,
    /// The pool buffer's allocation size (bytes).
    pool_capacity: u64,
    /// The BLAS build-input buffer (trimmed AABBs, Region-local).
    aabb_buffer_id: Id<Buffer>,
    /// How many AABBs the AABB buffer + BLAS storage can hold.
    aabb_capacity: u32,
    /// The Region's procedural AABB BLAS. Its device address is stable while
    /// the Region is resident (content edits rebuild it in place; a capacity
    /// growth replaces it — the TLAS instance then moves, which is the only
    /// non-transition TLAS rebuild).
    blas: Arc<AccelerationStructure>,
    /// The BLAS storage size (free-list reuse unit).
    blas_storage_size: u64,
}

/// A freed pool buffer awaiting reuse (capacity = allocation size).
struct FreedPool {
    buffer_id: Id<Buffer>,
    capacity: u64,
}

/// A freed (AABB buffer + BLAS storage) pair awaiting reuse.
struct FreedBlas {
    aabb_buffer_id: Id<Buffer>,
    aabb_capacity: u32,
    blas: Arc<AccelerationStructure>,
    blas_storage_size: u64,
}

/// The reusable free lists: memory whose referencing TLAS instance was
/// dropped by an executed rebuild.
#[derive(Default)]
struct FreeLists {
    pools: Vec<FreedPool>,
    blas: Vec<FreedBlas>,
}

/// Memory freed by the current change cycle, not yet reusable: the rebuild
/// that dropped the referencing instance must execute.
#[derive(Default)]
struct PendingFrees {
    pools: Vec<FreedPool>,
    blas: Vec<FreedBlas>,
}

/// Allocation probes (harness/tests): fresh allocations vs free-list reuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub pool_allocations: u64,
    pub pool_reuses: u64,
    pub blas_allocations: u64,
    pub blas_reuses: u64,
}

/// The outcome of one change cycle ([`RegionStore::apply`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    /// Regions that became resident this cycle.
    pub became_resident: Vec<IVec3>,
    /// Regions that left residency this cycle.
    pub left_resident: Vec<IVec3>,
    /// Resident regions whose content changed (BLAS rebuilt in place).
    pub dirty: Vec<IVec3>,
    /// Resident regions whose pack outgrew their BLAS capacity (the BLAS was
    /// replaced and the instance address moved — a documented non-transition
    /// TLAS rebuild, see [`RegionStore::rebuild`]).
    pub blas_replaced: Vec<IVec3>,
    /// The TLAS was rebuilt this cycle (iff any residency transition or BLAS
    /// replacement happened — instance data is static otherwise).
    pub tlas_rebuilt: bool,
    /// The ordered rebuild log: what the ordered
    /// rebuild nodes did this cycle, in node order (upload → BLAS → TLAS).
    /// A content edit logs an in-place [`RebuildLogEntry::BuildBlas`] and
    /// **no** [`RebuildLogEntry::BuildTlas`]; a residency transition logs
    /// [`RebuildLogEntry::RewriteInstances`] + [`RebuildLogEntry::BuildTlas`].
    pub rebuild_log: Vec<RebuildLogEntry>,
    /// Per-node GPU timings for this cycle.
    pub timings: NodeTimings,
    /// The resident-instance count before this cycle (the TLAS instance set
    /// size — the harness's ±1 transition probe).
    pub instance_count_before: usize,
    /// The resident-instance count after this cycle.
    pub instance_count: usize,
}

/// Best-fit allocation: removes the smallest entry with capacity ≥ `needed`
/// (pure — unit-tested). `None` means allocate fresh.
fn take_best_fit<T>(entries: &mut Vec<T>, needed: u64, capacity: impl Fn(&T) -> u64) -> Option<T> {
    let mut best: Option<(usize, u64)> = None;
    for (i, entry) in entries.iter().enumerate() {
        let cap = capacity(entry);
        if cap >= needed && best.is_none_or(|(_, c)| cap < c) {
            best = Some((i, cap));
        }
    }
    best.map(|(i, _)| entries.swap_remove(i))
}

/// The pool allocation for one Region (fresh or free-list reused).
struct PoolAllocation {
    buffer_id: Id<Buffer>,
    capacity: u64,
}

/// The BLAS allocation for one Region: the AABB build-input buffer plus,
/// when reused, the existing AS storage to build into.
struct BlasAllocation {
    aabb_buffer_id: Id<Buffer>,
    aabb_capacity: u32,
    /// `Some((as, storage_size))` when reused from the free list (build in
    /// place); `None` when fresh (create storage + build).
    as_storage: Option<(Arc<AccelerationStructure>, u64)>,
}

/// The full static lattice's GPU side: 4096 Region slots (12-bit id), the
/// free lists, the instance set and the stable TLAS.
pub struct RegionStore {
    // --- static (per-world) buffers + bindless ids ---------------------
    pub camera_buffer_id: Id<Buffer>,
    /// The Scene buffer (ticket 06): the analytic lights' constants (Sun
    /// direction/illuminance, sky knots, disk), updated every frame from
    /// the render context (tunable). The capture pipeline never reads it
    /// (its miss shader stays black) — the byte-exact validator is
    /// unchanged.
    pub scene_buffer_id: Id<Buffer>,
    pub region_table_storage_id: StorageBufferId,
    pub camera_storage_id: StorageBufferId,
    pub scene_storage_id: StorageBufferId,
    pub palette_storage_id: StorageBufferId,
    /// The bindless Material table (ADR 0008): one entry per palette index
    /// (albedo+metallic / emission+roughness), uploaded once at startup — the
    /// GPU twin of `world::voxel::get_material_table`'s mirror. Read by the
    /// DDA closest-hit (surface color — albedo == palette, so the byte-exact
    /// capture path is unchanged) and the production raygen in Voxel mode.
    pub material_table_storage_id: StorageBufferId,
    pub acceleration_structure_id: AccelerationStructureId,
    region_table_buffer_id: Id<Buffer>,
    /// The bindless id of the region -> AABB-buffer table (the DDA's and the
    /// debug Hull shader's lookup, parallel to `region_table`). The buffer
    /// itself stays alive via the bindless registration; this id is copied
    /// into the render task each frame.
    pub aabb_table_storage_id: StorageBufferId,

    // --- instance set + TLAS --------------------------------------------
    /// Lattice-static instance data, one slot per Region id: transform
    /// (translation by the Region origin), custom index = id, mask 0xFF.
    /// Only `acceleration_structure_reference` ever changes (residency).
    instances: Vec<AccelerationStructureInstance>,
    /// The packed resident ids, sorted — the TLAS build's primitive set.
    /// Changes only on residency transitions.
    resident_ids: Vec<u32>,
    pub instance_buffer_id: Id<Buffer>,
    /// The stable TLAS, rebuilt in place on transitions (storage sized for
    /// the full lattice — the bindless id never moves).
    tlas: Arc<AccelerationStructure>,
    tlas_storage_size: u64,
    /// The TLAS has been built at least once (the empty-world corner: a
    /// world with no initial Regions still needs one legal build).
    tlas_initialized: bool,

    // --- per-Region residency -------------------------------------------
    /// One slot per Region id (4096); `Some` iff resident.
    regions: Vec<Option<ResidentRegion>>,
    /// Region id → pool device address (the region table's CPU mirror).
    table_addresses: Vec<u64>,
    free: FreeLists,
    pending_free: PendingFrees,

    /// A never-hit dummy BLAS: keeps the TLAS build legal (≥1 primitive)
    /// when the resident set is empty, without null AS references.
    dummy_blas: Arc<AccelerationStructure>,

    /// Allocation probes (harness/tests): fresh vs free-list reuse.
    pub alloc_stats: AllocStats,
}

impl RegionStore {
    /// Builds the static lattice over the world's initial snapshot batch —
    /// the one-shot pre-loop build (user story 25): every initial Region
    /// becomes resident through the same rebuild path as change cycles.
    pub fn new(gpu: &GpuStack, voxel_data: &DotVoxData, initial: Vec<RegionData>) -> Self {
        // --- static buffers ---------------------------------------------
        let camera_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::Camera>(),
            )
            .unwrap();

        // The Scene buffer (ticket 06): the analytic lights' constants —
        // the Sun (direction + illuminance), the Procedural sky's μ-gradient
        // knots, and the disk. Written with the defaults at creation and
        // updated every frame from the render context (tunable).
        let scene_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::Scene>(),
            )
            .unwrap();

        let palette_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::Palette>(),
            )
            .unwrap();

        // The Material table (ADR 0008): one entry per palette index, packed
        // as two vec4[256] columns — albedo.rgb+metallic and
        // emission.rgb+roughness — uploaded once at startup from the CPU
        // mirror (`world::voxel::get_material_table`, the single source of
        // truth). Beside the Palette, bindless like it.
        let material_table_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::MaterialTable>(),
            )
            .unwrap();

        let region_table_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::RegionTable>(),
            )
            .unwrap();

        // The region -> AABB-buffer device-address table, parallel to the
        // region table above: the DDA (and the debug Hull mode) read a
        // Micro-chunk's trimmed hull back through it by primitive id.
        let aabb_table_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<capture_raygen::AabbTable>(),
            )
            .unwrap();

        // --- the stable instance buffer + TLAS ---------------------------
        let instance_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[AccelerationStructureInstance]>(REGION_COUNT as u64)
                    .unwrap(),
            )
            .unwrap();

        let instance_subbuffer =
            Subbuffer::new(gpu.resources.buffer(instance_buffer_id).buffer().clone())
                .cast_aligned::<AccelerationStructureInstance>();
        let (tlas, tlas_storage_size) = acceleration_structure::create_tlas_storage(
            &instance_subbuffer,
            REGION_COUNT as u32,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
        );

        let dummy_blas = create_dummy_blas(gpu);

        // --- palette + material table content (one-shot) -----------------
        // Both are world-static (the world is static per the effort scope),
        // uploaded once and never rewritten — the palette from the .vox
        // palette, the material table as the packed twin of the CPU mirror.
        let palette = get_palette(voxel_data).map(|color| [color.x, color.y, color.z, 1.0]);
        let material_table = get_material_table(voxel_data);
        let albedo_metallic: [[f32; 4]; 256] = std::array::from_fn(|i| {
            let m = &material_table[i];
            [m.albedo[0], m.albedo[1], m.albedo[2], m.metallic]
        });
        let rough_emit: [[f32; 4]; 256] = std::array::from_fn(|i| {
            let m = &material_table[i];
            [m.emission[0], m.emission[1], m.emission[2], m.roughness]
        });
        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<capture_raygen::Palette>(palette_buffer_id, ..) =
                        capture_raygen::Palette { colors: palette };
                    *tcx.write_buffer::<capture_raygen::MaterialTable>(material_table_buffer_id, ..) =
                        capture_raygen::MaterialTable {
                            albedo_metallic,
                            rough_emit,
                        };
                    *tcx.write_buffer::<capture_raygen::Scene>(scene_buffer_id, ..) =
                        crate::region::render::default_scene();
                    Ok(())
                },
                [
                    (palette_buffer_id, HostAccessType::Write),
                    (material_table_buffer_id, HostAccessType::Write),
                    (scene_buffer_id, HostAccessType::Write),
                ],
                [],
                [],
            )
            .unwrap();
        }
        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();

        // --- bindless registrations ---------------------------------------
        let bcx = gpu.resources.bindless_context().unwrap();
        let region_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                region_table_buffer_id,
                0,
                Some(size_of::<capture_raygen::RegionTable>() as DeviceSize),
            )
            .unwrap();
        let camera_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                camera_buffer_id,
                0,
                Some(size_of::<capture_raygen::Camera>() as DeviceSize),
            )
            .unwrap();
        let palette_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                palette_buffer_id,
                0,
                Some(size_of::<capture_raygen::Palette>() as DeviceSize),
            )
            .unwrap();
        let material_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                material_table_buffer_id,
                0,
                Some(size_of::<capture_raygen::MaterialTable>() as DeviceSize),
            )
            .unwrap();
        let scene_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                scene_buffer_id,
                0,
                Some(size_of::<capture_raygen::Scene>() as DeviceSize),
            )
            .unwrap();
        let acceleration_structure_id = bcx.global_set().add_acceleration_structure(tlas.clone());
        let aabb_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                aabb_table_buffer_id,
                0,
                Some(size_of::<capture_raygen::AabbTable>() as DeviceSize),
            )
            .unwrap();

        let mut store = Self {
            camera_buffer_id,
            scene_buffer_id,
            region_table_storage_id,
            camera_storage_id,
            scene_storage_id,
            palette_storage_id,
            material_table_storage_id,
            acceleration_structure_id,
            region_table_buffer_id,
            aabb_table_storage_id,
            instances: static_instances(),
            resident_ids: Vec::new(),
            instance_buffer_id,
            tlas,
            tlas_storage_size,
            tlas_initialized: false,
            regions: (0..REGION_COUNT).map(|_| None).collect(),
            table_addresses: vec![0; REGION_COUNT],
            free: FreeLists::default(),
            pending_free: PendingFrees::default(),
            dummy_blas,
            alloc_stats: AllocStats::default(),
        };

        // --- the initial residency (the one-shot pre-loop build, user
        // story 25): every initial Region becomes resident through the same
        // ordered rebuild path as change cycles.
        let packs: Vec<(IVec3, Option<RegionData>)> = initial
            .into_iter()
            .map(|region| (region.region_index, Some(region)))
            .collect();
        let report = store.rebuild(gpu, packs);
        debug_assert!(
            report.left_resident.is_empty(),
            "the initial batch only creates residency"
        );

        // The empty-world corner: no initial Regions means the rebuild above
        // built no TLAS; still make one legal build (the never-hit dummy
        // instance) so the first traced frame is well-defined.
        if !store.tlas_initialized {
            let instance_buffer = Subbuffer::new(
                gpu.resources
                    .buffer(store.instance_buffer_id)
                    .buffer()
                    .clone(),
            )
            .cast_aligned::<AccelerationStructureInstance>();
            let sizes = tlas_build_sizes(gpu, &instance_buffer, 1);
            debug_assert!(
                store.tlas_storage_size >= sizes.acceleration_structure_size,
                "the empty-world dummy TLAS build must fit the stable storage"
            );
            store.rebuild_with_plan(
                gpu,
                RebuildPlan {
                    instances: Some(store.packed_instance_prefix()),
                    tlas: Some(TlasBuild {
                        instance_count: 1,
                        scratch: allocate_scratch(gpu, sizes.build_scratch_size),
                    }),
                    ..Default::default()
                },
            );
        }

        // The region -> AABB-buffer table is written once, after the initial
        // residency — the world is static, so the AABB-buffer device addresses
        // never change after startup (a live edit path that replaces a BLAS
        // would have to move this write into the rebuild graph).
        store.write_aabb_table(gpu, aabb_table_buffer_id);

        store
    }

    /// Writes the region -> AABB-buffer device-address table once (one entry
    /// per Region id, 0 for non-resident), parallel to the pool region table.
    /// The world is static, so unlike the region table this is a one-shot
    /// startup write rather than a rebuild-graph node.
    fn write_aabb_table(&self, gpu: &GpuStack, aabb_table_buffer_id: Id<Buffer>) {
        let mut bdas = [0u64; REGION_COUNT];
        for (id, region) in self.regions.iter().enumerate() {
            if let Some(region) = region {
                bdas[id] = gpu
                    .resources
                    .buffer(region.aabb_buffer_id)
                    .buffer()
                    .device_address()
                    .get();
            }
        }
        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<capture_raygen::AabbTable>(aabb_table_buffer_id, ..) =
                        capture_raygen::AabbTable { bdas };
                    Ok(())
                },
                [(aabb_table_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
            .unwrap();
        }
        gpu.resources
            .flight(gpu.graphics_flight_id)
            .wait_idle()
            .unwrap();
    }

    /// Consumes one change cycle: the dirty-Region set the input contract
    /// published since the last call, each packed from its mirror (the world
    /// never reaches the pipeline). Requires the worker to be idle (the
    /// harness calls [`RendererInput::wait_until_idle`] first), so the dirty
    /// set and the packs are a consistent snapshot.
    pub fn apply(&mut self, gpu: &GpuStack, input: &RendererInput) -> ApplyReport {
        let dirty = input.take_dirty_regions();
        if dirty.is_empty() {
            return ApplyReport::default();
        }
        let packs: Vec<(IVec3, Option<RegionData>)> = dirty
            .iter()
            .map(|&region| (region, input.packed_region(region)))
            .collect();
        self.rebuild(gpu, packs)
    }

    /// The number of resident Regions.
    pub fn resident_count(&self) -> usize {
        self.resident_ids.len()
    }

    /// The packed resident ids, sorted (the TLAS's instance set).
    pub fn resident_ids(&self) -> &[u32] {
        &self.resident_ids
    }

    /// The resident BLASes — lifetime anchors for the render task (the TLAS
    /// instances reference the BLASes by device address only).
    pub fn blases(&self) -> Vec<Arc<AccelerationStructure>> {
        self.regions
            .iter()
            .filter_map(|region| region.as_ref().map(|region| region.blas.clone()))
            .collect()
    }

    /// The region table buffer id (the rebuild graph's host-write access).
    pub(crate) fn region_table_buffer_id(&self) -> Id<Buffer> {
        self.region_table_buffer_id
    }

    /// The stable TLAS — rebuilt **in place** on residency transitions; the
    /// bindless acceleration-structure id never moves. Exactly one (the new
    /// path has no back-AS double buffer and no flip atomic, so a
    /// double-buffered design would need an array of ASes and an index, which
    /// the store does not have).
    pub(crate) fn tlas(&self) -> Arc<AccelerationStructure> {
        self.tlas.clone()
    }

    /// Applies a change cycle to the lattice: the **plan phase** makes every
    /// CPU-side decision (residency transitions, in-place BLAS rebuilds for
    /// content edits, the TLAS rebuild on transitions, allocations from the
    /// free lists), then the **ordered rebuild nodes** ([`RebuildGraph`])
    /// record and execute the GPU half — pool upload → BLAS build → TLAS
    /// build — between the consuming trace and the next frame. The pending
    /// frees whose dropping rebuild executed are released after the graph
    /// completes.
    fn rebuild(&mut self, gpu: &GpuStack, packs: Vec<(IVec3, Option<RegionData>)>) -> ApplyReport {
        let mut report = ApplyReport {
            instance_count_before: self.resident_ids.len(),
            ..Default::default()
        };

        let mut plan = RebuildPlan::default();
        let mut tlas_dirty = false;
        let mut table_changed = false;

        for (region_index, pack) in packs {
            let id = region_id(region_index) as usize;
            debug_assert!(
                id < REGION_COUNT,
                "region id {id} outside the 12-bit lattice"
            );

            let was_resident = self.regions[id].is_some();
            match (was_resident, pack) {
                // (false, None): a dirty region whose mirror emptied before
                // this cycle — nothing to do (an empty plan means no rebuild
                // graph is compiled or submitted — the idle renderer costs
                // zero, no pending rebuilds, no wakeups).
                (false, None) => {}

                // Become resident: allocate from the free lists, upload the
                // pool + AABBs, build the BLAS into its storage, and add the
                // instance. The storage is created here (CPU) and the node
                // builds into it in place — the AS object and its device
                // address never move after creation.
                (false, Some(pack)) => {
                    let pool = self.allocate_pool(gpu, pack.blocks.len() as u64);
                    let blas_alloc = self.allocate_blas(gpu, pack.aabbs.len() as u32);

                    let aabb_count = pack.aabbs.len() as u32;
                    let aabb_buffer = Subbuffer::new(
                        gpu.resources
                            .buffer(blas_alloc.aabb_buffer_id)
                            .buffer()
                            .clone(),
                    )
                    .cast_aligned::<AabbPositions>();
                    let (blas, blas_storage_size) =
                        resolve_blas_storage(gpu, &aabb_buffer, aabb_count, &blas_alloc);

                    plan.uploads.push(RegionUpload {
                        region_index,
                        pool_buffer_id: pool.buffer_id,
                        pool_bytes: pack.blocks,
                        aabb_buffer_id: blas_alloc.aabb_buffer_id,
                        aabbs: pack.aabbs,
                    });
                    plan.blas_builds.push(plan_blas_build(
                        gpu,
                        region_index,
                        blas_alloc.aabb_buffer_id,
                        &aabb_buffer,
                        aabb_count,
                        blas.clone(),
                        blas_storage_size,
                        true,
                    ));

                    let address = gpu
                        .resources
                        .buffer(pool.buffer_id)
                        .buffer()
                        .device_address()
                        .get();
                    self.instances[id].acceleration_structure_reference =
                        blas.device_address().into();
                    self.insert_resident(id as u32);
                    self.table_addresses[id] = address;
                    table_changed = true;
                    self.regions[id] = Some(ResidentRegion {
                        pool_buffer_id: pool.buffer_id,
                        pool_capacity: pool.capacity,
                        aabb_buffer_id: blas_alloc.aabb_buffer_id,
                        aabb_capacity: blas_alloc.aabb_capacity,
                        blas,
                        blas_storage_size,
                    });
                    report.became_resident.push(region_index);
                    tlas_dirty = true;
                }

                // Leave residency: drop the instance, zero the table entry,
                // and return the memory to the pending frees (reusable only
                // after the dropping rebuild executes). No upload — the
                // instance removal rides the instances rewrite + TLAS build.
                (true, None) => {
                    let region = self.regions[id].take().unwrap();
                    self.remove_resident(id as u32);
                    self.table_addresses[id] = 0;
                    table_changed = true;
                    self.pending_free.pools.push(FreedPool {
                        buffer_id: region.pool_buffer_id,
                        capacity: region.pool_capacity,
                    });
                    self.pending_free.blas.push(FreedBlas {
                        aabb_buffer_id: region.aabb_buffer_id,
                        aabb_capacity: region.aabb_capacity,
                        blas: region.blas.clone(),
                        blas_storage_size: region.blas_storage_size,
                    });
                    report.left_resident.push(region_index);
                    tlas_dirty = true;
                }

                // Content edit: re-upload the pool and rebuild the BLAS in
                // place (device address stable → TLAS untouched). A pack that
                // outgrows its allocations replaces them — the old ones join
                // the pending frees, and a BLAS replacement moves the
                // instance address (the only non-transition TLAS rebuild).
                (true, Some(pack)) => {
                    let pool_grows =
                        self.regions[id].as_ref().unwrap().pool_capacity < pack.blocks.len() as u64;
                    let blas_grows =
                        self.regions[id].as_ref().unwrap().aabb_capacity < pack.aabbs.len() as u32;

                    let new_pool =
                        pool_grows.then(|| self.allocate_pool(gpu, pack.blocks.len() as u64));
                    let new_blas =
                        blas_grows.then(|| self.allocate_blas(gpu, pack.aabbs.len() as u32));
                    let mut blas_replacement: Option<BlasAllocation> = None;

                    if let Some(pool) = new_pool {
                        {
                            let region = self.regions[id].as_mut().unwrap();
                            self.pending_free.pools.push(FreedPool {
                                buffer_id: region.pool_buffer_id,
                                capacity: region.pool_capacity,
                            });
                            region.pool_buffer_id = pool.buffer_id;
                            region.pool_capacity = pool.capacity;
                        }
                        let address = gpu
                            .resources
                            .buffer(pool.buffer_id)
                            .buffer()
                            .device_address()
                            .get();
                        self.table_addresses[id] = address;
                        table_changed = true;
                    }
                    if let Some(alloc) = new_blas {
                        {
                            let region = self.regions[id].as_mut().unwrap();
                            self.pending_free.blas.push(FreedBlas {
                                aabb_buffer_id: region.aabb_buffer_id,
                                aabb_capacity: region.aabb_capacity,
                                blas: region.blas.clone(),
                                blas_storage_size: region.blas_storage_size,
                            });
                            region.aabb_buffer_id = alloc.aabb_buffer_id;
                            region.aabb_capacity = alloc.aabb_capacity;
                        }
                        blas_replacement = Some(alloc);
                    }

                    // The upload node copies the new content at record time;
                    // the BLAS node rebuilds after it (the ordered edge) —
                    // the build reads the AABB buffer, so it must follow the
                    // upload.
                    let (pool_id, aabb_id) = {
                        let region = self.regions[id].as_ref().unwrap();
                        (region.pool_buffer_id, region.aabb_buffer_id)
                    };
                    let aabb_count = pack.aabbs.len() as u32;
                    plan.uploads.push(RegionUpload {
                        region_index,
                        pool_buffer_id: pool_id,
                        pool_bytes: pack.blocks,
                        aabb_buffer_id: aabb_id,
                        aabbs: pack.aabbs,
                    });

                    match blas_replacement {
                        Some(alloc) => {
                            // Fresh storage (CPU-side) + build in place in
                            // the BLAS node; the instance address moves.
                            let aabb_buffer = Subbuffer::new(
                                gpu.resources.buffer(alloc.aabb_buffer_id).buffer().clone(),
                            )
                            .cast_aligned::<AabbPositions>();
                            let (blas, blas_storage_size) =
                                resolve_blas_storage(gpu, &aabb_buffer, aabb_count, &alloc);
                            plan.blas_builds.push(plan_blas_build(
                                gpu,
                                region_index,
                                alloc.aabb_buffer_id,
                                &aabb_buffer,
                                aabb_count,
                                blas.clone(),
                                blas_storage_size,
                                true,
                            ));
                            {
                                let region = self.regions[id].as_mut().unwrap();
                                region.blas = blas.clone();
                                region.blas_storage_size = blas_storage_size;
                            }
                            self.instances[id].acceleration_structure_reference =
                                blas.device_address().into();
                            report.blas_replaced.push(region_index);
                            tlas_dirty = true;
                        }
                        None => {
                            // In-place: the BLAS object and its device
                            // address stay — the TLAS instance is untouched.
                            let (blas, blas_storage_size) = {
                                let region = self.regions[id].as_ref().unwrap();
                                (region.blas.clone(), region.blas_storage_size)
                            };
                            let aabb_buffer =
                                Subbuffer::new(gpu.resources.buffer(aabb_id).buffer().clone())
                                    .cast_aligned::<AabbPositions>();
                            plan.blas_builds.push(plan_blas_build(
                                gpu,
                                region_index,
                                aabb_id,
                                &aabb_buffer,
                                aabb_count,
                                blas,
                                blas_storage_size,
                                false,
                            ));
                        }
                    }
                    report.dirty.push(region_index);
                }
            }
        }

        // --- the plan's GPU half -----------------------------------------
        if table_changed {
            plan.table = Some(
                self.table_addresses
                    .clone()
                    .try_into()
                    .expect("the region table has REGION_COUNT entries"),
            );
        }
        if tlas_dirty {
            let instance_buffer = Subbuffer::new(
                gpu.resources
                    .buffer(self.instance_buffer_id)
                    .buffer()
                    .clone(),
            )
            .cast_aligned::<AccelerationStructureInstance>();
            let instance_count = self.resident_ids.len().max(1) as u32;
            let sizes = tlas_build_sizes(gpu, &instance_buffer, instance_count);
            debug_assert!(
                self.tlas_storage_size >= sizes.acceleration_structure_size,
                "in-place TLAS build for {instance_count} instances exceeds the stable storage"
            );
            plan.instances = Some(self.packed_instance_prefix());
            plan.tlas = Some(TlasBuild {
                instance_count,
                scratch: allocate_scratch(gpu, sizes.build_scratch_size),
            });
        }

        report.rebuild_log = plan.log();
        report.tlas_rebuilt = tlas_dirty;

        if plan.is_empty() {
            // Nothing changed (every dirty region emptied before the cycle):
            // no rebuild graph, no GPU work — the idle renderer costs zero
            // (no pending rebuilds, no wakeups).
            report.instance_count = self.resident_ids.len();
            return report;
        }

        report.timings = self.rebuild_with_plan(gpu, plan);
        report.instance_count = self.resident_ids.len();
        report
    }

    /// Executes one rebuild plan through the ordered nodes, then releases
    /// the pending frees whose dropping rebuild executed (the graph above executed and waited idle, so
    /// the freed memory is safe to reuse). Shared by change cycles and the
    /// empty-world corner's forced dummy build.
    fn rebuild_with_plan(&mut self, gpu: &GpuStack, plan: RebuildPlan) -> NodeTimings {
        let tlas_rebuilds = plan.tlas.is_some();
        let graph = RebuildGraph::new(gpu, self, plan);
        let timings = graph.execute(gpu);
        if tlas_rebuilds {
            self.tlas_initialized = true;
        }
        self.release_pending_frees();
        timings
    }

    /// The packed instance prefix the upload node rewrites: one instance per
    /// resident Region (sorted by id), or the never-hit dummy instance when
    /// nothing is resident (keeps the TLAS build legal).
    fn packed_instance_prefix(&self) -> Vec<AccelerationStructureInstance> {
        packed_prefix(&self.instances, &self.resident_ids, self.dummy_instance())
    }

    /// Allocates a pool buffer (best-fit from the free lists, else fresh).
    fn allocate_pool(&mut self, gpu: &GpuStack, needed: u64) -> PoolAllocation {
        if let Some(freed) = take_best_fit(&mut self.free.pools, needed, |f| f.capacity) {
            self.alloc_stats.pool_reuses += 1;
            PoolAllocation {
                buffer_id: freed.buffer_id,
                capacity: freed.capacity,
            }
        } else {
            let buffer_id = gpu
                .resources
                .create_buffer(
                    &BufferCreateInfo {
                        usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    DeviceLayout::new_unsized::<[u8]>(needed).unwrap(),
                )
                .unwrap();
            self.alloc_stats.pool_allocations += 1;
            PoolAllocation {
                buffer_id,
                capacity: needed,
            }
        }
    }

    /// Allocates a (AABB buffer, BLAS storage) pair (best-fit reuse first).
    fn allocate_blas(&mut self, gpu: &GpuStack, aabb_count: u32) -> BlasAllocation {
        if let Some(freed) = take_best_fit(&mut self.free.blas, aabb_count as u64, |f| {
            f.aabb_capacity as u64
        }) {
            self.alloc_stats.blas_reuses += 1;
            BlasAllocation {
                aabb_buffer_id: freed.aabb_buffer_id,
                aabb_capacity: freed.aabb_capacity,
                as_storage: Some((freed.blas, freed.blas_storage_size)),
            }
        } else {
            let aabb_buffer_id = gpu
                .resources
                .create_buffer(
                    &BufferCreateInfo {
                        // STORAGE_BUFFER lets the debug Hull shader read the
                        // trimmed hulls back as a buffer_reference; the DDA
                        // path (AS build input + device address) is unchanged.
                        usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                            | BufferUsage::SHADER_DEVICE_ADDRESS
                            | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    DeviceLayout::new_unsized::<[AabbPositions]>(aabb_count as u64).unwrap(),
                )
                .unwrap();
            self.alloc_stats.blas_allocations += 1;
            BlasAllocation {
                aabb_buffer_id,
                aabb_capacity: aabb_count,
                as_storage: None,
            }
        }
    }

    /// The never-hit dummy instance (mask 0 → culled by the hardware).
    fn dummy_instance(&self) -> AccelerationStructureInstance {
        AccelerationStructureInstance {
            instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
            acceleration_structure_reference: self.dummy_blas.device_address().into(),
            ..Default::default()
        }
    }

    fn insert_resident(&mut self, id: u32) {
        match self.resident_ids.binary_search(&id) {
            Ok(_) => panic!("region {id} already resident"),
            Err(position) => self.resident_ids.insert(position, id),
        }
    }

    fn remove_resident(&mut self, id: u32) {
        let position = self
            .resident_ids
            .binary_search(&id)
            .unwrap_or_else(|_| panic!("region {id} not resident"));
        self.resident_ids.remove(position);
    }

    /// Releases the pending frees into the reusable lists. Call only after
    /// the rebuild sequence that dropped the referencing instances executed.
    fn release_pending_frees(&mut self) {
        self.free.pools.append(&mut self.pending_free.pools);
        self.free.blas.append(&mut self.pending_free.blas);
    }
}

/// Resolves the storage a BLAS build records into: the free-list-reused AS,
/// or a fresh storage sized exactly for `aabb_count` (created here, built
/// into **in place** by the BLAS node — the AS object and its device address
/// never move after creation).
fn resolve_blas_storage(
    gpu: &GpuStack,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
    alloc: &BlasAllocation,
) -> (Arc<AccelerationStructure>, u64) {
    match &alloc.as_storage {
        Some((blas, storage_size)) => (blas.clone(), *storage_size),
        None => acceleration_structure::create_blas_aabbs_storage(
            aabb_buffer,
            aabb_count,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
        ),
    }
}

/// Builds the plan's BLAS-build entry for one Region: sizes the scratch,
/// asserts the build fits the (resolved) storage, and packs the entry the
/// BLAS node records. `fresh` marks become-resident/replacement builds
/// (new storage) vs in-place content-edit rebuilds.
#[allow(clippy::too_many_arguments)]
fn plan_blas_build(
    gpu: &GpuStack,
    region_index: IVec3,
    aabb_buffer_id: Id<Buffer>,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
    blas: Arc<AccelerationStructure>,
    blas_storage_size: u64,
    fresh: bool,
) -> BlasBuild {
    let sizes = blas_build_sizes(gpu, aabb_buffer, aabb_count);
    debug_assert!(
        blas_storage_size >= sizes.acceleration_structure_size,
        "BLAS build for {aabb_count} AABBs exceeds its {blas_storage_size}-byte storage"
    );
    BlasBuild {
        region_index,
        aabb_buffer_id,
        aabb_count,
        blas,
        scratch: allocate_scratch(gpu, sizes.build_scratch_size),
        fresh,
    }
}

/// Packs the resident ids' instance slots into the TLAS build-input prefix
/// (sorted by id — the order [`RegionStore::resident_ids`] keeps); the
/// never-hit dummy instance when nothing is resident, so the TLAS build
/// stays legal (pure — unit-tested).
fn packed_prefix(
    instances: &[AccelerationStructureInstance],
    resident_ids: &[u32],
    empty_dummy: AccelerationStructureInstance,
) -> Vec<AccelerationStructureInstance> {
    if resident_ids.is_empty() {
        vec![empty_dummy]
    } else {
        resident_ids
            .iter()
            .map(|&id| instances[id as usize])
            .collect()
    }
}

/// The lattice-static instance data: one slot per Region id, transform =
/// translation by the Region origin, custom index = id, mask 0xFF. Only the
/// BLAS address is filled in on residency (pure — unit-tested).
fn static_instances() -> Vec<AccelerationStructureInstance> {
    let mut out = vec![AccelerationStructureInstance::default(); REGION_COUNT];
    for x in -8..8 {
        for y in -8..8 {
            for z in -8..8 {
                let index = IVec3::new(x, y, z);
                let id = region_id(index) as usize;
                let origin = (index * REGION_EDGE).as_vec3().to_array();
                out[id] = AccelerationStructureInstance {
                    transform: [
                        [1.0, 0.0, 0.0, origin[0]],
                        [0.0, 1.0, 0.0, origin[1]],
                        [0.0, 0.0, 1.0, origin[2]],
                    ],
                    instance_custom_index_and_mask: Packed24_8::new(region_id(index), 0xFF),
                    acceleration_structure_reference: 0,
                    ..Default::default()
                };
            }
        }
    }
    out
}

/// A never-hit procedural BLAS (one AABB ~1e9 away — beyond the ray t range
/// RAY_T_MAX = 10000): keeps the TLAS build legal when nothing is resident.
fn create_dummy_blas(gpu: &GpuStack) -> Arc<AccelerationStructure> {
    let aabb = AabbPositions {
        min: [1.0e9; 3],
        max: [1.0e9 + 1.0; 3],
    };
    let buffer = Buffer::from_iter(
        &gpu.memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        std::iter::once(aabb),
    )
    .expect("dummy AABB buffer creation failed");

    acceleration_structure::build_blas_aabbs_fresh(
        buffer,
        1,
        gpu.memory_allocator.clone(),
        gpu.device.clone(),
        gpu.compute_queue.clone(),
        &gpu.resources,
        gpu.compute_flight_id,
    )
    .0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The free list hands out the smallest entry that fits (best-fit), and
    /// never an entry too small.
    #[test]
    fn free_list_best_fit() {
        let invalid = Id::INVALID;
        let mut pools = vec![
            FreedPool {
                buffer_id: invalid,
                capacity: 100,
            },
            FreedPool {
                buffer_id: invalid,
                capacity: 8,
            },
            FreedPool {
                buffer_id: invalid,
                capacity: 200,
            },
        ];

        let taken = take_best_fit(&mut pools, 64, |f| f.capacity).unwrap();
        assert_eq!(taken.capacity, 100, "smallest fitting entry wins");
        assert_eq!(pools.len(), 2);

        // Nothing big enough: None, the list is untouched.
        assert!(take_best_fit(&mut pools, 500, |f| f.capacity).is_none());
        assert_eq!(pools.len(), 2);

        // An exact fit is preferred over a larger one.
        let taken = take_best_fit(&mut pools, 8, |f| f.capacity).unwrap();
        assert_eq!(taken.capacity, 8);
    }

    /// Pending frees are not reusable until released (the ordering invariant
    /// at the list level: allocation only sees the released lists).
    #[test]
    fn pending_frees_release_into_reusable_lists() {
        let mut pending = PendingFrees::default();
        pending.pools.push(FreedPool {
            buffer_id: Id::INVALID,
            capacity: 64,
        });

        // Not yet released: allocation cannot see it.
        let mut free = FreeLists::default();
        assert!(take_best_fit(&mut free.pools, 64, |f| f.capacity).is_none());

        // Release (after the dropping rebuild executed) → reusable.
        free.pools.append(&mut pending.pools);
        let taken = take_best_fit(&mut free.pools, 64, |f| f.capacity).unwrap();
        assert_eq!(taken.capacity, 64);
        assert!(pending.pools.is_empty());
    }

    /// Every Region id in the full ±2048/axis lattice (16^3 = 4096 Regions)
    /// fits the 12-bit budget and is unique.
    #[test]
    fn region_ids_fit_12bit_budget_at_full_lattice() {
        let mut seen = std::collections::HashSet::new();
        for x in -8..8 {
            for y in -8..8 {
                for z in -8..8 {
                    let id = region_id(IVec3::new(x, y, z));
                    assert!(id < (1 << 12) as u32, "region id {id} exceeds 12 bits");
                    assert!(seen.insert(id), "region id {id} collides");
                }
            }
        }
        assert_eq!(seen.len(), REGION_COUNT);
    }

    /// The lattice-static instance data: one slot per id, transform =
    /// translation by the Region origin, custom index = id, mask 0xFF, and
    /// no BLAS address until residency.
    #[test]
    fn static_instance_data_is_lattice_static() {
        let instances = static_instances();
        assert_eq!(instances.len(), REGION_COUNT);

        for index in [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 2, 3),
            IVec3::new(7, -8, 0),
        ] {
            let id = region_id(index) as usize;
            let instance = &instances[id];
            let origin = (index * REGION_EDGE).as_vec3().to_array();
            assert_eq!(instance.transform[0], [1.0, 0.0, 0.0, origin[0]]);
            assert_eq!(instance.transform[1], [0.0, 1.0, 0.0, origin[1]]);
            assert_eq!(instance.transform[2], [0.0, 0.0, 1.0, origin[2]]);
            assert_eq!(instance.instance_custom_index_and_mask.low_24(), id as u32);
            assert_eq!(instance.instance_custom_index_and_mask.high_8(), 0xFF);
            assert_eq!(instance.acceleration_structure_reference, 0);
        }
    }

    /// The packed instance prefix is the resident ids' instance slots in
    /// sorted order — exactly the TLAS build input — and the never-hit dummy
    /// when nothing is resident (the empty-world corner stays legal).
    #[test]
    fn packed_prefix_rewrites_resident_instances() {
        let instances = static_instances();
        let dummy = AccelerationStructureInstance {
            instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
            ..Default::default()
        };

        // Sorted resident ids → their lattice-static slots in order.
        let prefix = packed_prefix(&instances, &[2, 5, 9], dummy);
        assert_eq!(prefix.len(), 3);
        assert_eq!(prefix[0].instance_custom_index_and_mask.low_24(), 2);
        assert_eq!(prefix[1].instance_custom_index_and_mask.low_24(), 5);
        assert_eq!(prefix[2].instance_custom_index_and_mask.low_24(), 9);

        // Empty resident set → exactly one dummy instance (legal TLAS build).
        let prefix = packed_prefix(&instances, &[], dummy);
        assert_eq!(prefix, vec![dummy]);
    }
}
