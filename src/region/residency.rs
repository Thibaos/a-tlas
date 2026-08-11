//! Multi-region residency (renderer-impl ticket 04): the full static lattice.
//!
//! The renderer owns the lattice (ADR 0004): Region = 256^3 voxels,
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
//!
//! The ticket-04 rebuilds run synchronously (execute + wait_idle) between
//! frames; ticket 05 turns them into ordered taskgraph nodes. The free-list
//! ordering invariant is structural here: memory freed by a residency-leave
//! goes to [`PendingFrees`] and is only released to the reusable lists after
//! the TLAS rebuild that dropped the referencing instance has executed — a
//! later cycle's allocation can reuse it only from that point on.

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
    region::{
        input::RendererInput,
        pack::{REGION_COUNT, REGION_EDGE, RegionData, region_id},
        render::capture_raygen,
    },
    rt::acceleration_structure,
    world::voxel::get_palette,
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
/// that dropped the referencing instance must execute first (ticket 04's
/// ordering invariant — see [`RegionStore::rebuild`]).
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
    pub region_table_storage_id: StorageBufferId,
    pub camera_storage_id: StorageBufferId,
    pub palette_storage_id: StorageBufferId,
    pub acceleration_structure_id: AccelerationStructureId,
    region_table_buffer_id: Id<Buffer>,

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

        // --- palette content (one-shot) ----------------------------------
        let palette = get_palette(voxel_data).map(|color| [color.x, color.y, color.z, 1.0]);
        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<capture_raygen::Palette>(palette_buffer_id, ..) =
                        capture_raygen::Palette { colors: palette };
                    Ok(())
                },
                [(palette_buffer_id, HostAccessType::Write)],
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
        let acceleration_structure_id = bcx.global_set().add_acceleration_structure(tlas.clone());

        let mut store = Self {
            camera_buffer_id,
            region_table_storage_id,
            camera_storage_id,
            palette_storage_id,
            acceleration_structure_id,
            region_table_buffer_id,
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

        // --- the initial residency (same rebuild path as change cycles) ---
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
        // built no TLAS; still make one legal build (the dummy instance) so
        // the first traced frame is well-defined.
        if !store.tlas_initialized {
            store.rewrite_instances(gpu);
            store.rebuild_tlas(gpu);
        }

        store
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

    /// Applies a change cycle to the lattice: residency transitions, in-place
    /// BLAS rebuilds for content edits, and the TLAS rebuild on transitions —
    /// then releases the frees whose dropping rebuild executed.
    fn rebuild(&mut self, gpu: &GpuStack, packs: Vec<(IVec3, Option<RegionData>)>) -> ApplyReport {
        let mut report = ApplyReport::default();
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
                // this cycle — nothing to do.
                (false, None) => {}

                // Become resident: allocate from the free lists, upload the
                // pool + AABBs, build the BLAS, and add the instance.
                (false, Some(pack)) => {
                    let pool = self.allocate_pool(gpu, pack.blocks.len() as u64);
                    let blas_alloc = self.allocate_blas(gpu, pack.aabbs.len() as u32);

                    self.upload_region(gpu, pool.buffer_id, &pack, blas_alloc.aabb_buffer_id);
                    let (blas, blas_storage_size) =
                        self.build_region_blas(gpu, &blas_alloc, pack.aabbs.len() as u32);

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
                // after the dropping rebuild executes).
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

                    // Upload the new content, then rebuild the BLAS — the
                    // build reads the AABB buffer, so it must follow the
                    // upload.
                    let (pool_id, aabb_id) = {
                        let region = self.regions[id].as_ref().unwrap();
                        (region.pool_buffer_id, region.aabb_buffer_id)
                    };
                    self.upload_region(gpu, pool_id, &pack, aabb_id);

                    match blas_replacement {
                        Some(alloc) => {
                            let (blas, blas_storage_size) =
                                self.build_region_blas(gpu, &alloc, pack.aabbs.len() as u32);
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
                            let (blas, storage_size) = {
                                let region = self.regions[id].as_ref().unwrap();
                                (region.blas.clone(), region.blas_storage_size)
                            };
                            let subbuffer =
                                Subbuffer::new(gpu.resources.buffer(aabb_id).buffer().clone())
                                    .cast_aligned::<AabbPositions>();
                            acceleration_structure::build_blas_aabbs_in_place(
                                subbuffer,
                                pack.aabbs.len() as u32,
                                &blas,
                                storage_size,
                                gpu.memory_allocator.clone(),
                                gpu.device.clone(),
                                gpu.compute_queue.clone(),
                                &gpu.resources,
                                gpu.compute_flight_id,
                            );
                        }
                    }
                    report.dirty.push(region_index);
                }
            }
        }

        // --- publish the cycle to the GPU buffers -------------------------
        if table_changed {
            self.write_region_table(gpu);
        }
        if tlas_dirty {
            self.rewrite_instances(gpu);
            self.rebuild_tlas(gpu);
            self.tlas_initialized = true;
        }

        // The rebuild sequence that dropped the referencing instances has
        // executed (synchronous: every execute above waited idle) — the
        // freed memory is now safe to reuse.
        self.release_pending_frees();

        report.tlas_rebuilt = tlas_dirty;
        report
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
                        usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                            | BufferUsage::SHADER_DEVICE_ADDRESS,
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

    /// Uploads one Region's pool bytes and trimmed AABBs (synchronous).
    fn upload_region(
        &self,
        gpu: &GpuStack,
        pool_id: Id<Buffer>,
        pack: &RegionData,
        aabb_id: Id<Buffer>,
    ) {
        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[u8]>(pool_id, 0..pack.blocks.len() as DeviceSize)
                        .copy_from_slice(&pack.blocks);
                    let dst = tcx.write_buffer::<[AabbPositions]>(
                        aabb_id,
                        0..(pack.aabbs.len() as DeviceSize
                            * size_of::<AabbPositions>() as DeviceSize),
                    );
                    for (slot, aabb) in dst.iter_mut().zip(pack.aabbs.iter().copied()) {
                        *slot = aabb;
                    }
                    Ok(())
                },
                [
                    (pool_id, HostAccessType::Write),
                    (aabb_id, HostAccessType::Write),
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
    }

    /// Builds a Region's BLAS: into the reused storage, or a fresh one sized
    /// for `aabb_count`. Returns the BLAS and its storage size.
    fn build_region_blas(
        &self,
        gpu: &GpuStack,
        alloc: &BlasAllocation,
        aabb_count: u32,
    ) -> (Arc<AccelerationStructure>, u64) {
        let subbuffer = Subbuffer::new(gpu.resources.buffer(alloc.aabb_buffer_id).buffer().clone())
            .cast_aligned::<AabbPositions>();
        match &alloc.as_storage {
            Some((blas, storage_size)) => (
                acceleration_structure::build_blas_aabbs_in_place(
                    subbuffer,
                    aabb_count,
                    blas,
                    *storage_size,
                    gpu.memory_allocator.clone(),
                    gpu.device.clone(),
                    gpu.compute_queue.clone(),
                    &gpu.resources,
                    gpu.compute_flight_id,
                ),
                *storage_size,
            ),
            None => acceleration_structure::build_blas_aabbs_fresh(
                subbuffer,
                aabb_count,
                gpu.memory_allocator.clone(),
                gpu.device.clone(),
                gpu.compute_queue.clone(),
                &gpu.resources,
                gpu.compute_flight_id,
            ),
        }
    }

    /// Writes the region table (Region id → pool device address) wholesale.
    fn write_region_table(&self, gpu: &GpuStack) {
        let table = capture_raygen::RegionTable {
            bdas: self
                .table_addresses
                .clone()
                .try_into()
                .expect("the region table has REGION_COUNT entries"),
        };
        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<capture_raygen::RegionTable>(
                        self.region_table_buffer_id,
                        ..,
                    ) = table;
                    Ok(())
                },
                [(self.region_table_buffer_id, HostAccessType::Write)],
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

    /// Rewrites the instance buffer's packed resident prefix (the TLAS build
    /// input). With an empty resident set, writes the never-hit dummy
    /// instance so the TLAS build stays legal.
    fn rewrite_instances(&self, gpu: &GpuStack) {
        let count = self.resident_ids.len().max(1);
        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    let dst = tcx.write_buffer::<[AccelerationStructureInstance]>(
                        self.instance_buffer_id,
                        0..(count as DeviceSize
                            * size_of::<AccelerationStructureInstance>() as DeviceSize),
                    );
                    if self.resident_ids.is_empty() {
                        dst[0] = self.dummy_instance();
                    } else {
                        for (slot, id) in dst.iter_mut().zip(self.resident_ids.iter().copied()) {
                            *slot = self.instances[id as usize];
                        }
                    }
                    Ok(())
                },
                [(self.instance_buffer_id, HostAccessType::Write)],
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

    /// Rebuilds the stable TLAS in place over the packed resident prefix.
    fn rebuild_tlas(&self, gpu: &GpuStack) {
        let instance_subbuffer = Subbuffer::new(
            gpu.resources
                .buffer(self.instance_buffer_id)
                .buffer()
                .clone(),
        )
        .cast_aligned::<AccelerationStructureInstance>();
        acceleration_structure::build_tlas_in_place(
            instance_subbuffer,
            self.resident_ids.len().max(1) as u32,
            &self.tlas,
            self.tlas_storage_size,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
            gpu.compute_queue.clone(),
            &gpu.resources,
            gpu.compute_flight_id,
        );
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
    /// the rebuild sequence that dropped the referencing instances executed
    /// (ticket 04's ordering invariant; ticket 05 keeps the same structure
    /// with ordered taskgraph nodes).
    fn release_pending_frees(&mut self) {
        self.free.pools.append(&mut self.pending_free.pools);
        self.free.blas.append(&mut self.pending_free.blas);
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
}
