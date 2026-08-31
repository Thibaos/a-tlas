//! Multi-region residency: the full static lattice.
//!
//! The renderer owns the lattice: Region = 256^3 voxels,
//! origin-aligned, v1 extent ±2048/axis → 16^3 = 4096 Regions, exactly the
//! 12-bit region-id budget. [`RegionStore`] is the GPU half of that lattice:
//! per-Region voxel pools and trimmed-AABB BLASes exist across the lattice;
//! a Region becomes resident on its first non-empty Micro-chunk (a pool
//! buffer + a procedural AABB BLAS, allocated from free lists) and leaves
//! residency on its last (memory returned to the free lists; the CPU mirror
//! is freed with the Region by the input contract). The TLAS holds one
//! instance per resident region. Lattice-static transform, custom index =
//! region id, mask 0xFF, added on residency, removed on region-empty, and
//! rebuilt in place so the bindless acceleration-structure id never
//! moves.
pub(crate) const CACHE_TABLE_ENTRIES: u64 = 1 << 23;
const CACHE_KEY_BYTES: u64 = CACHE_TABLE_ENTRIES * 8;
const CACHE_RECORD_BYTES: u64 = CACHE_TABLE_ENTRIES * 16;
pub(crate) const CACHE_DIRTY_WORDS: usize = 128;

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

use crate::core::{
    render::{
        accel,
        gpu::GpuDesc,
        region::{
            alloc::{
                AllocStats, BlasAllocation, FreeLists, FreedBlas, FreedPool, PendingFrees,
                allocate_blas, allocate_pool,
            },
            decision::{RegionEffect, RegionSlot, decide},
            feed::RendererInput,
            pack::{REGION_COUNT, RegionData},
            rebuild::{
                BlasBuild, RebuildGraph, RebuildLogEntry, RebuildPlan, RegionUpload, TlasBuild,
            },
            task::{default_scene, production_raygen},
        },
    },
    world::{
        format::get_palette,
        grid::{REGION_LENGTH, region_id},
        material::{get_material_table, MATFLAG_GLASS},
    },
};

struct ResidentRegion {
    pool_buffer_id: Id<Buffer>,
    pool_capacity: u64,
    aabb_buffer_id: Id<Buffer>,
    aabb_capacity: u32,
    blas: Arc<AccelerationStructure>,
    blas_storage_size: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub became_resident: Vec<IVec3>,
    pub left_resident: Vec<IVec3>,
    pub dirty: Vec<IVec3>,
    pub blas_replaced: Vec<IVec3>,
    pub tlas_rebuilt: bool,
    pub rebuild_log: Vec<RebuildLogEntry>,
    pub instance_count_before: usize,
    pub instance_count: usize,
}

#[derive(Clone, Copy)]
pub struct RegionBindings {
    pub camera_buffer_id: Id<Buffer>,
    pub scene_buffer_id: Id<Buffer>,
    pub cache_state_buffer_id: Id<Buffer>,
    pub region_table_storage_id: StorageBufferId,
    pub camera_storage_id: StorageBufferId,
    pub scene_storage_id: StorageBufferId,
    pub palette_storage_id: StorageBufferId,
    pub material_table_storage_id: StorageBufferId,
    pub acceleration_structure_id: AccelerationStructureId,
    pub aabb_table_storage_id: StorageBufferId,
    pub cache_dirty_buffer_id: Id<Buffer>,
    pub cache_state_storage_id: StorageBufferId,
    pub instance_buffer_id: Id<Buffer>,
}

pub struct RegionStore {
    pub bindings: RegionBindings,
    region_table_buffer_id: Id<Buffer>,
    aabb_table_buffer_id: Id<Buffer>,
    cache_keys_buffer_id: Id<Buffer>,
    cache_accum_buffer_id: Id<Buffer>,
    cache_resolved_buffer_id: Id<Buffer>,
    cache_dirty_buffer_id: Id<Buffer>,
    cache_stats_buffer_id: Id<Buffer>,
    instances: Vec<AccelerationStructureInstance>,
    resident_ids: Vec<u32>,
    tlas: Arc<AccelerationStructure>,
    tlas_storage_size: u64,
    tlas_initialized: bool,
    regions: Vec<Option<ResidentRegion>>,
    table_addresses: Vec<u64>,
    cache_stats_enabled: bool,
    free: FreeLists,
    pending_free: PendingFrees,
    dummy_blas: Arc<AccelerationStructure>,
    alloc_stats: AllocStats,
}

impl RegionStore {
    pub fn new(gpu: &GpuDesc, voxel_data: &DotVoxData, input: &RendererInput) -> Self {
        input.wait_until_idle();
        let initial = input.packed_regions();

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
                DeviceLayout::new_sized::<production_raygen::Camera>(),
            )
            .unwrap();

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
                DeviceLayout::new_sized::<production_raygen::Scene>(),
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
                DeviceLayout::new_sized::<production_raygen::Palette>(),
            )
            .unwrap();

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
                DeviceLayout::new_sized::<production_raygen::MaterialTable>(),
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
                DeviceLayout::new_sized::<production_raygen::RegionTable>(),
            )
            .unwrap();

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
                DeviceLayout::new_sized::<production_raygen::AabbTable>(),
            )
            .unwrap();

        let cache_keys_buffer_id = gpu
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
                DeviceLayout::new_unsized::<[u8]>(CACHE_KEY_BYTES).unwrap(),
            )
            .unwrap();

        let cache_accum_buffer_id = gpu
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
                DeviceLayout::new_unsized::<[u8]>(CACHE_RECORD_BYTES).unwrap(),
            )
            .unwrap();

        let cache_resolved_buffer_id = gpu
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
                DeviceLayout::new_unsized::<[u8]>(CACHE_RECORD_BYTES).unwrap(),
            )
            .unwrap();

        let cache_dirty_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<[u32; CACHE_DIRTY_WORDS]>(),
            )
            .unwrap();

        let cache_state_buffer_id = gpu
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
                DeviceLayout::new_sized::<production_raygen::CacheState>(),
            )
            .unwrap();

        let cache_stats_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<[u32; 7 * REGION_COUNT]>(),
            )
            .unwrap();

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
        let (tlas, tlas_storage_size) = accel::create_tlas_storage(
            &instance_subbuffer,
            REGION_COUNT as u32,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
        );

        let dummy_blas = create_dummy_blas(gpu);

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
        let flags: [u32; 256] =
            std::array::from_fn(|i| if material_table[i].glass { MATFLAG_GLASS } else { 0 });

        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<production_raygen::Palette>(palette_buffer_id, ..) =
                        production_raygen::Palette { colors: palette };
                    *tcx.write_buffer::<production_raygen::MaterialTable>(
                        material_table_buffer_id,
                        ..,
                    ) = production_raygen::MaterialTable {
                        albedo_metallic,
                        rough_emit,
                        flags,
                    };
                    *tcx.write_buffer::<production_raygen::Scene>(scene_buffer_id, ..) =
                        default_scene();
                    tcx.write_buffer::<[u8]>(cache_keys_buffer_id, 0..CACHE_KEY_BYTES).fill(0);
                    tcx.write_buffer::<[u8]>(cache_accum_buffer_id, 0..CACHE_RECORD_BYTES)
                        .fill(0);
                    tcx.write_buffer::<[u8]>(cache_resolved_buffer_id, 0..CACHE_RECORD_BYTES)
                        .fill(0);
                    tcx.write_buffer::<[u32; CACHE_DIRTY_WORDS]>(cache_dirty_buffer_id, ..)
                        .fill(0);
                    Ok(())
                },
                [
                    (palette_buffer_id, HostAccessType::Write),
                    (material_table_buffer_id, HostAccessType::Write),
                    (scene_buffer_id, HostAccessType::Write),
                    (cache_keys_buffer_id, HostAccessType::Write),
                    (cache_accum_buffer_id, HostAccessType::Write),
                    (cache_resolved_buffer_id, HostAccessType::Write),
                    (cache_dirty_buffer_id, HostAccessType::Write),
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

        let bcx = gpu.resources.bindless_context().unwrap();

        let region_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                region_table_buffer_id,
                0,
                Some(size_of::<production_raygen::RegionTable>() as DeviceSize),
            )
            .unwrap();

        let camera_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                camera_buffer_id,
                0,
                Some(size_of::<production_raygen::Camera>() as DeviceSize),
            )
            .unwrap();

        let palette_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                palette_buffer_id,
                0,
                Some(size_of::<production_raygen::Palette>() as DeviceSize),
            )
            .unwrap();

        let material_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                material_table_buffer_id,
                0,
                Some(size_of::<production_raygen::MaterialTable>() as DeviceSize),
            )
            .unwrap();

        let scene_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                scene_buffer_id,
                0,
                Some(size_of::<production_raygen::Scene>() as DeviceSize),
            )
            .unwrap();

        let acceleration_structure_id = bcx.global_set().add_acceleration_structure(tlas.clone());

        let aabb_table_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                aabb_table_buffer_id,
                0,
                Some(size_of::<production_raygen::AabbTable>() as DeviceSize),
            )
            .unwrap();

        let cache_state_storage_id = bcx
            .global_set()
            .create_storage_buffer(
                cache_state_buffer_id,
                0,
                Some(size_of::<production_raygen::CacheState>() as DeviceSize),
            )
            .unwrap();

        let bindings = RegionBindings {
            camera_buffer_id,
            scene_buffer_id,
            cache_state_buffer_id,
            region_table_storage_id,
            camera_storage_id,
            scene_storage_id,
            palette_storage_id,
            material_table_storage_id,
            acceleration_structure_id,
            aabb_table_storage_id,
            cache_dirty_buffer_id,
            cache_state_storage_id,
            instance_buffer_id,
        };

        let mut store = Self {
            bindings,
            region_table_buffer_id,
            aabb_table_buffer_id,
            cache_keys_buffer_id,
            cache_accum_buffer_id,
            cache_resolved_buffer_id,
            cache_dirty_buffer_id,
            cache_stats_buffer_id,
            instances: static_instances(),
            resident_ids: Vec::new(),
            tlas,
            tlas_storage_size,
            tlas_initialized: false,
            regions: (0..REGION_COUNT).map(|_| None).collect(),
            table_addresses: vec![0; REGION_COUNT],
            cache_stats_enabled: std::env::var("ATLAS_RT_CACHE_STATS").is_ok(),
            free: FreeLists::default(),
            pending_free: PendingFrees::default(),
            dummy_blas,
            alloc_stats: AllocStats::default(),
        };

        let packs: Vec<(IVec3, Option<RegionData>)> = initial
            .into_iter()
            .map(|region| (region.region_index, Some(region)))
            .collect();
        let report = store.rebuild(gpu, packs);
        debug_assert!(
            report.left_resident.is_empty(),
            "the initial batch only creates residency"
        );

        if !store.tlas_initialized {
            let instance_buffer = Subbuffer::new(
                gpu.resources
                    .buffer(store.bindings.instance_buffer_id)
                    .buffer()
                    .clone(),
            )
            .cast_aligned::<AccelerationStructureInstance>();

            let sizes = accel::tlas_build_sizes(gpu, &instance_buffer, 1);

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
                        scratch: accel::allocate_scratch(gpu, sizes.build_scratch_size),
                    }),
                    ..Default::default()
                },
            );
        }

        store.write_aabb_table(gpu, aabb_table_buffer_id);

        if store.cache_stats_enabled() {
            store.print_table_stats();
        }

        store
    }

    fn write_aabb_table(&self, gpu: &GpuDesc, aabb_table_buffer_id: Id<Buffer>) {
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
                    *tcx.write_buffer::<production_raygen::AabbTable>(aabb_table_buffer_id, ..) =
                        production_raygen::AabbTable { bdas };
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

    pub fn apply(&mut self, gpu: &GpuDesc, input: &RendererInput) -> ApplyReport {
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

    pub fn blases(&self) -> Vec<Arc<AccelerationStructure>> {
        self.regions
            .iter()
            .filter_map(|region| region.as_ref().map(|region| region.blas.clone()))
            .collect()
    }

    pub(crate) fn region_table_buffer_id(&self) -> Id<Buffer> {
        self.region_table_buffer_id
    }

    pub(crate) fn tlas(&self) -> Arc<AccelerationStructure> {
        self.tlas.clone()
    }

    fn rebuild(&mut self, gpu: &GpuDesc, packs: Vec<(IVec3, Option<RegionData>)>) -> ApplyReport {
        let slots: Vec<Option<RegionSlot>> = self
            .regions
            .iter()
            .map(|region| {
                region.as_ref().map(|region| RegionSlot {
                    pool_capacity: region.pool_capacity,
                    aabb_capacity: region.aabb_capacity,
                })
            })
            .collect();

        let instance_count_before = self.resident_ids.len();
        let decision = decide(&slots, &self.resident_ids, packs);
        self.resident_ids = decision.resident_ids;

        let mut report = ApplyReport {
            instance_count_before,
            became_resident: decision.became_resident,
            left_resident: decision.left_resident,
            dirty: decision.dirty,
            blas_replaced: decision.blas_replaced,
            tlas_rebuilt: decision.tlas_dirty,
            ..Default::default()
        };

        let mut plan = RebuildPlan::default();

        for (id, effect) in decision.effects {
            match effect {
                RegionEffect::Ignore => {}

                RegionEffect::Enter {
                    pool_bytes,
                    aabbs,
                    pack,
                } => {
                    let region_index = pack.region_index;
                    let pool =
                        allocate_pool(gpu, &mut self.free, &mut self.alloc_stats, pool_bytes);
                    let blas_alloc =
                        allocate_blas(gpu, &mut self.free, &mut self.alloc_stats, aabbs);
                    let aabb_count = aabbs;
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
                    self.instances[id as usize].acceleration_structure_reference =
                        blas.device_address().into();
                    self.table_addresses[id as usize] = address;
                    self.regions[id as usize] = Some(ResidentRegion {
                        pool_buffer_id: pool.buffer_id,
                        pool_capacity: pool.capacity,
                        aabb_buffer_id: blas_alloc.aabb_buffer_id,
                        aabb_capacity: blas_alloc.aabb_capacity,
                        blas,
                        blas_storage_size,
                    });
                }

                RegionEffect::Exit {
                    retire_pool,
                    retire_blas,
                } => {
                    let region = self.regions[id as usize].take().unwrap();
                    self.table_addresses[id as usize] = 0;

                    self.pending_free.pools.push(FreedPool {
                        buffer_id: region.pool_buffer_id,
                        capacity: retire_pool,
                    });

                    self.pending_free.blas.push(FreedBlas {
                        aabb_buffer_id: region.aabb_buffer_id,
                        aabb_capacity: retire_blas,
                        blas: region.blas.clone(),
                        blas_storage_size: region.blas_storage_size,
                    });
                }

                RegionEffect::Update {
                    pool_bytes,
                    aabbs,
                    retire_pool,
                    retire_blas,
                    pack,
                } => {
                    let region_index = pack.region_index;
                    let new_pool = retire_pool.is_some().then(|| {
                        allocate_pool(gpu, &mut self.free, &mut self.alloc_stats, pool_bytes)
                    });

                    let new_blas = retire_blas
                        .is_some()
                        .then(|| allocate_blas(gpu, &mut self.free, &mut self.alloc_stats, aabbs));

                    let mut blas_replacement: Option<BlasAllocation> = None;

                    if let Some(pool) = new_pool {
                        {
                            let region = self.regions[id as usize].as_mut().unwrap();
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
                        self.table_addresses[id as usize] = address;
                    }

                    if let Some(alloc) = new_blas {
                        {
                            let region = self.regions[id as usize].as_mut().unwrap();
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

                    let (pool_id, aabb_id) = {
                        let region = self.regions[id as usize].as_ref().unwrap();
                        (region.pool_buffer_id, region.aabb_buffer_id)
                    };

                    let aabb_count = aabbs;

                    plan.uploads.push(RegionUpload {
                        region_index,
                        pool_buffer_id: pool_id,
                        pool_bytes: pack.blocks,
                        aabb_buffer_id: aabb_id,
                        aabbs: pack.aabbs,
                    });

                    match blas_replacement {
                        Some(alloc) => {
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
                                let region = self.regions[id as usize].as_mut().unwrap();
                                region.blas = blas.clone();
                                region.blas_storage_size = blas_storage_size;
                            }

                            self.instances[id as usize].acceleration_structure_reference =
                                blas.device_address().into();
                        }
                        None => {
                            let (blas, blas_storage_size) = {
                                let region = self.regions[id as usize].as_ref().unwrap();
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

                }
            }
        }

        if decision.table_changed {
            plan.table = Some(
                self.table_addresses
                    .clone()
                    .try_into()
                    .expect("the region table has REGION_COUNT entries"),
            );
        }

        if decision.tlas_dirty {
            let instance_buffer = Subbuffer::new(
                gpu.resources
                    .buffer(self.bindings.instance_buffer_id)
                    .buffer()
                    .clone(),
            )
            .cast_aligned::<AccelerationStructureInstance>();

            let instance_count = self.resident_ids.len().max(1) as u32;
            let sizes = accel::tlas_build_sizes(gpu, &instance_buffer, instance_count);

            debug_assert!(
                self.tlas_storage_size >= sizes.acceleration_structure_size,
                "in-place TLAS build for {instance_count} instances exceeds the stable storage"
            );

            plan.instances = Some(self.packed_instance_prefix());
            plan.tlas = Some(TlasBuild {
                instance_count,
                scratch: accel::allocate_scratch(gpu, sizes.build_scratch_size),
            });
        }

        report.rebuild_log = plan.log();

        if plan.is_empty() {
            report.instance_count = self.resident_ids.len();
            return report;
        }

        self.rebuild_with_plan(gpu, plan);

        if report.tlas_rebuilt {
            self.write_aabb_table(gpu, self.aabb_table_buffer_id);
        }

        report.instance_count = self.resident_ids.len();
        report
    }
    fn rebuild_with_plan(&mut self, gpu: &GpuDesc, plan: RebuildPlan) {
        let tlas_rebuilds = plan.tlas.is_some();
        let graph = RebuildGraph::new(gpu, self, plan);
        graph.execute(gpu);

        if tlas_rebuilds {
            self.tlas_initialized = true;
        }

        self.release_pending_frees();
    }

    fn packed_instance_prefix(&self) -> Vec<AccelerationStructureInstance> {
        packed_prefix(&self.instances, &self.resident_ids, self.dummy_instance())
    }

    fn dummy_instance(&self) -> AccelerationStructureInstance {
        AccelerationStructureInstance {
            instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
            acceleration_structure_reference: self.dummy_blas.device_address().into(),
            ..Default::default()
        }
    }

    fn release_pending_frees(&mut self) {
        self.free.pools.append(&mut self.pending_free.pools);
        self.free.blas.append(&mut self.pending_free.blas);
    }

    pub(crate) fn cache_stats_enabled(&self) -> bool {
        self.cache_stats_enabled
    }

    pub(crate) fn initial_cache_state(&self, gpu: &GpuDesc) -> production_raygen::CacheState {
        production_raygen::CacheState {
            stats_bda: gpu
                .resources
                .buffer(self.cache_stats_buffer_id)
                .buffer()
                .device_address()
                .get(),
            keys_bda: gpu
                .resources
                .buffer(self.cache_keys_buffer_id)
                .buffer()
                .device_address()
                .get(),
            accum_bda: gpu
                .resources
                .buffer(self.cache_accum_buffer_id)
                .buffer()
                .device_address()
                .get(),
            resolved_bda: gpu
                .resources
                .buffer(self.cache_resolved_buffer_id)
                .buffer()
                .device_address()
                .get(),
            dirty_bda: gpu
                .resources
                .buffer(self.cache_dirty_buffer_id)
                .buffer()
                .device_address()
                .get(),
            frame_index: 1,
            event_frames: 0,
            stats_enabled: self.cache_stats_enabled as u32,
        }
    }

    // 02's global tier: whatever resets NRD's history clears the table, so
    // no entry survives an event as fresh state.
    pub(crate) fn clear_cache_table(&self, gpu: &GpuDesc) {
        let clears: [(Id<Buffer>, u64); 3] = [
            (self.cache_keys_buffer_id, CACHE_KEY_BYTES),
            (self.cache_accum_buffer_id, CACHE_RECORD_BYTES),
            (self.cache_resolved_buffer_id, CACHE_RECORD_BYTES),
        ];

        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    for (buffer_id, bytes) in &clears {
                        tcx.write_buffer::<[u8]>(*buffer_id, 0..*bytes).fill(0);
                    }

                    Ok(())
                },
                clears
                    .iter()
                    .map(|(buffer_id, _)| (*buffer_id, HostAccessType::Write))
                    .collect::<Vec<_>>(),
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

    // Sums the per-region counters and zeroes them; the frames are
    // serialized by the wait_idle at the top of run_frame, so the read
    // races nothing.
    pub(crate) fn cache_stats_tick(&self, gpu: &GpuDesc) -> (u64, u64, u64, u64, u64, u64, u64) {
        let mut sums = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);

        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    let words = tcx.read_buffer::<[u32; 7 * REGION_COUNT]>(
                        self.cache_stats_buffer_id,
                        ..,
                    );

                    for (index, word) in words.iter().enumerate() {
                        match index / REGION_COUNT {
                            0 => sums.0 += *word as u64,
                            1 => sums.1 += *word as u64,
                            2 => sums.2 += *word as u64,
                            3 => sums.3 += *word as u64,
                            4 => sums.4 += *word as u64,
                            5 => sums.5 += *word as u64,
                            _ => sums.6 += *word as u64,
                        }
                    }

                    tcx.write_buffer::<[u32; 7 * REGION_COUNT]>(self.cache_stats_buffer_id, ..)
                        .fill(0);

                    Ok(())
                },
                [
                    (self.cache_stats_buffer_id, HostAccessType::Read),

                    (self.cache_stats_buffer_id, HostAccessType::Write),
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

        sums
    }

    pub(crate) fn print_table_stats(&self) {
        println!(
            "cache table: {CACHE_TABLE_ENTRIES} entries — keys {} MiB, accumulation {} MiB, \
             resolved {} MiB",
            CACHE_KEY_BYTES >> 20,
            CACHE_RECORD_BYTES >> 20,
            CACHE_RECORD_BYTES >> 20,
        );
    }
}

fn resolve_blas_storage(
    gpu: &GpuDesc,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
    alloc: &BlasAllocation,
) -> (Arc<AccelerationStructure>, u64) {
    match &alloc.as_storage {
        Some((blas, storage_size)) => (blas.clone(), *storage_size),
        None => accel::create_blas_aabbs_storage(
            aabb_buffer,
            aabb_count,
            gpu.memory_allocator.clone(),
            gpu.device.clone(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_blas_build(
    gpu: &GpuDesc,
    region_index: IVec3,
    aabb_buffer_id: Id<Buffer>,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
    blas: Arc<AccelerationStructure>,
    blas_storage_size: u64,
    fresh: bool,
) -> BlasBuild {
    let sizes = accel::blas_build_sizes(gpu, aabb_buffer, aabb_count);

    debug_assert!(
        blas_storage_size >= sizes.acceleration_structure_size,
        "BLAS build for {aabb_count} AABBs exceeds its {blas_storage_size}-byte storage"
    );

    BlasBuild {
        region_index,
        aabb_buffer_id,
        aabb_count,
        blas,
        scratch: accel::allocate_scratch(gpu, sizes.build_scratch_size),
        fresh,
    }
}

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

fn static_instances() -> Vec<AccelerationStructureInstance> {
    let mut out = vec![AccelerationStructureInstance::default(); REGION_COUNT];

    for x in -8..8 {
        for y in -8..8 {
            for z in -8..8 {
                let index = IVec3::new(x, y, z);
                let id = region_id(index) as usize;
                let origin = (index * REGION_LENGTH).as_vec3().to_array();
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

fn create_dummy_blas(gpu: &GpuDesc) -> Arc<AccelerationStructure> {
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

    accel::build_blas_aabbs_fresh(
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
            let origin = (index * REGION_LENGTH).as_vec3().to_array();
            assert_eq!(instance.transform[0], [1.0, 0.0, 0.0, origin[0]]);
            assert_eq!(instance.transform[1], [0.0, 1.0, 0.0, origin[1]]);
            assert_eq!(instance.transform[2], [0.0, 0.0, 1.0, origin[2]]);
            assert_eq!(instance.instance_custom_index_and_mask.low_24(), id as u32);
            assert_eq!(instance.instance_custom_index_and_mask.high_8(), 0xFF);
            assert_eq!(instance.acceleration_structure_reference, 0);
        }
    }

    #[test]
    fn packed_prefix_rewrites_resident_instances() {
        let instances = static_instances();
        let dummy = AccelerationStructureInstance {
            instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
            ..Default::default()
        };

        let prefix = packed_prefix(&instances, &[2, 5, 9], dummy);
        assert_eq!(prefix.len(), 3);
        assert_eq!(prefix[0].instance_custom_index_and_mask.low_24(), 2);
        assert_eq!(prefix[1].instance_custom_index_and_mask.low_24(), 5);
        assert_eq!(prefix[2].instance_custom_index_and_mask.low_24(), 9);

        let prefix = packed_prefix(&instances, &[], dummy);
        assert_eq!(prefix, vec![dummy]);
    }
}
