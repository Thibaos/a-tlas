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
            feed::RendererInput,
            pack::{REGION_COUNT, RegionData},
            rebuild::{
                BlasBuild, RebuildGraph, RebuildLogEntry, RebuildPlan, RegionUpload, TlasBuild,
                allocate_scratch, blas_build_sizes, tlas_build_sizes,
            },
            task::{capture_raygen, default_scene},
        },
    },
    world::{
        format::get_palette,
        grid::{REGION_LENGTH, region_id},
        material::get_material_table,
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

pub struct RegionStore {
    pub camera_buffer_id: Id<Buffer>,
    pub scene_buffer_id: Id<Buffer>,
    pub region_table_storage_id: StorageBufferId,
    pub camera_storage_id: StorageBufferId,
    pub scene_storage_id: StorageBufferId,
    pub palette_storage_id: StorageBufferId,
    pub material_table_storage_id: StorageBufferId,
    pub acceleration_structure_id: AccelerationStructureId,
    region_table_buffer_id: Id<Buffer>,
    aabb_table_buffer_id: Id<Buffer>,
    pub aabb_table_storage_id: StorageBufferId,
    instances: Vec<AccelerationStructureInstance>,
    resident_ids: Vec<u32>,
    pub instance_buffer_id: Id<Buffer>,
    tlas: Arc<AccelerationStructure>,
    tlas_storage_size: u64,
    tlas_initialized: bool,
    regions: Vec<Option<ResidentRegion>>,
    table_addresses: Vec<u64>,
    free: FreeLists,
    pending_free: PendingFrees,
    dummy_blas: Arc<AccelerationStructure>,
    pub alloc_stats: AllocStats,
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
                DeviceLayout::new_sized::<capture_raygen::Camera>(),
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

        unsafe {
            vulkano_taskgraph::execute(
                &gpu.transfer_queue,
                &gpu.resources,
                gpu.graphics_flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer::<capture_raygen::Palette>(palette_buffer_id, ..) =
                        capture_raygen::Palette { colors: palette };
                    *tcx.write_buffer::<capture_raygen::MaterialTable>(
                        material_table_buffer_id,
                        ..,
                    ) = capture_raygen::MaterialTable {
                        albedo_metallic,
                        rough_emit,
                    };
                    *tcx.write_buffer::<capture_raygen::Scene>(scene_buffer_id, ..) =
                        default_scene();
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
            aabb_table_buffer_id,
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

        store.write_aabb_table(gpu, aabb_table_buffer_id);

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

    #[allow(dead_code)]
    pub fn resident_count(&self) -> usize {
        self.resident_ids.len()
    }

    #[allow(dead_code)]
    pub fn resident_ids(&self) -> &[u32] {
        &self.resident_ids
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
                (false, None) => {}

                (false, Some(pack)) => {
                    let pool = allocate_pool(
                        gpu,
                        &mut self.free,
                        &mut self.alloc_stats,
                        pack.blocks.len() as u64,
                    );
                    let blas_alloc = allocate_blas(
                        gpu,
                        &mut self.free,
                        &mut self.alloc_stats,
                        pack.aabbs.len() as u32,
                    );

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

                (true, Some(pack)) => {
                    let pool_grows =
                        self.regions[id].as_ref().unwrap().pool_capacity < pack.blocks.len() as u64;
                    let blas_grows =
                        self.regions[id].as_ref().unwrap().aabb_capacity < pack.aabbs.len() as u32;

                    let new_pool = pool_grows.then(|| {
                        allocate_pool(
                            gpu,
                            &mut self.free,
                            &mut self.alloc_stats,
                            pack.blocks.len() as u64,
                        )
                    });

                    let new_blas = blas_grows.then(|| {
                        allocate_blas(
                            gpu,
                            &mut self.free,
                            &mut self.alloc_stats,
                            pack.aabbs.len() as u32,
                        )
                    });

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
            report.instance_count = self.resident_ids.len();
            return report;
        }

        self.rebuild_with_plan(gpu, plan);

        let aabbs_moved = !report.became_resident.is_empty()
            || !report.left_resident.is_empty()
            || !report.blas_replaced.is_empty();
        if aabbs_moved {
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

    fn release_pending_frees(&mut self) {
        self.free.pools.append(&mut self.pending_free.pools);
        self.free.blas.append(&mut self.pending_free.blas);
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
