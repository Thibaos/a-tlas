use std::sync::Arc;

use anyhow::Context;
use dot_vox::DotVoxData;
use glam::IVec3;
use vulkano::{
    DeviceSize, Packed24_8,
    acceleration_structure::{AabbPositions, AccelerationStructure, AccelerationStructureInstance},
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
};
use vulkano_taskgraph::{
    Id,
    descriptor_set::{AccelerationStructureId, BindlessContext, StorageBufferId},
    resource::HostAccessType,
};

use crate::{
    render::{
        accel,
        context::RenderContext,
        region::{
            alloc::{
                AllocStats, BlasAllocation, FreeLists, FreedBlas, FreedPool, PendingFrees,
                PoolAllocation, allocate_blas, allocate_pool,
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

struct SceneBuffers {
    camera: Id<Buffer>,
    scene: Id<Buffer>,
    palette: Id<Buffer>,
    region_table: Id<Buffer>,
    aabb_table: Id<Buffer>,
    instance: Id<Buffer>,
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
pub struct RegionBindingsIds {
    pub camera_buffer: Id<Buffer>,
    pub scene_buffer: Id<Buffer>,
    pub region_table_storage: StorageBufferId,
    pub camera_storage: StorageBufferId,
    pub scene_storage: StorageBufferId,
    pub palette_storage: StorageBufferId,
    pub acceleration_structure: AccelerationStructureId,
    pub aabb_table_storage: StorageBufferId,
    pub instance_buffer: Id<Buffer>,
}

pub struct RegionStore {
    pub bindings: RegionBindingsIds,
    region_table_buffer_id: Id<Buffer>,
    aabb_table_buffer_id: Id<Buffer>,
    instances: Vec<AccelerationStructureInstance>,
    resident_ids: Vec<u32>,
    tlas: Arc<AccelerationStructure>,
    tlas_storage_size: u64,
    tlas_initialized: bool,
    regions: Vec<Option<ResidentRegion>>,
    table_addresses: Vec<u64>,
    free: FreeLists,
    pending_free: PendingFrees,
    dummy_blas: Arc<AccelerationStructure>,
    alloc_stats: AllocStats,
}

impl RegionStore {
    pub fn new(
        gpu: &RenderContext,
        voxel_data: &DotVoxData,
        input: &RendererInput,
    ) -> anyhow::Result<Self> {
        input.wait_until_idle()?;
        let initial = input.packed_regions()?;

        let buffers = create_scene_buffers(gpu)?;
        let (tlas, tlas_storage_size) = create_tlas(gpu, buffers.instance)?;
        let dummy_blas = create_dummy_blas(gpu)?;

        upload_initial_globals(gpu, &buffers, voxel_data)?;

        let bindings = create_bindings(gpu, &buffers, &tlas)?;

        let mut store = Self {
            bindings,
            region_table_buffer_id: buffers.region_table,
            aabb_table_buffer_id: buffers.aabb_table,
            instances: static_instances()?,
            resident_ids: Vec::new(),
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

        let report = store.rebuild(gpu, packs)?;

        debug_assert!(
            report.left_resident.is_empty(),
            "the initial batch only creates residency"
        );

        store.ensure_tlas_initialized(gpu)?;
        store.write_aabb_table(gpu, buffers.aabb_table)?;

        Ok(store)
    }

    fn write_aabb_table(
        &self,
        gpu: &RenderContext,
        aabb_table_buffer_id: Id<Buffer>,
    ) -> anyhow::Result<()> {
        let mut bdas = vec![0u64; REGION_COUNT];

        for (id, region) in self.regions.iter().enumerate() {
            if let Some(region) = region {
                *bdas
                    .get_mut(id)
                    .context(format!("bda slot {id} out of range"))? = gpu
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
                    tcx.write_buffer::<production_raygen::AabbTable>(aabb_table_buffer_id, ..)
                        .bdas
                        .copy_from_slice(&bdas);
                    Ok(())
                },
                [(aabb_table_buffer_id, HostAccessType::Write)],
                [],
                [],
            )?;
        }

        gpu.resources.flight(gpu.graphics_flight_id).wait_idle()?;

        Ok(())
    }

    fn ensure_tlas_initialized(&mut self, gpu: &RenderContext) -> anyhow::Result<()> {
        if self.tlas_initialized {
            return Ok(());
        }

        let mut plan = RebuildPlan::default();
        self.plan_tlas_build(gpu, &mut plan, 1)?;

        self.rebuild_with_plan(gpu, plan)
    }

    fn plan_tlas_build(
        &self,
        gpu: &RenderContext,
        plan: &mut RebuildPlan,
        instance_count: u32,
    ) -> anyhow::Result<()> {
        let instance_buffer = Subbuffer::new(
            gpu.resources
                .buffer(self.bindings.instance_buffer)
                .buffer()
                .clone(),
        )
        .cast_aligned::<AccelerationStructureInstance>();

        let sizes = accel::tlas_build_sizes(gpu, &instance_buffer, instance_count)?;

        debug_assert!(
            self.tlas_storage_size >= sizes.acceleration_structure_size,
            "in-place TLAS build for {instance_count} instances exceeds the stable storage"
        );

        plan.instances = Some(self.packed_instance_prefix()?);

        plan.tlas = Some(TlasBuild {
            instance_count,
            scratch: accel::allocate_scratch(gpu, sizes.build_scratch_size)?,
        });

        Ok(())
    }

    pub fn apply(
        &mut self,
        gpu: &RenderContext,
        input: &RendererInput,
    ) -> anyhow::Result<ApplyReport> {
        let dirty = input.take_dirty_regions();

        if dirty.is_empty() {
            return Ok(ApplyReport::default());
        }

        let packs: Vec<(IVec3, Option<RegionData>)> = dirty
            .iter()
            .map(|&region| Ok((region, input.packed_region(region)?)))
            .collect::<anyhow::Result<_>>()?;

        self.rebuild(gpu, packs)
    }

    pub fn blases(&self) -> Vec<Arc<AccelerationStructure>> {
        self.regions
            .iter()
            .filter_map(|region| region.as_ref().map(|region| region.blas.clone()))
            .collect()
    }

    pub(crate) const fn region_table_buffer_id(&self) -> Id<Buffer> {
        self.region_table_buffer_id
    }

    pub(crate) fn tlas(&self) -> Arc<AccelerationStructure> {
        self.tlas.clone()
    }

    fn rebuild(
        &mut self,
        gpu: &RenderContext,
        packs: Vec<(IVec3, Option<RegionData>)>,
    ) -> anyhow::Result<ApplyReport> {
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
        let decision = decide(&slots, &self.resident_ids, packs)?;
        self.resident_ids = decision.resident_ids;

        let mut report = ApplyReport {
            instance_count_before,
            became_resident: decision.became_resident,
            left_resident: decision.left_resident,
            dirty: decision.dirty,
            blas_replaced: decision.blas_replaced,
            tlas_rebuilt: decision.tlas_dirty,
            ..ApplyReport::default()
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
                    self.enter_region(gpu, &mut plan, id, pool_bytes, aabbs, pack)?;
                }

                RegionEffect::Exit {
                    retire_pool,
                    retire_blas,
                } => {
                    self.exit_region(id, retire_pool, retire_blas)?;
                }

                RegionEffect::Update {
                    pool_bytes,
                    aabbs,
                    retire_pool,
                    retire_blas,
                    pack,
                } => {
                    self.update_region(
                        gpu,
                        &mut plan,
                        id,
                        pool_bytes,
                        aabbs,
                        retire_pool,
                        retire_blas,
                        pack,
                    )?;
                }
            }
        }

        if decision.table_changed {
            let addresses: [u64; REGION_COUNT] = self
                .table_addresses
                .clone()
                .try_into()
                .ok()
                .context("region address table length differs from REGION_COUNT")?;
            plan.table = Some(addresses);
        }

        if decision.tlas_dirty {
            self.plan_tlas_build(
                gpu,
                &mut plan,
                u32::try_from(self.resident_ids.len().max(1))?,
            )?;
        }

        report.rebuild_log = plan.log()?;

        if plan.is_empty() {
            report.instance_count = self.resident_ids.len();
            return Ok(report);
        }

        self.rebuild_with_plan(gpu, plan)?;

        if report.tlas_rebuilt {
            self.write_aabb_table(gpu, self.aabb_table_buffer_id)?;
        }

        report.instance_count = self.resident_ids.len();

        Ok(report)
    }

    fn enter_region(
        &mut self,
        gpu: &RenderContext,
        plan: &mut RebuildPlan,
        id: u32,
        pool_bytes: u64,
        aabbs: u32,
        pack: RegionData,
    ) -> anyhow::Result<()> {
        let region_index = pack.region_index;

        let pool = allocate_pool(gpu, &mut self.free, &mut self.alloc_stats, pool_bytes)?;
        let blas_alloc = allocate_blas(gpu, &mut self.free, &mut self.alloc_stats, aabbs)?;

        let aabb_buffer = Subbuffer::new(
            gpu.resources
                .buffer(blas_alloc.aabb_buffer_id)
                .buffer()
                .clone(),
        )
        .cast_aligned::<AabbPositions>();

        let (blas, blas_storage_size) =
            resolve_blas_storage(gpu, &aabb_buffer, aabbs, &blas_alloc)?;

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
            aabbs,
            blas.clone(),
            blas_storage_size,
            true,
        )?);

        let address = gpu
            .resources
            .buffer(pool.buffer_id)
            .buffer()
            .device_address()
            .get();

        self.set_instance_reference(id, &blas)?;
        self.set_table_address(id, address)?;

        *self
            .regions
            .get_mut(usize::try_from(id)?)
            .context(format!("region {id} out of range"))? = Some(ResidentRegion {
            pool_buffer_id: pool.buffer_id,
            pool_capacity: pool.capacity,
            aabb_buffer_id: blas_alloc.aabb_buffer_id,
            aabb_capacity: blas_alloc.aabb_capacity,
            blas,
            blas_storage_size,
        });

        Ok(())
    }

    fn exit_region(&mut self, id: u32, retire_pool: u64, retire_blas: u32) -> anyhow::Result<()> {
        let region = self
            .regions
            .get_mut(usize::try_from(id)?)
            .context(format!("region {id} out of range"))?
            .take()
            .context(format!("region {id} is not resident"))?;

        self.set_table_address(id, 0)?;

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

        Ok(())
    }

    fn update_region(
        &mut self,
        gpu: &RenderContext,
        plan: &mut RebuildPlan,
        id: u32,
        pool_bytes: u64,
        aabbs: u32,
        retire_pool: Option<u64>,
        retire_blas: Option<u32>,
        pack: RegionData,
    ) -> anyhow::Result<()> {
        let region_index = pack.region_index;

        let blas_replacement = if retire_blas.is_some() {
            let alloc = allocate_blas(gpu, &mut self.free, &mut self.alloc_stats, aabbs)?;

            self.replace_blas(id, &alloc)?;

            Some(alloc)
        } else {
            None
        };

        if retire_pool.is_some() {
            let pool = allocate_pool(gpu, &mut self.free, &mut self.alloc_stats, pool_bytes)?;

            self.replace_pool(gpu, id, &pool)?;
        }

        let (pool_id, aabb_id) = self.region_buffer_ids(id)?;

        plan.uploads.push(RegionUpload {
            region_index,
            pool_buffer_id: pool_id,
            pool_bytes: pack.blocks,
            aabb_buffer_id: aabb_id,
            aabbs: pack.aabbs,
        });

        match blas_replacement {
            Some(alloc) => {
                self.plan_replacement_blas_build(gpu, plan, id, region_index, aabbs, &alloc)?;
            }
            None => self.plan_in_place_blas_build(gpu, plan, id, region_index, aabb_id, aabbs)?,
        }

        Ok(())
    }

    fn replace_pool(
        &mut self,
        gpu: &RenderContext,
        id: u32,
        pool: &PoolAllocation,
    ) -> anyhow::Result<()> {
        let region = self
            .regions
            .get_mut(usize::try_from(id)?)
            .context(format!("region {id} out of range"))?
            .as_mut()
            .context(format!("region {id} is not resident"))?;

        self.pending_free.pools.push(FreedPool {
            buffer_id: region.pool_buffer_id,
            capacity: region.pool_capacity,
        });

        region.pool_buffer_id = pool.buffer_id;
        region.pool_capacity = pool.capacity;

        let address = gpu
            .resources
            .buffer(pool.buffer_id)
            .buffer()
            .device_address()
            .get();

        self.set_table_address(id, address)?;

        Ok(())
    }

    fn replace_blas(&mut self, id: u32, alloc: &BlasAllocation) -> anyhow::Result<()> {
        let region = self
            .regions
            .get_mut(usize::try_from(id)?)
            .context(format!("region {id} out of range"))?
            .as_mut()
            .context(format!("region {id} is not resident"))?;

        self.pending_free.blas.push(FreedBlas {
            aabb_buffer_id: region.aabb_buffer_id,
            aabb_capacity: region.aabb_capacity,
            blas: region.blas.clone(),
            blas_storage_size: region.blas_storage_size,
        });

        region.aabb_buffer_id = alloc.aabb_buffer_id;
        region.aabb_capacity = alloc.aabb_capacity;

        Ok(())
    }

    fn region_buffer_ids(&self, id: u32) -> anyhow::Result<(Id<Buffer>, Id<Buffer>)> {
        let region = self
            .regions
            .get(usize::try_from(id)?)
            .context(format!("region {id} out of range"))?
            .as_ref()
            .context(format!("region {id} is not resident"))?;

        Ok((region.pool_buffer_id, region.aabb_buffer_id))
    }

    fn plan_replacement_blas_build(
        &mut self,
        gpu: &RenderContext,
        plan: &mut RebuildPlan,
        id: u32,
        region_index: IVec3,
        aabb_count: u32,
        alloc: &BlasAllocation,
    ) -> anyhow::Result<()> {
        let aabb_buffer =
            Subbuffer::new(gpu.resources.buffer(alloc.aabb_buffer_id).buffer().clone())
                .cast_aligned::<AabbPositions>();

        let (blas, blas_storage_size) = resolve_blas_storage(gpu, &aabb_buffer, aabb_count, alloc)?;

        plan.blas_builds.push(plan_blas_build(
            gpu,
            region_index,
            alloc.aabb_buffer_id,
            &aabb_buffer,
            aabb_count,
            blas.clone(),
            blas_storage_size,
            true,
        )?);

        let region = self
            .regions
            .get_mut(usize::try_from(id)?)
            .context(format!("region {id} out of range"))?
            .as_mut()
            .context(format!("region {id} is not resident"))?;

        region.blas = blas.clone();
        region.blas_storage_size = blas_storage_size;

        self.set_instance_reference(id, &blas)?;

        Ok(())
    }

    fn plan_in_place_blas_build(
        &self,
        gpu: &RenderContext,
        plan: &mut RebuildPlan,
        id: u32,
        region_index: IVec3,
        aabb_id: Id<Buffer>,
        aabb_count: u32,
    ) -> anyhow::Result<()> {
        let (blas, blas_storage_size) = {
            let region = self
                .regions
                .get(usize::try_from(id)?)
                .context(format!("region {id} out of range"))?
                .as_ref()
                .context(format!("region {id} is not resident"))?;

            (region.blas.clone(), region.blas_storage_size)
        };

        let aabb_buffer = Subbuffer::new(gpu.resources.buffer(aabb_id).buffer().clone())
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
        )?);

        Ok(())
    }

    fn set_table_address(&mut self, id: u32, address: u64) -> anyhow::Result<()> {
        *self
            .table_addresses
            .get_mut(usize::try_from(id)?)
            .context(format!("table address slot {id} out of range"))? = address;

        Ok(())
    }

    fn set_instance_reference(
        &mut self,
        id: u32,
        blas: &Arc<AccelerationStructure>,
    ) -> anyhow::Result<()> {
        self.instances
            .get_mut(usize::try_from(id)?)
            .context(format!("instance slot {id} out of range"))?
            .acceleration_structure_reference = blas.device_address().into();

        Ok(())
    }

    fn rebuild_with_plan(&mut self, gpu: &RenderContext, plan: RebuildPlan) -> anyhow::Result<()> {
        let tlas_rebuilds = plan.tlas.is_some();
        let graph = RebuildGraph::new(gpu, self, plan)?;

        graph.execute(gpu)?;

        if tlas_rebuilds {
            self.tlas_initialized = true;
        }

        self.release_pending_frees();

        Ok(())
    }

    fn packed_instance_prefix(&self) -> anyhow::Result<Vec<AccelerationStructureInstance>> {
        packed_prefix(&self.instances, &self.resident_ids, self.dummy_instance())
    }

    fn dummy_instance(&self) -> AccelerationStructureInstance {
        AccelerationStructureInstance {
            instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
            acceleration_structure_reference: self.dummy_blas.device_address().into(),
            ..AccelerationStructureInstance::default()
        }
    }

    fn release_pending_frees(&mut self) {
        self.free.pools.append(&mut self.pending_free.pools);
        self.free.blas.append(&mut self.pending_free.blas);
    }
}

fn resolve_blas_storage(
    gpu: &RenderContext,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
    alloc: &BlasAllocation,
) -> anyhow::Result<(Arc<AccelerationStructure>, u64)> {
    match &alloc.as_storage {
        Some((blas, storage_size)) => Ok((blas.clone(), *storage_size)),
        None => accel::create_blas_aabbs_storage(
            aabb_buffer,
            aabb_count,
            &gpu.memory_allocator,
            &gpu.device,
        ),
    }
}

fn plan_blas_build(
    gpu: &RenderContext,
    region_index: IVec3,
    aabb_buffer_id: Id<Buffer>,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
    blas: Arc<AccelerationStructure>,
    blas_storage_size: u64,
    fresh: bool,
) -> anyhow::Result<BlasBuild> {
    let sizes = accel::blas_build_sizes(gpu, aabb_buffer, aabb_count)?;

    debug_assert!(
        blas_storage_size >= sizes.acceleration_structure_size,
        "BLAS build for {aabb_count} AABBs exceeds its {blas_storage_size}-byte storage"
    );

    Ok(BlasBuild {
        region_index,
        aabb_buffer_id,
        aabb_count,
        blas,
        scratch: accel::allocate_scratch(gpu, sizes.build_scratch_size)?,
        fresh,
    })
}

fn packed_prefix(
    instances: &[AccelerationStructureInstance],
    resident_ids: &[u32],
    empty_dummy: AccelerationStructureInstance,
) -> anyhow::Result<Vec<AccelerationStructureInstance>> {
    if resident_ids.is_empty() {
        return Ok(vec![empty_dummy]);
    }

    resident_ids
        .iter()
        .map(|&id| {
            let index = usize::try_from(id)?;

            instances
                .get(index)
                .copied()
                .context(format!("resident region {id} has no instance slot"))
        })
        .collect()
}

fn static_instances() -> anyhow::Result<Vec<AccelerationStructureInstance>> {
    let mut out = vec![AccelerationStructureInstance::default(); REGION_COUNT];

    for x in -8..8 {
        for y in -8..8 {
            for z in -8..8 {
                let index = IVec3::new(x, y, z);
                let id = usize::try_from(region_id(index))?;
                let region_length = REGION_LENGTH.cast_signed();
                let origin = IVec3::new(
                    x.strict_mul(region_length),
                    y.strict_mul(region_length),
                    z.strict_mul(region_length),
                )
                .as_vec3()
                .to_array();

                *out.get_mut(id)
                    .context(format!("instance slot {id} out of range"))? =
                    AccelerationStructureInstance {
                        transform: [
                            [1.0, 0.0, 0.0, origin[0]],
                            [0.0, 1.0, 0.0, origin[1]],
                            [0.0, 0.0, 1.0, origin[2]],
                        ],
                        instance_custom_index_and_mask: Packed24_8::new(region_id(index), 0xFF),
                        acceleration_structure_reference: 0,
                        ..AccelerationStructureInstance::default()
                    };
            }
        }
    }

    Ok(out)
}

fn create_scene_buffers(gpu: &RenderContext) -> anyhow::Result<SceneBuffers> {
    Ok(SceneBuffers {
        camera: create_storage_buffer::<production_raygen::Camera>(gpu)?,
        scene: create_storage_buffer::<production_raygen::Scene>(gpu)?,
        palette: create_storage_buffer::<production_raygen::Palette>(gpu)?,
        region_table: create_storage_buffer::<production_raygen::RegionTable>(gpu)?,
        aabb_table: create_storage_buffer::<production_raygen::AabbTable>(gpu)?,
        instance: create_instance_buffer(gpu)?,
    })
}

fn create_storage_buffer<T: BufferContents>(gpu: &RenderContext) -> anyhow::Result<Id<Buffer>> {
    Ok(gpu.resources.create_buffer(
        &BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..BufferCreateInfo::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        DeviceLayout::new_sized::<T>(),
    )?)
}

fn create_instance_buffer(gpu: &RenderContext) -> anyhow::Result<Id<Buffer>> {
    let layout =
        DeviceLayout::new_unsized::<[AccelerationStructureInstance]>(u64::try_from(REGION_COUNT)?)
            .context("device layout for the instance buffer is invalid")?;

    Ok(gpu.resources.create_buffer(
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..BufferCreateInfo::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        layout,
    )?)
}

fn create_tlas(
    gpu: &RenderContext,
    instance_buffer_id: Id<Buffer>,
) -> anyhow::Result<(Arc<AccelerationStructure>, u64)> {
    let instance_buffer = Subbuffer::new(gpu.resources.buffer(instance_buffer_id).buffer().clone())
        .cast_aligned::<AccelerationStructureInstance>();

    accel::create_tlas_storage(
        &instance_buffer,
        u32::try_from(REGION_COUNT)?,
        &gpu.memory_allocator,
        &gpu.device,
    )
}

fn upload_initial_globals(
    gpu: &RenderContext,
    buffers: &SceneBuffers,
    voxel_data: &DotVoxData,
) -> anyhow::Result<()> {
    let palette = get_palette(voxel_data).map(|color| [color.x, color.y, color.z, 1.0]);

    unsafe {
        vulkano_taskgraph::execute(
            &gpu.transfer_queue,
            &gpu.resources,
            gpu.graphics_flight_id,
            |_cbf, tcx| {
                *tcx.write_buffer::<production_raygen::Palette>(buffers.palette, ..) =
                    production_raygen::Palette { colors: palette };
                *tcx.write_buffer::<production_raygen::Scene>(buffers.scene, ..) = default_scene();
                Ok(())
            },
            [
                (buffers.palette, HostAccessType::Write),
                (buffers.scene, HostAccessType::Write),
            ],
            [],
            [],
        )?;
    }

    gpu.resources.flight(gpu.graphics_flight_id).wait_idle()?;

    Ok(())
}

fn bindless_storage_buffer<T>(
    bcx: &BindlessContext,
    buffer_id: Id<Buffer>,
) -> anyhow::Result<StorageBufferId> {
    let size = DeviceSize::try_from(size_of::<T>())?;

    Ok(bcx
        .global_set()
        .create_storage_buffer(buffer_id, 0, Some(size))?)
}

fn create_bindings(
    gpu: &RenderContext,
    buffers: &SceneBuffers,
    tlas: &Arc<AccelerationStructure>,
) -> anyhow::Result<RegionBindingsIds> {
    let bcx = gpu
        .resources
        .bindless_context()
        .context("bindless context not found")?;

    let region_table_storage =
        bindless_storage_buffer::<production_raygen::RegionTable>(bcx, buffers.region_table)?;

    let camera_storage = bindless_storage_buffer::<production_raygen::Camera>(bcx, buffers.camera)?;

    let palette_storage =
        bindless_storage_buffer::<production_raygen::Palette>(bcx, buffers.palette)?;

    let scene_storage = bindless_storage_buffer::<production_raygen::Scene>(bcx, buffers.scene)?;

    let acceleration_structure = bcx.global_set().add_acceleration_structure(tlas.clone());

    let aabb_table_storage =
        bindless_storage_buffer::<production_raygen::AabbTable>(bcx, buffers.aabb_table)?;

    Ok(RegionBindingsIds {
        camera_buffer: buffers.camera,
        scene_buffer: buffers.scene,
        region_table_storage,
        camera_storage,
        scene_storage,
        palette_storage,
        acceleration_structure,
        aabb_table_storage,
        instance_buffer: buffers.instance,
    })
}

fn create_dummy_blas(gpu: &RenderContext) -> anyhow::Result<Arc<AccelerationStructure>> {
    let aabb = AabbPositions {
        min: [1.0e9; 3],
        max: [1.0e9 + 1.0; 3],
    };

    let buffer = Buffer::from_iter(
        &gpu.memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..BufferCreateInfo::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        std::iter::once(aabb),
    )?;

    let result = accel::build_blas_aabbs_fresh(
        &buffer,
        1,
        &gpu.memory_allocator,
        &gpu.device,
        &gpu.compute_queue,
        &gpu.resources,
        gpu.compute_flight_id,
    )?;

    Ok(result.0)
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
        let instances = static_instances().unwrap();
        assert_eq!(instances.len(), REGION_COUNT);

        for index in [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 2, 3),
            IVec3::new(7, -8, 0),
        ] {
            let id = region_id(index) as usize;
            let instance = &instances[id];
            let region_length = REGION_LENGTH.cast_signed();
            let origin = IVec3::new(
                index.x.strict_mul(region_length),
                index.y.strict_mul(region_length),
                index.z.strict_mul(region_length),
            )
            .as_vec3()
            .to_array();
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
        let instances = static_instances().unwrap();
        let dummy = AccelerationStructureInstance {
            instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
            ..AccelerationStructureInstance::default()
        };

        let prefix = packed_prefix(&instances, &[2, 5, 9], dummy).unwrap();
        assert_eq!(prefix.len(), 3);
        assert_eq!(prefix[0].instance_custom_index_and_mask.low_24(), 2);
        assert_eq!(prefix[1].instance_custom_index_and_mask.low_24(), 5);
        assert_eq!(prefix[2].instance_custom_index_and_mask.low_24(), 9);

        let prefix = packed_prefix(&instances, &[], dummy).unwrap();
        assert_eq!(prefix, vec![dummy]);
    }
}
