use std::sync::Arc;

use glam::IVec3;
use vulkano::{
    DeviceSize,
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, Subbuffer},
};
use vulkano_taskgraph::{
    Id, QueueFamilyType, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{AccessTypes, HostAccessType},
    resource_map,
};

use crate::core::render::{
    accel,
    gpu::GpuDesc,
    region::{pack::REGION_COUNT, residency::RegionStore, task::production_raygen},
};

pub struct RegionUpload {
    pub region_index: IVec3,
    pub pool_buffer_id: Id<Buffer>,
    pub pool_bytes: Vec<u8>,
    pub aabb_buffer_id: Id<Buffer>,
    pub aabbs: Vec<AabbPositions>,
}

pub struct BlasBuild {
    pub region_index: IVec3,
    pub aabb_buffer_id: Id<Buffer>,
    pub aabb_count: u32,
    pub blas: Arc<AccelerationStructure>,
    pub scratch: Arc<Buffer>,
    pub fresh: bool,
}

pub struct TlasBuild {
    pub instance_count: u32,
    pub scratch: Arc<Buffer>,
}

#[derive(Default)]
pub struct RebuildPlan {
    pub uploads: Vec<RegionUpload>,
    pub blas_builds: Vec<BlasBuild>,
    pub table: Option<[u64; REGION_COUNT]>,
    pub instances: Option<Vec<AccelerationStructureInstance>>,
    pub tlas: Option<TlasBuild>,
}

impl RebuildPlan {
    pub const fn is_empty(&self) -> bool {
        self.uploads.is_empty()
            && self.blas_builds.is_empty()
            && self.table.is_none()
            && self.instances.is_none()
            && self.tlas.is_none()
    }

    pub fn log(&self) -> anyhow::Result<Vec<RebuildLogEntry>> {
        let mut log = Vec::new();

        for upload in &self.uploads {
            log.push(RebuildLogEntry::Upload {
                region_index: upload.region_index,
                pool_bytes: u64::try_from(upload.pool_bytes.len())?,
                aabbs: u32::try_from(upload.aabbs.len())?,
            });
        }

        if self.table.is_some() {
            log.push(RebuildLogEntry::WriteRegionTable);
        }

        if let Some(instances) = &self.instances {
            log.push(RebuildLogEntry::RewriteInstances {
                instance_count: u32::try_from(instances.len())?,
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

        Ok(log)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildLogEntry {
    Upload {
        region_index: IVec3,
        pool_bytes: u64,
        aabbs: u32,
    },
    BuildBlas {
        region_index: IVec3,
        aabb_count: u32,
        fresh: bool,
    },
    WriteRegionTable,
    RewriteInstances {
        instance_count: u32,
    },
    BuildTlas {
        instance_count: u32,
    },
}

struct UploadRegionsTask {
    uploads: Vec<RegionUpload>,
    table: Option<[u64; REGION_COUNT]>,
    instances: Option<Vec<AccelerationStructureInstance>>,
    instance_buffer_id: Id<Buffer>,
    region_table_buffer_id: Id<Buffer>,
}
impl Task for UploadRegionsTask {
    type World = ();

    unsafe fn execute(
        &self,
        _cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        for upload in &self.uploads {
            let Ok(pool_byte_size) = DeviceSize::try_from(upload.pool_bytes.len()) else {
                eprintln!("upload pool bytes len could not be cast to device size");
                return Ok(());
            };
            let Ok(aabbs_size) = DeviceSize::try_from(upload.aabbs.len()) else {
                eprintln!("upload aabbs len could not be cast to device size");
                return Ok(());
            };
            let Ok(aabb_position_size) = DeviceSize::try_from(size_of::<AabbPositions>()) else {
                eprintln!("AabbPositions size could not be cast to device size");
                return Ok(());
            };

            tcx.write_buffer::<[u8]>(upload.pool_buffer_id, 0..pool_byte_size)
                .copy_from_slice(&upload.pool_bytes);

            let dst = tcx.write_buffer::<[AabbPositions]>(
                upload.aabb_buffer_id,
                0..(aabbs_size.strict_mul(aabb_position_size)),
            );
            for (slot, aabb) in dst.iter_mut().zip(upload.aabbs.iter().copied()) {
                *slot = aabb;
            }
        }

        if let Some(table) = &self.table {
            tcx.write_buffer::<production_raygen::RegionTable>(self.region_table_buffer_id, ..)
                .bdas
                .copy_from_slice(table);
        }

        if let Some(instances) = &self.instances {
            let Ok(instances_size) = DeviceSize::try_from(instances.len()) else {
                eprintln!("instances could not be cast to device size");
                return Ok(());
            };
            let Ok(as_size) = DeviceSize::try_from(size_of::<AccelerationStructureInstance>())
            else {
                eprintln!("AccelerationStructureInstance size could not be cast to device size");
                return Ok(());
            };

            let dst = tcx.write_buffer::<[AccelerationStructureInstance]>(
                self.instance_buffer_id,
                0..(instances_size.strict_mul(as_size)),
            );
            for (slot, instance) in dst.iter_mut().zip(instances) {
                *slot = *instance;
            }
        }

        Ok(())
    }
}

struct BuildBlasTask {
    builds: Vec<BlasBuild>,
}

impl Task for BuildBlasTask {
    type World = ();

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        for build in &self.builds {
            let aabb_buffer = Subbuffer::new(tcx.buffer(build.aabb_buffer_id).buffer().clone())
                .cast_aligned::<AabbPositions>();
            let geometries = accel::aabb_geometries(&aabb_buffer);

            if let Ok(geometries) = geometries {
                let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
                    ty: AccelerationStructureType::BottomLevel,
                    mode: BuildAccelerationStructureMode::Build,
                    flags: accel::build_flags(AccelerationStructureType::BottomLevel),
                    geometries: &geometries,
                    ..AccelerationStructureBuildGeometryInfo::new()
                };
                build_geometry_info.dst_acceleration_structure = Some(&build.blas);
                build_geometry_info.scratch_data = build.scratch.device_address().get();

                accel::as_build_pre_barrier(cbf);

                unsafe {
                    cbf.as_raw().build_acceleration_structure(
                        &build_geometry_info,
                        &[AccelerationStructureBuildRangeInfo {
                            primitive_count: build.aabb_count,
                            ..Default::default()
                        }],
                    )
                };

                accel::as_build_post_barrier(cbf);
            }
        }

        Ok(())
    }
}

struct BuildTlasTask {
    instance_count: u32,
    instance_buffer_id: Id<Buffer>,
    tlas: Arc<AccelerationStructure>,
    scratch: Arc<Buffer>,
}

impl Task for BuildTlasTask {
    type World = ();

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        let instance_buffer = Subbuffer::new(tcx.buffer(self.instance_buffer_id).buffer().clone())
            .cast_aligned::<AccelerationStructureInstance>();
        let geometries = accel::instance_geometries(&instance_buffer);

        if let Ok(geometries) = geometries {
            let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
                ty: AccelerationStructureType::TopLevel,
                mode: BuildAccelerationStructureMode::Build,
                flags: accel::build_flags(AccelerationStructureType::TopLevel),
                geometries: &geometries,
                ..AccelerationStructureBuildGeometryInfo::new()
            };

            build_geometry_info.dst_acceleration_structure = Some(&self.tlas);
            build_geometry_info.scratch_data = self.scratch.device_address().get();

            accel::as_build_pre_barrier(cbf);

            unsafe {
                cbf.as_raw().build_acceleration_structure(
                    &build_geometry_info,
                    &[AccelerationStructureBuildRangeInfo {
                        primitive_count: self.instance_count,
                        ..Default::default()
                    }],
                )
            };

            accel::as_build_post_barrier(cbf);
        }

        Ok(())
    }
}

pub struct RebuildGraph {
    executable: ExecutableTaskGraph<()>,
}

impl RebuildGraph {
    pub fn new(gpu: &GpuDesc, store: &RegionStore, plan: RebuildPlan) -> anyhow::Result<Self> {
        let blas_buffers = blas_buffer_ids(&plan);

        let mut task_graph = TaskGraph::new(&gpu.resources);

        for upload in &plan.uploads {
            task_graph.add_host_buffer_access(upload.pool_buffer_id, HostAccessType::Write);
            task_graph.add_host_buffer_access(upload.aabb_buffer_id, HostAccessType::Write);
        }

        if plan.table.is_some() {
            task_graph
                .add_host_buffer_access(store.region_table_buffer_id(), HostAccessType::Write);
        }

        if plan.instances.is_some() {
            task_graph
                .add_host_buffer_access(store.bindings.instance_buffer, HostAccessType::Write);
        }

        let upload_node = task_graph
            .create_task_node(
                "Rebuild upload",
                QueueFamilyType::Compute,
                UploadRegionsTask {
                    uploads: plan.uploads,
                    table: plan.table,
                    instances: plan.instances,
                    instance_buffer_id: store.bindings.instance_buffer,
                    region_table_buffer_id: store.region_table_buffer_id(),
                },
            )
            .build();

        let blas_node = if plan.blas_builds.is_empty() {
            None
        } else {
            let mut builder = task_graph.create_task_node(
                "Rebuild BLAS",
                QueueFamilyType::Compute,
                BuildBlasTask {
                    builds: plan.blas_builds,
                },
            );

            for build in blas_buffers {
                builder.buffer_access(
                    build,
                    AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_READ,
                );
            }

            Some(builder.build())
        };

        let tlas_node = if let Some(tlas) = plan.tlas {
            Some(
                task_graph
                    .create_task_node(
                        "Rebuild TLAS",
                        QueueFamilyType::Compute,
                        BuildTlasTask {
                            instance_count: tlas.instance_count,
                            instance_buffer_id: store.bindings.instance_buffer,
                            tlas: store.tlas(),
                            scratch: tlas.scratch,
                        },
                    )
                    .buffer_access(
                        store.bindings.instance_buffer,
                        AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                    )
                    .build(),
            )
        } else {
            None
        };

        let mut previous = upload_node;
        if let Some(blas_node) = blas_node {
            task_graph.add_edge(previous, blas_node)?;
            previous = blas_node;
        }

        if let Some(tlas_node) = tlas_node {
            task_graph.add_edge(previous, tlas_node)?;
        }

        let executable = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&gpu.compute_queue],
                present_queue: None,
                flight_id: gpu.compute_flight_id,
                ..CompileInfo::default()
            })
        }?;

        Ok(Self { executable })
    }

    pub fn execute(self, gpu: &GpuDesc) -> anyhow::Result<()> {
        let resource_map = resource_map!(&self.executable)?;

        unsafe { self.executable.execute(resource_map, &(), || {}) }?;

        gpu.resources.flight(gpu.compute_flight_id).wait_idle()?;

        Ok(())
    }
}

fn blas_buffer_ids(plan: &RebuildPlan) -> Vec<Id<Buffer>> {
    plan.blas_builds
        .iter()
        .map(|build| build.aabb_buffer_id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_plan_means_no_rebuild_work() {
        let plan = RebuildPlan::default();
        assert!(plan.is_empty());
        assert!(plan.log().unwrap().is_empty());
    }

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
            plan.log().unwrap(),
            vec![RebuildLogEntry::Upload {
                region_index: IVec3::new(0, 0, 0),
                pool_bytes: 64,
                aabbs: 1,
            }]
        );

        plan.instances = Some(vec![AccelerationStructureInstance::default()]);

        plan.tlas = None;

        let log = plan.log().unwrap();

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
