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
    pub fn is_empty(&self) -> bool {
        self.uploads.is_empty()
            && self.blas_builds.is_empty()
            && self.table.is_none()
            && self.instances.is_none()
            && self.tlas.is_none()
    }

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
            *tcx.write_buffer::<production_raygen::RegionTable>(self.region_table_buffer_id, ..) =
                production_raygen::RegionTable { bdas: *table };
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
    pub fn new(gpu: &GpuDesc, store: &RegionStore, plan: RebuildPlan) -> Self {
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
                .add_host_buffer_access(store.bindings.instance_buffer_id, HostAccessType::Write);
        }

        let upload_node = task_graph
            .create_task_node(
                "Rebuild upload",
                QueueFamilyType::Compute,
                UploadRegionsTask {
                    uploads: plan.uploads,
                    table: plan.table,
                    instances: plan.instances,
                    instance_buffer_id: store.bindings.instance_buffer_id,
                    region_table_buffer_id: store.region_table_buffer_id(),
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
                            instance_buffer_id: store.bindings.instance_buffer_id,
                            tlas: store.tlas(),
                            scratch: tlas.scratch,
                        },
                    )
                    .buffer_access(
                        store.bindings.instance_buffer_id,
                        AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                    )
                    .build(),
            );
        }

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

        Self { executable }
    }

    pub fn execute(self, gpu: &GpuDesc) {
        let resource_map = resource_map!(&self.executable).unwrap();

        unsafe { self.executable.execute(resource_map, &(), || {}) }.unwrap();

        gpu.resources
            .flight(gpu.compute_flight_id)
            .wait_idle()
            .unwrap();
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
        assert!(plan.log().is_empty());
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
            plan.log(),
            vec![RebuildLogEntry::Upload {
                region_index: IVec3::new(0, 0, 0),
                pool_bytes: 64,
                aabbs: 1,
            }]
        );

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
