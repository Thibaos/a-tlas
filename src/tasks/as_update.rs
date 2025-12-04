use std::sync::Arc;

use crate::RenderContext;
use vulkano::{
    DeviceSize, Packed24_8,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureInstance, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode,
    },
    buffer::{Buffer, Subbuffer},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
};

const UPDATES_PER_FRAME: u64 = 2u64.pow(8);

pub struct UpdateAccelerationStructureTask {
    pub blas_reference: u64,
    pub instance_buffer_id: Id<Buffer>,
    pub scratch_buffer_id: Id<Buffer>,
    pub tlas: Arc<AccelerationStructure>,
    pub max_instance_count: u64,
}

impl Task for UpdateAccelerationStructureTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _rcx: &Self::World,
    ) -> TaskResult {
        const AS_SIZE: DeviceSize = size_of::<AccelerationStructureInstance>() as DeviceSize;

        let start_slice =
            rand::random_range(0..(self.max_instance_count - UPDATES_PER_FRAME)) * AS_SIZE;

        let write_instance_buffer = tcx
            .write_buffer::<[AccelerationStructureInstance]>(
                self.instance_buffer_id,
                start_slice..(start_slice + UPDATES_PER_FRAME * AS_SIZE),
            )
            .unwrap();

        for o in write_instance_buffer.iter_mut() {
            *o = AccelerationStructureInstance {
                acceleration_structure_reference: self.blas_reference,
                instance_custom_index_and_mask: Packed24_8::new(0, 0x00),
                transform: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                ..Default::default()
            };
        }

        let instance_buffer = Subbuffer::new(
            tcx.buffer(self.instance_buffer_id)
                .expect("Instance buffer not found")
                .buffer()
                .clone(),
        );

        let instance_buffer = Subbuffer::new(
            tcx.buffer(self.instance_buffer_id)
                .unwrap()
                .buffer()
                .clone(),
        )
        .cast_aligned::<AccelerationStructureInstance>();

        let geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer.clone())),
        );

        let geometries = AccelerationStructureGeometries::Instances(geometry_instances_data);
        let primitive_count = self.max_instance_count as u32;

        let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
            mode: BuildAccelerationStructureMode::Update(self.tlas.clone()),
            flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
            ..AccelerationStructureBuildGeometryInfo::new(geometries)
        };

        let scratch_buffer =
            Subbuffer::new(tcx.buffer(self.scratch_buffer_id).unwrap().buffer().clone());

        build_geometry_info.dst_acceleration_structure = Some(self.tlas.clone());
        build_geometry_info.scratch_data = Some(scratch_buffer);

        unsafe {
            cbf.as_raw().build_acceleration_structure(
                &build_geometry_info,
                &[AccelerationStructureBuildRangeInfo {
                    primitive_count,
                    ..Default::default()
                }],
            )
        }?;

        Ok(())
    }
}
