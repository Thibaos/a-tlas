use std::sync::Arc;

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

use crate::app::App;

const UPDATES_PER_FRAME: u64 = 2u64.pow(10);

pub struct UpdateAccelerationStructureTask {
    blas_reference: u64,
    pub instance_buffer_id: Id<Buffer>,
    staging_tlas: [Arc<AccelerationStructure>; 2],
    scratch_buffer_id: Id<Buffer>,
    max_instance_count: u64,
}

impl UpdateAccelerationStructureTask {
    pub fn new(
        app: &App,
        instance_buffer_id: Id<Buffer>,
        scratch_buffer_id: Id<Buffer>,
        staging_tlas: [Arc<AccelerationStructure>; 2],
        blas_reference: u64,
    ) -> Self {
        Self {
            blas_reference,
            instance_buffer_id,
            staging_tlas,
            scratch_buffer_id,
            max_instance_count: app.max_instance_count,
        }
    }
}

pub struct AsyncRenderContext {
    pub tlas: Arc<AccelerationStructure>,
}

impl Task for UpdateAccelerationStructureTask {
    type World = AsyncRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        const AS_SIZE: DeviceSize = size_of::<AccelerationStructureInstance>() as DeviceSize;

        let start_instance = rand::random_range(0..(self.max_instance_count - UPDATES_PER_FRAME));
        let start_slice = start_instance * AS_SIZE;

        let write_instance_buffer = tcx.write_buffer::<[AccelerationStructureInstance]>(
            self.instance_buffer_id,
            start_slice..(start_slice + UPDATES_PER_FRAME * AS_SIZE),
        )?;

        for instance in write_instance_buffer.iter_mut() {
            const RANGE: i32 = 256;
            let x = rand::random_range(-RANGE..=RANGE) as f32;
            let y = rand::random_range(-RANGE..=RANGE) as f32;
            let z = rand::random_range(-RANGE..=RANGE) as f32;

            *instance = AccelerationStructureInstance {
                acceleration_structure_reference: self.blas_reference,
                instance_custom_index_and_mask: Packed24_8::new(rand::random::<u8>() as u32, 0xFF),
                transform: [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z]],
                ..Default::default()
            };
        }

        let instance_buffer = Subbuffer::new(
            tcx.buffer(self.instance_buffer_id)
                .expect("Instance buffer not found")
                .buffer()
                .clone(),
        )
        .cast_aligned::<AccelerationStructureInstance>();

        let geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer.clone())),
        );

        let geometries = AccelerationStructureGeometries::Instances(geometry_instances_data);

        let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
            mode: BuildAccelerationStructureMode::Update(rcx.tlas.clone()),
            flags: BuildAccelerationStructureFlags::PREFER_FAST_BUILD
                | BuildAccelerationStructureFlags::ALLOW_UPDATE,
            ..AccelerationStructureBuildGeometryInfo::new(geometries)
        };

        let scratch_buffer = Subbuffer::new(
            tcx.buffer(self.scratch_buffer_id)
                .expect("Scratch buffer not found")
                .buffer()
                .clone(),
        );

        build_geometry_info.dst_acceleration_structure = Some(rcx.tlas.clone());
        build_geometry_info.scratch_data = Some(scratch_buffer);

        unsafe {
            cbf.as_raw().build_acceleration_structure(
                &build_geometry_info,
                &[AccelerationStructureBuildRangeInfo {
                    primitive_count: self.max_instance_count as u32,
                    ..Default::default()
                }],
            )
        }?;

        Ok(())
    }
}
