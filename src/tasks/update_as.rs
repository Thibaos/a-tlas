use std::{collections::HashSet, sync::Arc};

use vulkano::{
    DeviceSize, Packed24_8,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryInstancesDataType, AccelerationStructureInstance,
        BuildAccelerationStructureFlags, BuildAccelerationStructureMode, GeometryInstanceFlags,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
};

use crate::app::App;

const UPDATES_PER_FRAME: u64 = 1000;
// const UPDATES_PER_FRAME: u64 = 2u64.pow(10);

pub struct UpdateAccelerationStructureTask {
    blas_reference: u64,
    pub instance_buffer_id: Id<Buffer>,
    scratch_buffer_id: Id<Buffer>,
}

impl UpdateAccelerationStructureTask {
    pub fn new(app: &App, instance_buffer_id: Id<Buffer>, blas_reference: u64) -> Self {
        let geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(None),
        );

        let geometries = AccelerationStructureGeometries::Instances(geometry_instances_data);

        let build_info = AccelerationStructureBuildGeometryInfo::new(geometries);

        let build_sizes_info = app
            .device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &build_info,
                &[UPDATES_PER_FRAME as u32],
            )
            .unwrap();

        let update_scratch_buffer = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
                DeviceLayout::new_unsized::<[u8]>(build_sizes_info.build_scratch_size).unwrap(),
            )
            .unwrap();

        Self {
            blas_reference,
            instance_buffer_id,
            scratch_buffer_id: update_scratch_buffer,
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

        let write_instance_buffer = tcx.write_buffer::<[AccelerationStructureInstance]>(
            self.instance_buffer_id,
            0..(UPDATES_PER_FRAME * AS_SIZE),
        )?;

        for instance in write_instance_buffer.iter_mut() {
            const RANGE: i32 = 32;
            let x = rand::random_range(-RANGE..=RANGE);
            let y = rand::random_range(-RANGE..=RANGE);
            let z = rand::random_range(-RANGE..=RANGE);

            *instance = AccelerationStructureInstance {
                acceleration_structure_reference: self.blas_reference,
                instance_custom_index_and_mask: Packed24_8::new(rand::random::<u8>() as u32, 0xFF),
                transform: [
                    [1.0, 0.0, 0.0, x as f32],
                    [0.0, 1.0, 0.0, y as f32],
                    [0.0, 0.0, 1.0, z as f32],
                ],
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

        let mut build_geometry_info = AccelerationStructureBuildGeometryInfo::new(geometries);

        let scratch_buffer = Subbuffer::new(
            tcx.buffer(self.scratch_buffer_id)
                .expect("Scratch buffer not found")
                .buffer()
                .clone(),
        );

        build_geometry_info.mode = BuildAccelerationStructureMode::Update(rcx.tlas.clone());
        build_geometry_info.flags = BuildAccelerationStructureFlags::ALLOW_UPDATE;
        build_geometry_info.dst_acceleration_structure = Some(rcx.tlas.clone());
        build_geometry_info.scratch_data = Some(scratch_buffer);

        unsafe {
            cbf.as_raw().build_acceleration_structure(
                &build_geometry_info,
                &[AccelerationStructureBuildRangeInfo {
                    primitive_count: UPDATES_PER_FRAME as u32,
                    ..Default::default()
                }],
            )
        }?;

        Ok(())
    }
}
