use std::sync::atomic::Ordering;

use vulkano::{
    acceleration_structure::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureBuildRangeInfo,
        AccelerationStructureBuildType, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureInstance, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout},
    sync::{AccessFlags, PipelineStages},
    DeviceSize,
};
use vulkano_taskgraph::{
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    Id, Task, TaskContext, TaskResult,
};

use crate::app::{App, AsyncRenderContext};

const AS_SIZE: DeviceSize = size_of::<AccelerationStructureInstance>() as DeviceSize;

pub struct UpdateAccelerationStructureTask {
    instance_count: u32,
    blas_reference: u64,
    pub instance_buffer_id: Id<Buffer>,
    pub scratch_buffer_id: Id<Buffer>,
    geometries: AccelerationStructureGeometries,
}

impl UpdateAccelerationStructureTask {
    pub fn new(
        app: &App,
        instance_count: u32,
        instance_buffer_id: Id<Buffer>,
        blas_reference: u64,
    ) -> Self {
        let instance_buffer =
            Subbuffer::new(app.resources.buffer(instance_buffer_id).buffer().clone())
                .cast_aligned::<AccelerationStructureInstance>();

        let geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer.clone())),
        );

        let geometries = AccelerationStructureGeometries::Instances(geometry_instances_data);

        let build_geometry_info = AccelerationStructureBuildGeometryInfo::new(geometries.clone());

        let build_sizes_info = app.device.acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &build_geometry_info,
            &[instance_count],
        );

        let scratch_size = build_sizes_info.build_scratch_size;

        let scratch_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
                DeviceLayout::new_unsized::<[u8]>(scratch_size).unwrap(),
            )
            .unwrap();

        Self {
            instance_count,
            blas_reference,
            instance_buffer_id,
            scratch_buffer_id,
            geometries,
        }
    }
}

impl Task for UpdateAccelerationStructureTask {
    type World = AsyncRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let write_instance_buffer = tcx.write_buffer::<[AccelerationStructureInstance]>(
            self.instance_buffer_id,
            0..(self.instance_count as u64 * AS_SIZE),
        );

        let new_instances = rcx.world.to_instances(
            0,
            &rcx.position,
            self.blas_reference,
            self.instance_count as u64,
            rcx.frustum.as_ref(),
        );

        write_instance_buffer
            .iter_mut()
            .zip(new_instances)
            .for_each(|(instance, new_instance)| {
                *instance = new_instance;
            });

        let pre_memory_barrier = MemoryBarrier {
            src_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE
                | AccessFlags::TRANSFER_WRITE
                | AccessFlags::SHADER_READ,
            dst_access: AccessFlags::ACCELERATION_STRUCTURE_READ
                | AccessFlags::ACCELERATION_STRUCTURE_WRITE
                | AccessFlags::SHADER_READ,
            src_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD
                | PipelineStages::ALL_TRANSFER
                | PipelineStages::RAY_TRACING_SHADER,
            dst_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD,
            ..Default::default()
        };

        let mut build_geometry_info =
            AccelerationStructureBuildGeometryInfo::new(self.geometries.clone());

        let scratch_buffer = Subbuffer::new(tcx.buffer(self.scratch_buffer_id).buffer().clone());

        let back_index = usize::from(!rcx.current_as_index.load(Ordering::Acquire));
        let dst = rcx.acceleration_structures[back_index].clone();

        build_geometry_info.mode = BuildAccelerationStructureMode::Build;
        build_geometry_info.flags = BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE;
        build_geometry_info.dst_acceleration_structure = Some(dst);
        build_geometry_info.scratch_data = Some(scratch_buffer);

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[pre_memory_barrier],
                ..Default::default()
            })
        };

        unsafe {
            cbf.as_raw().build_acceleration_structure(
                &build_geometry_info,
                &[AccelerationStructureBuildRangeInfo {
                    primitive_count: self.instance_count,
                    ..Default::default()
                }],
            )
        };

        let post_memory_barrier = MemoryBarrier {
            src_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE,
            dst_access: AccessFlags::ACCELERATION_STRUCTURE_READ | AccessFlags::SHADER_READ,
            src_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD,
            dst_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD
                | PipelineStages::RAY_TRACING_SHADER,
            ..Default::default()
        };

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[post_memory_barrier],
                ..Default::default()
            })
        };

        Ok(())
    }
}
