use std::sync::atomic::Ordering;

use vulkano::{
    DeviceSize, Packed24_8,
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
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
};

use crate::{
    app::{App, RenderContext},
    utils,
};

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
        let instance_buffer = Subbuffer::new(
            app.resources
                .buffer(instance_buffer_id)
                .unwrap()
                .buffer()
                .clone(),
        )
        .cast_aligned::<AccelerationStructureInstance>();

        let geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer.clone())),
        );

        let geometries = AccelerationStructureGeometries::Instances(geometry_instances_data);

        let build_geometry_info = AccelerationStructureBuildGeometryInfo::new(geometries.clone());

        let build_sizes_info = app
            .device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &build_geometry_info,
                &[instance_count],
            )
            .unwrap();

        // `mode` is ignored by vulkano's size query, so the driver computes for Build mode.
        // `build_scratch_size` is a safe upper bound for any update operation.
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
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        if !rcx.tlas_update_requested {
            return Ok(());
        }

        let write_instance_buffer = tcx.write_buffer::<[AccelerationStructureInstance]>(
            self.instance_buffer_id,
            0..(self.instance_count as u64 * AS_SIZE),
        )?;

        for instance in write_instance_buffer.iter_mut() {
            let radius = self.instance_count.ilog2().pow(2) as f32;
            let (x, y, z) = utils::sample_uniform_sphere(radius);

            *instance = AccelerationStructureInstance {
                acceleration_structure_reference: self.blas_reference,
                instance_custom_index_and_mask: Packed24_8::new(1, 0xFF),
                transform: [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z]],
                ..Default::default()
            };
        }

        // Make the instance buffer write (TRANSFER) visible to the AS build's
        // internal read of the instance buffer (SHADER_READ in AS_BUILD stage),
        // as well as making the previous frame's AS writes visible to this update.
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

        let scratch_buffer =
            Subbuffer::new(tcx.buffer(self.scratch_buffer_id).unwrap().buffer().clone());

        // Build a fresh TLAS into the back buffer (the one NOT currently rendered),
        // then flip the index so the render task uses the new one.
        let back_index = usize::from(!rcx.current_as_index.load(Ordering::Relaxed));
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
        }?;

        unsafe {
            cbf.as_raw().build_acceleration_structure(
                &build_geometry_info,
                &[AccelerationStructureBuildRangeInfo {
                    primitive_count: self.instance_count,
                    ..Default::default()
                }],
            )
        }?;

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
        }?;

        // Flip the index so the render task traces against the freshly-built TLAS.
        rcx.current_as_index.store(
            !rcx.current_as_index.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );

        Ok(())
    }
}
