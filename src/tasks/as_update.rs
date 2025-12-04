use crate::RenderContext;
use vulkano::{
    DeviceSize, Packed24_8, acceleration_structure::AccelerationStructureInstance, buffer::Buffer,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
};

pub const MAX_INSTANCE_COUNT: u64 = 2u64.pow(20);
const UPDATES_PER_FRAME: u64 = 4096;

pub struct UpdateAccelerationStructureTask {
    pub blas_reference: u64,
    pub instance_buffer_id: Id<Buffer>,
}

impl Task for UpdateAccelerationStructureTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        _cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _rcx: &Self::World,
    ) -> TaskResult {
        let default_instance: AccelerationStructureInstance = AccelerationStructureInstance {
            acceleration_structure_reference: self.blas_reference,
            instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            ..Default::default()
        };

        const AS_SIZE: DeviceSize = size_of::<AccelerationStructureInstance>() as DeviceSize;

        let start_slice = rand::random_range(0..(MAX_INSTANCE_COUNT - UPDATES_PER_FRAME)) * AS_SIZE;

        let write_instance_buffer = tcx
            .write_buffer::<[AccelerationStructureInstance]>(
                self.instance_buffer_id,
                start_slice..(start_slice + UPDATES_PER_FRAME * AS_SIZE),
            )
            .unwrap();

        for o in write_instance_buffer.iter_mut() {
            *o = default_instance;
        }

        Ok(())
    }
}
