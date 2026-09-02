use std::sync::Arc;

use anyhow::Ok;
use vulkano::{
    DeviceSize,
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureCreateInfo,
        AccelerationStructureGeometry, AccelerationStructureGeometryAabbsData,
        AccelerationStructureGeometryData, AccelerationStructureGeometryInstancesData,
        AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    sync::{AccessFlags, PipelineStages},
};
use vulkano_taskgraph::{
    Id,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    resource::{Flight, Resources},
};

use crate::core::render::gpu::GpuDesc;

pub type BuildGeometries = Vec<AccelerationStructureGeometry<'static>>;

pub fn as_build_pre_barrier(cbf: &mut RecordingCommandBuffer<'_>) {
    let pre_memory_barrier = MemoryBarrier {
        src_access: AccessFlags::TRANSFER_WRITE
            | AccessFlags::SHADER_WRITE
            | AccessFlags::HOST_WRITE
            | AccessFlags::ACCELERATION_STRUCTURE_WRITE,
        dst_access: AccessFlags::ACCELERATION_STRUCTURE_WRITE
            | AccessFlags::ACCELERATION_STRUCTURE_READ,
        src_stages: PipelineStages::ALL_TRANSFER
            | PipelineStages::COMPUTE_SHADER
            | PipelineStages::HOST
            | PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        dst_stages: PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        ..Default::default()
    };

    unsafe {
        cbf.pipeline_barrier(&DependencyInfo {
            memory_barriers: &[pre_memory_barrier],
            ..Default::default()
        });
    }
}

pub fn as_build_post_barrier(cbf: &mut RecordingCommandBuffer<'_>) {
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
        });
    }
}

pub fn build_flags(ty: AccelerationStructureType) -> BuildAccelerationStructureFlags {
    match ty {
        AccelerationStructureType::TopLevel => {
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE
                | BuildAccelerationStructureFlags::ALLOW_UPDATE
        }
        AccelerationStructureType::BottomLevel => {
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE
        }
        _ => unimplemented!(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_acceleration_structure_in_place(
    geometries: &BuildGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    dst: &Arc<AccelerationStructure>,
    storage_capacity: u64,
    memory_allocator: &Arc<dyn MemoryAllocator>,
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> anyhow::Result<Arc<AccelerationStructure>> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty,
        mode: BuildAccelerationStructureMode::Build,
        flags: build_flags(ty),
        geometries,
        ..AccelerationStructureBuildGeometryInfo::new()
    };

    let as_build_sizes_info = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[primitive_count],
    );
    debug_assert!(
        as_build_sizes_info.acceleration_structure_size <= storage_capacity,
        "in-place build of {primitive_count} primitives needs {} bytes but the storage holds {storage_capacity}",
        as_build_sizes_info.acceleration_structure_size
    );

    let scratch_buffer = Buffer::new_slice::<u8>(
        memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )?;

    as_build_geometry_info.dst_acceleration_structure = Some(dst);
    as_build_geometry_info.scratch_data = scratch_buffer.device_address()?.get();

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    unsafe {
        vulkano_taskgraph::execute(
            queue,
            resources,
            flight_id,
            |cbf, _tcx| {
                as_build_pre_barrier(cbf);
                cbf.as_raw()
                    .build_acceleration_structure(&as_build_geometry_info, &[as_build_range_info]);
                as_build_post_barrier(cbf);

                Result::<(), _>::Ok(())
            },
            [],
            [],
            [],
        )
    }?;

    resources.flight(flight_id).wait_idle()?;

    Ok(dst.clone())
}

pub fn acceleration_structure_build_sizes(
    device: &Arc<Device>,
    geometries: &[AccelerationStructureGeometry<'static>],
    ty: AccelerationStructureType,
    primitive_count: u32,
) -> vulkano::acceleration_structure::AccelerationStructureBuildSizesInfo {
    let as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty,
        mode: BuildAccelerationStructureMode::Build,
        flags: build_flags(ty),
        geometries,
        ..AccelerationStructureBuildGeometryInfo::new()
    };

    device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[primitive_count],
    )
}

pub fn create_blas_aabbs_storage(
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    primitive_count: u32,
    memory_allocator: &Arc<dyn MemoryAllocator>,
    device: &Arc<Device>,
) -> anyhow::Result<(Arc<AccelerationStructure>, u64)> {
    let geometries = aabb_geometries(aabb_buffer)?;

    let as_build_sizes_info = acceleration_structure_build_sizes(
        device,
        &geometries,
        AccelerationStructureType::BottomLevel,
        primitive_count,
    );

    let as_buffer = Buffer::new_slice::<u8>(
        memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.acceleration_structure_size,
    )?;

    let as_create_info = AccelerationStructureCreateInfo {
        size: as_build_sizes_info.acceleration_structure_size,
        ty: AccelerationStructureType::BottomLevel,
        ..AccelerationStructureCreateInfo::new(as_buffer.buffer())
    };

    let acceleration = unsafe { AccelerationStructure::new(device, &as_create_info) }?;

    Ok((
        acceleration,
        as_build_sizes_info.acceleration_structure_size,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn build_acceleration_structure_fresh(
    geometries: &BuildGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: &Arc<dyn MemoryAllocator>,
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> anyhow::Result<(Arc<AccelerationStructure>, u64)> {
    let as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty,
        mode: BuildAccelerationStructureMode::Build,
        flags: build_flags(ty),
        geometries,
        ..AccelerationStructureBuildGeometryInfo::new()
    };

    let as_build_sizes_info = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[primitive_count],
    );

    let as_buffer = Buffer::new_slice::<u8>(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.acceleration_structure_size,
    )?;

    let as_create_info = AccelerationStructureCreateInfo {
        size: as_build_sizes_info.acceleration_structure_size,
        ty,
        ..AccelerationStructureCreateInfo::new(as_buffer.buffer())
    };

    let acceleration = unsafe { AccelerationStructure::new(&device, &as_create_info) }?;

    let built = build_acceleration_structure_in_place(
        &geometries,
        primitive_count,
        ty,
        &acceleration,
        as_build_sizes_info.acceleration_structure_size,
        &memory_allocator,
        &device,
        &queue,
        resources,
        flight_id,
    )?;

    Ok((built, as_build_sizes_info.acceleration_structure_size))
}

pub fn build_blas_aabbs_fresh(
    aabb_buffer: Subbuffer<[AabbPositions]>,
    primitive_count: u32,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> anyhow::Result<(Arc<AccelerationStructure>, u64)> {
    build_acceleration_structure_fresh(
        &aabb_geometries(&aabb_buffer)?,
        primitive_count,
        AccelerationStructureType::BottomLevel,
        &memory_allocator,
        &device,
        &queue,
        resources,
        flight_id,
    )
}

pub fn create_tlas_storage(
    instance_buffer: &Subbuffer<[AccelerationStructureInstance]>,
    max_instances: u32,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
) -> anyhow::Result<(Arc<AccelerationStructure>, u64)> {
    let geometries = instance_geometries(instance_buffer)?;

    let as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty: AccelerationStructureType::TopLevel,
        mode: BuildAccelerationStructureMode::Build,
        flags: build_flags(AccelerationStructureType::TopLevel),
        geometries: &geometries,
        ..AccelerationStructureBuildGeometryInfo::new()
    };

    let as_build_sizes_info = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[max_instances],
    );

    let as_buffer = Buffer::new_slice::<u8>(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.acceleration_structure_size,
    )?;

    let as_create_info = AccelerationStructureCreateInfo {
        size: as_build_sizes_info.acceleration_structure_size,
        ty: AccelerationStructureType::TopLevel,
        ..AccelerationStructureCreateInfo::new(as_buffer.buffer())
    };

    let acceleration = unsafe { AccelerationStructure::new(&device, &as_create_info) }?;

    Ok((
        acceleration,
        as_build_sizes_info.acceleration_structure_size,
    ))
}

pub fn aabb_geometries(
    aabb_buffer: &Subbuffer<[AabbPositions]>,
) -> anyhow::Result<BuildGeometries> {
    let aabb_data = AccelerationStructureGeometryAabbsData {
        data: aabb_buffer.device_address()?.get(),
        stride: size_of::<AabbPositions>() as u32,
        ..Default::default()
    };

    Ok(vec![AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Aabbs(aabb_data),
    )])
}

pub fn instance_geometries(
    instance_buffer: &Subbuffer<[AccelerationStructureInstance]>,
) -> anyhow::Result<BuildGeometries> {
    let instances_data = AccelerationStructureGeometryInstancesData {
        data: instance_buffer.device_address()?.get(),
        ..Default::default()
    };

    Ok(vec![AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Instances(instances_data),
    )])
}

pub fn blas_build_sizes(
    gpu: &GpuDesc,
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    aabb_count: u32,
) -> anyhow::Result<AccelerationStructureBuildSizesInfo> {
    Ok(acceleration_structure_build_sizes(
        &gpu.device,
        &aabb_geometries(aabb_buffer)?,
        AccelerationStructureType::BottomLevel,
        aabb_count,
    ))
}

pub fn tlas_build_sizes(
    gpu: &GpuDesc,
    instance_buffer: &Subbuffer<[AccelerationStructureInstance]>,
    instance_count: u32,
) -> anyhow::Result<AccelerationStructureBuildSizesInfo> {
    Ok(acceleration_structure_build_sizes(
        &gpu.device,
        &instance_geometries(instance_buffer)?,
        AccelerationStructureType::TopLevel,
        instance_count,
    ))
}

pub fn allocate_scratch(gpu: &GpuDesc, size: DeviceSize) -> Arc<Buffer> {
    Buffer::new_slice::<u8>(
        &gpu.memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        size.max(1),
    )
    .unwrap()
    .buffer()
    .clone()
}
