use std::{sync::Arc, time::Instant};

use vulkano::{
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryAabbsData, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
        AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode, GeometryFlags,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, Queue},
    format::Format,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};
use vulkano_taskgraph::{
    Id,
    resource::{Flight, Resources},
};

use crate::world::Vertex3D;

#[allow(clippy::too_many_arguments)]
pub fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    mode: BuildAccelerationStructureMode,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> Arc<AccelerationStructure> {
    let now = Instant::now();

    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: mode.clone(),
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    let scratch_buffer = Buffer::new_slice::<u8>(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    let as_buffer = Buffer::new_slice::<u8>(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.acceleration_structure_size,
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        ty,
        ..AccelerationStructureCreateInfo::new(&as_buffer)
    };

    let acceleration = unsafe { AccelerationStructure::new(&device, &as_create_info) }.unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    unsafe {
        vulkano_taskgraph::execute(
            &queue,
            resources,
            flight_id,
            |cbf, _tcx| {
                cbf.as_raw()
                    .build_acceleration_structure(&as_build_geometry_info, &[as_build_range_info])
                    .unwrap();

                Ok(())
            },
            [],
            [],
            [],
        )
        .unwrap()
    };

    match ty {
        AccelerationStructureType::TopLevel => {
            print!("build tlas ");
        }
        AccelerationStructureType::BottomLevel => {
            print!("build blas ");
        }
        _ => {}
    }

    let elapsed = now.elapsed();

    print!("cmd: {elapsed:.2?}, ");

    resources.flight(flight_id).unwrap().wait_idle().unwrap();

    let elapsed = now.elapsed();
    println!("wait: {elapsed:.2?}");

    acceleration
}

pub fn build_blas(
    vertex_buffer: Subbuffer<[Vertex3D]>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> Arc<AccelerationStructure> {
    let primitive_count = (vertex_buffer.len() / 3) as u32;
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        vertex_stride: size_of::<Vertex3D>() as _,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    build_acceleration_structure_common(
        geometries,
        BuildAccelerationStructureMode::Build,
        primitive_count,
        AccelerationStructureType::BottomLevel,
        memory_allocator,
        device,
        queue,
        resources,
        flight_id,
    )
}

pub fn build_tlas(
    as_instances: Vec<AccelerationStructureInstance>,
    allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> Arc<AccelerationStructure> {
    let primitive_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
        &allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        as_instances,
    )
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    build_acceleration_structure_common(
        geometries,
        BuildAccelerationStructureMode::Build,
        primitive_count,
        AccelerationStructureType::TopLevel,
        allocator,
        device,
        queue,
        resources,
        flight_id,
    )
}

pub fn build_blas_aabb(
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> Arc<AccelerationStructure> {
    let aabbs = [AabbPositions {
        min: [-0.5; 3],
        max: [0.5; 3],
    }];

    let primitive_count = aabbs.len() as u32;

    let data = Buffer::from_iter(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        aabbs,
    )
    .unwrap();

    let as_geometry_aabb_data = AccelerationStructureGeometryAabbsData {
        flags: GeometryFlags::OPAQUE,
        data: Some(data.into_bytes()),
        stride: 4 * 4,
        ..AccelerationStructureGeometryAabbsData::new()
    };

    let geometries = AccelerationStructureGeometries::Aabbs(vec![as_geometry_aabb_data]);

    build_acceleration_structure_common(
        geometries,
        BuildAccelerationStructureMode::Build,
        primitive_count,
        AccelerationStructureType::BottomLevel,
        memory_allocator,
        device,
        queue,
        resources,
        flight_id,
    )
}
