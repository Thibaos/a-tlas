use std::sync::Arc;

use vulkano::{
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometry,
        AccelerationStructureGeometryAabbsData, AccelerationStructureGeometryData,
        AccelerationStructureGeometryInstancesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
};

/// One or more geometry entries for an acceleration structure build (the new
/// flat-slice geometry API: the data types hold raw device addresses, so the
/// geometry list borrows nothing and is `'static`).
pub type BuildGeometries = Vec<AccelerationStructureGeometry<'static>>;

// The build's pre/post memory barriers (shared by every build path): the
// build input (TRANSFER_WRITE | SHADER_WRITE) must be visible before the
// build, and the built AS must be visible to later traces/rebuilds.
fn as_build_barriers(cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>) {
    let pre_memory_barrier = vulkano_taskgraph::command_buffer::MemoryBarrier {
        src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE
            | vulkano::sync::AccessFlags::SHADER_WRITE,
        dst_access: vulkano::sync::AccessFlags::ACCELERATION_STRUCTURE_WRITE
            | vulkano::sync::AccessFlags::ACCELERATION_STRUCTURE_READ,
        src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER
            | vulkano::sync::PipelineStages::COMPUTE_SHADER,
        dst_stages: vulkano::sync::PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        ..Default::default()
    };
    // SAFETY: `pipeline_barrier` is unsafe (synchronization must be
    // correct); the caller holds the command buffer inside an execute
    // closure's recording context, and these are the standard pre/post
    // build barriers from the original build helper.
    unsafe {
        cbf.pipeline_barrier(&vulkano_taskgraph::command_buffer::DependencyInfo {
            memory_barriers: &[pre_memory_barrier],
            ..Default::default()
        });
    }

    let post_memory_barrier = vulkano_taskgraph::command_buffer::MemoryBarrier {
        src_access: vulkano::sync::AccessFlags::ACCELERATION_STRUCTURE_WRITE,
        dst_access: vulkano::sync::AccessFlags::ACCELERATION_STRUCTURE_READ
            | vulkano::sync::AccessFlags::SHADER_READ,
        src_stages: vulkano::sync::PipelineStages::ACCELERATION_STRUCTURE_BUILD,
        dst_stages: vulkano::sync::PipelineStages::ACCELERATION_STRUCTURE_BUILD
            | vulkano::sync::PipelineStages::RAY_TRACING_SHADER,
        ..Default::default()
    };
    unsafe {
        cbf.pipeline_barrier(&vulkano_taskgraph::command_buffer::DependencyInfo {
            memory_barriers: &[post_memory_barrier],
            ..Default::default()
        });
    }
}

/// The flags for a fresh/in-place build of `ty` (TLAS builds keep
/// ALLOW_UPDATE so an update-mode build remains possible; BLAS builds do
/// not need it).
pub(crate) fn build_flags(ty: AccelerationStructureType) -> BuildAccelerationStructureFlags {
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

/// Builds `geometries` into the existing AS `dst` (mode Build) — an
/// **in-place build**: the AS object and its storage never move, so the
/// device address (and any TLAS instance referencing it) stays valid. The
/// storage was sized for `storage_capacity` at creation; the caller asserts
/// the new build fits.
///
/// Used by the fresh-build path (`build_acceleration_structure_fresh`, which
/// creates the storage sized exactly and builds into it — the dummy BLAS
/// path). The residency manager's ordered rebuild nodes record the same in-place builds into ordered taskgraph nodes
/// instead of calling this synchronously.
#[allow(clippy::too_many_arguments)]
pub fn build_acceleration_structure_in_place(
    geometries: BuildGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    dst: &Arc<AccelerationStructure>,
    storage_capacity: u64,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> Arc<AccelerationStructure> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty,
        mode: BuildAccelerationStructureMode::Build,
        flags: build_flags(ty),
        geometries: &geometries,
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
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(dst);
    as_build_geometry_info.scratch_data = scratch_buffer.device_address().unwrap().get();

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
                as_build_barriers(cbf);
                cbf.as_raw()
                    .build_acceleration_structure(&as_build_geometry_info, &[as_build_range_info]);
                Ok(())
            },
            [],
            [],
            [],
        )
        .unwrap()
    };

    resources.flight(flight_id).wait_idle().unwrap();

    dst.clone()
}

/// Computes the build sizes (acceleration-structure storage + scratch) for
/// `geometries` with `primitive_count` primitives of type `ty` — the
/// CPU-side sizing the ordered rebuild nodes use
/// to create fresh storage and scratch buffers before recording.
pub(crate) fn acceleration_structure_build_sizes(
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

/// Creates the AS storage for a procedural AABB BLAS over `aabb_buffer`,
/// sized exactly for `primitive_count` AABBs — **without building**. The
/// ordered rebuild nodes create the storage in the
/// plan phase (CPU) and record the build later, so a become-resident BLAS
/// never moves after creation (the node builds into the pre-created storage
/// in place). Returns the BLAS and its storage size (the residency manager's
/// free-list reuse unit).
pub(crate) fn create_blas_aabbs_storage(
    aabb_buffer: &Subbuffer<[AabbPositions]>,
    primitive_count: u32,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
) -> (Arc<AccelerationStructure>, u64) {
    let aabb_data = AccelerationStructureGeometryAabbsData {
        data: aabb_buffer.device_address().unwrap().get(),
        stride: size_of::<AabbPositions>() as u32,
        ..Default::default()
    };

    let geometries = vec![AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Aabbs(aabb_data),
    )];

    let as_build_sizes_info = acceleration_structure_build_sizes(
        &device,
        &geometries,
        AccelerationStructureType::BottomLevel,
        primitive_count,
    );

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
        size: as_build_sizes_info.acceleration_structure_size,
        ty: AccelerationStructureType::BottomLevel,
        ..AccelerationStructureCreateInfo::new(as_buffer.buffer())
    };

    let acceleration = unsafe { AccelerationStructure::new(&device, &as_create_info) }.unwrap();

    (
        acceleration,
        as_build_sizes_info.acceleration_structure_size,
    )
}

/// Creates a fresh AS (storage sized exactly for `primitive_count`) and
/// builds it. Returns the AS and its storage size (the residency manager
/// tracks it for free-list reuse checks).
#[allow(clippy::too_many_arguments)]
pub fn build_acceleration_structure_fresh(
    geometries: BuildGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> (Arc<AccelerationStructure>, u64) {
    let as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty,
        mode: BuildAccelerationStructureMode::Build,
        flags: build_flags(ty),
        geometries: &geometries,
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
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        size: as_build_sizes_info.acceleration_structure_size,
        ty,
        ..AccelerationStructureCreateInfo::new(as_buffer.buffer())
    };

    let acceleration = unsafe { AccelerationStructure::new(&device, &as_create_info) }.unwrap();

    let built = build_acceleration_structure_in_place(
        // The geometry info is re-derived inside; pass the geometries again.
        // (The call below re-runs build_sizes with the identical geometry,
        // so the assertion against the freshly-sized storage holds.)
        geometries,
        primitive_count,
        ty,
        &acceleration,
        as_build_sizes_info.acceleration_structure_size,
        memory_allocator,
        device,
        queue,
        resources,
        flight_id,
    );

    (built, as_build_sizes_info.acceleration_structure_size)
}

use vulkano_taskgraph::{
    Id,
    resource::{Flight, Resources},
};

/// Builds a fresh procedural AABB BLAS over `aabb_buffer` (a taskgraph-owned
/// build-input buffer) and returns it plus its AS storage size (the residency
/// manager's free-list reuse unit). Storage is sized exactly for
/// `primitive_count` AABBs.
pub fn build_blas_aabbs_fresh(
    aabb_buffer: Subbuffer<[AabbPositions]>,
    primitive_count: u32,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> (Arc<AccelerationStructure>, u64) {
    let aabb_data = AccelerationStructureGeometryAabbsData {
        data: aabb_buffer.device_address().unwrap().get(),
        stride: size_of::<AabbPositions>() as u32,
        ..Default::default()
    };

    build_acceleration_structure_fresh(
        vec![AccelerationStructureGeometry::new(
            AccelerationStructureGeometryData::Aabbs(aabb_data),
        )],
        primitive_count,
        AccelerationStructureType::BottomLevel,
        memory_allocator,
        device,
        queue,
        resources,
        flight_id,
    )
}

/// Creates a TLAS object whose storage is pre-sized for `max_instances` —
/// the residency manager's **stable TLAS**: rebuilt
/// in place on every residency transition with up to `max_instances`
/// instances, so the object, its storage, and the bindless
/// acceleration-structure id never move. Returns the TLAS and its storage
/// size.
pub fn create_tlas_storage(
    instance_buffer: &Subbuffer<[AccelerationStructureInstance]>,
    max_instances: u32,
    memory_allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
) -> (Arc<AccelerationStructure>, u64) {
    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData {
        data: instance_buffer.device_address().unwrap().get(),
        ..Default::default()
    };

    let geometries = vec![AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Instances(as_geometry_instances_data),
    )];

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
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        size: as_build_sizes_info.acceleration_structure_size,
        ty: AccelerationStructureType::TopLevel,
        ..AccelerationStructureCreateInfo::new(as_buffer.buffer())
    };

    let acceleration = unsafe { AccelerationStructure::new(&device, &as_create_info) }.unwrap();

    (
        acceleration,
        as_build_sizes_info.acceleration_structure_size,
    )
}
