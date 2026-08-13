// Shared declarations for the Region render path.
//
// Voxel pools: each Region owns one GPU buffer that starts with a
// u32 offset table (32768 Micro-chunk slots, sentinel 0xFFFFFFFF for empty)
// followed by compact blocks, one per non-empty Micro-chunk: a 64-byte
// Occupancy mask (512 bits, bit idx = x + 8*y + 64*z, little-endian) then the
// popcount-compacted u8 material indices in increasing bit order, padded to
// 8-byte alignment. The intersection shader reaches a Region's pool through
// the Region table: a bindless u64[4096] of pool device addresses indexed by
// the 12-bit Region id that rides the instance custom index.
//
// The ray t-range equals the camera near/far: rays clip exactly
// like the camera. The CPU reference tracer mirrors these constants.

#define RAY_T_MIN 0.01
#define RAY_T_MAX 10000.0

#define REGION_TABLE_ENTRIES 4096
#define OFFSET_SENTINEL 0xFFFFFFFFu

layout(buffer_reference, std430) readonly buffer RegionPool {
    uint words[];
};

VKO_DECLARE_STORAGE_BUFFER(region_table, RegionTable {
    uint64_t bdas[REGION_TABLE_ENTRIES];
})

#define region_table vko_buffer(region_table, region_table_buffer_id)

// Region push constants, shared by every stage of the region pipeline. The raygen
// uses the images, the camera and the palette; the intersection shader uses
// only `region_table_buffer_id` (the DDA resolves the pool through the
// Region table). All stages declare the same block so the pipeline has one
// push-constant range.
layout(push_constant) uniform RegionPushConstants {
    StorageImageId image_id;
    // Validation only: the ray pass additionally writes payload.t to this
    // image for the validator's per-pixel {color, t} comparison. The
    // production raygen passes INVALID and never dereferences it.
    StorageImageId t_image_id;
    AccelerationStructureId acceleration_structure_id;
    StorageBufferId camera_buffer_id;
    StorageBufferId palette_buffer_id;
    StorageBufferId region_table_buffer_id;
};
