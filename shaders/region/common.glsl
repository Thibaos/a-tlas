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

// The region -> AABB-buffer device-address table, parallel to `region_table`
// (one entry per Region id, 0 for non-resident). The DDA reads the invoking
// Micro-chunk's trimmed hull back through it by primitive id; the debug Hull
// mode uses the same table. Allocated, registered and written once at startup.
VKO_DECLARE_STORAGE_BUFFER(aabb_table, AabbTable {
    uint64_t bdas[REGION_TABLE_ENTRIES];
})

#define aabb_table vko_buffer(aabb_table, aabb_table_buffer_id)

// The march-and-miss counter: three uint counters the DDA intersection shader
// increments by atomicAdd — hull_crossed (the slab passed), march_and_miss
// (the march hit nothing), and empty_and (the forward box has no occupancy).
// Attached only under --measure; the increments are gated by the
// COUNTER_ENABLED specialization constant (false by default), so the
// default/validator pipelines render byte-identical output.
VKO_DECLARE_STORAGE_BUFFER(counter, Counter {
    uint hull_crossed;
    uint march_and_miss;
    uint empty_and;
})

#define counter vko_buffer(counter, counter_buffer_id)

// The per-pixel hull-crossed count buffer (debug builds): one uint per ray
// pass pixel, incremented by the DDA intersection shader's atomicAdd at
// slab-pass when the hull-crossed mode selects its hit group. Attached only in
// debug builds; the validator and release paths push INVALID and never
// dereference it (PER_PIXEL_COUNTER is false there).
VKO_DECLARE_STORAGE_BUFFER(hull_count, HullCountBuffer {
    uint pixels[];
})

#define hull_count vko_buffer(hull_count, hull_count_buffer_id)

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
    // The bindless Material table (ticket 03, ADR 0008): one entry per
    // palette index, packed albedo.rgb+metallic / emission.rgb+roughness.
    // The DDA closest-hit reads it for the surface color (albedo == palette
    // color by construction, so the byte-exact capture path is unchanged)
    // and the production raygen reads it through the payload's hit_kind in
    // Voxel mode (real metalness + emission-as-albedo-light in the stub).
    StorageBufferId material_table_buffer_id;
    StorageBufferId region_table_buffer_id;
    // The region -> AABB-buffer device-address table (the DDA's and the
    // Hull intersection shader's lookup, parallel to `region_table`).
    StorageBufferId aabb_table_buffer_id;
    // The march-and-miss counter buffer (bindless); INVALID when not
    // measuring — the DDA's increments are gated by COUNTER_ENABLED, so the
    // field is never dereferenced in the default/validator pipelines.
    StorageBufferId counter_buffer_id;
    // The per-pixel hull-crossed count buffer (bindless); INVALID when not in
    // the hull-crossed debug mode — the intersection shader's PER_PIXEL_COUNTER
    // gate folds the atomicAdd away in the default/validator pipelines.
    StorageBufferId hull_count_buffer_id;
    // Render mode: 0 = Voxel, 1 = Hull, 2 = Ray latency, 3 = hull-crossed —
    // what the raygen paints. The raygen maps it to a hit-group record offset
    // (Ray latency traces hit-region 0, the DDA, and only changes the paint).
    // Always present; always 0 in release (Voxel is the only mode).
    uint mode;
    // Path-tracing output contract (ADR 0007): the trace pass's noisy
    // radiance pair and auxiliary guide buffers, written by the production
    // raygen in Voxel mode (diffuse+specular radiance with in-lobe hit
    // distance in alpha, normal+roughness, linear viewZ, backward motion
    // vectors, albedo+metalness). The composite node exposes them (and, from
    // ticket 08, the Denoise pass consumes them). INVALID for the debug
    // modes, which paint the swapchain directly, and in the validator (the
    // capture raygen never writes them).
    StorageImageId diff_radiance_image_id;
    StorageImageId spec_radiance_image_id;
    StorageImageId normal_roughness_image_id;
    StorageImageId viewz_image_id;
    StorageImageId mv_image_id;
    StorageImageId albedo_metal_image_id;
};
