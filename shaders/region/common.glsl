#include "../common/contract.glsl"

layout(buffer_reference, std430) readonly buffer RegionPool {
    uint words[];
};

VKO_DECLARE_STORAGE_BUFFER(region_table, RegionTable {
    uint64_t bdas[REGION_TABLE_ENTRIES];
})

#define region_table vko_buffer(region_table, region_table_buffer_id)

VKO_DECLARE_STORAGE_BUFFER(aabb_table, AabbTable {
    uint64_t bdas[REGION_TABLE_ENTRIES];
})

#define aabb_table vko_buffer(aabb_table, aabb_table_buffer_id)

layout(push_constant) uniform RegionPushConstants {
    StorageImageId image_id;
    AccelerationStructureId acceleration_structure_id;
    StorageBufferId camera_buffer_id;
    StorageBufferId palette_buffer_id;
    StorageBufferId material_table_buffer_id;
    StorageBufferId scene_buffer_id;
    StorageBufferId region_table_buffer_id;
    StorageBufferId aabb_table_buffer_id;
    StorageBufferId cache_table_buffer_id;
    StorageBufferId cache_state_buffer_id;
    uint mode;
    uint frame_seed;
    StorageImageId diff_radiance_image_id;
    StorageImageId spec_radiance_image_id;
    StorageImageId normal_roughness_image_id;
    StorageImageId viewz_image_id;
    StorageImageId mv_image_id;
    StorageImageId albedo_metal_image_id;
};
