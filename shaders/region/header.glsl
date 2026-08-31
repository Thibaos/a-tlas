#extension GL_EXT_ray_tracing : require

#define VKO_ACCELERATION_STRUCTURE_ENABLED 1

#include <vulkano.glsl>

VKO_DECLARE_STORAGE_BUFFER(camera, Camera{
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 view_prev;
    mat4 proj_prev;
})

VKO_DECLARE_STORAGE_BUFFER(palette, Palette{
    vec4[256] colors;
})

VKO_DECLARE_STORAGE_BUFFER(material_table, MaterialTable{
    vec4[256] albedo_metallic;
    vec4[256] rough_emit;
    uint[256] flags;
})

VKO_DECLARE_STORAGE_BUFFER(scene, Scene{
    vec4 sun_dir;
    vec4 sky_knots;
    vec4 sun_disk;
})

#define camera vko_buffer(camera, camera_buffer_id)
#define palette vko_buffer(palette, palette_buffer_id)
#define material_table vko_buffer(material_table, material_table_buffer_id)
#define scene vko_buffer(scene, scene_buffer_id)
