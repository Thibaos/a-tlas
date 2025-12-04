#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#define VKO_ACCELERATION_STRUCTURE_ENABLED 1

#include <vulkano.glsl>

VKO_DECLARE_STORAGE_BUFFER(camera, Camera{
    // Camera view * projection
    mat4 view_proj;
    // Camera inverse view matrix
    mat4 view_inverse;
    // Camera inverse projection matrix
    mat4 proj_inverse;
})

VKO_DECLARE_STORAGE_BUFFER(palette, Palette{
    vec4[256] colors;
})

VKO_DECLARE_STORAGE_BUFFER(sunlight, Sunlight{
    vec3 direction;
})

#define camera vko_buffer(camera, camera_buffer_id)
#define palette vko_buffer(palette, palette_buffer_id)
#define sunlight vko_buffer(sunlight, sunlight_buffer_id)

layout(push_constant) uniform PushConstants {
    StorageImageId image_id;
    AccelerationStructureId acceleration_structure_id;
    StorageBufferId camera_buffer_id;
    StorageBufferId palette_buffer_id;
    StorageBufferId sunlight_buffer_id;
};
