#extension GL_EXT_ray_tracing : require

#define VKO_ACCELERATION_STRUCTURE_ENABLED 1

#include <vulkano.glsl>

VKO_DECLARE_STORAGE_BUFFER(camera, Camera{
    mat4 view_inverse;
    mat4 proj_inverse;
})

VKO_DECLARE_STORAGE_BUFFER(palette, Palette{
    vec4[256] colors;
})

VKO_DECLARE_STORAGE_BUFFER(scene, Scene{
    vec4 sky_knots;
})

#define camera vko_buffer(camera, camera_buffer_id)
#define palette vko_buffer(palette, palette_buffer_id)
#define scene vko_buffer(scene, scene_buffer_id)
