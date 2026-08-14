// The Region path's header: the Camera and Palette buffer declarations shared
// by every Region stage (raygen, miss, intersection, closest-hit).

#extension GL_EXT_ray_tracing : require

#define VKO_ACCELERATION_STRUCTURE_ENABLED 1

#include <vulkano.glsl>

VKO_DECLARE_STORAGE_BUFFER(camera, Camera{
    // Camera inverse view matrix
    mat4 view_inverse;
    // Camera inverse projection matrix
    mat4 proj_inverse;
})

VKO_DECLARE_STORAGE_BUFFER(palette, Palette{
    vec4[256] colors;
})

#define camera vko_buffer(camera, camera_buffer_id)
#define palette vko_buffer(palette, palette_buffer_id)
