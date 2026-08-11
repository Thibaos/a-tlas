// The Region path's own header (renderer-impl ticket 02). Deliberately NOT
// shaders/rt/deps.glsl: that file also declares the retired path's nameless
// push-constant block, whose members leak into the global scope and collide
// with the Region push constants. The Camera and Palette buffer declarations
// are byte-identical to deps.glsl (the struct registry validates them equal
// across shaders), so both paths share the bindless layout.

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
