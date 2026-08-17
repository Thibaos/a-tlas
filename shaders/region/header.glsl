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

// The Material table (ticket 03, ADR 0008): one entry per palette index
// (the MATL chunk's material id == the palette index), packed as two vec4
// columns — albedo.rgb + metallic, and emission.rgb + roughness — the GPU
// twin of src/world/voxel.rs's Material mirror (uploaded once at startup;
// the albedo column equals the Palette by construction, so the byte-exact
// capture path is unchanged). Indexed by the 8-bit hitKind in closest-hit;
// the raygen reads it through the payload's hit_kind.
VKO_DECLARE_STORAGE_BUFFER(material_table, MaterialTable{
    vec4[256] albedo_metallic;
    vec4[256] rough_emit;
})

// The Scene buffer (ticket 06): the analytic lights' constants — the Sun
// (delta: world direction + illuminance) and the Procedural sky (the
// μ-gradient knots + the disk). Read by the production sky miss shader and
// the production raygen's Voxel mode (NEE); the capture stages declare the
// binding but never dereference it, so the byte-exact validator is
// unchanged. All values are data (packed by the app); the CPU path tracer
// (07) mirrors the same packed values.
VKO_DECLARE_STORAGE_BUFFER(scene, Scene{
    vec4 sun_dir;   // xyz = unit sun direction (world); w = unused
    vec4 sky_knots; // x = ground radiance (μ = -1), y = horizon (μ = 0), z = zenith (μ = 1); w = unused
    vec4 sun_disk;  // x = E_sun (illuminance, lux), y = cos(θ_disk), z = L_disk (disk radiance); w = unused
})

#define camera vko_buffer(camera, camera_buffer_id)
#define palette vko_buffer(palette, palette_buffer_id)
#define material_table vko_buffer(material_table, material_table_buffer_id)
#define scene vko_buffer(scene, scene_buffer_id)
