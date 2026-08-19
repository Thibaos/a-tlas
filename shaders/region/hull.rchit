#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../common/common.glsl"
#include "header.glsl"
#include "common.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

// Hull mode's closest hit: expand the 8-bit coordinate hash (the hit kind) to
// an RGB color in-shader via a golden-ratio HSV ramp. No palette lookup (that
// is the DDA closest-hit's job, and no color buffer is needed here). The
// golden-ratio conjugate spreads adjacent hash values far apart in hue, so
// neighboring chunks stay distinct even though the hash is folded to 8 bits.

vec3 hsv_to_rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // hue = hash * golden-ratio conjugate (fractional part); saturation and
    // value fixed, so the ramp is 256 distinct hues.
    float hue = fract(float(gl_HitKindEXT) * 0.618033988749895);
    vec3 rgb = hsv_to_rgb(vec3(hue, 1.0, 1.0));
    incoming_payload.color = vec4(rgb, 1.0);
    incoming_payload.t = gl_RayTmaxEXT;
    // Surface identity (never read by the Hull paint path; set for hygiene
    // so a payload inspection is well-defined).
    incoming_payload.hit_kind = gl_HitKindEXT;
    // No surface normal in Hull mode (the hull AABB face is not a voxel
    // face; the paint path never reads it), zeroed for hygiene.
    incoming_payload.normal = vec3(0.0);
}
