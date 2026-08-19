#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../common/common.glsl"
#include "header.glsl"
#include "common.glsl"
#include "sky.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

// The production miss shader (ticket 06): the Procedural sky's radiance.
// The gradient evaluated at the ray's world direction (the disk is the
// camera's direct view, added by the raygen's primary-miss branch, so the
// transport's BSDF-miss samples and the env NEE see the gradient only).
// t = 0 (no-hit sentinel), hit_kind = 0, normal = 0. The payload contract
// the raygen reads.
void main() {
    incoming_payload.color = vec4(sky_radiance(gl_WorldRayDirectionEXT), 1.0);
    incoming_payload.t = 0.0;
    incoming_payload.hit_kind = 0u;
    incoming_payload.normal = vec3(0.0);
}
