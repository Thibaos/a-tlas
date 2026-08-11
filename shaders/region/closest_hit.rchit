#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../rt/common.glsl"
#include "header.glsl"
#include "common.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

// The material rides the 8-bit hitKind reported by the intersection shader.
// Palette index 0 is a real color — the palette lookup is unconditional, and
// the Occupancy mask (not a sentinel material) defined existence upstream.
void main() {
    incoming_payload.color = palette.colors[gl_HitKindEXT];
    incoming_payload.t = gl_RayTmaxEXT;
}
