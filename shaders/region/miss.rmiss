#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../common/common.glsl"
#include "header.glsl"
#include "common.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

void main() {
    incoming_payload.color = vec4(vec3(0.0), 1.0);
    incoming_payload.t = 0.0;
    incoming_payload.hit_kind = 0u;
    incoming_payload.normal = vec3(0.0);
}
