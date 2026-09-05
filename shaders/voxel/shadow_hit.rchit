#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../common/header.glsl"
#include "../common/bindings.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload shadow_payload;

void main() {
    shadow_payload.attenuation = 0.25;
}
