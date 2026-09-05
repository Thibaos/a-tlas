#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "common/header.glsl"
#include "common/bindings.glsl"
#include "common/sky.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;
layout(location = 1) rayPayloadInEXT ShadowPayload shadow_payload;

void main() {
    incoming_payload.color = vec4(sky_radiance(gl_WorldRayDirectionEXT), 1.0);
    incoming_payload.normal = vec3(0.0);
    shadow_payload.attenuation = 1.0;
}
