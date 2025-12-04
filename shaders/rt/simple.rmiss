#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "common.glsl"
#include "deps.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

void main() {
    // incoming_payload.color = vec4(gl_WorldRayDirectionEXT, 1.0);
    incoming_payload.color = vec4(vec3(0.0), 1.0);
    incoming_payload.t = 0.0;
}
