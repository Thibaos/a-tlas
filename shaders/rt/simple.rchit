#version 460

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference2 : require

#include "common.glsl"
#include "deps.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_static_payload;

float xor(float a, float b) {
    return a + b - a * b * (1.0 + a + b - a * b);
}

float compute_edge(float edge_width, vec3 box_position) {
    float edge_x = (step(0.5 - edge_width, box_position.r) + step(box_position.r, -0.5 + edge_width));
    float edge_y = (step(0.5 - edge_width, box_position.g) + step(box_position.g, -0.5 + edge_width));
    float edge_z = (step(0.5 - edge_width, box_position.b) + step(box_position.b, -0.5 + edge_width));

    float edge = xor(xor(edge_x, edge_y), edge_z) - (edge_x * edge_y * edge_z);

    return edge;
}

void main() {
    // incoming_static_payload.color = vec4(1.0);
    incoming_static_payload.color = palette.colors[gl_InstanceCustomIndexEXT];
    incoming_static_payload.t = gl_RayTmaxEXT;
}
