#version 450

layout(location = 0) in vec4 vertex_color;
layout(location = 1) in float vertex_depth;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(vertex_color.rgb, 1. - vertex_depth);
}