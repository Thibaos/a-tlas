#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 vertex_color;
layout(location = 1) out float vertex_depth;

layout(push_constant) uniform PushConstants {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    vertex_color = color;

    mat4 worldview = uniforms.view * uniforms.world;
    vec4 position = uniforms.proj * worldview * vec4(position, 1.0);
    gl_Position = position;
    vertex_depth = clamp(position.z / 300.0, 0.0, 1.0);
}