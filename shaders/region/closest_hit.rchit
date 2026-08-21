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
    incoming_payload.color = vec4(material_table.albedo_metallic[gl_HitKindEXT].rgb, 1.0);
    incoming_payload.t = gl_RayTmaxEXT;
    incoming_payload.hit_kind = gl_HitKindEXT;

    vec3 hit_point = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_RayTmaxEXT;
    int face = -1;

    for (int a = 0; a < 3; ++a) {
        if (gl_ObjectRayDirectionEXT[a] == 0.0) {
            continue;
        }

        float f = hit_point[a] - floor(hit_point[a]);
        float eps = 8.0 * 1.1920929e-07 * max(abs(hit_point[a]), 1.0);

        if (f < eps || f > 1.0 - eps) {
            face = a;
            break;
        }
    }

    vec3 normal = -normalize(gl_ObjectRayDirectionEXT);

    if (face >= 0) {
        normal = vec3(0.0);
        normal[face] = -sign(gl_ObjectRayDirectionEXT[face]);
    }

    incoming_payload.normal = normal;
}
