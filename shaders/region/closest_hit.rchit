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
    const bool glass = (material_table.flags[gl_HitKindEXT] & MATFLAG_GLASS) != 0u;

    // Palette entries are sRGB; the display path (ACES + gamma) is linear.
    incoming_payload.color = vec4(pow(palette.colors[gl_HitKindEXT].rgb, vec3(2.2)), 1.0);
    incoming_payload.t = gl_RayTmaxEXT;
    incoming_payload.hit_kind = gl_HitKindEXT;

    vec3 hit_point = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_RayTmaxEXT;
    int face = -1;

    for (int a = 0; a < 3; ++a) {
        if (gl_ObjectRayDirectionEXT[a] == 0.0) {
            continue;
        }

        float f = hit_point[a] - floor(hit_point[a]);
        // The reported t is the division-form crossing of the entered face
        // (intersect.rint snaps it), so p = o + d*t sits on the boundary to
        // within the residual of that division plus the mul/add here — up to
        // a dozen ULP of the coordinate scale. Object space caps |p| at the
        // region edge, so even 32 ULP keeps the false-positive window on
        // non-crossed axes a fraction of a percent of a cell.
        float eps = 32.0 * 1.1920929e-07 * max(abs(hit_point[a]), 1.0);

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
