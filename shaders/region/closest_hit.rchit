#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../common/common.glsl"
#include "header.glsl"
#include "common.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

// The material rides the 8-bit hitKind reported by the intersection shader.
// Palette index 0 is a real color. The palette lookup is unconditional, and
// the Occupancy mask (not a sentinel material) defined existence upstream.
void main() {
    // The committed voxel's Material (ADR 0008): the table's albedo column
    // equals the Palette color by construction, so this stays the same
    // surface color the capture path always wrote, byte-identical frames,
    // while the read itself exercises the GPU table on every hit. Alpha is
    // forced 1.0: palette alpha is not a material property (the palette
    // buffer is uploaded with alpha 1.0; the reference tracer forces it too).
    incoming_payload.color = vec4(material_table.albedo_metallic[gl_HitKindEXT].rgb, 1.0);
    incoming_payload.t = gl_RayTmaxEXT;
    // The surface identity the raygen needs for its own table read (ticket
    // 03: real metalness + emission in Voxel mode; later, the normal).
    incoming_payload.hit_kind = gl_HitKindEXT;

    // The geometric normal at the hit (ticket 04, ADR 0009): the face the
    // DDA's march entered the committed voxel through. The intersection
    // shader knows it exactly (the last Amanatides-Woo step axis), but
    // reportIntersectionEXT carries only (t, 8-bit hitKind) and the payload
    // is opaque to intersection shaders, so the closest-hit recovers the
    // face from the hit point: the reported t is the cell-entry boundary
    // crossing, so p[a] is an integer (within epsilon) exactly on the
    // crossed axis, and strictly interior on the others. Ties (edge/corner
    // entries) break to the first axis in x, y, z order, the DDA's own
    // preference order; the canonical rule is documented in ADR 0009.
    // The epsilon is relative, ~8 ULP of |p[a]| (f32 at world scale),
    // because the reconstruction error grows with the coordinate magnitude:
    // the DDA's t and the p recomputation each round to ~1 ULP of their
    // operands, so a fixed epsilon that covers far-away voxels (p ~ 2000 →
    // ULP ~ 2.4e-4) would swallow near-corner entries nearby (p ~ 10 →
    // ULP ~ 1e-6). 8 ULP keeps the false-positive window on non-crossed
    // axes a fraction of a thousandth of a cell.
    vec3 hit_point = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_RayTmaxEXT;
    int face = -1;
    for (int a = 0; a < 3; ++a) {
        // A parallel axis never crosses a boundary, so it can't be the face
        // even if p[a] sits on one by coincidence (grazing an edge plane).
        if (gl_ObjectRayDirectionEXT[a] == 0.0) {
            continue;
        }
        float f = hit_point[a] - floor(hit_point[a]);
        float eps = 8.0 * 1.1920929e-07 * max(abs(hit_point[a]), 1.0);
        // Low-face (p = k, entered moving +a) and high-face (p = k + 1,
        // entered moving -a) both give f ~ 0 up to float error; the sign of
        // the direction orients the outward normal.
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
