#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "../rt/common.glsl"
#include "header.glsl"
#include "common.glsl"

layout(location = 0) rayPayloadInEXT MainPassPayload incoming_payload;

// The material rides the 8-bit hitKind reported by the intersection shader.
// Palette index 0 is a real color — the palette lookup is unconditional, and
// the Occupancy mask (not a sentinel material) defined existence upstream.
void main() {
    // The committed voxel's Material (ADR 0008): the table's albedo column
    // equals the Palette color by construction, so this stays the same
    // surface color the capture path always wrote — byte-identical frames —
    // while the read itself exercises the GPU table on every hit. Alpha is
    // forced 1.0: palette alpha is not a material property (the palette
    // buffer is uploaded with alpha 1.0; the reference tracer forces it too).
    incoming_payload.color = vec4(material_table.albedo_metallic[gl_HitKindEXT].rgb, 1.0);
    incoming_payload.t = gl_RayTmaxEXT;
    // The surface identity the raygen needs for its own table read (ticket
    // 03: real metalness + emission in Voxel mode; later, the normal).
    incoming_payload.hit_kind = gl_HitKindEXT;
}
