# Material table: MATL chunk → bindless GPU table + CPU mirror

Shading needs per-voxel surface properties beyond the palette color: metallic, roughness, emission. The .vox MATL chunk carries them per palette index (one Material per Palette index, ≤ 256; `dot_vox` already parses it, atlas-rt read only the palette). This ADR records the table that brings them into the pipeline: the GPU side, the CPU mirror, the defaults, and the emission mapping.

## Status

accepted (path-tracing ticket 03, 2026-08-17)

## Decision

- **Table shape**: a 256-entry bindless storage buffer beside the Palette (both world-static, uploaded once at startup). Two `vec4[256]` columns, `albedo_metallic` (albedo.rgb + metallic) and `rough_emit` (emission.rgb + roughness), indexed by the 8-bit hitKind (the material index) in closest-hit; the raygen reads it through the payload's `hit_kind` (the payload grows one uint).
- **CPU mirror**: `Material` (albedo / metallic / roughness / emission) and `get_material_table` in src/core/world/material.rs, the single source of truth. `RegionStore` uploads its packed twin.
- **Albedo == Palette color by construction**: the table's albedo column is the palette (closest-hit forces alpha 1.0; palette alpha is not a material property).
- **Defaults** (no MATL entry, or missing property): diffuse, metallic 0, roughness 0.3, emission 0. Properties clamp to [0, 1]; malformed material ids (≥ 256) are skipped, not fatal. `_type` is informational in v1. All types keep the PBR triad (`glass` is treated as opaque per the map's out-of-scope).
- **Emission mapping**: linear RGB radiance = `_emit` × albedo × 10 (`EMISSION_SCALE`, tunable. Bright enough to read through the ACES tonemap and to act as a real path-hit light later). The result is unclamped; the firefly-clamp policy stays in the effort's fog.

## Considered Options

- **Material values in the payload**. Rejected: 3+ vec4s per bounce against one `uint hit_kind`; the raygen does the same bindless lookup the path tracer needs per bounce (ticket 05).
- **Texture / material system**. Rejected (CONTEXT.md: a Material is not a system); a bindless table beside the Palette mirrors the Palette's shape.
- **Clamping emission radiance**. Deferred: the "Firefly control" fog item decides the clamp policy when ticket 05 shows the actual noise.
- **Per-type defaults** (`_type` drives the table). Rejected for v1: the MATL types beyond diffuse/emissive (glass, etc.) are out of scope; the properties are the contract.

## Consequences

- `RegionStore` owns one more static buffer (8 KB); the shared push-constant block grows one bindless id (always valid; closest-hit reads the table in every pipeline).
- The production raygen's Voxel mode writes the real MATL metalness into the albedo+metalness buffer, and the stub diffuse radiance becomes albedo + emission, the "emission as albedo-light" pattern, so the table's emission column is visibly correct before ticket 05 traces real paths (05 replaces the stub with de-modulated radiance; emission still rides the diffuse signal per ADR 0007).
- Verification: unit tests pin the mirror's mapping (defaults, clamps, `_emit` × albedo × scale) in src/core/world/material.rs.
