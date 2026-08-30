# Transparency: the MATL glass marker and binary opacity

Straight-transmission glass needs two things settled before the transport ticket can hang off them: which palette indices transmit, and how much light a transmitting voxel passes. ADR 0008's material table carries the PBR triad and treats glass as opaque; this ADR re-opens transmission as straight pass-through — the transparency-and-overrides effort's locked floor — grounded in the MATL surface research (docs/research-dot-vox-matl-surface.md, ticket 01).

## Status

accepted (transparency-and-overrides ticket 03, 2026-08-30). Supersedes one clause of ADR 0008: glass is no longer treated as opaque. Revised 2026-08-30 (ticket 09): the per-world override file is dropped — glass data collapses to the marker alone, and the transmittance law is fixed in ADR 0013.

## Decision

- **Signal**: MATL `_type == "_glass"` — checked as `Material::material_type() == Some("_glass")` — is the only transparency marker. Absence degrades correctly: files without MATL chunks and entries without a preset both mean opaque diffuse, which is what those files mean. `_blend` and `_media` never transmit despite the editor giving them the same Transparency slider; palette alpha stays display-only (closest-hit forces 1.0, ADR 0008).
- **Binary opacity**: a glass voxel fully transmits. Light loses only tint absorption (ADR 0013's law; R = 0, no reflection); there is no stochastic pass-through and no partial opacity. MATL `_trans` is not consulted in v1 — it is the recorded plug-in point if a continuous model is ever adopted.
- **Glass data is the marker alone** (revised 2026-08-30, ticket 09): the chart-time override file — a per-world sidecar tuning metal/rough/emission and glass parameters over MATL — was dropped before any code landed. MagicaVoxel cannot author absorption strength or diffuse roughness, and every field it can author already rides the MATL chunk, so a sidecar would re-expose only defaults; the look ceiling is accepted (glass is one fixed look).
- **No glass parameters**: no tint field (the absorption color is the Palette color; albedo == Palette stands), no strength knob, no IOR in the v1 table. MATL `_ri` stays the recorded future source — read it from the raw property map, since dot_vox's `ri()` accessor reads `_ior` instead (ticket 01). The GPU table grows only the glass flag beside the PBR triad; the transmittance law is ADR 0013's.

## Considered Options

- **Palette alpha / `_alpha` as the signal**. Rejected: display-only, unbound in the modern material panel, and ADR 0008 forces alpha 1.0 at closest-hit.
- **An override-file flag as the signal**. Rejected: forks the truth about which voxels are glass across two files; the destination names MATL-marked glass.
- **Continuous opacity weighted by `_trans`**. Deferred, not rejected: at 1 spp it injects per-pixel Bernoulli noise into the Beauty channel, forfeits the trivial primary-surface-replacement degeneration the NRD guidance relies on (ticket 02), and complicates Reference-tracer parity. `_trans` stays the plug-in point.
- **Fixed reflectance instead of an IOR knob**. Overtaken (2026-08-30): transport set R = 0 (ADR 0013) so nothing consumes IOR in v1, and ticket 09 dropped it from the table; a future refraction effort re-authors it.

## Consequences

- Execution grows the GPU table with the glass flag beside the PBR triad, resolved at table build (src/core/world/material.rs), and consumes it in shaders.
- The removed 2026-08-27 Reference tracer never grew transmission (the validation stance is ticket 10's); parity stays cheap because transmission is deterministic.
- ADR 0008's other clauses stand unchanged: albedo == Palette, defaults, clamps, `_rough` honored only on `_type`-carrying entries (glass entries carry `_type`, so their `_rough` keeps flowing).
