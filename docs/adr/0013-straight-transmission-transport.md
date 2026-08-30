# Straight-transmission transport: panes, absorption, and channel attribution

The transport half of the transparency effort: what happens to a path at a
glass hit, how tint absorption is accounted, and which Beauty channel carries
see-through radiance. Decided with the owner (ticket 04, one grilling round,
2026-08-30) on top of ADR 0012's binary glass model and ticket 05's look
preferences (per-pane factor, R = 0), grounded in the production raygen loop
(ADR 0010) and the NRD guidance (ticket 02).

## Status

accepted (transparency-and-overrides ticket 04, 2026-08-30). Consumes ADR
0012's transparency marker; leaves ADR 0010's loop structure and ADR 0011's
MIS untouched. Amended 2026-08-30 (ticket 06): the primary-miss branch fires
at depth 0 regardless of crossing count — the Sun disk shows through glass,
tinted by the pane product. Revised 2026-08-30 (ticket 09): the override
file is dropped — the law simplifies to T = the Pane's Palette color (the s
parameter is deleted) and IOR leaves the v1 table.
Amended 2026-08-30 (ticket 10): the validation stance is set — GPU-side
closed-form worlds, no CPU mirror (spec.md in the effort directory).

## Decision

- **Transmission is deterministic continuation, not a lobe.** With R = 0 a
  glass hit has one outcome: no third lobe, no pick probability, no RNG draw.
  The crossing multiplies throughput by the Pane's transmittance and continues
  in the same direction (traversal mechanism: ticket 08). Glass has no NEE and
  no direct response of its own; light reaches surfaces behind glass only via
  transmitted paths and shadow rays (ticket 06's attenuated policy).
- **A Pane is a maximal run of contiguous glass voxels with the same palette
  index along the ray.** The transmittance applies once per Pane, at entry
  (the transition non-glass → glass(i)). Thickness-blind, and symmetric under
  ray reversal when different glass indices touch (T_A·T_B either way). The
  run containing the camera origin is a Pane: its T applies when the ray
  exits.
- **The law, per channel: T = the Pane's Palette color** (linear-decoded; the
  plain tint multiply). Revised 2026-08-30 (ticket 09): the override file
  that authored s is gone, so the s parameter and its [0.25, 4] clamp are
  deleted — T = max(0, 1 + s·(tint − 1)) at s = 1 reduces to exactly this.
- **Depth and Russian roulette**: crossings do not consume MAX_BOUNCES — a
  crossing is not a scattering event, so two glass panes still leave 4 real
  bounces behind them. A hard cap of 32 crossings per path bounds pathological
  worlds; past it the hit shades as opaque (fail-dark). RR is untouched: T
  multiplies throughput and the existing max-component RR (floor 0.05)
  applies; dark tints raise kill probability, unbiased.
- **Attribution (the denoiser hook)**: a glass primary reports its whole path
  to the specular channel with weight 1 (no 1/lobe_p division); the diffuse
  channel reports no-data at glass pixels (0 radiance, nhd 0 — the
  skipped-lobe convention); the g-buffer describes the revealed surface
  behind the glass (ticket 02's primary surface replacement). Behind-glass
  transport runs the depth>0-style full-mixture sampling (50/50 pick, ×2
  weights) and NEE there evaluates the full BSDF mixture. Hit-distance
  semantics across the transmitted segment are ticket 07's.
- **Emission**: behind glass it attenuates through the throughput's pane
  product (what the prototype rendered). On a glass voxel (`_emit` on a glass
  entry) it is added at the hit un-attenuated by its own Pane, still
  attenuated by panes crossed before reaching it; at a glass primary, own
  emission joins the specular-attributed path — a recorded deviation from
  ADR 0010's primary-emission-to-diffuse rule. Sky seen through glass needs
  no rule: the ray crosses the pane (×T) then misses.
- **The Sun disk shows through glass**: the primary-miss branch fires at
  depth 0 regardless of crossing count, so the disk radiance carries the
  pane product — a tinted disk (decided in ticket 06's round). Branch
  bookkeeping across the re-traced crossings is ticket 07's.
- **IOR is dropped from v1** (revised 2026-08-30, ticket 09): with R = 0
  nothing consumes it and the override schema that carried it is gone; ADR
  0012 keeps MATL `_ri` as the recorded future source. Glass
  roughness/metallic are equally unconsumed in v1 (the g-buffer at a glass
  primary describes the revealed surface). Refraction — at most a go/no-go
  recommendation from this map's fog — would be their first consumer.
- **MIS is untouched**: glass appears in no pdf — no BSDF sampling of glass,
  no light sampling of glass; the sky-miss balance weight behind glass uses
  the scattering surface's mixture density as today (ADR 0011).

## Considered Options

- **Third lobe / folded into the specular pick.** Rejected: at R = 0 the pick
  has probability 1 — vacuous randomness spending a draw and a code path.
- **Fresnel-weighted split** (the chart-time floor). Rejected by the owner in
  ticket 05's renders: glass read as colored gelatin either way, and R = 0
  deletes the reflection machinery entirely.
- **Crossings consume the bounce budget.** Rejected: glass-heavy interiors go
  dark at MAX_BOUNCES; the prototype's per-crossing depth++ was plumbing, not
  a rendered preference.
- **Per-voxel absorption.** Rejected in ticket 05: thickness-dependent,
  editor-unpredictable.
- **Any-glass-run pane with entry-voxel parameters.** Rejected: asymmetric
  under ray reversal (A|B ≠ B|A) and one material's tint governs its
  neighbor's body.
- **Drop IOR from the schema.** Rejected while the schema existed (it churned
  ADR 0012 and the override format for a parameter the refraction go/no-go
  would want back; authored-but-idle was free). Overtaken 2026-08-30 (ticket
  09): the format is gone, the free ride ended, and IOR is dropped from v1.

## Consequences

- Execution grows the GPU material table with the glass marker per palette
  index (ADR 0012 revised); the raygen loop gains a glass-advance branch
  (throughput ×= T, re-trace, unchanged draw order, crossing counter, opaque
  fallback past 32); closest-hit needs a glass flag per hit. Ticket 08 owns
  the mechanism: same-index run-boundary detection, previous-hit material
  tracking, the interior-camera commit, and the cap.
- Ticket 07 inherits the hook: at glass primaries the aux buffers describe the
  revealed surface, the transmitted segment's hit-distance semantics are its
  to settle, and RGBA8 packing pressure reports back to the map's fog.
- Ticket 09 dropped the override file (2026-08-30): no schema, no s clamp, no
  IOR in the v1 table; the GPU table grows only the glass marker (ADR 0012
  revised).
- Validation stays cheap: transmission is deterministic; ticket 10 set the
  stance — GPU-side closed-form worlds, no CPU mirror (spec.md). Test
  worlds want a single-pane slab, mixed-index adjacency (T_A·T_B), a
  camera-inside-glass world, and a >32-crossing pathological world.
- CONTEXT.md: Transmission loses the Fresnel fraction; Pane enters the
  glossary.
