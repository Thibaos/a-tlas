# Per-pixel traversal latency diagnostics

The renderer gains two debug-only Render modes for seeing where a frame's ray
latency goes: `Ray latency` (a `clockRealtime` wall-clock delta per pixel) and
`hull-crossed` (the per-pixel spatial form of the march-and-miss counter's
`hull_crossed` word). We pair the clock with hull-crossed, not DDA iterations,
because traversal latency is set by the geometry along the ray — how many hulls
it enters — which is the hardware BVH walk's regime; the DDA march is a
separate memory-latency regime, deliberately left out of scope.

## Status

accepted (grilling session, 2026-08-16)

## Considered Options

- **Per-pixel GPU timestamps** — impossible: `QueryType::Timestamp` is
  per-pass, not per-pixel.
- **A "traversal time" heatmap from work counts (hull-crossed + DDA
  iterations)** — rejected: a count is shader-side work, not traversal; the
  hardware BVH walk (node tests, close-but-rejected AABB tests) runs before any
  shader, so no counter observes it.
- **`clockRealtime` delta paired with DDA iterations** — rejected: DDA
  iterations measures the march (occupancy/mask reads), a different latency
  regime than the hardware traversal; the pair would compare incomparable
  things.
- **`clockRealtime` delta paired with hull-crossed** — chosen: both are set by
  the traversal itself (geometry along the ray), so the pair answers "is
  latency proportional to hulls entered"; hull-crossed is the spatial form of
  the already-measured `hull_crossed` counter.

## Consequences

- `Ray latency` is wall-clock, not cost: it includes stalls, occupancy
  contention, and the slowest warp lane, and is not reproducible
  frame-to-frame — it must never be read as the glossary's GPU timestamp
  (cost), on pain of fixing CPU/present stalls with traversal work.
- `hull-crossed` is a lower bound on hardware traversal work (hulls entered,
  not rejected AABB tests) and an upper bound per-pixel (Vulkan may invoke
  intersection shaders redundantly) — the same caveats as the existing
  counter.
- The two modes are debug-build-only and app-only; the validator's capture
  raygen and the default Voxel pipeline stay byte-identical.
- `DDA iteration count` is deliberately not a mode: the march is a separate
  regime, revisitable only if its cost is ever to be studied alone.
