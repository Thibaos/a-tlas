# Independent CPU reference tracer for validation

The renderer's correctness is validated against a **naive per-voxel CPU ray
tracer over the world's source of truth** (chunk HashMaps + palette) — an
independent algorithm that shares only camera inputs and the palette with the
GPU path — rather than a CPU mirror of the renderer's own representation. The
point of the reference is to be a different implementation, so a divergence
points at the renderer, never at assumptions shared by both sides.

## Status

accepted (rendering-core ticket 06, 2026-08-10)

## Considered Options

- **Naive per-voxel tracer over the world's voxels** — chosen: per-pixel
  ray-vs-voxel stepping over the 64^3 grid, no DDA, no AABBs, no pools.
  Validates the whole renderer path (micro-chunk snapshots -> region pools ->
  trimmed AABBs -> BLAS -> intersection DDA -> hitKind -> palette) against an
  implementation that shares none of it. A bug in the DDA algorithm itself
  cannot pass silently on both sides.
- **DDA-mirror over the renderer's own representation** — rejected for v1:
  same algorithm on CPU, so it isolates GPU-implementation bugs but shares
  every algorithmic assumption — a DDA bug passes on both sides. Kept as the
  escalation path: when a diff appears, a mirror of the renderer's
  representation localizes it to GPU-vs-algorithm.

## Consequences

- The reference tracer and the renderer are deliberately allowed to disagree
  in the middle; only the final per-pixel {color, t} is compared (colors
  exact — same palette — t within relative tolerance 1e-3·max(t,1)).
- The reference reads the world's side of the renderer input contract
  (ticket 07) directly, so it also catches contract violations between the
  world and the renderer.
- The harness runs offline on demand (capture before the debug overlay
  draws); the reference tracer is not budget-accounted against 16 ms.
- The DDA-mirror is not built until a diff needs localizing.
