# Renderer input contract: snapshots, change queue, BLAS residency

The world hands the renderer **Micro-chunk snapshots**, {global coords, 64-byte Occupancy mask, u8 material indices}, and create, update, and removal are the same message: an emptied Micro-chunk re-snapshots with a zero mask, and coalescing is last-wins per Micro-chunk. The renderer owns the region lattice, derives each region id from global coords (region = 256^3 voxels, origin-aligned; v1 extent ±2048/axis → 16^3 = 4096 regions, exactly the 12-bit region-id budget), and keeps a CPU-side mirror per region as the source for wholesale pool re-packing. A content edit rebuilds only the **region's BLAS, in place**. The TLAS instance references the BLAS by device address, stable across rebuilds, so the **TLAS rebuilds only when a region transitions empty ↔ non-empty** (its instance is created or removed); instance transforms are lattice-static, the custom index is the static region id, and masks are always 0xFF (no instance-level culling. The hardware TLAS rejects per-ray). All rebuilds are **ordered taskgraph nodes** between the consuming trace and the next (pool upload → BLAS build → TLAS build on residency), which makes in-place rebuilds race-free and removes the need for double-buffered back ASes and a flip atomic; the worker keeps only the CPU-side drain/pack. Region memory (BLAS + pool buffer) is **free-listed**: a region becomes resident on its first non-empty Micro-chunk and leaves on its last, with memory returned and reused only after the rebuild sequence that dropped the referencing instance has executed; streaming is batches of snapshots.

## Status

accepted (rendering-core ticket 07, 2026-08-10). Amended (grilling
session, 2026-08-21): the consume cycle is owned by `RegionStore` —
`new` and `apply` are the only drain points of the dirty-region set,
and the app calls `apply` unconditionally each frame. The "ordered
taskgraph nodes" above is realized as host-side sequencing between
frames (graphics flight idle → compute-flight rebuild graph → next
trace), not as nodes of one taskgraph; the race-freedom argument holds
through the flight waits.

## Considered Options

- **TLAS rebuild on every content edit**. Rejected: the instance references the BLAS by device address, stable across in-place rebuilds; a per-edit TLAS rebuild is needless work.
- **Per-voxel delta messages**. Rejected: non-idempotent, demands renderer-side incremental mask maintenance, contradicts the wholesale region pool rebuild (ADR 0001).
- **World-derived region ids / pull-diff interface**. Rejected: duplicates lattice constants or couples the renderer to the world's change tracking; the renderer owns the lattice and the world stays region-agnostic.
- **Camera-driven frustum culling via instance masks**. Rejected (owner): per-frame instance changes break change-driven rebuilds; the hardware TLAS already rejects per-ray. Revises 03's "region-granularity culling".
- **Double-buffered back AS + flip atomic worker**. Rejected for rebuilds (revises 03's mechanics): in-place rebuilds of objects the front frame traces cannot be gated by wait-for-frame-advance; ordered taskgraph nodes give correctness without the double buffer or the flip.
- **Chunk-level change events / visibility flag**. Rejected: no chunk-level events; chunk-visibility semantics are the world side's concern, expressed as snapshots if at all.

## Consequences

- The world side (loading/editing/streaming, implemented later) calls `submit_microchunk` / `submit_batch`; enqueue-only, any thread, never blocks on GPU.
- Content edits cost one in-place region BLAS rebuild; TLAS rebuilds only on region residency transitions; rebuild GPU time is inline in one frame (measured by ticket 06); startup is a one-shot pre-loop build.
- The reference tracer (ticket 06 / ADR 0003) exercises the contract directly, so contract violations between world and renderer surface in validation.
- Region BLAS + pool memory churn under streaming is absorbed by free lists; reuse ordering prevents use-after-free.
- 03's "region-granularity frustum culling" and "async double-buffered worker (back AS + flip)" are revised by this ADR.
