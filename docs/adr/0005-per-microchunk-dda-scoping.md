# Per-Micro-chunk DDA scoping

The intersection shader drifted to a Region-wide (256^3) DDA march, contradicting
ADR 0001 and CONTEXT.md, which specify resolving a voxel per Micro-chunk (8^3).
We decide the DDA resolves its invoking Micro-chunk: it reads the trimmed AABB via
gl_PrimitiveID through the region->AABB device-address table (promoted to release),
slab-tests that hull, and marches the 8^3 lattice, bounded to at most 8 cells per
axis per invocation.

## Status

accepted (grilling session, 2026-08-14)

## Considered Options

- **Region-slab march (status quo)**. Correct but re-walks, cell by cell, the
  empty space the BVH already skipped (up to 256 cells/axis per invocation); the
  likely step-count driver behind the intersection shader's 93% of SM time.
- **Per-Micro-chunk march via gl_PrimitiveID + AABB table**. Chosen: bounded to
  <=8 steps, matches ADR 0001's "trimmed hulls avoid wasted invocations" and the
  Teardown precedent.
- **Full 8^3 hulls**. Already rejected in ADR 0001 (sparse chunks would
  re-introduce wasted invocations).

## Consequences

- The DDA reads its invoking AABB back through gl_PrimitiveID; this is the
  mechanism for ADR 0001's "the hit position alone yields the micro-chunk". The
  shader has no hit position, so the AABB read supplies it. It adds no new
  per-voxel metadata.
- aabb_table and the AABB buffer's STORAGE_BUFFER usage move from debug-only into
  release.
- The cross-chunk tie-break now delegates to hardware traversal order + closest-hit;
  the x,y,z tie-break remains within each chunk. Any byte-identical validator
  divergence is a stop-condition, not a reason to loosen the gate.
