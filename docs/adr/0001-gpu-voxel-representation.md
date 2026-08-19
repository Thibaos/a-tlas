# GPU voxel pools and in-shader voxel resolution

The renderer represents voxels on the GPU as per-Region buffer voxel pools (BDA-addressed; per non-empty Micro-chunk a 64-byte Occupancy mask plus popcount-compacted u8 material bytes, found through a u32 offset table), and resolves a voxel from an acceleration-structure hit entirely in the intersection shader: micro-chunk AABBs are trimmed to their occupied bounds and authored in absolute region-local coordinates, so the hit position alone yields the micro-chunk and cell; the DDA commits via reportIntersectionEXT(t, material_index), the material riding the 8-bit hitKind so closest-hit stays a palette lookup and the payload stays {color, t}.

## Status

accepted (rendering-core ticket 04, 2026-08-10)

## Considered Options

- **Textures / bindless sampled images per micro-chunk**. Rejected: no filtering need for discrete palette indices; sparse micro-chunks demand indirection regardless.
- **Dense fixed 512B material slabs per micro-chunk**. Rejected: ~8x memory waste on sparse micro-chunks; the one popcount per committed hit is negligible (material sampling is not the bottleneck).
- **Material via SBT record offset or payload**. Rejected: cannot vary per primitive within a region / wastes payload bandwidth closest-hit doesn't need.
- **Full 8^3 hulls**. Rejected by owner: trimmed hulls (Teardown finding 3) avoid intersection-shader invocations for rays that miss a sparse
  micro-chunk's occupied sub-volume.

## Consequences

- Voxel edits are region-scoped wholesale pool rebuilds + region BLAS rebuilds (compacted block sizes change with popcount); no in-place patching.
- f32 precision confines to the ray origin/direction and instance transforms (region-local coordinates <= 256); ticket 05's precision question shrinks accordingly.
- The world's input contract is Micro-chunk snapshots {coords, mask, materials}; the renderer owns the region lattice (the change path is ticket 07).
