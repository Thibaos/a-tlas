# DDA "bit-wise optimization": video research

## 0. Correction: the video shows the ray-mask occupancy filter, not the empty-space skip

The original version of this note could not identify the video and then
*guessed* the technique was the bit-scan empty-space skip (§2a) and
recommended implementing it. That guess was wrong. The user watched the video
and reports its actual content: **bit masking over the occupancy mask,
skipping the empty occupancy bits by filtering, ANDing a precomputed
ray-direction bit representation against the bit representation of occupied
voxels.** That is the *ray-mask occupancy filter / subspace cull* (§2f), not
the find-first-set empty-space skip. The empty-space skip (§2a) remains a
real sibling technique but is **not** what the video shows. §6's verdict is
corrected accordingly.

**Status:** video identity UNCONFIRMED; technique now identified from the user's first-hand account (§0).

## 1. Video identity (and confidence)

**I could not identify the video from the URL/ID.** Every web_search query
for the ID, `P2bGF6GPmfc`, `youtube.com/watch?v=P2bGF6GPmfc`,
`site:youtube.com P2bGF6GPmfc`, plus keyword variants ("DDA", "voxel",
"ray tracing", "shader"), returned either no results or a single
false-positive. The search backend in this session returns source URLs only
(no answer summaries or snippets), and the sandbox blocks all outbound TLS
(HTTPS), so I could not fetch the YouTube page, oEmbed, or the one Chinese
weekly that appears to contain the link.

The single real trace: the string `P2bGF6GPmfc` is indexed inside
`ruanyf/weekly` `docs/issue-284.md` (阮一峰's "科技爱好者周刊 #284"),
visible via Sourcegraph. That file is titled around "YouTube 有多少个视频"
(how many videos YouTube has) and is HTTPS-only with no Wayback snapshot, so
I could not extract the video's title/channel from it. **Treat the specific
video as unidentified**; the *technique* it describes is almost certainly one
of the "bit-wise DDA" family below.

**Closest technique match (high confidence):** "A guide to fast voxel ray
tracing using sparse 64-trees" by **dubiousconst282** (a.k.a.
**Teknologicus**, author of the *Vorxel* voxel engine and
`dubiousconst282/VoxelRT`). It is the most detailed modern primary source
for exactly this class of optimization and may be the video author's written
companion (unconfirmed).

## 2. What the "bit-wise optimization" actually is

"Bit-wise DDA optimization" is a family, not one trick. Per the user's
first-hand account of the video, the member it shows is **(f) the ray-mask
occupancy filter**: AND a precomputed ray-direction bit mask against the
occupancy mask and skip the empty bits. The bit-scan empty-space skip (§2a),
the technique this note originally, wrongly, bet on, is a sibling, and the
remaining members (integer tMax, mantissa/octant/tree tricks) are covered
below for completeness.

### 2a. Bit-scan / find-first-set empty-space skip (a sibling, not the video's technique)

Standard Amanatides-Woo marches one cell per loop iteration and tests
`mask & (1 << bit)` per cell. The bit-wise form holds the current run of
cells along the dominant axis as a bit field and uses a single bit-scan
(`findLSB`/first-set, or `findMSB`/count-leading-zeros) to jump straight
to the next *occupied* cell, skipping up to the whole run of empty cells in
one iteration.

GLSL has the intrinsics: `findLSB(uint)` (= first set bit from LSB),
`findMSB(uint)` (= index of MSB set, i.e. 31 − clz), and `bitCount`.
In SPIR-V/HW this compiles to `BallotFindFirstBit`-family or an integer
`clz`/bit-scan instruction (a single ALU op on most GPUs).

Concretely, for a flat 8×8×8 chunk with occupancy laid out as
`idx = x + 8*y + 64*z` (a 512-bit mask = 8 × 64-bit words, `word[z]` =
the 8×8 (x,y) plane at height z):

- Stepping along **x**: the 8 x-bits at a fixed (y,z) are a contiguous
  `0xFF << (8*y)` field inside `word[z]`. One bit-scan skips up to 8
  empty cells: `bits = word[z] & (0xFFu << (8*y));` then
  `next = findLSB(bits) /* or findMSB for -x */` gives the x of the next
  occupied voxel.
- Stepping along **y**: the 8 y-bits are strided by 8 within `word[z]`;
  extract with a byte-per-y mask and scan.
- Stepping along **z**: one bit per word (stride 64); requires gathering 8
  bits across the 8 words (or keeping the whole 512-bit mask and scanning a
  64-bit "column" mask built once per (x,y)).

This is the same primitive Laine & Karras use in their octree traversal:
maintain a bit mask of candidate children, AND it with the valid (child)
mask, and take the first set bit to descend. See §5.

### 2f. Ray-mask occupancy filter (subspace cull), the video's technique

Per the user's first-hand account of the video: hold the occupancy mask (the
bit representation of occupied voxels); for a ray, look up (or compute) a
*ray mask*, a precomputed bit representation of the cells the ray traverses,
keyed by ray direction; AND the two. The surviving bits are the occupied
voxels the ray actually crosses; the empty ones are filtered out in one
operation instead of being marched cell-by-cell.

This is the AMD "Subspace Culling for Ray–Box Intersection" family (I3D
2023): embed a binary occupancy mask in each primitive and AND it against a
ray mask, rejecting the primitive when the AND is empty, already cited in
`docs/research-voxel-rt-cost-model.md` finding 9 (−37.5% intersections,
−13.1% time on a 12.1M-triangle hair scene, a 64-bit mask over a 4³ grid).
Laine & Karras' sparse-voxel-octree traversal is the same AND + bit-scan
primitive.

Difference from the empty-space skip (§2a): the skip *marches* in t-order and
uses find-first-set to jump over empty runs inside the loop; the ray-mask
filter *filters* the whole occupancy mask by a direction-derived mask up
front, collapsing empty cells in one AND + bit-scan rather than per-cell
stepping.

### 2b. Integer / fixed-point DDA (replaces per-axis float compares)

The A-W loop's per-step cost is the `min(t_next.x, t_next.y, t_next.z)`
select chain. A variant keeps `tMax` in scaled integer form (multiply the
ray direction by a fixed scale so the dominant axis advances exactly one
integer unit per step), turning the crossing test into integer compares and
adds. This is what the request calls "(a)" / "(d) 1D DDA integer-only
traversal". It removes the float divide/reciprocal per step but does **not**
remove the 3-way min select. That is inherent to a grid walk.

### 2c. Mantissa bit-manipulation (dubiousconst282)

For power-of-two grids, the cell index can be read straight out of the IEEE
754 mantissa instead of `floor()` + float→int + mod. dubiousconst282's
key helpers (Slang):

```hlsl
int GetNodeCellIndex(float3 pos, int scaleExp) {
    uint3 cellPos = asuint(pos) >> scaleExp & 3;
    return cellPos.x + cellPos.z * 4 + cellPos.y * 16;   // 4^3 node
}
float3 FloorScale(float3 pos, int scaleExp) {           // floor(pos/scale)*scale
    uint mask = ~0u << scaleExp;
    return asfloat(asuint(pos) & mask);                 // erase low mantissa bits
}
```

The recursion divides the cube by 4, so each tree level is a 2-bit chunk of
the mantissa, addressed by shift-and-mask with **no** float→int conversion and
**no** reciprocal scaling. This is the trick that is genuinely "bit-wise".

### 2d. Ancestor backtracking via XOR + count-leading-zeros (dubiousconst282)

To re-enter the tree at the right depth after a step, XOR the old and new
mantissa and take `firstbithigh` (findMSB / clz) to get the highest changed
bit. That names the common ancestor level, replacing a descent-from-root
every step:

```hlsl
uint3 diffPos = asuint(pos) ^ asuint(cellMin);
int diffExp = firstbithigh((diffPos.x | diffPos.y | diffPos.z) & 0xFFAAAAAA);
if (diffExp > scaleExp) { scaleExp = diffExp; nodeIdx = stack[scaleExp >> 1]; ... }
```

### 2e. Empty-cuboid coalescing + octant mirroring (dubiousconst282)

- **Coalesce**: if the child population mask shows an entire aligned 2³ (or
  larger) block empty, advance the whole block at once:
  ```hlsl
  if ((node.ChildMask >> (childIdx & 0b101010) & 0x00330033) == 0) advScaleExp++;
  ```
- **Octant mirroring**: fold the ray into the negative octant by XOR-ing the
  mantissa (since for a power-of-two cell count, "flip" == XOR of the index
  bits), which bakes the per-axis sign/offset out of the inner loop.

## 3. What it replaces in a standard DDA

| Standard A-W DDA step | Bit-wise replacement |
|---|---|
| step one cell; test `mask & (1<<bit)` | bit-scan a run/column mask; jump to next set bit |
| `min(t_next.x,y,z)` float select every step | integer `tMax` accumulate (2b); select stays |
| `floor(pos)` + float→int + `mod 8` addressing | shift/mask the float mantissa (2c) |
| descend from root every iteration (trees) | XOR + clz backtrack to common ancestor (2d) |
| per-axis sign/step select in the loop | octant mirror via XOR (2e) |

## 4. Claimed benefits and tradeoffs

Measured by dubiousconst282 (integrated GPU, 4K "Bistro" scene, cycles/ray):

| Change | cycles/ray | gain |
|---|---|---|
| baseline (naive march) | ~16903 | — |
| + ancestor memoization (2d) | ~8896 | ~2.0× |
| + empty-cuboid coalescing (2e) | ~7052 | +21% |
| + octant mirroring (2e) | ~6358 | +10% |

**Tradeoffs:**

- **Precision.** Mantissa tricks (2c/2d/2e) are only exact on a fixed
  exponent range (`[1.0, 2.0)` in dubiousconst282's scheme) and break
  outside it; he keeps an explicit `3.0 - pos` fallback. NaNs/denormals and
  rays originating outside the normalized cube need guards.
- **Divergence.** Data-dependent skip lengths make per-lane iteration counts
  diverge; this is inherent to DDA but bit-scan makes the variance larger
  (long empty runs vs. dense hits). On HW RT, divergence already dominates.
- **Layout coupling.** The trick is only clean if the occupancy bit order
  matches the step axis (contiguous runs). `x + 8y + 64z` gives contiguous
  x-runs; y and z runs are strided, so y/z skips cost more ALU.
- **Helps sparse, not dense.** Empty-space skip reduces *step count*; in a
  fully-occupied chunk there is nothing to skip and the per-step overhead of
  building a run mask can make it a net loss.
- **Register pressure.** Keeping the whole 512-bit mask or an ancestor stack
  live costs registers; on an integrated GPU (their case) that eats latency
  hiding.

## 5. Sources / citations

- Amanatides & Woo (1987), "A Fast Voxel Traversal Algorithm for Ray Tracing", the baseline DDA. https://www.researchgate.net/publication/2611491_A_Fast_Voxel_Traversal_Algorithm_for_Ray_Tracing
- Laine & Karras (2010), "Efficient Sparse Voxel Octrees — Analysis, Extensions, and Implementation" (NVIDIA Tech Report NVR-2010-001), the canonical bitmask + find-first-set voxel traversal. https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees
- dubiousconst282, "A guide to fast voxel ray tracing using sparse 64-trees" (Oct 3, 2024), primary source for §2c/2d/2e and the measured numbers. https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/ (archived: https://web.archive.org/web/20260419004803/https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/ )
- dubiousconst282/VoxelRT (code): https://github.com/dubiousconst282/VoxelRT
- expenses/tree64 (independent re-implementation of the above): https://github.com/expenses/tree64
- AMD (I3D 2023), "Subspace Culling for Ray–Box Intersection", occupancy-mask AND ray-mask culling (already cited in docs/research-voxel-rt-cost-model.md). https://gpuopen.com/download/I3D2023_SubspaceCulling.pdf
- javidx9, "Super Fast Ray Casting in Tiled Worlds using DDA", the most-cited *DDA* video, but it is 2D tile raycasting, not voxel, and not the "bit trick". https://www.youtube.com/watch?v=NbSee-XM7WA

## 6. Applicability to atlas-rt's intersection-shader DDA

The target is `shaders/region/intersect.rint` (A-W DDA over the Region
lattice, per-cell Occupancy-mask test, commit via `reportIntersectionEXT`).
Three facts from the repo change the calculus vs. the video's software
raymarcher framing:

1. **The DDA already runs in small integer lattice space.** `mc = cell >> 3`,
   `c = cell & 7`, `idx = x + 8y + 64z` are already bit ops; `floor(entry)`
   happens once at setup. So the mantissa/octant-mirroring tricks (2c/2e) buy
   almost nothing here. There is no per-step float→int or sign select to
   remove (sign/step is set once, lines 132–149).
2. **The measured bottleneck is per-step ALU chains, not memory, not step
   count in dense views.** Nsight: intersection shader = 93% of SM time;
   warp states WAIT 37% / Not Selected 14% / Selected 13%. Ticket 08
   (`.scratch/renderer-impl/issues/08-dda-mask-caching.md`) already tried
   hoisting the mask/offset loads into registers and got **+11% regression**.
   Per-step ALU is the floor, and the A-W min-axis select chain
   (FADD→CMP→SEL→FADD) is "scalar-irreducible".
3. **The renderer is HW RT with a buffer_reference pool**, not a compute
   raymarcher. The DDA runs inside an intersection shader per trimmed-hull
   AABB; there is no workgroup/shared memory (dubiousconst282's groupshared
   stack is unavailable), and `reportIntersectionEXT` is the only output.

**Verdict per trick:**

- **2f ray-mask occupancy filter (the video's technique):** the candidate to
  evaluate as a *cull*: AND a per-micro-chunk ray mask against the 512-bit
  Occupancy mask and reject the candidate (return without marching) when the
  AND is empty, attacking the wasted empty-hull DDA invocation the cost-model
  research flags (finding 8/9) rather than the per-step ALU floor ticket 08
  measured. Exact-t and entry-offset precision are open questions (§0).
- **2a empty-space skip (a sibling, not the video's):** the one worth evaluating,
  but only for *sparse* views and grazing rays. It reduces step count, which
  is exactly the axis the ticket already lists as "occupancy run-skip (sparse
  views)". In the dense looking-down view (their 22–26 ms worst case) a
  mostly-occupied chunk has no runs to skip, so expect ~zero there. The
  contiguous-run property means an **x-major skip is cheapest**; y/z skips
  cost more ALU. Also: skips cannot cross a micro-chunk boundary without
  first testing the next chunk's `block_offset != SENTINEL` (the
  buffer_reference pool is chunk-indexed, not a single flat mask), so the
  inner skip is bounded to ≤8 cells.
- **2b integer tMax DDA:** marginal. Their `t_next`/min-select is the
  irreducible floor the ticket already identified; int vs float compare is
  not the differentiator on modern GPUs.
- **2c/2d/2e:** not applicable. Tree-level and mantissa-level tricks assume a
  hierarchical sparse tree and float-fractional coordinates; atlas-rt is a
  flat 8³ lattice addressed by small integers.

**Bottom line for the user (corrected):** the video's technique is the
*ray-mask occupancy filter* (§2f), not the empty-space skip. Evaluate it as a
*cull*: AND a per-micro-chunk ray mask against the 512-bit Occupancy mask
and reject the candidate (return without marching) when the AND is empty.
That lever attacks wasted empty-hull DDA invocations, which the
cost-model research already flags as the procedural-path pitfall. Its
precision (a direction-only ray mask vs. the ray's fractional entry offset
into a chunk) and the exact-t commitment it leaves behind are open questions
to settle in grilling, not to assume. The empty-space skip (§2a) and the
mantissa/octant/tree tricks (§2c–§2e) are siblings that do not match the
video and are de-prioritised until a decision says otherwise.
