# atlas-rt

A real-time voxel renderer using hardware ray tracing (Vulkan RT pipelines,
vulkano). Renders sparse voxel worlds loaded from .vox files.

## Language

**World**:
The scene loaded from a .vox file: a sparse set of occupied voxels (a flat
map keyed by global coordinates) plus a 256-color palette. Worlds are loaded
once at startup today.
_Avoid_: Scene, level, map

**Voxel Scale**:
1 voxel = 1/16 meter (0.0625 m), per VOXEL_PHYSICAL_LENGTH. Physical
dimensions (player size, speeds) are expressed in meters.
_Avoid_: Block size, grid resolution
_Note_: differs from wgpu-rt (1/8 m) — do not assume a shared scale.

**Palette**:
A 256-entry RGBA8 color table from the .vox file mapping material indices to
surface colors. GPU-side: a bindless vec4[256] storage buffer.
_Avoid_: Color table, LUT

**Micro-chunk**:
The renderer's 8x8x8 render/acceleration-structure unit, tightly wrapped to
occupied voxels (owner requirement; named by rendering-core ticket 03). One
AABB per non-empty micro-chunk. That AABB is the trimmed hull (tight occupied
bounds), not the full 8x8x8 cell box.
_Avoid_: cell

**Region**:
The renderer's grouping of Micro-chunks that share one acceleration-structure
build: 32^3 micro-chunks (256^3 voxels). The TLAS holds one
instance per region; a region's structure exists only while it holds >=1
non-empty Micro-chunk.
_Avoid_: Super-chunk, block

## Voxel storage

**Voxel pool**:
The renderer's GPU-side storage for voxel data, organized per Region: for
each non-empty Micro-chunk, one Occupancy mask plus the material indices of
the occupied voxels. Built by the renderer from the world's Micro-chunk
snapshots; the world never writes it.
_Avoid_: Voxel buffer, voxel data store

**Occupancy mask**:
The 512-bit presence bitmap of a Micro-chunk: one bit per voxel, set iff
the voxel is occupied. The mask — not a sentinel material — defines which
voxels exist (palette index 0 is a real color). Material indices hang off
it.
_Avoid_: Bitmask, presence bitmap

## Renderer input

**Snapshot**:
The unit of change the world hands the renderer: a Micro-chunk's global
coords, 64-byte Occupancy mask, and u8 material indices. Create, update,
and removal are the same message — an emptied Micro-chunk re-snapshots with
a zero mask.
_Avoid_: Edit message, delta

**Change queue**:
The renderer's inbound queue of Snapshots plus its dirty-region set; the
world enqueues, the renderer drains. Coalescing is last-wins per
Micro-chunk.
_Avoid_: Event bus, message bus

**Resident region**:
A Region holding at least one non-empty Micro-chunk: it owns a BLAS, a
voxel pool, and a TLAS instance. It becomes resident on its first non-empty
Micro-chunk and leaves residency when the last one empties.
_Avoid_: Active region, loaded region

**Dirty region**:
A Resident region whose content changed since the last rebuild — queued for
a rebuild.
_Avoid_: Changed region

## Ray tracing

**Ray tracing pipeline**:
The full pipeline-based hardware ray tracing mechanism (ray generation, miss,
closest-hit, intersection shaders, shader binding table) via vulkano.
atlas-rt's only acceleration mechanism.
_Avoid_: Ray query (below)

**Ray query**:
The inline hardware ray-intersection mechanism (wgpu's ray queries / Vulkan
ray-query) used without a dedicated pipeline. Not used in atlas-rt; reference
term only.
_Avoid_: Ray tracing pipeline

**DDA**:
The renderer's voxel-resolution algorithm: a ray marches cell-by-cell through
the 8x8x8 Micro-chunk lattice (Amanatides-Woo), rejecting empty cells against
the Occupancy mask and committing the first occupied one. The Reference
tracer marches an independent DDA, deliberately not a mirror of the
renderer's.
_Avoid_: voxel ray march, ray walk

**Background**:
The color produced where no geometry is hit (the miss shader's output);
black today. Rays that leave the loaded world hit nothing and report the
Background color. The ray pass's t-range equals the camera's near/far, so
Background also appears beyond the far plane.
_Avoid_: Sky (implies atmosphere/secondary-ray semantics — out of scope)

**Void**:
The space outside the loaded world; rays there hit nothing and report the
Background color.
_Avoid_: Sky, empty space ("empty" is a property of the sparse world,
not a place)

## Render mode

**Render mode**:
What the renderer resolves a primary ray into. `Voxel` (default): the DDA
commits the surface voxel, shaded from the Palette. `Hull`: each Micro-chunk's
trimmed AABB is the surface, colored by a coordinate hash, with no DDA.
_Avoid_: shading mode, visualization mode

## Validation

**Reference tracer**:
The independent CPU renderer that validates the GPU path: a naive per-voxel
ray tracer over the world's source of truth (the flat voxel map + palette), sharing
only camera inputs and the palette with the GPU. Deliberately not a mirror of
the renderer's DDA/AABB/pool representation, so a divergence points at the
renderer rather than at shared algorithm assumptions (rendering-core ticket
06).
_Avoid_: Reference renderer, oracle

## Measurement

**GPU timestamp**:
A QueryType::Timestamp sample from the graphics queue (the compute queue's
rebuild nodes are timestamped the same way). Runs on demand: only the app
attaches a pool (`atlas-rt --measure`); the validator never measures.
_Avoid_: Timer, clock (the wall-clock is a different thing, below)

**Per-stage attribution**:
The FPS log's breakdown of the frame's GPU time into trace_rays, AS rebuild
(the ordered rebuild nodes' upload+BLAS+TLAS), and flight lines — a rebuild
spike shows up in the AS-rebuild line, never in trace_rays.
_Avoid_: Total frame time (that is flight's job)

**Gate**:
The 16 ms/frame budget; computed as the **GPU timestamp sum** (trace_rays +
AS rebuilds), with the wall-clock frame interval reported beside it. A
wall-clock over 16 ms with a small GPU sum means CPU/present-bound — a
different fix than traversal.
_Avoid_: FPS target, frame-time target

**Flight**:
The frame's whole GPU interval on the graphics queue, bracketed around the
render node's work (the app-only debug overlay draws after and is excluded).
_Avoid_: Frame interval (the wall-clock's term)
