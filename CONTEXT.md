# a-tlas

A real-time voxel renderer using hardware ray tracing (Vulkan RT pipelines,
vulkano). Renders sparse voxel worlds loaded from .vox files.

## Language

**World**:
The scene loaded from a .vox file: a sparse set of occupied voxels (per-chunk
HashMaps) plus a 256-color palette. Worlds are loaded once at startup today.
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

**Chunk**:
The world's 64^3-voxel storage unit (current code). Pre-allocated 64x64x64
grid, sparse per-chunk voxel maps.
_Avoid_: Micro-chunk (below)

**Micro-chunk**:
The renderer's 8x8x8 render/acceleration-structure unit, tightly wrapped to
occupied voxels (owner requirement; named by rendering-core ticket 03). One
AABB per non-empty micro-chunk; 512 micro-chunks fill one Chunk. a-tlas's
"Chunk" and "Micro-chunk" are different units (storage vs render).
_Avoid_: Chunk, cell

**Region**:
The renderer's grouping of Micro-chunks that share one acceleration-structure
build: 32^3 micro-chunks (256^3 voxels, 4x4x4 Chunks). The TLAS holds one
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
a-tlas's only acceleration mechanism.
_Avoid_: Ray query (below)

**Ray query**:
The inline hardware ray-intersection mechanism (wgpu's ray queries / Vulkan
ray-query) used without a dedicated pipeline. Not used in a-tlas; reference
term only.
_Avoid_: Ray tracing pipeline

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

## Validation

**Reference tracer**:
The independent CPU renderer that validates the GPU path: a naive per-voxel
ray tracer over the world's source of truth (chunk HashMaps + palette), sharing
only camera inputs and the palette with the GPU. Deliberately not a mirror of
the renderer's DDA/AABB/pool representation, so a divergence points at the
renderer rather than at shared algorithm assumptions (rendering-core ticket
06).
_Avoid_: Reference renderer, oracle
