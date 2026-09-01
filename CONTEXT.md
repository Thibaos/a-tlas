# atlas-rt

A real-time voxel renderer using hardware ray tracing (Vulkan RT pipelines,
vulkano). Renders sparse voxel worlds loaded from .vox files.

## Language

**World**:
The scene loaded from a .vox file: a sparse set of occupied voxels (a flat
map keyed by global coordinates) plus a 256-color palette. Worlds are loaded
once at startup today.
_Avoid_: Scene, level, map

**Palette**:
A 256-entry RGBA8 color table from the .vox file mapping material indices to
display colors; kept sRGB-encoded end to end. The ray pass converts a hit's
entry to linear for the display path. GPU-side: a bindless vec4[256] storage
buffer.
_Avoid_: Color table, LUT

**Material index**:
The per-voxel u8 the voxel pool carries beside the Occupancy mask: the
Palette entry the voxel paints with. There is no surface property table —
the renderer shades from the Palette alone.
_Avoid_: material id, MATL, material system

**Normal**:
The geometric surface normal at a voxel hit: the face the DDA's march
entered the committed voxel through (the entered face, never a neighbor
average). The DDA intersection shader knows it exactly as the last
Amanatides-Woo step axis, but reportIntersectionEXT carries only (t, 8-bit
hitKind) and the payload is opaque to intersection shaders, so the closest
hit reconstructs it from the hit point (the reported t is the cell-entry
boundary crossing): p[a] is an integer, within epsilon, exactly on the
crossed axis. Ties (edge/corner entries) break to the first axis in x, y, z
order, the DDA's own preference order. A camera embedded in a voxel (the
t_min commit, no crossed face) gets the camera-facing direction instead.
Object space == world space up to the translation instance transform.
Carried in the ray payload; the Normal debug Render mode paints it as a
heatmap (x red, y green, z blue).
_Avoid_: facet normal, interpolated normal (no raster interpolants exist)

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
the voxel is occupied. The mask, not a sentinel index, defines which
voxels exist (palette index 0 is a real color). Material indices hang off
it.
_Avoid_: Bitmask, presence bitmap

## Renderer input

**Snapshot**:
The unit of change the world hands the renderer: a Micro-chunk's global
coords, 64-byte Occupancy mask, and u8 material indices. Create, update,
and removal are the same message. An emptied Micro-chunk re-snapshots with
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
A Resident region whose content changed since the last rebuild, queued for
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
the Occupancy mask and committing the first occupied one.
_Avoid_: voxel ray march, ray walk

**t pre-pass**:
A candidate primary-visibility optimization under evaluation: a coarse
(lower-resolution) ray pass records each tile's nearest-hit t, which the
full-resolution pass then uses to skip nearer empty space. Named by its
mechanism, a lower-res t pass, not an effect.
_Avoid_: beam (classic beam tracing is secondary-ray cone tracing, out of
scope), depth pre-pass (implies a raster depth buffer this renderer lacks)

**Procedural sky**:
The analytic Background: a piecewise-linear radiance gradient in
μ = cos(elevation), knots at ground/horizon/zenith (all positive),
evaluated by the miss shader. No assets, no Sun disk.
_Avoid_: skybox, environment map (a sampled asset; the Procedural sky is
analytic), sky

**Background**:
The radiance produced where no geometry is hit (the miss shader's output):
the Procedural sky. Rays that leave the loaded world hit nothing and report
the Background color. The ray pass's t-range equals the camera's near/far,
so Background also appears beyond the far plane.
_Avoid_: empty space ("empty" is a property of the sparse world,
not a place)

**Void**:
The space outside the loaded world; rays there hit nothing and report the
Background color.
_Avoid_: Sky, empty space

## Display path

**Composite**:
The node that exposes the ray pass's color image to the swapchain: the ACES
curve at a fixed identity exposure, gamma, and a one-LSB display dither.
Debug Render modes paint the swapchain directly and bypass it.
_Avoid_: post-processing (beyond exposure/tonemap, out of scope), final pass,
eye adaptation (the exposure is a constant, not a meter)

## Frame lifecycle

**Frame images**:
The renderer's extent-bound image set: the ray pass's color output and the
swapchain's bindless storage views. A resize destroys and recreates them
together.
_Avoid_: render targets, trace-pass images, G-buffer

**Frame input**:
What the app reports to the renderer each frame: the player's view, a
swapchain resize, and a render-mode request. The renderer derives the
projection itself.
_Avoid_: camera update, render parameters

## Render mode

**Render mode**:
What the ray pass paints each pixel with: surface identity (`Voxel`, `Hull`) or
a diagnostic quantity (the `Normal` heatmap).
`Voxel` (default): the
DDA commits the surface voxel, painted with its Palette entry.
`Hull`: each
Micro-chunk's trimmed AABB is the surface, colored by a coordinate hash, with
no DDA. The diagnostic modes are debug-build-only.
_Avoid_: shading mode, visualization mode

**Normal (Render mode)**:
A diagnostic Render mode (debug builds): each pixel is colored by its hit's
geometric Normal, -1..1 mapped to 0..1 per channel, voxel faces paint by
their axis (x red, y green, z blue; + side bright, - side dark), background
gray. Traces the DDA hit group like Voxel; the normal rides the payload.
_Avoid_: normal map visualization (a texture-space concept)
