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
A 256-entry RGBA8 color table from the .vox file mapping Material indices to
display colors; kept sRGB-encoded end to end (the debug paint and the byte
comparison depend on it). Materials decode it to linear reflectance for
lighting. GPU-side: a bindless vec4[256] storage buffer.
_Avoid_: Color table, LUT

**Material**:
The per-palette-index surface properties: albedo (the Palette color),
metallic, roughness, and Emission. Loaded from the .vox MATL chunk; one
Material per Palette index (256 max); GPU-side a bindless table beside the
Palette.
_Avoid_: texture, material system (the PBR triad is a Material, not a
system)

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
Object space == world space up to the translation instance transform. Carried
in the ray payload; the Normal debug Render mode paints it as a heatmap
(x red, y green, z blue).
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
the voxel is occupied. The mask, not a sentinel material, defines which
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
the Occupancy mask and committing the first occupied one. The Reference
tracer marches an independent DDA, deliberately not a mirror of the
renderer's.
_Avoid_: voxel ray march, ray walk

**t pre-pass**:
A candidate primary-visibility optimization under evaluation: a coarse
(lower-resolution) ray pass records each tile's nearest-hit t, which the
full-resolution pass then uses to skip nearer empty space. Named by its
mechanism, a lower-res t pass, not an effect.
_Avoid_: beam (classic beam tracing is secondary-ray cone tracing, out of
scope), depth pre-pass (implies a raster depth buffer this renderer lacks)

**Background**:
The radiance produced where no geometry is hit (the miss shader's output):
the Procedural sky. Rays that leave the loaded world hit nothing and report
the Background color. The ray pass's t-range equals the camera's near/far, so
Background also appears beyond the far plane. The camera's direct view of the
Background adds the Sun disk (the Sun's visual), evaluated by the raygen's
primary-miss branch.
_Avoid_: skybox, environment map (a sampled asset; the Procedural sky is
analytic)

**Void**:
The space outside the loaded world; rays there hit nothing and report the
Background color.
_Avoid_: Sky, empty space ("empty" is a property of the sparse world,
not a place)

## Light transport

**Path tracing**:
The renderer's lighting algorithm: per-pixel light transport, a primary ray
plus up to N BSDF-scattered Bounces, terminated by Russian roulette and a
depth cap. Path tracing replaces flat palette shading as the default Render
mode's output.
_Avoid_: raytracing (that is the _mechanism_; path tracing is the algorithm),
GI (an effect path tracing delivers. Describe what is seen)

**Sample**:
One path per pixel, produced once per frame (1 spp by design); samples become
an image through the Denoise pass, never accumulation.
_Avoid_: accumulation, spp ("per frame" is the point)

**Bounce**:
One BSDF-scattered secondary trace in a path; the default cap is depth 4 plus
Russian roulette.

**Emission**:
The per-Material emissive radiance; a voxel whose Material has emission > 0
is an Emissive voxel and a light source. Emissive light reaches pixels only
via path hits. There is no next-event sampling of Emissive voxels.
_Avoid_: light (say Sun, Procedural sky, or Emissive voxel, which one)

**Sun**:
The analytic directional light: a delta light at infinity, with fixed world
direction and illuminance (`E_sun`, lux on a surface perpendicular to its
rays) constants, sampled by NEE with MIS weight 1. A delta has no solid
angle, so the BSDF sampler can never produce exactly its direction.

**Procedural sky**:
The analytic environment light, a piecewise-linear radiance gradient in
μ = cos(elevation), knots at ground/horizon/zenith (all positive), evaluated
by the miss shader; the Background. Importance-sampleable by analytic CDF
inversion; no assets. The Sun disk (below) is the Sun's visual, not part of
the transport radiance. NEE and BSDF-miss samples see the gradient only.
_Avoid_: skybox, environment map

**Sun disk**:
The Sun's visual: the measure-zero radiance bump on the Procedural sky in
the Sun's direction, detected by a dot test. Seen only by the camera's
direct view of the sky (the primary-miss branch); the transport never
importance-samples it. The delta Sun light carries the light, and sampling
a bright bump with a gradient-matched pdf would firefly at 1 spp.
_Avoid_: the Sun (the disk is the look; the Sun is the light)

**NEE**:
Next-event estimation: a Bounce samples a light directly (Sun or Procedural
sky) rather than waiting for a path hit; one light is picked per Bounce
(equal probability), with a shadow ray against the world; combined with the
BSDF estimate by MIS.
_Avoid_: direct lighting (unqualified), light sampling (unqualified)

**MIS**:
Multiple importance sampling: the balance-heuristic weighting combining the
NEE and BSDF estimators for the Sun and Procedural sky so neither dominates.
The Sun's delta is NEE-only (weight 1); the sky and the BSDF split by their
direction pdfs.

**Russian roulette**:
Probabilistic path termination: a Bounce continues with probability equal to
its throughput weight, keeping the estimator unbiased.

**Lobe selection**:
The per-pixel coin flip at the primary hit that picks which of the diffuse
or specular lobes the first Bounce samples (p = 0.5, inside NRD's [1/4, 3/4]
clamp for its AREA_3X3 hit-distance reconstruction mode); the whole path's
radiance is attributed to the selected lobe's channel, the other channel gets
0 that frame, and the Denoise pass's temporal accumulation fills both.
_Avoid_: split path (per-pixel the path is single-lobe by design.
subsequent Bounces sample the full BSDF)

**Trace pass**:
The ray pass under the path-tracing output contract (ADR 0007): in Voxel
mode it writes the Beauty buffer (the noisy radiance pair) and the
auxiliary buffers instead of the swapchain; the debug Render modes still
paint the swapchain directly.
_Avoid_: g-buffer pass (a raster concept this renderer lacks)

**Denoise pass**:
The real-time denoiser (NRD ReBLUR, bound through FFI) turning the 1-Sample
radiance into a clean image from the Beauty buffer, motion vectors, and the
auxiliary buffers (normal+roughness, viewZ, motion vectors,
albedo+metalness).
_Avoid_: filter, temporal AA

**Beauty buffer**:
The trace pass's noisy radiance output, a diffuse + specular RGBA16F pair
whose alpha holds the in-lobe hit distance, written per pixel and consumed by the
Denoise pass; exposed to the swapchain after exposure and tonemap in the
Composite.
_Avoid_: render target, output image

**De-modulation**:
The removal of the surface's color response from radiance before denoising:
the trace pass writes diffuse radiance divided by albedo (eps-guarded) and
specular radiance divided by its BRDF average, so the Denoise pass filters
pure light at a uniform noise level instead of albedo-tinted light; the
Composite re-modulates (multiplies the color back) before exposure and
tonemap. Emission is albedo-proportional, so it de-modulates to a constant
and denoises as pure light.
_Avoid_: unmodulated, normalized radiance ("de-modulated" names the NRD
input contract specifically)

**Composite**:
The node that exposes the (denoised) radiance to the swapchain:
re-modulation by albedo/metallic, manual EV exposure, and the ACES tonemap.
A no-op for the debug Render modes, which paint the swapchain directly.
_Avoid_: post-processing (beyond exposure/tonemap, out of scope), final
pass

## Render mode

**Render mode**:
What the ray pass paints each pixel with: surface identity (`Voxel`, `Hull`) or
a diagnostic quantity (the `Normal` heatmap).
`Voxel` (default): the
DDA commits the surface voxel, shaded by Path tracing from the surface's
Material. `Hull`: each
Micro-chunk's trimmed AABB is the surface, colored by a coordinate hash, with
no DDA. The diagnostic modes are debug-build-only.
_Avoid_: shading mode, visualization mode

**Normal (Render mode)**:
A diagnostic Render mode (debug builds): each pixel is colored by its hit's
geometric Normal, -1..1 mapped to 0..1 per channel, voxel faces paint by
their axis (x red, y green, z blue; + side bright, - side dark), background
gray. Traces the DDA hit group like Voxel; the normal rides the payload.
_Avoid_: normal map visualization (a texture-space concept)

## Validation

**Reference tracer**:
The independent CPU renderer that validates the GPU path: a naive per-voxel
ray tracer over the world's source of truth (the flat voxel map + palette), sharing
only camera inputs and the palette with the GPU. Deliberately not a mirror of
the renderer's DDA/AABB/pool representation, so a divergence points at the
renderer rather than at shared algorithm assumptions (rendering-core ticket
06).
_Avoid_: Reference renderer, oracle
