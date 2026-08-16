# t pre-pass precedents and BVH redundancy (ticket 01)

## Executive answer

For atlas-rt's hardware-BVH primary-visibility voxel renderer, **neither shape of
the t pre-pass has a credible prior-art case in a hardware-BVH renderer, and the
redundancy finding is strong enough to short-circuit to no-go.** The t-bound
shape is a real technique — but every primary source for it (hierarchical-Z /
min-max depth pyramids, depth pre-passes, min-t mip chains) is a **raster
depth-buffer** or **software ray-marcher** device, and in all of them the coarse
hierarchy is the *only* hierarchy, doing the empty-space skip that atlas-rt's
hardware BVH already does per-ray in dedicated hardware. After the per-Micro-chunk
DDA re-scope (ADR 0005), there is no "empty-space march" left to skip: the BVH
descends to the nearest 8^3 hull in log-depth hardware box tests, and the DDA is
bounded to ≤8 cells/axis — so a t-bound can only try to shave the already-cheap
BVH node tests in front of the first hit, while adding a whole coarse pass through
the *same* intersection shader (~6% extra ray work at quarter-res per axis). The
hit-reuse shape is approximate **by construction** (a fine ray's nearest hit is
its own), its closest prior art is *temporal* reuse / *lighting* reuse (not
spatial coarse-to-fine primary hits), and in an edge-dense 0.0625 m voxel world it
mismatches at every silhouette, depth edge, and 0.5 m micro-chunk boundary —
while breaking the byte-exact validator seam (ADR 0003). **No published measured
precedent exists** for a coarse min-t pass bounding hardware-BVH primary rays (the
sources searched are listed); the closest hardware-RT traversal-reduction number
is AMD Subspace Culling's −37.5% intersections / −13.1% time, which is a 3D
occupancy-mask cull, not a screen-space t bound, and is already deferred as the
sibling candidate (renderer-impl tickets 10–11).

## The question and scope

The renderer is already coarse-to-fine **in 3D**: one procedural-AABB BLAS per
Region (256^3 voxels), one TLAS instance per Region, one trimmed AABB per
non-empty Micro-chunk (8^3), and the hardware BVH skips empty space and descends
to the nearest Micro-chunk AABB before the intersection shader runs an
Amanatides-Woo DDA bounded to the invoking chunk ([intersect.rint](shaders/region/intersect.rint),
[ADR 0005](docs/adr/0005-per-microchunk-dda-scoping.md)). The ticket asks whether
a *second*, orthogonal coarse-to-fine hierarchy **in screen space** — a coarse
lower-resolution ray pass that records nearest-hit t, then bounds (t-bound) or
seeds (hit-reuse) the full-resolution primary pass — earns its keep, or is
redundant with the hardware BVH. Output is written straight to swapchain storage
images via `imageStore`; there is no depth buffer and no raster pass, so a
"depth pre-pass" is a misnomer and the t pre-pass would write t to a new storage
image ([map.md](.scratch/t-pre-pass/map.md)). Two shapes: **t-bound**
(conservative, output byte-identical) and **hit-reuse** (approximate, wrong at
silhouettes).

## Findings

1. **t-bound prior art is raster depth-buffer or software ray-marcher — none of
   it is a hardware-BVH primary-ray technique.** The canonical source,
   Greene/Kass/Miller's "Hierarchical Z-Buffer Visibility" (SIGGRAPH 1993), is an
   *occlusion-culling* algorithm built on a raster z-buffer: a z-pyramid whose
   coarser levels store the farthest depth, used to reject polygons hidden behind
   nearer rasterized geometry. [Hierarchical Z-Buffer Visibility](https://dl.acm.org/doi/10.1145/166117.166147)
   ATI's HyperZ (2000) is the commercial hardware realization of the same
   raster-only idea — hierarchical-Z early rejection, z-compression, and
   fast-z-clear, all depth-buffer (raster) features; the same early-Z / Hi-Z
   rejection is documented in NVIDIA's own raster guidance as "the GPU can
   quickly reject fragments based on depth or stencil testing before the fragment
   shader is executed." [NVIDIA OpenGL ES Programming Tips](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/Graphics/GraphicsProgramming/OpenglEsProgrammingTips.html)
   The "min-t mip chain" analog lives in software ray marching: Tevs/Ihrke/Seidel's
   "Maximum Mipmaps for Fast, Accurate, and Scalable Dynamic Height Field
   Rendering" (I3D 2008) builds min/max mipmaps of a height field to accelerate a
   *software* relief-mapping march, and McGuire/Mara's "Efficient GPU
   Screen-Space Ray Tracing" (JCGT 2014) plus the follow-on screen-space
   acceleration structures (HPG 2015, HPG 2017) march *screen-space reflection*
   rays against a mip-mapped **raster depth buffer** — they all require a
   depth buffer the renderer does not have, and they accelerate a rasterized
   depth field, not hardware BVH traversal. [Maximum Mipmaps](https://dl.acm.org/doi/10.1145/1342250.1342279),
   [Efficient GPU Screen-Space Ray Tracing](https://jcgt.org/published/0003/04/04/),
   [An Adaptive Acceleration Structure for Screen-space Ray Tracing (HPG 2015)](https://research.nvidia.com/node/2780),
   [Hierarchical multi-layer screen-space ray tracing (HPG 2017)](https://dl.acm.org/doi/abs/10.1145/3105762.3105781)

2. **hit-reuse prior art exists, but it is coherence/temporal/lighting reuse —
   not spatial coarse-to-fine primary-hit reuse — and it is always approximate.**
   The coherence-exploitation ancestor is Arvo/Kirk's "Fast Ray Tracing by Ray
   Classification" (SIGGRAPH 1987), which groups rays by direction class to
   exploit object-space coherence in a *software* tracer. [Fast Ray Tracing by Ray Classification](https://dl.acm.org/doi/10.1145/37402.37409)
   The modern measured reuse results are about *lighting*, not primary hits:
   ReSTIR (Bitterli et al., SIGGRAPH 2020) resamples direct-lighting *light
   samples* across pixels and frames with provable bias — reuse of a reservoir,
   not of a primary-visibility hit. [Spatiotemporal reservoir resampling (NVIDIA)](https://research.nvidia.com/index.php/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct),
   [ReSTIR (ACM TOG)](https://dl.acm.org/doi/10.1145/3386569.3392481)
   Temporal *hit/sample* reuse across frames (the other coherence axis) is
   documented in NVIDIA's "Temporally Dense Ray Tracing" (HPG 2019), Frisvad et
   al.'s "Stable Sample Caching" work, and NVIDIA's "Stable ray tracing" patent on
   reprojecting ray-traced samples — all *temporal* (previous frame → this frame),
   all approximate, and all reporting reprojection-induced error as the central
   failure mode. [Temporally Dense Ray Tracing (NVIDIA)](https://research.nvidia.com/labs/rtr/publication/andersson2019temporally/),
   [Stable Sample Caching](https://orbit.dtu.dk/en/publications/stable-sample-caching-for-interactive-stereoscopic-ray-tracing/),
   [Stable ray tracing (US10388059)](https://patents.google.com/patent/US10388059)
   No source found does the ticket's *spatial* coarse-to-fine hit reuse (coarse
   pass → same-frame fine pass) for primary visibility in a hardware-BVH renderer.

3. **"Two-level traversal" in the hardware-RT literature is 3D, not screen-space.**
   The ticket's lead list mentions "two-level traversal guidance." The term's
   primary meaning in this literature is Reshetov/Soupikov/Hurley's "Multi-Level
   Ray Tracing Algorithm" (SIGGRAPH 2005) — a two-level BVH over object-space
   geometry (the conceptual ancestor of DXR/Vulkan's TLAS+BLAS), not a
   screen-space second pass. [Multi-Level Ray Tracing Algorithm](https://dl.acm.org/doi/abs/10.1145/1186822.1073329)
   Conflating "two-level" with a screen-space t pass is a category error the
   go/no-go must avoid: TLAS+BLAS is *both* levels in 3D, and atlas-rt already has
   it.

4. **The crux: the hardware BVH already provides per-ray 3D coarse-to-fine plus
   closest-hit t narrowing; a screen-space t hierarchy is orthogonal in mechanism
   but targets waste the BVH has already eliminated, and it is not found in
   hardware-RT primary renderers.** The ray carries a [TMin, TMax] interval
   (DXR `RayDesc::TMin/TMax`; GLSL `gl_RayTminEXT`/`gl_RayTmaxEXT`), and the
   acceleration-structure traversal returns the *closest* committed hit within it;
   the acceleration structure itself is an opaque, implementation-defined
   hierarchy that the implementation searches. [DXR Functional Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html),
   [Vulkan Ray Traversal chapter](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html),
   [Vulkan RayTmaxKHR](https://docs.vulkan.org/refpages/latest/refpages/source/RayTmaxKHR.html),
   [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix9/guide/optix_guide.250130.LTR.pdf)
   On the hardware side, the BVH traversal and box/intersection tests run in
   dedicated RT units, not shader cores (NVIDIA's Ampere whitepaper: second-gen RT
   cores; the Turing whitepaper introduced the RT core doing BVH traversal +
   ray-triangle intersection). [NVIDIA Ampere GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf),
   [NVIDIA Turing whitepaper](https://www.nvidia.com/en-gb/geforce/news/geforce-rtx-20-series-turing-architecture-whitepaper/)
   So the renderer's coarse-to-fine *spatial* traversal is already hardware —
   there is no shader-side empty-space march left for a t-bound to skip. A
   screen-space t hierarchy would exploit *cross-pixel* (inter-ray) coherence,
   which the per-ray BVH does **not** exploit — so it is not strictly redundant in
   *mechanism*. But its entire value is "skip near empty space," and the BVH
   already skips exactly that empty space at log-depth hardware cost; the t-bound
   can only shave the BVH's own traversal in the near range, which is the
   cheapest, hardware-accelerated part, while paying a full coarse pass through
   the same intersection shader. And the techniques themselves are raster (HZB/SSR,
   which reuse an existing full-res depth buffer) or software-DDA (SVO/mip, where
   the mip *is* the only hierarchy) — they do not transfer to a renderer whose
   hierarchy is already in hardware. The flagship hardware-BVH voxel renderer
   (Teardown's next-gen engine) corroborates the absence: it debugged wasted
   intersection-shader invocations and fixed them with tight hulls + regular
   chunking, not with a screen-space t pass. [research-teardown-hardware-ray-tracing.md, findings 4/9](docs/research-teardown-hardware-ray-tracing.md),
   [GPC 2025 recording](https://www.youtube.com/watch?v=IM1Dr98f3xU)

5. **Measured precedents: none exists for a coarse min-t pass bounding HW-RT
   primary rays.** Across the sources searched (raster HZB/HyperZ, screen-space
   RT, two-level BVH, temporal/reuse, and NVIDIA/AMD traversal guidance), **no
   paper, spec, or first-party talk reports win/loss numbers for a coarse min-t
   pass bounding hardware-BVH primary rays.** This is stated as an explicit
   negative of this search, not a proof of non-existence. The nearest *measured*
   hardware-RT traversal-cost reductions found are: (a) AMD "Subspace Culling for
   Ray–Box Intersection" (I3D 2023) — an occupancy-mask cull embedded in the AABB
   (3D, not screen-space t) that cut intersections by **37.5%** and rendering time
   by **13.1%** on a 12.1M-triangle hair scene; this is the sibling candidate
   already deferred in renderer-impl tickets 10–11. [Subspace Culling (I3D 2023)](https://gpuopen.com/download/I3D2023_SubspaceCulling.pdf)
   (b) Teardown's "ray tracing the primary view was the fastest solution — even on
   AMD," achieved **without** any t pre-pass. [GPC 2025 recording](https://www.youtube.com/watch?v=IM1Dr98f3xU)
   Raster HZB occlusion-culling and SSR numbers (the depth-pyramid domain) are
   real but measure a different problem (raster visibility/reflections) and do not
   transfer to bounding HW-RT primary-ray t. The two-pass technique that *is*
   canonical in path tracing — adaptive sampling — is image-space sample
   allocation for variance reduction, not a geometric t pre-pass. [PBRT v4: Improving Efficiency](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency)

6. **The conservative near-bound recipe: "min over samples" is not provably safe
   at silhouettes/thin geometry; the truly safe bound needs a full-res min-depth
   pyramid the renderer cannot cheaply produce.** A provably-safe near-bound for a
   tile must be ≤ the nearest surface t of *every* fine ray in the tile. The
   min-direction of the classic min/max depth pyramid is exactly that: a coarser
   level stores the **minimum** (nearest) depth over its children, so querying a
   node covering the tile yields a bound that cannot skip anything nearer.
   [Hierarchical Z-Buffer Visibility](https://dl.acm.org/doi/10.1145/166117.166147),
   [Maximum Mipmaps (I3D 2008)](https://dl.acm.org/doi/10.1145/1342250.1342279)
   But that bound is conservative **because it is built from the full-resolution
   depth of every covered pixel** (or a conservative rasterization) — and atlas-rt
   has no depth buffer, so the only cheap substitute is **min over the coarse
   samples**, which is a *sampled* min, not a *regional* min. It is safe only
   under a surface-coverage assumption (the near surface is continuous and spans
   the tile), which fails precisely at the cases that matter: a **single-voxel
   wall seen edge-on** or a **distant one-voxel pillar** can fall between every
   coarse sample center, so min-over-samples overshoots and the fine ray misses
   nearer geometry — the bound is then not conservative at all. At **grazing
   angles**, the near surface runs nearly parallel to the ray, so the depth varies
   wildly across the tile and min-over-samples is far from the per-pixel surface t
   → it skips almost nothing. **frustum-min** (the true minimum over the tile's
   frustum volume) is inherently conservative but requires testing geometry
   against each tile frustum — i.e., re-running a coarse spatial query per tile,
   which is the BVH's own job and is the very work the pre-pass was meant to
   avoid. Net: the only safe recipe is unavailable (no depth buffer), the
   available recipe is unsafe exactly where near empty space is large, and the
   frustum recipe is redundant with the BVH.

7. **hit-reuse failure modes: silhouette/depth-edge misses are intrinsic, and
   prior art treats reuse as approximate with explicit rejection — which a
   byte-exact validator forbids.** Reusing a coarse hit for a tile of fine rays is
   wrong wherever a fine ray's nearest surface is not the coarse sample's surface:
   silhouettes, depth discontinuities, thin geometry, and (for voxels) every
   micro-chunk boundary. The reuse/reprojection literature does not make this
   *correct*; it detects and rejects or accepts the error — NVIDIA's "Stable ray
   tracing" patent documents reprojection-induced sample motion/error at low
   sampling rates and mitigates with history clamping/rejection, and ReSTIR
   resampling is explicitly biased and validated on diffuse/direct lighting, not
   on primary visibility. [Stable ray tracing (US10388059)](https://patents.google.com/patent/US10388059),
   [ReSTIR (ACM TOG)](https://dl.acm.org/doi/10.1145/3386569.3392481)
   For atlas-rt the consequence is structural, not just visual: the correctness
   seam is **byte-exact** against the independent CPU reference tracer (ADR 0003),
   so an approximate shape implies reworking the validator — a cost the go/no-go
   must weigh — and the map already names the expected hole class (edge-on walls,
   distant pillars). [ADR 0003](docs/adr/0003-validation-reference-tracer.md),
   [map.md](.scratch/t-pre-pass/map.md)

8. **Voxel-specific: after ADR 0005 there is no near empty space to skip that the
   BVH is not already skipping, and hit-reuse mismatch is high at voxel scale.**
   The near empty space a t-bound could target is the distance from the ray origin
   to the first Micro-chunk hull entry. The BVH already skips it *without marching
   it* — it descends from TLAS instance (Region) to the nearest BLAS AABB in
   log-depth hardware box tests, and the DDA then marches only the 8^3 hull (≤8
   cells/axis). [intersect.rint](shaders/region/intersect.rint),
   [ADR 0005](docs/adr/0005-per-microchunk-dda-scoping.md)
   So a t-bound's marginal win is limited to shortening the (hardware,
   log-depth) BVH traversal in front of the first hit — not "skipping empty space,"
   because nothing steps through empty space anymore. Dense view (camera near the
   city): the near empty space is small, so there is little to win. Grazing /
   edge-on view: the near empty space is large, but finding 6 shows the
   conservative bound is too pessimistic (or unsafe) exactly there. On the
   hit-reuse side, a 4×4 coarse tile at typical distances spans multiple 0.0625 m
   voxels (≈4 cm at 10 m, ≈43 cm at 100 m at 1080p/60°-class FOV — a derivation
   from the repo's VOXEL_PHYSICAL_LENGTH and quarter-res tile, flagged as an
   estimate, not a measurement), and micro-chunks are 8 voxels = 0.5 m on a side,
   so tiles routinely straddle chunk boundaries, depth edges, and 90° voxel edges;
   the "flat surface spans the tile" coherence assumption behind hit-reuse fails
   constantly in an edge-dense voxel world. [CONTEXT.md (Voxel Scale, Micro-chunk)](CONTEXT.md)

## Sources

- Kept (primary):
  - Greene, Kass, Miller. "Hierarchical Z-Buffer Visibility." SIGGRAPH 1993. https://dl.acm.org/doi/10.1145/166117.166147 — canonical z-pyramid occlusion culling (raster).
  - Tevs, Ihrke, Seidel. "Maximum Mipmaps…Dynamic Height Field Rendering." I3D 2008. https://dl.acm.org/doi/10.1145/1342250.1342279 — min/max mip in software relief mapping.
  - McGuire, Mara. "Efficient GPU Screen-Space Ray Tracing." JCGT 3(4), 2014. https://jcgt.org/published/0003/04/04/ — depth-mip screen-space ray marching (raster).
  - "An Adaptive Acceleration Structure for Screen-space Ray Tracing." HPG 2015. https://research.nvidia.com/node/2780 — screen-space (raster) RT acceleration.
  - "Hierarchical multi-layer screen-space ray tracing." HPG 2017. https://dl.acm.org/doi/abs/10.1145/3105762.3105781 — screen-space (raster) RT.
  - Arvo, Kirk. "Fast Ray Tracing by Ray Classification." SIGGRAPH 1987. https://dl.acm.org/doi/10.1145/37402.37409 — coherence exploitation (software).
  - Reshetov, Soupikov, Hurley. "Multi-Level Ray Tracing Algorithm." SIGGRAPH 2005. https://dl.acm.org/doi/abs/10.1145/1186822.1073329 — two-level BVH is 3D.
  - Bitterli et al. "Spatiotemporal reservoir resampling…" SIGGRAPH 2020. https://dl.acm.org/doi/10.1145/3386569.3392481 — approximate reuse (lighting), biased.
  - Andersson et al. "Temporally Dense Ray Tracing." HPG 2019. https://research.nvidia.com/labs/rtr/publication/andersson2019temporally/ — temporal reuse.
  - Frisvad et al. "Stable Sample Caching for Interactive Stereoscopic Ray Tracing." https://orbit.dtu.dk/en/publications/stable-sample-caching-for-interactive-stereoscopic-ray-tracing/ — temporal sample reuse.
  - NVIDIA. "Stable ray tracing." US10388059. https://patents.google.com/patent/US10388059 — reprojection-induced error + clamping/rejection.
  - AMD. "Subspace Culling for Ray–Box Intersection." I3D 2023. https://gpuopen.com/download/I3D2023_SubspaceCulling.pdf — 3D occupancy-mask cull, −37.5%/−13.1%.
  - DXR Functional Spec. https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html — RayDesc TMin/TMax, opaque acceleration structure, closest-hit.
  - Vulkan Ray Traversal + RayTmaxKHR. https://docs.vulkan.org/spec/latest/chapters/raytraversal.html , https://docs.vulkan.org/refpages/latest/refpages/source/RayTmaxKHR.html — ray interval + closest-hit.
  - NVIDIA OptiX Programming Guide. https://raytracing-docs.nvidia.com/optix9/guide/optix_guide.250130.LTR.pdf — optixTrace tmin/tmax, traversal.
  - NVIDIA Ampere GA102 whitepaper. https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf — RT core BVH traversal + intersection.
  - NVIDIA Turing whitepaper. https://www.nvidia.com/en-gb/geforce/news/geforce-rtx-20-series-turing-architecture-whitepaper/ — RT core introduction.
  - NVIDIA OpenGL ES Programming Tips. https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/Graphics/GraphicsProgramming/OpenglEsProgrammingTips.html — early-Z/Hi-Z raster rejection.
  - PBRT v4 "Improving Efficiency." https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency — adaptive/stratified sampling = variance, not t pre-pass.
  - Teardown GPC 2025 recording + repo research doc. https://www.youtube.com/watch?v=IM1Dr98f3xU , docs/research-teardown-hardware-ray-tracing.md — HW-BVH voxel renderer without a t pre-pass.
  - Repo-internal: CONTEXT.md, docs/adr/0003-validation-reference-tracer.md, docs/adr/0005-per-microchunk-dda-scoping.md, shaders/region/intersect.rint, .scratch/t-pre-pass/map.md.
- Dropped (secondary): HyperZ Wikipedia article and AnandTech 2000 HyperZ reviews — press/secondary, used only to locate the feature; the technique is anchored on Greene 1993 + NVIDIA early-Z docs instead. GPUReporter and other "hybrid rendering/upscaling" write-ups — secondary. HardwareTimes/WCCFTech RDNA2-vs-Ampere comparisons — secondary.

## Gaps

- The explicit negative of finding 5 is bounded by the search terms used; it is a
  "not found in these sources" statement, not a proof no such number exists
  anywhere.
- ATI HyperZ's original 2000 whitepaper is no longer first-party-hosted; the
  feature is named here but its *technique* is anchored on the hierarchical-Z
  paper and NVIDIA early-Z docs (see Sources).
- The 4×4-tile world-size numbers in finding 8 are a derivation from the repo's
  0.0625 m voxel length and a quarter-res coarse pass at 1080p, not a measurement;
  they set the scale of the mismatch argument, and ticket 03's prototype should
  confirm the actual mismatch rate with the validator.
- The "~6% extra ray work" coarse-pass figure is from the map's own estimate
  (quarter-res per axis = 1/16 of rays) and was not independently measured here.

## Decision-ready verdict

**No-go — and the redundancy finding is strong enough to short-circuit before
either prototype's performance question matters.** Neither shape has a credible
prior-art case in a hardware-BVH renderer: the t-bound is a raster/software-DDA
technique whose coarse hierarchy is redundant with the BVH atlas-rt already has,
and whose conservative recipe is unavailable (no depth buffer), unsafe at
silhouettes/thin geometry (min-over-samples), or redundant (frustum-min); the
hit-reuse shape is approximate by construction, breaks the byte-exact validator,
and mismatches constantly in an edge-dense 0.0625 m voxel world. The measured
precedent that *does* exist and is transferable is the 3D occupancy-mask cull
(AMD Subspace Culling, −37.5%/−13.1%), which is already deferred as the sibling
candidate — that, not a screen-space t pass, is where the traversal-reduction
budget should go. Recommend: resolve ticket 01 as the redundancy verdict, close
the t-bound/hit-reuse prototypes (03/04) as no-go, and let ticket 05 rank the
ray-mask cull alone against the ticket-02 baseline.
