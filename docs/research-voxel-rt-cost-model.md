# Voxel-RT cost model and precedents (ticket 02)

## Executive answer

For the hybrid design (hardware acceleration structure over tightly-wrapped 8×8×8
chunk AABBs + in-shader voxel resolution), the precedent-set (Teardown's GPC 2025
talk, NVIDIA/AMD/Epic guidance, DXR/Vulkan specs) converges on one granularity
answer: **chunk AABBs belong in BLASes as procedural primitives; the TLAS should
contain one instance per region/volume, never one instance per chunk**. Teardown
inserts all chunk AABBs of a voxel volume into a single per-volume BLAS and lets
the TLAS see one instance per volume, with regular grid subdivision of the world
beating designer-split irregular shapes [Teardown research doc, findings 3–4]. At
world scale, per-chunk *instances* break down quickly: a 2048³ world in 8³ chunks
has 256³ = 2²⁴ chunk slots, which *equals* the DXR hard cap of 2²⁴ instances per
TLAS at 100% occupancy and is ~16× over Epic's "≤100k instances" 30 fps guidance
at typical 10% occupancy [DXR spec; UE docs]. Per-chunk *primitives* (up to 2²⁹
per BLAS in DXR) are fine at any plausible world size. TLAS rebuild every frame on
GPU is the norm (cost is linear in instance count; NVIDIA says build, don't refit;
keep total build/update ≤ ~2 ms, overlapped on async compute); BLAS rebuilds are
event-driven (stream-in/out, voxel edits) and must be spread over frames.
Count-changing edits *cannot* be refit, only rebuilt, per DXR update constraints. When a ray crosses empty space inside a tight hull, the intersection
shader (pipeline) or ray-query loop simply declines to report a hit and traversal
continues; the cost is the wasted invocation + DDA, which is why hulls must be
tight (Teardown finding 3) and why occupancy-mask culling exists as a proven
mitigation (−37.5% intersections in the AMD I3D 2023 paper). Opacity micromaps:
**confirmed absent in wgpu** (tracking issue #6762 marks them unimplemented), but
present in DXR 1.1+/VK_EXT|KHR_opacity_micromap, Ada-native hardware
acceleration with software emulation on pre-Ada (incl. RTX 30), and irrelevant to
this design anyway because OMM is sub-triangle opacity for triangle geometry only,
and a BLAS cannot mix triangles and AABBs.

## The question and scope

Hybrid design: hardware BVH over tightly-fitted 8×8×8-chunk AABBs, with voxel
resolution in-shader (DDA) when a hull is entered. The ticket asks for:
granularity cost behavior + instance math at world scale; DXR/Vulkan instance
limits and RTX-30 behavior; TLAS rebuild vs refit cost/cadence for chunk
streaming; AABB empty-hull semantics and overhead; precedents (Teardown findings
3/4/5/6 + other shipped engines/papers/open-source, incl. WebGPU/wgpu demos);
opacity-micromap status in DXR/Vulkan and wgpu (expected: none).

Working note: this research supersedes the "no portable WebGPU/wgpu ray-tracing
API today" caveat in `docs/research-teardown-hardware-ray-tracing.md`. wgpu has
had *experimental* ray queries + BLAS/TLAS since v27 and they are present in
wgpu 30 (the version pinned in `Cargo.toml`), native Vulkan/DX12 only. Details in
findings 9 and 12.

## Findings

1. **DXR instance limit: 2²⁴ (16,777,216) per TLAS; primitive limit: 2²⁹ per BLAS.**
   The DXR functional spec caps instances per top-level acceleration structure at
   2²⁴ and primitives per bottom-level structure (summed across geometries) at
   2²⁹, geometries per BLAS at 2²⁴; the cap is exposed as
   `D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE =
   16777216`. [DXR Functional Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html),
   [windows-docs-rs constant](https://microsoft.github.io/windows-docs-rs/doc/windows/Win32/Graphics/Direct3D12/constant.D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE.html)

2. **Vulkan exposes the same limits as queryable properties, not constants.**
   `VkPhysicalDeviceAccelerationStructurePropertiesKHR` carries
   `maxGeometryCount`, `maxInstanceCount`, `maxPrimitiveCount`, and builds are
   validated against them (VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03801:
   TLAS instance count ≤ `maxInstanceCount`); `maxPrimitiveCount` applies to
   triangles *and* AABBs. There is no fixed public value in the spec. Actual
   device numbers must be queried at runtime (uncertainty: NVIDIA/AMD values not
   pinned in this research). [Vulkan refpage for the properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceAccelerationStructurePropertiesKHR.html),
   [vkCmdBuildAccelerationStructuresKHR VUIDs](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBuildAccelerationStructuresKHR.html)

3. **Instance math at world scale (8³ chunks).** Chunk-slot counts: 512³ world →
   64³ = 262,144 slots; 2048³ world → 256³ = 16,777,216 slots = 2²⁴ (exactly the
   DXR TLAS instance cap). At terrain-like occupancy (10–30%), non-empty chunks ≈
   26k–79k (512³) and ≈ 1.7M–5.0M (2048³).
   - **Chunks as TLAS instances**: 512³ at 25% ≈ 66k instances, feasible but at
     Epic's 30 fps budget edge; 2048³ at 10% ≈ 1.7M, ~16× over that budget, and
     100% occupancy hits the 2²⁴ hard cap. TLAS build cost scales linearly with
     instance count, so this is the wrong axis to spend it on. [UE Ray Tracing Performance Guide](https://dev.epicgames.com/documentation/unreal-engine/ray-tracing-performance-guide-in-unreal-engine?lang=en-US),
     [DXR spec limits](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)
   - **Chunks as BLAS primitives, regions as instances (Teardown's model)**: a
     region of 16³–32³ chunks (128³–256³ voxels) holds ≤ 32,768 AABB primitives
     per BLAS, far below the 2²⁹ primitive cap, and yields 512–4,096 region
     instances at full 512³ occupancy, or 4,096–32,768 at full 2048³ occupancy;
     only loaded/resident regions enter the TLAS, so in-practice instance counts
     are low thousands. Teardown puts *one BLAS per voxel volume* and instances
     only the volumes, with chunk AABBs as primitives inside; their terrain is
     subdivided on a regular grid because that produces a better TLAS than
     designer-split irregular shapes. [Teardown research doc, findings 3–4](research-teardown-hardware-ray-tracing.md),
     [GPC 2025 recording](https://www.youtube.com/watch?v=IM1Dr98f3xU)
   - Memory estimate (flagged as estimate): a DXR/Vulkan instance descriptor is
     64 bytes, so 66k instances ≈ 4 MB and 1.7M ≈ 107 MB of staging per rebuild;
     chunk AABB input data is 24 bytes/AABB (two vec3), so a 32k-chunk region ≈
     0.8 MB. [D3D12_RAYTRACING_INSTANCE_DESC](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_raytracing_instance_desc)

4. **BLAS granularity: one BLAS can hold many geometries cheaply; the cost driver
   is TLAS instance overlap, not primitive count.** NVIDIA: "Tracing performance
   against a BLAS doesn't depend on the number of geometries in it"; split a BLAS
   when its *instance* world-space AABB contains a lot of empty space (unnecessary
   traversal), but merge BLASes when their instance AABBs overlap significantly
   (every ray through the overlap region must process all of them). AMD's RRA
   guide says the same for terrain: one giant terrain BLAS overlaps everything and
   hurts traversal; splitting into chunks trades "longer TLAS build time" for
   much better traversal, and warns that splitting BLASes further (e.g., per
   material or per orientation) often *hurts* because of instance overlap and TLAS
   build cost. Practical guidance for voxels: chunk AABBs inside per-region BLASes
   gives tight instance hulls with no inter-instance overlap, which is exactly the
   NVIDIA "split on empty space" + AMD "split terrain into chunks" sweet spot.
   [NVIDIA best practices](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/),
   [AMD RRA guide](https://gpuopen.com/learn/improving-rt-perf-with-rra/)

5. **TLAS cadence: rebuild every frame, on GPU; don't refit.** Epic: "The Top
   Level Acceleration Structure is rebuilt every frame", with Rendering-Thread,
   RHI-Thread and GPU cost proportional to instance count, and "scenes should
   generally have 100,000 instances or fewer" after culling for a 30 fps console
   target. NVIDIA: for TLAS "perform only rebuilds" with `PREFER_FAST_TRACE`
   ("it's just easier to manage in most circumstances"), keep total BLAS/TLAS
   build+update ≤ ~2 ms via pruning/selective updates, and overlap builds with
   G-Buffer/shadow passes on async compute; generate build command lists on worker
   threads. "Games typically rebuild the TLAS every frame to handle dynamic
   objects" (NVIDIA stochastic-LOD article). [UE RT performance guide](https://dev.epicgames.com/documentation/unreal-engine/ray-tracing-performance-guide-in-unreal-engine?lang=en-US),
   [NVIDIA RTX best practices](https://developer.nvidia.com/blog/rtx-best-practices/),
   [NVIDIA best practices (updated)](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/),
   [NVIDIA stochastic LOD](https://developer.nvidia.com/blog/implementing-stochastic-lod-with-microsoft-dxr/)

6. **Rebuild vs refit (update): order-of-magnitude cheaper, but constraints bite.**
   Refit is ~10× cheaper than build at the BLAS level (Keller's SIGGRAPH numbers:
   refit ≈ 1000 Mtris/s vs build ≈ 100 Mtris/s; a BFV case went 64 ms → 1.15 ms
   after build optimization). But DXR update mode allows changing *only* vertex
   positions (triangles) or the AABB values (procedurals), not primitive counts,
   geometry flags, or formats. BLAS updates don't allow inactive primitives
   to re-activate. For voxel chunks, adding/removing chunks (streaming, edits) is
   a topology change → **rebuild, not refit**. Refit quality degrades under large
   deformations: NVIDIA rebuilds after large deformations and recommends
   distributing rebuilds over frames to avoid stutter; AMD RRA demonstrates a door
   whose update-mode BVH costs ~1100 traversal steps vs rebuild, and recommends
   "rebuild every n frames and update in between"; Epic similarly forces periodic
   rebuilds (`r.RayTracing.DynamicGeometry.ForceBuild.MaxPrimitivesPerFrame`) and
   spreads dynamic builds round-robin with a ~10k–30k primitives/frame budget.
   [DXR update constraints](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html),
   [Keller SIGGRAPH 2019 slides](https://cg.informatik.uni-freiburg.de/intern/seminar/raytracing%20-%20Keller%20-%20SIGGRAPH%202019%202%20Acceleration%20Data%20Structures.pdf),
   [NVIDIA best practices (updated)](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/),
   [AMD RRA guide](https://gpuopen.com/learn/improving-rt-perf-with-rra/),
   [UE RT performance guide](https://dev.epicgames.com/documentation/unreal-engine/ray-tracing-performance-guide-in-unreal-engine?lang=en-US)

7. **Streaming-specific guidance.** Only loaded/resident geometry should enter the
   TLAS: NVIDIA culls instances (frustum + distance + size) and updates only
   visible BLASes; Epic's ray-tracing culling defaults to a 100 m radius, and its
   Reference-Based-Residency system evicts BLASes not referenced by the TLAS under
   a memory budget (a direct analog for a chunk-BLAS pool). NVIDIA's
   `vk_lod_clusters` sample documents host-driven BLAS caching/streaming with BLAS
   reuse across frames. DXR 1.2's Partitioned TLAS (PTLAS) is the forward-looking
   fix for "the entire TLAS must be rebuilt even if only a small number of
   instances are modified": persistent, partially rebuilt, with partitions of
   ~100–1000 instances offering most of the build speedup with least trace
   penalty; it is DX-only for now and not in wgpu. [UE RT performance guide](https://dev.epicgames.com/documentation/unreal-engine/ray-tracing-performance-guide-in-unreal-engine?lang=en-US),
   [NVIDIA best practices (updated)](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/),
   [nvpro blas_caching doc](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/blas_caching.md),
   [DXR spec part 2 (PTLAS)](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing2.html),
   [wgpu tracking issue #6762](https://github.com/gfx-rs/wgpu/issues/6762)

8. **AABB empty-hull semantics: candidates, not hits, and the hull may be
   enlarged by the implementation.** In DXR, an AABB primitive only generates an
   intersection *candidate*; the application intersection shader decides whether
   an actual hit exists and reports it (or not. Traversal then continues). The
   spec is explicit that implementations "may replace the AABBs provided... with
   more or fewer AABBs (or other representation)" and "may have chosen some larger
   volume than the input AABB for which to invoke intersection shaders", and that
   "intersection shaders may be executed redundantly" for the same primitive/ray.
   Vulkan's ray-traversal spec says the same: "In the case of AABB geometries,
   implementations may increase their size in an acceleration structure in order
   to mitigate precision issues. This may result in false positive intersections
   being reported to the application", and a ray originating inside an AABB
   counts as hitting it (camera-inside-terrain always invokes the shader). In ray
   queries (the wgpu-viable path), an AABB candidate makes `Proceed()` return
   true so the shader can manually resolve and then `Confirm`/`Generate` or just
   continue; in wgpu's naga binding, an AABB candidate carries no `t` ("an AABB
   which has a volume"). Net: the "empty space inside the hull" case costs one
   shader invocation + the DDA, per candidate, per ray. Hardware cannot skip it.
   [DXR spec (procedural geometry, AABB volume)](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html),
   [Vulkan ray traversal spec](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html),
   [wgpu ray_tracing.md](https://github.com/gfx-rs/wgpu/blob/trunk/docs/api-specs/ray_tracing.md)

9. **Overhead precedents for in-shader resolution.** Teardown: because chunk
   volumes are only 8×8×8, the intersection shader's DDA needs "only a few steps"
   before hitting or exiting, no mip hierarchy needed, and their compression
   experiments showed material sampling is *not* the bottleneck (block compression
   bought ~1% throughput; traversal is). Their instrumentation (wave-wide clock
   sampling around expensive work, mapped to a heatmap) exposed occluded
   intersection-shader invocations and TLAS problems; a ray crossing a sparse
   hull that resolves to "no voxel" is pure wasted invocation cost, minimized by
   tight hulls (flagpoles!), regular chunking, and small chunks. AMD's I3D 2023
   paper "Subspace Culling for Ray–Box Intersection" is the published mitigation:
   embed a binary occupancy mask in each AABB and AND it against a ray mask. On a
   12.1M-triangle hair scene it cut intersections by 37.5% and rendering time by
   13.1%, at 64 bits per AABB (4³ grid) with LUT compression. NVIDIA's guidance to
   "use triangle geometries when possible" (hardware excels at ray-triangle
   tests; ray-box is accelerated but less) and DXR's "intersection shaders... are
   less efficient but offer far more flexibility" bound the expected overhead of
   the procedural path. [Teardown research doc, findings 3–6](research-teardown-hardware-ray-tracing.md),
   [GPC 2025 recording](https://www.youtube.com/watch?v=IM1Dr98f3xU),
   [Subspace Culling paper](https://gpuopen.com/download/I3D2023_SubspaceCulling.pdf),
   [NVIDIA best practices (updated)](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/),
   [DXR spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)

10. **Precedents, shipped and open-source.**
    - **Teardown next-gen engine (shipped prototype)**, the flagship: 8×8×8
      procedural chunk AABBs in per-volume BLASes, custom intersection shaders,
      no triangles; hardware RT of the primary view was the fastest option "even
      on AMD" vs GPU-driven re-meshing and direct chunk re-marching; full path
      tracing ~10 ms on high-end GPUs; per-volume BLAS means TLAS instances =
      objects, not chunks. [Teardown research doc](research-teardown-hardware-ray-tracing.md),
      [talk](https://www.youtube.com/watch?v=IM1Dr98f3xU)
    - **Minecraft RTX (shipped, NVIDIA + Microsoft)**, cadence precedent: BLAS
      builds per frame for animated objects only, TLAS built per frame, then
      multiple ray dispatches (primary+shadows, indirect diffuse/specular) with
      A-SVGF denoising + DLSS 2.0. The world's exact primitive representation
      (triangle block models vs AABBs) is **not confirmed** in the sources found;
      flagged as a gap. [Frame analysis (beta)](https://medium.com/swlh/frame-analysis-minecraft-rtx-beta-5602081cd90c),
      [NVIDIA GTC blog](https://developer.nvidia.com/blog/gtc-digital-crafting-a-real-time-path-tracer-for-minecraft-rtx/)
    - **Unreal Engine 5 (shipped engine)**, instance budget (≤100k), per-frame
      TLAS rebuild, dynamic-BLAS budgets and round-robin, BLAS residency
      streaming, update-quality/rebuild policies. [UE RT performance guide](https://dev.epicgames.com/documentation/unreal-engine/ray-tracing-performance-guide-in-unreal-engine?lang=en-US)
    - **UntitledDXRVoxels (open source, DXR)**, a 100%-raytraced voxel game with
      a 64³ chunk and runtime block creation/destruction; notably it ray-traces
      *triangles* (per-face cube quads, `TriangleHitGroup`, vertex/index buffers),
      i.e., the naive contrast to the AABB-procedural path, and uses
      `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE` for
      shadows. [README](https://github.com/RMichelsen/UntitledDXRVoxels),
      [MainRays.hlsl](https://github.com/RMichelsen/UntitledDXRVoxels/blob/master/Untitled/Resources/Shaders/MainRays.hlsl)
    - **Microsoft D3D12RaytracingProceduralGeometry sample (official)**,
      multiple intersection shaders (SDF/volumetric/fractal) over procedural
      AABBs, extended SBT layouts across geometries/BLASes; the reference for
      procedural-hit pipelines. [sample readme](https://github.com/microsoft/directx-graphics-samples/blob/master/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingProceduralGeometry/readme.md)
    - **SDF_D3D12 (open source)**, procedural AABBs + volume data + sphere-tracing
      in an intersection shader, with volumes compacted so "no empty space is
      wasted", an explicit empty-hull mitigation. [repo](https://github.com/jambuttenshaw/SDF_D3D12)
    - **AMD Subspace Culling (I3D 2023)**, see finding 9. [paper](https://gpuopen.com/download/I3D2023_SubspaceCulling.pdf)
    - **WebGPU/wgpu demos**: public WebGPU voxel ray tracers are compute-based
      (no acceleration structures exist in the WebGPU standard): e.g.,
      AddisonPrairie/WebGPU-.vox (MagicaVoxel path tracer in compute),
      TwentyFiveSoftware/webgpu-ray-tracing ("Ray Tracing in One Weekend" in a
      WebGPU compute shader), benjaminaster.com WebGPU ray tracer. The only
      AS-based path is wgpu's *experimental native-only* ray query (finding 12);
      no WebGPU/wgpu AABB-procedural voxel-RT precedent with BLAS/TLAS was found.
      [WebGPU-.vox](https://github.com/AddisonPrairie/WebGPU-.vox),
      [webgpu-ray-tracing](https://github.com/TwentyFiveSoftware/webgpu-ray-tracing),
      [WebGPU Ray Tracer](https://benjaminaster.com/ray-tracer/)

11. **Opacity micromaps: "none" is confirmed for wgpu only; DXR/Vulkan have them,
    and they don't apply to AABB voxels anyway.** DXR 1.1+ defines OMM arrays
    (built like ASes, referenced per-triangle) and D3D12 exposes
    `D3D12_RAYTRACING_GEOMETRY_OMM_TRIANGLES_DESC`; Vulkan has
    `VK_EXT_opacity_micromap`, promoted to `VK_KHR_opacity_micromap` (the KHR
    version has significant API changes). Hardware acceleration is Ada-native:
    NVIDIA's OMM SDK states "OMMs are supported on any RTX GPU. Ada Lovelace GPUs
    have native hardware that accelerate the performance", with software
    emulation for pre-Ada (i.e., RTX 30) per the OMM integration guide; Microsoft
    announced OMM with "out of the gate all NVIDIA raytracing capable hardware
    supports OMMs (currently developer preview drivers)". No AMD OMM support was
    found in sources (uncertainty flag). **wgpu: none**. The experimental RT
    scope (ray queries, BLAS/TLAS) has no OMM, and tracking issue #6762 lists
    "Micromaps (opacity and displacement)" as unchecked. OMM is sub-triangle opacity for *triangle* geometry, and DXR forbids mixing geometry
    types in one BLAS, so a voxel AABB BLAS can't use OMM regardless. The
    correct knob is `FORCE_OPAQUE`/opaque flags and accept-first-hit for shadow
    rays. [NVIDIA OMM SDK](https://developer.nvidia.com/rtx/ray-tracing/opacity-micro-map/get-started),
    [OMM integration guide](https://github.com/NVIDIA-RTX/OMM/blob/main/docs/integration_guide.md),
    [Microsoft OMM blog](https://devblogs.microsoft.com/directx/omm/),
    [VK_KHR_opacity_micromap proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_opacity_micromap.html),
    [DXR spec OMM](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html),
    [wgpu tracking issue #6762](https://github.com/gfx-rs/wgpu/issues/6762)

12. **RTX-30-series behavior and what wgpu 30 actually offers.** Ampere's
    second-generation RT cores give up to 2× the ray throughput of Turing and can
    run RT concurrently with shading/denoising (Ampere whitepaper); a concrete
    cross-API benchmark (81 instances, 490k triangles, primary+shadow+reflection
    rays at 1600×900) shows GeForce 3080 ≈ 0.54–0.60 ms vs 2080 Ti ≈ 0.95–1.06 ms
    (≈1.8–2×), with BLAS build ~7 ms (18 ms unoptimized; 33 MB → 15 MB compacted)
    and compute-shader RT 7.7× slower than hardware RT; the same page calls AMD's
    AS size/build times (76 MB, 15 ms, >500 ms first build) the main AMD weakness,
    "especially for dynamic TLAS with thousands of instances". Driver-side BLAS
    storage spreads ~3× across vendors (NVIDIA ≈ 19–27 B/triangle, AMD ≈ 48–57,
    radv ≈ 137; measured on triangles. AABB-BLAS numbers not found, flagged).
    Note Ampere has no AS-build hardware unit (builds run on shader cores; Ada's
    DMM "10× faster BVH build" claim applies to displaced-micromesh geometry only).
    [Ampere GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf),
    [Tellusim RT comparison](https://tellusim.com/rt-perf/),
    [zeux.io measuring AS](https://zeux.io/2025/03/31/measuring-acceleration-structures/),
    [Ada Lovelace architecture](https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/)
    - **wgpu status (v27+; pinned wgpu 30 in this repo)**: experimental ray
      queries + BLAS/TLAS under `Features::EXPERIMENTAL_RAY_QUERY` (the old
      separate acceleration-structure feature was merged into it in v27), GPU-side
      builds via `CommandEncoder::build_acceleration_structures`, async BLAS
      compaction (`prepare_compaction_async`/`compact_blas`, compacted BLAS cannot
      be rebuilt), and, explicitly, **"Updating (performing a partial rebuild)
      is currently unsupported"**, so wgpu is rebuild-only. naga exposes
      `rayQuery*` in WGSL (`enable wgpu_ray_query`), AABB candidates as
      `RAY_QUERY_INTERSECTION_AABB` without `t`, `SKIP_AABBS`/`SKIP_TRIANGLES`
      flags, and 24-bit `TlasInstance` custom data; ray-tracing *pipelines* are in
      development (hal RT pipelines landed in wgpu 30; naga-side still maturing).
      It is native-only (Vulkan/DX12). Browser WebGPU has no RT, and explicitly
      "may have major bugs... subject to breaking changes". A real wgpu-specific
      pitfall at scale: TLAS build validation/residency was O(instance count) per
      queue submit (a landscape scene stalled ~50 ms/frame until deduplicated by
      unique BLAS, PR #9835). [wgpu ray_tracing.md](https://github.com/gfx-rs/wgpu/blob/trunk/docs/api-specs/ray_tracing.md),
      [wgpu v27 changelog](https://github.com/gfx-rs/wgpu/blob/v27.0.1/CHANGELOG.md),
      [wgpu v30.0.0 release](https://github.com/gfx-rs/wgpu/releases/tag/v30.0.0),
      [wgpu PR #9835](https://github.com/gfx-rs/wgpu/pull/9835)

13. **Small-BLAS memory and command-list pitfalls.** One D3D resource per small
    BLAS wastes memory (64 KiB alignment) and thrashes the TLB; NVIDIA recommends
    pooling suballocations at 256-byte alignment. Many small builds also have
    real CPU-side command-build overhead: a developer benchmark of ~2000 separate
    BLAS builds found the cost in command-list build time, not GPU time. Compaction
    saves up to ~50% on static structures. This argues for *fewer, pooled,
    compacted* chunk BLASes rather than one BLAS per tiny chunk. [NVIDIA memory blog](https://developer.nvidia.com/blog/managing-memory-for-acceleration-structures-in-dxr/),
    [NVIDIA forum thread](https://forums.developer.nvidia.com/t/solved-multi-infocount-of-vkcmdbuildaccelerationstructureskhr-gpu-time-in-nsight-shows-short-time-but-long-fence-wait/346407),
    [NVIDIA best practices (updated)](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/)

## Decision-ready summary

- **Granularity**: 8×8×8 chunk AABBs as procedural primitives inside per-region
  BLASes (16³–32³ chunks per region); TLAS instances = regions only (plus
  dynamic objects). Never one TLAS instance per chunk at 2048³ scale.
- **Instance math**: 512³ world → 262k chunk slots → ~26k–79k non-empty chunks →
  512–4,096 region instances at full occupancy; 2048³ world → 2²⁴ chunk slots
  (at the DXR instance cap) → ~1.7M–5.0M non-empty chunks → 4,096–32,768 regions
  full-world, low thousands once only resident regions are instanced. All far
  under the 2²⁴ instance cap and Epic's ≤100k guidance; primitive counts per BLAS
  (≤32k) are ~4 orders of magnitude under the 2²⁹ cap.
- **Cadence**: TLAS rebuild every frame on GPU (`PREFER_FAST_TRACE`), cost linear
  in instances, hidden on async compute, CPU command-gen on worker threads; BLAS
  rebuild only on stream-in/out/edits, spread across frames (topology changes
  forbid refit; DXR update allows AABB *values* only); compact static chunk
  BLASes, pool the allocations; keep total build/update ≈ ≤2 ms.
- **Pitfalls**: (1) empty-hull invocations are unavoidable per candidate.
  Tighten hulls, prefer occupancy masks over big hulls, keep chunks small;
  (2) ray queries return AABB candidates without `t`. Resolve manually;
  (3) intersection shaders may run redundantly and may be invoked for
  implementation-enlarged or degenerate AABBs; (4) BLAS cannot mix triangles and
  AABBs; (5) instance-overlap/empty instance AABBs are the top TLAS traversal
  cost. Chunked regular subdivision fixes both; (6) wgpu has no update/refit and
  no OMM, is native-only and experimental; submit validation was O(instances);
  (7) don't build thousands of tiny BLASes (TLB/alignment/command-build
  overhead); (8) OMM is triangle-only and Ada-native anyway. Irrelevant here.

## Relevance to wgpu-rt

- The design is implementable in wgpu 30 today via experimental ray queries +
  BLAS/TLAS (native Vulkan/DX12), but only in rebuild-only mode; the in-shader
  DDA over chunk AABBs maps exactly onto wgpu's AABB-candidate ray-query loop.
- wgpu's `TlasInstance` 24-bit custom data can carry chunk/region IDs; per-region
  BLAS rebuilds on chunk edits + per-frame TLAS rebuild is the natural streaming
  cadence; async compaction fits the "frozen chunk BLAS" lifecycle.
- The WebGPU standard still has no RT. Anything built on this is native-only and
  behind an experimental feature.

## Sources

- Kept:
  - DXR Functional Spec (Microsoft, primary): https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html, limits, AABB semantics, update constraints, OMM, intersection-shader redundancy.
  - DXR Functional Spec part 2 (PTLAS/clustered geometry): https://microsoft.github.io/DirectX-Specs/d3d/Raytracing2.html, streaming/partial-rebuild future work.
  - Vulkan ray-traversal spec: https://docs.vulkan.org/spec/latest/chapters/raytraversal.html, AABB candidate/confirmation semantics, AABB enlargement → false positives, ray flags.
  - Vulkan refpages (acceleration-structure properties/VUIDs): https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceAccelerationStructurePropertiesKHR.html, maxInstanceCount/maxPrimitiveCount.
  - NVIDIA "Best Practices for Using NVIDIA RTX Ray Tracing (Updated)": https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/, granularity, empty-space splitting, rebuild-vs-update, compaction, flags.
  - NVIDIA "Tips and Tricks: Ray Tracing Best Practices": https://developer.nvidia.com/blog/rtx-best-practices/, 2 ms budget, async compute, TLAS build-not-update.
  - NVIDIA "Managing Memory for Acceleration Structures in DXR": https://developer.nvidia.com/blog/managing-memory-for-acceleration-structures-in-dxr/, pooling/TLB.
  - NVIDIA OMM SDK + integration guide: https://developer.nvidia.com/rtx/ray-tracing/opacity-micro-map/get-started, https://github.com/NVIDIA-RTX/OMM/blob/main/docs/integration_guide.md, Ada-native, pre-Ada emulation.
  - Microsoft D3D12 OMM blog: https://devblogs.microsoft.com/directx/omm/, DXR 1.1 OMM announcement.
  - VK_EXT/KHR_opacity_micromap: https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_opacity_micromap.html, Vulkan OMM status.
  - UE Ray Tracing Performance Guide: https://dev.epicgames.com/documentation/unreal-engine/ray-tracing-performance-guide-in-unreal-engine, instance budget, cadence, residency streaming, dynamic budgets.
  - AMD GPUOpen RRA guide: https://gpuopen.com/learn/improving-rt-perf-with-rra/, terrain chunking, overlap, update-mode quality.
  - AMD "Subspace Culling for Ray–Box Intersection" (I3D 2023): https://gpuopen.com/download/I3D2023_SubspaceCulling.pdf, occupancy-mask culling numbers.
  - NVIDIA Ampere GA102 whitepaper: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf, 2nd-gen RT cores (2×).
  - Tellusim RT performance comparison: https://tellusim.com/rt-perf/, RTX 3080 vs 2080 Ti, BLAS build/size, AMD weakness.
  - zeux.io "Measuring acceleration structures": https://zeux.io/2025/03/31/measuring-acceleration-structures/, driver/vendor BLAS memory spread, BVH internals.
  - Keller SIGGRAPH 2019 slides: https://cg.informatik.uni-freiburg.de/intern/seminar/raytracing%20-%20Keller%20-%20SIGGRAPH%202019%202%20Acceleration%20Data%20Structures.pdf, refit vs build rates, BFV numbers.
  - wgpu ray_tracing.md + CHANGELOG v27 + v30 release + PR #9835: https://github.com/gfx-rs/wgpu/blob/trunk/docs/api-specs/ray_tracing.md, https://github.com/gfx-rs/wgpu/blob/v27.0.1/CHANGELOG.md, https://github.com/gfx-rs/wgpu/releases/tag/v30.0.0, https://github.com/gfx-rs/wgpu/pull/9835, experimental RT scope, no update mode, submit-validation pitfall.
  - wgpu tracking issue #6762: https://github.com/gfx-rs/wgpu/issues/6762, OMM/PTLAS unchecked.
  - Minecraft RTX frame analysis: https://medium.com/swlh/frame-analysis-minecraft-rtx-beta-5602081cd90c, cadence precedent.
  - UntitledDXRVoxels: https://github.com/RMichelsen/UntitledDXRVoxels, triangle-based voxel-RT contrast.
  - Microsoft D3D12RaytracingProceduralGeometry sample: https://github.com/microsoft/directx-graphics-samples/blob/master/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingProceduralGeometry/readme.md, procedural pipeline reference.
  - SDF_D3D12: https://github.com/jambuttenshaw/SDF_D3D12, empty-space compaction in procedural AABBs.
  - nvpro blas_caching: https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/blas_caching.md, streaming BLAS reuse.
  - WebGPU voxel RT demos (compute-based): https://github.com/AddisonPrairie/WebGPU-.vox, https://github.com/TwentyFiveSoftware/webgpu-ray-tracing
  - Repo-internal: docs/research-teardown-hardware-ray-tracing.md (findings 3/4/5/6) and its primary sources (GPC 2025 recording/slides, Voxagon blog).
- Dropped:
  - Generic Khronos tutorials (Vulkan-Tutorial ray-query course), background only.
  - DGriffin91/tray_racing Embree-vs-GPU benchmark, CPU-ray-tracing oriented, redundant with Tellusim/zeux data.
  - PixelPics ElementIntersector (Minecraft-clone Java source), software raytracer, not hardware RT.
  - StormCreeper/Minecraft-Raytracing and MahoganyTown shaderpacks, software/OpenGL voxel RT, out of scope.
  - NVIDIA community Q&A / marketing pages on RTX 30 ("up to 2×"), superseded by the whitepaper citation.

## Gaps

- No public RTX-30 (or any) TLAS-build-time-vs-instance-count table was found;
  cadence guidance is qualitative (UE/NVIDIA/AMD). Tellusim covers whole-scene
  trace + BLAS build, not TLAS scaling.
- Minecraft RTX's world primitive representation (triangle block models vs
  AABB/procedural) could not be confirmed from primary sources; only its
  per-frame BLAS/TLAS cadence is verified.
- Actual `maxInstanceCount`/`maxPrimitiveCount` device values on NVIDIA/AMD
  Vulkan drivers are queryable but not pinned here (DXR numbers are spec-fixed).
- BLAS storage/build numbers for *AABB* primitives specifically were not found
  (zeux.io and Tellusim measure triangles); AABB BLASes will differ (box nodes
  dominate, no triangle nodes).
- wgpu-30-specific API details verified against trunk/v27–v30 docs, not the exact
  30.x release docs. Minor drift risk on names like `create_blas`/`TlasInstance`.
- AMD RDNA OMM support status not confirmed in any source found.
- Follow-up suggestion: micro-benchmark chunk-BLAS build time (e.g., 8³ vs 16³ vs
  32³ chunks per region) on the target GPU via wgpu experimental ray queries, and
  measure per-frame TLAS rebuild at region-instance counts.

## Sources note

All claims above are cited inline. Anything marked "estimate" or "flagged" is an
extrapolation or unverified-in-this-run; treat those as assumptions to validate
on target hardware.
