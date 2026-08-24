# Face-radiance cache: integration points and cost model (ticket 02)

## Executive answer

A per-voxel-face radiance cache hooks into exactly two places: the DDA hit path that already computes `hit_kind`/normal/material (the single choke point every traceRayEXT result routes through, `shaders/region/production.rgen`'s payload contract), and a small post-hit step in the path loop. At 1080p with up to 4 bounces plus NEE shadow rays, worst-case hit traffic is ~20–33M hits/frame (realistically ~10–15M after sky escape and Russian roulette); blending RGBA16F + age into a per-face table at that rate is ~180 MB/frame ≈ 11 GB/s raw write traffic at 60 fps — trivial for an RTX 3070's ~448 GB/s L2/DRAM — but only if blends are packed into ≤ 2× u32 atomics with warp aggregation; naive 4-component `atomicAdd` (~60M atomics/frame) is the real contention risk. Structure verdict: a fixed sparse grid keyed per Micro-chunk (8³ voxels × 6 faces = 3,072 entries/chunk), allocated by the existing Region/Resident lifecycle in `src/render/`, beats a general GPU open-addressed hash map because keys are dense within resident chunks — hashing buys nothing here. Storage at per-voxel-face granularity (RGBA16F + age = 12 B/face): **~630 MB all-faces / ~315 MB visible-only for a solid 512³ world; ~1.2 GB visible-only at 25% occupancy** — fits comfortably in the 3070's 8 GB beside existing pools under a hard cap; a 1024³ world (~5–10 GB) does not fit and needs camera-local residency gating.

## Findings

### Integration points

1. **Single write choke point already exists: the DDA hit resolution.** Every ray result (primary, bounce, NEE shadow via `shadowed()`) routes through one payload contract in `shaders/region/production.rgen`: `payload.t`, `payload.hit_kind` (material-table row index, ADR 0008), `payload.normal`. The hit path under `shaders/` (DDA inside region/hull modes, selected by `sbt_offset`) already computes voxel coord + entry face — exactly the cache key `(voxel, face)` and the shading inputs. A cache write is a few lines appended there; no new traversal machinery.

2. **Read path option A (shading-only reuse):** at each bounce hit, look up the face entry; if fresh, use it as the incoming-radiance estimate instead of tracing the continuation `traceRayEXT` (the call at the bottom of the depth loop, `depth <= MAX_BOUNCES=4`). Preserves the loop, RR floor (`RR_FLOOR=0.05`), and both NEE branches untouched. Caveat: cosine-weighted sampling of cached *outgoing radiance* is not the same estimator as BSDF sampling — store per-face **irradiance** (hemisphere integral) and modulate by albedo, or restrict the cache to diffuse lobes and keep tracing when the specular lobe was picked (`LOBE_P = 0.5` on the primary).

3. **Read path option B (secondary-ray termination):** bounce rays hitting a face with a valid entry terminate immediately with the cached value. Saves full sub-paths (bigger win than A) but changes the path PDF, so MIS weights against the Sun/sky pick (`SUN_PICK_P = 0.5`, `sky_light_pdf`) must be recomputed: cached radiance replaces the BSDF-sampled continuation with weight 1; NEE stays as-is. Standard radiance-cache/MIS split (Pantaleoni's RCU, ReSTIR-PT final-gather). Recommend A first — zero estimator change — B later for diffuse lobes at depth ≥ 1.

4. **Emissive hits get covered for free.** There is no NEE for emissive voxels today (`rough_emit.rgb` is added directly at each hit). Because *all* paths write the cache, emitter radiance propagates one face-hop per frame — cheap emitter visibility without new light sampling. Side benefit, not a design driver. Note the ticket's standing decision: the cache stores all transport hitting a face (direct sun/sky included), which is what makes this work.

### Write path cost

5. **Hit traffic at 1080p × 4 bounces.** 1920×1080 ≈ 2.07M primary pixels; per pixel up to 5 path segments (depths 0..4), each with one continuation trace plus a shadow trace with probability `SUN_PICK_P = 0.5`. Upper bound ≈ 2.07M × 5 × 1.5 ≈ **15.5M traces/frame** (worst case ~31M counting both-lobe paths); after sky escape (~30–50% in typical .vox scenes) and Russian roulette (survival usually ≥ 0.5 past bounce 2, floor 0.05), realistic hit volume is **~8–15M hits/frame**. These are the numbers to instrument-confirm; the `clockRealtime2x32EXT` scaffolding behind `ATLAS_RT_RAY_LATENCY` shows the pattern exists.

6. **Write bandwidth and atomic contention.** Entry = RGBA16F radiance (8 B) + packed age/confidence u32 (4 B) = 12 B. At 15M writes/frame: ~180 MB/frame ≈ **11 GB/s at 60 fps raw**, before coherence savings; warp-aggregated atomics (NVIDIA's canonical pattern: warp-wide reduction, one atomic per unique target line) cut DRAM traffic toward the unique-touched-face count — typically 3–10× fewer than hits given spatial coherence. **Bandwidth is not the constraint; atomic throughput is.** Naive `atomicAdd` on 4 components × 15M hits ≈ 60M atomics/frame ≈ 1M/ms sustained — near a 3070's practical ceiling and will serialize hot faces (e.g., ground planes under sky light). Mitigation: pack RGB into one u32 (R11G11B10-style or RGB10+E5 exponent) and blend via `atomicCompSwap` retry loops or `atomicAdd` on pre-quantized increments; two atomics per hit max. This mirrors how voxel cone-tracing octrees (Crassin et al.) and voxelization pipelines handle fragment-to-voxel writes.

7. **Contention structure.** Hot spots are faces receiving many hits per frame (sun-facing ground, emitter surfaces). With per-face entries and u32-packed blends, contention cost is bounded by the compSwap retry loop; the alternative — staging in a per-pixel/per-warp scratch buffer and resolving in a post-pass — adds a full extra dispatch and ordering complexity. Start with direct atomic blends + packing; measure before adding a resolve pass.

### Structure choice

8. **Fixed sparse grid keyed per Micro-chunk beats a general GPU open-addressed hash map.** The codebase's Region/Resident-region lifecycle in `src/render/` (pool allocation, stream-in/out, dirty tracking — the machinery the BLAS research doc already leans on) provides chunk residency events for free. Keying the cache by Micro-chunk reuses it exactly: slab allocated on resident, freed on evict, index = `chunk_base + voxel_in_chunk*6 + face`. No probing, no resize, no tombstones; lookup is one indirection off the DDA's chunk address. A generic open-addressed map keyed by (voxel, face) only wins for scattered sparse keys; within resident chunks the key space is dense, and every GPU-hash precedent found (SlabHash, warp-cooperative schemes) targets the dynamic-sparse regime, paying probe overhead this design doesn't need. Chunk validity bit doubles as the "is this face backed" check, giving graceful fallback to full tracing for non-cached chunks.

9. **Storage math (RGBA16F 8 B + u32 age 4 B = 12 B/face).**
   - Per fully-populated surface voxel: 6 faces × 12 B = 72 B worst case; eliding buried faces (typical .vox: 2–3 visible faces per surface voxel) → ~24–36 B.
   - **Solid 512³ world**: ~134M voxels → 630 MB all-faces, ~315 MB visible-only.
   - **512³ at 25% occupancy** (~34M solid voxels): ~2.5 GB all-faces, **~1.2 GB visible-only**.
   - **1024³ at 25% occupancy**: ×8 of the above → ~10 GB all-faces / ~5 GB visible-only — does not fit; needs camera-local residency (only chunks near the view frustum carry cache slabs).
   - **RTX 3070 (8 GB)**: voxel pool + material table (ADR 0008: two vec4 rows/material — negligible) + region metadata leave roughly 4–6 GB headroom. The 512³-class scene at ~1.2 GB fits easily; enforce a hard cap (e.g., 2 GB slab) in the region allocator with per-chunk fallback flags. Age stamp can double as LRU for eviction pressure.

10. **Age/confidence is one compare.** Store a frame-stamp u32 alongside each entry (written with the blend); fresh = `current_frame − stamp < T`. Recency is a sufficient proxy for confidence initially; add a separate confidence field only if measured staleness artifacts demand it.

### What becomes redundant

11. **Same-voxel resampling across pixels and frames** — every pixel currently re-traces identical diffuse transport for nearby static surfaces. The cache converts O(pixels × bounces) duplicate transport into O(unique touched faces) updates/frame. In static regions this is where >90% of diffuse-bounce work dies.

12. **Multi-bounce re-tracing collapses under option B:** paths terminate at the first fresh face, cutting average traced hops from ~4 to ~1–2. Roughly 2–3× fewer traceRayEXT invocations in diffuse-dominant scenes, plus lower variance into NRD ReBLUR (shorter paths converge faster) — though note the map's open item that ReBLUR rewiring is deliberately deferred until irradiance stability is judged visually.

## Decision-ready summary

- Hook: DDA hit shader writes (key = Micro-chunk-relative voxel+face; value = RGBA16F irradiance + frame-stamp u32); read at diffuse bounces — shading-only (A) first, termination (B) for depth ≥ 1 later with MIS weight 1 replacing the BSDF term; NEE unchanged.
- Structure: per-Micro-chunk fixed slabs owned by the existing Region lifecycle; no GPU hash map.
- Writes: ~8–15M/frame at 1080p×4 bounces; ≤ 2 packed-u32 atomic blends per hit with warp aggregation. Bandwidth ~11 GB/s raw is fine; atomic count/packing is the thing to benchmark.
- Storage: ~1.2 GB visible-face granularity for a populated 512³ world; fits RTX 3070 with a hard cap; 1024³ needs camera-local residency gating.
- Emissive coverage falls out for free since the cache carries all transport (standing owner decision).

## Sources

Kept:
- `shaders/region/production.rgen` — path loop constants (MAX_BOUNCES=4, SUN_PICK_P=0.5, LOBE_P=0.5, RR_FLOOR=0.05), material reads via `payload.hit_kind`, NEE branches, trace call sites. Primary grounding for findings 1–5, 11–12.
- shaders/ hit/DDA path — cache-key provenance (voxel coord + face), sbt mode selection. Findings 1, 8.
- src/render/ voxel pool + Region/Resident lifecycle — slab allocation/eviction host. Finding 8.
- docs/adr/0007, 0008, 0010 — trace-pass contract, material table layout, fog/clamp context.
- Warp-aggregated atomics (NVIDIA CUDA Pro Tip): https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics — atomic-traffic reduction pattern. Finding 6.
- SlabHash (GPU compute hash table): https://github.com/owensgroup/SlabHash — contrast case showing hashes pay off only for dynamic sparse keys. Finding 8.
- Crassin et al., voxel cone tracing / sparse voxel octree — precedent for fragment→voxel radiance writes with packed formats. Findings 6, 9.
- Pantaleoni et al. RCU / ReSTIR-PT radiance caching — MIS treatment of cached radiance replacing the BSDF continuation. Finding 3.

Dropped:
- Generic Vulkan SSBO-atomics tutorial pages — background only, spec covers it.
- Octree-based radiance caches (VXGI-style) — different structure (hierarchical vs flat per-face), redundant once the Micro-chunk decision is grounded in the local lifecycle.

## Gaps

- Hit-rate and unique-touched-face counts are estimated from shader constants, not measured; one instrumentation run (pattern already exists behind `ATLAS_RT_RAY_LATENCY`) pins findings 5 and 6.
- RTX 3070 sustained throughput for device-address 32-bit atomics under compSwap retries not pinned; validate with a microbench before freezing the entry packing.
- Buried-face elision ratio (2–3×) estimated from typical MagicaVoxel scenes, not measured against the repo's test .vox assets.
