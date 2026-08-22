# ReSTIR, RTXDI, and importance sampling for atlas-rt

> Research request: NVIDIA path tracing optimizations — importance-sampling
> foundations, the ReSTIR family, the RTXDI SDK, adjacent hardware features —
> and what they mean for a 1-spp voxel path tracer that already does NEE+MIS
> over an analytic sun + procedural sky and denoises with NRD ReBLUR. No owning
> ticket; requested directly via /research (August 2026).
> Verification date: 2026-08-22. Claims checked against paper pages
> (research.nvidia.com, diglib.eg.org, dl.acm.org, cemyuksel.com), Khronos and
> NVIDIA blogs, the in-tree vendored Vulkan spec (`docs/vulkan/appendices/`),
> the local vulkano git checkout, pinned ash 0.38.0+1.3.281 registry sources,
> and — for everything RTXDI — a local clone of tag v3.0.0 (tagged commit
> 274141a, 2026-03-10; RTXDI-Library submodule head d28e20f, 2026-04-29) with
> README, ChangeLog, LICENSE, all Doc/*.md guides and Include/Rtxdi headers read
> directly. Primary sources only; remaining gaps are collected at the end.

## Executive answer

**Nothing in the ReSTIR/RTXDI toolbox pays off for atlas-rt today, and that is a
consequence of a locked decision, not an accident: ReSTIR exists to fight light
selection noise across many lights, and atlas-rt samples exactly two analytic
lights (Sun + procedural sky) whose NEE+MIS is already near-optimal.** The value
unlocks the moment emissive voxels or local area lights become directly
sampleable — voxel scenes turn every `_emit` palette entry into thousands of
tiny emitters, which is precisely the "millions of lights" regime RTXDI was
built for. The right posture is: adopt blue-noise sampling now (NRD already
recommends Heitz Owen-scrambled Sobol), keep a ReSTIR-DI-shaped design sketch
(reservoirs over candidate light samples, temporal + spatial + visibility
reuse, feeding ReBLUR unchanged) for when the no-emissive-NEE decision is
revisited, not adopt the RTXDI SDK (proprietary EULA like NRD's, 36 `RAB_` bridge
functions, ~2.7k LOC minimal-sample floor — hand-write the ~100-line reservoir
core against the cloned SDK as executable spec instead), defer
ReSTIR GI/PT until measured indirect quality demands them, and treat Shader
Execution Reordering as the one hardware feature worth profiling — vulkano
master already models `nv_ray_tracing_invocation_reorder` and voxel scenes
have exactly the incoherent-hit divergence SER targets, but adoption means a
hit-object raygen rewrite and both pinned Rust bindings lack the late-2025
EXT extension — and this machine's GPU is an RTX 3070 (Ampere), below the
Ada line where dedicated reordering hardware appeared.

## 1. Importance-sampling foundations (compact)

1. **NEE + MIS is the baseline atlas-rt already implements.** Next-event
   estimation samples light sources explicitly; multiple importance sampling
   (balance heuristic) combines those with BSDF samples without double counting.
   This is textbook material (Veach's thesis; pbr-book) and settled in the
   effort's map (ADR 0011 sun/sky MIS). It stops paying once the *light set* is
   too large to sample exhaustively per bounce.
2. **RIS is the bridge to ReSTIR.** Importance resampling
   ([Talbot et al., "Importance Resampling for Global Illumination", Eurographics 2005](https://diglib.eg.org/items/7b8d7c38-ee96-4415-acdd-3dd164fa8fad))
   draws M candidates from a cheap pdf, picks one proportionally to target/candidate,
   and divides by the average weight — unbiased sampling of an arbitrary target
   distribution through a finite candidate pool. Every ReSTIR variant is RIS
   with the candidate pool drawn spatiotemporally.
3. **Blue-noise placement is the cheapest 1-spp win.** Owen-scrambled Sobol with
   blue-noise permutation of pixel sample sets decorrelates neighbor errors;
   NRD's own guidance recommends Heitz's scheme over white noise and warns that
   probabilistic lobe selection must add global temporal jitter to avoid
   directional bias ([NRD README](https://github.com/NVIDIA-RTX/NRD/blob/master/README.md),
   established in [docs/research-nrd-reblur-path-tracing.md](research-nrd-reblur-path-tracing.md)
   §6.4). atlas-rt should do this in the Sample pass regardless of anything else
   in this document.

## 2. The ReSTIR family

1. **ReSTIR DI** ([Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz,
   "Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic
   Direct Lighting", SIGGRAPH 2020 / TOG 39(4)](https://research.nvidia.com/index.php/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct))
   keeps per-pixel **reservoirs** — running RIS over candidate light samples
   where each reservoir stores the selected light sample, a combined weight, and
   the count M of candidates considered. **Temporal reuse** updates this frame's
   reservoir with last frame's selected sample after reprojection through motion
   vectors; **spatial reuse** merges k neighbors' reservoirs on the same plane;
   optional **visibility reuse** shoots one shadow ray per reused reservoir so
   occluded selections don't poison neighbors. Bias enters when candidates
   cross pixels with different target pdfs; corrections are the classic
   1/M-count weighting, an MIS-style combination, or pairwise-MIS weighting
   (paper §6; formalized in [Wyman & Yuksel, "Generalized Resampled Importance
   Sampling: Foundations of ReSTIR", SIGGRAPH 2022 / TOG 41(4)](https://dl.acm.org/doi/abs/10.1145/3528223.3530158),
   [author PDF](https://www.cemyuksel.com/research/papers/sig22_GRIS.pdf)).
2. **ReSTIR GI** ([Ouyang et al., HPG 2021](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing))
   applies the same reservoir machinery to one indirect bounce: the "sample"
   becomes a short path segment (neighbor point + direction), reused
   spatiotemporally instead of shooting full GI rays per pixel.
3. **ReSTIR PT** ([Lin, Kettunen, Gilcher, Yuksel, SIGGRAPH 2022 Talks](https://research.nvidia.com/labs/rtr/tag/resampling/))
   extends reuse to full paths via **shift mappings**: when reusing neighbor
   history, the reused path must be re-evaluated under the new pixel's bsdf —
   random-replay shifts replay rng decisions, reconnection shifts reconnect the
   path at a vertex near the camera (dominant practical choice), hybrid mixes
   both. Follow-up: ["ReSTIR PT Enhanced"](https://www.semanticscholar.org/paper/ce50617e11dfac0469f993576ffc9bba421b938f)
   (faster, more robust shifts). A gentle overview of the whole family is the
   [SIGGRAPH 2023 course "A Gentle Introduction to ReSTIR Path Reuse in Real-Time"](http://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Welcome.pdf);
   defocus/antialiasing variants exist too ([Area ReSTIR, Zhang et al. 2024](https://research.nvidia.com/labs/rtr/publication/zhang2024area/zhang2024area.pdf)).
4. **Cost profile.** RTXDI packs a direct-lighting reservoir into **24 B/pixel**
   (`RTXDI_PackedDIReservoir`, 6 dwords: selected light id, radiance/pdf
   terms, combined weight, M-count, target pdf, sample position; GI 32 B, PT
   64 B). Two screen-sized ping-pong arrays cost **≈ 95 MiB at 1080p internal**.
   Passes per frame: initial candidate sampling → temporal reuse → spatial
   reuse (often 2+ rounds) → visibility rays → shade — each one fullscreen
   dispatch at roughly raygen cost, trading a handful of cheap passes for
   orders-of-magnitude fewer shadow rays than per-pixel NEE over huge light
   sets. NVIDIA's documented operating points: **temporal-only reuse reads as
   blotches to a temporal denoiser — one spatial pass fixes the signal
   structure**, and **visibility reuse cuts the final shadow-ray count to
   ~30%**. The failure modes that matter operationally: disocclusions and
   camera cuts invalidate temporal history (clamp/clear the reservoir M **and**
   give the denoiser its own reset — see §5), moving/emissive-geometry changes
   poison reused samples without validity checks, and biased-correction
   shortcuts trade energy error for speed.

## 3. RTXDI SDK

1. **Identity and version.** [NVIDIA-RTX/RTXDI](https://github.com/NVIDIA-RTX/RTXDI)
   packages ReSTIR DI/GI/PT as a shipping SDK: HLSL shader includes (GLSL
   via `RTXDI_GLSL`) compiled into *your* shaders, a light registry,
   reservoir management, and runtime parameters; MinimalSample and a
   Donut/NVRHI FullSample for D3D12 + Vulkan. Current release **v3.0.0**
   (tagged commit 274141a, 2026-03-10, "ReSTIR PT"); the integrable runtime
   lives in the [RTXDI-Library](https://github.com/NVIDIA-RTX/RTXDI-Library)
   submodule (head d28e20f, 2026-04-29) — verified from a local clone of the
   tag (README, ChangeLog, LICENSE, all Doc/*.md, headers read directly).
   The registry is one buffer of **polymorphic lights** (kSphere, kCylinder,
   kDisk, kRect, kTriangle, kDirectional, kEnvironment, kPoint),
   double-buffered odd/even frames so temporal resampling reads previous-frame
   light data (`RAB_TranslateLightIndex` maps indices across frames);
   emissive meshes split per-triangle with precomputed average radiance (the
   docs name the pain points: emissive detection, material integration over
   the triangle, shader permutations). Optional accelerators: power-PDF
   texture + RIS presampling over local lights, environment-PDF presampling,
   ReGIR world-space grids. Two negative findings, grep-verified in v3.0.0:
   **no "implicit lights" registry format and no `RTXDIParticipatingMedia`
   API exist**. NVIDIA's positioning:
   ["Lighting Scenes with Millions of Lights Using RTX Direct Illumination"](https://developer.nvidia.com/blog/lighting-scenes-with-millions-of-lights-using-rtx-direct-illumination/).
2. **Requirements and pairing.** Targets DXR-1.1-class ray tracing (DX12 and
   Vulkan backends ship in the repo) — i.e., strictly more capability than
   atlas-rt's enabled set (`khr_acceleration_structure` +
   `khr_ray_tracing_pipeline` + maintenance1,
   `src/core/gpu.rs`), but check the programming guide's exact Vulkan feature
   list against vulkano's exposure before assuming. RTXDI is designed to feed a
   separate denoiser; NRD is the reference pairing, and ReLAX is NRD's
   "designed for RTXDI" denoiser (established in
   [docs/research-nrd-reblur-path-tracing.md](research-nrd-reblur-path-tracing.md)
   §1.3) — ReBLUR remains fine for diffuse+specular DI output at 1 spp.
3. **License — resolved: proprietary EULA, same family as NRD.** RTXDI's
   `LICENSE.txt` (read from the v3.0.0 clone) is the "NVIDIA RTX SDKs
   LICENSE": object-code distribution only inside an application with
   "material additional functionality", the NVIDIA attribution notice,
   downstream terms at least as protective, and clause 4(e) forbidding any use
   that would subject the SDK to an open-source license; library sources carry
   `LicenseRef-NvidiaProprietary` SPDX headers. Not MIT, cannot merge into
   the MIT tree — the same distinctly-licensed-vendored-component treatment
   NRD got (ticket 01).
4. **Fit for a Rust+vulkano codebase.** The library makes **zero graphics-API
   calls** (pure C++ parameter/size math + shader includes), but the app
   contract is **36 documented `RAB_` bridge entries** (5 structs + 31
   functions, Doc/RtxdiApplicationBridge.md; the central hook is
   `RAB_GetLightSampleTargetPdfForSurface`), a light registry with mesh
   preprocessing, and even NVIDIA's MinimalSample floor is **≈ 2.7k LOC**
   (counted locally). Shaders can take the HLSL-include + dxc-to-SPIR-V path
   (the pattern already planned for NRD) or the `RTXDI_GLSL` macro path —
   the latter untested against vulkano-shaders' GLSL dialect. The serious
   alternative inverts the NRD situation: NRD's value was closed machinery
   worth binding, while here **the valuable part is public math** —
   `RTXDI_StreamSample`/`RTXDI_CombineReservoirs`/`RTXDI_FinalizeResampling`
   are the papers' Algorithm 3/4 + Eq. 6, ≈ **100 lines of scalar math** — so
   hand-write the reservoir core in GLSL using the cloned SDK as executable
   reference and its docs as the spec. Options ranked: (a) hand-written
   ReSTIR-DI in GLSL when triggered; (c) wait until the many-light scenario
   exists; (b) vendor the SDK only if ReGIR-scale emitter counts arrive and
   the EULA is accepted. Nice-to-haves the SDK would hand over: a boiling
   filter (`RTXDI_BoilingFilter`, mandatory once rare bright lights exist)
   and an optional confidence channel from sampling statistics that speeds
   denoiser convergence (Doc/Confidence.md).

## 4. Adjacent NVIDIA optimizations

1. **Shader Execution Reordering (SER).** Groups invocations executing the same
   shader branch/hit group to restore warp coherence after divergent
   intersections. Vulkan story: [`VK_EXT_ray_tracing_invocation_reorder`](https://www.khronos.org/blog/boosting-ray-tracing-performance-with-shader-execution-reordering-introducing-vk-ext-ray-tracing-invocation-reorder)
   provides API support for `SPV_NV_shader_invocation_reorder` — the vendored
   spec copy lives in-tree at
   `docs/vulkan/appendices/VK_EXT_ray_tracing_invocation_reorder.adoc`
   (last modified 2025-11-12). NVIDIA's production data point:
   [Indiana Jones path tracing used SER + live state reductions](https://developer.nvidia.com/blog/path-tracing-optimization-in-indiana-jones-shader-execution-reordering-and-live-state-reductions/)
   for meaningful frame-time wins. Local support check: **vulkano master
   (checkout fb4cfdb) models the NV device extension**
   (`DeviceExtensions::nv_ray_tracing_invocation_reorder`,
   `ray_tracing_invocation_reorder_reordering_hint` properties) but names no
   `ext_*` field — so the reachable-today path is the NV extension on NVIDIA
   hardware, with hit-object usage in SPIR-V; the standardized EXT path would
   need raw-Vulkan plumbing around vulkano until it autogenerates the field.
   Bind-level corroboration: pinned **ash 0.38.0+1.3.281** headers contain
   only `VK_NV_ray_tracing_invocation_reorder` (#491) —
   `VK_EXT_ray_tracing_invocation_reorder` (#582, late 2025) is absent from
   both pinned bindings, so the EXT path needs binding upgrades too.
   Requires `khr_ray_tracing_pipeline` (already enabled). Measurement is
   cheap on the existing per-stage timestamp harness, but adoption is a
   hit-object raygen rewrite, not a flag flip — profile divergence first.
   Hardware gate: dedicated reordering arrived with Ada (RTX 40);
   implementations without it expose hit objects while skipping the actual
   reordering — `rayTracingInvocationReorderReorderingHint`
   (vulkano: `reordering_hint`) exists precisely to report that. This
   machine pairs a GeForce RTX 3070 (Ampere) with an AMD Radeon iGPU, so SER
   buys ~nothing here today; revisit only on Ada-class hardware.
2. **DLSS Ray Reconstruction** is NVIDIA's closed AI denoiser trained for path
   traced rendering, replacing hand-written denoisers in shipped titles (current
   line DLSS 4.x; the 4.5 update retrained it on broader data with a
   second-generation transformer, per
   [heise, 2025](https://www.heise.de/en/news/DLSS-4-5-Nvidia-improves-Ray-Reconstruction-for-raytracing-games-11313209.html)).
   It is NVIDIA-hardware-bound, closed, and NGX-integrated — the opposite trade
   from the committed cross-vendor NRD choice. Precise framing: RR may be
   *better* where it runs (trained model, whole-signal temporal state; untested
   against ReBLUR here — that claim rests on NVIDIA positioning), but it is not
   better *suited*: closed NGX binaries inside the DLSS upscaling chain rather
   than a standalone modular denoiser, none of the per-signal control atlas-rt
   relies on (diffuse/specular separation, the radiance +
   normalized-hit-distance contract, validation overlay, split-screen QA).
   ReBLUR stays; the question reopens only if atlas-rt ever ships as an
   NVIDIA-only product where maximum 1-spp image quality justifies NGX
   integration.
3. **Opacity Micromaps / Displacement Micro-Meshes / RTX Mega Geometry** attack
   *triangle* acceleration-structure costs (alpha-tested geometry without shader
   invocations; clustered BLAS builds — see
   [RTX Mega Geometry with new Vulkan samples](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/)).
   All irrelevant while atlas-rt intersects **DDA voxel grids inside procedural
   intersection shaders** — there are no triangles, no alpha tests, no BLAS
   refits. Revisit only if triangle-mesh terrain ever lands. (`VK_EXT_opacity_micromap`
   appendix is already vendored in-tree for that day.)
4. **Many-light sampling without ReSTIR** — the classical GPU answer is
   light hierarchies: [Conty Estévez & Kulla, "Importance Sampling of Many
   Lights on the GPU" (Ray Tracing Gems, ch. 18)](https://scholar.archive.org/fatcat/release/sj24zbmzincknalg6mxzocrble)
   build a BVH over lights sampled by importance (cont triangulation for mesh
   lights) and traverse it per shading point. Lower machinery than ReSTIR, no
   temporal state, but weaker quality at extreme counts; worth considering as
   the *first* step above exhaustive NEE before committing to reservoirs.
5. Adjacent curiosity: ReSTIR-style reservoir resampling also covers volumes
   ([Fast Volume Rendering with Spatiotemporal Reservoir Resampling, NVIDIA Research 2021](https://research.NVIDIA.com/index.php/publication/2021-11_fast-volume-rendering-spatiotemporal-reservoir-resampling)) —
   noted in case atlas-rt grows volumetrics.

## 5. Applicability to atlas-rt

1. **Now: nothing structural changes.** With Sun + sky as the only NEE targets,
   ReSTIR's core value proposition (avoiding per-pixel selection among huge
   light sets) has nothing to bite on; MIS already handles the two-source
   combination, and adding reservoirs would add bias risk and temporal state for
   zero variance reduction. The locked no-emissive-NEE decision (map) keeps the
   light set at two by construction. At 1 spp the residual variance comes from
   BSDF-lobe shape and from visibility (shadow edges, sky seen through gaps) —
   not from choosing between lights — so reservoir sharing cannot touch it; the
   denoiser handles what sampling hygiene leaves.
2. **Trigger: direct sampling of emissive voxels / local lights.** If that
   decision flips, voxel worlds generate enormous emitter sets (every `_emit`
   palette cell × lit faces), and the migration order should be: (a) importance-
   weighted light registry built from the material table; (b) hierarchical
   many-light sampling (§4.4) as the low-tech baseline; (c) ReSTIR-DI reservoirs
   with temporal+spatial+visibility reuse once measurement shows the hierarchy
   is not enough. Denoising contract is unchanged — de-modulated radiance into
   ReBLUR as today.
3. **Validator compatibility is the quiet constraint.** ReSTIR is
   seed-deterministic, so the CPU diff harness stays viable, but reservoir state
   is temporal cross-frame state: the CPU mirror must reproduce reuse decisions
   frame-by-frame, and camera-cut resets need the same explicit reset semantics
   NRD taught us (one-frame RESTART analogues). Budget validation complexity for
   that, not just shader LOC.
4. **SER is the one cheap experiment available now** — NVIDIA-hardware-only,
   one device-extension flag through vulkano's existing NV modeling, hit-object
   intrinsics in the raygen loop, measurable immediately on the timestamp
   harness. Do it as its own micro-ticket if primary-hit material divergence
   shows up in profiling; it composes with everything else here.
5. **Blue noise first — but swap it atomically.** Owen-scrambled Sobol with
   blue-noise permutations is the highest-certainty, lowest-cost improvement
   for a 1-spp renderer, is already NRD-aligned, and touches only RNG indexing
   in the Sample pass. It is a new sample *sequence*, not a filter: white-noise
   RNG calls become Owen-scrambled Sobol samples, decorrelated across pixels by
   small hash-based per-tile rank/scramble tables (Burley 2020 — no textures
   strictly required). Same estimator in and out; the leftover error becomes
   screen-spread and frame-stable instead of clumpy, which is exactly what NRD
   wants at 1 spp. It regenerates every sample index, so the CPU reference's
   seeds must be regenerated in the same commit or the diff harness breaks.
   The repo's RNG is structurally ready: ADR 0010 fixes a stateless PCG-XSH-RR
   hash whose draws are pure functions of `(seed, draw_index)`
   (`shaders/region/production.rgen`), so a Sobol sample is just a scrambled
   sequence element XORed with an existing hash draw — the byte-exact CPU
   mirror stays reproducible when both sides switch together. The `p = 0.5`
   diffuse/specular coin flip already sits inside NRD's recommended `[1/4,
   3/4]` probabilistic-lobe clamp, precisely where the global-temporal-jitter
   warning applies. ("1 spp" counts camera paths — one primary ray per pixel
   plus up to four BSDF bounces under Russian roulette at a 0.05 floor — not
   trace calls.)
6. **Camera cuts couple the resets.** A cut must clamp/clear reservoir M
   **and** send NRD `accumulationMode::RESTART` for that frame together —
   spatial reuse over stale M propagates ghosts straight into the denoiser's
   history.
7. **Validation policy must be chosen up front.** Reservoir state makes frames
   history-dependent: the seed-identical CPU diff must either reset both sides
   per frame (validates single-frame behavior) or evolve both temporally
   (validates reuse). Decide before implementing, not after.
8. **Emissive voxels are never one-light-per-voxel.** Presample by power PDF
   over an emissive-light registry and consider a ReGIR-shaped world-space
   grid keyed to the chunk lattice; add a boiling filter once rare bright
   lights exist.

## Sources

- Kept:
  - Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz — ReSTIR DI, SIGGRAPH 2020 / TOG 39(4). https://research.nvidia.com/index.php/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct
  - Ouyang et al. — ReSTIR GI, HPG 2021. https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
  - Lin, Kettunen, Gilcher, Yuksel — ReSTIR PT, SIGGRAPH 2022 Talks (+ Enhanced follow-up). https://research.nvidia.com/labs/rtr/tag/resampling/
  - Wyman & Yuksel — Generalized RIS: Foundations of ReSTIR, SIGGRAPH 2022 / TOG 41(4). https://dl.acm.org/doi/abs/10.1145/3528223.3530158 · https://www.cemyuksel.com/research/papers/sig22_GRIS.pdf
  - Wyman, Kettunen et al. — SIGGRAPH 2023 course, A Gentle Introduction to ReSTIR Path Reuse. http://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Welcome.pdf
  - Zhang et al. — Area ReSTIR, 2024. https://research.nvidia.com/labs/rtr/publication/zhang2024area/zhang2024area.pdf
  - Talbot et al. — Importance Resampling for Global Illumination, EG 2005. https://diglib.eg.org/items/7b8d7c38-ee96-4415-acdd-3dd164fa8fad
  - Conty Estévez & Kulla — Importance Sampling of Many Lights on the GPU, Ray Tracing Gems ch. 18. https://scholar.archive.org/fatcat/release/sj24zbmzincknalg6mxzocrble
  - NVIDIA-RTX/RTXDI v3.0.0 (local clone, tag commit 274141a, 2026-03-10): README, ChangeLog, LICENSE.txt ("NVIDIA RTX SDKs LICENSE"), Doc/Integration.md, Doc/ShaderAPI.md, Doc/RtxdiApplicationBridge.md, Doc/RestirPT.md, Doc/Confidence.md, MinimalSample LOC count. https://github.com/NVIDIA-RTX/RTXDI
  - NVIDIA-RTX/RTXDI-Library (local clone, head d28e20f, 2026-04-29): Include/Rtxdi tree (RTXDI_PackedDIReservoir, RTXDI_StreamSample/CombineReservoirs/FinalizeResampling), polymorphic light types, negative grep results (no "implicit lights", no RTXDIParticipatingMedia). https://github.com/NVIDIA-RTX/RTXDI-Library
  - NVIDIA blog — Lighting Scenes with Millions of Lights Using RTX Direct Illumination. https://developer.nvidia.com/blog/lighting-scenes-with-millions-of-lights-using-rtx-direct-illumination/
  - Bind-level checks: pinned ash 0.38.0+1.3.281 registry sources (VK_NV_ray_tracing_invocation_reorder present, VK_EXT #582 absent); vulkano git checkout fb4cfdb (nv_ray_tracing_invocation_reorder modeling).
  - Khronos blog — VK_EXT_ray_tracing_invocation_reorder. https://www.khronos.org/blog/boosting-ray-tracing-performance-with-shader-execution-reordering-introducing-vk-ext-ray-tracing-invocation-reorder
  - NVIDIA blog — Path Tracing Optimization in Indiana Jones (SER + live state reductions). https://developer.nvidia.com/blog/path-tracing-optimization-in-indiana-jones-shader-execution-reordering-and-live-state-reductions/
  - NVIDIA blog — RTX Mega Geometry with Vulkan samples. https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/
  - heise — DLSS 4.5 Ray Reconstruction improvements. https://www.heise.de/en/news/DLSS-4-5-Nvidia-improves-Ray-Reconstruction-for-raytracing-games-11313209.html
  - NVIDIA Research — Fast Volume Rendering with Spatiotemporal Reservoir Resampling, 2021. https://research.NVIDIA.com/index.php/publication/2021-11_fast-volume-rendering-spatiotemporal-reservoir-resampling
  - In-tree primaries: docs/vulkan/appendices/VK_EXT_ray_tracing_invocation_reorder.adoc (Khronos, CC-BY-4.0, 2025-11-12); src/core/gpu.rs (enabled extensions/features); vulkano git checkout fb4cfdb (nv_ray_tracing_invocation_reorder modeling).
- Gaps (not verified against primary sources): Bitterli 2020 per-pass
  millisecond tables (paper PDF unfetchable); Falcor's RTXDI module; vulkano
  RawDeviceExtensions API at the pinned commit; dxc flags for
  `SPV_EXT_shader_invocation_reorder`; any official RTXDI cross-vendor
  statement; RTXDI minimum shader model; Heitz 2019 co-authors beyond
  Heitz/Belcour; DLSS-RR hardware requirements from license documents rather
  than marketing pages. None blocks the verdict.
