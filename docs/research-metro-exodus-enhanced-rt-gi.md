# Metro Exodus (2019) and Enhanced Edition (2021): what 4A Games actually rendered with ray tracing

> Research request: how 4A Games' Metro Exodus (2019) and especially Metro
> Exodus PC Enhanced Edition (2021) produced deep, noise-free interior
> diffusion of sun and sky light with ray tracing, and at what performance
> cost — as reference for a 1-spp voxel path tracer (1 primary ray + up to 4
> BSDF bounces, NEE over analytic sun + procedural sky, NRD ReBLUR, fixed
> exposure). No owning ticket; requested directly in-session (August 2026).
> Verification date: 2026-08-31. Claims checked against the 4A Games GTC/GDC
> 2019 slide deck (developer.download.nvidia.com s9985 PDF, read in full),
> both 4A Games 4a-dna technical blogs (PC Enhanced Edition, May 2021;
> Gen-9 console upgrade, June 2021), the Digital Foundry Enhanced Edition
> analysis and Jon Bloch tech interview (digitalfoundry.net, raw HTML read),
> NVIDIA developer-blog articles on Metro, and the NVIDIA Game Ready driver
> article for the Enhanced launch. Primary sources only; every claim is
> attributed to 4A first-party, NVIDIA-vendor, or Digital Foundry measurement
> at point of use. Unverifiable items are in Dropped/Gaps, not the body.

## Executive answer

**4A's noise-free interior look comes from a hybrid design, not from brute-force
path tracing: rasterized primary visibility plus a *deferred*, one-pass-per-pixel
diffuse GI ray whose hit point is lit by the *full deferred lighting stack*
(sun + sky + up to 256 analytic lights + emissive surfaces since the DLCs), with
recursive bounces supplied by a DDGI probe grid that samples the previous frame,
all denoised by 4A's own in-house spatiotemporal filter and averaged into TAA.**
The 2019 original lit the GI ray's hit point with only sun + sky (N-dot-L);
Enhanced extended that same ray budget to the whole light set and moved deep
bounces into RTXGI-style DDGI. 4A's own conference material is the GTC/GDC 2019
talk "Exploring Raytraced Future in Metro Exodus"; **no GDC or other conference
talk on the Enhanced Edition renderer was found** — the technical record for
Enhanced is 4A's own two blog posts plus the Digital Foundry analysis/interview.
Nothing first-party states the Enhanced per-pixel ray count on PC (consoles
"halved per-pixel ray-count" per 4A), and nothing first-party describes exposure
or tonemapping strategy.

## 1. Metro Exodus 2019: the RT GI implementation

1. **Talk identity.** The 4A conference talk is "Exploring Raytraced Future in
   Metro Exodus" (GDC 2019 session, GDC Vault id 1026159; distributed as NVIDIA
   GTC Silicon Valley 2019 session S9985 with slides). Slides credit four
   speakers: Oles Shyshkovtsov, Sergei Karmalsky, Benjamin Archard (4A Games)
   and Dmitry Zhdan (NVIDIA)
   ([GDC Vault](https://gdcvault.com/play/1026159/Exploring-the-Ray-Traced-Future),
   [GTC S9985 slides PDF](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9985-exploring-ray-traced-future-in-metro-exodus.pdf)).
   NVIDIA's Martin Stich separately presented the Metro Exodus and Control RT
   implementations in the GDC 2019 "Graphics Reinvented: RTX Update" talk;
   NVIDIA's summary blog quotes him: "The game computes one bounce of indirect
   light from sunlight… Ray tracing here is used to compute an AO pass and a GI
   pass with the same ray, which are then added to the lighting terms that are
   already being computed for the raster passes"
   ([NVIDIA blog](https://developer.nvidia.com/blog/ray-tracing-in-4as-metro-exodus-and-remedys-control/)).

2. **Ray budget: one diffuse ray per pixel, full resolution, shared between AO
   and GI.** The deck's GI slide: "Shoot rays at every pixel in all directions
   (ok, according to BRDF lobe). Gather lighting at the contact point; multiplied
   by albedo of that point. Accumulate that! Hit distance gives us 'free' RTAO"
   (S9985, RTGI slide). So the per-pixel trace is one cosine-lobe-distributed
   diffuse ray delivering *two* terms (occlusion from hit distance + irradiance
   from the deferred-lit hit), which is how the talk's framing "Global
   Illumination… in less than 1 ray per pixel?" squares with "shoot rays at
   every pixel". **No first-party statement that the GI term was traced at half
   or quarter resolution exists**; the deck describes a full-pixel trace plus a
   screen-space pre-trace, and resolution-scaling options for the GI trace are a
   2021 (Enhanced) feature (§2.6). S9985,
   [PDF](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9985-exploring-ray-traced-future-in-metro-exodus.pdf).

3. **Screen-space pre-trace.** Before any real ray, the same ray generator is
   ray-marched against the depth buffer (async compute, hidden under BVH
   builds); only pixels where the pre-trace finds no intersection spawn a real
   DXR ray. It also fixes missing alpha-tested geometry. Outputs are
   hit-distance + albedo packed into a single UINT; the trace payload is a
   single UINT; only a closest-hit shader is used. S9985, pre-trace and
   raytracing slides.

4. **What the ray sampled.** First-bounce indirect from sunlight only, against
   simplified BLAS geometry (position-only VBs reused from sun-shadow impostors,
   ≈4x smaller than "real" geometry, ~1 GB instead of ~4 GB). The deferred
   lighting pass re-runs the identical ray generation to reconstruct the hit
   position and computes illumination there — MISS samples the skybox, HIT
   computes lighting — initially with a single N-dot-L light (sun/moon) plus an
   area-light term sampling the skybox texture. Albedo at the hit point is a
   precomputed per-instance average ("we store average albedo per-instance"),
   pre-filtered deliberately because "integration across the whole hemisphere is
   a low-pass filter in essence". Hit-point lighting from other analytic lights
   was prototyped (~0.2 ms on a 2080 Ti) but rejected as conflicting with
   hand-crafted lighting and the stealth mechanic — "we were out of time".
   S9985, deferred-lighting and color-transport slides; first-party.

5. **What it replaced.** RTGI superseded the voxel-GI system (RSM-fed SH voxel
   grid, "coarse and only available close to the camera") and RTAO superseded
   SSAO + geometric ESM-AO; removed stages: RSM rendering, ESM-AO
   ("approximation of 16 rays"), SH-voxel-grid computation/gather/temporal
   blend/screen-space resolve. SSAO survives as an accumulation-weight pass
   ("it's cheap and helps guide the denoiser"); the legacy pipeline expects AO
   in 79 shader call sites, and the raster pipeline blends out of RTGI into
   regular AO over a 250 m transition for open-world foreground. S9985,
   implementation slides; first-party. 4A's 2021 blog adds the 2019 raster
   baseline: GI was "approximated with a voxel grid, which transported lighting
   data from a reflective shadow map… very coarse and only available close to
   the camera"
   ([4A tech dive](https://www.4a-games.com.mt/4a-dna/in-depth-technical-dive-into-metro-exodus-pc-enhanced-edition)).

6. **Denoising.** Convolution-based, spatial + temporal, in-house ("Our approach
   is convolution-based and has spatial and temporal components"; "TAA is your
   friend — it's a free pass of denoising"). Key first-party mechanics:
   temporal accumulation always runs *before* spatial denoising and history is
   rejected on z-occlusion/out-of-screen; two GI denoiser passes (pass 1 no
   normals, 6 m base radius; pass 2 normal-weighted, 3 m radius); the output of
   each denoiser is a lerp of denoised and noisy input
   (`lerp(denoised, input, 0.5·accumSpeed)`, accumSpeed 0.93 static) to preserve
   detail; blur radius scaled by view distance, signal variance, and AO; per-pixel
   kernel rotation rejected as "noise on top of noise" in favor of per-frame
   random rotation; AO path adaptively samples up to 64 samples/pixel, 2 pixels
   per thread. Irradiance is stored as L1 SH of Y (4×FP16) + CoCg (2×FP16), 96
   bits/pixel — the SH also encodes a dominant direction + degraded roughness to
   reconstruct indirect specular GGX without extra rays. Rays are seeded from a
   64×64 blue-noise texture indexed by pixel and frame. S9985, denoising slides;
   first-party.

7. **Performance budget.** Deck's own table, RTX 2080 @ 2560×1440: pre-trace
   ~0.4 ms (High) / ~0.8 ms (Ultra); BLAS/TLAS ~0.5 ms (hidden by async compute);
   raytracing 1–3 ms (High) / 2–6 ms (Ultra); AO denoising ~0.6/0.9 ms; GI
   computation ~0.6/1.0 ms; GI denoising ~1.6/2.1 ms; **total frame-time
   overhead vs RTX off ~20% (High) / ~30% (Ultra)**. Deferred lighting of hit
   positions was measured at ~0.2 ms on an RTX 2080 Ti. S9985, performance
   slide; first-party.
   Derived, not published: those add-on sums (~5 ms High / ~9 ms Ultra) against
   the deck's own overhead fractions back-solve to ~26 ms / ~30 ms total frames
   — **~38 fps (High) / ~33 fps (Ultra)** at 1440p on a 2080. No total fps is
   stated anywhere first-party; this is arithmetic on the deck's own table.

## 2. Metro Exodus PC Enhanced Edition (2021): what became fully ray traced

1. **What was removed with the raster lighting.** DF (Battaglia): "For the new
   Metro, the standard rasterised versions of each map — including all
   individually, artist-placed lights — are gone. The tricks, fake light sources
   and other legacy elements are replaced with a proper, real-time RT solution"
   ([DF analysis](https://www.digitalfoundry.net/articles/digitalfoundry-2021-inside-metro-exodus-enhanced-edition-pc-exclusive)).
   4A's own framing of the same cutover: with RTR added, static IBLs/cube-maps
   were deleted and "no longer does any part of our lighting system consist of
   'baked', pre-generated data. It can now all be generated in real time"; and
   "we have always found this to be undesirable… the results are completely
   static" ([4A tech dive](https://www.4a-games.com.mt/4a-dna/in-depth-technical-dive-into-metro-exodus-pc-enhanced-edition)).
   The console blog extends it: "lighting was one of the biggest and most
   all-encompassing examples of the lot and Ray Tracing solves it"
   ([4A console blog](https://www.4a-games.com.mt/4a-dna/everything-technical-about-metro-exodus-upgrade-for-playstation-5-and-xbox-series-x-s)).
   Note the game never used lightmaps per se — the 2019 deck already stated
   "4A-Engine doesn't really have a concept of something static (prebaked)"; the
   removed baked artifacts are IBL cube-maps and hand-placed fake lights, not
   lightmaps.

2. **Direct lighting: still rasterized shading, with the GI ray lit by the full
   deferred stack.** DF's summary verdict: "the new Metro is not a fully
   path-traced game… but rather a hybrid renderer where global illumination,
   lighting and shadows are handled by ray tracing, while other elements of the
   game still use traditional rasterisation" — DF's characterization
   ([DF interview](https://www.digitalfoundry.net/articles/digitalfoundry-2021-metro-exodus-tech-interview)).
   4A's own mechanics are more precise: the on-screen pixels are still shaded by
   the deferred pipeline ("we use each ray as a mean of sampling the environment…"
   — the path-tracing recursion is entered only at the GI-ray hit point), where
   the hit is lit by sun + sky + "up to 256 analytic light sources… each with
   their own shadow map to provide accurate secondary occlusion," N-dot-L +
   inverse-square falloff, culled through a world-space cluster grid. So per
   frame, sun/sky reach the GI term through the traced ray's deferred hit
   lighting, while on-screen direct sun shadows remain the existing shadow-map
   path; DF observed emissive surfaces give "physically accurate direct
   shadowing from area light sources" through the traced term. Emissive
   materials in the RT lighting date from the Two Colonels DLC and are in
   Enhanced ("surfaces to also have emissive materials… that surface is then a
   large area light source in its own right") — 4A tech dive, first-party;
   corroborated by [NVIDIA driver article](https://www.nvidia.com/en-us/geforce/news/metro-exodus-enhanced-edition-game-ready-driver/)
   ("ray-traced emissive lighting, previously seen exclusively in The Two
   Colonels DLC").

3. **Bounces: one traced bounce + DDGI for "infinite" bounces.** 4A: "We have
   introduced a raytraced probe grid as part of a system known as Dynamic
   Diffuse Global Illumination (DDGI). It is based heavily on and indeed uses
   much of the same shader technology as NVIDIA's RTXGI SDK." DDGI rays are the
   same diffuse rays, lit by the same pipeline; grid probes sample the grid from
   the previous frame, "simulating the effect of rays bouncing recursively";
   the effect "adds a small and noticeable lift to dark parts of the scene that
   light would not normally penetrate to with just a single bounce"
   (4A tech dive, first-party). DF measured the cost of that design: "only the
   first light bounce is calculated in real-time, with other bounces the result
   of temporal accumulation… it took around 20 full frames for all the bounces
   to accrue with the game running at 60fps" when the system is deliberately
   gamed; occasional glowing surfaces and some noise at the low sample count
   were also noted (DF, measurement).

4. **GI ray budget: no 4A-published number.** Neither 4A blog nor the DF
   interviews state Enhanced's rays/pixel or bounce count numerically for PC.
   The one 4A-published budget change is console-side: "denoiser's frontend (its
   temporal accumulation component) had already been made scalable, allowing
   for any resolution input while still running at full resolution itself. We
   halved per-pixel ray-count and it soon became apparent… all of the major
   features we were hoping to implement could then actually be feasible"
   (4A console blog, first-party). What exists on PC is the *resolution axis*:
   the RT setting "controls the internal resolution of the global illumination"
   — Normal = "quarter of the internal resolution", High = "a checkerboarded
   rendition of your resolution", Ultra = native (DF, measurement/observation of
   shipped settings). Both sources are explicit that this is resolution scaling
   of the GI trace, not a per-pixel ray-count statement.

5. **Denoiser: 4A's own, overhauled from linear to non-linear.** 4A: "In our
   image processing pipeline, we have completely overhauled our denoising
   filters. Where our previous denoisers used a linear filtering method, blurring
   the dataset a little bit each frame… a non-linear filter now evaluates
   surrounding pixels to rapidly home in on an estimate… able to adapt to rapid
   changes in the lighting environment… better able to reconstruct details and
   in a fraction of the time of previous iterations" (4A tech dive, first-party).
   Console blog adds that the old denoiser "was of high quality but was quite
   heavy on the GPU and its cost was far from being constant," and that the
   denoiser's recurrent blur and pre-trace were tuned to console cache sizes.
   Nothing first-party names an NVIDIA-supplied denoiser for either game — the
   2019 deck and both blogs describe in-house filters only (contrast: DDGI uses
   NVIDIA RTXGI *SDK* shader tech, a fact 4A itself volunteers).

6. **Upscaling.** PC: DLSS 2.0 (DF observed DLSS 2.1 behavior) plus a 4A TAA
   upsampler for non-RTX GPUs; 4A states the internal dynamic-scaling system
   already ran the RT step "at lower than full-screen resolution… yet still
   generate an accurate lighting dataset at the end," and DLSS extends that to
   the whole pipeline (4A tech dive, first-party; DF analysis for the TAA
   upsampler and DLSS 2.1 observation). Consoles: 4A's own dynamic resolution +
   temporal upscaling targeting 60 fps: "4K at 60FPS output resolution with
   typical internal dynamic resolution hovering around 5 mega-pixels on our
   heaviest scenes (on Xbox Series X and PlayStation 5)" (4A console blog,
   first-party).

7. **Measured performance.**
   - PC, DF: "an RTX 2060 ran the game just as well as the 2019 version — and
     usually a fair lick faster: anything up to 16 percent to the better";
     CPU-bound scenes regressed ~13% on a Ryzen 5 3600. DF's article headline
     framing: Enhanced "runs at the same speed — or better — than its 2019
     counterpart" ([DF analysis](https://www.digitalfoundry.net/articles/digitalfoundry-2021-inside-metro-exodus-enhanced-edition-pc-exclusive),
     DF measurement). **No 2070/2080 Ti/3070-specific numbers exist from 4A,
     NVIDIA, or DF**; NVIDIA never published the promised Enhanced performance
     article (the launch driver article says "head back… soon to get further
     details on the game's ray tracing and NVIDIA DLSS performance"; no such
     article was found on GeForce news as of 2026-08-31 — the plausible URL
     returns 404).
   - Consoles, DF: both target 60 fps; dynamic resolution measured — Xbox Series
     X typically 1512p–1728p (rare minimums ~1080p), PlayStation 5 typically
     1296p–1512p (~80% of Series X throughput) in open-world scenes, PS5
     slightly smoother (Series X shows stutter), Taiga noted as most demanding
     ([DF PS5 vs XSX](https://www.digitalfoundry.net/articles/digitalfoundry-2021-metro-exodus-enhanced-edition-ps5-vs-xbox-series-x),
     DF measurement).
   - Console engineering envelope, 4A: "our typical 16ms frame has about 12ms of
     work on async-queue"; bottleneck balancing for Gen 9 "made a huge
     difference… (on the order of being 30% faster)… evident in the PC Enhanced
     version as well, with a typical 18% performance gain on Nvidia hardware
     compared to async-off"; first menu-screen build ran <2 MP at 60 fps and
     climbed to the 4K/5 MP state (4A console blog, first-party).
   - Console-specific RT optimizations, 4A: ray-binning (direction-coherent
     thread groups to cut divergence and memory traffic), BVH LoDs, FP16
     throughout, DSBR/DCC/VRS where the platform offers them (4A console blog,
     first-party).

## 3. Why interior sun/sky diffusion looks deep and noise-free

1. **The direct sun/sky entry into interiors resolves through geometry the
   traced term actually sees.** In 2019 the deck already showcased "GI FROM
   LIGHT SOURCES — Interiors fully lit by sun": the GI ray's deferred hit
   lighting samples the skybox for MISS rays, so sunlight streaming through
   apertures is evaluated at per-pixel ray resolution, not via a coarse
   volumetric or probe. S9985, first-party. In Enhanced this is unchanged in
   kind — sun and sky remain the two lights every GI sample gets — while the
   sampled set *grows* (256 analytic lights + emissive). 4A tech dive,
   first-party.

2. **Diffuse hemisphere integration is deliberately pre-filtered.** 4A: "All of
   the fine detail generated by PBR lighting models would be lost as the rays of
   light scatter randomly… for this purpose… N-dot-L will suffice"; hit-point
   albedo is a per-instance average chosen precisely because hemisphere
   integration is "a low-pass filter in essence… It is a good idea to pre-filter
   signal to lower denoiser's input noise level. We do that pre-filtering
   extremely aggressively" (S9985, first-party). Enhanced repeats the move at
   the material level: "their materials consist of a block, average colour for
   each object… We even avoid loading textures to reduce memory bandwidth"
   (4A tech dive, first-party). This is the same contract NRD demands of
   atlas-rt: low-variance, de-modulated radiance in, detail out of the filter.

3. **Temporal accumulation is the bounce-depth mechanism.** The "infinite
   bounce" label is a temporal construction: one traced bounce per frame, DDGI
   probes resampling their own previous frame, and the SH-encoded irradiance
   buffer accumulated over time ("Temporal accumulation always happens before
   denoising"; "These technologies act as an approximate GI term that can be
   added to the RTGI light sources themselves"). DF's 20-frames-to-converge
   measurement is the visible signature. 4A blogs + DF, as cited in §2.3/§2.5.

4. **The denoiser, not the sampler, absorbs the residual noise.** 2019: adaptive
   two-pass radius (6 m → 3 m with normal weighting), variance- and AO-scaled
   blur ("blur less in 'dark corners', i.e. multiply by AO"), blue-noise ray
   seeding, and the explicit acceptance that "perfection in image 'cleanness' is
   not needed" because TAA adds one more denoise pass. Enhanced: the non-linear
   filter rewrite specifically to make the filter faster and less dependent on
   blur, so it "adapts to rapid changes in the lighting environment" — i.e.
   interiors with moving sun or muzzle flash converge without long blur tails.
   S9985 + 4A tech dive, first-party. DF's caveat stands: at Enhanced's low
   sample count "you can see some 'noise' sometimes in the lighting" and rare
   glowing surfaces (DF).

5. **Exposure/tonemapping: nothing first-party.** Neither the S9985 deck nor
   either 4A blog nor the DF interviews state anything about exposure strategy,
   HDR calibration, or tonemapping interaction with the RTGI term. The only
   adjacent first-party datum is the deck's YCoCg HDR radiance encoding ("Y in
   range [0..HDR]"). Do not guess from the games' look; if this matters for
   atlas-rt, treat it as unmeasured (see Gaps).

## 4. Enhanced vs original cost, as published

1. **4A never published a like-for-like ms comparison between the 2019 RTGI and
   Enhanced.** The comparison points that exist: (a) DF's measurement that
   Enhanced runs same-or-faster than 2019 on RTX 2060 (§2.7) — i.e. the
   removed SSAO/IBL/legacy-GI work and the async/FP16/denoiser overhaul
   absorbed the added lights, DDGI, and RTR; (b) 4A's async-compute balancing
   claim of ~18% gain on NVIDIA PC hardware (console blog); (c) the 2019 deck's
   own ~20%/~30% (High/Ultra) frame-time overhead as the baseline the Enhanced
   pipeline had to beat (S9985). (d) Directionally, DF's interview records
   reflections as the expensive feature 4A deprioritized: "They are one of the
   most expensive features out there… ultimately, they just come second to GI
   in terms of importance to our game" (Jon Bloch, 4A, quoted by DF).
2. **Attribution warning.** Widely repeated claims that Enhanced is "fully path
   traced" originate in marketing shorthand; the on-record technical sources
   (4A's blog, DF's review) describe one traced diffuse bounce + DDGI +
   shadow-mapped analytic lighting at hit points, not multi-bounce path
   tracing. DF says it outright: "not a fully path-traced game" (DF interview).

## Sources

- Kept:
  - GTC/GDC 2019 slide deck, "Exploring Raytraced Future in Metro Exodus"
    (Shyshkovtsov, Karmalsky, Archard — 4A; Zhdan — NVIDIA): all 2019
    implementation, denoising, and performance numbers.
    https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9985-exploring-ray-traced-future-in-metro-exodus.pdf (also https://gdcvault.com/play/1026159/Exploring-the-Ray-Traced-Future)
  - 4A Games, "In-depth Technical Dive into Metro Exodus PC Enhanced Edition"
    (May 6, 2021): Enhanced feature set, 256 analytic lights at GI hit points,
    emissive materials, DDGI/RTXGI, denoiser overhaul, DLSS 2.0, baked-data
    removal, RTR.
    https://www.4a-games.com.mt/4a-dna/in-depth-technical-dive-into-metro-exodus-pc-enhanced-edition
  - 4A Games, "Everything Technical About Metro Exodus Upgrade for PlayStation 5
    and Xbox Series X|S" (June 17, 2021): console RT port — halved ray count,
    ray-binning, BVH LoDs, async envelope (12 ms of 16 ms async), 4K60 output /
    ~5 MP internal, denoiser scalability.
    https://www.4a-games.com.mt/4a-dna/everything-technical-about-metro-exodus-upgrade-for-playstation-5-and-xbox-series-x-s
  - Digital Foundry (Alex Battaglia), "Inside Metro Exodus Enhanced Edition"
    (April 28, 2021): removal of artist-placed raster lighting, RTGI resolution
    settings (quarter/checkerboard/native), 20-frame bounce accrual, noise and
    glowing-surface caveats, RTX 2060 +13–16% vs 2019, Ryzen CPU −13%, DLSS
    2.1/TAA upsampler observations.
    https://www.digitalfoundry.net/articles/digitalfoundry-2021-inside-metro-exodus-enhanced-edition-pc-exclusive
  - Digital Foundry (Alex Battaglia), "The Making of Metro Exodus Enhanced
    Edition" tech interview, part 1 with executive producer Jon Bloch (May 16,
    2021): "not a fully path-traced game" framing, fake-light removal and
    lighting rebalance, DDGI as "the only option for secondary GI", reflections
    deprioritized as most expensive.
    https://www.digitalfoundry.net/articles/digitalfoundry-2021-metro-exodus-tech-interview
  - Digital Foundry, "Metro Exodus Enhanced Edition: how does PS5 compare to
    Xbox Series X?" (June 18, 2021): measured console dynamic resolution ranges
    and 60 fps behavior.
    https://www.digitalfoundry.net/articles/digitalfoundry-2021-metro-exodus-enhanced-edition-ps5-vs-xbox-series-x
  - NVIDIA developer blog, "Ray Tracing in 4A's Metro Exodus and Remedy's
    Control" (June 17, 2019): Martin Stich's one-bounce AO+GI-same-ray
    description of the 2019 game.
    https://developer.nvidia.com/blog/ray-tracing-in-4as-metro-exodus-and-remedys-control/
  - NVIDIA developer blog, "Global Illumination in Metro Exodus: An Artist's
    Point of View" (May 14, 2019): Sergei Karmalsky quote (50x ray lengths,
    pixel-perfect detail) and pointer to the GDC-19 excerpt.
    https://developer.nvidia.com/blog/global-illumination-in-metro-exodus/
  - NVIDIA GeForce news, "Metro Exodus PC Enhanced Edition Game Ready Driver"
    (April 29, 2021): vendor-side feature list for Enhanced (every light
    contributes to GI, RT emissive from Two Colonels, DLSS 2.0), and the
    (unfulfilled) promise of a later performance article.
    https://www.nvidia.com/en-us/geforce/news/metro-exodus-enhanced-edition-game-ready-driver/
  - NVIDIA GeForce news, "Metro Exodus Enhanced With NVIDIA RTX Ray Traced
    Effects" (August 20, 2018): launch announcement of RTGI+RTAO pairing.
    https://www.nvidia.com/en-us/geforce/news/metro-exodus-rtx-ray-traced-global-illumination-ambient-occlusion/
- Dropped:
  - DSOGaming Enhanced Edition benchmark article (RTX 3070-class fps numbers) —
    third-party enthusiast outlet, outside this document's source rules; the
    acceptance criteria require 4A/NVIDIA/DF numbers only.
  - Tom's Hardware "Metro Exodus: Ray-Traced Global Illumination" Pascal RT
    testing — secondary outlet, non-RT hardware, adds nothing first-party.
  - TechSpot / GameSkinny / Geeky Gadgets / gameunion.tv Enhanced Edition
    recaps — content-farm restatements of the DF analysis and NVIDIA driver
    article.
  - Reddit r/pcgaming threads comparing Enhanced to Cyberpunk Overdrive —
    hearsay, excluded by constraint.
  - The promised-but-unpublished NVIDIA Enhanced performance article —
    `metro-exodus-enhanced-edition-performance/` returns 404; absence recorded
    in §2.7 rather than substituted.
  - GDC Vault 2019 video page as a citation for implementation numbers — the
    video content could not be re-read here; all 2019 numbers are cited to the
    verbatim slide deck instead.
- Gaps:
  - **No Enhanced-Edition conference talk exists that this search could find.**
    The GDC Vault public browse listings (2019–2026 sessions shown in full)
    contain only the 2019 talk (play/1026159); no 4A GDC/HPD/SIGGRAPH session on
    the Enhanced renderer surfaced, and the DF interview promised a second
    deep-dive ("4A's CTO Oles Shishkovstov and senior rendering programmer Ben
    Archard go into extreme depth") that does not appear under any slug in the
    Wayback CDX listing of digitalfoundry.net/eurogamer.net 2021 Metro URLs —
    it may exist only in DF video/Patreon form. If that part-2 interview is ever
    located, it is the single most likely source for Enhanced ray-count numbers.
  - **Enhanced per-pixel GI ray count (PC): absent first-party.** Only console
    "halved per-pixel ray-count" (4A) and the PC GI-resolution settings axis
    (DF) are on record.
  - **2019 trace resolution: absent first-party.** "Less than one ray per pixel"
    in the talk's framing is the AO+GI shared-ray design, not a half-res trace;
    no slide or blog states the GI buffer resolution for the 2019 game.
  - **Exposure/tonemapping strategy: absent first-party** (§3.5).
  - **Bounce count in the S9985 Q&A:** the deck says "Not limited to 1st bounce
    at all, but… Even 2nd bounce gives diminishing returns compared to cost" —
    a design comment, not a shipped-config statement for either release.
