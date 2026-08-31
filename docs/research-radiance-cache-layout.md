# Radiance cache layout: dense per-face vs DDGI probe grid (ticket 01)

> Research request: the cache's physical shape for ADR 0019's hit-lighting
> integration in a sparse voxel world — dense per-exposed-face storage beside
> the voxel pool, a DDGI-style free-space probe grid, or a hybrid — judged
> against the two measured killers of the v1 prototype (ADR 0017): uncovered
> regions and transient staleness under camera and sun motion.
> Starting material: docs/research-radiance-cache-precedents.md,
> docs/research-radiance-cache-integration-cost.md,
> docs/research-metro-exodus-enhanced-rt-gi.md, ADR 0017/0019,
> .scratch/radiance-cache/map.md, shaders/region/production.rgen.
> Verification date: 2026-08-31. New claims checked against the DDGI papers
> (Majercik et al., JCGT 2019; production extensions, JCGT 2021, read in
> full), the RTXGI-DDGI SDK docs and shaders at
> github.com/NVIDIAGameWorks/RTXGI-DDGI (main, 2026-08), and the SHaRC
> integration guide at github.com/NVIDIA-RTX/SHARC (read in full). Primary
> sources only; unverifiable items are in Gaps, not the body. The 2019 DDGI
> paper's own PDF could not be fetched (137 MB); its claims here are limited
> to the verified abstract, with mechanics cited to the 2021 paper and the
> shipped RTXGI shaders instead.

## Executive answer

**Dense per-face.** The DDA hit already produces the exact key (voxel, face)
and the exact hemisphere bound (the face normal), so the cache entry is the
cosine-weighted integral DDGI spends eight probes, two octahedral atlases,
Chebyshev visibility, and a self-shadow bias to approximate. Every RTXGI
artifact control — relocation, classification, view/normal bias, wall
thickness rules — exists because the query point is not the data. In a sparse
voxel world of 1-voxel-thick walls and interior rooms those controls are
unmeetable, and the probe grid's own documented costs (scroll-in planes
"must reconverge"; "an unavoidable minimum latency when lighting changes
occur" — RTXGI's words) land directly on the two v1 killers the gates are
written against.

The probe grid's update policy is the part worth keeping verbatim, and it is
fully specified in primary sources: old-weight hysteresis with event-driven
reduction, gamma-5.0 perception encoding, impulse clamping. Those numbers
feed ticket 02. SHaRC's spatial hash stays as the escape hatch if dense
storage blows the region-allocator cap at 1024³-class worlds; that is a
substitution for candidate 1, not a hybrid. A hybrid (per-face + camera
probe volume) has no remaining job: diffuse fill only, no volumetrics, no
dynamic geometry, and the out-of-scope list already bars RTXGI SDK adoption.

## Question

The map locks integration at hit lighting (never stamped into primary
radiance), temporal multi-bounce with hysteresis, a path budget dropping
4 → 1–2 + cache, and pre-filtered inputs. What is left open is storage: where
the previous-frame state lives and what the read and write look like inside
`shaders/region/production.rgen`'s loop (primary + ≤4 bounces, NEE/MIS over
delta sun and procedural sky, 1 spp, NRD ReBLUR consuming de-modulated
diffuse, ~15 ms frame on the RTX 3070). Candidates:

1. **Dense per-face.** One entry per exposed voxel face, keyed by global
   voxel coord + face axis. Value: RGBA16F irradiance + age/confidence.
   Direct O(1) index off the DDA hit; slabs owned by the existing Region
   lifecycle (integration-cost doc, finding 8). This is v1's structure with
   ADR 0019's integration point — the structure is not what was rejected;
   the stamp was.
2. **DDGI probe grid.** Regular 3D grid of free-space probes, each an
   octahedral 8×8 irradiance + 16×16 visibility map, updated per frame by
   ~256 traced rays/probe with hysteresis, queried by trilinear interpolation
   over the 8 surrounding probes with visibility weighting.
3. **SHaRC-style spatial hash.** Surface-keyed `(position, normal)` hashed
   into logarithmic grid levels; 64-bit key + 16 B accumulation + 16 B
   resolved data; Update/Resolve/Query passes. The middle shape: keyed to
   surfaces like candidate 1, machinery-heavy like candidate 2.
4. **Hybrid** (per-face + a camera-following probe volume), addressed and
   rejected in the Recommendation.

## Candidates, as verified

### DDGI probe grid (RTXGI)

- **Storage.** Six texture arrays per volume: ray data, irradiance, distance,
  probe data (offsets + states), variability ×2. Octahedral probes with
  1-texel borders for hardware bilinear. The production paper selects 8×8
  irradiance and 16×16 visibility per probe for bandwidth, SIMD fit, and
  occupancy (Majercik et al. 2021, §7.2, Table 2).
- **Update.** Per probe: trace ~256 spherical-Fibonacci rays (RTXGI default
  `probeNumRays = 256`), shade them against the *previous frame's* probe
  grid (that is where recursive bounces come from), then blend per texel:
  `lerp(new, old, hysteresis)`. RTXGI's `probeHysteresis` defaults to 0.97;
  its own comment says values "closer to 0.9 or lower will rapidly react to
  scene changes, but will exhibit flickering" (DDGIVolume.h). The 2021 paper
  reduces hysteresis on change: per texel, change > 25% of max → −0.15,
  > 80% → 0.0; scene events — small light change → −15% for 4 frames, large
  light or object change → −50% for 10 frames (§4.3). Irradiance is stored
  through a gamma-5.0 perception curve so light-to-dark transitions read as
  linear drops (§4.2; RTXGI `probeIrradianceEncodingGamma = 5.f`), with a
  0.10 brightness threshold blocking single-texel impulses (DDGIVolume.h).
- **Query.** Bias the shaded point along normal and view ray, loop the 8
  surrounding probes: trilinear weight, a wrap-shading backface weight
  `(dot+1)²/4 + 0.2`, a Chebyshev visibility weight against the probe's mean
  distance/variance floored at 0.05, weight crushing below 0.2, then
  bilinear-filtered octahedral irradiance samples decoded from the gamma
  curve (RTXGI Irradiance.hlsl, read in full). This is the per-shaded-point
  cost probes add that candidate 1 does not.
- **Placement machinery.** Probes that land in geometry: relocation shifts
  them up to 45% of a grid cell based on backface-hit ratios; classification
  disables useless probes using 32 fixed rays (states Off/Sleeping/Awake/
  Vigilant in the 2021 paper, §6, worth 30–50% update time). Infinite
  scrolling volumes follow the camera by leapfrogging edge planes; the docs
  are explicit that scrolled-in planes are "invalidated and must reconverge"
  while interior probes persist (DDGIVolume.md, Volume Movement).
- **Content rules.** "We recommend a probe every 2–3 meters"; walls must have
  thickness "proportional to the probe density" or light leaks; sparse grids
  often beat dense ones because dense grids "can reveal the structure of the
  probe grid" (DDGIVolume.md, Rules of Thumb).
- **Admitted limits.** "Irradiance temporally accumulates in probes, so
  there is an unavoidable minimum latency when lighting changes occur";
  low-frequency signal only; "probe data storage can become memory intensive
  in large environments" (RTXGI Algorithms.md, Limitations). The 2021 paper
  still reports ghosting for small bright lights after all its heuristics
  (§8.1). Some probe texels "may never fully converge"; RTXGI tracks a
  variability statistic so the app can pause updates when the field settles.

### Dense per-face

Structure per the precedents and integration-cost docs; the new work here is
checking it against ADR 0019's gates under hit-lighting integration:

- **Read.** One indexed fetch at the payload choke point (`payload.hit_kind`
  / `payload.normal` in production.rgen). No interpolation code, no bias
  terms, no probe-index arithmetic. The face bounds the incoming hemisphere,
  so a scalar RGB entry is the exact cosine-weighted integral for that
  surface, not an octahedral approximation of it.
- **Write.** The same hit blends ≤ 2 packed-u32 atomics. Traffic ~11 GB/s
  raw at 1080p×4 bounces is fine; atomic packing is the thing to benchmark
  (integration-cost doc, findings 5–7).
- **Propagation.** Because the cache lights traced-ray hit positions, the
  existing path traffic *is* the propagation pass: bounce rays landing on
  cached faces pick up previous-frame fill, and their continuations write new
  faces, one hop per frame. This is the same 1-hop-per-frame recursion 4A
  get from probes resampling their own grid (metro doc §2.3), achieved with
  zero extra dispatches. DDGI instead pays a dedicated ~256-ray/probe trace
  pass; per-face reuses the 8–15M hits/frame the renderer already pays for.
- **State.** Entries are world-anchored. Camera motion creates no
  invalidation, no reconvergence, no scroll bookkeeping. Stale entries in a
  static world with a static sun are correct fill, not artifacts; the
  transient case that matters is sun motion, which is global and
  event-shaped.

### SHaRC spatial hash

Verified against the SHaRC integration guide: three passes (Update — a
modified path trace over a small pixel fraction, e.g. 1-in-5×5 ≈ 4%;
Resolve — compute pass combining accumulation with previous-frame data and
evicting stale entries; Query — early path termination on cache hits), 40 B
per voxel (8 + 16 + 16), 2²² elements as baseline = 160 MiB fixed, occupancy
10–20% at a static camera. Two features map directly onto this effort's
gates: `staleFrameNumMax` eviction (an aging mechanism exactly like
per-face recency) and a "responsive lighting" mode that treats transient
lights separately because "standard SHaRC accumulation may react too slowly"
(SHaRC Integration.md). It is the strongest surface-keyed published design
and the right fallback — but in this engine it re-derives, with hash
collisions, barriers, and a log-level grid, a key the DDA hands over
exactly, and it inserts two extra passes with UAV barriers that the
single-rgen-file renderer does not currently have.

## Judgement

### Uncovered regions (v1 killer #1)

- **Probes** cover space by construction: any point inside a volume has 8
  surrounding probes (Irradiance.hlsl). But data coverage is not light
  coverage. An interior probe's rays must still thread the same 1-voxel
  window apertures to see sun or sky, and a camera-scrolling volume ships
  new, unlit planes exactly when the player enters a room — the moment the
  interior fill is on screen. Probes also cannot express per-surface
  confidence: the query blends 8 probes and the variance is averaged into
  the Chebyshev floor, so "this face has never seen a sample" is not a
  thing the data structure knows.
- **Per-face** makes uncovered a per-entry property with a direct reading:
  the age stamp. A face never hit has no entry, and ADR 0019's gate —
  fall back to the direct BSDF estimate where confidence is low — is a
  compare on that stamp. Coverage is driven by where paths land: faces
  visible through a window sliver are hit every frame by every pixel on the
  sliver; faces deeper in the room accumulate only through aperture-threading
  bounce paths, which is the same physics the probe rays face, accumulated
  over the same frames. Camera motion uncovers nothing because entries do
  not move.
- **Verdict: per-face, clearly.** The gate 0019 demands is native to the
  structure.

### Transient staleness (v1 killer #2)

- **Probes** carry two lag sources: the blend lag every temporal cache has
  (RTXGI's "unavoidable minimum latency") and scroll reconvergence, which is
  a *camera*-motion transient — the exact shape of v1's second killer. The
  2021 paper's §4.3 heuristics exist to fight the lighting-change case and
  still leave flashlight ghosting on the table (§8.1).
- **Per-face** has the same blend lag, no more. Its transient inventory is
  short: the sun moves (global, one event, reset or hysteresis reduction
  handles every entry at once — RTXGI's `OnGlobalLightChange` /
  `OnSmallLightChange` event hooks are the shipped pattern for exactly
  this); a region streams in (per-region cold start); a cut resets history
  (NRD already does). Camera motion is free. There are no dynamic objects
  in the estimator's world, so the per-object heuristics DDGI needs have
  nothing to attach to.
- **Verdict: per-face.** Same blend physics, minus the camera-motion
  component, with the event-driven mitigation transferable from the probe
  literature at no structural cost.

### Geometric fit

- **Per-face**: the hit *is* the key. No bleed-through is possible because
  no data is read from anywhere but the face it was sampled on. A 1-voxel
  wall is not a leak hazard; it is six exact entries. The production paper's
  own gloss (§4.4) confirms probes store the hemisphere integral as an
  approximation; a face stores it exactly.
- **Probes**: every mitigation feature is an answer to "the query point is
  not the data": relocation, classification, normal/view bias (defaults 0.1
  each in DDGIVolume.h), the self-shadow bias (2021 §4.1, default 0.3,
  backface rays zeroed and depth shortened 80%), Chebyshev weighting,
  wall-thickness content rules. RTXGI's 2–3 m spacing against 1-voxel walls
  is the leak configuration the docs warn about; matching wall thickness to
  probe density would force probe spacing toward voxel scale and blow up
  probe counts (their own storage note). Interiors with window apertures
  maximize all of it.
- **Verdict: per-face, decisively.** This is the same call the precedents
  doc reached, and the 0019 re-adoption did not change it — it changed what
  the cache is allowed to touch.

### Integration cost

- **Per-face**: append a blend at the existing choke point and a fetch in
  the loop. No new dispatches, no barriers, no octahedral code, no
  relocation/classification/scrolling passes, no second RT invocation.
  Sits inside the single-shader-file constraint.
- **Probes**: a probe-trace dispatch (256 rays/probe — at a modest 32³-probe
  volume that is 8.4M extra trace rays/frame) plus two blending dispatches,
  relocation, classification, border copies, and variability reduction,
  before the query path adds an 8-probe weighted loop with two texture
  samples each to every shaded hit. ADR 0017's third measured ground —
  machinery cost — recurs in full.
- **SHaRC**: two extra passes with UAV barriers and hash atomics; designed
  for engines that lack an exact spatial key.
- **Verdict: per-face cheapest, SHaRC middle, probes heaviest.**

### Perf plausibility against ~15 ms

- **Per-face** is net-negative on trace work: the locked 4 → 1–2 bounce drop
  cuts 2–3× of the frame's `traceRayEXT` volume (integration-cost finding
  12) and shorter paths converge faster under ReBLUR. The added cost is
  blend traffic (~11 GB/s raw, before warp aggregation) and the atomic
  throughput risk, both bounded by packing. Storage: ~0.4–1.2 GB
  visible-only at 512³-class worlds against 4–6 GB headroom; a hard slab cap
  in the region allocator, with per-region fallback, keeps the worst case
  bounded. This is the only candidate that *reduces* frame time.
- **Probes** are purely additive on top of today's frame: probe rays,
  blending, and query weight on every hit. Metro Exodus Enhanced shipped
  DDGI on a 2060, but as a *replacement* inside a rebuilt pipeline with GI
  traced internally at quarter resolution (DF, metro doc §2.4), not as an
  additional pass over an existing 15 ms budget. Update frequency can be
  amortized (DDGIVolume.md permits scheduling below frame rate), but every
  amortization step buys latency, and latency is killer #2. No first-party
  additive-cost number for a probe grid exists anywhere in the sources.
- **Memory** is the one axis probes and SHaRC win (tens to ~160 MiB fixed
  vs ~1 GB dense). At 1024³-class worlds dense needs residency gating, and
  if the cap proves unacceptable the SHaRC hash is the pre-verified
  substitute at 40 B/entry.
- **Verdict: per-face plausible and self-funding; probes unproven under
  this budget.** [INFERENCE: probe-volume sizing and additive cost on a
  3070 are estimates; no shipped voxel-world DDGI was found in any source.]

## Recommendation

Candidate 1, dense per-face, under ADR 0019's hit-lighting contract:

- Key: global voxel coord + face axis, slabs allocated per Micro-chunk/Region
  by the existing residency lifecycle, hard-capped in the allocator.
- Entry: RGBA16F irradiance + u32 age/confidence stamp, blended by ≤ 2
  packed-u32 atomics at the DDA hit.
- Read: direct fetch at traced-ray hit positions inside the production.rgen
  loop; shaded-only reuse first (integration-cost option A), termination
  with MIS weight later (option B).
- Reject the hybrid: a camera probe volume adds a second cache whose only
  theoretical advantage — queries in free space — has no consumer in this
  renderer (diffuse fill only, no volumetrics), and its costs land on both
  gates. Reject probes as primary; steal their update policy. Reject SHaRC
  as primary; keep it named as the substitute if the storage cap fails on
  the 3070 harness.

Evidence is thin on exactly one axis: nothing in the sources measures a
dense per-face cache or a probe grid added to an existing 15 ms voxel path
tracer. The choice is made on mechanism, not measurement, and the boring
option wins on both: per-face is the shape with no interpolation artifacts
to debug, no placement machinery to port, and a direct implementation of
both acceptance gates. Say this plainly in the Answer: the recommendation
is the conservative one, and ticket 03's instrumentation is what converts
it from defensible to measured.

## Implications for tickets 02–04

- **02 (blend policy).** The probe literature supplies verified starting
  numbers; adopt them rather than re-deriving: old-weight hysteresis near
  RTXGI's 0.97 default for stability, reduced on events (−15% for 4 frames
  on small light changes, −50% for 10 on large ones — sun motion is the
  large case here); per-entry change thresholds 0.25/0.8 with reductions
  −0.15/→0; a brightness impulse clamp (~0.10) so a single bright sample
  cannot jump a texel; gamma-5.0 perception encoding on stored irradiance,
  which also buys dynamic range in RGBA16F. Note the paper's caveat that a
  denoiser's own accumulation lowers the tolerable base hysteresis — ReBLUR
  already smooths ~30 frames (map.md), so start lower than 0.97 and let the
  transient gate tune it.
- **03 (coverage + transients).** Uncovered-region fallback is a per-entry
  compare on the age stamp, falling back to the direct BSDF estimate where
  the entry is absent or stale beyond threshold. Transient handling: global
  frame-stamp invalidation on sun change and history reset (the
  `OnGlobalLightChange` pattern), per-region cold-start fill on stream-in.
  No scroll reconvergence mechanism exists to build — camera motion needs
  nothing. Instrument unique-touched-face counts to validate the
  integration-cost doc's estimated hit rates.
- **04 (rgen integration).** Read at the payload choke point; the face
  entry stores hemisphere irradiance pre-modulation, so the cache-supplied
  term reaches ReBLUR already de-modulated (multiply by albedo/π at the
  hit, keep the direct term's demodulation untouched). Respect the
  standing estimator rules: cosine-weighted cached irradiance replaces only
  the diffuse lobe (specular stays traced, LOBE_P on the primary
  unchanged), NEE branches untouched, and option-B termination changes the
  path PDF, so it waits for MIS rework. Emissive coverage rides the
  all-transport entry as decided in the integration-cost doc.

## Sources

Kept:

- Majercik, Guertin, Nowrouzezahrai, McGuire, "Dynamic Diffuse Global
  Illumination with Ray-Traced Irradiance Fields", JCGT 8(2), 2019 —
  canonical definition; abstract verified (moment-based visibility
  interpolant, compact irradiance-field encoding).
  https://jcgt.org/published/0008/02/01/ (full PDF unfetchable at 137 MB;
  mechanics cited to the 2021 paper and RTXGI instead)
- Majercik, Marrs, Spjut, McGuire, "Scaling Probe-Based Real-Time Dynamic
  Global Illumination for Production", JCGT 10(2), 2021 — read in full:
  8×8/16×16 octahedral encodings (§7.2, Table 2), hysteresis update and
  per-texel/event reduction heuristics (§2.3, §4.3), gamma-5.0 encoding
  (§4.2), self-shadow bias and backface handling (§4.1), probe-position
  optimizer (§5), probe states and 30–50% sleeping savings (§6–7.1),
  tracking-window leapfrog (§7.3), flashlight ghosting limitation (§8.1).
  https://jcgt.org/published/0010/02/01/paper.pdf
- RTXGI-DDGI SDK, docs/Algorithms.md — stated limitations (temporal latency,
  low-frequency signal, memory). https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/docs/Algorithms.md
- RTXGI-DDGI SDK, docs/DDGIVolume.md — six texture arrays, octahedral
  borders, infinite scrolling ("invalidated and must reconverge"),
  relocation (45% of grid cell), classification (32 fixed rays), probe
  variability, rules of thumb (2–3 m probes, wall thickness, dense-grid
  structure reveal). https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/docs/DDGIVolume.md
- RTXGI-DDGI SDK, rtxgi-sdk/shaders/ddgi/Irradiance.hlsl — full query path:
  8-probe trilinear, wrap-shading weight, Chebyshev visibility floor 0.05,
  weight crushing, gamma decode, R10G10B10A2 energy correction.
  https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/rtxgi-sdk/shaders/ddgi/Irradiance.hlsl
- RTXGI-DDGI SDK, rtxgi-sdk/include/rtxgi/ddgi/DDGIVolume.h — defaults
  verified: probeNumRays 256, probeHysteresis 0.97 (with the flicker
  comment), encoding gamma 5, thresholds 0.25/0.10, backface thresholds
  0.1/0.25, view/normal bias 0.1, light-change event hooks.
  https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/rtxgi-sdk/include/rtxgi/ddgi/DDGIVolume.h
- NVIDIA-RTX/SHARC docs/Integration.md — read in full: Update/Resolve/Query
  passes, 40 B/voxel (8+16+16), 2²² baseline = 160 MiB, ~4% update fraction,
  stale eviction, responsive-lighting mode, occupancy norms.
  https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md
- In-repo: shaders/region/production.rgen (path loop constants and payload
  contract), docs/adr/0017-voxel-radiance-cache-rejected.md and
  docs/adr/0019-radiance-cache-hit-lighting.md (gates), docs/research-radiance-cache-precedents.md
  and docs/research-radiance-cache-integration-cost.md (structure, storage
  math, atomic packing, Region-lifecycle slabs),
  docs/research-metro-exodus-enhanced-rt-gi.md (4A's shipped DDGI design,
  quarter-res GI trace, 20-frame bounce accrual, thick-wall content),
  .scratch/radiance-cache/map.md (locked decisions), CONTEXT.md
  (Micro-chunk, Region, Radiance cache).

Dropped:

- The 2019 DDGI paper's full PDF — fetch failed on size; nothing cited from
  it beyond the verified landing-page abstract.
- DDGI follow-up literature (ADGI, adaptive probe placement) — refinements
  of a placement problem the per-face key dissolves.
- RTXGI planar probes / RTXGI v2 SDK (SHaRC integration) — planar probe
  variants target thin-wall content via a different probe class; the
  surface-keyed need is already served by SHaRC and, here, by exact faces.
- Web search for additional postmortems — providers unavailable this
  session; the 4A record is already fully mined in the metro doc (no
  Enhanced-Edition conference talk exists).

Gaps:

- No measured additive cost for a DDGI probe volume on a 3070 under this
  frame budget exists in any source; the perf judgement for probes rests on
  structure (additive vs self-funding), not numbers.
- Per-face atomic throughput and unique-touched-face counts remain estimates
  (integration-cost doc gaps); ticket 03's instrumentation pins them.
- Probe-grid settings 4A actually shipped (spacing, rays/probe, volume
  count) were never published; the metro doc records the absence.
- The 2019 paper's specifics (moment-based visibility internals, original
  hysteresis defaults) remain verified only at abstract level; all mechanics
  cited here come from the 2021 paper and shipped RTXGI code.
