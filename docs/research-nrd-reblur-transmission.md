# NRD ReBLUR and transmissive hits

How glass pixels should feed our REBLUR_DIFFUSE_SPECULAR instance (NRD
v4.17.3, vendored under third_party/nrd). Researched for wayfinder ticket
[02](../.scratch/transparency-and-overrides/issues/02-nrd-reblur-transmission-guidance.md);
grounds the G-buffer/denoiser contract decision (ticket 07).

## Headline

NRD has **no native transparency mode** — its own README opens with "not
natively designed for volumetrics or transparency". Two application-side
conventions exist:

1. **Primary Surface Replacement (PSR)** — the documented mechanism for
   delta-event chains (pure mirrors, refractions): NRD never sees the glass;
   it sees the first non-delta surface behind it, with the g-buffer rewritten
   into *virtual world space*.
2. A **"denoising-free" glass path** in the NRD sample (SHARC radiance cache +
   reprojection of the already-denoised frame + dithering, polished by
   TAA/upscaler). Not applicable here: we run no TAA/upscaler, and SHARC is a
   separate effort-scale dependency.

PSR is the fit. The README's own assessment: "*PSR* is perfect for flat
mirrors" — and every voxel face is exactly that. Straight (non-refractive)
transmission collapses the hardest parts of the contract:

- No bending ⇒ the virtual position lies **on** the view vector at the real
  revealed point (`Bvirtual = A + viewVector · |B − A|` degenerates to `B`).
  Virtual space == world space for transmitted rays.
- Flat faces ⇒ curvature corrections are identically zero (the README's
  "potentially adjusted several times by curvature" clauses all vanish).

## The PSR contract, mapped onto our six buffers (ADR 0007)

For a camera ray hitting glass at `A`, passing through (possibly several
glass voxels), reaching the first opaque surface `B`, then scattering:

| Buffer | Content |
|---|---|
| albedo_metal | material of `B` |
| normal_roughness | normal of `B`'s face (virtual == real here) + **roughness at `B`** |
| viewZ | viewZ of `B` (no elongation needed without bending) |
| mv | motion of `B`'s point (our backward-pointing pixel/frame MVs, evaluated at `B`) |
| diff/spec radiance `.a` | in-lobe hitT measured **from `B`** (`length(C − B)` for next hit `C`), not from `A` |

The primary→`B` segment is deliberately excluded from hitT: its length is
already folded into viewZ, keeping REBLUR's `normHitDist` scaling coherent
(the same trick PSR uses for mirror elongation).

Channel attribution: a transmitted segment is a **specular (delta) event**, so
see-through radiance rides the specular Beauty buffer regardless of `B`'s
material — identical to how mirror reflections behave today. `B`'s roughness
drives filter breadth: smooth glass yields near-mirror statistics (tight
filter, firefly-prone highlights), rough `_rough` glass widens it naturally.

## Disocclusion handling

Glass pixels are PSR pixels; README warns their disocclusion logic needs
looser thresholds than opaque pixels, and mixing both on one screen requires
per-pixel control. v4.17.3 ships exactly that:

- `CommonSettings::disocclusionThreshold` (default 0.01) and
  `disocclusionThresholdAlternate` ([0.02; 0.2]) mixed per-pixel by
  `IN_DISOCCLUSION_THRESHOLD_MIX` (R8+, optional input on
  REBLUR_DIFFUSE_SPECULAR) when `isDisocclusionThresholdMixAvailable = true`
  (Include/NRDDescriptors.h, Include/NRDSettings.h).
- Our raygen already knows which primaries are glass ⇒ writing the mix
  texture is one more RGBA8 output or a constant-per-classification value.

## Shadows (hooks for ticket 06)

SIGMA_SHADOW_TRANSLUCENCY denoises **stochastic translucent shadows**
(pseudo-translucency, LDR [0;1]; packed via `SIGMA_FrontEnd_PackTranslucency`,
distance-to-occluder + rgb translucency). It is a separate denoiser instance
(~0.45 ms @1440p RTX 4080 per README). If ticket 06 picks stochastic shadow
transmission, this is the supported denoising route; opaque-blocker or
analytic-attenuation policies need no new instance.

## Lobe selection / noise interplay

Nothing new: probabilistic diffuse/specular selection at the primary hit is
the README's recommended configuration (AREA_3X3 hit-distance reconstruction,
which we already run); its blue-noise/temporal-jitter guidance applies
unchanged. Transmission adds delta events, i.e. more variance into the
specular channel — the firefly-policy conversation stays on the path-tracing
map (ticket 10 there).

## Implementation implications (ticket 07's menu)

- Raygen classifies each primary as opaque or glass-delta; glass pixels emit
  PSR-bookkept aux data (revealed-surface material/normal/viewZ/MV, hitT
  from the reveal point). For straight transmission this is loop-local state
  the path tracer already has when it commits through voxels — no payload
  growth beyond what traversal (ticket 08) needs anyway.
- One extra optional bind: IN_DISOCCLUSION_THRESHOLD_MIX (or fold the 0/1
  classification into an existing spare channel if one survives ticket 07's
  packing review).
- The Reference tracer mirrors the same bookkeeping; the shading diff
  compares final radiances, so PSR conventions need no validator changes
  beyond matching the convention itself.
- Refraction (the map's fog item) would reuse this exact machinery plus
  virtual-space elongation/curvature — the go/no-go can price it precisely.

## Sources

- third_party/nrd/README.md — "not natively designed for…" (l.11), PSR
  section (ll. 601–639) incl. virtual-space rules and the flat-mirror
  endorsement, blue-noise/lobe notes (ll. 588–597), SIGMA translucency
  (ll. 877–925), perf table (l. 955).
- third_party/nrd/Include/NRDDescriptors.h — IN_DISOCCLUSION_THRESHOLD_MIX,
  REBLUR_DIFFUSE_SPECULAR optional inputs; Include/NRDSettings.h —
  disocclusionThreshold(Alternate), isDisocclusionThresholdMixAvailable.
- third_party/nrd/Shaders/NRD.hlsli — SIGMA_FrontEnd_PackTranslucency.
- NVIDIA, "Rendering perfect reflections and refractions in path-traced
  games" (PSR origin), linked from README l. 603:
  https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/
- NRD-Sample (simplex branch) — the "denoising-free" glass path referenced by
  README l. 11: https://github.com/NVIDIA-RTX/NRD-Sample
