# G-buffer and denoiser contract at transmissive primaries

The denoiser half of the transparency effort: what the Trace pass records
when the primary hit is glass, so NRD ReBLUR sees a contract it can filter
(wayfinder ticket 07, one grilling round, 2026-08-30). Consumes ADR 0013's
attribution hook and ticket 02's Primary Surface Replacement research;
leaves ADR 0007's six-buffer contract and the single
REBLUR_DIFFUSE_SPECULAR instance unchanged.

## Status

accepted (transparency-and-overrides ticket 07, 2026-08-30). Operates on
ADR 0013's transport: a glass primary's whole path rides the specular
channel with weight 1; the diffuse channel reports no-data (0 radiance,
nhd 0).

## Decision

- **Primary Surface Replacement**: at a glass primary NRD never sees the
  glass. The aux buffers describe the revealed surface B — the first
  non-glass surface the path reaches:
  - albedo_metal, normal_roughness: B's material and entered face;
  - viewZ: B's depth, the crossings' segment lengths folded into the
    accumulated primary t — no virtual-space elongation, since straight
    transmission keeps the virtual point on the real one and flat voxel
    faces make curvature corrections identically zero;
  - motion vectors: evaluated at B's world position (backward, camera
    reprojection; voxels are static);
  - in-lobe hit distance: the first scattering segment after B,
    length(C − B), normalized by B's roughness and viewZ_B; no in-lobe hit
    → 0 (the no-data convention).
- **Sky through glass is a Background pixel**: a primary path that crosses
  only glass and then misses reports sky conventions wholesale — VIEWZ_SKY,
  zero aux buffers, zero MV, and the pane-tinted radiance (Procedural sky
  plus the Sun disk, ADR 0013's amendment) in the diffuse channel,
  un-modulated. The radiance is deterministic — nothing to denoise — and
  the Composite's is_sky branch reads only the diffuse channel. The
  g-buffer's surface test becomes "the path ended on a surface", not "the
  first trace hit".
- **Disocclusion mix ships**: glass primaries are PSR pixels, and NRD's
  README requires looser disocclusion thresholds for them, per pixel on
  mixed screens. IN_DISOCCLUSION_THRESHOLD_MIX (R8 optional input; the
  CommonSettings fields are already mirrored) carries the classification:
  1.0 at glass primaries, 0.0 elsewhere, against
  disocclusionThresholdAlternate — starting at the vendored default 0.05,
  tuned by validation worlds, not decided here.
  is_disocclusion_threshold_mix_available flips on.
- **No packing pressure**: albedo_metal and normal_roughness keep their
  layout — they merely describe B. The classification is the only new
  channel and rides the dedicated R8, not the RGBA8 aux pair. The map's
  packing-pressure fog item closes on this verdict.
- **Cap fallback is opaque**: past the 32-crossing cap the glass voxel
  shades as an opaque surface (ADR 0013) and the g-buffer describes it
  normally — the pixel stops being a PSR pixel, no special denoiser case.
- **Hull mode untouched**: hull SBT hits carry no glass markers; the
  contract is voxel-mode only.

## Considered Options

- **Specular-carried tinted sky** (delta-rule uniformity: everything a
  glass primary produces rides specular). Rejected: the Composite's
  is_sky branch drops the specular channel, so it would need a composite
  special case — and the delta rule exists to route noise, of which a
  crossed-only miss has none.
- **Defer the disocclusion mix** (keep the global 0.01 threshold).
  Rejected: it bakes NRD-documented PSR ghosting into v1's validation
  baseline to save one R8 image whose binding seams all exist.
- **hitT measured from the glass entry** (the whole transmitted chain in
  the in-lobe distance). Rejected: that length is already folded into
  viewZ; double-counting breaks normHitDist's scaling — the same reason
  PSR folds mirror elongation into viewZ.
- **MV at the glass entry point**. Rejected: the pixel's temporal history
  is B's history; reprojection from the entry point pairs the radiance
  with the wrong surface.

## Consequences

- Execution: the raygen accumulates primary t across crossings and carries
  loop-local B state (hit_kind, normal, world position) to the buffer
  writes; bounce_t is recorded at the first post-B scattering segment; a
  scattered flag distinguishes crossed-only misses from behind-glass sky.
  FrameImages gains one R8 kind; NrdInputs and the descriptor set gain the
  mix bind; CommonSettings flips the availability flag.
- Behind-glass misses (after a scattering event) keep the depth>0 MIS sky
  path and ride the specular channel with the path — only crossed-only
  misses are Background pixels.
- CONTEXT.md gains Revealed surface; Background gains the through-glass
  clause.
