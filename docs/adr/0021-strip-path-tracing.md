# Strip path tracing from main

Path-traced lighting (ADRs 0007-0020) and the simpler renderer it replaced grew up on one
branch, so main's tip carried both. The effort is complete and preserved whole; main
continues as the simpler renderer underneath it.

## Status

accepted (strip decision, 2026-09-01)

## Decision

- The full path-tracing state lives on the **path-tracing** branch at 7171953, frozen.
  main's history still contains every PT commit; the branch gives the last PT-bearing
  tree a name.
- main deletes the PT subsystems going forward: the NRD denoiser and its frame history,
  the radiance cache, the BSDF transport (bounces, NEE/MIS, Russian roulette, lobe
  selection), the glass transmission walk, the material table and MATL consumption, and
  auto exposure.
- The display keeps the ACES tonemap at a fixed identity exposure (the one kept piece of
  the composite). The Background is the sky gradient without the Sun disk. Surfaces shade
  from the Palette alone.
- Kept: the core/render/app layer layout, FramePipeline (now Render then Composite), the
  residency decision module, the DDA snapping fix (0009's amendment), the Hull and Normal
  debug modes, and the true-average fps measurement.

## Consequences

- The payload is color, t, hit kind, and normal; the frame images are one color image
  plus the swapchain views. contract.glsl shrinks to the region and mode constants.
- Glass voxels paint opaque from their palette entry; transmission is a PT-era concept.
- The fixed exposure is a starting point — retune `EXPOSURE_EV` in composite.comp by eye.
- Resuming path tracing means branching from path-tracing, not reverting this strip.
