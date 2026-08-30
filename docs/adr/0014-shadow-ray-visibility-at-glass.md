# Shadow-ray visibility at glass

The NEE half of the transparency effort: what a shadow ray reports when its
segment to the light crosses glass. Decided with the owner (ticket 06, one
grilling round, 2026-08-30) on top of ADR 0013's R = 0 transport and ticket
05's rendered preference; ADR 0011's MIS is untouched.

## Status

accepted (transparency-and-overrides ticket 06, 2026-08-30). Consumes ADR
0013's Pane law; leaves ADR 0011 untouched — no MIS weight moves.

## Decision

- **Attenuated**: shadow visibility is the product of the crossed Panes' T
  (ADR 0013's law) along the shadow segment. With R = 0 and binary glass
  this is exactly the transmittance the v1 transport pays over the same
  segment, so shadow visibility and bounce transport are consistent by
  construction — the "biased approximation" framing belonged to the Fresnel
  model, and with no reflected fraction there is nothing to ignore. No RNG
  draw, no new noise into ReBLUR's input, colored shadows.
- **Both analytic lights**: the Sun's delta and the Procedural sky's env
  samples take the same visibility; no light-type branch.
- **The origin run counts**: a shading face that opens directly into glass
  pays that Pane's T at its exit, mirroring the camera-run rule — a floor
  flush under glass receives direct sun attenuated exactly once, matching
  the transport path that pays T entering the same Pane.
- **The 32-crossing cap applies**: past 32 Panes on a shadow segment the
  visibility is 0 — shadow rays and transport fail dark together.
- **Emissive voxels unchanged**: no NEE exists for them; their light behind
  glass reaches surfaces only via transport's pane product (ADR 0013).

## Considered Options

- **Opaque blockers**. Rejected: dark shadows under glass contradict the
  transmitted look the transport delivers; the owner rejected it in ticket
  05's renders.
- **Stochastic pass-through**. Rejected: injects Bernoulli noise into shadow
  visibility at 1 spp for a bias the R = 0 model does not even have.
- **Sun-only attenuation**. Rejected: sky NEE through a glass roof would
  treat glass as opaque and black out interiors outside the sunbeam, for no
  saved cost — the same segment walk serves both lights.
- **Exempt the origin run**. Rejected: a face opening into glass would get
  un-attenuated direct light while transport pays T entering that Pane.
- **Amend ADR 0011**. Rejected: the ticket's own condition — amend if MIS
  weights shift — is not met; visibility multiplies the estimate, no pdf
  changes.

## Consequences

- The shadow trace (`shadowed()`, shaders/region/production.rgen) becomes a
  transmittance walk. Today it is a binary test under `gl_RayFlagsOpaqueEXT`
  (any-hit disabled) and the payload is opaque to intersection shaders, so
  per-crossing accumulation constrains ticket 08's mechanism: an any-hit
  route drops the Opaque flag for shadow rays and needs the committed
  voxel's material in the any-hit; commit-and-continue needs recursion. The
  interface is fixed regardless of route: shadow visibility = Π Pane T.
- Ticket 07 inherits settled branch bookkeeping context: the primary-miss
  branch fires at depth 0 regardless of crossing count (the Sun disk shows
  through glass, tinted — amended into ADR 0013 during this round).
- Ticket 10's validation gains: colored shadows under a glass slab (Sun and
  sky), the flush-floor origin run, mixed-index stacking (T_A·T_B
  symmetric), and a >32-Pane world failing dark on both estimators.
- CONTEXT.md: the NEE entry carries the attenuation clause.
