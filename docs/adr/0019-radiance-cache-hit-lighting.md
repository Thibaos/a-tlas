# Radiance cache: re-adopted at hit lighting

Interiors receive fill only through low-probability BSDF threads of the
aperture (measured: ~1/255 at calibrated exposure; transport verified
unbiased; unresolved at 64 spp and by the ~30-frame NRD window), so the
cacheless estimator cannot light them at 1 spp. Metro Exodus Enhanced ships
the mechanism (docs/research-metro-exodus-enhanced-rt-gi.md): one traced
diffuse bounce per pixel, the hit position lit by the full light stack,
recursive bounces temporal via a DDGI probe grid, residual noise absorbed by
denoiser + TAA. This ADR re-adopts the radiance cache ADR 0017 rejected —
with a different integration point, and with 0017's measured grounds as
acceptance gates.

## Status

accepted (2026-08-31, owner verdict in-session). Supersedes the Decision in
ADR 0017; 0017's rejection grounds (uncovered regions, stale transients, machinery
cost) remain binding acceptance gates. Starting material: 0017's research
legs (docs/research-radiance-cache-precedents.md,
docs/research-radiance-cache-integration-cost.md) and the Metro doc.

## Decision

- **The cache lights traced-ray hit positions; it is never stamped into the
  primary-surface radiance.** v1's accumulate→resolve→stamp measured its
  killers exactly at the stamp; 4A's shipped design keeps the primary
  surface on the direct path and feeds hit lighting from cache data.
- **Multi-bounce is temporal.** Cache samples resolve against previous-frame
  state with hysteresis; the path budget drops from 4 traced bounces to
  1–2 + cache, cutting per-frame trace work.
- **Materials feeding the cache are pre-filtered**, extending the existing
  NRD de-modulation contract; per-object average albedo is the 4A precedent.
- **Acceptance gates, measured on the 3070 harness:** an explicit
  uncovered-region mechanism (fallback to the direct BSDF estimate where
  cache confidence is low) and a transient mechanism (bounded blend; reset
  on light change, camera cut, or history reset).

## Considered Options

- Re-adopt v1's stamp integration with fixes. Rejected: uncovered regions
  and transients are properties of stamping into primary radiance, not
  implementation defects.
- Keep 0017's verdict. Rejected: interiors stay at ~1/255; neither more spp
  (64-spp test) nor the denoiser's accumulation window reaches the
  aperture-threading paths.

## Consequences

- On acceptance, ADR 0017's Decision compresses to a superseded stub
  pointing here; this thread's doc trail converges.
- Tickets own the variant space: cache layout (dense per-face vs probe grid
  in a voxel world), update/blend policy, coverage mechanism, integration
  into production.rgen, NRD interplay, and the perf budget against the
  current ~15 ms frame.
