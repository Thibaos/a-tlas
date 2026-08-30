# Voxel radiance cache: prototyped and rejected

A per-face GPU radiance cache — dense slabs beside the voxel pool, per-frame
accumulation with an EMA blend, a resolve dispatch, and a debug irradiance
render mode — was charted as its own wayfinder effort, researched (two legs:
precedents and integration cost), and prototyped to a working
accumulate→resolve→stamp pipeline measured on the RTX 3070. The owner
judged the result and rejected it; all prototype code was reverted to HEAD.

## Status

rejected (voxel-radiance-cache ticket 04, owner verdict 2026-08-24; code
reverted to HEAD). The research legs remain tracked:
docs/research-radiance-cache-precedents.md and
docs/research-radiance-cache-integration-cost.md.

## Decision

- **No face-radiance cache in the renderer.** The cacheless path-tracing
  estimator with NRD ReBLUR stays the lighting architecture.
- Grounds of the rejection: the uncovered-region budget (faces reached by
  few or no cached samples), stale transients under camera motion, and the
  cost/complexity of the accumulate→resolve→stamp machinery not justified
  by what it bought at 1 spp.
- Alternatives recorded during research: DDGI-style probes rejected early —
  their machinery fixes a probe-position problem exact DDA hits do not
  have; SHaRC-style spatial hashing was only a size fallback for the dense
  design, and fell with it.

## Considered Options

- Ship the prototype behind a debug Render mode. Rejected with the
  prototype: the same grounds apply, and keeping the machinery alive
  invites drift from the production trace pass contract.
- Leave the design parked without a verdict. Rejected: the prototype
  measurements were enough to settle it; an open question would outlive
  its evidence.

## Consequences

- No cache plumbing exists under src/core/render; nothing to maintain or
  tear out.
- A future radiance-cache effort starts from the two research docs and
  this verdict, not from the reverted code.
