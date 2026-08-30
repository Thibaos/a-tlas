# Refraction: no-go for straight-transmission glass

The transparency effort (ADRs 0012–0016) shipped straight transmission as
its floor and carried refraction as a go/no-go fog item. The verdict, taken
with the owner against the ticket-05 prototype's bent-render evidence,
closes the question for the foreseeable renderer.

## Status

rejected (transparency-and-overrides ticket 11, 2026-08-30). Straight
transmission — no ray bending — is the end state; a future refraction
effort must redraw that effort's destination from scratch, not resume it.

## Decision

- **No-go.** Glass transmits straight (ADR 0013); no Snell bending, no TIR
  branches, and no IOR consumption — MATL `_ri` stays the recorded future
  source (ADR 0012).
- Evidence (the prototype's bent-render group,
  .scratch/transparency-and-overrides/prototype-absorption/renders/):
  plane-parallel panes cancel bending — bent rays shift sub-voxel with
  seams at pane borders; lensing survives only at non-parallel face stacks
  (box corners); TIR bands appear at high IOR; everything deterministic.
- The owner's verdict: the effect is real but skippable against its cost —
  re-opening ADR 0015's virtual==real and zero-curvature degeneracies,
  adding Snell/TIR branches, and reworking the 32-crossing cap reach.

## Considered Options

- Go (bent transmission in v1). Rejected on the evidence: the dominant
  visual case — flat glass panes — shows almost nothing, while the
  denoiser contract, the traversal, and the cap all pay.
- Defer without a verdict. Rejected: the fog item would outlive its
  evidence; the prototype renders settled it cheaply.

## Consequences

- ADR 0013's transport has no refraction branch to reserve for; ADR 0016's
  crossing walk and ADR 0015's g-buffer degeneracies stand as final for
  this destination.
- A future refraction effort redraws the transparency destination: it
  re-opens ADR 0015's degeneracy guarantees and the pane law, and owns the
  IOR story end to end.
