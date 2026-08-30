# Transmission traversal: the raygen crossing walk

How a path physically continues through glass on the GPU: where the crossing
loop lives and how the pane law (ADR 0013) is detected per hit. Decided with
the owner (ticket 08, two grilling rounds, 2026-08-30) on top of ADR 0013's
transport, ADR 0014's shadow visibility, and ADR 0015's g-buffer contract;
grounded in the production raygen loop (ADR 0010) and the DDA intersection
shader's commit semantics.

## Status

accepted (transparency-and-overrides ticket 08, 2026-08-30). Mechanism for
ADR 0013's transport and ADR 0014's shadow walk; ADR 0015 consumes its
outputs at glass primaries. ADR 0010's loop structure and the pipeline
recursion depth are unchanged.
Corrected 2026-08-30 (ticket 10): the table grows only the glass flag (ADR
0012, revised by ticket 09 — tint is the Palette color; no strength, no
IOR); the any-hit rejection was re-examined by the owner and stands.

## Decision

- **The crossing loop lives in the raygen.** On a committed glass hit (the
  material table's glass flag, ADR 0012/0013's table growth) the raygen
  multiplies throughput by the pane's T, advances the origin one voxel past
  the crossed cell, and re-traces in the same direction. intersect.rint and
  closest_hit.rchit are untouched; the recursion depth stays 1
  (src/core/render/region/task.rs); the payload is unchanged — all walk
  state is raygen locals; crossings draw no RNG (T is deterministic).
- **Pane detection is raygen-local successor comparison.** The walk tracks
  the open pane as (index, expected successor cell); the successor is the
  Amanatides-Woo next cell, computed during the advance as the
  exit-boundary minimum. A committed glass voxel is mid-pane iff it is
  exactly that successor with the same index; otherwise it is a new pane
  entry and pays T. World-space cells make this exact across micro-chunk
  and region boundaries, with no pool lookups.
- **The advance.** From the entry point p (the division-form snapped t of
  intersect.rint keeps p on the entered face), the origin moves to
  p + d·(t_exit + ε), ε ≈ 1e-3 in t. A mid-pane re-entry commits the
  successor voxel at t ≈ t_min with no entered face; its reconstructed
  normal is unused — glass crossings never shade.
- **Interior camera.** The camera's own voxel commits at t ≈ t_min with no
  crossed face; its run pays T at the first observed crossing.
  Product-identical to ADR 0013's "at exit" phrasing — the ADR text stands;
  this is the mechanism reading.
- **Shadow rays run the same walk.** shadowed() loops: trace → glass →
  visibility ×= T (same pane rule; a flush-floor origin run pays at its
  first observed crossing) → advance → re-trace; an opaque hit → 0; a miss
  → ΠT; past 32 panes on the segment → 0. One pane-law implementation
  serves both estimators, light-agnostic; gl_RayFlagsOpaqueEXT stays
  everywhere.
- **Caps unchanged**: 32 crossings per path for transport (ADR 0013), 32
  panes per shadow segment (ADR 0014). Past the transport cap the current
  glass voxel is already the committed hit — it shades opaque (fail-dark)
  and the g-buffer describes it normally (ADR 0015).
- **Hull mode untouched**: the glass branch is voxel-mode only.
- **Debug modes keep seeing glass**: glass voxels commit, so the Normal and
  palette debug paints show them — an any-hit route would have made glass
  invisible to every voxel-mode trace.

## Considered Options

- **Any-hit accumulation** (drop gl_RayFlagsOpaqueEXT, ignore glass in the
  traversal, accumulate ΠT in the payload): one trace per segment and no
  restarts, but the dropped flag taxes every candidate of every voxel-mode
  trace even in glass-free worlds; ~16 B of payload growth; the pane test
  needs cross-region pool lookups or payload state with an invocation-order
  risk at region boundaries; the cap fallback needs stop-ignoring logic;
  debug modes lose glass. The documented fallback if glass-heavy validation
  worlds show the walk's cost — not v1.
- **Closest-hit recursion**: needs max_pipeline_ray_recursion_depth 33+
  against 1 today. Rejected on hardware grounds.
- **rint run-skipping** (march through same-index runs, report at run
  boundaries): one trace per pane segment and no new stage, but the
  single-u32 report channel cannot carry pane continuation across
  invocations — a run exiting an AABB and a run ending at an empty gap
  report identically — forcing disambiguation lookups, open-run-exit
  reporting, and cross-region behind-cell lookups inside the DDA. Rejected
  on complexity.

## Consequences

- Execution touches only production.rgen: the glass-advance loop and the
  shadowed() walk; the material table gains the glass flag alone (ADR 0012
  revised — tint is the Palette color, no strength, no IOR). Nothing else
  in the pipeline, SBT, payload, or shader set moves.
- **Accepted discretization**: a cell grazed in less than ε of t can be
  skipped by the advance (a sliver pinhole) — measure-zero, the same
  epsilon-guard class as BOUNCE_OFFSET and COS_EPS; recorded here, not
  special-cased.
- The cost profile is per-voxel traces bounded by the caps: a k-voxel pane
  costs k+1 transport traces and k+1 shadow-walk traces per crossing
  segment (up to 5 shadow segments per path).
- Ticket 10's validation mirrors these semantics; the ticket-05 prototype's
  run-level walk (glass_runs / shadow_transmittance) is the semantic
  equivalent on boxes. The camera-inside-glass and >32-crossing test worlds
  exercise the origin run and both caps.
- CONTEXT.md gains Crossing.
