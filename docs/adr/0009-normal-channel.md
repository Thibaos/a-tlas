# Surface normal: entered-face reconstruction in closest-hit

Shading (BRDF eval, NEE cosine, the BSDF's reflection) needs the geometric
surface normal at a hit. The DDA intersection shader knows it exactly — the
last Amanatides-Woo step axis, the face the ray crossed into the committed
voxel — but the intersection shader cannot communicate it to the closest hit
through the ray pipeline. This ADR records the channel and the convention.

## Status

accepted (path-tracing ticket 04, 2026-08-17)

## Decision

- **Channel**: a `vec3 normal` field in the ray payload (`MainPassPayload`,
  object space — equal to world space up to the translation instance
  transform). Zero on miss; only read when `t > 0`.
- **Reconstruction**: the closest hit recovers the entered face from the hit
  point. `reportIntersectionEXT` carries only `(t, 8-bit hitKind)` and the
  payload is opaque to intersection shaders, so the DDA's exact step axis
  cannot be passed. But the reported `t` is the cell-entry boundary crossing
  time, so `p = origin + dir * t` lies on the entered face: `p[a]` is an
  integer (within epsilon) exactly on the crossed axis and strictly interior
  on the others. The closest hit scans x, y, z for the first axis with
  `fract(p[a]) < eps` or `> 1 - eps` and a non-zero direction component
  (a parallel axis never crosses a boundary, even if `p[a]` sits on one by
  coincidence), then orients the outward normal by the direction's sign:
  `normal = -sign(dir[face]) * e_face`. The epsilon is relative — ~8 ULP of
  `|p[a]|` (`8 * 2^-23 * max(|p[a]|, 1)`) — because the reconstruction error
  is the f32 rounding of the DDA's t and of `p` itself, which grows with the
  coordinate magnitude (ULP ~ 1e-6 near the origin, ~ 2.4e-4 at p = 2000);
  a fixed epsilon would either miss far-away boundaries or swallow
  near-corner entries nearby. 8 ULP gives 4x margin over the ~2 ULP worst
  case while keeping the false-positive window on non-crossed axes a
  fraction of a thousandth of a cell.
- **Tie-break**: edge/corner entries (`p` on two or three boundaries) break
  to the first axis in x, y, z order — the DDA's own preference order. The
  true last-step face at a corner depends on the occupancy of intermediate
  cells (unknowable without re-marching), so a canonical order is the
  contract; near-corner float ties (a boundary crossed a hair before the
  commit) resolve the same way — deterministic, and a benign misattribution
  at grazing silhouettes.
- **Camera embedded in a voxel**: the t_min commit crosses no face (`p` is
  interior) — the normal is the camera-facing direction (`-normalize(dir)`).
  A camera exactly on a face degrades to the face normal of the enclosing
  cell, which is also correct-looking.
- **Normal heatmap**: a new debug Render mode (`Normal` = 4) paints the
  payload normal, -1..1 mapped to 0..1 per channel — faces color by axis
  (x red, y green, z blue; + side bright, - side dark), background gray.
  It traces the DDA hit group like Voxel; the mode is a per-frame push
  constant, so the TAB cycle gains one entry, never a pipeline rebuild.

## Considered Options

- **Side-channel the face from the DDA** — rejected: the payload is opaque
  to intersection shaders and hitKind's 8 bits are all consumed by the
  material index. A per-pixel scratch buffer cannot index secondary rays
  (ticket 05 bounces), and nudging the reported `t` to encode the face would
  pollute the depth/hitT consumers (viewZ, NRD de-modulation) and fight the
  validator's t comparison.
- **Slab-based face derivation in closest-hit** — rejected as an
  implementation: it needs the *entered* cell, which itself requires an
  epsilon nudge to disambiguate (a high-face entry makes `floor(p)` land one
  cell past), replacing one epsilon with two.
- **Largest-|dir| face heuristic** — rejected: not the entered face (a ray
  can enter through a small-|dir| face); the map's contract is the entered
  face, never a neighbor average.

## Consequences

- The payload grows one `vec3` (24 → 36 bytes, padded to 40); the capture
  validator compares {color, t} only, and the closest hit's added ALU
  changes no written byte — verified byte-identical across the full suite
  (stash + rebuild + rerun, all 18 gpu.png hashes equal).
- The production raygen's Voxel mode now writes the real octahedrally
  encoded normal and the Material's linear roughness into the
  normal+roughness auxiliary buffer (ADR 0007's "04 (normals)" stub note),
  plus the heatmap mode for visible verification.
- Ticket 05 shades with `payload.normal` at every bounce; ticket 07's CPU
  path tracer reports the face exactly (its DDA knows the step axis) and
  must implement the same canonical tie-break — the two agree except at
  float corner ties.
