# Voxel radiance cache: prototyped and rejected

Superseded by ADR 0019 (radiance cache re-adopted at hit lighting; accepted
2026-08-31). History: prototyped as an accumulate→resolve→stamp pipeline and
rejected on the RTX 3070 (owner verdict 2026-08-24, ticket 04); reopened
2026-08-31 by the Metro Exodus research
(docs/research-metro-exodus-enhanced-rt-gi.md).

The verdict's measured grounds — uncovered regions, stale transients,
machinery cost — were not refuted; they carry into ADR 0019 as binding
acceptance gates. Research legs:
docs/research-radiance-cache-precedents.md and
docs/research-radiance-cache-integration-cost.md.
