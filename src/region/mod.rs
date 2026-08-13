//! The destination render path: the Region
//! pipeline, its input contract, and multi-region residency.
//!
//! End-to-end for every Region the world occupies: the world hands the
//! renderer Micro-chunk snapshots through the input contract; the mirrors are
//! packed into per-Region voxel pools (offset table + compact blocks,
//! `pack`); the residency manager owns the full
//! static lattice (Region = 256^3 voxels, 4096 Regions, 12-bit ids), the
//! free lists, the lattice-static instance set, and the stable TLAS — a
//! Region becomes Resident on its first non-empty Micro-chunk (pool buffer +
//! procedural AABB BLAS) and leaves on its last, with freed memory reused
//! only after the rebuild that dropped the referencing instance executed.
//! Each Region is one procedural AABB BLAS with a trimmed AABB per non-empty
//! Micro-chunk, traversed by the lattice DDA in the intersection shader
//! (materials via 8-bit hitKind), and ray-passed by a per-pixel raygen with
//! world-space f32 rays, traceRay Opaque (deliberately NOT
//! TerminateOnFirstHit — the first found hit is not the closest; see
//! shaders/region/production.rgen), to the swapchain storage images.

pub mod input;
pub mod pack;
pub mod rebuild;
pub mod render;
pub mod residency;
pub mod snapshot;
