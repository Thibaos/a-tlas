//! The destination render path (renderer-impl tickets 02-04): the Region
//! pipeline, its input contract, and multi-region residency.
//!
//! End-to-end for every Region the world occupies: the world hands the
//! renderer Micro-chunk snapshots through the input contract (`input`,
//! ticket 03: enqueue-only `submit_microchunk` / `submit_batch`, a worker
//! drain into per-Region CPU mirrors, condvar-sleep idle); the mirrors are
//! packed into per-Region voxel pools (offset table + compact blocks,
//! `pack`); the residency manager (`residency`, ticket 04) owns the full
//! static lattice (Region = 256^3 voxels, 4096 Regions, 12-bit ids), the
//! free lists, the lattice-static instance set, and the stable TLAS — a
//! Region becomes Resident on its first non-empty Micro-chunk (pool buffer +
//! procedural AABB BLAS) and leaves on its last, with freed memory reused
//! only after the rebuild that dropped the referencing instance executed.
//! Each Region is one procedural AABB BLAS with a trimmed AABB per non-empty
//! Micro-chunk, traversed by the lattice DDA in the intersection shader
//! (materials via 8-bit hitKind), and ray-passed by a per-pixel raygen with
//! world-space f32 rays, traceRay Opaque | TerminateOnFirstHit, to the
//! swapchain storage images (ADR 0001/0002/0004).
//!
//! The ticket-04 rebuilds ran synchronously (execute + wait_idle per step)
//! between frames; ticket 05 turns them into **ordered taskgraph nodes**
//! (`rebuild`: pool upload → BLAS build → TLAS build on residency
//! transitions — no back-AS double buffer, no flip atomic), with per-node
//! GPU timestamps feeding ticket 07's measurement. The worker keeps only
//! the CPU-side drain/pack.

pub mod input;
pub mod pack;
pub mod rebuild;
pub mod render;
pub mod residency;
pub mod snapshot;
