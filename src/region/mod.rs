//! The destination render path (renderer-impl tickets 02/03): the Region
//! pipeline and its input contract.
//!
//! End-to-end for every Region the world occupies: the world hands the
//! renderer Micro-chunk snapshots through the input contract (`input`,
//! ticket 03: enqueue-only `submit_microchunk` / `submit_batch`, a worker
//! drain into per-Region CPU mirrors, condvar-sleep idle); the mirrors are
//! packed into per-Region voxel pools (offset table + compact blocks,
//! `pack`); each Region becomes one procedural AABB BLAS with a trimmed AABB
//! per non-empty Micro-chunk, traversed by the lattice DDA in the
//! intersection shader (materials via 8-bit hitKind), and ray-passed by a
//! per-pixel raygen with world-space f32 rays, traceRay Opaque |
//! TerminateOnFirstHit, to the swapchain storage images (ADR 0001/0002/0004).
//!
//! The TLAS holds one instance per Region — lattice-static transform
//! (translation by the Region origin), custom index = the 12-bit Region id,
//! mask 0xFF — and the Region table maps that id to the Region's pool device
//! address. The GPU-side residency/free-list and ordered-rebuild work are
//! tickets 04/05; the worker here keeps only the CPU-side drain/pack.

pub mod input;
pub mod pack;
pub mod render;
pub mod snapshot;

pub use pack::REGION_COUNT;
