//! The destination render path (renderer-impl ticket 02): the Region
//! pipeline.
//!
//! End-to-end for the single-Region case (extended to every Region the world
//! occupies at startup): the world's voxels are emitted as Micro-chunk
//! snapshots (the minimal emitter, `snapshot`), packed into per-Region voxel
//! pools (offset table + compact blocks, `pack`), built as one procedural
//! AABB BLAS per Region with a trimmed AABB per non-empty Micro-chunk,
//! traversed by the lattice DDA in the intersection shader (materials via
//! 8-bit hitKind), and ray-passed by a per-pixel raygen with world-space f32
//! rays, traceRay Opaque | TerminateOnFirstHit, to the swapchain storage
//! images (ADR 0001/0002/0004).
//!
//! The TLAS holds one instance per Region — lattice-static transform
//! (translation by the Region origin), custom index = the 12-bit Region id,
//! mask 0xFF — and the Region table maps that id to the Region's pool device
//! address. Residency transitions, free lists, the change queue and the
//! ordered rebuild nodes are tickets 03-05; this ticket builds the static
//! lattice over the initial snapshot batch.

pub mod pack;
pub mod render;
pub mod snapshot;

pub use pack::REGION_COUNT;
