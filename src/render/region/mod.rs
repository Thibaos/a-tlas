//! The Region pipeline: the renderer's region-shaped half of the lattice —
//! feeding (the input contract's worker side), packing, residency, ordered
//! rebuilds, and the per-frame render task. See `world::snapshot` for the
//! producer side of the contract.

pub mod alloc;
pub mod feed;
pub mod pack;
pub mod rebuild;
pub mod residency;
pub mod task;
