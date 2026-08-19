//! Allocation: the free lists a Region's pool buffer and (AABB buffer +
//! BLAS storage) pair are drawn from. Memory freed by a change cycle is not
//! reusable until the rebuild that dropped its referencing TLAS instance
//! executes — see `residency`.

use std::sync::Arc;

use vulkano::{
    acceleration_structure::{AabbPositions, AccelerationStructure},
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
};
use vulkano_taskgraph::Id;

use crate::core::gpu::GpuStack;

/// A freed pool buffer awaiting reuse (capacity = allocation size).
pub(crate) struct FreedPool {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

/// A freed (AABB buffer + BLAS storage) pair awaiting reuse.
pub(crate) struct FreedBlas {
    pub(crate) aabb_buffer_id: Id<Buffer>,
    pub(crate) aabb_capacity: u32,
    pub(crate) blas: Arc<AccelerationStructure>,
    pub(crate) blas_storage_size: u64,
}

/// The reusable free lists: memory whose referencing TLAS instance was
/// dropped by an executed rebuild.
#[derive(Default)]
pub(crate) struct FreeLists {
    pub(crate) pools: Vec<FreedPool>,
    pub(crate) blas: Vec<FreedBlas>,
}

/// Memory freed by the current change cycle, not yet reusable: the rebuild
/// that dropped the referencing instance must execute.
#[derive(Default)]
pub(crate) struct PendingFrees {
    pub(crate) pools: Vec<FreedPool>,
    pub(crate) blas: Vec<FreedBlas>,
}

/// Allocation probes (harness/tests): fresh allocations vs free-list reuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub pool_allocations: u64,
    pub pool_reuses: u64,
    pub blas_allocations: u64,
    pub blas_reuses: u64,
}

/// Best-fit allocation: removes the smallest entry with capacity ≥ `needed`
/// (pure — unit-tested). `None` means allocate fresh.
fn take_best_fit<T>(entries: &mut Vec<T>, needed: u64, capacity: impl Fn(&T) -> u64) -> Option<T> {
    let mut best: Option<(usize, u64)> = None;
    for (i, entry) in entries.iter().enumerate() {
        let cap = capacity(entry);
        if cap >= needed && best.is_none_or(|(_, c)| cap < c) {
            best = Some((i, cap));
        }
    }
    best.map(|(i, _)| entries.swap_remove(i))
}

/// The pool allocation for one Region (fresh or free-list reused).
pub(crate) struct PoolAllocation {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

/// The BLAS allocation for one Region: the AABB build-input buffer plus,
/// when reused, the existing AS storage to build into.
pub(crate) struct BlasAllocation {
    pub(crate) aabb_buffer_id: Id<Buffer>,
    pub(crate) aabb_capacity: u32,
    /// `Some((as, storage_size))` when reused from the free list (build in
    /// place); `None` when fresh (create storage + build).
    pub(crate) as_storage: Option<(Arc<AccelerationStructure>, u64)>,
}

/// Allocates a pool buffer (best-fit from the free lists, else fresh).
pub(crate) fn allocate_pool(
    gpu: &GpuStack,
    free: &mut FreeLists,
    stats: &mut AllocStats,
    needed: u64,
) -> PoolAllocation {
    if let Some(freed) = take_best_fit(&mut free.pools, needed, |f| f.capacity) {
        stats.pool_reuses += 1;
        PoolAllocation {
            buffer_id: freed.buffer_id,
            capacity: freed.capacity,
        }
    } else {
        let buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[u8]>(needed).unwrap(),
            )
            .unwrap();
        stats.pool_allocations += 1;
        PoolAllocation {
            buffer_id,
            capacity: needed,
        }
    }
}

/// Allocates a (AABB buffer, BLAS storage) pair (best-fit reuse first).
pub(crate) fn allocate_blas(
    gpu: &GpuStack,
    free: &mut FreeLists,
    stats: &mut AllocStats,
    aabb_count: u32,
) -> BlasAllocation {
    if let Some(freed) = take_best_fit(&mut free.blas, aabb_count as u64, |f| {
        f.aabb_capacity as u64
    }) {
        stats.blas_reuses += 1;
        BlasAllocation {
            aabb_buffer_id: freed.aabb_buffer_id,
            aabb_capacity: freed.aabb_capacity,
            as_storage: Some((freed.blas, freed.blas_storage_size)),
        }
    } else {
        let aabb_buffer_id = gpu
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    // STORAGE_BUFFER lets the debug Hull shader read the
                    // trimmed hulls back as a buffer_reference; the DDA
                    // path (AS build input + device address) is unchanged.
                    usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[AabbPositions]>(aabb_count as u64).unwrap(),
            )
            .unwrap();
        stats.blas_allocations += 1;
        BlasAllocation {
            aabb_buffer_id,
            aabb_capacity: aabb_count,
            as_storage: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Best-fit: the smallest entry with enough capacity wins; `None` when
    /// nothing fits.
    #[test]
    fn free_list_best_fit() {
        let mut entries = vec![
            FreedPool {
                buffer_id: Id::INVALID,
                capacity: 64,
            },
            FreedPool {
                buffer_id: Id::INVALID,
                capacity: 256,
            },
            FreedPool {
                buffer_id: Id::INVALID,
                capacity: 128,
            },
        ];
        let taken = take_best_fit(&mut entries, 100, |f| f.capacity).unwrap();
        assert_eq!(taken.capacity, 128);
        assert_eq!(entries.len(), 2);

        // Nothing fits → None (allocate fresh).
        let none = take_best_fit(&mut entries, 1024, |f| f.capacity);
        assert!(none.is_none());
    }

    /// Pending frees are not reusable until released (the ordering invariant
    /// at the list level: allocation only sees the released lists).
    #[test]
    fn pending_frees_release_into_reusable_lists() {
        let mut pending = PendingFrees::default();
        pending.pools.push(FreedPool {
            buffer_id: Id::INVALID,
            capacity: 64,
        });

        // Not yet released: allocation cannot see it.
        let mut free = FreeLists::default();
        assert!(take_best_fit(&mut free.pools, 64, |f| f.capacity).is_none());

        // Release (after the dropping rebuild executed) → reusable.
        free.pools.append(&mut pending.pools);
        let taken = take_best_fit(&mut free.pools, 64, |f| f.capacity).unwrap();
        assert_eq!(taken.capacity, 64);
        assert!(pending.pools.is_empty());
    }
}
