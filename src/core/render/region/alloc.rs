use std::sync::Arc;

use vulkano::{
    acceleration_structure::{AabbPositions, AccelerationStructure},
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
};
use vulkano_taskgraph::Id;

use crate::core::render::gpu::GpuDesc;

pub(crate) struct FreedPool {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

pub(crate) struct FreedSlab {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

pub(crate) struct FreedBlas {
    pub(crate) aabb_buffer_id: Id<Buffer>,
    pub(crate) aabb_capacity: u32,
    pub(crate) blas: Arc<AccelerationStructure>,
    pub(crate) blas_storage_size: u64,
}

#[derive(Default)]
pub(crate) struct FreeLists {
    pub(crate) pools: Vec<FreedPool>,
    pub(crate) blas: Vec<FreedBlas>,
    pub(crate) slabs: Vec<FreedSlab>,
}

#[derive(Default)]
pub(crate) struct PendingFrees {
    pub(crate) pools: Vec<FreedPool>,
    pub(crate) blas: Vec<FreedBlas>,
    pub(crate) slabs: Vec<FreedSlab>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub pool_allocations: u64,
    pub pool_reuses: u64,
    pub blas_allocations: u64,
    pub blas_reuses: u64,
    pub slab_allocations: u64,
    pub slab_reuses: u64,
    pub slab_budget_refusals: u64,
}

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

pub(crate) struct PoolAllocation {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

pub(crate) struct BlasAllocation {
    pub(crate) aabb_buffer_id: Id<Buffer>,
    pub(crate) aabb_capacity: u32,
    pub(crate) as_storage: Option<(Arc<AccelerationStructure>, u64)>,
}

pub(crate) fn allocate_pool(
    gpu: &GpuDesc,
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

pub(crate) struct SlabAllocation {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

pub(crate) const CACHE_SLAB_BUDGET: u64 = 2 * 1024 * 1024 * 1024;

// 28 B per face: 3 state words (2 packed-half u32 + a frame stamp) + 4
// accumulator words (contract.glsl's CACHE_ENTRY_STRIDE * 4, cross-checked
// by the pack contract test).
pub(crate) const CACHE_ENTRY_BYTES: u64 = 28;

pub(crate) fn slab_budget_ok(allocated: u64, needed: u64) -> bool {
    allocated + needed <= CACHE_SLAB_BUDGET
}

// The Radiance cache's per-Region slabs recycle through their own free
// list; the budget counts allocated bytes (live and retired), and a
// refusal hands back no slab — the shader's bda-0 check turns that into
// ADR 0019's per-region fallback.
pub(crate) fn allocate_slab(
    gpu: &GpuDesc,
    free: &mut FreeLists,
    stats: &mut AllocStats,
    allocated: &mut u64,
    needed: u64,
) -> Option<SlabAllocation> {
    if let Some(freed) = take_best_fit(&mut free.slabs, needed, |f| f.capacity) {
        stats.slab_reuses += 1;

        return Some(SlabAllocation {
            buffer_id: freed.buffer_id,
            capacity: freed.capacity,
        });
    }

    if !slab_budget_ok(*allocated, needed) {
        stats.slab_budget_refusals += 1;

        return None;
    }

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

    *allocated += needed;
    stats.slab_allocations += 1;

    Some(SlabAllocation {
        buffer_id,
        capacity: needed,
    })
}
pub(crate) fn allocate_blas(
    gpu: &GpuDesc,
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

        let none = take_best_fit(&mut entries, 1024, |f| f.capacity);
        assert!(none.is_none());
    }

    #[test]
    fn pending_frees_release_into_reusable_lists() {
        let mut pending = PendingFrees::default();
        pending.pools.push(FreedPool {
            buffer_id: Id::INVALID,
            capacity: 64,
        });

        let mut free = FreeLists::default();
        assert!(take_best_fit(&mut free.pools, 64, |f| f.capacity).is_none());

        free.pools.append(&mut pending.pools);
        let taken = take_best_fit(&mut free.pools, 64, |f| f.capacity).unwrap();
        assert_eq!(taken.capacity, 64);
        assert!(pending.pools.is_empty());
    }

    #[test]
    fn slab_budget_bounds_allocation() {
        assert!(slab_budget_ok(0, CACHE_SLAB_BUDGET));
        assert!(slab_budget_ok(CACHE_SLAB_BUDGET - CACHE_ENTRY_BYTES, CACHE_ENTRY_BYTES));
        assert!(!slab_budget_ok(CACHE_SLAB_BUDGET, CACHE_ENTRY_BYTES));
    }

    #[test]
    fn slab_reuse_prefers_retired_slab() {
        let mut free = FreeLists::default();
        free.slabs.push(FreedSlab {
            buffer_id: Id::INVALID,
            capacity: 1024,
        });

        let taken = take_best_fit(&mut free.slabs, 512, |s| s.capacity).unwrap();
        assert_eq!(taken.capacity, 1024);
        assert!(take_best_fit(&mut free.slabs, 2048, |s| s.capacity).is_none());
    }
}
