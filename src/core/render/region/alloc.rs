use std::sync::Arc;

use anyhow::Context;
use vulkano::{
    acceleration_structure::{AabbPositions, AccelerationStructure},
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
};
use vulkano_taskgraph::Id;

use crate::core::render::gpu::GpuDesc;

pub struct FreedPool {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

pub struct FreedBlas {
    pub(crate) aabb_buffer_id: Id<Buffer>,
    pub(crate) aabb_capacity: u32,
    pub(crate) blas: Arc<AccelerationStructure>,
    pub(crate) blas_storage_size: u64,
}

#[derive(Default)]
pub struct FreeLists {
    pub(crate) pools: Vec<FreedPool>,
    pub(crate) blas: Vec<FreedBlas>,
}

#[derive(Default)]
pub struct PendingFrees {
    pub(crate) pools: Vec<FreedPool>,
    pub(crate) blas: Vec<FreedBlas>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub pool_allocations: u64,
    pub pool_reuses: u64,
    pub blas_allocations: u64,
    pub blas_reuses: u64,
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

pub struct PoolAllocation {
    pub(crate) buffer_id: Id<Buffer>,
    pub(crate) capacity: u64,
}

pub struct BlasAllocation {
    pub(crate) aabb_buffer_id: Id<Buffer>,
    pub(crate) aabb_capacity: u32,
    pub(crate) as_storage: Option<(Arc<AccelerationStructure>, u64)>,
}

pub fn allocate_pool(
    gpu: &GpuDesc,
    free: &mut FreeLists,
    stats: &mut AllocStats,
    needed: u64,
) -> anyhow::Result<PoolAllocation> {
    if let Some(freed) = take_best_fit(&mut free.pools, needed, |f| f.capacity) {
        stats.pool_reuses = stats.pool_reuses.saturating_add(1);

        Ok(PoolAllocation {
            buffer_id: freed.buffer_id,
            capacity: freed.capacity,
        })
    } else {
        let buffer_id = gpu.resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u8]>(needed).context("device layout creation failed")?,
        )?;

        stats.pool_allocations = stats.pool_allocations.saturating_add(1);

        Ok(PoolAllocation {
            buffer_id,
            capacity: needed,
        })
    }
}

pub fn allocate_blas(
    gpu: &GpuDesc,
    free: &mut FreeLists,
    stats: &mut AllocStats,
    aabb_count: u32,
) -> anyhow::Result<BlasAllocation> {
    if let Some(freed) = take_best_fit(&mut free.blas, u64::from(aabb_count), |f| {
        u64::from(f.aabb_capacity)
    }) {
        stats.blas_reuses = stats.blas_reuses.strict_add(1);

        Ok(BlasAllocation {
            aabb_buffer_id: freed.aabb_buffer_id,
            aabb_capacity: freed.aabb_capacity,
            as_storage: Some((freed.blas, freed.blas_storage_size)),
        })
    } else {
        let aabb_buffer_id = gpu.resources.create_buffer(
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
            DeviceLayout::new_unsized::<[AabbPositions]>(u64::from(aabb_count))
                .context("device layout creation failed")?,
        )?;

        stats.blas_allocations = stats.blas_allocations.saturating_add(1);

        Ok(BlasAllocation {
            aabb_buffer_id,
            aabb_capacity: aabb_count,
            as_storage: None,
        })
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
}
