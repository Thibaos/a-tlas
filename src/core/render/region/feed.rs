use std::{
    collections::HashMap,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};

use anyhow::{Context, bail};
use glam::IVec3;

use crate::core::{
    render::region::pack::{RegionData, pack_region},
    world::{
        grid::{assert_region_index_in_lattice, region_index_of},
        snapshot::MicroChunkSnapshot,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionMirror {
    region_index: IVec3,
    microchunks: HashMap<IVec3, MicroChunkSnapshot>,
}

impl RegionMirror {
    pub fn new(region_index: IVec3) -> Self {
        Self {
            region_index,
            microchunks: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.microchunks.is_empty()
    }

    #[cfg(test)]
    pub fn microchunk(&self, global_coords: IVec3) -> Option<&MicroChunkSnapshot> {
        self.microchunks.get(&global_coords)
    }

    pub fn apply(&mut self, snapshot: MicroChunkSnapshot) -> bool {
        if snapshot.occupied_count() == 0 {
            return self.microchunks.remove(&snapshot.global_coords).is_some();
        }
        match self.microchunks.get(&snapshot.global_coords) {
            Some(previous) if *previous == snapshot => false,
            _ => {
                self.microchunks.insert(snapshot.global_coords, snapshot);
                true
            }
        }
    }

    pub fn pack(&self) -> anyhow::Result<RegionData> {
        debug_assert!(!self.is_empty(), "packing an empty mirror");

        let snapshots: Vec<&MicroChunkSnapshot> = self.microchunks.values().collect();

        pack_region(self.region_index, &snapshots)
    }
}

struct ChangeQueueInner {
    pending: Mutex<HashMap<IVec3, MicroChunkSnapshot>>,
    wake_worker: Condvar,
    wake_renderer: Condvar,
    mirrors: Mutex<HashMap<IVec3, RegionMirror>>,
    applied_regions: Mutex<Vec<IVec3>>,
    idle: AtomicBool,
    busy: AtomicBool,
    shutdown: AtomicBool,
}

impl ChangeQueueInner {
    fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            wake_worker: Condvar::new(),
            wake_renderer: Condvar::new(),
            mirrors: Mutex::new(HashMap::new()),
            applied_regions: Mutex::new(Vec::new()),
            idle: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
        }
    }
}

#[derive(Clone)]
pub struct ChangeQueue {
    inner: Arc<ChangeQueueInner>,
}

impl ChangeQueue {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(ChangeQueueInner::new()),
        }
    }

    #[cfg(test)]
    pub fn submit_microchunk(&self, snapshot: MicroChunkSnapshot) {
        assert_region_index_in_lattice(region_index_of(snapshot.global_coords));
        self.inner
            .pending
            .lock()
            .unwrap()
            .insert(snapshot.global_coords, snapshot);
        self.inner.wake_worker.notify_all();
    }

    pub fn submit_batch<I>(&self, snapshots: I) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = MicroChunkSnapshot>,
    {
        let validated: Vec<MicroChunkSnapshot> = snapshots
            .into_iter()
            .inspect(|snapshot| {
                assert_region_index_in_lattice(region_index_of(snapshot.global_coords));
            })
            .collect();
        {
            let Ok(mut pending) = self.inner.pending.lock() else {
                bail!("pending lock poisoned");
            };

            for snapshot in validated {
                pending.insert(snapshot.global_coords, snapshot);
            }
        }

        self.inner.wake_worker.notify_all();

        Ok(())
    }

    #[cfg(test)]
    pub fn pending_count(&self) -> usize {
        self.inner.pending.lock().unwrap().len()
    }

    #[cfg(test)]
    pub(crate) fn drain_pending(&self) -> Vec<MicroChunkSnapshot> {
        let mut pending = self.inner.pending.lock().unwrap();
        std::mem::take(&mut *pending).into_values().collect()
    }

    pub(crate) fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.inner.wake_worker.notify_all();
    }
}

pub struct RendererInput {
    queue: ChangeQueue,
    worker: Option<JoinHandle<anyhow::Result<()>>>,
}

impl RendererInput {
    pub fn new() -> anyhow::Result<Self> {
        let queue = ChangeQueue::new();
        let inner = queue.inner.clone();
        let worker = std::thread::Builder::new()
            .name("region-input-worker".to_string())
            .spawn(move || worker_loop(&inner))?;

        Ok(Self {
            queue,
            worker: Some(worker),
        })
    }

    #[cfg(test)]
    pub fn change_queue(&self) -> ChangeQueue {
        self.queue.clone()
    }

    #[cfg(test)]
    pub fn submit_microchunk(&self, snapshot: MicroChunkSnapshot) {
        self.queue.submit_microchunk(snapshot);
    }

    pub fn submit_batch<I>(&self, snapshots: I) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = MicroChunkSnapshot>,
    {
        self.queue.submit_batch(snapshots)
    }

    pub fn wait_until_idle(&self) -> anyhow::Result<()> {
        let Ok(mut pending) = self.queue.inner.pending.lock() else {
            bail!("pending lock poisoned");
        };

        while !pending.is_empty() || self.queue.inner.busy.load(Ordering::SeqCst) {
            pending = match self.queue.inner.wake_renderer.wait(pending) {
                Ok(next) => next,
                Err(_) => bail!("pending lock poisoned; the input worker panicked"),
            };
        }

        Ok(())
    }

    pub(in crate::core::render) fn take_dirty_regions(&self) -> Vec<IVec3> {
        let Ok(mut regions) = self.queue.inner.applied_regions.lock() else {
            return vec![];
        };

        let mut dirty = std::mem::take(&mut *regions);
        dirty.sort_unstable_by_key(IVec3::to_array);

        dirty
    }

    pub fn packed_region(&self, region_index: IVec3) -> anyhow::Result<Option<RegionData>> {
        let Ok(mirrors) = self.queue.inner.mirrors.lock() else {
            bail!("mirrors lock poisoned");
        };

        Ok(mirrors
            .get(&region_index)
            .map(RegionMirror::pack)
            .context(format!("region {region_index} has no mirror"))?
            .ok())
    }

    pub fn packed_regions(&self) -> anyhow::Result<Vec<RegionData>> {
        let Ok(mirrors) = self.queue.inner.mirrors.lock() else {
            bail!("mirrors lock poisoned");
        };

        let mut regions: Vec<RegionData> = mirrors
            .values()
            .map(RegionMirror::pack)
            .collect::<anyhow::Result<_>>()?;

        regions.sort_unstable_by_key(RegionData::region_id);

        Ok(regions)
    }

    #[cfg(test)]
    pub fn region_count(&self) -> usize {
        self.queue.inner.mirrors.lock().unwrap().len()
    }

    #[cfg(test)]
    pub fn worker_idle(&self) -> bool {
        self.queue.inner.idle.load(Ordering::SeqCst)
    }
}

impl Drop for RendererInput {
    fn drop(&mut self) {
        self.queue.shutdown();
        if let Some(worker) = self.worker.take()
            && let Err(e) = worker.join()
        {
            eprintln!("{e:?}");
        }
    }
}

fn worker_loop(inner: &Arc<ChangeQueueInner>) -> anyhow::Result<()> {
    loop {
        let taken = {
            let Ok(mut pending) = inner.pending.lock() else {
                bail!("pending lock poisoned");
            };

            loop {
                if !pending.is_empty() {
                    break;
                }

                if inner.shutdown.load(Ordering::SeqCst) {
                    bail!("worker shutdown");
                }

                inner.idle.store(true, Ordering::SeqCst);

                let Ok(next) = inner.wake_worker.wait(pending) else {
                    bail!("pending lock poisoned");
                };

                pending = next;

                inner.idle.store(false, Ordering::SeqCst);
            }

            inner.busy.store(true, Ordering::SeqCst);

            std::mem::take(&mut *pending)
                .into_values()
                .collect::<Vec<MicroChunkSnapshot>>()
        };

        let dirty = {
            let Ok(mut mirrors) = inner.mirrors.lock() else {
                bail!("mirrors lock poisoned");
            };

            apply_snapshots(&mut mirrors, taken)
        };

        {
            let Ok(mut applied) = inner.applied_regions.lock() else {
                bail!("applied regions lock poisoned");
            };

            for region in dirty {
                if !applied.contains(&region) {
                    applied.push(region);
                }
            }
        }

        {
            if inner.pending.lock().is_err() {
                bail!("pending lock poisoned");
            }

            inner.busy.store(false, Ordering::SeqCst);
        }

        inner.wake_renderer.notify_all();
    }
}

pub fn apply_snapshots(
    mirrors: &mut HashMap<IVec3, RegionMirror>,
    snapshots: Vec<MicroChunkSnapshot>,
) -> Vec<IVec3> {
    let mut dirty: Vec<IVec3> = Vec::new();

    for snapshot in snapshots {
        let region_index = region_index_of(snapshot.global_coords);
        let mirror = mirrors
            .entry(region_index)
            .or_insert_with(|| RegionMirror::new(region_index));

        if mirror.apply(snapshot) && !dirty.contains(&region_index) {
            dirty.push(region_index);
        }
    }

    mirrors.retain(|_, mirror| !mirror.is_empty());
    dirty.sort_unstable_by_key(IVec3::to_array);

    dirty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        render::region::pack::pack_regions,
        world::{
            World,
            grid::{MICRO_CHUNK_LENGTH, region_index_of},
            snapshot::emit_snapshots,
        },
    };

    fn snapshot(coords: IVec3, cells: &[(u32, u8)]) -> MicroChunkSnapshot {
        let mut mask = [0u8; 64];
        let mut materials = Vec::new();
        let mut cells: Vec<_> = cells.iter().copied().collect();
        cells.sort_unstable_by_key(|&(idx, _)| idx);
        for (idx, material) in cells {
            mask[(idx / 8) as usize] |= 1 << (idx % 8);
            materials.push(material);
        }
        debug_assert_eq!(
            materials.len(),
            mask.iter().map(|b| b.count_ones() as usize).sum()
        );
        MicroChunkSnapshot {
            global_coords: coords,
            mask,
            materials,
        }
    }

    fn zero(coords: IVec3) -> MicroChunkSnapshot {
        MicroChunkSnapshot {
            global_coords: coords,
            mask: [0u8; 64],
            materials: Vec::new(),
        }
    }

    #[test]
    fn pending_set_coalesces_last_wins() {
        let queue = ChangeQueue::new();
        let coords = IVec3::new(8, 0, 0);
        queue.submit_microchunk(snapshot(coords, &[(0, 1)]));
        queue.submit_microchunk(snapshot(coords, &[(0, 2), (3, 5)]));

        assert_eq!(queue.pending_count(), 1);
        let drained = queue.drain_pending();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].global_coords, coords);
        assert_eq!(drained[0].materials, vec![2, 5]);
    }

    #[test]
    fn submit_batch_coalesces_duplicates() {
        let queue = ChangeQueue::new();

        queue
            .submit_batch([
                snapshot(IVec3::new(0, 0, 0), &[(0, 1)]),
                snapshot(IVec3::new(8, 0, 0), &[(1, 2)]),
                snapshot(IVec3::new(0, 0, 0), &[(2, 9)]),
            ])
            .unwrap();

        assert_eq!(queue.pending_count(), 2);

        let drained = queue.drain_pending();
        let by_coords: HashMap<_, _> = drained
            .iter()
            .map(|s| (s.global_coords, s.materials.clone()))
            .collect();

        assert_eq!(by_coords[&IVec3::new(0, 0, 0)], vec![9]);
        assert_eq!(by_coords[&IVec3::new(8, 0, 0)], vec![2]);
    }

    #[test]
    fn apply_is_idempotent_and_last_wins() {
        let mut mirrors = HashMap::new();
        let coords = IVec3::new(0, 0, 0);
        let region = region_index_of(coords);

        let first = snapshot(coords, &[(0, 1)]);
        assert_eq!(
            apply_snapshots(&mut mirrors, vec![first.clone()]),
            vec![region]
        );
        assert_eq!(mirrors[&region].microchunk(coords), Some(&first));

        assert!(apply_snapshots(&mut mirrors, vec![first.clone()]).is_empty());

        let second = snapshot(coords, &[(0, 2), (5, 3)]);
        assert_eq!(
            apply_snapshots(&mut mirrors, vec![second.clone()]),
            vec![region]
        );
        assert_eq!(mirrors[&region].microchunk(coords), Some(&second));

        assert_eq!(
            apply_snapshots(&mut mirrors, vec![zero(coords)]),
            vec![region]
        );
        assert!(mirrors.is_empty(), "emptied mirror must be dropped");

        assert!(apply_snapshots(&mut mirrors, vec![zero(IVec3::new(8, 8, 8))]).is_empty());
        assert!(mirrors.is_empty());
    }

    #[test]
    fn region_ids_derived_from_global_coords() {
        let mut mirrors = HashMap::new();
        apply_snapshots(
            &mut mirrors,
            vec![
                snapshot(IVec3::new(-8, -8, -8), &[(0, 1)]),
                snapshot(IVec3::new(0, 0, 0), &[(0, 2)]),
            ],
        );
        let mut keys: Vec<_> = mirrors.keys().copied().collect();
        keys.sort_unstable_by_key(|key| key.to_array());
        assert_eq!(keys, vec![IVec3::new(-1, -1, -1), IVec3::ZERO]);
    }

    #[test]
    #[should_panic(expected = "exceeds the renderer lattice")]
    fn submit_rejects_out_of_lattice_snapshot() {
        let input = RendererInput::new().unwrap();
        input.submit_microchunk(snapshot(IVec3::new(2048, 0, 0), &[(0, 1)]));
    }

    #[test]
    fn lattice_boundary_is_exclusive_high() {
        let input = RendererInput::new().unwrap();
        input.submit_microchunk(snapshot(IVec3::new(2047, 2047, 2047), &[(0, 1)]));
        input.wait_until_idle().unwrap();

        assert_eq!(region_index_of(IVec3::new(2047, 0, 0)), IVec3::new(7, 0, 0));
        assert_eq!(region_index_of(IVec3::new(2048, 0, 0)), IVec3::new(8, 0, 0));
    }

    #[test]
    fn rejected_submit_does_not_poison_queue() {
        let input = RendererInput::new().unwrap();

        let queue = input.change_queue();
        let rejected = std::thread::spawn(move || {
            queue
                .submit_batch([snapshot(IVec3::new(2048, 0, 0), &[(0, 1)])])
                .unwrap();
        });

        assert!(rejected.join().is_err(), "out-of-lattice batch must panic");

        input.submit_microchunk(snapshot(IVec3::new(0, 0, 0), &[(0, 1)]));
        input.wait_until_idle().unwrap();

        assert_eq!(input.region_count(), 1);
        assert_eq!(input.take_dirty_regions(), vec![IVec3::ZERO]);
    }

    #[test]
    fn out_of_order_snapshots_converge() {
        let input = RendererInput::new().unwrap();
        let coords_a = IVec3::new(0, 0, 0);
        let coords_b = IVec3::new(8, 0, 0);
        let coords_c = IVec3::new(16, 0, 0);

        input
            .submit_batch([snapshot(coords_a, &[(0, 1)]), snapshot(coords_b, &[(0, 2)])])
            .unwrap();

        input.submit_microchunk(snapshot(coords_a, &[(1, 9)])); // A updated, out of order
        input.submit_microchunk(snapshot(coords_b, &[(0, 2)])); // B re-sent (identical)
        input.submit_microchunk(snapshot(coords_c, &[(0, 3)]));
        input.wait_until_idle().unwrap();

        let expected = vec![
            snapshot(coords_a, &[(1, 9)]),
            snapshot(coords_b, &[(0, 2)]),
            snapshot(coords_c, &[(0, 3)]),
        ];

        let direct = pack_regions(&expected);
        let through_contract = input.packed_regions().unwrap();

        assert_eq!(through_contract.len(), 1);
        assert_eq!(through_contract[0].blocks, direct[0].blocks);
        assert_eq!(through_contract[0].aabbs, direct[0].aabbs);
    }

    #[test]
    fn startup_batch_matches_direct_pack() {
        let mut world = World::default();

        world.insert_voxel_at(IVec3::new(7, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(255, 0, 0), 2);
        world.insert_voxel_at(IVec3::new(256, 0, 0), 3);

        let snapshots = emit_snapshots(&world).unwrap();
        let input = RendererInput::new().unwrap();
        input.submit_batch(snapshots.iter().cloned()).unwrap();
        input.wait_until_idle().unwrap();

        let direct = pack_regions(&snapshots);
        let through_contract = input.packed_regions().unwrap();

        assert_eq!(through_contract.len(), 2);

        for (a, b) in through_contract.iter().zip(&direct) {
            assert_eq!(a.region_index, b.region_index);
            assert_eq!(a.blocks, b.blocks);
            assert_eq!(a.aabbs, b.aabbs);
        }
    }

    #[test]
    fn enqueue_from_threads_converges() {
        let input = RendererInput::new().unwrap();
        let queue = input.change_queue();

        const THREADS: i32 = 8;
        const PER_THREAD: i32 = 16;

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    for m in 0..PER_THREAD {
                        let coords = IVec3::new(
                            (t * PER_THREAD + m) * i32::try_from(MICRO_CHUNK_LENGTH).unwrap(),
                            t,
                            0,
                        );
                        queue.submit_microchunk(snapshot(coords, &[(0, (m % 256) as u8)]));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        input.wait_until_idle().unwrap();

        assert_eq!(
            input.region_count(),
            4,
            "regions derived from global coords"
        );

        let expected: Vec<_> = (0..THREADS)
            .flat_map(|t| {
                (0..PER_THREAD).map(move |m| {
                    snapshot(
                        IVec3::new(
                            (t * PER_THREAD + m) * i32::try_from(MICRO_CHUNK_LENGTH).unwrap(),
                            t,
                            0,
                        ),
                        &[(0, (m % 256) as u8)],
                    )
                })
            })
            .collect();

        let direct = pack_regions(&expected);
        let through_contract = input.packed_regions().unwrap();

        assert_eq!(through_contract.len(), direct.len());

        for (a, b) in through_contract.iter().zip(&direct) {
            assert_eq!(a.blocks, b.blocks);
            assert_eq!(a.aabbs, b.aabbs);
        }
    }

    #[test]
    fn worker_sleeps_when_idle_and_wakes_on_submit() {
        let input = RendererInput::new().unwrap();
        input.wait_until_idle().unwrap();

        let idle = (0..2000).any(|_| {
            if input.worker_idle() {
                true
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
                false
            }
        });

        assert!(
            idle,
            "worker must block on the condvar when idle (no polling)"
        );

        input.submit_microchunk(snapshot(IVec3::new(0, 0, 0), &[(0, 1)]));
        input.wait_until_idle().unwrap();
        assert_eq!(input.region_count(), 1);

        let idle_again = (0..2000).any(|_| {
            if input.worker_idle() {
                true
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
                false
            }
        });

        assert!(
            idle_again,
            "worker must return to the condvar wait after draining"
        );
    }

    #[test]
    fn dirty_regions_dedupe_and_clear() {
        let input = RendererInput::new().unwrap();
        input
            .submit_batch([
                snapshot(IVec3::new(0, 0, 0), &[(0, 1)]),
                snapshot(IVec3::new(8, 0, 0), &[(0, 2)]),
                snapshot(IVec3::new(256, 0, 0), &[(0, 3)]),
            ])
            .unwrap();
        input.wait_until_idle().unwrap();

        let dirty = input.take_dirty_regions();
        assert_eq!(dirty, vec![IVec3::ZERO, IVec3::new(1, 0, 0)]);

        input.wait_until_idle().unwrap();
        assert!(input.take_dirty_regions().is_empty());

        input.submit_microchunk(snapshot(IVec3::new(0, 0, 0), &[(1, 4)]));
        input.wait_until_idle().unwrap();
        let dirty = input.take_dirty_regions();
        assert_eq!(dirty, vec![IVec3::ZERO]);
        assert!(input.take_dirty_regions().is_empty());
    }

    #[test]
    fn emptying_a_region_drops_its_mirror() {
        let input = RendererInput::new().unwrap();

        let a = IVec3::new(0, 0, 0);
        let b = IVec3::new(8, 0, 0);

        input
            .submit_batch([snapshot(a, &[(0, 1)]), snapshot(b, &[(0, 2)])])
            .unwrap();
        input.wait_until_idle().unwrap();
        assert_eq!(input.region_count(), 1);

        input.submit_microchunk(zero(a));
        input.wait_until_idle().unwrap();
        assert_eq!(input.region_count(), 1, "one Micro-chunk still occupied");

        input.submit_microchunk(zero(b));
        input.wait_until_idle().unwrap();
        assert_eq!(
            input.region_count(),
            0,
            "last Micro-chunk removed → mirror dropped"
        );
        assert!(input.packed_regions().unwrap().is_empty());
    }
}
