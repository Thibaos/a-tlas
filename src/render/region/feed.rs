//! The renderer input contract: the world-facing
//! enqueue-only API plus the CPU-side change machinery.
//!
//! The world hands the renderer **Micro-chunk snapshots**: {global coords,
//! 64-byte Occupancy mask, u8 material indices}. Then create, update, and
//! removal are the same message: an emptied Micro-chunk re-snapshots with a
//! zero mask. Submitting never blocks on GPU: it inserts into a
//! mutex-protected pending set (last-wins per Micro-chunk) and signals a
//! worker thread via a condvar. The worker drains **everything pending per
//! cycle**, applies the snapshots into per-Region CPU mirrors
//! ([`RegionMirror`]). It derives each Region id from the snapshot's global
//! coords, never from the world, and publishes the dirty-Region set
//! ([`RendererInput::take_dirty_regions`]). The renderer repacks mirrors
//! through [`RegionData`] (see `crate::render::region::pack`) and feeds the
//! Region pipeline.
//!
//! Idle cost: with nothing pending, the worker blocks on the condvar. No
//! polling, zero cost. Startup is one `submit_batch` (the world's initial
//! snapshots) followed by a single drain cycle.

use std::{
    collections::HashMap,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};

use glam::IVec3;

use super::pack::{RegionData, pack_region};
use crate::core::grid::{assert_region_index_in_lattice, region_index_of};
use crate::world::snapshot::MicroChunkSnapshot;

/// One Region's CPU-side mirror: the authoritative per-Micro-chunk
/// snapshot state, the source for wholesale pool re-packing. Empty
/// Micro-chunks are never stored. A zero-mask snapshot removes the
/// Micro-chunk from the mirror, and an emptied mirror is dropped by
/// [`apply_snapshots`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionMirror {
    region_index: IVec3,
    /// Micro-chunk origin (global coords) → latest snapshot. One entry per
    /// non-empty Micro-chunk; last-wins by construction.
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn microchunk(&self, global_coords: IVec3) -> Option<&MicroChunkSnapshot> {
        self.microchunks.get(&global_coords)
    }

    /// Applies one snapshot. Create, update, and removal are the same
    /// message (removal = zero-mask re-snapshot). Returns whether the mirror
    /// actually changed (an identical re-snapshot is a no-op: idempotent).
    pub fn apply(&mut self, snapshot: MicroChunkSnapshot) -> bool {
        if snapshot.occupied_count() == 0 {
            // Removal: drop any previous state for this Micro-chunk.
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

    /// Packs the mirror into the pipeline-consumable [`RegionData`]: pool
    /// blocks (offset table + compact blocks) + trimmed hull AABBs, the
    /// packing step the Region pipeline consumes.
    pub fn pack(&self) -> RegionData {
        debug_assert!(!self.is_empty(), "packing an empty mirror");
        let snapshots: Vec<&MicroChunkSnapshot> = self.microchunks.values().collect();
        pack_region(self.region_index, &snapshots)
    }
}

/// The shared change state behind [`ChangeQueue`] and [`RendererInput`].
struct ChangeQueueInner {
    /// Snapshots enqueued but not yet drained, keyed by Micro-chunk origin:
    /// repeated/out-of-order submissions coalesce. Only the last one
    /// survives (last-wins, idempotent).
    pending: Mutex<HashMap<IVec3, MicroChunkSnapshot>>,
    /// Wakes the worker when snapshots arrive (or on shutdown).
    wake_worker: Condvar,
    /// Wakes renderer-side waiters (`wait_until_idle`) at the end of a
    /// drain cycle.
    wake_renderer: Condvar,
    /// The per-Region CPU mirrors the worker applies into.
    mirrors: Mutex<HashMap<IVec3, RegionMirror>>,
    /// The dirty-Region set published by the drain cycles since the last
    /// [`RendererInput::take_dirty_regions`] (deduped by the worker).
    applied_regions: Mutex<Vec<IVec3>>,
    /// The worker is blocked on `wake_worker` with nothing pending. The
    /// idle invariant (no polling, zero cost). Probe for tests.
    idle: AtomicBool,
    /// The worker is mid-cycle (drained, not yet published). Makes
    /// `wait_until_idle` precise (see its docs for the condvar argument).
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

/// The world-facing, enqueue-only half of the input contract. Clone to share
/// across threads; submitting never blocks on GPU. It inserts into the
/// pending set (a short mutex hold at most) and signals the worker.
#[derive(Clone)]
pub struct ChangeQueue {
    inner: Arc<ChangeQueueInner>,
}

impl ChangeQueue {
    /// Creates a detached queue with no worker (unit-testing / composition;
    /// the app gets its queue from [`RendererInput`]).
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(ChangeQueueInner::new()),
        }
    }

    /// Enqueues one Micro-chunk snapshot (create/update/removal are the same
    /// message; removal = zero-mask re-snapshot). Last-wins per Micro-chunk;
    /// safe from any thread; never blocks on GPU.
    pub fn submit_microchunk(&self, snapshot: MicroChunkSnapshot) {
        // The world→renderer boundary: a Micro-chunk whose Region index falls
        // outside the v1 lattice (±2048/axis) cannot be represented (the
        // 12-bit region-id budget). Reject loudly and unconditionally.
        // Release mode must not silently alias and index past the fixed tables.
        //
        // The check runs *before* the `pending` lock is taken: a rejection
        // panics here on the caller thread holding no lock, so the queue's
        // Mutex is never poisoned and the worker thread (whose
        // `wake_worker.wait(..).unwrap()` would panic on a poisoned lock, then
        // trip `RendererInput::drop`'s `join().expect`) is never affected.
        assert_region_index_in_lattice(region_index_of(snapshot.global_coords));
        self.inner
            .pending
            .lock()
            .unwrap()
            .insert(snapshot.global_coords, snapshot);
        self.inner.wake_worker.notify_all();
    }

    /// Enqueues a batch of snapshots (one pending-lock hold). Same
    /// last-wins semantics as [`ChangeQueue::submit_microchunk`].
    pub fn submit_batch<I>(&self, snapshots: I)
    where
        I: IntoIterator<Item = MicroChunkSnapshot>,
    {
        // See [`ChangeQueue::submit_microchunk`]: the lattice boundary check
        // runs *before* taking the `pending` lock, so a rejection cannot
        // poison the queue's Mutex and kill the worker thread. The whole batch
        // is validated up front: either every snapshot is in-lattice (then
        // all are inserted under one lock hold) or the batch is rejected
        // atomically, before any state changes.
        let validated: Vec<MicroChunkSnapshot> = snapshots
            .into_iter()
            .map(|snapshot| {
                assert_region_index_in_lattice(region_index_of(snapshot.global_coords));
                snapshot
            })
            .collect();
        {
            let mut pending = self.inner.pending.lock().unwrap();
            for snapshot in validated {
                pending.insert(snapshot.global_coords, snapshot);
            }
        }
        self.inner.wake_worker.notify_all();
    }

    /// How many snapshots are pending (enqueued, not yet drained).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn pending_count(&self) -> usize {
        self.inner.pending.lock().unwrap().len()
    }

    /// Takes everything pending (the worker's per-cycle drain). Last-wins
    /// coalescing already happened at submit time (keyed insert).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn drain_pending(&self) -> Vec<MicroChunkSnapshot> {
        let mut pending = self.inner.pending.lock().unwrap();
        std::mem::take(&mut *pending).into_values().collect()
    }

    /// Asks the worker to exit after its current cycle (used by Drop).
    pub(crate) fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.inner.wake_worker.notify_all();
    }
}

/// The renderer's end of the input contract: owns the worker thread and the
/// per-Region mirrors. The renderer drains dirty regions each cycle and feeds
/// the packed mirrors to the Region pipeline.
pub struct RendererInput {
    queue: ChangeQueue,
    worker: Option<JoinHandle<()>>,
}

impl RendererInput {
    /// Spawns the worker thread and returns the ready input machinery.
    pub fn new() -> Self {
        let queue = ChangeQueue::new();
        let inner = queue.inner.clone();
        let worker = std::thread::Builder::new()
            .name("region-input-worker".to_string())
            .spawn(move || worker_loop(&inner))
            .expect("failed to spawn the input worker thread");
        Self {
            queue,
            worker: Some(worker),
        }
    }

    /// The world-facing handle (cloneable, callable from any thread).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn change_queue(&self) -> ChangeQueue {
        self.queue.clone()
    }

    /// Convenience: enqueue one snapshot (see [`ChangeQueue::submit_microchunk`]).
    pub fn submit_microchunk(&self, snapshot: MicroChunkSnapshot) {
        self.queue.submit_microchunk(snapshot);
    }

    /// Convenience: enqueue a batch (see [`ChangeQueue::submit_batch`]).
    pub fn submit_batch<I>(&self, snapshots: I)
    where
        I: IntoIterator<Item = MicroChunkSnapshot>,
    {
        self.queue.submit_batch(snapshots);
    }

    /// Blocks until every pending snapshot has been drained and applied to
    /// the mirrors (the worker is asleep again). Used by the harness and
    /// tests for deterministic startup/edit points; the production renderer
    /// would consume [`RendererInput::take_dirty_regions`] instead.
    ///
    /// Condvar soundness: the worker clears `busy` while holding the pending
    /// lock, and this check-then-wait happens under the same lock, so the
    /// transition cannot be missed.
    pub fn wait_until_idle(&self) {
        let mut pending = self.queue.inner.pending.lock().unwrap();
        while !pending.is_empty() || self.queue.inner.busy.load(Ordering::SeqCst) {
            pending = self.queue.inner.wake_renderer.wait(pending).unwrap();
        }
    }

    /// The dirty-Region set since the last call: every Region whose mirror
    /// changed (deduped, sorted). Empty between change cycles. The renderer
    /// does no work when nothing changed. Consumed by the residency manager
    /// ([`crate::render::region::residency::RegionStore::apply`]); the validator
    /// rebuilds wholesale for now.
    pub fn take_dirty_regions(&self) -> Vec<IVec3> {
        let mut dirty = std::mem::take(&mut *self.queue.inner.applied_regions.lock().unwrap());
        dirty.sort_unstable_by_key(|region| region.to_array());
        dirty
    }

    /// Packs one Region's mirror into pipeline-consumable [`RegionData`]
    /// (the per-Region pack the residency manager applies on a change
    /// cycle); `None` when the mirror is empty (the Region left residency).
    pub fn packed_region(&self, region_index: IVec3) -> Option<RegionData> {
        let mirrors = self.queue.inner.mirrors.lock().unwrap();
        mirrors.get(&region_index).map(RegionMirror::pack)
    }

    /// The packing step the Region pipeline consumes: every non-empty mirror
    /// packed into [`RegionData`] (pool blocks + offset table + trimmed
    /// hulls), sorted by Region id (deterministic instance order).
    pub fn packed_regions(&self) -> Vec<RegionData> {
        let mirrors = self.queue.inner.mirrors.lock().unwrap();
        let mut regions: Vec<RegionData> = mirrors.values().map(|mirror| mirror.pack()).collect();
        regions.sort_unstable_by_key(|region| region.region_id());
        regions
    }

    /// The number of resident (non-empty) Region mirrors.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn region_count(&self) -> usize {
        self.queue.inner.mirrors.lock().unwrap().len()
    }

    /// True when the worker is blocked on the condvar with nothing pending. The idle invariant (no polling, zero cost). Probe for tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn worker_idle(&self) -> bool {
        self.queue.inner.idle.load(Ordering::SeqCst)
    }
}

impl Drop for RendererInput {
    fn drop(&mut self) {
        self.queue.shutdown();
        if let Some(worker) = self.worker.take() {
            worker.join().expect("input worker thread panicked");
        }
    }
}

fn worker_loop(inner: &Arc<ChangeQueueInner>) {
    loop {
        let taken = {
            let mut pending = inner.pending.lock().unwrap();
            loop {
                if !pending.is_empty() {
                    break;
                }
                if inner.shutdown.load(Ordering::SeqCst) {
                    return;
                }
                inner.idle.store(true, Ordering::SeqCst);
                pending = inner.wake_worker.wait(pending).unwrap();
                inner.idle.store(false, Ordering::SeqCst);
            }
            inner.busy.store(true, Ordering::SeqCst);
            std::mem::take(&mut *pending)
                .into_values()
                .collect::<Vec<MicroChunkSnapshot>>()
        };

        let dirty = {
            let mut mirrors = inner.mirrors.lock().unwrap();
            apply_snapshots(&mut mirrors, taken)
        };

        {
            let mut applied = inner.applied_regions.lock().unwrap();
            for region in dirty {
                if !applied.contains(&region) {
                    applied.push(region);
                }
            }
        }

        {
            let _pending = inner.pending.lock().unwrap();
            inner.busy.store(false, Ordering::SeqCst);
        }
        inner.wake_renderer.notify_all();
    }
}

/// Applies a drained batch into the per-Region mirrors. The Region id is
/// derived from each snapshot's global coords. The renderer owns the
/// lattice; the world never computes or passes Region ids. Returns the dirty
/// Region indices (deduped, sorted); Regions whose mirrors empty out are
/// dropped.
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
    dirty.sort_unstable_by_key(|region| region.to_array());
    dirty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::grid::{MICRO_CHUNK_EDGE, region_index_of},
        render::region::pack::pack_regions,
        world::snapshot::emit_snapshots,
        world::World,
    };

    /// Builds a snapshot for one Micro-chunk from (bit index, material) cells.
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

    /// The pending set coalesces per Micro-chunk: two submissions for the
    /// same Micro-chunk leave one entry, and the drain yields the last one.
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

    /// A batch is one atomic submission; duplicate keys inside it also
    /// coalesce last-wins.
    #[test]
    fn submit_batch_coalesces_duplicates() {
        let queue = ChangeQueue::new();
        queue.submit_batch([
            snapshot(IVec3::new(0, 0, 0), &[(0, 1)]),
            snapshot(IVec3::new(8, 0, 0), &[(1, 2)]),
            snapshot(IVec3::new(0, 0, 0), &[(2, 9)]),
        ]);
        assert_eq!(queue.pending_count(), 2);
        let drained = queue.drain_pending();
        let by_coords: HashMap<_, _> = drained
            .iter()
            .map(|s| (s.global_coords, s.materials.clone()))
            .collect();
        assert_eq!(by_coords[&IVec3::new(0, 0, 0)], vec![9]);
        assert_eq!(by_coords[&IVec3::new(8, 0, 0)], vec![2]);
    }

    /// Apply is idempotent and last-wins: create, identical re-snapshot (no
    /// change → not dirty), update, removal, removal-of-absent (no change).
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

        // Identical re-snapshot: no change, no dirty Region.
        assert!(apply_snapshots(&mut mirrors, vec![first.clone()]).is_empty());

        // Update (last-wins replaces the content).
        let second = snapshot(coords, &[(0, 2), (5, 3)]);
        assert_eq!(
            apply_snapshots(&mut mirrors, vec![second.clone()]),
            vec![region]
        );
        assert_eq!(mirrors[&region].microchunk(coords), Some(&second));

        // Removal (zero-mask re-snapshot) empties the mirror.
        assert_eq!(
            apply_snapshots(&mut mirrors, vec![zero(coords)]),
            vec![region]
        );
        assert!(mirrors.is_empty(), "emptied mirror must be dropped");

        // Removal of a never-present Micro-chunk: no mirror, no dirty Region.
        assert!(apply_snapshots(&mut mirrors, vec![zero(IVec3::new(8, 8, 8))]).is_empty());
        assert!(mirrors.is_empty());
    }

    /// The renderer derives Region ids from global coords, including floor
    /// division for negative coords; the world never passes them (the
    /// snapshot type has no Region field).
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

    /// The world→renderer boundary rejects a snapshot whose Region index falls
    /// outside the v1 lattice (the 12-bit region-id budget, ±2048/axis). An
    /// over-lattice model must fail loudly here. Release mode never aliases.
    #[test]
    #[should_panic(expected = "exceeds the renderer lattice")]
    fn submit_rejects_out_of_lattice_snapshot() {
        let input = RendererInput::new();
        // Voxel 2048 → Region index 8, just past the [-8, 8) budget.
        input.submit_microchunk(snapshot(IVec3::new(2048, 0, 0), &[(0, 1)]));
    }

    /// The lattice boundary is inclusive at the high edge: voxel 2047 (Region
    /// index 7) is valid, voxel 2048 (Region index 8) is not.
    #[test]
    fn lattice_boundary_is_exclusive_high() {
        let input = RendererInput::new();
        input.submit_microchunk(snapshot(IVec3::new(2047, 2047, 2047), &[(0, 1)]));
        input.wait_until_idle();
        // Region index 7 fits; the batch below would panic at 2048, asserted
        // by the should_panic test, not duplicated here.
        assert_eq!(region_index_of(IVec3::new(2047, 0, 0)), IVec3::new(7, 0, 0));
        assert_eq!(region_index_of(IVec3::new(2048, 0, 0)), IVec3::new(8, 0, 0));
    }

    /// A rejected (out-of-lattice) submit must not poison the queue: the
    /// boundary check runs before the `pending` lock is taken, so a panic on
    /// the caller thread leaves the Mutex healthy and the worker able to drain
    /// later valid submits. (Without that ordering, the poisoned lock would
    /// kill the worker on its next `wait(..).unwrap()` and trip
    /// `RendererInput::drop`'s `join().expect`.)
    #[test]
    fn rejected_submit_does_not_poison_queue() {
        let input = RendererInput::new();

        // The out-of-lattice submit panics on a separate thread (caught here),
        // exactly as a caller thread would experience it.
        let queue = input.change_queue();
        let rejected = std::thread::spawn(move || {
            queue.submit_batch([snapshot(IVec3::new(2048, 0, 0), &[(0, 1)])]);
        });
        assert!(rejected.join().is_err(), "out-of-lattice batch must panic");

        // The queue is still healthy: a valid submit drains and becomes a
        // mirror. The worker did not die on a poisoned lock.
        input.submit_microchunk(snapshot(IVec3::new(0, 0, 0), &[(0, 1)]));
        input.wait_until_idle();
        assert_eq!(input.region_count(), 1);
        assert_eq!(input.take_dirty_regions(), vec![IVec3::ZERO]);
    }

    /// Out-of-order, repeated snapshots across multiple drain cycles converge
    /// to the last-wins state, and the packed mirrors equal the direct pack
    /// of that state (the pack the Region pipeline consumes).
    #[test]
    fn out_of_order_snapshots_converge() {
        let input = RendererInput::new();
        let coords_a = IVec3::new(0, 0, 0);
        let coords_b = IVec3::new(8, 0, 0);
        let coords_c = IVec3::new(16, 0, 0);

        input.submit_batch([snapshot(coords_a, &[(0, 1)]), snapshot(coords_b, &[(0, 2)])]);
        input.submit_microchunk(snapshot(coords_a, &[(1, 9)])); // A updated, out of order
        input.submit_microchunk(snapshot(coords_b, &[(0, 2)])); // B re-sent (identical)
        input.submit_microchunk(snapshot(coords_c, &[(0, 3)]));
        input.wait_until_idle();

        let expected = vec![
            snapshot(coords_a, &[(1, 9)]),
            snapshot(coords_b, &[(0, 2)]),
            snapshot(coords_c, &[(0, 3)]),
        ];
        let direct = pack_regions(&expected);
        let through_contract = input.packed_regions();
        assert_eq!(through_contract.len(), 1);
        assert_eq!(through_contract[0].blocks, direct[0].blocks);
        assert_eq!(through_contract[0].aabbs, direct[0].aabbs);
    }

    /// Startup via submit_batch over a real world matches the direct pack
    /// path exactly, across Regions (the pack the pipeline consumes).
    #[test]
    fn startup_batch_matches_direct_pack() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(7, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(255, 0, 0), 2);
        world.insert_voxel_at(IVec3::new(256, 0, 0), 3);

        let snapshots = emit_snapshots(&world);
        let input = RendererInput::new();
        input.submit_batch(snapshots.iter().cloned());
        input.wait_until_idle();

        let direct = pack_regions(&snapshots);
        let through_contract = input.packed_regions();
        assert_eq!(through_contract.len(), 2);
        for (a, b) in through_contract.iter().zip(&direct) {
            assert_eq!(a.region_index, b.region_index);
            assert_eq!(a.blocks, b.blocks);
            assert_eq!(a.aabbs, b.aabbs);
        }
    }

    /// Enqueuing from any thread never blocks on GPU and converges: many
    /// threads submit concurrently while the worker drains; everything lands
    /// in the mirrors exactly once.
    #[test]
    fn enqueue_from_threads_converges() {
        let input = RendererInput::new();
        let queue = input.change_queue();

        const THREADS: i32 = 8;
        const PER_THREAD: i32 = 16;

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    for m in 0..PER_THREAD {
                        let coords = IVec3::new((t * PER_THREAD + m) * MICRO_CHUNK_EDGE, t, 0);
                        queue.submit_microchunk(snapshot(coords, &[(0, (m % 256) as u8)]));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        input.wait_until_idle();

        // All Micro-chunks land in the 4 Regions their x-extent spans (x up
        // to 1016 → region x 0..4, y = t < 8 → region y = 0, z = 0): 128
        // Micro-chunks across 4 resident Regions, region ids derived from
        // global coords.
        assert_eq!(
            input.region_count(),
            4,
            "regions derived from global coords"
        );

        let expected: Vec<_> = (0..THREADS)
            .flat_map(|t| {
                (0..PER_THREAD).map(move |m| {
                    snapshot(
                        IVec3::new((t * PER_THREAD + m) * MICRO_CHUNK_EDGE, t, 0),
                        &[(0, (m % 256) as u8)],
                    )
                })
            })
            .collect();
        let direct = pack_regions(&expected);
        let through_contract = input.packed_regions();
        assert_eq!(through_contract.len(), direct.len());
        for (a, b) in through_contract.iter().zip(&direct) {
            assert_eq!(a.blocks, b.blocks);
            assert_eq!(a.aabbs, b.aabbs);
        }
    }

    /// The idle invariant: with an empty Change queue the worker sleeps on
    /// the condvar (no polling, zero cost), wakes on submit, drains, and
    /// sleeps again.
    #[test]
    fn worker_sleeps_when_idle_and_wakes_on_submit() {
        let input = RendererInput::new();
        input.wait_until_idle();

        // Bounded poll: the worker transitions to the condvar wait right
        // after finishing its cycle. Sleeps instead of spin-yields so the
        // poll is robust under parallel test load.
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
        input.wait_until_idle();
        assert_eq!(input.region_count(), 1);

        // And back to sleep. The whole cycle is condvar-driven.
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

    /// The dirty-Region set dedupes across cycles and clears on take.
    #[test]
    fn dirty_regions_dedupe_and_clear() {
        let input = RendererInput::new();
        input.submit_batch([
            snapshot(IVec3::new(0, 0, 0), &[(0, 1)]),
            snapshot(IVec3::new(8, 0, 0), &[(0, 2)]),
            snapshot(IVec3::new(256, 0, 0), &[(0, 3)]),
        ]);
        input.wait_until_idle();

        let dirty = input.take_dirty_regions();
        assert_eq!(dirty, vec![IVec3::ZERO, IVec3::new(1, 0, 0)]);

        // Nothing changed since the take.
        input.wait_until_idle();
        assert!(input.take_dirty_regions().is_empty());

        // A second cycle for an already-dirty Region dedupes against the
        // previous take (the applied set is per-take, so it reappears, but
        // within one cycle it appears once).
        input.submit_microchunk(snapshot(IVec3::new(0, 0, 0), &[(1, 4)]));
        input.wait_until_idle();
        let dirty = input.take_dirty_regions();
        assert_eq!(dirty, vec![IVec3::ZERO]);
        assert!(input.take_dirty_regions().is_empty());
    }

    #[test]
    fn emptying_a_region_drops_its_mirror() {
        let input = RendererInput::new();
        let a = IVec3::new(0, 0, 0);
        let b = IVec3::new(8, 0, 0);
        input.submit_batch([snapshot(a, &[(0, 1)]), snapshot(b, &[(0, 2)])]);
        input.wait_until_idle();
        assert_eq!(input.region_count(), 1);

        input.submit_microchunk(zero(a));
        input.wait_until_idle();
        assert_eq!(input.region_count(), 1, "one Micro-chunk still occupied");

        input.submit_microchunk(zero(b));
        input.wait_until_idle();
        assert_eq!(
            input.region_count(),
            0,
            "last Micro-chunk removed → mirror dropped"
        );
        assert!(input.packed_regions().is_empty());
    }
}
