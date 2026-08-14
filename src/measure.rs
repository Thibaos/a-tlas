//! GPU measurement.
//!
//! The measurement closes the effort: correctness is proven
//! at the frame seam, and performance is measured against the 16 ms gate.
//! Per-stage GPU timestamps ([`QueryType::Timestamp`]) attribute the frame's
//! GPU time:
//!
//! - **trace_rays** — the ray pass's GPU time, bracketed around the
//!   `trace_rays` call in the render task (slots 2/3);
//! - **flight** — the frame's whole GPU interval on the graphics queue,
//!   bracketed around the render node's execution (slots 0/1 — the first
//!   and last commands of the production path's frame work; the app-only
//!   debug overlay, when present, draws after and is excluded);
//! - **as rebuild** — the ordered rebuild nodes' GPU time (upload + BLAS +
//!   TLAS), fed in from [`NodeTimings`](crate::region::rebuild::NodeTimings)
//!   when a change cycle ran (0 otherwise).
//!
//! The [`Measurement`] accumulates one [`FrameSample`] per completed frame
//! into a rolling ~60-frame window; the FPS log prints min/avg/p95 per
//! stage. The **16 ms gate** is the GPU timestamp sum (`trace + rebuild`);
//! the wall-clock frame interval is reported beside it — a wall-clock above
//! 16 ms with a small GPU sum points at CPU/present-bound, a different fix
//! than traversal.
//!
//! The measurement runs **on demand**: only the app attaches a timestamp
//! pool (`atlas-rt --measure`); the validator never constructs a
//! [`Measurement`], so the harness's captured frames are bit-identical
//! (no extra commands are recorded in the capture path).

use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
};
use vulkano_taskgraph::{descriptor_set::StorageBufferId, Id};

use crate::{app::GpuStack, region::rebuild::NodeTimings};

/// The measurement window: min/avg/p95 over ~60 completed frames (the
/// spec's ~60-frame readback).
pub const MEASURE_WINDOW: usize = 60;

/// The march-and-miss counter buffer: three uint words the DDA intersection
/// shader increments by atomicAdd (word 0 = hull-crossed, word 1 =
/// march-and-miss, word 2 = empty-and). Host-visible so the app reads it back
/// each frame after the flight idle; the render task resets it with a fill
/// each frame.
#[derive(Clone, Copy)]
pub struct CounterBuffer {
    /// The buffer id the render task fills (reset) and the app reads back.
    pub buffer_id: Id<Buffer>,
    /// The bindless id pushed into the shader's push constants.
    pub storage_id: StorageBufferId,
}

/// The three counter words the DDA shader writes, in buffer order (word 0 =
/// hull-crossed, word 1 = march-and-miss, word 2 = empty-and). `repr(C)` keeps
/// its size exactly the shader Counter block's byte size.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CounterWords {
    hull_crossed: u32,
    march_and_miss: u32,
    empty_and: u32,
}

/// The counter buffer's byte size — derived from the word struct so a future
/// fourth word cannot silently desync it from the shader Counter block.
const COUNTER_BYTES: u64 = std::mem::size_of::<CounterWords>() as u64;

// The query-pool slot layout (shared with the render task, which records
// the writes; [`Measurement`] owns the pool and reads them back).
/// The flight begin timestamp: the first command of the render node.
pub const FLIGHT_BEGIN_SLOT: u32 = 0;
/// The flight end timestamp: the last command of the render node (after
/// `trace_rays` — the app-only debug overlay, when present, draws after and
/// is excluded from "flight").
pub const FLIGHT_END_SLOT: u32 = 1;
/// The trace begin timestamp: immediately before `trace_rays`.
pub const TRACE_BEGIN_SLOT: u32 = 2;
/// The trace end timestamp: immediately after `trace_rays`.
pub const TRACE_END_SLOT: u32 = 3;
/// The pool's query count (two begin/end pairs).
pub const TIMESTAMP_SLOT_COUNT: u32 = 4;

/// One completed frame's measured GPU stages (nanoseconds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameSample {
    /// The wall-clock interval of the frame (CPU, `delta_time`).
    pub wall_ns: u64,
    /// The ray pass's GPU time (`trace_rays`).
    pub trace_ns: u64,
    /// The frame's whole GPU interval on the graphics queue.
    pub flight_ns: u64,
    /// The ordered rebuild nodes' GPU time this frame (0 when no change
    /// cycle ran).
    pub rebuild_ns: u64,
    /// Hull-crossed invocations this frame (the DDA slab passed and the
    /// march began) — the denominator of the march-and-miss rate.
    pub hull_crossed: u64,
    /// March-and-miss invocations this frame (the DDA marched the whole hull
    /// with no hit) — the numerator, and an upper bound on the ray-mask
    /// cull's win (Vulkan may re-run intersection shaders redundantly).
    pub march_and_miss: u64,
    /// Empty-AND invocations this frame (the provisional forward-box mask ANDs
    /// against the occupancy to zero) — the lower bound on the cull's win.
    pub empty_and: u64,
}

impl FrameSample {
    /// The 16 ms gate: the GPU timestamp sum (`trace_rays` + AS rebuilds).
    /// The write-out is inside `trace_ns` (the ray pass writes the
    /// swapchain storage image itself), so the sum is the renderer's whole
    /// per-frame GPU budget.
    pub fn gate_ns(&self) -> u64 {
        self.trace_ns.saturating_add(self.rebuild_ns)
    }
}

/// Min/avg/p95 over a sample set (nanoseconds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageStats {
    pub min_ns: u64,
    pub avg_ns: u64,
    pub p95_ns: u64,
}

impl StageStats {
    /// Computes min/avg/p95 over `samples`. p95 is the value at the 95th
    /// percentile position of the sorted set (`ceil(0.95·n)`-th smallest,
    /// 1-indexed); empty input yields the default (all zeros).
    pub fn from_samples(samples: &[u64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self::default();
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        let mut sum = 0u64;
        for &value in &sorted {
            sum = sum.saturating_add(value);
        }

        let p95_index = ((0.95 * n as f64).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);

        Self {
            min_ns: sorted[0],
            avg_ns: sum / n as u64,
            p95_ns: sorted[p95_index],
        }
    }
}

/// The app-side measurement accumulator. Owns the timestamp
/// query pool (shared with the render task) and the rolling frame window;
/// prints the per-stage min/avg/p95 into the FPS log.
#[derive(Default)]
pub struct Measurement {
    pool: Option<Arc<QueryPool>>,
    /// The march-and-miss counter buffer (created unconditionally with the
    /// measurement — only the --measure app constructs a Measurement).
    counter: Option<CounterBuffer>,
    /// The frame currently being assembled: the change cycle's rebuild time
    /// lands here when it runs; the trace/flight readback and the wall
    /// interval complete it at the next frame boundary.
    pending: FrameSample,
    /// Completed frames (rolling window of [`MEASURE_WINDOW`]).
    window: Vec<FrameSample>,
    /// `record_frame` calls so far (the pool holds nothing until the first
    /// frame executed — the readback starts at frame 2).
    frames: u64,
}

impl Measurement {
    /// Creates the measurement. The pool exists only when the graphics
    /// queue family supports timestamp queries; otherwise every read is
    /// skipped and the log reports timestamps unsupported.
    pub fn new(gpu: &GpuStack) -> Self {
        let pool = timestamp_supported(gpu).then(|| {
            QueryPool::new(
                &gpu.device,
                &QueryPoolCreateInfo {
                    query_count: TIMESTAMP_SLOT_COUNT,
                    ..QueryPoolCreateInfo::new(QueryType::Timestamp)
                },
            )
            .unwrap()
        });

        // The march-and-miss counter buffer: three uint words the DDA
        // increments by atomicAdd, host-visible for the per-frame readback.
        // TRANSFER_DST lets the render task reset it with a fill each frame.
        let counter = {
            let buffer_id = gpu
                .resources
                .create_buffer(
                    &BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    DeviceLayout::new_unsized::<[u8]>(COUNTER_BYTES).unwrap(),
                )
                .unwrap();
            let storage_id = gpu
                .resources
                .bindless_context()
                .unwrap()
                .global_set()
                .create_storage_buffer(buffer_id, 0, Some(COUNTER_BYTES))
                .unwrap();
            CounterBuffer {
                buffer_id,
                storage_id,
            }
        };

        Self {
            pool,
            counter: Some(counter),
            ..Default::default()
        }
    }

    /// The shared pool for the render task (which records the timestamp
    /// writes). `None` → the render task records no timestamps.
    pub fn pool(&self) -> Option<Arc<QueryPool>> {
        self.pool.clone()
    }

    /// The march-and-miss counter buffer (its buffer id + bindless storage id),
    /// for the render task to reset and push. `None` would mean the counter is
    /// absent, but a [`Measurement`] always creates one.
    pub fn counter(&self) -> Option<&CounterBuffer> {
        self.counter.as_ref()
    }

    /// Whether timestamp queries are available on this device.
    pub fn enabled(&self) -> bool {
        self.pool.is_some()
    }

    /// The change cycle's rebuild time,
    /// attributed to the frame being assembled. A rebuild spike lands in
    /// the AS-rebuild line, not in trace_rays.
    pub fn record_rebuild(&mut self, timings: &NodeTimings) {
        if !timings.supported {
            return;
        }
        self.pending.rebuild_ns = timings
            .upload_ns
            .saturating_add(timings.blas_ns)
            .saturating_add(timings.tlas_ns);
    }

    /// Completes the previous frame's sample: reads the pool (the previous
    /// execute's trace/flight — the caller waits the flight idle first),
    /// attaches the wall interval, and pushes the window.
    pub fn record_frame(&mut self, gpu: &GpuStack, wall_ns: u64) {
        self.frames += 1;
        // The pool holds nothing until the first frame executed.
        if self.frames < 2 {
            return;
        }

        self.pending.wall_ns = wall_ns;
        if let Some(pool) = &self.pool {
            self.pending.flight_ns = read_pool(gpu, pool, FLIGHT_BEGIN_SLOT, FLIGHT_END_SLOT);
            self.pending.trace_ns = read_pool(gpu, pool, TRACE_BEGIN_SLOT, TRACE_END_SLOT);
        }
        // The counter lags one frame, like the timestamps: the flight idle
        // before this call completed the previous execute, so the words are
        // that frame's counts (the render task resets them with its fill).
        if let Some(counter) = &self.counter {
            let words = read_counter(gpu, counter);
            self.pending.hull_crossed = u64::from(words.hull_crossed);
            self.pending.march_and_miss = u64::from(words.march_and_miss);
            self.pending.empty_and = u64::from(words.empty_and);
        }

        self.push_sample(self.pending);
        self.pending = FrameSample::default();
    }

    /// Pushes a completed sample into the rolling window (trims to
    /// [`MEASURE_WINDOW`]).
    fn push_sample(&mut self, sample: FrameSample) {
        self.window.push(sample);
        if self.window.len() > MEASURE_WINDOW {
            self.window.remove(0);
        }
    }

    /// Prints the per-stage min/avg/p95 over the window into the FPS log,
    /// with the 16 ms gate as
    /// the GPU timestamp sum and the wall-clock beside it.
    pub fn print_log(&self) {
        let n = self.window.len();
        if n == 0 {
            return;
        }

        // The march-and-miss rate is independent of timestamp support, so it
        // prints first. The ratio is the redundancy-robust invariant (Vulkan
        // may re-run intersection shaders, so the raw totals are upper bounds).
        let miss: u64 = self.window.iter().map(|s| s.march_and_miss).sum();
        let hull: u64 = self.window.iter().map(|s| s.hull_crossed).sum();
        let empty: u64 = self.window.iter().map(|s| s.empty_and).sum();
        if hull > 0 {
            let miss_ratio = miss as f64 / hull as f64;
            let empty_ratio = empty as f64 / hull as f64;
            println!(
                "  march-and-miss {miss_ratio:.4} ({miss} / {hull} hull-crossed, n={n})"
            );
            println!(
                "  empty-and     {empty_ratio:.4} ({empty} / {hull} hull-crossed) — lower bound on the cull's win"
            );
            if miss > 0 {
                println!(
                    "    cull win in [{empty_ratio:.4}, {miss_ratio:.4}] of hull-crossed; empty-and rejects {:.4} of wasted marches",
                    empty as f64 / miss as f64,
                );
            }
            println!(
                "    raw totals are upper bounds — Vulkan may re-run intersection shaders; the ratios are the invariant"
            );
        } else {
            println!("  march-and-miss n/a (0 hull-crossed invocations, n={n})");
        }

        if !self.enabled() {
            println!("  (measurement: timestamp queries unsupported on this device)");
            return;
        }

        let trace: Vec<u64> = self.window.iter().map(|s| s.trace_ns).collect();
        let rebuild: Vec<u64> = self.window.iter().map(|s| s.rebuild_ns).collect();
        let flight: Vec<u64> = self.window.iter().map(|s| s.flight_ns).collect();
        let gate: Vec<u64> = self.window.iter().map(|s| s.gate_ns()).collect();
        let wall: Vec<u64> = self.window.iter().map(|s| s.wall_ns).collect();

        let stage = |stats: StageStats| {
            format!(
                "{:>8.3} / {:>8.3} / {:>8.3}",
                ns_to_ms(stats.min_ns),
                ns_to_ms(stats.avg_ns),
                ns_to_ms(stats.p95_ns),
            )
        };

        println!("  gpu ms (min/avg/p95, n={n}):");
        println!("    trace_rays {}", stage(StageStats::from_samples(&trace)));
        println!(
            "    as rebuild {}",
            stage(StageStats::from_samples(&rebuild))
        );
        println!(
            "    flight     {}",
            stage(StageStats::from_samples(&flight))
        );
        println!(
            "  gate (trace + as rebuild) {} ms | wall {} ms — 16 ms budget",
            stage(StageStats::from_samples(&gate)),
            stage(StageStats::from_samples(&wall)),
        );
    }
}

/// Reads one begin/end timestamp pair from the pool (nanoseconds elapsed ×
/// `timestamp_period`). Zero when the results are not yet available — the
/// caller's flight idle makes them available; nothing is written otherwise.
fn read_pool(gpu: &GpuStack, pool: &QueryPool, begin_slot: u32, end_slot: u32) -> u64 {
    let mut values = [0u64; TIMESTAMP_SLOT_COUNT as usize];
    let available = pool
        .get_results::<u64>(
            0,
            TIMESTAMP_SLOT_COUNT,
            &mut values,
            QueryResultFlags::empty(),
        )
        .unwrap();
    if !available {
        return 0;
    }

    let period = gpu.device.physical_device().properties().timestamp_period as f64;
    let begin = values[begin_slot as usize];
    let end = values[end_slot as usize];
    (end.wrapping_sub(begin) as f64 * period) as u64
}

/// Reads the three counter words back from host-visible memory. The caller
/// waits the flight idle first, so the words are the completed frame's counts
/// (the render task resets them with a fill at the top of the next execute).
fn read_counter(gpu: &GpuStack, counter: &CounterBuffer) -> CounterWords {
    let buffer = gpu.resources.buffer(counter.buffer_id).buffer().clone();
    let sub = Subbuffer::new(buffer).cast_aligned::<u32>();
    let guard = sub.read().unwrap();
    CounterWords {
        hull_crossed: guard[0],
        march_and_miss: guard[1],
        empty_and: guard[2],
    }
}

/// Whether the graphics queue family supports timestamp queries.
fn timestamp_supported(gpu: &GpuStack) -> bool {
    let index = gpu.graphics_queue.queue_family_index() as usize;
    gpu.device.physical_device().queue_family_properties()[index]
        .timestamp_valid_bits
        .is_some()
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate is the GPU timestamp sum (trace + rebuild), saturating.
    #[test]
    fn gate_is_trace_plus_rebuild() {
        let sample = FrameSample {
            trace_ns: 3_000_000,
            rebuild_ns: 1_500_000,
            ..Default::default()
        };
        assert_eq!(sample.gate_ns(), 4_500_000);
        assert_eq!(FrameSample::default().gate_ns(), 0);
    }

    /// min/avg/p95 over the window: p95 is the 95th-percentile position of
    /// the sorted set (ceil(0.95·n)-th smallest).
    #[test]
    fn stage_stats_min_avg_p95() {
        let samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let stats = StageStats::from_samples(&samples);
        assert_eq!(stats.min_ns, 1);
        assert_eq!(stats.avg_ns, 5);
        // n=10: ceil(9.5)=10 → index 9 → the largest value.
        assert_eq!(stats.p95_ns, 10);

        let stats = StageStats::from_samples(&[]);
        assert_eq!(stats, StageStats::default());
    }

    /// The p95 index math over the ~60-frame window: the 57th smallest of
    /// 60 (0-indexed 56).
    #[test]
    fn p95_index_at_window_size() {
        let samples: Vec<u64> = (1..=60).collect();
        let stats = StageStats::from_samples(&samples);
        assert_eq!(stats.p95_ns, 57);
        assert_eq!(stats.min_ns, 1);
        assert_eq!(stats.avg_ns, 30);
    }

    /// The window keeps the last MEASURE_WINDOW samples (no growth) and
    /// rebuild attribution lands on the pending frame.
    #[test]
    fn window_trims_and_rebuild_lands_on_pending() {
        let mut m = Measurement::default();
        for frame in 1..=100u64 {
            m.push_sample(FrameSample {
                wall_ns: frame * 1000,
                trace_ns: frame * 2,
                flight_ns: frame * 3,
                ..Default::default()
            });
        }
        assert_eq!(m.window.len(), MEASURE_WINDOW);
        assert_eq!(m.window[0].wall_ns, 41_000);
        assert_eq!(m.window[MEASURE_WINDOW - 1].wall_ns, 100_000);

        // record_rebuild attributes the cycle's rebuild to the pending
        // frame (the one the next record_frame will complete).
        let mut m = Measurement::default();
        m.record_rebuild(&NodeTimings {
            upload_ns: 10,
            blas_ns: 20,
            tlas_ns: 30,
            supported: true,
        });
        assert_eq!(m.pending.rebuild_ns, 60);

        // Unsupported devices contribute nothing.
        let mut m = Measurement::default();
        m.record_rebuild(&NodeTimings {
            supported: false,
            ..Default::default()
        });
        assert_eq!(m.pending.rebuild_ns, 0);
    }
}
