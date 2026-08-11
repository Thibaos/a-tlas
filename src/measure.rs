//! GPU measurement (renderer-impl ticket 07).
//!
//! The measurement closes the effort: correctness (tickets 01-06) is proven
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
//! than traversal (rendering-core ticket 06, Q5).
//!
//! The measurement runs **on demand**: only the app attaches a timestamp
//! pool (`atlas-rt --measure`); the validator never constructs a
//! [`Measurement`], so the harness's captured frames are bit-identical
//! (no extra commands are recorded in the capture path).

use std::sync::Arc;

use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};

use crate::{app::GpuStack, region::rebuild::NodeTimings};

/// The measurement window: min/avg/p95 over ~60 completed frames (the
/// spec's ~60-frame readback).
pub const MEASURE_WINDOW: usize = 60;

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
}

impl FrameSample {
    /// The 16 ms gate: the GPU timestamp sum (`trace_rays` + AS rebuilds).
    /// The write-out is inside `trace_ns` (the ray pass writes the
    /// swapchain storage image itself), so the sum is the renderer's whole
    /// per-frame GPU budget (rendering-core ticket 06, Q5).
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

/// The app-side measurement accumulator (ticket 07). Owns the timestamp
/// query pool (shared with the render task) and the rolling frame window;
/// prints the per-stage min/avg/p95 into the FPS log.
#[derive(Default)]
pub struct Measurement {
    pool: Option<Arc<QueryPool>>,
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

        Self {
            pool,
            ..Default::default()
        }
    }

    /// The shared pool for the render task (which records the timestamp
    /// writes). `None` → the render task records no timestamps.
    pub fn pool(&self) -> Option<Arc<QueryPool>> {
        self.pool.clone()
    }

    /// Whether timestamp queries are available on this device.
    pub fn enabled(&self) -> bool {
        self.pool.is_some()
    }

    /// The change cycle's rebuild time (ticket 05's per-node timings),
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

    /// Prints the per-stage min/avg/p95 over the window into the FPS log
    /// (ticket 07's log surface — no separate HUD), with the 16 ms gate as
    /// the GPU timestamp sum and the wall-clock beside it.
    pub fn print_log(&self) {
        let n = self.window.len();
        if n == 0 {
            return;
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
