//! The render task's measurement contract: the timestamp-pool slot layout the
//! render task records into, and the march-and-miss counter buffer it resets
//! and pushes. The host-side accumulator lives in `app::stats`.

use vulkano::buffer::Buffer;
use vulkano_taskgraph::{descriptor_set::StorageBufferId, Id};

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
pub struct CounterWords {
    pub hull_crossed: u32,
    pub march_and_miss: u32,
    pub empty_and: u32,
}

/// The counter buffer's byte size — derived from the word struct so a future
/// fourth word cannot silently desync it from the shader Counter block.
pub const COUNTER_BYTES: u64 = std::mem::size_of::<CounterWords>() as u64;

// The query-pool slot layout (shared with the render task, which records
// the writes; `app::stats::Measurement` owns the pool and reads them back).
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
