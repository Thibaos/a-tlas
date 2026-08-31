// The Rust↔GLSL seam: constants the CPU writer (pack.rs, task.rs RenderMode)
// and the GPU readers must agree on, plus the demodulation and viewZ-sky
// values shared between production.rgen and composite.comp. pack.rs
// cross-checks every value; declare none of these anywhere else.

#define RAY_T_MIN 0.01
#define RAY_T_MAX 10000.0

#define REGION_TABLE_ENTRIES 4096
#define REGION_ID_MASK 0xFFFu

#define MC_STRIDE_Y 32u
#define MC_STRIDE_Z 1024u
#define VOXEL_STRIDE_Y 8u
#define VOXEL_STRIDE_Z 64u
#define MASK_BYTES 64u
#define OFFSET_SENTINEL 0xFFFFFFFFu

#define MODE_VOXEL 0u
#define MODE_HULL 1u
#define MODE_VALIDATION 2u
#define MODE_NORMAL 4u

#define ALBEDO_EPS 1e-3

// Radiance cache: SHaRC's fixed sparse table (06) replaces the per-Region
// slabs. One entry per exposed voxel face, keyed by the DDA's exact hit
// (region 12 bits | Micro-chunk block 15 | voxel 9 | face 3), hashed into
// CACHE_TABLE_ENTRIES slots and resolved through a fixed linear-probe
// bucket. Three buffers per entry: an 8 B key, a 16 B accumulator (fixed-
// point rgb at ACC_SCALE + a chain count) that deposits blend into, and a
// 16 B resolved record (2 packed-half u32 of gamma-encoded irradiance rgb +
// a frame stamp) that the resolve pass owns. The resolve also runs 02's
// tiers: dirty-Region sweeps on edit frames (region bits ride the key) and
// EVICT_T aging (an entry no deposit touched since its last blend leaves
// the table). STALE_T bounds the age a face is trusted past its last
// blend; EVENT_* drive the global light-change hysteresis reduction;
// LADDER_*/IMPULSE are 02's per-frame change thresholds and brightness
// clamp; DIRTY_WORDS is the per-frame edit bitset (one bit per Region).
#define CACHE_TABLE_BITS 23u
#define CACHE_TABLE_ENTRIES 8388608u
#define CACHE_EVICT_T 1024u
#define CACHE_DIRTY_WORDS 128u
#define CACHE_ACC_SCALE 1024.0
#define CACHE_ACC_TICK_CAP 1048575.0
#define CACHE_BASE_HYSTERESIS 0.97
#define CACHE_LADDER_LOW 0.25
#define CACHE_LADDER_HIGH 0.8
#define CACHE_LADDER_STEP 0.15
#define CACHE_IMPULSE 1.10
#define CACHE_STALE_T 4096u
#define CACHE_IRRADIANCE_GAMMA 5.0
#define CACHE_EVENT_FRAMES 10u
#define CACHE_EVENT_REDUCTION 0.5


// Sky pixels write this viewZ (beyond NRD's denoisingRange, so they are
// excluded from denoising); composite classifies a pixel as sky at >= this.
#define VIEWZ_SKY 1.0e6
