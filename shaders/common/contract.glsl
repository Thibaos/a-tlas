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

// Radiance cache per-face entries (ADR 0019): one entry per exposed voxel
// face, block = the hit's AABB ordinal, entry = block*ENTRIES_PER_MC +
// voxel*6 + face. Each entry carries 3 state words (2 packed-half u32 of
// gamma-encoded irradiance rgb + a frame stamp; stamp 0 is absent) and 4
// accumulator words (fixed-point rgb at ACC_SCALE + a chain count) that
// deposits blend into and the resolve pass consumes and zeroes. STALE_T
// bounds the age a face is trusted past its last blend; EVENT_* drive the
// global light-change hysteresis reduction; LADDER_*/IMPULSE are 02's
// per-frame change thresholds and brightness clamp on the resolve blend.
#define CACHE_ENTRIES_PER_MC 3072u
#define CACHE_ENTRY_STRIDE 7u
#define CACHE_ACC_OFFSET 3u
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
