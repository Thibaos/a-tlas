// The Rust↔GLSL seam: constants the CPU writer (pack.rs, task.rs RenderMode)
// and the GPU readers must agree on. pack.rs cross-checks every value;
// declare none of these anywhere else.

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
#define MODE_NORMAL 4u

// Material-table flags word (uint[256] flags column, one per palette index):
// bit set iff the MATL entry is the glass type (ADR 0012).
#define MATFLAG_GLASS 1u
