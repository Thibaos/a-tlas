struct MainPassPayload {
    float t;
    vec4 color;
    vec3 normal;
};

struct ShadowPayload {
    float attenuation;
};

#define EPSILON 1e-20
#define FLT_MAX 3.402823466e+38

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
#define MODE_NORMAL 2u
