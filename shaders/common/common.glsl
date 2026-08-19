struct MainPassPayload {
    vec4 color;
    float t;
    // The 8-bit material index (== palette index) of the committed voxel,
    // written by the DDA closest-hit from gl_HitKindEXT (the payload's
    // surface-identity channel; ticket 03. The raygen reads the bindless
    // Material table through it). Zero on miss; only read when t > 0.
    uint hit_kind;
    // The geometric surface normal at the hit (ticket 04, ADR 0009): the
    // face the DDA's march entered the committed voxel through (object
    // space == world space up to the translation instance transform). The
    // DDA closest-hit reconstructs it from the hit point; zero on miss;
    // only read when t > 0. For a camera embedded in a voxel (the t_min
    // commit, no crossed face) it points back along the ray.
    vec3 normal;
};

#define FLT_MAX 3.402823466e+38
