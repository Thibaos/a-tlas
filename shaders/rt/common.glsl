struct MainPassPayload {
    vec4 color;
    float t;
    // The 8-bit material index (== palette index) of the committed voxel,
    // written by the DDA closest-hit from gl_HitKindEXT (the payload's
    // surface-identity channel; ticket 03 — the raygen reads the bindless
    // Material table through it). Zero on miss; only read when t > 0.
    uint hit_kind;
};

#define FLT_MAX 3.402823466e+38
