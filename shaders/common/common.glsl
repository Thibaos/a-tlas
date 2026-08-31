struct MainPassPayload {
    vec4 color;
    float t;
    uint hit_kind;
    vec3 normal;
    // The radiance cache key, packed by closest_hit for voxel hits: key =
    // region | block << 12; meta = voxel slot | face << 9 | faced << 12.
    uint cache_key;
    uint cache_meta;
};

#define FLT_MAX 3.402823466e+38
