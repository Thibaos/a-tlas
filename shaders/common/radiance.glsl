const float FP16_MAX = 65504.0;
const float PACK_EPS = 1e-6;

const vec3 HIT_DISTANCE_PARAMETERS = vec3(3.0, 0.1, 20.0);

float spec_magic_curve(float roughness) {
    return (1.0 - exp2(-200.0 * roughness * roughness)) * pow(clamp(roughness, 0.0, 1.0), 0.5);
}

float get_norm_hit_dist(float hit_dist, float view_z, float roughness) {
    float smc = spec_magic_curve(roughness);
    float f = (HIT_DISTANCE_PARAMETERS.x + abs(view_z) * HIT_DISTANCE_PARAMETERS.y)
            * mix(HIT_DISTANCE_PARAMETERS.z, 1.0, smc);
    return max(clamp(hit_dist / f, 0.0, 1.0), PACK_EPS);
}

vec4 pack_radiance_and_norm_hit_dist(vec3 radiance, float norm_hit_dist) {
    radiance = clamp(radiance, 0.0, FP16_MAX);

    float y = dot(radiance, vec3(0.25, 0.5, 0.25));
    float co = dot(radiance, vec3(0.5, 0.0, -0.5));
    float cg = dot(radiance, vec3(-0.25, 0.5, -0.25));

    return vec4(y, co, cg, clamp(norm_hit_dist, 0.0, 1.0));
}

vec3 unpack_radiance(vec4 data) {
    float t = data.x - data.z;

    vec3 rgb;
    rgb.y = data.x + data.z;
    rgb.x = t + data.y;
    rgb.z = t - data.y;

    return max(rgb, 0.0);
}

vec4 pack_normal_and_roughness(vec3 normal, float roughness) {
    normal /= max(abs(normal.x), max(abs(normal.y), abs(normal.z)));

    return vec4(normal * 0.5 + 0.5, roughness);
}
