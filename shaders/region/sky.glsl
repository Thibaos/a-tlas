// Clear-day blue normalized to unit Rec.709 luminance: the knots' scale
// survives, only the hue shifts.
const vec3 SKY_TINT = vec3(0.608, 1.013, 2.026);

float sky_gradient(float mu) {
    vec4 k = scene.sky_knots;
    float t = clamp(mu, -1.0, 1.0);
    return (t < 0.0) ? mix(k.x, k.y, t + 1.0) : mix(k.y, k.z, t);
}

// The Background radiance at the direction dir: the gradient only.
vec3 sky_radiance(vec3 dir) {
    return SKY_TINT * vec3(sky_gradient(dir.y));
}
