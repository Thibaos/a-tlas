// Clear-day blue normalized to unit Rec.709 luminance: the knots' scale
// survives, only the hue shifts.
const vec3 SKY_TINT = vec3(0.608, 1.013, 2.026);

float sky_gradient(float mu) {
    vec4 k = scene.sky_knots;
    float t = clamp(mu, -1.0, 1.0);
    return (t < 0.0) ? mix(k.x, k.y, t + 1.0) : mix(k.y, k.z, t);
}

float sky_pdf_norm() {
    vec4 k = scene.sky_knots;
    return 0.5 * ((k.x + k.y) + (k.y + k.z));
}

// The marginal pdf of μ: L(μ)/Z (uniform solid-angle measure in μ: dω = dφ·dμ).
float sky_mu_pdf(float mu) {
    return sky_gradient(mu) / sky_pdf_norm();
}

// Inverse-CDF sample of μ from u ∈ [0, 1): per-segment quadratic inversion.
// The CDF of a piecewise-linear function is piecewise-quadratic. Degenerate
// segments (equal end knots: the gradient is constant there) fall back to
// the linear form. The +root is the correct branch for both signs of the
// quadratic coefficient (the CDF is monotone, all-positive knots).
float sky_sample_mu(float u) {
    vec4 k = scene.sky_knots;
    float Z = sky_pdf_norm();
    float c0 = (0.5 * (k.x + k.y)) / Z; // the CDF at the horizon (μ = 0)
    if (u < c0) {
        // Segment [-1, 0], t = μ + 1 ∈ [0, 1]: L = k.x + (k.y - k.x)·t and
        // ∫ L dt = k.x·t + (k.y - k.x)·t²/2. Solve for t: a·t² + b·t - uZ = 0
        // with a = (k.y - k.x)/2, b = k.x.
        float a = 0.5 * (k.y - k.x);
        float b = k.x;
        if (a == 0.0) {
            return u * Z / b - 1.0;
        }
        float disc = max(b * b + 4.0 * a * u * Z, 0.0);
        float t = (-b + sqrt(disc)) / (2.0 * a);
        return t - 1.0;
    }
    // Segment [0, 1], μ ∈ [0, 1]: L = k.y + (k.z - k.y)·μ and
    // ∫ L dμ = k.y·μ + (k.z - k.y)·μ²/2. Solve for μ: a·μ² + b·μ - (u-c0)Z = 0
    // with a = (k.z - k.y)/2, b = k.y.
    float a = 0.5 * (k.z - k.y);
    float b = k.y;
    if (a == 0.0) {
        return (u - c0) * Z / b;
    }
    float disc = max(b * b + 4.0 * a * (u - c0) * Z, 0.0);
    return (-b + sqrt(disc)) / (2.0 * a);
}

// The transport radiance at the direction dir: the gradient only (the disk
// is not part of the transport. See the header comment). Used by the sky
// miss shader (BSDF-miss samples) and the env NEE.
vec3 sky_radiance(vec3 dir) {
    return SKY_TINT * vec3(sky_gradient(dir.y));
}

// The camera's direct view of the sky: the gradient plus the Sun disk, a
// measure-zero radiance bump detected by a pure dot test (no atan needed).
// Evaluated only by the production raygen's primary-miss branch.
vec3 sky_radiance_with_disk(vec3 dir) {
    vec3 L = sky_radiance(dir);
    if (dot(dir, scene.sun_dir.xyz) > scene.sun_disk.y) {
        L += scene.sun_disk.z;
    }
    return L;
}
