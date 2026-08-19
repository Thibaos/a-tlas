// Shared heat-map color helpers for the diagnostic Render modes (Ray latency,
// hull-crossed): HSV -> RGB and a blue -> red heat ramp. The same ramp the
// debug Hull mode's hsv_to_rgb uses for its coordinate hash.
vec3 hsv_to_rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// A blue -> red heat ramp: t = 0 (fast/cold) -> blue, t = 1 (slow/hot) -> red.
vec4 heat_color(float t) {
    float hue = mix(2.0 / 3.0, 0.0, clamp(t, 0.0, 1.0));
    return vec4(hsv_to_rgb(vec3(hue, 1.0, 1.0)), 1.0);
}
