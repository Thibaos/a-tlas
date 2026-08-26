void build_primary_ray(out vec3 origin, out vec3 direction) {
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    const vec2 ndc = in_uv * 2.0 - 1.0;

    const vec4 clip_pos = vec4(ndc, -1.0, 1.0);
    vec4 eye_pos = camera.proj_inverse * clip_pos;
    eye_pos /= eye_pos.w;

    origin = (camera.view_inverse * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    direction = normalize((camera.view_inverse * vec4(eye_pos.xyz, 0.0)).xyz);
}
