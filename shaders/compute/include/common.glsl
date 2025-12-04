#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 inv_dir;
};

struct HitInfo {
    vec3 position;
    vec3 normal;
    bool hit;
};

uint popcnt_var64(uint64_t mask, uint width) {
    uint himask = uint(mask);
    uint count = 0;

    if (width >= 32) {
        count = bitCount(himask);
        himask = uint(mask >> 32);
    }
    uint m = 1u << (width & 31u);
    count += bitCount(himask & (m - 1u));

    return count;
}

uint popcnt64(uint64_t mask) {
    uint a = bitCount(uint(mask >> 0));
    uint b = bitCount(uint(mask >> 32));
    return a + b;
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 saturate(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 inferno_quintic(float x) {
	x = saturate(x);
	vec4 x1 = vec4(1.0, x, x * x, x * x * x); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return saturate(
        vec3(
		    dot(x1.xyzw, vec4(-0.027780558, +1.228188385, +0.278906882, +3.892783760)) + dot(x2.xy, vec2(-8.490712758, +4.069046086)),
		    dot(x1.xyzw, vec4(+0.014065206, +0.015360518, +1.605395918, -4.821108251)) + dot(x2.xy, vec2(+8.389314011, -4.193858954)),
		    dot(x1.xyzw, vec4(-0.019628385, +3.122510347, -5.893222355, +2.798380308)) + dot(x2.xy, vec2(-3.608884658, +4.324996022))
        )
    );
}

uint get_node_cell_index(vec3 pos, int scale_exp) {
    uvec3 cell_position = uvec3(pos) >> scale_exp & 3;
    return cell_position.x + cell_position.z * 4 + cell_position.y * 16;
}

vec3 floor_scale(vec3 pos, int scale_exp) {
    uint mask = ~0u << scale_exp;
    return vec3(uvec3(pos) & mask); // erase bits lower than scale
}