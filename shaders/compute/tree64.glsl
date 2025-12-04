#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_shader_clock : enable

#include <vulkano.glsl>
#include "include/common.glsl"
#include "include/node.glsl"
#include "include/material.glsl"

#define EPSILON 0.0001
#define STORAGE_SIZE 32 * 32 * 32

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    StorageImageId image_id;
    StorageBufferId scene_params_id;
    StorageBufferId node_buffer_id;
    StorageBufferId material_buffer_id;
};

VKO_DECLARE_STORAGE_IMAGE(image, image2D, rgba8)

VKO_DECLARE_STORAGE_BUFFER(params, SceneParams {
    mat4 inv_proj_mat;
    vec3 camera_position;
    ivec2 resolution;
})

VKO_DECLARE_STORAGE_BUFFER(nodes, Nodes {
    Node[STORAGE_SIZE] data;
})

VKO_DECLARE_STORAGE_BUFFER(materials, Materials {
    Material[STORAGE_SIZE] data;
})

#define image vko_image(image, image_id)
#define params vko_buffer(params, scene_params_id)
#define nodes vko_buffer(nodes, node_buffer_id)
#define materials vko_buffer(materials, material_buffer_id)

Ray get_primary_ray(ivec2 screen_position) {
    vec2 uv = (vec2(screen_position) + 0.5) / vec2(params.resolution);
    uv = uv * 2.0 - 1.0;
    vec4 far = params.inv_proj_mat * vec4(uv, 1.0, 1.0);

    Ray ray;
    ray.origin = params.camera_position;
    ray.direction = normalize(far.xyz / far.w);
    ray.inv_dir = 1.0 / ray.direction; 

    return ray;
}

vec3 get_mirrored_position(vec3 position, vec3 direction, bool range_check) {
    vec3 mirrored = vec3(uvec3(position) ^ 8388607);
    // XOR-ing will only work for coords in range [1.0, 2.0),
    // fallback to subtractions if that's not the case.
    bool lt = position.x < 1.0 || position.y < 1.0 || position.z < 1.0;
    bool ge = position.x >= 2.0|| position.y >= 2.0|| position.z >= 2.0;
    if (range_check && (lt || ge)) mirrored = 3.0 - position;
    vec3 o = vec3(
        direction.x > 0.0 ? mirrored.x : position.x,
        direction.y > 0.0 ? mirrored.y : position.y,
        direction.z > 0.0 ? mirrored.z : position.z
    );

    return o;
}

bool is_voxel(vec3 position) {
    vec3 sqr = position * position;
    return sqr.x + sqr.y + sqr.z < 100.0;
}

bool intersect_aabb(Ray ray, vec3 voxel_position, float half_scale) {
    vec3 min_box = voxel_position - vec3(half_scale);
    vec3 max_box = voxel_position + vec3(half_scale);

    float tx1 = (min_box.x - ray.origin.x) * ray.inv_dir.x;
    float tx2 = (max_box.x - ray.origin.x) * ray.inv_dir.x;

    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);

    float ty1 = (min_box.y - ray.origin.y) * ray.inv_dir.y;
    float ty2 = (max_box.y - ray.origin.y) * ray.inv_dir.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (min_box.z - ray.origin.z) * ray.inv_dir.z;
    float tz2 = (max_box.z - ray.origin.z) * ray.inv_dir.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    return tmax >= tmin;
}

HitInfo ray_cast(Ray ray) {
    HitInfo hit;
    hit.hit = false;

    vec3 origin = ray.origin;
    vec3 position = ray.origin + ray.direction * EPSILON;

    int i = 0;
    while (i < 64) {
        Node node = nodes.data[0];
        int scale_exp = 21;
        uint child_index = get_node_cell_index(position, scale_exp);

        while (!is_leaf(node) && (child_mask(node) >> child_index & 1) != 0) {
            uint node_index = child_ptr(node) + popcnt64(child_mask(node) & ((uint64_t(1) << child_index) - 1));
            node = nodes.data[node_index];

            scale_exp -= 2;
            child_index = get_node_cell_index(position, scale_exp);
        }

        if (is_leaf(node) && (child_mask(node) >> child_index & 1) != 0 && i != 0) {
            hit.hit = true;
            break;
        }

        // scale = 2.0 ^ (scaleExp - 23)
        float scale = float((scale_exp - 23 + 127) << 23);

        // int adv_scale_exp = scale_exp;
        // 0b101010 = 42
        // 0x00330033 = 3342387
        // if ((child_mask(node) >> (child_index & 42) & 3342387) == 0) adv_scale_exp++;

        vec3 cell_min = floor_scale(position, scale_exp);
        vec3 cell_size = vec3(scale);

        vec3 side_position = cell_min + step(0.0, ray.direction) * cell_size;
        vec3 side_distance = (side_position - ray.origin) * ray.inv_dir;

        float t = min(min(side_distance.x, side_distance.y), side_distance.z) + EPSILON;

        // vec3 neighbor_min = cell_min + vec3(
        //     abs(side_distance.x - t) < EPSILON ? sign(ray.direction.x) * scale : 0,
        //     abs(side_distance.y - t) < EPSILON ? sign(ray.direction.y) * scale : 0,
        //     abs(side_distance.z - t) < EPSILON ? sign(ray.direction.z) * scale : 0
        // );
        // vec3 neighbor_max = neighbor_min + cell_size;
        // position = clamp(origin + ray.direction * t, neighbor_min, neighbor_max);
        position = origin + ray.direction * t;

        i++;
    }

    return hit;
}

void main() {
    ivec2 location = ivec2(int(gl_GlobalInvocationID.x), int(gl_GlobalInvocationID.y));

    uint64_t start = clockARB();

    Ray ray = get_primary_ray(location);
    HitInfo result = ray_cast(ray);

    uint64_t elapsed = clockARB() - start;

    vec3 albedo = ray.direction;

    if (result.hit) {
        albedo = vec3(1.0);
    }

    memoryBarrierBuffer();
    imageStore(image, location, vec4(albedo, 1.0));
}