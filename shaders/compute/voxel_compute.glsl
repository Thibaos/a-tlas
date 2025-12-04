#version 450

#extension GL_GOOGLE_include_directive : enable

#include <vulkano.glsl>

const float MAX_RANGE = 1000.0;
const float EPSILON = 0.01;
const float NO_HIT = -1000.0;
const uint SVO_DEPTH = 8;
const uint MAX_RAYCAST_ITERATIONS = 256;
const uint CAST_STACK_DEPTH = 23;
const uint STACK_SIZE = 128;

struct SVONode {
    uint index;
    uint size;
    uint depth;
    uint parent_ptr;
    vec3 position;
    vec3 color;
    uint children_ptr;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct AABB {
    vec3 min_corner;
    vec3 max_corner;
};

struct CastStack {
    uint top_index;
    uint[STACK_SIZE] node_indices;
    float[STACK_SIZE] t_values;
};

struct StackReadInfo {
    uint node_ptr;
    float t_max;
};

struct RayCastResult {
    float t;
    uint iterations;
    uint node;
    vec3 color;
};

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    StorageImageId image_id;
    StorageBufferId camera_buffer_id;
    StorageBufferId resolution_buffer_id;
    StorageBufferId voxel_buffer_id;
};

VKO_DECLARE_STORAGE_IMAGE(image, image2D, rgba8)

VKO_DECLARE_STORAGE_BUFFER(camera, Camera {
    vec3 position;
    vec3 direction;
    vec3 right;
    vec3 up;
    float fov;
})

VKO_DECLARE_STORAGE_BUFFER(resolution, Resolution {
    float x;
    float y;
})

VKO_DECLARE_STORAGE_BUFFER(voxel_buffer, Voxels {
    SVONode voxels[];
})

#define image vko_image(image, image_id)
#define camera vko_buffer(camera, camera_buffer_id)
#define resolution vko_buffer(resolution, resolution_buffer_id)
#define voxel_buffer vko_buffer(voxel_buffer, voxel_buffer_id)

vec4 white = vec4(1., 1., 1., 1.);
vec4 red = vec4(1., 0., 0., 1.);
vec4 black = vec4(0., 0., 0., 0.);

float intersect(Ray ray, AABB box) {
    double tx1 = (box.min_corner.x - ray.origin.x) * ray.direction.x;
    double tx2 = (box.max_corner.x - ray.origin.x) * ray.direction.x;

    double tmin = min(tx1, tx2);
    double tmax = max(tx1, tx2);

    double ty1 = (box.min_corner.y - ray.origin.y) * ray.direction.y;
    double ty2 = (box.max_corner.y - ray.origin.y) * ray.direction.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    double tz1 = (box.min_corner.z - ray.origin.z) * ray.direction.z;
    double tz2 = (box.max_corner.z - ray.origin.z) * ray.direction.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    if (tmax >= tmin) {
        return float(tmin);
    }

    return NO_HIT;
}

RayCastResult traverse(Ray ray) {
    SVONode node = voxel_buffer.voxels[0];

    RayCastResult res = RayCastResult(NO_HIT, 0, 0, vec3(((1.0 / ray.direction) + 1.0) / 2.0));

    uint[STACK_SIZE] default_uint;
    float[STACK_SIZE] default_float;
    CastStack stack = CastStack(0, default_uint, default_float);
    stack.node_indices[0] = 0;
    float root_half_size = float(node.size >> 1);
    stack.t_values[0] = intersect(ray, AABB(node.position - root_half_size, node.position + root_half_size));

    for (uint n = 0; n <= 100; n++) {
        uint parent_index = stack.node_indices[stack.top_index];
        float t = stack.t_values[stack.top_index];
        stack.top_index--;

        SVONode parent = voxel_buffer.voxels[parent_index];

        if (parent.depth >= SVO_DEPTH || parent.children_ptr == 0 || parent.size == 1) {
            // leaf
            res.color = parent.color;
            res.t = t;
            break;
        }

        if (t <= EPSILON) {
            continue;
        }

        uint children_ptr = parent.children_ptr;
        float min_t = 1000.;
        CastStack child_hit_list = CastStack(0, default_uint, default_float);

        for (uint child_index = children_ptr; child_index < children_ptr + 8; child_index++) {
            SVONode child = voxel_buffer.voxels[child_index];
            float child_half_size = float(child.size >> 1);
            AABB child_aabb = AABB(child.position - child_half_size, child.position + child_half_size);
            float child_t = intersect(ray, child_aabb);

            if (child_t > 0.0 && child_t < min_t) {
                min_t = child_t;
                child_hit_list.node_indices[child_hit_list.top_index] = child_index;
                child_hit_list.t_values[child_hit_list.top_index] = child_t;
                child_hit_list.top_index++;
            }
        }

        // push children
        for (uint j = 0; j < child_hit_list.top_index; j++) {
            stack.node_indices[stack.top_index] = child_hit_list.node_indices[j];
            stack.t_values[stack.top_index] = child_hit_list.t_values[j];
            stack.top_index++;
        }


        if (stack.top_index < 1 || stack.top_index >= STACK_SIZE) {
            break;
        }
    }

    return res;
}

vec4 render(vec2 pixel_pos) {
    vec2 screenPos = pixel_pos * 2. - 1.;
    float tanFovY = tan(camera.fov * 0.5);

    vec3 forward = vec3(camera.direction.x, camera.direction.y, camera.direction.z);
    vec3 right = vec3(camera.right.x, camera.right.y, camera.right.z) * tanFovY * resolution.x / resolution.y;
    vec3 up = -vec3(camera.up.x, camera.up.y, camera.up.z) * tanFovY;

    vec3 ray_origin = vec3(camera.position.x, camera.position.y, camera.position.z);
    vec3 ray_direction = forward + screenPos.x * right + screenPos.y * up;
    Ray ray = Ray(ray_origin, ray_direction);
    ray.direction = 1.0 / ray.direction;

    RayCastResult result = traverse(ray);

    return vec4(result.color, 1.0);
}

void main() {
    ivec2 location = ivec2(int(gl_GlobalInvocationID.x), int(gl_GlobalInvocationID.y));
    vec2 pixel = vec2(float(gl_GlobalInvocationID.x) / resolution.x, float(gl_GlobalInvocationID.y) / resolution.y);
    vec4 color = render(pixel);
    memoryBarrierBuffer();
    imageStore(image, location, color);
}