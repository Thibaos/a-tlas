struct Material {
    uvec2 data;
};

vec3 get_material_color(Material mat) {
    uvec3 mask = uvec3(31, 63, 31);
    vec3 color = vec3(mat.data.xxx >> uvec3(11, 5, 0) & mask) / vec3(mask);
    return color * color;
}