#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

struct Node {
    uint is_leaf;
    uint child_ptr;
    uint child_mask_0;
    uint child_mask_1;
    // uint[3] packed_data;
};

// bool is_leaf(Node node) {
//     return (node.packed_data[0] & 1) != 0;
// }

// uint child_ptr(Node node) {
//     return node.packed_data[0] >> 1;
// }

// uint64_t child_mask(Node node) {
//     return node.packed_data[1] | uint64_t(node.packed_data[2]) << 32;
// }

bool is_leaf(Node node) {
    return bool(node.is_leaf);
}

uint child_ptr(Node node) {
    return node.child_ptr;
}

uint64_t child_mask(Node node) {
    return uint64_t(node.child_mask_0) | uint64_t(node.child_mask_1) << 32;
}