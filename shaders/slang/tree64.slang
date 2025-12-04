import "./common";

struct DispatchParams
{
    VoxelTree scene;
    float4x4 inv_proj_mat;
    float3 camera_origin;
    [vk::image_format("rgba8")]
    WTexture2D<float4> output_albedo_tex;
};

[vk::push_constant]
DispatchParams pc;

public struct VoxelTree : IRayCastable
{
    Node *NodePool;
    uint8_t *LeafData;
    Material *Palette;

    public HitInfo RayCast(float3 origin, float3 dir, bool coarse)
    {
        uint groupId = 0;
        // uint groupId = spirv_asm
        // { result:$$uint = OpLoad builtin(LocalInvocationIndex:uint); };

        static groupshared uint gs_stack[64][11];
        // uint *stack = &gs_stack[groupId][0];
        int scaleExp = 21; // 0.25

        uint nodeIdx = 0; // root
        Node node = NodePool[nodeIdx];

        // Mirror coordinates to negative ray octant to simplify cell intersections
        uint mirrorMask = 0;
        if (dir.x > 0)
            mirrorMask |= 3 << 0;
        if (dir.y > 0)
            mirrorMask |= 3 << 4;
        if (dir.z > 0)
            mirrorMask |= 3 << 2;

        origin = GetMirroredPos(origin, dir, true);

        // Clamp to prevent traversal from completely breaking for rays starting outside tree
        float3 pos = clamp(origin, 1.0f, 1.9999999f);
        float3 invDir = 1.0 / -abs(dir);

        float3 sideDist;
        int i;

        for (i = 0; i < 256; i++)
        {
            if (coarse && i > 20 && node.IsLeaf)
                break;

            uint childIdx = GetNodeCellIndex(pos, scaleExp) ^ mirrorMask;

            // Descend
            while ((node.ChildMask >> childIdx & 1) != 0 && !node.IsLeaf)
            {
                gs_stack[groupId][scaleExp >> 1] = nodeIdx;

                nodeIdx = node.ChildPtr + popcnt_var64(node.ChildMask, childIdx);
                node = NodePool[nodeIdx];

                scaleExp -= 2;
                childIdx = GetNodeCellIndex(pos, scaleExp) ^ mirrorMask;
            }
            if ((node.ChildMask >> childIdx & 1) != 0 && node.IsLeaf)
                break;

            // 2³ steps
            int advScaleExp = scaleExp;
            if ((node.ChildMask >> (childIdx & 0b101010) & 0x00330033) == 0)
                advScaleExp++;

            // Compute next pos by intersecting with max cell sides
            float3 cellMin = FloorScale(pos, advScaleExp);

            sideDist = (cellMin - origin) * invDir;
            float tmax = min(min(sideDist.x, sideDist.y), sideDist.z);

            int3 neighborMax = asint(cellMin) + select(sideDist == tmax, -1, (1 << advScaleExp) - 1);
            pos = min(origin - abs(dir) * tmax, asfloat(neighborMax));

            // Find common ancestor based on left-most carry bit
            // We only care about changes in the exponent and high bits of
            // each cell position (10'10'10'...), so the odd bits are masked.
            uint3 diffPos = asuint(pos) ^ asuint(cellMin);
            int diffExp = firstbithigh((diffPos.x | diffPos.y | diffPos.z) & 0xFFAAAAAA); // 31 - lzcnt, or findMSB in GLSL

            if (diffExp > scaleExp)
            {
                scaleExp = diffExp;
                if (diffExp > 21)
                    break; // going out of root?

                nodeIdx = gs_stack[groupId][scaleExp >> 1];
                node = NodePool[nodeIdx];
            }
        }

        HitInfo hit;
        hit.MaterialId = 0;

        if (node.IsLeaf && scaleExp <= 21)
        {
            pos = GetMirroredPos(pos, dir, false);
            uint childIdx = GetNodeCellIndex(pos, scaleExp);

            hit.MaterialId = LeafData[node.ChildPtr + popcnt_var64(node.ChildMask, childIdx)];
            hit.Pos = pos;

            float tmax = min(min(sideDist.x, sideDist.y), sideDist.z);
            bool3 sideMask = tmax >= sideDist;
            hit.Normal = select(sideMask, -sign(dir), 0.0);
        }
        return hit;
    }
}

struct Node
{
    uint packed_data[3];

    property bool IsLeaf
    {
        get {
            return (packed_data[0] & 1) != 0;
        }
    }
    property uint ChildPtr
    {
        get {
            return packed_data[0] >> 1;
        }
    }
    property uint64_t ChildMask
    {
        get {
            return packed_data[1] | uint64_t(packed_data[2]) << 32;
        }
    }
}

void GetPrimaryRay(int2 screenPos, out float3 rayPos, out float3 rayDir)
{
    uint width, height, depth;
    pc.output_albedo_tex.GetDimensions(0u, width, height, depth);
    // nit: UV re-scaling and anti-alias jitter can be pre-baked in the matrix.
    float2 uv = (screenPos + 0.5) / float2(width, height);
    uv = uv * 2 - 1;
    float4 far = mul(pc.inv_proj_mat, float4(uv, 1, 1));
    rayDir = normalize(far.xyz / far.w);
    rayPos = pc.camera_origin;
}

static int GetNodeCellIndex(float3 pos, int scaleExp)
{
    uint3 cellPos = asuint(pos) >> scaleExp & 3;
    return cellPos.x + cellPos.z * 4 + cellPos.y * 16;
}

// floor(pos / scale) * scale
static float3 FloorScale(float3 pos, int scaleExp)
{
    uint mask = ~0u << scaleExp;
    return asfloat(asuint(pos) & mask); // erase bits lower than scale
}

// Reverses `pos` from range [1.0, 2.0) to (2.0, 1.0] if `dir > 0`.
static float3 GetMirroredPos(float3 pos, float3 dir, bool rangeCheck)
{
    float3 mirrored = asfloat(asuint(pos) ^ 0x7FFFFF);
    // XOR-ing will only work for coords in range [1.0, 2.0),
    // fallback to subtractions if that's not the case.
    if (rangeCheck && any(pos < 1.0 || pos >= 2.0))
        mirrored = 3.0 - pos;
    return select(dir > 0, mirrored, pos);
}

// Count number of set bits in variable range [0..width]
static uint popcnt_var64(uint64_t mask, uint width)
{
    // return popcnt64(mask & ((1ull << mask) - 1));
    uint himask = uint(mask);
    uint count = 0;

    if (width >= 32)
    {
        count = countbits(himask);
        himask = uint(mask >> 32);
    }
    uint m = 1u << (width & 31u);
    count += countbits(himask & (m - 1u));
    return count;
}

float3 GetRandomSphereDir()
{
    float2 u = (NextRandomU32() >> uint2(0, 16) & 65535) / 65536.0;

    float phi = u.x * 6.283185307179586;
    float y = u.y * 2 - 1;
    float r = sqrt(1.0 - y * y);
    return float3(sin(phi) * r, y, cos(phi) * r);
}

// PCG random
static uint g_RandomSeed = 0;

uint NextRandomU32()
{
    uint state = g_RandomSeed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return g_RandomSeed = (word >> 22u) ^ word;
}

void InitRandom(uint2 pos, uint frameNo)
{
    g_RandomSeed = pos.x ^ (pos.y << 16);
    NextRandomU32();
    g_RandomSeed ^= frameNo * 1234;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pos: SV_DispatchThreadID)
{
    uint w, h, d;
    pc.output_albedo_tex.GetDimensions(0, w, h, d);

    // Needed in case texture size is not aligned to workgroup size.
    if (any(pos >= uint2(w, h)))
        return;

    float3 rayPos, rayDir;
    GetPrimaryRay(pos, rayPos, rayDir);

    HitInfo hit = pc.scene.RayCast(rayPos, rayDir, false);
    float3 albedo;

    if (!hit.Miss)
    {
        let material = pc.scene.Palette[hit.MaterialId];
        albedo = material.Color;
    }
    else
    {
        albedo = rayDir;
    }

    pc.output_albedo_tex.Store(pos, float4(albedo, 1));
}
