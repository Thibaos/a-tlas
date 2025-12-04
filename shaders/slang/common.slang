public interface IRayCastable
{
    public HitInfo RayCast(float3 origin, float3 dir, bool coarse);
}

public struct HitInfo
{
    public float Dist;
    public float3 Pos;
    public float3 Normal;
    public float2 FaceUV;
    public uint MaterialId;

    public property bool Miss
    {
        get {
            return MaterialId == 0;
        }
    }
}

public struct Material
{
    public uint2 Data;

    public property float3 Color
    {
        get {
            uint3 mask = uint3(31, 63, 31);
            float3 color = float3(Data.xxx >> uint3(11, 5, 0) & mask) / float3(mask);
            return color * color; // srgb gamma hack
        }
    }
    public property float Emission
    {
        get {
            return f16tof32(Data.x >> 16);
        }
    }
    public property float MetalFuzziness
    {
        get {
            return (Data.y & 255) / 255.0;
        }
    }
}
