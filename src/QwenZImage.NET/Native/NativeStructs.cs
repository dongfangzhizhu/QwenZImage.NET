using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>Layout slot indices reported by <c>qzi_abi_layout</c>.</summary>
internal static class LayoutSlots
{
    internal const int EngineConfig = 0;
    internal const int Qwen21Request = 1;
    internal const int ZImageRequest = 2;
    internal const int Callbacks = 3;
    internal const int Image = 4;
    internal const int Health = 5;
    internal const int Count = 6;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziEngineConfig
{
    public uint AbiVersion;
    public byte* Qwen21Dir;
    public byte* ZimageDir;
    public byte* CacheDir;
    public int DeviceId;
    public int ThreadCount;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziQwen21Request
{
    public byte* Prompt;
    public byte* NegativePrompt;
    public byte* Tier;
    public byte* Lora;
    public int Width;
    public int Height;
    public int Steps;
    public float Guidance;
    public float LoraScale;
    public ulong Seed;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziZImageRequest
{
    public byte* Prompt;
    public byte* Checkpoint;
    public int Width;
    public int Height;
    public ulong Seed;
    public uint Flags;
}

/// <summary>Function pointers are passed as IntPtr so the source-generated
/// marshaller treats the whole struct as blittable (no marshalling stubs).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct QziCallbacks
{
    public IntPtr Progress;
    public IntPtr ShouldCancel;
    public IntPtr User;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziImage
{
    public byte* Data;
    public nuint Size;
    public uint Width;
    public uint Height;
    public ulong Token;
}

[StructLayout(LayoutKind.Sequential)]
internal struct QziHealth
{
    public int ResidentModel;
    public int Busy;
    public int QueueDepth;
    public int DeviceId;
    public long FreeVramMib;
    public long TotalVramMib;
    public ulong Served;
    public ulong Rejected;
    public double UptimeSeconds;
    public int ReferenceBackend;
}
