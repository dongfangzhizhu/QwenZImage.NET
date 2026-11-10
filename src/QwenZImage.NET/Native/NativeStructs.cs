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

/// <summary>
/// Additive ABI v1 image-edit request: the ordinary qwen21 request followed by
/// the reference/mask pointers and the two WanGP strengths. Field order and
/// packing must match <c>qzi_qwen21_edit_request</c> in native/include/qzi.h;
/// <see cref="AbiGuard"/> verifies sizeof() against qzi_qwen21_edit_layout().
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziQwen21EditRequest
{
    public QziQwen21Request Base;
    public byte* ReferenceRgb;
    public uint ReferenceWidth;
    public uint ReferenceHeight;
    public byte* Mask;
    public float Strength;
    public float MaskingStrength;
    public uint Flags;
}

/// <summary>
/// Third additive ABI v1 request: the ordinary image-edit request followed by
/// WanGP's Red Canvas margins, the LanPaint model mode, the RGBA switch and
/// reserved flags. Field order and packing must match
/// <c>qzi_qwen21_edit_ex_request</c> in native/include/qzi.h;
/// <see cref="AbiGuard"/> verifies sizeof() against qzi_qwen21_edit_ex_layout().
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziQwen21EditExRequest
{
    public QziQwen21EditRequest Base;
    public uint MarginTop;
    public uint MarginBottom;
    public uint MarginLeft;
    public uint MarginRight;
    public uint ModelMode;
    public uint Rgba;
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

/// <summary>
/// Additive ABI v1 Z-Image Turbo Fun ControlNet Union request: the ordinary
/// zimage request followed by the planar control image, its dimensions, the
/// control type string, the control scale and reserved flags. Field order and
/// packing must match <c>qzi_zimage_control_request</c> in native/include/qzi.h;
/// <see cref="AbiGuard"/> verifies sizeof() against qzi_zimage_control_layout().
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct QziZImageControlRequest
{
    public QziZImageRequest Base;
    public byte* ControlRgb;
    public uint ControlWidth;
    public uint ControlHeight;
    public byte* ControlType;
    public float ControlScale;
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
