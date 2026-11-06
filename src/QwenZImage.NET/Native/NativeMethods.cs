using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Source-generated P/Invoke over the stable C ABI declared in
/// <c>native/include/qzi.h</c>. Every parameter is blittable: no reflection,
/// no runtime marshalling stubs, no string allocations on the generation path.
/// </summary>
internal static unsafe partial class NativeMethods
{
    internal const string LibraryName = "qzi";
    internal const uint ExpectedAbiVersion = 1;
    internal const int ErrorCapacity = 512;

    [LibraryImport(LibraryName, EntryPoint = "qzi_abi_version")]
    internal static partial uint AbiVersion();

    [LibraryImport(LibraryName, EntryPoint = "qzi_status_text")]
    internal static partial IntPtr StatusText(int status);

    [LibraryImport(LibraryName, EntryPoint = "qzi_abi_layout")]
    internal static partial int AbiLayout(nuint* sizes, nuint count);

    [LibraryImport(LibraryName, EntryPoint = "qzi_engine_create")]
    internal static partial int EngineCreate(QziEngineConfig* config,
                                             IntPtr* outEngine,
                                             byte* error, nuint errorCap);

    [LibraryImport(LibraryName, EntryPoint = "qzi_engine_destroy")]
    internal static partial void EngineDestroy(IntPtr engine);

    [LibraryImport(LibraryName, EntryPoint = "qzi_engine_health")]
    internal static partial int EngineHealth(IntPtr engine, QziHealth* health,
                                             byte* error, nuint errorCap);

    [LibraryImport(LibraryName, EntryPoint = "qzi_engine_unload")]
    internal static partial int EngineUnload(IntPtr engine,
                                             byte* error, nuint errorCap);

    [LibraryImport(LibraryName, EntryPoint = "qzi_generate_qwen21")]
    internal static partial int GenerateQwen21(IntPtr engine,
                                               QziQwen21Request* request,
                                               QziCallbacks* callbacks,
                                               QziImage* image,
                                               byte* error, nuint errorCap);

    [LibraryImport(LibraryName, EntryPoint = "qzi_generate_zimage")]
    internal static partial int GenerateZImage(IntPtr engine,
                                               QziZImageRequest* request,
                                               QziCallbacks* callbacks,
                                               QziImage* image,
                                               byte* error, nuint errorCap);

    [LibraryImport(LibraryName, EntryPoint = "qzi_image_release")]
    internal static partial void ImageRelease(IntPtr engine, QziImage* image);
}
