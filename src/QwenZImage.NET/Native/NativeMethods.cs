using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Source-generated P/Invoke over the stable C ABI declared in
/// <c>native/include/qzi.h</c>. Every parameter is blittable: no reflection,
/// no runtime marshalling stubs, no string allocations on the generation path.
/// </summary>
internal static unsafe partial class NativeMethods
{
    // Platform-neutral name: resolves to QwenZImageNative.dll on Windows,
    // libQwenZImageNative.so on Linux and libQwenZImageNative.dylib on macOS.
    internal const string LibraryName = "QwenZImageNative";
    internal const uint ExpectedAbiVersion = 1;
    internal const int ErrorCapacity = 512;

    /// <summary>Capability bit for Qwen-Image 2.1 image editing.</summary>
    internal const uint FeatureQwen21Edit = 0x1u;

    /// <summary>Capability bit for Z-Image Turbo Fun ControlNet Union.</summary>
    internal const uint FeatureZImageControl = 0x2u;

    /// <summary>
    /// Capability bit for advanced Qwen-Image 2.1 editing (WanGP outpainting,
    /// LanPaint model modes and RGBA output). Added without bumping the ABI.
    /// </summary>
    internal const uint FeatureQwen21EditAdv = 0x4u;

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

    /// <summary>
    /// Additive in ABI v1. Older libraries do not export this symbol, so the
    /// source-generated stub is resolved lazily: never call it until
    /// <see cref="AbiGuard.Qwen21EditSupported"/> has confirmed the export
    /// exists (otherwise an EntryPointNotFoundException escapes).
    /// </summary>
    [LibraryImport(LibraryName, EntryPoint = "qzi_generate_qwen21_edit")]
    internal static partial int GenerateQwen21Edit(IntPtr engine,
                                                   QziQwen21EditRequest* request,
                                                   QziCallbacks* callbacks,
                                                   QziImage* image,
                                                   byte* error, nuint errorCap);

    /// <summary>
    /// Additive in ABI v1. Older libraries do not export this symbol, so the
    /// source-generated stub is resolved lazily: never call it until
    /// <see cref="AbiGuard.ZImageControlSupported"/> has confirmed the export
    /// exists (otherwise an EntryPointNotFoundException escapes). The layout
    /// export <c>qzi_zimage_control_layout</c> is probed on demand with
    /// <c>NativeLibrary.TryGetExport</c> in <see cref="AbiGuard"/> rather than
    /// bound here, matching the image-edit capability probe.
    /// </summary>
    [LibraryImport(LibraryName, EntryPoint = "qzi_generate_zimage_control")]
    internal static partial int GenerateZImageControl(IntPtr engine,
                                                      QziZImageControlRequest* request,
                                                      QziCallbacks* callbacks,
                                                      QziImage* image,
                                                      byte* error, nuint errorCap);

    /// <summary>
    /// Additive in ABI v1, third increment. Older libraries do not export this
    /// symbol, so never call it until
    /// <see cref="AbiGuard.Qwen21EditAdvSupported"/> has confirmed the export
    /// exists. The matching <c>qzi_qwen21_edit_ex_layout</c> is probed on
    /// demand with <c>NativeLibrary.TryGetExport</c> in <see cref="AbiGuard"/>.
    /// </summary>
    [LibraryImport(LibraryName, EntryPoint = "qzi_generate_qwen21_edit_ex")]
    internal static partial int GenerateQwen21EditEx(IntPtr engine,
                                                      QziQwen21EditExRequest* request,
                                                      QziCallbacks* callbacks,
                                                      QziImage* image,
                                                      byte* error, nuint errorCap);

    [LibraryImport(LibraryName, EntryPoint = "qzi_image_release")]
    internal static partial void ImageRelease(IntPtr engine, QziImage* image);
}
