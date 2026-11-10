using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Checks the native ABI version and every POD struct size once per process.
/// A packing mismatch would otherwise silently corrupt the request struct, so
/// this fails fast with an actionable message.
/// </summary>
internal static unsafe class AbiGuard
{
    private static int _verified;
    // 0 = not probed, 1 = no editing support, 2 = editing support verified.
    private static int _editState;
    // 0 = not probed, 1 = no ControlNet support, 2 = ControlNet support verified.
    private static int _controlState;

    internal static void Verify()
    {
        if (Volatile.Read(ref _verified) == 1) return;

        uint nativeVersion = NativeMethods.AbiVersion();
        if (nativeVersion != NativeMethods.ExpectedAbiVersion)
        {
            throw new QziException(QziStatus.Unsupported,
                $"Native qzi ABI version {nativeVersion} does not match the managed wrapper " +
                $"(expected {NativeMethods.ExpectedAbiVersion}). Rebuild the native library from " +
                "the matching native/include/qzi.h.");
        }

        nuint* sizes = stackalloc nuint[LayoutSlots.Count];
        int status = NativeMethods.AbiLayout(sizes, (nuint)LayoutSlots.Count);
        if (status != 0)
        {
            throw new QziException(QziStatus.InternalError,
                "qzi_abi_layout failed; the native library did not report struct sizes.");
        }

        Check("qzi_engine_config", sizes[LayoutSlots.EngineConfig], sizeof(QziEngineConfig));
        Check("qzi_qwen21_request", sizes[LayoutSlots.Qwen21Request], sizeof(QziQwen21Request));
        Check("qzi_zimage_request", sizes[LayoutSlots.ZImageRequest], sizeof(QziZImageRequest));
        Check("qzi_callbacks", sizes[LayoutSlots.Callbacks], sizeof(QziCallbacks));
        Check("qzi_image", sizes[LayoutSlots.Image], sizeof(QziImage));
        Check("qzi_health", sizes[LayoutSlots.Health], sizeof(QziHealth));

        Volatile.Write(ref _verified, 1);
    }

    /// <summary>
    /// True when the loaded library implements the additive ABI v1 image-edit
    /// surface. Editing was added WITHOUT bumping QZI_ABI_VERSION, so the only
    /// safe test is export discovery: an older v1 library has none of the three
    /// symbols. The result is cached per process (one library is loaded once).
    /// A library that declares the feature but reports a different edit struct
    /// size fails fast through <see cref="Check"/>.
    /// </summary>
    internal static bool Qwen21EditSupported
    {
        get
        {
            int state = Volatile.Read(ref _editState);
            if (state != 0) return state == 2;
            // A P/Invoke inside Verify() triggers the DllImportResolver, which
            // records the handle used by the export probe below.
            Verify();
            bool supported = ProbeEditFeatures();
            Volatile.Write(ref _editState, supported ? 2 : 1);
            return supported;
        }
    }

    private static bool ProbeEditFeatures()
    {
        IntPtr handle = NativeLibraryLoader.Handle;
        if (handle == IntPtr.Zero) return false;
        if (!NativeLibrary.TryGetExport(handle, "qzi_abi_features", out IntPtr features) ||
            !NativeLibrary.TryGetExport(handle, "qzi_qwen21_edit_layout", out IntPtr layout) ||
            !NativeLibrary.TryGetExport(handle, "qzi_generate_qwen21_edit", out _))
            return false;

        uint bits = ((delegate* unmanaged<uint>)features)();
        if ((bits & NativeMethods.FeatureQwen21Edit) == 0) return false;

        nuint nativeSize = ((delegate* unmanaged<nuint>)layout)();
        Check("qzi_qwen21_edit_request", nativeSize, sizeof(QziQwen21EditRequest));
        return true;
    }

    /// <summary>
    /// True when the loaded library implements the additive ABI v1 Z-Image
    /// Turbo Fun ControlNet Union surface. Like editing, this was added WITHOUT
    /// bumping QZI_ABI_VERSION, so the only safe test is export discovery: an
    /// older v1 library has none of the control symbols. The result is cached
    /// per process. A library that declares the feature but reports a different
    /// control struct size fails fast through <see cref="Check"/>.
    /// </summary>
    internal static bool ZImageControlSupported
    {
        get
        {
            int state = Volatile.Read(ref _controlState);
            if (state != 0) return state == 2;
            // A P/Invoke inside Verify() triggers the DllImportResolver, which
            // records the handle used by the export probe below.
            Verify();
            bool supported = ProbeControlFeatures();
            Volatile.Write(ref _controlState, supported ? 2 : 1);
            return supported;
        }
    }

    private static bool ProbeControlFeatures()
    {
        IntPtr handle = NativeLibraryLoader.Handle;
        if (handle == IntPtr.Zero) return false;
        if (!NativeLibrary.TryGetExport(handle, "qzi_abi_features", out IntPtr features) ||
            !NativeLibrary.TryGetExport(handle, "qzi_zimage_control_layout", out IntPtr layout) ||
            !NativeLibrary.TryGetExport(handle, "qzi_generate_zimage_control", out _))
            return false;

        uint bits = ((delegate* unmanaged<uint>)features)();
        if ((bits & NativeMethods.FeatureZImageControl) == 0) return false;

        nuint nativeSize = ((delegate* unmanaged<nuint>)layout)();
        Check("qzi_zimage_control_request", nativeSize, sizeof(QziZImageControlRequest));
        return true;
    }

    private static void Check(string name, nuint native, int managed)
    {
        if (native != (nuint)managed)
        {
            throw new QziException(QziStatus.Unsupported,
                $"ABI layout mismatch for {name}: native {native} bytes, managed {managed} bytes. " +
                "The struct declaration and native/include/qzi.h have drifted apart.");
        }
    }
}

/// <summary>Maps native status codes onto the managed exception.</summary>
internal static class NativeError
{
    internal static unsafe string Read(byte* buffer)
        => Marshal.PtrToStringUTF8((IntPtr)buffer) ?? string.Empty;

    internal static unsafe QziException ToException(int status, byte* buffer)
    {
        string message = Read(buffer);
        if (message.Length == 0)
        {
            IntPtr text = NativeMethods.StatusText(status);
            message = Marshal.PtrToStringUTF8(text) ?? $"native error {status}";
        }
        return new QziException((QziStatus)status, message);
    }
}
