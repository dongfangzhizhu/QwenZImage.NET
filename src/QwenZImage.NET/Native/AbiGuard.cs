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
