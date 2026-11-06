using Microsoft.Win32.SafeHandles;

namespace QwenZImage.NET.Native;

/// <summary>Owns the native qzi_engine pointer and destroys it exactly once.</summary>
internal sealed class QziEngineHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal QziEngineHandle(IntPtr handle) : base(ownsHandle: true) => SetHandle(handle);

    protected override bool ReleaseHandle()
    {
        NativeMethods.EngineDestroy(handle);
        return true;
    }
}
