using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Bridges managed progress / cancellation to the native callback pair using
/// static <see cref="UnmanagedCallersOnlyAttribute"/> entry points and a
/// pinned <see cref="GCHandle"/> as the opaque user pointer. No delegate
/// allocations, no reverse P/Invoke stubs, no per-step string allocations.
/// </summary>
internal sealed unsafe class CallbackScope : IDisposable
{
    private GCHandle _stateHandle;

    public QziCallbacks Callbacks { get; }

    public CallbackScope(IProgress<GenerationProgress>? progress,
                         CancellationToken cancellationToken)
    {
        var state = new CallbackState(progress, cancellationToken);
        _stateHandle = GCHandle.Alloc(state);
        Callbacks = new QziCallbacks
        {
            Progress = (IntPtr)(delegate* unmanaged[Cdecl]<byte*, int, int, double, void*, void>)&OnProgress,
            ShouldCancel = (IntPtr)(delegate* unmanaged[Cdecl]<void*, int>)&OnShouldCancel,
            User = GCHandle.ToIntPtr(_stateHandle)
        };
    }

    private sealed class CallbackState(IProgress<GenerationProgress>? progress,
                                       CancellationToken cancellationToken)
    {
        public IProgress<GenerationProgress>? Progress { get; } = progress;
        public CancellationToken CancellationToken { get; } = cancellationToken;
    }

    public void Dispose()
    {
        if (_stateHandle.IsAllocated) _stateHandle.Free();
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static void OnProgress(byte* phase, int completed, int total,
                                   double elapsedSeconds, void* user)
    {
        if (user is null) return;
        if (GCHandle.FromIntPtr((IntPtr)user).Target is not CallbackState state) return;
        state.Progress?.Report(new GenerationProgress(
            ParsePhase(phase), completed, total, TimeSpan.FromSeconds(elapsedSeconds)));
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static int OnShouldCancel(void* user)
    {
        if (user is null) return 0;
        if (GCHandle.FromIntPtr((IntPtr)user).Target is not CallbackState state) return 0;
        return state.CancellationToken.IsCancellationRequested ? 1 : 0;
    }

    private static GenerationPhase ParsePhase(byte* phase)
    {
        if (phase is null) return GenerationPhase.Unknown;
        ReadOnlySpan<byte> text = MemoryMarshal.CreateReadOnlySpanFromNullTerminated(phase);
        if (text.SequenceEqual("model load"u8)) return GenerationPhase.ModelLoad;
        if (text.SequenceEqual("text"u8)) return GenerationPhase.Text;
        if (text.SequenceEqual("image encode"u8)) return GenerationPhase.ImageEncode;
        if (text.SequenceEqual("dit"u8)) return GenerationPhase.Dit;
        if (text.SequenceEqual("vae"u8)) return GenerationPhase.Vae;
        if (text.SequenceEqual("encode"u8)) return GenerationPhase.Encode;
        return GenerationPhase.Unknown;
    }
}
