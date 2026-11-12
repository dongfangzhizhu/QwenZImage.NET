using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Bridges managed progress / cancellation to the native callback pair using
/// static <see cref="UnmanagedCallersOnlyAttribute"/> entry points. The opaque
/// user pointer is an integer token into a process-wide live-scope table, NOT
/// a raw <see cref="GCHandle"/>: a late or stale callback from a native worker
/// can therefore never dereference a freed handle (which used to be able to
/// surface as a fatal <c>System.ExecutionEngineException</c>, HResult
/// 0x80131506, at the generation P/Invoke call site).
/// </summary>
/// <remarks>
/// <para>
/// An <c>[UnmanagedCallersOnly]</c> method is entered from native code. A
/// managed exception escaping from it has to unwind through arbitrary native
/// frames (CUDA / ggml / C++): depending on the backend and the debugger that
/// can propagate normally, fail fast, or corrupt the CLR. Both callback entry
/// points therefore catch <em>every</em> managed exception, remember the first
/// one, and ask the engine to stop. <see cref="TakeFault"/> rethrows it in
/// managed code after the P/Invoke returns, where normal exception handling
/// applies.
/// </para>
/// <para>
/// No delegate allocations, no reverse P/Invoke stubs and no per-step string
/// allocations remain on the hot path.
/// </para>
/// </remarks>
internal sealed unsafe class CallbackScope : IDisposable
{
    private static readonly ConcurrentDictionary<long, CallbackState> Live = new();
    private static long _nextToken;

    private readonly long _token;
    private readonly CallbackState _state;
    private int _disposed;

    public QziCallbacks Callbacks { get; }

    public CallbackScope(IProgress<GenerationProgress>? progress,
                         CancellationToken cancellationToken)
    {
        _state = new CallbackState(progress, cancellationToken);
        _token = Interlocked.Increment(ref _nextToken);
        Live[_token] = _state;
        Callbacks = new QziCallbacks
        {
            Progress = (IntPtr)(delegate* unmanaged[Cdecl]<byte*, int, int, double, void*, void>)&OnProgress,
            ShouldCancel = (IntPtr)(delegate* unmanaged[Cdecl]<void*, int>)&OnShouldCancel,
            // An integer token, deliberately not GCHandle.ToIntPtr(). Native
            // code only ever round-trips this opaque value.
            User = (IntPtr)_token
        };
    }

    private sealed class CallbackState(IProgress<GenerationProgress>? progress,
                                       CancellationToken cancellationToken)
    {
        public IProgress<GenerationProgress>? Progress { get; } = progress;
        public CancellationToken CancellationToken { get; } = cancellationToken;
        public Exception? Fault;
    }

    /// <summary>
    /// Returns the first callback exception (clearing it) so the caller can
    /// rethrow it in managed code. Called after the native call has returned.
    /// </summary>
    public Exception? TakeFault() => Interlocked.Exchange(ref _state.Fault, null);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 1) return;
        Live.TryRemove(_token, out _);
        GC.SuppressFinalize(this);
    }

    private static CallbackState? Resolve(void* user)
    {
        if (user is null) return null;
        long token = (long)(IntPtr)user;
        // A stale, reused or garbage token simply resolves to nothing. No
        // GCHandle.FromIntPtr() on a possibly freed handle, no CLR crash.
        return token > 0 && Live.TryGetValue(token, out CallbackState? state) ? state : null;
    }

    private static void Record(CallbackState state, Exception exception)
    {
        // Only the first failure is reported; subsequent progress ticks keep
        // the engine's shutdown path simple.
        Interlocked.CompareExchange(ref state.Fault, exception, null);
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static void OnProgress(byte* phase, int completed, int total,
                                   double elapsedSeconds, void* user)
    {
        try
        {
            CallbackState? state = Resolve(user);
            if (state?.Progress is null) return;
            state.Progress.Report(new GenerationProgress(
                ParsePhase(phase), completed, total, TimeSpan.FromSeconds(elapsedSeconds)));
        }
        catch (Exception ex)
        {
            // NEVER let a managed exception escape into the native frame.
            // The cancellation callback below will then stop the engine and
            // QziClient.Invoke rethrows this fault on the managed side.
            if (Resolve(user) is { } state) Record(state, ex);
        }
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static int OnShouldCancel(void* user)
    {
        try
        {
            CallbackState? state = Resolve(user);
            if (state is null) return 0;
            // A failed progress handler cannot be recovered from inside native
            // code: stop the denoise loop and surface the fault to the caller.
            if (Volatile.Read(ref state.Fault) is not null) return 1;
            return state.CancellationToken.IsCancellationRequested ? 1 : 0;
        }
        catch (Exception ex)
        {
            if (Resolve(user) is { } state) Record(state, ex);
            return 1;
        }
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
