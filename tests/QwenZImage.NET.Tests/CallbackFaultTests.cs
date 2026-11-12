using System.Runtime.InteropServices;
using QwenZImage.NET.Native;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Regression tests for the fatal <c>System.ExecutionEngineException</c>
/// (HResult 0x80131506) that used to surface at the generation P/Invoke call
/// site. Root cause: an <see cref="IProgress{T}"/> handler (typically an SSE
/// subscriber) throwing inside an <c>[UnmanagedCallersOnly]</c> progress
/// callback let a managed exception unwind through CUDA / ggml native frames,
/// and a late callback could also dereference an already-freed GCHandle.
///
/// The callback boundary now catches every managed exception, stores it and
/// asks the engine to stop; <c>QziClient.Invoke</c> rethrows it in managed
/// code. These tests run against the GPU-free reference backend, so CI can
/// prove the contract without weights or a GPU.
/// </summary>
public class CallbackFaultTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private sealed class ThrowingProgress : IProgress<GenerationProgress>
    {
        public int Calls;

        public void Report(GenerationProgress value)
        {
            Interlocked.Increment(ref Calls);
            throw new InvalidOperationException("simulated SSE subscriber failure");
        }
    }

    private sealed class CountingProgress : IProgress<GenerationProgress>
    {
        public int Calls;

        public void Report(GenerationProgress value) => Interlocked.Increment(ref Calls);
    }

    private static QwenImageOptions Options() => new()
    {
        Prompt = "a red fox in snow",
        Width = 32,
        Height = 32,
        Seed = 42
    };

    [Fact]
    public void Throwing_Progress_Handler_Is_Contained_And_Rethrown_In_Managed_Code()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        var progress = new ThrowingProgress();

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => client.GenerateQwen21(Options(), progress));

        Assert.Contains("progress callback", error.Message);
        Assert.IsType<InvalidOperationException>(error.InnerException);
        Assert.Equal("simulated SSE subscriber failure", error.InnerException!.Message);
        Assert.True(progress.Calls >= 1, "the throwing handler must actually have been invoked");

        // The CLR and the engine survived: health is still queryable and the
        // single-flight slot was released rather than left busy. A cancelled
        // run does not increment `served` by design, so Busy is the assertion
        // that matters here; Engine_Remains_Usable_After_A_Callback_Fault
        // proves the engine can still produce images.
        EngineHealth health = client.GetHealth();
        Assert.False(health.Busy);
    }

    [Fact]
    public void Engine_Remains_Usable_After_A_Callback_Fault()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        QwenImageOptions options = Options();

        Assert.Throws<InvalidOperationException>(
            () => client.GenerateQwen21(options, new ThrowingProgress()));

        var progress = new CountingProgress();
        using GenerationResult result = client.GenerateQwen21(options, progress);
        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.True(progress.Calls > 1, "later generations must still report progress");
        Assert.Equal(32, result.Width);
        Assert.Equal(32, result.Height);
    }

    [Fact]
    public void Callback_Fault_Does_Not_Change_The_Deterministic_Output()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        QwenImageOptions options = Options();

        using GenerationResult baseline = client.GenerateQwen21(options);
        byte[] expected = baseline.PngSpan.ToArray();

        Assert.Throws<InvalidOperationException>(
            () => client.GenerateQwen21(options, new ThrowingProgress()));

        using GenerationResult after = client.GenerateQwen21(options);
        Assert.Equal(expected, after.PngSpan.ToArray());
    }

    [Fact]
    public unsafe void Callback_Fault_Is_Recorded_And_Stops_The_Engine()
    {
        using var scope = new CallbackScope(new ThrowingProgress(), CancellationToken.None);
        QziCallbacks callbacks = scope.Callbacks;

        var onProgress = (delegate* unmanaged[Cdecl]<byte*, int, int, double, void*, void>)callbacks.Progress;
        var onShouldCancel = (delegate* unmanaged[Cdecl]<void*, int>)callbacks.ShouldCancel;

        fixed (byte* phase = "dit\0"u8)
            onProgress(phase, 1, 8, 0.25, (void*)callbacks.User);

        // The native loop polls ShouldCancel: a recorded callback fault must
        // stop generation even when the caller's CancellationToken is idle.
        Assert.Equal(1, onShouldCancel((void*)callbacks.User));

        Exception fault = Assert.IsType<InvalidOperationException>(scope.TakeFault());
        Assert.Equal("simulated SSE subscriber failure", fault.Message);
        Assert.Null(scope.TakeFault()); // faults are delivered exactly once
    }

    [Fact]
    public unsafe void Stale_Token_Callbacks_Are_Ignored_Instead_Of_Dereferencing_A_Freed_Handle()
    {
        QziCallbacks callbacks;
        using (var scope = new CallbackScope(new CountingProgress(), CancellationToken.None))
            callbacks = scope.Callbacks; // scope is disposed here; token is now stale

        var onProgress = (delegate* unmanaged[Cdecl]<byte*, int, int, double, void*, void>)callbacks.Progress;
        var onShouldCancel = (delegate* unmanaged[Cdecl]<void*, int>)callbacks.ShouldCancel;

        // A late callback from a native worker must be a harmless no-op. With
        // the old GCHandle-based user pointer this could crash the CLR.
        fixed (byte* phase = "dit\0"u8)
            onProgress(phase, 1, 8, 0.0, (void*)callbacks.User);
        Assert.Equal(0, onShouldCancel((void*)callbacks.User));

        // A garbage / negative pointer is equally harmless.
        onProgress(null, 1, 8, 0.0, (void*)(-1));
        Assert.Equal(0, onShouldCancel((void*)(-1)));
    }

    [Fact]
    public unsafe void Cancellation_Still_Works_Through_The_Token_Table()
    {
        using CancellationTokenSource cts = new();
        using var scope = new CallbackScope(new CountingProgress(), cts.Token);
        QziCallbacks callbacks = scope.Callbacks;
        var onShouldCancel = (delegate* unmanaged[Cdecl]<void*, int>)callbacks.ShouldCancel;

        Assert.Equal(0, onShouldCancel((void*)callbacks.User));
        cts.Cancel();
        Assert.Equal(1, onShouldCancel((void*)callbacks.User));
        Assert.Null(scope.TakeFault());
    }

    [Fact]
    public async Task Queue_Publish_Swallows_A_Broken_Observer()
    {
        // The WebUI job queue publishes progress synchronously on the native
        // generation thread. A throwing event subscriber must never abort the
        // job or escape into the native frame.
        using QziGenerationQueue queue = new(new RecordingExecutor());
        queue.JobChanged += (_, _) => throw new InvalidOperationException("broken observer");

        QziJob job = queue.Submit(Options());
        Assert.NotNull(job.Request); // the dispatcher may already have started it

        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));
        Assert.All(queue.ListJobs(), j => Assert.Equal(QziJobState.Done, j.State));
        Assert.All(queue.ListJobs(), j => Assert.NotNull(j.OutputLength));
    }
}
