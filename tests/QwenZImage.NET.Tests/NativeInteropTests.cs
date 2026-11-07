using QwenZImage.NET.Native;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// End-to-end tests against the bundled deterministic reference backend:
/// same ABI, same phases, same buffer ownership as a real CUDA engine, but no
/// GPU, weights or network needed.
/// </summary>
public class NativeInteropTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    [Fact]
    public void AbiGuard_Matches_The_Native_Layout()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Assert.Equal(1u, client.AbiVersion);
        EngineHealth health = client.GetHealth();
        Assert.True(health.IsReferenceBackend);
        Assert.Equal(GenerationModel.None, health.ResidentModel);
        Assert.False(health.Busy);
    }

    [Fact]
    public void Qwen21_Generates_A_Png_And_Reports_Phases()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        List<GenerationProgress> progress = [];
        var collector = new CollectingProgress<GenerationProgress>(progress);

        using GenerationResult result = client.GenerateQwen21(
            new QwenImageOptions { Prompt = "a red fox in snow", Width = 64, Height = 64, Seed = 42 },
            collector);

        Assert.True(result.Length > 33);
        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.Equal(64, result.Width);
        Assert.Equal(64, result.Height);
        Assert.Equal(42ul, result.Seed);
        Assert.Equal(GenerationModel.Qwen21, result.Model);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Dit);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Vae);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Text);
        Assert.Equal(8, progress.Count(p => p.Phase == GenerationPhase.Dit));
    }

    [Fact]
    public void Identical_Request_Is_Byte_Identical()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        var options = new QwenImageOptions { Prompt = "a red fox", Width = 32, Height = 32, Seed = 7 };
        using GenerationResult a = client.GenerateQwen21(options);
        using GenerationResult b = client.GenerateQwen21(options);
        Assert.Equal(a.PngSpan.ToArray(), b.PngSpan.ToArray());
    }

    [Fact]
    public void ZImage_Generates_A_Png()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        using GenerationResult result = client.GenerateZImage(
            new ZImageOptions { Prompt = "a red fox", Width = 64, Height = 64, Seed = 9 });
        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.Equal(GenerationModel.ZImage, result.Model);
        Assert.Equal(GenerationModel.ZImage, client.GetHealth().ResidentModel);
    }

    [Fact]
    public void WriteTo_ZeroCopy_Matches_Generate()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        var options = new ZImageOptions { Prompt = "same", Width = 32, Height = 32, Seed = 3 };
        using GenerationResult expected = client.GenerateZImage(options);
        using MemoryStream stream = new();
        client.WriteTo(stream, options);
        Assert.Equal(expected.PngSpan.ToArray(), stream.ToArray());
    }

    [Fact]
    public void Health_Counters_Track_Served_And_Rejected()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        var options = new QwenImageOptions { Prompt = "a fox", Width = 32, Height = 32 };
        client.GenerateQwen21(options).Dispose();
        client.GenerateQwen21(options).Dispose();
        Assert.Throws<ArgumentException>(() => client.GenerateQwen21(options with { Width = 100 }));
        EngineHealth health = client.GetHealth();
        Assert.Equal(2ul, health.Served);
        Assert.Equal(0ul, health.Rejected); // managed validator rejects before the native call
        Assert.Equal(GenerationModel.Qwen21, health.ResidentModel);
        client.Unload();
    }

    [Fact]
    public async Task Async_Wrapper_Produces_The_Same_Bytes()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        var options = new QwenImageOptions { Prompt = "async fox", Width = 32, Height = 32 };
        using GenerationResult sync = client.GenerateQwen21(options);
        using GenerationResult asyncResult = await client.GenerateQwen21Async(options);
        Assert.Equal(sync.PngSpan.ToArray(), asyncResult.PngSpan.ToArray());
    }

    [Fact]
    public void PreCancelled_Token_Throws_Without_Calling_The_Engine()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        using CancellationTokenSource cts = new();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => client.GenerateQwen21(
            new QwenImageOptions { Prompt = "never runs", Width = 32, Height = 32 },
            cancellationToken: cts.Token));
        Assert.Equal(0ul, client.GetHealth().Served);
    }

    [Fact]
    public unsafe void Native_Layer_Rejects_A_Misaligned_Canvas()
    {
        // Bypasses the managed validator on purpose: the native ABI repeats
        // the checks for integrators that call qzi_* directly from C/C++.
        NativeLibraryLoader.Configure(null);
        AbiGuard.Verify();
        using var dir = new Utf8PinnedString(".");
        QziEngineConfig config = new()
        {
            AbiVersion = NativeMethods.ExpectedAbiVersion,
            Qwen21Dir = dir.Pointer,
            ZimageDir = dir.Pointer,
            CacheDir = dir.Pointer,
            DeviceId = -1
        };
        byte* error = stackalloc byte[NativeMethods.ErrorCapacity];
        IntPtr engine = IntPtr.Zero;
        Assert.Equal(0, NativeMethods.EngineCreate(&config, &engine, error,
                                                   NativeMethods.ErrorCapacity));
        try
        {
            using var prompt = new Utf8PinnedString("a red fox");
            QziQwen21Request request = new()
            {
                Prompt = prompt.Pointer,
                Width = 100,
                Height = 64,
                Steps = 8,
                Guidance = 1.0f,
                LoraScale = 0.0f,
                Seed = 1
            };
            QziCallbacks callbacks = default;
            QziImage image = default;
            int status = NativeMethods.GenerateQwen21(engine, &request, &callbacks,
                                                      &image, error,
                                                      NativeMethods.ErrorCapacity);
            Assert.Equal((int)QziStatus.InvalidArgument, status);
            Assert.Contains("multiples of 32", NativeError.Read(error));
        }
        finally
        {
            NativeMethods.EngineDestroy(engine);
        }
    }

    private sealed class CollectingProgress<T>(List<T> items) : IProgress<T>
    {
        public void Report(T value) => items.Add(value);
    }
}
