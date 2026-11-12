using System.Buffers.Binary;
using System.Reflection;
using System.Text;
using QwenZImage.NET;

namespace QwenZImage.NET.PackageVerification;

/// <summary>
/// Smoke test for the published NuGet package. Only a PackageReference is
/// allowed here; pointing at the repository sources would verify the wrong
/// binary. No GPU, model weights or config file are required.
/// </summary>
internal static class Program
{
    private const string ExpectedAssemblyVersion = "0.1.0.0";

    private static readonly byte[] PngSignature =
        { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

    private static readonly string[] RequiredTypes =
    {
        "QwenZImage.NET.QziClient",
        "QwenZImage.NET.QziClientOptions",
        "QwenZImage.NET.QziGenerationQueue",
        "QwenZImage.NET.QziPresets",
        "QwenZImage.NET.QziRasterImage",
        "QwenZImage.NET.QziException",
        "QwenZImage.NET.GenerationResult",
        "QwenZImage.NET.GenerationProgress",
        "QwenZImage.NET.EngineHealth",
        "QwenZImage.NET.QwenImageOptions",
        "QwenZImage.NET.ZImageOptions",
        "QwenZImage.NET.Qwen21EditExOptions",
        "QwenZImage.NET.ZImageControlOptions",
        "QwenZImage.NET.ZImageInpaintOptions",
    };

    private static int _failures;

    private static void Check(string name, bool passed, string? detail = null)
    {
        Console.WriteLine($"  [{(passed ? "PASS" : "FAIL")}] {name}{(detail is null ? "" : $"  ({detail})")}");
        if (!passed) _failures++;
    }

    private static async Task<int> Main()
    {
        Console.WriteLine("QwenZImage.NET - published NuGet package verification");
        Console.WriteLine("Package: QwenZImage.NET 0.1.0 via PackageReference (no repo ProjectReference)");
        Console.WriteLine();

        // Verify only what ships in the package; ignore machine-level overrides
        // that would point at a locally built native library or config file.
        Environment.SetEnvironmentVariable("QZI_NATIVE_LIBRARY", null);
        Environment.SetEnvironmentVariable("QZI_CONFIG", null);

        try
        {
            VerifyAssembly();
            VerifyPackageAssets();
            await VerifyEngineAsync();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"  [FAIL] unexpected {ex.GetType().Name}: {ex.Message}");
            _failures++;
        }

        Console.WriteLine();
        Console.WriteLine(_failures == 0 ? "ALL CHECKS PASSED" : $"{_failures} CHECK(S) FAILED");
        return _failures == 0 ? 0 : 1;
    }

    private static void VerifyAssembly()
    {
        Console.WriteLine("[1/3] Managed assembly");
        Assembly assembly = typeof(QziClient).Assembly;
        string? version = assembly.GetName().Version?.ToString();
        Check("assembly version is 0.1.0.0", version == ExpectedAssemblyVersion, version);

        string location = Path.GetFullPath(assembly.Location);
        string baseDir = Path.GetFullPath(AppContext.BaseDirectory);
        Check("loaded from restored package output",
              location.StartsWith(baseDir, StringComparison.OrdinalIgnoreCase), location);

        string[] missing = RequiredTypes.Where(t => assembly.GetType(t) is null).ToArray();
        Check($"{RequiredTypes.Length} public API types present", missing.Length == 0,
              missing.Length == 0 ? null : string.Join(", ", missing));
    }

    private static void VerifyPackageAssets()
    {
        Console.WriteLine("[2/3] Package RID assets");
        string baseDir = AppContext.BaseDirectory;
        (string Label, string Relative)[] assets =
        {
            ("win-x64 native",   Path.Combine("runtimes", "win-x64", "native", "QwenZImageNative.dll")),
            ("linux-x64 native", Path.Combine("runtimes", "linux-x64", "native", "libQwenZImageNative.so")),
        };
        foreach ((string label, string relative) in assets)
        {
            string path = Path.Combine(baseDir, relative);
            bool exists = File.Exists(path);
            Check(label, exists, exists ? $"{new FileInfo(path).Length} bytes" : "missing");
        }
    }

    private static async Task VerifyEngineAsync()
    {
        Console.WriteLine("[3/3] Native ABI, callbacks and generation (no GPU / weights needed)");

        // The bundled reference backend only requires non-empty model pack
        // paths and never opens the weights. A temporary empty directory keeps
        // this check independent of any QwenZImage.config.json on the machine.
        string modelRoot = Path.Combine(Path.GetTempPath(), $"qzi-verify-{Guid.NewGuid():N}");
        Directory.CreateDirectory(modelRoot);
        try
        {
            using QziClient client = new(new QziClientOptions
            {
                Qwen21Directory = modelRoot,
                ZImageDirectory = modelRoot,
                CacheDirectory = modelRoot,
            });
            Console.WriteLine($"    native library: {client.NativeLibraryPath ?? "runtime default probing"}");

            Check("ABI v1 negotiated", client.AbiVersion >= 1, $"v{client.AbiVersion}");
            EngineHealth health = client.GetHealth();
            Check("engine health readable", !health.Busy, $"served={health.Served}");
            Check("bundled reference backend", health.IsReferenceBackend,
                  health.IsReferenceBackend ? "deterministic, GPU-free" : "a real CUDA build is loaded");

            SyncProgress progress = new();
            using GenerationResult first = await client.GenerateZImageAsync(Request(42), progress);
            byte[] pngA = first.Png.ToArray();
            Check("progress callbacks crossed the native boundary", progress.Count > 0, $"{progress.Count} reports");
            Check("PNG buffer length matches result", first.Length == pngA.Length, $"{first.Length} bytes");
            Check("valid 64x64 PNG", IsPngWithSize(pngA, 64, 64));
            Check("result metadata matches request", first.Width == 64 && first.Height == 64,
                  $"{first.Width}x{first.Height}");

            using GenerationResult second = await client.GenerateZImageAsync(Request(42));
            Check("same seed produces identical bytes", pngA.AsSpan().SequenceEqual(second.Png.Span));

            using GenerationResult third = await client.GenerateZImageAsync(Request(43));
            Check("different seed changes bytes", !pngA.AsSpan().SequenceEqual(third.Png.Span));

            string saved = Path.Combine(modelRoot, "saved.png");
            await first.SaveAsync(saved);
            Check("SaveAsync writes the same bytes", File.ReadAllBytes(saved).AsSpan().SequenceEqual(pngA));
        }
        finally
        {
            try { Directory.Delete(modelRoot, recursive: true); }
            catch (IOException) { /* best effort */ }
            catch (UnauthorizedAccessException) { /* best effort */ }
        }
    }

    private static ZImageOptions Request(ulong seed) => new()
    {
        Prompt = "nuget package verification",
        Width = 64,
        Height = 64,
        Seed = seed,
        Checkpoint = "bf16",
    };

    private static bool IsPngWithSize(byte[] png, int width, int height)
    {
        if (png.Length < 33 || !png.AsSpan(0, 8).SequenceEqual(PngSignature)) return false;
        if (Encoding.ASCII.GetString(png, 12, 4) != "IHDR") return false;
        return BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(16, 4)) == width
            && BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(20, 4)) == height;
    }

    private sealed class SyncProgress : IProgress<GenerationProgress>
    {
        private int _count;
        public int Count => Volatile.Read(ref _count);
        public void Report(GenerationProgress value) => Interlocked.Increment(ref _count);
    }
}
