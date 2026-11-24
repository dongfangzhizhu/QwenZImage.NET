using System.Buffers.Binary;
using System.Security.Cryptography;
using QwenZImage.NET;

// Standalone verification of the PUBLISHED NuGet package. There is no
// ProjectReference anywhere in this project: every API below is resolved from
// the QwenZImage.NET package restored from nuget.org. A CUDA native library and
// real model weights are required; the package's bundled reference backend
// would only emit deterministic noise and is explicitly rejected here.
internal static class Program
{
    private const ulong Seed = 42;
    private const int Size = 256;

    private static readonly byte[] PngSignature =
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    private static readonly string Prompt =
        "A red fox walks through fresh snow at dawn, cinematic light, sharp detail";

    private static int _failures;

    private static async Task<int> Main(string[] args)
    {
        if (args.Contains("--help", StringComparer.OrdinalIgnoreCase))
        {
            Console.WriteLine("""
                Published NuGet package smoke test (real CUDA engine + local models)

                  --native <path>   CUDA QwenZImageNative.dll / .so (required)
                  --config <path>   QwenZImage.config.json with local model packs
                  --out <dir>       output directory for PNGs (default: artifacts/published-smoke)
                  --quick           only the Z-Image check (skip Qwen21 / identity / VRAM)

                Environment: QZI_NATIVE_LIBRARY and QZI_CONFIG are also accepted.
                """);
            return 0;
        }

        string? native = GetArg(args, "native") ?? Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY");
        string? config = GetArg(args, "config");
        string outDir = GetArg(args, "out") ?? Path.Combine("artifacts", "published-smoke");
        bool quick = args.Contains("--quick", StringComparer.OrdinalIgnoreCase);

        if (string.IsNullOrWhiteSpace(native) || !File.Exists(native))
        {
            Console.Error.WriteLine($"CUDA native library not found: '{native ?? "(none)"}'.");
            Console.Error.WriteLine("Build it with scripts/build-native-cuda.bat and pass --native <path>.");
            return 2;
        }

        native = Path.GetFullPath(native);
        Environment.SetEnvironmentVariable("QZI_NATIVE_LIBRARY", native);
        // Never let a missing GPU silently fall back to the noise backend.
        Environment.SetEnvironmentVariable("QZI_REQUIRE_REAL_BACKEND", "1");
        if (config is not null)
            Environment.SetEnvironmentVariable("QZI_CONFIG", Path.GetFullPath(config));

        Directory.CreateDirectory(outDir);

        Console.WriteLine("QwenZImage.NET - published NuGet package verification");
        Console.WriteLine($"package     : {typeof(QziClient).Assembly.GetName().Version} (nuget.org, PackageReference only)");
        Console.WriteLine($"native      : {native}");
        Console.WriteLine($"config      : {config ?? "(library default lookup)"}");
        Console.WriteLine($"output      : {Path.GetFullPath(outDir)}");
        Console.WriteLine();

        try
        {
            QziClientOptions options = QziClientOptions.FromConfig(config);
            IProgress<GenerationProgress> progress = new Progress<GenerationProgress>(p =>
                Console.WriteLine($"    {p.Phase}: {p.Completed}/{p.Total}"));

            using (QziClient client = new(options))
            {
                EngineHealth health = client.GetHealth();
                Check("real CUDA backend loaded (not the reference/noise backend)",
                    !health.IsReferenceBackend, $"IsReferenceBackend={health.IsReferenceBackend}");
                if (health.IsReferenceBackend)
                    throw new InvalidOperationException("QZI_REQUIRE_REAL_BACKEND did not reject the reference backend.");
                Check("native library path is the requested CUDA build",
                    string.Equals(client.NativeLibraryPath, native, StringComparison.OrdinalIgnoreCase),
                    client.NativeLibraryPath ?? "(null)");
                Check("ABI v1 negotiation", client.AbiVersion >= 1, $"ABI v{client.AbiVersion}");

                Console.WriteLine("Z-Image Turbo (bf16, real weights)");
                string zHash = await SaveAndCheckAsync(
                    await client.GenerateZImageAsync(new ZImageOptions
                    {
                        Prompt = Prompt, Width = Size, Height = Size, Seed = Seed,
                        Checkpoint = "bf16"
                    }, progress), "zimage-bf16", outDir, Size, Size);
                Console.WriteLine($"    sha256={zHash}");

                if (quick)
                {
                    Console.WriteLine("--quick: skipping Qwen21 / identity / low-VRAM cases");
                    return Finish();
                }

                Console.WriteLine("Qwen-Image 2.1 Turbo with high memory profile");
                client.SetMemoryProfile(QziMemoryProfile.High);
                QziMemoryPlan highPlan = client.GetMemoryPlan();
                Check("high memory plan resolved", highPlan.Profile == QziMemoryProfile.High,
                    $"{highPlan.Profile}, pool={highPlan.PoolReleaseMib} MiB");
                string highHash = await SaveAndCheckAsync(
                    await client.GenerateQwen21Async(new QwenImageOptions
                    {
                        Prompt = Prompt, Width = Size, Height = Size, Seed = Seed,
                        Tier = QwenImageTier.Turbo
                    }, progress), "qwen21-high", outDir, Size, Size);
                Console.WriteLine($"    sha256={highHash}");

                Console.WriteLine("Qwen-Image 2.1 identity reference (Qwen3-VL visual tower + VAE latent)");
                await SaveAndCheckAsync(
                    await client.GenerateQwen21IdentityRefAsync(new Qwen21IdentityRefOptions
                    {
                        Prompt = "the same person in a bright red jacket, street fashion, front view",
                        Width = Size, Height = Size, Seed = Seed,
                        Tier = QwenImageTier.Turbo,
                        References = [SyntheticPortrait()],
                        IdentityRefWeight = 70,
                        ReferenceEncodeMaxSide = 512
                    }, progress), "qwen21-identity", outDir, Size, Size);

                Console.WriteLine("Qwen-Image 2.1 Turbo with low memory profile (DiT block swapping)");
            }

            // A profile change does not evict an already resident DiT, so use a
            // fresh engine (client) to make the low profile actually load the
            // checkpoint in streaming mode.
            using (QziClient lowClient = new(options))
            {
                lowClient.SetMemoryProfile(QziMemoryProfile.Low);
                QziMemoryPlan lowPlan = lowClient.GetMemoryPlan();
                Check("low memory plan resolved", lowPlan.Profile == QziMemoryProfile.Low,
                    $"{lowPlan.Profile}, encode cap={lowPlan.ReferenceEncodeCap} px");
                string lowHash = await SaveAndCheckAsync(
                    await lowClient.GenerateQwen21Async(new QwenImageOptions
                    {
                        Prompt = Prompt, Width = Size, Height = Size, Seed = Seed,
                        Tier = QwenImageTier.Turbo
                    }, progress), "qwen21-low", outDir, Size, Size);
                Console.WriteLine($"    sha256={lowHash}");

                // The high-profile hash is computed above in the first client.
                // Stored on a local variable outside the scope so the two
                // profiles can be compared byte for byte.
                Check("block swapping is numerically lossless (high == low)",
                    lowHash == _highHash,
                    _highHash is null ? "missing high baseline" : $"high={_highHash[..16]} low={lowHash[..16]}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[FAIL] {ex.GetType().Name}: {ex.Message}");
            if (ex is QziException qzi)
                Console.Error.WriteLine($"       native status={qzi.Status}");
            return 1;
        }

        return Finish();
    }

    // Set in the high-profile scope before the low-profile comparison.
    private static string? _highHash;

    private static async Task<string> SaveAndCheckAsync(GenerationResult result,
        string name, string outDir, int expectedWidth, int expectedHeight)
    {
        try
        {
            string path = Path.Combine(outDir, $"{name}.png");
            await result.SaveAsync(path);
            byte[] png = File.ReadAllBytes(path);
            bool signature = png.AsSpan(0, Math.Min(8, png.Length)).SequenceEqual(PngSignature);
            int width = png.Length >= 24 ? BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(16, 4)) : 0;
            int height = png.Length >= 24 ? BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(20, 4)) : 0;
            string hash = Convert.ToHexString(SHA256.HashData(png)).ToLowerInvariant();
            Check($"{name}: PNG signature / dimensions / non-empty payload",
                signature && width == expectedWidth && height == expectedHeight && png.Length > 4096,
                $"{width}x{height}, {png.Length} bytes, {result.Elapsed.TotalSeconds:F1}s");
            if (name == "qwen21-high") _highHash = hash;
            return hash;
        }
        finally
        {
            result.Dispose();
        }
    }

    // A deterministic synthetic reference avoids shipping a personal photo.
    // 128x128 is enough for the visual tower and keeps the identity case cheap.
    private static QziRasterImage SyntheticPortrait()
    {
        const int side = 128;
        byte[] rgb = new byte[side * side * 3];
        for (int y = 0; y < side; y++)
        {
            for (int x = 0; x < side; x++)
            {
                int i = (y * side + x) * 3;
                bool head = Math.Abs(x - 64) < 26 && y > 24 && y < 88;
                bool torso = Math.Abs(x - 64) < 42 && y >= 88;
                rgb[i] = head ? (byte)210 : torso ? (byte)40 : (byte)230;
                rgb[i + 1] = head ? (byte)165 : torso ? (byte)90 : (byte)235;
                rgb[i + 2] = head ? (byte)135 : torso ? (byte)160 : (byte)240;
            }
        }
        return QziRasterImage.FromInterleavedRgb(rgb, side, side);
    }

    private static string? GetArg(string[] args, string key)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals($"--{key}", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                return args[i + 1];
            if (args[i].StartsWith($"--{key}=", StringComparison.OrdinalIgnoreCase))
                return args[i][(key.Length + 3)..];
        }
        return null;
    }

    private static void Check(string name, bool passed, string? detail = null)
    {
        Console.WriteLine($"  [{(passed ? "PASS" : "FAIL")}] {name}{(detail is null ? "" : $"  ({detail})")}");
        if (!passed) _failures++;
    }

    private static int Finish()
    {
        Console.WriteLine();
        Console.WriteLine(_failures == 0
            ? "RESULT: PASS - published package works with local models."
            : $"RESULT: FAIL - {_failures} check(s) failed.");
        return _failures == 0 ? 0 : 1;
    }
}
