using System.Runtime.InteropServices;
using QwenZImage.NET.Native;
using QwenZImage.NET.Validation;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Z-Image Turbo Fun ControlNet Union (additive ABI v1) coverage against the
/// deterministic reference backend. Pure managed tests (layout, validation,
/// configuration, capability bits) always run. Generation tests require the
/// ControlNet-capable reference DLL and FAIL when a stale native library is
/// staged: a silent skip here once masked a whole untested path. A v1 library
/// without the new exports is covered separately by
/// ControlNet_Capability_And_Unsupported_Semantics.
/// </summary>
public class ZImageControlTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static QziRasterImage SolidRgb(int size = 32, byte value = 0x33)
        => QziRasterImage.FromPlanar(
            Enumerable.Repeat(value, size * size * 3).ToArray(), size, size, 3);

    private static ZImageControlOptions Control(int size = 32, byte value = 0x33)
        => new()
        {
            Prompt = "a pose-controlled red fox",
            Width = size,
            Height = size,
            Seed = 42,
            ControlImage = SolidRgb(size, value)
        };

    // Publish the ControlNet weights directory so the reference backend's
    // deterministic config fingerprint can observe it.
    private static QziClientOptions ControlClientOptions()
    {
        QziClientOptions baseline = NativeTestLibrary.Options();
        return new QziClientOptions
        {
            NativeLibraryPath = baseline.NativeLibraryPath,
            Qwen21Directory = baseline.Qwen21Directory,
            ZImageDirectory = baseline.ZImageDirectory,
            CacheDirectory = baseline.CacheDirectory,
            ZImageControlDirectory = "."
        };
    }

    private static QziClient RequireControlClient()
    {
        Assert.True(NativeTestLibrary.IsAvailable,
            "reference native library not found; build with scripts/build-native.ps1 " +
            "and stage it with scripts/stage-native.ps1");
        QziClient client = new(ControlClientOptions());
        if (client.SupportsControlNet) return client;
        client.Dispose();
        throw new InvalidOperationException(
            "artifacts/runtimes/win-x64/native/QwenZImageNative.dll is stale: it does not " +
            "advertise QZI_FEATURE_ZIMAGE_CONTROL. Rebuild the reference backend with " +
            "scripts/build-native.ps1 and restage it with scripts/stage-native.ps1. " +
            "Generation tests must never silently turn into no-ops.");
    }

    // ------------------------------------------------------------- layout

    [Fact]
    public unsafe void Control_Request_Layout_Matches_The_C_Struct()
    {
        // qzi_zimage_request: 2 ptr (16) + 2 int (8) + ulong (8) + uint (4),
        // padded to the 8-byte alignment = 40 bytes.
        Assert.Equal(40, sizeof(QziZImageRequest));
        // qzi_zimage_control_request = base (40) + control_rgb (8) +
        // width (4) + height (4) + control_type (8) + scale (4) + flags (4)
        // = 72 bytes, no trailing padding (already 8-aligned).
        Assert.Equal(72, sizeof(QziZImageControlRequest));
        Assert.Equal(sizeof(QziZImageRequest) + 32, sizeof(QziZImageControlRequest));
        Assert.Equal(72, Marshal.SizeOf<QziZImageControlRequest>());

        // Field ORDER must match native/include/qzi.h exactly.
        Assert.Equal(0, (int)Marshal.OffsetOf<QziZImageControlRequest>("Base"));
        Assert.Equal(40, (int)Marshal.OffsetOf<QziZImageControlRequest>("ControlRgb"));
        Assert.Equal(48, (int)Marshal.OffsetOf<QziZImageControlRequest>("ControlWidth"));
        Assert.Equal(52, (int)Marshal.OffsetOf<QziZImageControlRequest>("ControlHeight"));
        Assert.Equal(56, (int)Marshal.OffsetOf<QziZImageControlRequest>("ControlType"));
        Assert.Equal(64, (int)Marshal.OffsetOf<QziZImageControlRequest>("ControlScale"));
        Assert.Equal(68, (int)Marshal.OffsetOf<QziZImageControlRequest>("Flags"));
    }

    [Fact]
    public unsafe void Control_Layout_Matches_The_Native_Export()
    {
        using QziClient client = RequireControlClient();

        IntPtr handle = NativeLibraryLoader.Handle;
        Assert.NotEqual(IntPtr.Zero, handle);
        Assert.True(NativeLibrary.TryGetExport(handle, "qzi_zimage_control_layout",
                                               out IntPtr layout));
        nuint nativeSize = ((delegate* unmanaged<nuint>)layout)();
        Assert.Equal((nuint)sizeof(QziZImageControlRequest), nativeSize);
    }

    // ---------------------------------------------------------- validation

    [Fact]
    public void Validator_Requires_A_Valid_Control_Image()
    {
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Control() with { ControlImage = null! }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Control() with
                { ControlImage = QziRasterImage.SolidMask(32, 32, 255) }));

        // A within-limits control image is accepted; the native limits are
        // enforced by QziRasterImage at construction time.
        Assert.Equal(4096, QziRasterImage.MaxSide);
        Assert.Equal(2048L * 2048L, QziRasterImage.MaxPixels);
        QziRasterImage atSideLimit = QziRasterImage.FromPlanar(
            new byte[4096 * 3], 4096, 1, 3);
        RequestValidator.Validate(Control() with { ControlImage = atSideLimit });

        // Over the per-side limit / total-pixel limit is rejected at raster
        // construction (ValidateDimensions runs before the buffer-size check,
        // so a tiny stand-in buffer is enough to reach the dimension guard).
        Assert.Throws<ArgumentException>(() =>
            QziRasterImage.FromPlanar(new byte[3], 4097, 1, 3));
        Assert.Throws<ArgumentException>(() =>
            QziRasterImage.FromPlanar(new byte[3], 2048, 2049, 3));
    }

    [Fact]
    public void Control_Request_Reuses_ZImage_Base_Validation()
    {
        // Canvas must be a multiple of 16, prompt required, checkpoint rules apply.
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Control() with { Width = 100 }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Control() with { Prompt = "" }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Control() with { Checkpoint = "svdq-fp4-r256" }));
        RequestValidator.Validate(Control() with { Checkpoint = "nunchaku-r256" });
    }

    [Fact]
    public void Control_Type_Is_Normalized_And_Validated()
    {
        (string? Raw, string? Normalized)[] valid =
        [
            (null, null), ("", null), ("   ", null),
            ("auto", "auto"), ("PoSe", "pose"), ("CANNY", "canny"),
            ("DePtH", "depth"), ("ScRiBbLe", "scribble"), ("RAW", "raw"),
            (" Pose ", "pose")
        ];
        foreach ((string? raw, string? normalized) in valid)
        {
            ZImageControlOptions options = Control() with { ControlType = raw };
            Assert.Equal(normalized, options.EffectiveControlType);
            RequestValidator.Validate(options);
        }

        foreach (string bad in new[] { "hedge", "pose2", "normal", "lineart", "seg" })
            Assert.Throws<ArgumentException>(() =>
                RequestValidator.Validate(Control() with { ControlType = bad }));
    }

    [Fact]
    public void Control_Scale_Range_Is_Enforced()
    {
        // 0 is the "engine default 0.65" sentinel; the rest must be finite
        // and lie in 0.05..2.0.
        foreach (float good in new[] { 0.0f, 0.05f, 0.65f, 1.0f, 2.0f })
            RequestValidator.Validate(Control() with { ControlScale = good });

        foreach (float bad in new[]
                 { 0.049f, 2.01f, -0.5f, -1.0f, float.NaN,
                   float.PositiveInfinity, float.NegativeInfinity })
            Assert.Throws<ArgumentException>(() =>
                RequestValidator.Validate(Control() with { ControlScale = bad }));
    }

    [Fact]
    public void Validator_Rejects_Nonzero_Flags()
    {
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Control() with { Flags = 1u }));
        RequestValidator.Validate(Control() with { Flags = 0u });
    }

    // --------------------------------------------------------- capability

    [Fact]
    public void ControlNet_Feature_Bit_Is_Additive_0x2()
    {
        Assert.Equal(0x1u, NativeMethods.FeatureQwen21Edit);
        Assert.Equal(0x2u, NativeMethods.FeatureZImageControl);
        Assert.Equal(0u, NativeMethods.FeatureQwen21Edit & NativeMethods.FeatureZImageControl);
    }

    [Fact]
    public void ControlNet_Capability_And_Unsupported_Semantics()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(ControlClientOptions());

        Assert.Equal(client.SupportsControlNet, AbiGuard.ZImageControlSupported);

        if (client.SupportsControlNet)
            return; // rebuilt stub: the supported path is covered below

        // A v1 library without the control exports must fail with an explicit,
        // actionable "unsupported" error, never an EntryPointNotFoundException.
        QziException error = Assert.Throws<QziException>(() =>
            client.GenerateZImageControl(Control()));
        Assert.Equal(QziStatus.Unsupported, error.Status);
    }

    // ------------------------------------------------------ configuration

    [Fact]
    public void Config_Key_And_Environment_Override_For_Control_Dir()
    {
        string path = Path.Combine(Path.GetTempPath(),
            $"qzi-control-{Guid.NewGuid():N}.json");
        string name = QziConfig.ZImageControlDirectoryVariable;
        string? ambient = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, null);
            File.WriteAllText(path, """
                { "zimage_control_dir": "/from-file/control",
                  "windows": { "zimage_control_dir": "/from-windows/control" },
                  "linux":   { "zimage_control_dir": "/from-linux/control" },
                  "osx":     { "zimage_control_dir": "/from-osx/control" } }
                """);

            QziConfig config = QziConfig.Load(path);
            Assert.NotNull(config.ZImageControlDirectory);
            // A platform block overrides the top-level key.
            Assert.NotEqual("/from-file/control", config.ZImageControlDirectory);
            Assert.Equal(config.ZImageControlDirectory,
                         config.ToClientOptions().ZImageControlDirectory);

            QziConfig overridden = config.With("zimage_control_dir", "/overridden/control");
            Assert.Equal("/overridden/control", overridden.ZImageControlDirectory);

            // The QZI_* environment override has the highest precedence.
            Environment.SetEnvironmentVariable(name, "/from-env/control");
            Assert.Equal("/from-env/control",
                         QziClientOptions.FromConfig(path).ZImageControlDirectory);

            // Unknown keys must still be rejected (typo guard preserved).
            Environment.SetEnvironmentVariable(name, null);
            File.WriteAllText(path, """{ "zimage_control_dirr": "/typo" }""");
            Assert.Throws<InvalidDataException>(() => QziConfig.Load(path));
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, ambient);
            File.Delete(path);
        }
    }

    [Fact]
    public void NativeConfigScope_Publishes_And_Restores_Control_Dir()
    {
        string name = QziConfig.ZImageControlDirectoryVariable;
        string? original = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, "/ambient/control");
            var scoped = new QziClientOptions { ZImageControlDirectory = "/scoped/control" };
            using (new NativeConfigScope(scoped))
                Assert.Equal("/scoped/control", Environment.GetEnvironmentVariable(name));
            Assert.Equal("/ambient/control", Environment.GetEnvironmentVariable(name));

            // An unset option must leave the ambient value exactly as it was.
            using (new NativeConfigScope(new QziClientOptions()))
                Assert.Equal("/ambient/control", Environment.GetEnvironmentVariable(name));
            Assert.Equal("/ambient/control", Environment.GetEnvironmentVariable(name));

            // The variable is part of the published/restored set.
            Assert.Contains(name, NativeConfigScope.VariableNames);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, original);
        }
    }

    // -------------------------------------------------------- generation

    [Fact]
    public void Control_Generation_Produces_A_Png_And_The_ImageEncode_Phase()
    {
        using QziClient client = RequireControlClient();

        List<GenerationProgress> progress = [];
        var collector = new CollectingProgress<GenerationProgress>(progress);
        using GenerationResult result = client.GenerateZImageControl(Control(), collector);

        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.Equal(32, result.Width);
        Assert.Equal(32, result.Height);
        Assert.Equal(42ul, result.Seed);
        Assert.Equal(GenerationModel.ZImage, result.Model);
        // VAE-encoding the control image is a distinct ControlNet phase.
        Assert.Contains(progress, p => p.Phase == GenerationPhase.ImageEncode);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Text);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Vae);
    }

    [Fact]
    public void Control_Output_Depends_On_Control_Image_Scale_And_Type()
    {
        using QziClient client = RequireControlClient();

        ZImageControlOptions baseline = Control();
        using GenerationResult a = client.GenerateZImageControl(baseline);
        using GenerationResult b = client.GenerateZImageControl(baseline);
        Assert.Equal(a.PngSpan.ToArray(), b.PngSpan.ToArray()); // deterministic

        // The reference backend must fold the control image CONTENT into its
        // deterministic tag so a GPU-free test can observe parameter pass-through.
        using GenerationResult otherImage = client.GenerateZImageControl(Control(value: 0xC0));
        Assert.NotEqual(a.PngSpan.ToArray(), otherImage.PngSpan.ToArray());

        // ... as well as control_scale and control_type.
        using GenerationResult otherScale = client.GenerateZImageControl(
            baseline with { ControlScale = 1.25f });
        Assert.NotEqual(a.PngSpan.ToArray(), otherScale.PngSpan.ToArray());

        using GenerationResult otherType = client.GenerateZImageControl(
            baseline with { ControlType = "canny" });
        Assert.NotEqual(a.PngSpan.ToArray(), otherType.PngSpan.ToArray());

        // Case-insensitive normalisation reaches the engine: "PoSe" == "pose".
        using GenerationResult poseA = client.GenerateZImageControl(
            baseline with { ControlType = "pose" });
        using GenerationResult poseB = client.GenerateZImageControl(
            baseline with { ControlType = "PoSe" });
        Assert.Equal(poseA.PngSpan.ToArray(), poseB.PngSpan.ToArray());
    }

    [Fact]
    public async Task Control_Async_WriteTo_And_Dispatch_Map_To_The_Control_Entry()
    {
        using QziClient client = RequireControlClient();

        ZImageControlOptions options = Control();
        using GenerationResult direct = client.Generate(options); // dispatch switch
        using GenerationResult typed = client.GenerateZImageControl(options);
        Assert.Equal(direct.PngSpan.ToArray(), typed.PngSpan.ToArray());

        using MemoryStream stream = new();
        client.WriteTo(stream, options); // zero-copy path
        Assert.Equal(direct.PngSpan.ToArray(), stream.ToArray());

        using GenerationResult asyncResult = await client.GenerateZImageControlAsync(options);
        Assert.Equal(direct.PngSpan.ToArray(), asyncResult.PngSpan.ToArray());

        // A control request must not take the plain Z-Image entry point.
        using GenerationResult plain = client.GenerateZImage(new ZImageOptions
        {
            Prompt = options.Prompt, Width = options.Width,
            Height = options.Height, Seed = options.Seed
        });
        Assert.NotEqual(direct.PngSpan.ToArray(), plain.PngSpan.ToArray());
    }

    // ------------------------------------------------------------ helpers

    private sealed class CollectingProgress<T>(List<T> items) : IProgress<T>
    {
        public void Report(T value) => items.Add(value);
    }
}
