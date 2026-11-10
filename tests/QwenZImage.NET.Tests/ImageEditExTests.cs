using System.Runtime.InteropServices;
using QwenZImage.NET.Native;
using QwenZImage.NET.Validation;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Third additive ABI v1 increment: WanGP advanced Qwen-Image 2.1 editing
/// (Red Canvas outpainting, LanPaint model modes, RGBA output) against the
/// deterministic reference backend. The reference backend mixes every advanced
/// knob into its deterministic PNG tag and can emit colour type 6 PNGs, so the
/// whole path is observable without a GPU or model weights.
/// </summary>
public class ImageEditExTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static QziRasterImage SolidRgb(int size = 32, byte value = 0x33)
        => QziRasterImage.FromPlanar(
            Enumerable.Repeat(value, size * size * 3).ToArray(), size, size, 3);

    /// <summary>
    /// 64x64 is the FINAL output canvas: a 32 px margin leaves an inner
    /// 32x32 source rectangle, the smallest WanGP allows.
    /// </summary>
    private static Qwen21EditExOptions EditEx(byte value = 0x33)
        => new()
        {
            Prompt = "extend the snowy forest beyond the frame",
            Width = 64,
            Height = 64,
            Seed = 42,
            Reference = SolidRgb(32, value)
        };

    private static QziClient RequireAdvancedClient()
    {
        Assert.True(NativeTestLibrary.IsAvailable,
            "reference native library not found; build with scripts/build-native.ps1 " +
            "and stage it with scripts/stage-native.ps1");
        QziClient client = new(NativeTestLibrary.Options());
        if (client.SupportsAdvancedEditing) return client;
        client.Dispose();
        throw new InvalidOperationException(
            "the staged QwenZImageNative library is stale: it does not advertise " +
            "QZI_FEATURE_QWEN21_EDIT_ADV. Rebuild the reference backend from the " +
            "matching native/include/qzi.h. Generation tests must never silently " +
            "turn into no-ops.");
    }

    // ------------------------------------------------------------- layout

    [Fact]
    public unsafe void Advanced_Request_Layout_Matches_The_C_Struct()
    {
        int editSize = sizeof(QziQwen21EditRequest);
        // The embedded edit request is 8-byte aligned; seven uint32_t fields
        // occupy 28 bytes and tail padding rounds the total up to +32.
        Assert.Equal(editSize + 32, sizeof(QziQwen21EditExRequest));
        Assert.Equal(sizeof(QziQwen21EditExRequest),
                     Marshal.SizeOf<QziQwen21EditExRequest>());

        // Field ORDER must match qzi_qwen21_edit_ex_request in qzi.h exactly.
        Assert.Equal(0, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("Base"));
        Assert.Equal(editSize, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("MarginTop"));
        Assert.Equal(editSize + 4, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("MarginBottom"));
        Assert.Equal(editSize + 8, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("MarginLeft"));
        Assert.Equal(editSize + 12, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("MarginRight"));
        Assert.Equal(editSize + 16, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("ModelMode"));
        Assert.Equal(editSize + 20, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("Rgba"));
        Assert.Equal(editSize + 24, (int)Marshal.OffsetOf<QziQwen21EditExRequest>("Flags"));
    }

    [Fact]
    public unsafe void Advanced_Layout_Matches_The_Native_Export()
    {
        using QziClient client = RequireAdvancedClient();

        IntPtr handle = NativeLibraryLoader.Handle;
        Assert.NotEqual(IntPtr.Zero, handle);
        Assert.True(NativeLibrary.TryGetExport(handle, "qzi_qwen21_edit_ex_layout",
                                               out IntPtr layout));
        nuint nativeSize = ((delegate* unmanaged<nuint>)layout)();
        Assert.Equal((nuint)sizeof(QziQwen21EditExRequest), nativeSize);
    }

    // ---------------------------------------------------------- capability

    [Fact]
    public void Feature_Bits_Are_Additive_0x1_0x2_0x4()
    {
        Assert.Equal(0x1u, NativeMethods.FeatureQwen21Edit);
        Assert.Equal(0x2u, NativeMethods.FeatureZImageControl);
        Assert.Equal(0x4u, NativeMethods.FeatureQwen21EditAdv);
        Assert.Equal(0u, NativeMethods.FeatureQwen21EditAdv &
                         (NativeMethods.FeatureQwen21Edit |
                          NativeMethods.FeatureZImageControl));
    }

    [Fact]
    public void Advanced_Edit_Capability_Is_Detected_On_The_Matching_Library()
    {
        using QziClient client = RequireAdvancedClient();
        Assert.True(client.SupportsImageEditing);
        Assert.True(client.SupportsAdvancedEditing);
        Assert.Equal(client.SupportsAdvancedEditing, AbiGuard.Qwen21EditAdvSupported);
    }

    // ---------------------------------------------------------- validation

    [Fact]
    public void Managed_Validator_Mirrors_The_Native_Advanced_Rules()
    {
        Qwen21EditExOptions baseline = EditEx();
        RequestValidator.Validate(baseline);

        // Margins must be 32-aligned and leave at least a 32x32 source.
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { MarginTop = 16u }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { MarginLeft = 64u, MarginRight = 64u }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { MarginBottom = 64u, MarginTop = 64u }));

        // Unknown WanGP model modes and out-of-range RGBA flags.
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { ModelMode = (Qwen21EditMode)1 }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { ModelMode = (Qwen21EditMode)6 }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { Flags = 1u }));

        // LanPaint requires a mask, excludes outpainting and forces both
        // strengths to 1.0 (an explicit other value is rejected, not dropped).
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { ModelMode = Qwen21EditMode.LanPaint5 }));
        Qwen21EditExOptions lanpaint = baseline with
        {
            ModelMode = Qwen21EditMode.LanPaint5,
            Mask = QziRasterImage.SolidMask(32, 32, 255)
        };
        RequestValidator.Validate(lanpaint);
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            lanpaint with { MarginTop = 32u }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            lanpaint with { Strength = 0.5f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            lanpaint with { MaskingStrength = 0.5f }));
        RequestValidator.Validate(lanpaint with { Strength = 1.0f, MaskingStrength = 1.0f });

        // The generic dispatcher must pick the derived validator.
        GenerationOptions options = baseline with { MarginTop = 16u };
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(options));
    }

    [Fact]
    public void Advanced_Options_Expose_The_WanGP_Semantics()
    {
        Qwen21EditExOptions plain = EditEx();
        Assert.Equal(GenerationModel.Qwen21, plain.Model);
        Assert.False(plain.HasOutpainting);
        Assert.False(plain.IsLanPaint);
        Assert.Equal(0u, plain.EffectiveModelMode);
        Assert.Equal(32u, RequestValidator.OutpaintAlignment);

        Assert.True((plain with { MarginRight = 32u }).HasOutpainting);
        Assert.True((plain with { ModelMode = Qwen21EditMode.LanPaint15 }).IsLanPaint);
        Assert.Equal(5u, (plain with { ModelMode = Qwen21EditMode.LanPaint15 })
                          .EffectiveModelMode);
    }

    // ---------------------------------------------------------- generation

    [Fact]
    public void Advanced_Edit_Generates_A_Png_And_Outpainting_Runs_The_Full_Schedule()
    {
        using QziClient client = RequireAdvancedClient();
        List<GenerationProgress> progress = [];
        var collector = new ProgressCollector(progress);

        // Turbo has 8 steps. A plain edit at strength 0.75 skips 2 steps;
        // Red Canvas margins always start from step 0 (WanGP step_offset 0).
        using GenerationResult plain = client.GenerateQwen21EditEx(EditEx(), collector);
        Assert.Equal(6, progress.Count(p => p.Phase == GenerationPhase.Dit));

        progress.Clear();
        using GenerationResult outpaint = client.GenerateQwen21EditEx(
            EditEx() with { MarginTop = 32u }, collector);
        Assert.Equal(8, progress.Count(p => p.Phase == GenerationPhase.Dit));
        Assert.Contains(progress, p => p.Phase == GenerationPhase.ImageEncode);

        Assert.Equal(PngSignature, outpaint.PngSpan[..8].ToArray());
        Assert.Equal(64, outpaint.Width);
        Assert.Equal(64, outpaint.Height);
        Assert.Equal(42ul, outpaint.Seed);
        Assert.Equal(GenerationModel.Qwen21, outpaint.Model);
        Assert.Equal(2, (int)outpaint.PngSpan[25]); // PNG colour type 2 = RGB
        Assert.NotEqual(plain.PngSpan.ToArray(), outpaint.PngSpan.ToArray());
    }

    [Fact]
    public void Every_Advanced_Knob_Changes_The_Deterministic_Png()
    {
        using QziClient client = RequireAdvancedClient();
        Qwen21EditExOptions baseline = EditEx();

        using GenerationResult a = client.GenerateQwen21EditEx(baseline);
        using GenerationResult b = client.GenerateQwen21EditEx(baseline);
        Assert.Equal(a.PngSpan.ToArray(), b.PngSpan.ToArray());

        using GenerationResult otherReference = client.GenerateQwen21EditEx(EditEx(0xC0));
        Assert.NotEqual(a.PngSpan.ToArray(), otherReference.PngSpan.ToArray());

        using GenerationResult margin = client.GenerateQwen21EditEx(
            baseline with { MarginLeft = 32u });
        Assert.NotEqual(a.PngSpan.ToArray(), margin.PngSpan.ToArray());

        using GenerationResult lanpaint = client.GenerateQwen21EditEx(baseline with
        {
            ModelMode = Qwen21EditMode.LanPaint10,
            Mask = QziRasterImage.SolidMask(32, 32, 255)
        });
        Assert.NotEqual(a.PngSpan.ToArray(), lanpaint.PngSpan.ToArray());

        using GenerationResult rgba = client.GenerateQwen21EditEx(baseline with { Rgba = true });
        Assert.NotEqual(a.PngSpan.ToArray(), rgba.PngSpan.ToArray());
    }

    [Fact]
    public void Rgba_Request_Produces_A_Colour_Type_6_Png()
    {
        using QziClient client = RequireAdvancedClient();
        using GenerationResult rgb = client.GenerateQwen21EditEx(EditEx());
        using GenerationResult rgba = client.GenerateQwen21EditEx(EditEx() with { Rgba = true });

        Assert.Equal(2, (int)rgb.PngSpan[25]);  // truecolor RGB
        Assert.Equal(6, (int)rgba.PngSpan[25]); // truecolor + alpha
        Assert.True(rgba.PngSpan.Length > rgb.PngSpan.Length,
                    "four interleaved channels must be larger than three");
        Assert.Equal(64, rgba.Width);
        Assert.Equal(64, rgba.Height);
    }

    [Fact]
    public async Task Generate_WriteTo_Queue_And_Async_All_Accept_Advanced_Options()
    {
        using QziClient client = RequireAdvancedClient();
        Qwen21EditExOptions options = EditEx() with { MarginTop = 32u, Rgba = true };

        using GenerationResult direct = client.Generate(options);
        using GenerationResult typed = client.GenerateQwen21EditEx(options);
        Assert.Equal(direct.PngSpan.ToArray(), typed.PngSpan.ToArray());

        using GenerationResult asyncResult = await client.GenerateQwen21EditExAsync(options);
        Assert.Equal(direct.PngSpan.ToArray(), asyncResult.PngSpan.ToArray());

        using MemoryStream stream = new();
        client.WriteTo(stream, options);
        Assert.Equal(direct.PngSpan.ToArray(), stream.ToArray());

        var executor = new RecordingExecutor();
        using QziGenerationQueue queue = new(executor);
        QziJob job = queue.Submit(options);
        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Same(options, job.Request);
        Assert.Equal(QziJobState.Done, job.State);
        Assert.Contains(options, executor.Calls);
    }

    [Fact]
    public void Default_Advanced_Request_Differs_From_The_Ordinary_Edit_Tag()
    {
        using QziClient client = RequireAdvancedClient();
        Qwen21EditExOptions advanced = EditEx();
        Qwen21EditOptions ordinary = new()
        {
            Prompt = advanced.Prompt,
            Width = advanced.Width,
            Height = advanced.Height,
            Seed = advanced.Seed,
            Reference = advanced.Reference,
            Mask = advanced.Mask
        };

        using GenerationResult a = client.Generate(advanced);
        using GenerationResult b = client.Generate(ordinary);
        // The stub tags every advanced call with "ADVEDIT1" so both paths are
        // distinguishable in GPU-less CI; the sampling schedule is the same.
        Assert.NotEqual(a.PngSpan.ToArray(), b.PngSpan.ToArray());
    }

    private sealed class ProgressCollector(List<GenerationProgress> items)
        : IProgress<GenerationProgress>
    {
        public void Report(GenerationProgress value) => items.Add(value);
    }
}
