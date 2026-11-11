using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.InteropServices;
using QwenZImage.NET.Native;
using QwenZImage.NET.Validation;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Fourth additive ABI v1 increment: Z-Image Turbo Fun ControlNet Union 2.x
/// mask inpainting, against the deterministic reference backend (small canvas,
/// no GPU). Pure managed tests (feature bits, request layout, options and
/// validation) always run. Generation tests exercise the freshly staged stub
/// when it advertises QZI_FEATURE_ZIMAGE_INPAINT, and -- when a stale v1
/// library is staged -- assert the explicit "unsupported" path instead, so the
/// suite never turns into a silent no-op in either state.
/// </summary>
public class ZImageInpaintTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static QziRasterImage SolidRgb(int size = 32, byte value = 0x33)
        => QziRasterImage.FromPlanar(
            Enumerable.Repeat(value, size * size * 3).ToArray(), size, size, 3);

    private static QziRasterImage SolidMask(int size = 32, byte value = 0xFF)
        => QziRasterImage.SolidMask(size, size, value);

    private static ZImageInpaintOptions Inpaint(int size = 32, byte controlValue = 0x33,
                                                byte maskValue = 0xFF)
        => new()
        {
            Prompt = "repaint the masked region with a red fox",
            Width = size,
            Height = size,
            Seed = 42,
            ControlImage = SolidRgb(size, controlValue),
            Mask = SolidMask(size, maskValue)
        };

    // Publish the ControlNet weights directory so the reference backend's
    // deterministic config fingerprint can observe it (mirrors
    // ZImageControlTests.ControlClientOptions). Inpainting embeds the control
    // request, so the same directory applies.
    private static QziClientOptions InpaintClientOptions()
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

    /// <summary>
    /// On a stale v1 library (which still serves ControlNet) the managed
    /// inpaint API must fail with an actionable Unsupported error, never an
    /// EntryPointNotFoundException from a lazily resolved export.
    /// </summary>
    private static void AssertInpaintUnsupported(QziClient client)
    {
        Assert.True(client.SupportsControlNet);
        Assert.False(client.SupportsInpaint);
        QziException error = Assert.Throws<QziException>(() =>
            client.GenerateZImageInpaint(Inpaint()));
        Assert.Equal(QziStatus.Unsupported, error.Status);
    }

    // ------------------------------------------------------------- layout

    [Fact]
    public unsafe void Inpaint_Request_Layout_Matches_The_C_Struct()
    {
        // qzi_zimage_inpaint_request = control request (72) + mask ptr (8) +
        // mask_width (4) + mask_height (4) + flags (4), padded to 8-byte
        // alignment = 96 bytes. qzi_zimage_inpaint_layout() reports the same.
        Assert.Equal(72, sizeof(QziZImageControlRequest));
        Assert.Equal(96, sizeof(QziZImageInpaintRequest));
        Assert.Equal(sizeof(QziZImageControlRequest) + 24, sizeof(QziZImageInpaintRequest));
        Assert.Equal(96, Marshal.SizeOf<QziZImageInpaintRequest>());

        // Field ORDER must match native/include/qzi.h exactly.
        Assert.Equal(0, (int)Marshal.OffsetOf<QziZImageInpaintRequest>("Base"));
        Assert.Equal(72, (int)Marshal.OffsetOf<QziZImageInpaintRequest>("Mask"));
        Assert.Equal(80, (int)Marshal.OffsetOf<QziZImageInpaintRequest>("MaskWidth"));
        Assert.Equal(84, (int)Marshal.OffsetOf<QziZImageInpaintRequest>("MaskHeight"));
        Assert.Equal(88, (int)Marshal.OffsetOf<QziZImageInpaintRequest>("Flags"));
    }

    // ---------------------------------------------------------- capability

    [Fact]
    public void Feature_Bits_Are_Additive_0x1_0x2_0x4_0x8()
    {
        Assert.Equal(0x1u, NativeMethods.FeatureQwen21Edit);
        Assert.Equal(0x2u, NativeMethods.FeatureZImageControl);
        Assert.Equal(0x4u, NativeMethods.FeatureQwen21EditAdv);
        Assert.Equal(0x8u, NativeMethods.FeatureZImageInpaint);
        // The four increments together are exactly qzi_abi_features() == 0xF.
        Assert.Equal(0xFu, NativeMethods.FeatureQwen21Edit |
                           NativeMethods.FeatureZImageControl |
                           NativeMethods.FeatureQwen21EditAdv |
                           NativeMethods.FeatureZImageInpaint);
        Assert.Equal(0u, NativeMethods.FeatureZImageInpaint &
                         (NativeMethods.FeatureQwen21Edit |
                          NativeMethods.FeatureZImageControl |
                          NativeMethods.FeatureQwen21EditAdv));
    }

    [Fact]
    public unsafe void Native_Features_And_Layout_Report_The_Fourth_Increment()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(InpaintClientOptions());

        IntPtr handle = NativeLibraryLoader.Handle;
        Assert.NotEqual(IntPtr.Zero, handle);
        Assert.True(NativeLibrary.TryGetExport(handle, "qzi_abi_features",
                                               out IntPtr features));
        uint bits = ((delegate* unmanaged<uint>)features)();
        Assert.Equal(0u, bits & ~0xFu); // no stray bits beyond the four increments

        if (!client.SupportsInpaint)
        {
            // Stale v1 library: 0x8 and BOTH inpaint exports are absent.
            Assert.Equal(0u, bits & NativeMethods.FeatureZImageInpaint);
            Assert.False(NativeLibrary.TryGetExport(handle, "qzi_zimage_inpaint_layout", out _));
            Assert.False(NativeLibrary.TryGetExport(handle, "qzi_generate_zimage_inpaint", out _));
            return;
        }

        // Fresh library: all four bits and a matching layout export.
        Assert.Equal(0xFu, bits);
        Assert.True(NativeLibrary.TryGetExport(handle, "qzi_zimage_inpaint_layout",
                                               out IntPtr layout));
        Assert.True(NativeLibrary.TryGetExport(handle, "qzi_generate_zimage_inpaint", out _));
        Assert.Equal((nuint)sizeof(QziZImageInpaintRequest),
                     ((delegate* unmanaged<nuint>)layout)());
        // The bound source-generated stub must agree with the lazy probe.
        Assert.Equal((nuint)sizeof(QziZImageInpaintRequest),
                     NativeMethods.ZImageInpaintLayout());
    }

    [Fact]
    public void Inpaint_Capability_And_Unsupported_Semantics()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(InpaintClientOptions());
        Assert.Equal(client.SupportsInpaint, AbiGuard.ZImageInpaintSupported);

        if (!client.SupportsInpaint)
        {
            AssertInpaintUnsupported(client);
            return;
        }

        // Inpainting embeds the ControlNet request, so its prerequisite holds.
        Assert.True(client.SupportsControlNet);
        Assert.True(client.SupportsInpaint);
    }

    // ------------------------------------------------------------ options

    [Fact]
    public void Inpaint_Options_Derive_From_Control_And_Carry_A_Single_Channel_Mask()
    {
        ZImageInpaintOptions options = Inpaint();
        Assert.IsAssignableFrom<ZImageControlOptions>(options);
        Assert.IsAssignableFrom<ZImageOptions>(options);
        Assert.IsAssignableFrom<GenerationOptions>(options);
        Assert.Equal(GenerationModel.ZImage, options.Model);
        Assert.Equal(1, options.Mask.Channels);
        Assert.Equal(3, options.ControlImage.Channels);

        // ControlNet metadata is inherited and behaves identically.
        Assert.Null(options.EffectiveControlType);
        Assert.Equal("canny", (options with { ControlType = "CANNY" }).EffectiveControlType);

        // A one-channel mask decoded with FromPngBase64(data, mask: true):
        // WHITE (255) = repaint, BLACK (0) = keep. The decoder does not verify
        // the PNG CRC, so a four-zero CRC is enough for this test double.
        string base64 = Convert.ToBase64String(MakeGreyPng(2, 1, [255, 0]));
        QziRasterImage mask = QziRasterImage.FromPngBase64(base64, mask: true);
        Assert.Equal(1, mask.Channels);
        Assert.Equal(2, mask.Width);
        Assert.Equal(1, mask.Height);
        Assert.Equal(new byte[] { 255, 0 }, mask.Planar.ToArray());

        ZImageInpaintOptions withDecodedMask = options with { Mask = mask };
        Assert.Same(mask, withDecodedMask.Mask);
        RequestValidator.Validate(withDecodedMask);
    }

    // ---------------------------------------------------------- validation

    [Fact]
    public void Validator_Rejects_A_Missing_Or_Multichannel_Mask()
    {
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Inpaint() with { Mask = null! }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(Inpaint() with { Mask = SolidRgb() }));

        // The generic dispatcher must pick the derived inpaint validator.
        GenerationOptions options = Inpaint() with { Mask = null! };
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(options));
    }

    [Fact]
    public void Validator_Rejects_Oversized_Masks_And_Allows_Independent_Dimensions()
    {
        Assert.Equal(4096, QziRasterImage.MaxSide);
        Assert.Equal(2048L * 2048L, QziRasterImage.MaxPixels);

        // Oversized masks are rejected at raster construction, before the
        // request can ever be built (ValidateDimensions runs first).
        Assert.Throws<ArgumentException>(() =>
            QziRasterImage.FromPlanar(new byte[3], 4097, 1, 1));
        Assert.Throws<ArgumentException>(() =>
            QziRasterImage.FromPlanar(new byte[3], 2048, 2049, 1));

        // A mask at the per-side limit is accepted by the validator.
        QziRasterImage atLimit = QziRasterImage.FromPlanar(new byte[4096], 4096, 1, 1);
        RequestValidator.Validate(Inpaint() with { Mask = atLimit });

        // The mask is cover-fitted INDEPENDENTLY of the control image, so its
        // dimensions need not match (unlike Qwen21EditOptions, which requires
        // mask dimensions == reference dimensions).
        RequestValidator.Validate(Inpaint() with
            { Mask = QziRasterImage.SolidMask(16, 16, 255) });
    }

    [Fact]
    public void Validator_Rejects_Nonzero_Flags_And_Reuses_Control_Rules()
    {
        ZImageInpaintOptions baseline = Inpaint();
        RequestValidator.Validate(baseline);

        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { Flags = 1u }));
        RequestValidator.Validate(baseline with { Flags = 0u });

        // The embedded ControlNet / Z-Image rules still apply.
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { Width = 100 }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { Prompt = "" }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { Checkpoint = "svdq-fp4-r256" }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { ControlImage = null! }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { ControlType = "hedge" }));
        Assert.Throws<ArgumentException>(() =>
            RequestValidator.Validate(baseline with { ControlScale = 3.0f }));
    }

    [Fact]
    public void Generic_Validator_Dispatches_Inpaint_Before_Control()
    {
        ZImageInpaintOptions bad = Inpaint() with { Mask = null! };

        // Matched as ZImageInpaintOptions -> the mask rule applies.
        GenerationOptions generic = bad;
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(generic));

        // Statically typed as the base control options: the control overload
        // has NO mask rule, so the very same request would pass. This is
        // exactly why the derived case must come first in the switch.
        RequestValidator.Validate((ZImageControlOptions)bad);
    }

    // ------------------------------------------------------- dispatch

    [Fact]
    public void Generate_Dispatches_Inpaint_Options_To_The_Inpaint_Entry()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(InpaintClientOptions());

        // A request whose ONLY violation is the inpaint-specific mask rule.
        // If Generate() matched ZImageControlOptions first, the control path
        // (available on both stale and fresh v1 libraries) would produce a PNG
        // instead of validating the mask. Validation runs before the capability
        // check, so this holds regardless of whether inpaint is supported.
        GenerationOptions bad = Inpaint() with { Mask = null! };
        Assert.Throws<ArgumentException>(() => client.Generate(bad));

        // The ordinary ControlNet dispatch is still reachable.
        ZImageControlOptions control = new()
        {
            Prompt = "a pose-controlled red fox",
            Width = 32,
            Height = 32,
            Seed = 42,
            ControlImage = SolidRgb()
        };
        using GenerationResult controlResult = client.Generate(control);
        Assert.Equal(PngSignature, controlResult.PngSpan[..8].ToArray());
    }

    // -------------------------------------------------------- generation

    [Fact]
    public void Inpaint_Generation_Produces_A_Png_And_The_ImageEncode_Phase()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(InpaintClientOptions());
        if (!client.SupportsInpaint)
        {
            AssertInpaintUnsupported(client);
            return;
        }

        List<GenerationProgress> progress = [];
        var collector = new CollectingProgress<GenerationProgress>(progress);
        using GenerationResult result = client.GenerateZImageInpaint(Inpaint(), collector);

        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.Equal(32, result.Width);
        Assert.Equal(32, result.Height);
        Assert.Equal(42ul, result.Seed);
        Assert.Equal(GenerationModel.ZImage, result.Model);
        // The reference backend tags a SECOND VAE encode ("image encode") for
        // the masked image, plus the usual text / vae phases.
        Assert.Contains(progress, p => p.Phase == GenerationPhase.ImageEncode);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Text);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Vae);
    }

    [Fact]
    public void Inpaint_Output_Depends_On_The_Mask_And_Differs_From_Control()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(InpaintClientOptions());
        if (!client.SupportsInpaint)
        {
            AssertInpaintUnsupported(client);
            return;
        }

        ZImageInpaintOptions baseline = Inpaint();
        using GenerationResult a = client.GenerateZImageInpaint(baseline);
        using GenerationResult b = client.GenerateZImageInpaint(baseline);
        Assert.Equal(a.PngSpan.ToArray(), b.PngSpan.ToArray()); // deterministic

        // Mask CONTENT folds into the deterministic tag ("INPNT21_").
        using GenerationResult otherMask =
            client.GenerateZImageInpaint(Inpaint(maskValue: 0x00));
        Assert.NotEqual(a.PngSpan.ToArray(), otherMask.PngSpan.ToArray());

        // Mask GEOMETRY folds in too, and may differ from the control image.
        using GenerationResult otherSize = client.GenerateZImageInpaint(
            baseline with { Mask = QziRasterImage.SolidMask(16, 16, 255) });
        Assert.NotEqual(a.PngSpan.ToArray(), otherSize.PngSpan.ToArray());

        // An inpaint call must NOT degenerate into a plain ControlNet call.
        ZImageControlOptions control = new()
        {
            Prompt = baseline.Prompt,
            Width = baseline.Width,
            Height = baseline.Height,
            Seed = baseline.Seed,
            ControlImage = baseline.ControlImage
        };
        using GenerationResult controlResult = client.GenerateZImageControl(control);
        Assert.NotEqual(a.PngSpan.ToArray(), controlResult.PngSpan.ToArray());
    }

    [Fact]
    public async Task Generate_WriteTo_And_Async_Map_To_The_Inpaint_Entry()
    {
        if (!NativeTestLibrary.IsAvailable) return;
        using QziClient client = new(InpaintClientOptions());
        if (!client.SupportsInpaint)
        {
            AssertInpaintUnsupported(client);
            return;
        }

        ZImageInpaintOptions options = Inpaint();
        using GenerationResult direct = client.Generate(options); // dispatch switch
        using GenerationResult typed = client.GenerateZImageInpaint(options);
        Assert.Equal(direct.PngSpan.ToArray(), typed.PngSpan.ToArray());

        using MemoryStream stream = new();
        client.WriteTo(stream, options); // zero-copy path
        Assert.Equal(direct.PngSpan.ToArray(), stream.ToArray());

        using GenerationResult asyncResult = await client.GenerateZImageInpaintAsync(options);
        Assert.Equal(direct.PngSpan.ToArray(), asyncResult.PngSpan.ToArray());
    }

    // ------------------------------------------------------------ helpers

    private sealed class CollectingProgress<T>(List<T> items) : IProgress<T>
    {
        public void Report(T value) => items.Add(value);
    }

    /// <summary>Builds a tiny 8-bit greyscale PNG test double (CRC unchecked).</summary>
    private static byte[] MakeGreyPng(int width, int height, byte[] pixels)
    {
        using MemoryStream raw = new();
        for (int y = 0; y < height; y++)
        {
            raw.WriteByte(0); // filter: None
            raw.Write(pixels, y * width, width);
        }
        byte[] rawBytes = raw.ToArray();

        using MemoryStream zlib = new();
        using (ZLibStream compressor = new(zlib, CompressionLevel.Optimal, leaveOpen: true))
            compressor.Write(rawBytes);
        byte[] idat = zlib.ToArray();

        using MemoryStream png = new();
        png.Write(PngSignature);
        byte[] ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(0, 4), width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4, 4), height);
        ihdr[8] = 8; // 8-bit
        ihdr[9] = 0; // greyscale
        WriteChunk(png, "IHDR"u8, ihdr);
        WriteChunk(png, "IDAT"u8, idat);
        WriteChunk(png, "IEND"u8, []);
        return png.ToArray();
    }

    private static void WriteChunk(Stream target, ReadOnlySpan<byte> type,
                                   ReadOnlySpan<byte> data)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(length, data.Length);
        target.Write(length);
        target.Write(type);
        target.Write(data);
        // QziRasterImage.DecodePng skips the CRC bytes without verifying them.
        Span<byte> crc = stackalloc byte[4];
        crc.Clear();
        target.Write(crc);
    }
}
