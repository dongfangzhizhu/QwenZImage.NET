using System.Buffers.Binary;
using System.IO.Compression;
using QwenZImage.NET.Validation;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Managed image-editing surface against the deterministic reference backend:
/// capability probing, ABI round trip, WanGP step semantics, raster/PNG
/// decoding and the managed/native validation mirror.
/// </summary>
public class ImageEditTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static QziRasterImage SolidRgb(int size = 32, byte value = 0x33)
        => QziRasterImage.FromPlanar(new byte[size * size * 3].Select(_ => value).ToArray(),
                                     size, size, 3);

    private static Qwen21EditOptions Edit(int size = 32, byte value = 0x33,
                                          QziRasterImage? mask = null)
        => new()
        {
            Prompt = "replace the sky with a sunset",
            Width = size,
            Height = size,
            Seed = 42,
            Reference = SolidRgb(size, value),
            Mask = mask
        };

    [Fact]
    public void Edit_Capability_Is_Detected_On_The_Matching_Library()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Assert.True(client.SupportsImageEditing);
    }

    [Fact]
    public void Edit_Generates_A_Png_And_Reports_The_WanGP_Step_Count()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        List<GenerationProgress> progress = [];
        var collector = new CollectingProgress<GenerationProgress>(progress);

        // Turbo has 8 steps. WanGP: first_step = int(8 * (1 - 0.75)) = 2, so
        // only 6 denoise steps are re-run. A sigma-value rule would report 2.
        using GenerationResult result = client.GenerateQwen21Edit(Edit(), collector);

        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.Equal(32, result.Width);
        Assert.Equal(32, result.Height);
        Assert.Equal(42ul, result.Seed);
        Assert.Equal(GenerationModel.Qwen21, result.Model);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.ImageEncode);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Text);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Vae);
        Assert.Equal(6, progress.Count(p => p.Phase == GenerationPhase.Dit));
    }

    [Fact]
    public void Edit_Is_Deterministic_And_Changes_With_The_Reference_And_Mask()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Qwen21EditOptions baseline = Edit();

        using GenerationResult a = client.GenerateQwen21Edit(baseline);
        using GenerationResult b = client.GenerateQwen21Edit(baseline);
        Assert.Equal(a.PngSpan.ToArray(), b.PngSpan.ToArray());

        using GenerationResult otherReference = client.GenerateQwen21Edit(Edit(value: 0xC0));
        Assert.NotEqual(a.PngSpan.ToArray(), otherReference.PngSpan.ToArray());

        QziRasterImage mask = QziRasterImage.SolidMask(32, 32, 255);
        using GenerationResult masked = client.GenerateQwen21Edit(
            baseline with { Mask = mask, MaskingStrength = 0.6f });
        Assert.NotEqual(a.PngSpan.ToArray(), masked.PngSpan.ToArray());

        using GenerationResult stronger = client.GenerateQwen21Edit(
            baseline with { Strength = 1.0f });
        Assert.NotEqual(a.PngSpan.ToArray(), stronger.PngSpan.ToArray());
    }

    [Fact]
    public async Task Generate_WriteTo_And_Queue_All_Accept_Edit_Options()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Qwen21EditOptions options = Edit();

        using GenerationResult direct = client.Generate(options);
        using GenerationResult typed = client.GenerateQwen21Edit(options);
        Assert.Equal(direct.PngSpan.ToArray(), typed.PngSpan.ToArray());

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
    public async Task Async_Edit_Matches_The_Synchronous_Call()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Qwen21EditOptions options = Edit();
        using GenerationResult sync = client.GenerateQwen21Edit(options);
        using GenerationResult asyncResult = await client.GenerateQwen21EditAsync(options);
        Assert.Equal(sync.PngSpan.ToArray(), asyncResult.PngSpan.ToArray());
    }

    [Fact]
    public void Managed_Validator_Rejects_Inconsistent_Edit_Requests()
    {
        Qwen21EditOptions options = Edit();

        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { Reference = QziRasterImage.SolidMask(32, 32, 255) }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { Mask = QziRasterImage.SolidMask(16, 16, 255) }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { Mask = SolidRgb() }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { Strength = 0.0f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { Strength = 2.0f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { MaskingStrength = 0.5f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { MaskingStrength = float.NaN,
                           Mask = QziRasterImage.SolidMask(32, 32, 255) }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            options with { Width = 100 }));
        RequestValidator.Validate(options with
        {
            Mask = QziRasterImage.SolidMask(32, 32, 255),
            MaskingStrength = 0.5f
        });
    }

    [Fact]
    public void Raster_Planar_Helpers_Keep_Channel_Order_And_Validate_Size()
    {
        byte[] interleaved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        QziRasterImage planar = QziRasterImage.FromInterleavedRgb(interleaved, 2, 2);
        Assert.Equal(3, planar.Channels);
        Assert.Equal(4, planar.PlaneSize);
        Assert.Equal(new byte[] { 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12 },
                     planar.Planar.ToArray());

        QziRasterImage mask = QziRasterImage.SolidMask(3, 2, 255);
        Assert.Equal(1, mask.Channels);
        Assert.Equal(Enumerable.Repeat((byte)255, 6), mask.Planar.ToArray());

        Assert.Throws<ArgumentException>(() =>
            QziRasterImage.FromPlanar(new byte[11], 2, 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QziRasterImage.FromPlanar(new byte[4], 2, 2, 2));
        Assert.Throws<ArgumentException>(() =>
            QziRasterImage.SolidMask(0, 2, 255));
    }

    [Fact]
    public void Png_Decoder_Handles_Rgb_Rgba_Greyscale_And_Base64_Data_Urls()
    {
        byte[] rgba = new byte[2 * 2 * 4];
        // White opaque, white opaque, transparent, black opaque.
        rgba[0] = rgba[1] = rgba[2] = rgba[3] = 255;
        rgba[4] = rgba[5] = rgba[6] = rgba[7] = 255;
        rgba[12] = rgba[13] = rgba[14] = 0; rgba[15] = 255;
        byte[] png = MakePng(2, 2, 6, rgba);

        QziRasterImage reference = QziRasterImage.FromPng(png);
        Assert.Equal(3, reference.Channels);
        // Transparent pixels composite over white, black stays black.
        Assert.Equal(new byte[] { 255, 255, 255, 0,   // R plane
                                  255, 255, 255, 0,   // G plane
                                  255, 255, 255, 0 }, // B plane
                     reference.Planar.ToArray());

        QziRasterImage mask = QziRasterImage.FromPngMask(png);
        Assert.Equal(1, mask.Channels);
        Assert.Equal(255, mask.Planar.Span[0]);
        Assert.Equal(0, mask.Planar.Span[2]); // transparent => keep

        string dataUrl = "data:image/png;base64," + Convert.ToBase64String(png);
        Assert.Equal(reference.Planar.ToArray(),
                     QziRasterImage.FromPngBase64(dataUrl).Planar.ToArray());
        Assert.Equal(mask.Planar.ToArray(),
                     QziRasterImage.FromPngBase64(dataUrl, mask: true).Planar.ToArray());

        byte[] grey = MakePng(2, 1, 0, [10, 200]);
        Assert.Equal(new byte[] { 10, 200, 10, 200, 10, 200 },
                     QziRasterImage.FromPng(grey).Planar.ToArray());
    }

    [Fact]
    public void Png_Decoder_Rejects_Malformed_And_Unsupported_Input()
    {
        Assert.Throws<ArgumentException>(() => QziRasterImage.FromPng([1, 2, 3]));
        Assert.Throws<ArgumentException>(() => QziRasterImage.FromPngBase64("not base64 !!"));
        Assert.Throws<ArgumentException>(() => QziRasterImage.FromPngBase64("data:image/png;broken"));
        // Palette (color type 3) is deliberately unsupported.
        byte[] palette = MakePng(1, 1, 3, [0]);
        Assert.Throws<ArgumentException>(() => QziRasterImage.FromPng(palette));
    }

    [Fact]
    public void Generated_Png_Round_Trips_Back_As_An_Edit_Reference()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        using GenerationResult t2i = client.GenerateQwen21(
            new QwenImageOptions { Prompt = "round trip", Width = 32, Height = 32, Seed = 5 });

        QziRasterImage reference = QziRasterImage.FromPng(t2i.PngSpan);
        Assert.Equal(32, reference.Width);
        Assert.Equal(32, reference.Height);
        Assert.Equal(3, reference.Channels);

        using GenerationResult edited = client.GenerateQwen21Edit(
            Edit() with { Reference = reference, Seed = 5 });
        Assert.Equal(PngSignature, edited.PngSpan[..8].ToArray());
        Assert.NotEqual(t2i.PngSpan.ToArray(), edited.PngSpan.ToArray());
    }

    // --------------------------------------------------------------- helpers

    private sealed class CollectingProgress<T>(List<T> items) : IProgress<T>
    {
        public void Report(T value) => items.Add(value);
    }

    /// <summary>Builds a minimal non-interlaced 8-bit PNG with filter 0 rows.</summary>
    private static byte[] MakePng(int width, int height, int colorType, byte[] pixels)
    {
        int samples = colorType switch { 0 => 1, 2 => 3, 4 => 2, 3 => 1, _ => 4 };
        int stride = width * samples;
        using MemoryStream raw = new();
        for (int y = 0; y < height; y++)
        {
            raw.WriteByte(0);
            raw.Write(pixels, y * stride, stride);
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
        ihdr[8] = 8;
        ihdr[9] = (byte)colorType;
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
        byte[] crcInput = new byte[4 + data.Length];
        type.CopyTo(crcInput);
        data.CopyTo(crcInput.AsSpan(4));
        Span<byte> crc = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crc, Crc32(crcInput));
        target.Write(crc);
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte value in data)
        {
            crc ^= value;
            for (int bit = 0; bit < 8; bit++)
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
        }
        return crc ^ 0xFFFFFFFFu;
    }
}
