using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// The fourth additive ABI v1 increment: Z-Image Union 2.x inpaint. A control
/// request that also carries a single-channel brush mask (white = repaint,
/// black = keep) is upgraded to <see cref="ZImageInpaintOptions"/>. These tests
/// pin the model aliases, the auto-upgrade, the required rasters, the health
/// flag, the job badges and the capability gate. No GPU or native library is
/// needed: everything goes through RequestMapper / pure helpers.
/// </summary>
public class ZImageInpaintMappingTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static string ControlPng(int width = 2, int height = 2)
        => "data:image/png;base64," + Convert.ToBase64String(
            MakePng(width, height, 2, new byte[width * height * 3]));

    private static string MaskPng(int width = 2, int height = 2)
    {
        // RGBA: opaque white pixels are the brush stroke (repaint = 255),
        // transparent pixels are "keep" (0). FromPngBase64(mask: true) folds
        // this into a single-channel raster, exactly like qwen21-edit.
        byte[] rgba = new byte[width * height * 4];
        for (int i = 0; i < width * height / 2; i++)
            rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = rgba[i * 4 + 3] = 255;
        return "data:image/png;base64," + Convert.ToBase64String(
            MakePng(width, height, 6, rgba));
    }

    private static GenerateRequest InpaintRequest(string model = "zimage-inpaint",
                                                   Action<GenerateRequest>? tweak = null)
    {
        GenerateRequest request = new()
        {
            Model = model,
            Prompt = "a neon city street at night",
            Width = 64,
            Height = 64,
            Seed = 7,
            ControlImage = ControlPng(),
            MaskImage = MaskPng(),
            ControlType = "pose",
            ControlScale = 0.65f,
            Client = "webui"
        };
        tweak?.Invoke(request);
        return request;
    }

    [Fact]
    public void Inpaint_Aliases_And_A_Control_Image_With_Mask_Map_To_ZImageInpaintOptions()
    {
        foreach (string alias in new[] { "zimage-inpaint", "zimage_inpaint", "z-image-inpaint" })
        {
            ZImageInpaintOptions options = Assert.IsType<ZImageInpaintOptions>(
                RequestMapper.ToOptions(InpaintRequest(alias)));

            Assert.Equal(GenerationModel.ZImage, options.Model);
            Assert.Equal("a neon city street at night", options.Prompt);
            Assert.Equal(64, options.Width);
            Assert.Equal(64, options.Height);
            Assert.Equal(7ul, options.Seed);
            Assert.Equal("bf16", options.Checkpoint);
            Assert.Equal(2, options.ControlImage.Width);
            Assert.Equal(2, options.ControlImage.Height);
            Assert.Equal(3, options.ControlImage.Channels);
            Assert.NotNull(options.Mask);
            Assert.Equal(1, options.Mask.Channels);
            Assert.Equal(2, options.Mask.Width);
            Assert.Equal(2, options.Mask.Height);
            Assert.Equal("pose", options.ControlType);
            Assert.Equal(0.65f, options.ControlScale);
            Assert.Equal(0u, options.Flags);
        }
    }

    [Fact]
    public void Plain_ZImage_With_Control_Image_And_Mask_Upgrades_To_Inpaint()
    {
        // A script that already posts "zimage" (or "zimage-control") opts into
        // inpaint simply by adding mask_image alongside control_image.
        Assert.IsType<ZImageInpaintOptions>(
            RequestMapper.ToOptions(InpaintRequest("zimage")));
        Assert.IsType<ZImageInpaintOptions>(
            RequestMapper.ToOptions(InpaintRequest("zimage-control")));
    }

    [Fact]
    public void Mask_Image_Is_Required_For_Inpaint()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            InpaintRequest(tweak: r => r.MaskImage = null)));
        Assert.Contains("mask_image", ex.Message);

        // A blank / whitespace mask is equally missing.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            InpaintRequest(tweak: r => r.MaskImage = "   ")));
    }

    [Fact]
    public void Control_Image_Is_Required_For_Inpaint()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            InpaintRequest(tweak: r => r.ControlImage = null)));
        Assert.Contains("control_image", ex.Message);
    }

    [Fact]
    public void Inpaint_Is_Mutually_Exclusive_With_Image_Editing()
    {
        GenerateRequest both = InpaintRequest();
        both.ReferenceImage = ControlPng(); // also opts into editing
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(both));
        Assert.Contains("mutually exclusive", ex.Message);
    }

    [Fact]
    public void Mask_On_A_Control_Request_Is_Not_An_Edit_Parameter()
    {
        // Regression for HasEditParameters: mask_image used to be rejected as
        // a qwen21-edit field ("require reference_image") on control paths.
        // With a control image present it must instead upgrade to inpaint.
        ZImageInpaintOptions options = Assert.IsType<ZImageInpaintOptions>(
            RequestMapper.ToOptions(InpaintRequest("zimage-control")));
        Assert.NotNull(options.Mask);
        Assert.Equal("pose", options.ControlType);
    }

    [Fact]
    public void Health_Flags_Serialize_Supports_Inpaint()
    {
        Assert.False(new WebUiEngineHealth().SupportsInpaint);

        string json = JsonSerializer.Serialize(
            new WebUiEngineHealth { SupportsInpaint = true }, WebUiJson.Options);
        using (JsonDocument doc = JsonDocument.Parse(json))
            Assert.True(doc.RootElement.GetProperty("supports_inpaint").GetBoolean());

        // /api/health builds an anonymous object with the literal flag name.
        string endpointJson = JsonSerializer.Serialize(
            new { supports_inpaint = true }, WebUiJson.Options);
        using (JsonDocument doc = JsonDocument.Parse(endpointJson))
            Assert.True(doc.RootElement.GetProperty("supports_inpaint").GetBoolean());
    }

    [Fact]
    public void Job_Dto_Inpaint_State_And_Badges()
    {
        ZImageInpaintOptions inpaint = Assert.IsType<ZImageInpaintOptions>(
            RequestMapper.ToOptions(InpaintRequest()));
        ZImageControlOptions control = Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(new GenerateRequest
            {
                Model = "zimage-control", Prompt = "a fox", Width = 64, Height = 64,
                ControlImage = ControlPng()
            }));

        // The derived record is matched before its ControlNet base in the pure
        // helpers; inpaint still carries a control image, so Control stays true.
        Assert.Equal((true, true), JobDto.InpaintState(inpaint));
        Assert.Equal((true, true), JobDto.ControlState(inpaint));
        Assert.Equal((false, false), JobDto.InpaintState(control));

        JobDto dto = new() { Inpaint = true, HasMask = true };
        using JsonDocument doc = JsonDocument.Parse(
            JsonSerializer.Serialize(dto, WebUiJson.Options));
        Assert.True(doc.RootElement.GetProperty("inpaint").GetBoolean());
        Assert.True(doc.RootElement.GetProperty("has_mask").GetBoolean());
    }

    [Fact]
    public void Capability_Gate_Returns_Inpaint_Unavailable_When_Unsupported()
    {
        ZImageInpaintOptions inpaint = Assert.IsType<ZImageInpaintOptions>(
            RequestMapper.ToOptions(InpaintRequest()));

        string? error = WebUiCapabilities.ErrorFor(inpaint,
            supportsImageEditing: true, supportsControlNet: true,
            supportsAdvancedEditing: true, supportsInpaint: false);
        Assert.Equal(WebUiCapabilities.InpaintUnavailable, error);
        Assert.Contains("inpaint", error);

        Assert.Null(WebUiCapabilities.ErrorFor(inpaint, true, true, true, true));

        // The four-argument overload predates the increment: inpaint support
        // follows ControlNet there, so a ControlNet-less library is rejected.
        Assert.Equal(WebUiCapabilities.InpaintUnavailable,
            WebUiCapabilities.ErrorFor(inpaint, true, false, true));
        Assert.Null(WebUiCapabilities.ErrorFor(inpaint, true, true, true));
    }

    [Fact]
    public void Plain_Control_Without_A_Mask_Stays_On_The_Old_Path()
    {
        ZImageControlOptions control = Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(new GenerateRequest
            {
                Model = "zimage-control", Prompt = "a fox", Width = 64, Height = 64,
                ControlImage = ControlPng(), ControlType = "pose", ControlScale = 0.65f
            }));
        Assert.False(control is ZImageInpaintOptions);
        Assert.Equal("pose", control.ControlType);
        Assert.Equal(0.65f, control.ControlScale);
    }

    // --------------------------------------------------------------- helpers

    /// <summary>Builds a minimal non-interlaced 8-bit PNG with filter 0 rows.</summary>
    private static byte[] MakePng(int width, int height, int colorType, byte[] pixels)
    {
        int samples = colorType switch { 0 => 1, 2 => 3, 4 => 2, _ => 4 };
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

        using MemoryStream png = new();
        png.Write(PngSignature);
        byte[] ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(0, 4), width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4, 4), height);
        ihdr[8] = 8;
        ihdr[9] = (byte)colorType;
        WriteChunk(png, "IHDR"u8, ihdr);
        WriteChunk(png, "IDAT"u8, zlib.ToArray());
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
