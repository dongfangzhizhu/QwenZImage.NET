using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// The browser posts snake_case JSON with base64 PNG data URLs. These tests
/// pin the wire names and verify that RequestMapper decodes them into the same
/// strongly typed options QziClient validates (no silent parameter drops).
/// </summary>
public class WebUiEditMappingTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static string ReferencePng(int width = 2, int height = 2)
        => "data:image/png;base64," + Convert.ToBase64String(
            MakePng(width, height, 2, new byte[width * height * 3]));

    private static string MaskPng(int width = 2, int height = 2)
    {
        byte[] rgba = new byte[width * height * 4];
        // Half the canvas is an opaque white brush stroke, the rest is
        // transparent ("keep"). FromPngMask turns that into 255 / 0.
        for (int i = 0; i < width * height / 2; i++)
        {
            rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = rgba[i * 4 + 3] = 255;
        }
        return "data:image/png;base64," + Convert.ToBase64String(MakePng(width, height, 6, rgba));
    }

    private static GenerateRequest EditRequest(string model = "qwen21-edit",
                                               Action<GenerateRequest>? tweak = null)
    {
        GenerateRequest request = new()
        {
            Model = model,
            Prompt = "replace the sky with a sunset",
            Width = 32,
            Height = 32,
            Seed = 7,
            ReferenceImage = ReferencePng(),
            MaskImage = MaskPng(),
            Strength = 0.6f,
            MaskingStrength = 0.8f,
            Client = "webui"
        };
        tweak?.Invoke(request);
        return request;
    }

    [Fact]
    public void Snake_Case_Edit_Fields_Bind_To_The_Dto()
    {
        string json = """
            {
              "model": "qwen21-edit",
              "prompt": "a sunset",
              "negative_prompt": "blurry",
              "lora_scale": 1.5,
              "reference_image": "data:image/png;base64,AAAA",
              "mask_image": "data:image/png;base64,BBBB",
              "strength": 0.6,
              "masking_strength": 0.8
            }
            """;
        GenerateRequest? dto = JsonSerializer.Deserialize<GenerateRequest>(json, WebUiJson.Options);
        Assert.NotNull(dto);
        Assert.Equal("qwen21-edit", dto.Model);
        Assert.Equal("blurry", dto.NegativePrompt);
        Assert.Equal(1.5f, dto.LoraScale);
        Assert.StartsWith("data:image/png;base64,", dto.ReferenceImage);
        Assert.StartsWith("data:image/png;base64,", dto.MaskImage);
        Assert.Equal(0.6f, dto.Strength);
        Assert.Equal(0.8f, dto.MaskingStrength);
    }

    [Fact]
    public void Plain_Qwen21_Still_Maps_To_Text_To_Image()
    {
        GenerationOptions options = RequestMapper.ToOptions(new GenerateRequest
        {
            Model = "qwen21",
            Prompt = "a fox",
            Width = 64,
            Height = 64
        });
        QwenImageOptions qwen = Assert.IsType<QwenImageOptions>(options);
        Assert.Equal(GenerationModel.Qwen21, qwen.Model);
        Assert.Equal(QwenImageTier.Turbo, qwen.Tier);
    }

    [Fact]
    public void Explicit_Edit_Model_Decodes_Reference_And_Mask()
    {
        Qwen21EditOptions edit = Assert.IsType<Qwen21EditOptions>(
            RequestMapper.ToOptions(EditRequest()));

        Assert.Equal(GenerationModel.Qwen21, edit.Model);
        Assert.Equal(2, edit.Reference.Width);
        Assert.Equal(2, edit.Reference.Height);
        Assert.Equal(3, edit.Reference.Channels);
        Assert.NotNull(edit.Mask);
        Assert.Equal(1, edit.Mask.Channels);
        Assert.Equal(2, edit.Mask.Width);
        Assert.Equal(2, edit.Mask.Height);
        Assert.Equal(255, edit.Mask.Planar.Span[0]);
        Assert.Equal(0, edit.Mask.Planar.Span[3]);
        Assert.Equal(0.6f, edit.EffectiveStrength);
        Assert.Equal(0.8f, edit.EffectiveMaskingStrength);
        Assert.Equal(7ul, edit.Seed);
    }

    [Fact]
    public void A_Reference_Image_On_Plain_Qwen21_Opts_Into_Editing()
    {
        Qwen21EditOptions edit = Assert.IsType<Qwen21EditOptions>(
            RequestMapper.ToOptions(EditRequest("qwen21")));
        Assert.NotNull(edit.Mask);
        Assert.Equal(0.6f, edit.EffectiveStrength);
        Assert.Equal(0.8f, edit.EffectiveMaskingStrength);
    }

    [Fact]
    public void Without_A_Mask_Masking_Strength_Is_Not_Silently_Dropped()
    {
        GenerateRequest request = EditRequest(tweak: r =>
        {
            r.MaskImage = null;
            r.MaskingStrength = null;
            r.Strength = null;
        });
        Qwen21EditOptions edit = Assert.IsType<Qwen21EditOptions>(
            RequestMapper.ToOptions(request));
        Assert.Null(edit.Mask);
        Assert.Equal(0.0f, edit.EffectiveMaskingStrength);
        Assert.Equal(0.75f, edit.EffectiveStrength); // null Strength => default
    }

    [Fact]
    public void Common_Qwen21_Fields_Survive_The_Edit_Mapping()
    {
        GenerateRequest request = EditRequest();
        request.Tier = "base";
        request.Steps = 20;
        request.Guidance = 5.5f;
        request.NegativePrompt = "blurry";
        request.Lora = "dmd2.safetensors";
        request.LoraScale = 1.25f;

        Qwen21EditOptions edit = Assert.IsType<Qwen21EditOptions>(
            RequestMapper.ToOptions(request));
        Assert.Equal(QwenImageTier.Base, edit.Tier);
        Assert.Equal(20, edit.EffectiveSteps);
        Assert.Equal(5.5f, edit.EffectiveGuidance);
        Assert.Equal("blurry", edit.NegativePrompt);
        Assert.Equal("dmd2.safetensors", edit.Lora);
        Assert.Equal(1.25f, edit.EffectiveLoraScale);
    }

    [Fact]
    public void Edit_Parameters_Are_Rejected_Without_A_Reference()
    {
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "qwen21", Prompt = "a fox", Strength = 0.5f }));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "qwen21", Prompt = "a fox", MaskImage = MaskPng() }));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "qwen21-edit", Prompt = "a fox" }));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "qwen21_edit", Prompt = "a fox" }));
    }

    [Fact]
    public void Editing_Is_Rejected_On_ZImage()
    {
        GenerateRequest request = EditRequest("zimage");
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(request));
        Assert.Contains("Qwen-Image 2.1", ex.Message);
    }

    [Fact]
    public void Masking_Strength_Without_A_Mask_Is_Rejected_Clearly()
    {
        GenerateRequest request = EditRequest(tweak: r => r.MaskImage = null);
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(request));
        Assert.Contains("mask_image", ex.Message);
    }

    [Fact]
    public void Mask_Dimension_Mismatch_Is_Rejected_Before_Queueing()
    {
        GenerateRequest request = EditRequest(tweak: r => r.MaskImage = MaskPng(4, 4));
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(request));
        Assert.Contains("Mask dimensions must match", ex.Message);
    }

    [Fact]
    public void Malformed_Image_Payloads_Fail_With_400_Style_Argument_Errors()
    {
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(tweak: r => r.ReferenceImage = "not base64 !!")));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(tweak: r => r.ReferenceImage = Convert.ToBase64String([1, 2, 3, 4]))));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(tweak: r => r.MaskImage = "not base64 !!")));
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
