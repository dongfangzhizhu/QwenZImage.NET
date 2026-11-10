using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// The Z-Image Turbo Fun ControlNet Union mode posts snake_case JSON with a
/// base64 PNG control image. These tests pin the wire names, the mapping into
/// <see cref="ZImageControlOptions"/>, the capability gate and the job status
/// flags. No GPU, native library or real ControlNet weights are needed.
/// </summary>
public class WebUiControlMappingTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static string ControlPng(int width = 2, int height = 2)
        => "data:image/png;base64," + Convert.ToBase64String(
            MakePng(width, height, 2, new byte[width * height * 3]));

    private static GenerateRequest ControlRequest(string model = "zimage-control",
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
            ControlType = "pose",
            ControlScale = 0.65f,
            Client = "webui"
        };
        tweak?.Invoke(request);
        return request;
    }

    [Fact]
    public void Snake_Case_Control_Fields_Bind_To_The_Dto()
    {
        string json = """
            {
              "model": "zimage-control",
              "prompt": "a city",
              "control_image": "data:image/png;base64,AAAA",
              "control_type": "CaNnY",
              "control_scale": 0.8,
              "checkpoint": "nunchaku"
            }
            """;
        GenerateRequest? dto = JsonSerializer.Deserialize<GenerateRequest>(json, WebUiJson.Options);
        Assert.NotNull(dto);
        Assert.Equal("zimage-control", dto.Model);
        Assert.StartsWith("data:image/png;base64,", dto.ControlImage);
        Assert.Equal("CaNnY", dto.ControlType);
        Assert.Equal(0.8f, dto.ControlScale);
        Assert.Equal("nunchaku", dto.Checkpoint);
    }

    [Fact]
    public void Control_Model_Maps_To_ZImageControlOptions()
    {
        ZImageControlOptions options = Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest()));

        Assert.Equal(GenerationModel.ZImage, options.Model);
        Assert.Equal("a neon city street at night", options.Prompt);
        Assert.Equal(64, options.Width);
        Assert.Equal(64, options.Height);
        Assert.Equal(7ul, options.Seed);
        Assert.Equal("bf16", options.Checkpoint);
        Assert.Equal(2, options.ControlImage.Width);
        Assert.Equal(2, options.ControlImage.Height);
        Assert.Equal(3, options.ControlImage.Channels);
        Assert.Equal("pose", options.ControlType);
        Assert.Equal("pose", options.EffectiveControlType);
        Assert.Equal(0.65f, options.ControlScale);
        Assert.Equal(0u, options.Flags);
    }

    [Fact]
    public void Control_Aliases_And_A_Control_Image_On_Plain_ZImage_Opt_In()
    {
        Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest("zimage_control")));
        Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest("z-image-control")));
        // Scripts that already post "zimage" can opt in by adding one field.
        Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest("zimage")));
    }

    [Fact]
    public void Plain_ZImage_Still_Maps_To_Text_To_Image()
    {
        // A plain zimage request must NOT become a ControlNet request.
        ZImageOptions zimage = Assert.IsType<ZImageOptions>(RequestMapper.ToOptions(
            new GenerateRequest { Model = "zimage", Prompt = "a fox", Width = 64, Height = 64 }));
        Assert.Equal(GenerationModel.ZImage, zimage.Model);
        Assert.Equal("bf16", zimage.Checkpoint);
    }

    [Fact]
    public void Control_Image_Is_Required()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "zimage-control", Prompt = "a fox" }));
        Assert.Contains("control_image", ex.Message);

        // control_type / control_scale alone (no image) are not silently dropped.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "zimage", Prompt = "a fox", ControlType = "canny" }));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "zimage", Prompt = "a fox", ControlScale = 0.5f }));
    }

    [Fact]
    public void Control_Type_Is_Normalised_And_Whitelisted()
    {
        (string Input, string Expected)[] cases =
        [
            ("auto", "auto"), ("PoSe", "pose"), ("CANNY", "canny"),
            (" Depth ", "depth"), ("scribble", "scribble"), ("RAW", "raw")
        ];
        foreach ((string input, string expected) in cases)
        {
            ZImageControlOptions options = Assert.IsType<ZImageControlOptions>(
                RequestMapper.ToOptions(ControlRequest(tweak: r => r.ControlType = input)));
            Assert.Equal(expected, options.ControlType);
            Assert.Equal(expected, options.EffectiveControlType);
        }

        ZImageControlOptions none = Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest(tweak: r => r.ControlType = "  ")));
        Assert.Null(none.ControlType);
        Assert.Null(none.EffectiveControlType);

        ArgumentException ex = Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            ControlRequest(tweak: r => r.ControlType = "hedge")));
        Assert.Contains("control_type", ex.Message);
    }

    [Fact]
    public void Control_Scale_Boundaries()
    {
        // 0 (and null) are the engine-default sentinel 0.65; the rest must be
        // finite and lie in 0.05..2.0.
        foreach (float? scale in new float?[] { 0f, 0.05f, 0.65f, 2.0f, null })
        {
            ZImageControlOptions options = Assert.IsType<ZImageControlOptions>(
                RequestMapper.ToOptions(ControlRequest(tweak: r => r.ControlScale = scale)));
            Assert.Equal(scale ?? 0f, options.ControlScale);
        }

        foreach (float scale in new[] { -0.1f, 0.03f, 2.1f, float.NaN })
        {
            Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
                ControlRequest(tweak: r => r.ControlScale = scale)));
        }
    }

    [Fact]
    public void Control_Is_Rejected_On_Qwen21_And_Is_Mutually_Exclusive_With_Editing()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            ControlRequest("qwen21")));
        Assert.Contains("Z-Image", ex.Message);

        GenerateRequest both = ControlRequest();
        both.ReferenceImage = ControlPng(); // a reference would also opt into editing
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(both));
    }

    [Fact]
    public void Health_Flags_Serialize_As_Snake_Case()
    {
        // /api/health exposes supports_controlnet using this same WebUiJson
        // snake_case contract; pin the field name and its default here.
        Assert.False(new WebUiEngineHealth().SupportsControlNet);
        string json = JsonSerializer.Serialize(
            new WebUiEngineHealth { SupportsControlNet = true }, WebUiJson.Options);
        using JsonDocument doc = JsonDocument.Parse(json);
        Assert.True(doc.RootElement.GetProperty("supports_controlnet").GetBoolean());

        // /api/health builds an anonymous object with the literal flag name and
        // the very same WebUiJson options; pin that exact wire name too.
        string endpointJson = JsonSerializer.Serialize(
            new { supports_controlnet = true }, WebUiJson.Options);
        using JsonDocument endpointDoc = JsonDocument.Parse(endpointJson);
        Assert.True(endpointDoc.RootElement.GetProperty("supports_controlnet").GetBoolean());
    }

    [Fact]
    public void Capability_Gate_Returns_A_Clear_Error_When_Unsupported()
    {
        ZImageControlOptions control = Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest()));
        GenerationOptions edit = RequestMapper.ToOptions(new GenerateRequest
        {
            Model = "qwen21-edit", Prompt = "a fox", Width = 32, Height = 32,
            ReferenceImage = ControlPng()
        });

        // Without ControlNet support the mode must fail loudly, never degrade
        // to ordinary text-to-image or the noise backend.
        string? error = WebUiCapabilities.ErrorFor(control,
            supportsImageEditing: true, supportsControlNet: false);
        Assert.NotNull(error);
        Assert.Contains("ControlNet", error);
        Assert.Contains("does not implement", error);
        Assert.Equal(WebUiCapabilities.ControlNetUnavailable, error);

        // With support (and for plain t2i) there is no error.
        Assert.Null(WebUiCapabilities.ErrorFor(control, true, true));
        Assert.Null(WebUiCapabilities.ErrorFor(
            new ZImageOptions { Prompt = "a", Width = 16, Height = 16 }, false, false));

        // Existing image-edit gate keeps its message (regression).
        Assert.Equal(WebUiCapabilities.EditingUnavailable,
            WebUiCapabilities.ErrorFor(edit, supportsImageEditing: false, supportsControlNet: true));
    }

    [Fact]
    public void Job_Dto_Control_State()
    {
        ZImageControlOptions control = Assert.IsType<ZImageControlOptions>(
            RequestMapper.ToOptions(ControlRequest()));
        ZImageOptions plain = new() { Prompt = "a", Width = 16, Height = 16 };

        // JobDto.From uses this pure helper for its Control / HasControl flags.
        Assert.Equal((true, true), JobDto.ControlState(control));
        Assert.Equal((false, false), JobDto.ControlState(plain));

        // The flags must serialise under the documented snake_case names.
        JobDto dto = new() { Control = true, HasControl = true };
        using JsonDocument doc = JsonDocument.Parse(
            JsonSerializer.Serialize(dto, WebUiJson.Options));
        Assert.True(doc.RootElement.GetProperty("control").GetBoolean());
        Assert.True(doc.RootElement.GetProperty("has_control").GetBoolean());
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
