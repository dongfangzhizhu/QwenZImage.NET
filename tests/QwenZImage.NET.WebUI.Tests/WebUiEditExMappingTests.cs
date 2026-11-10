using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// WebUI mapping for the third additive ABI v1 increment: outpainting margins,
/// LanPaint model modes and RGBA output. The browser only sends the advanced
/// envelope when a knob differs from its default, so ordinary qwen21-edit jobs
/// must keep mapping to the basic Qwen21EditOptions.
/// </summary>
public class WebUiEditExMappingTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static string ReferencePng()
        => "data:image/png;base64," + Convert.ToBase64String(
            MakePng(2, 2, 2, new byte[2 * 2 * 3]));

    private static string MaskPng()
        => "data:image/png;base64," + Convert.ToBase64String(
            MakePng(2, 2, 6, Enumerable.Repeat((byte)255, 2 * 2 * 4).ToArray()));

    private static GenerateRequest EditRequest(Action<GenerateRequest>? tweak = null)
    {
        GenerateRequest request = new()
        {
            Model = "qwen21-edit",
            Prompt = "extend the snowy forest",
            Width = 64,
            Height = 64,
            Seed = 7,
            ReferenceImage = ReferencePng(),
            MaskImage = MaskPng()
        };
        tweak?.Invoke(request);
        return request;
    }

    [Fact]
    public void Snake_Case_Advanced_Fields_Bind_To_The_Dto()
    {
        string json = """
            {
              "model": "qwen21-edit",
              "prompt": "extend the frame",
              "margin_top": 32,
              "margin_bottom": 0,
              "margin_left": 64,
              "margin_right": 32,
              "model_mode": 3,
              "rgba": 1
            }
            """;
        GenerateRequest? dto = JsonSerializer.Deserialize<GenerateRequest>(json, WebUiJson.Options);
        Assert.NotNull(dto);
        Assert.Equal(32, dto.MarginTop);
        Assert.Equal(0, dto.MarginBottom);
        Assert.Equal(64, dto.MarginLeft);
        Assert.Equal(32, dto.MarginRight);
        Assert.Equal(3, dto.ModelMode);
        Assert.Equal(1, dto.Rgba);
    }

    [Fact]
    public void Without_Advanced_Fields_The_Job_Stays_An_Ordinary_Edit()
    {
        GenerationOptions options = RequestMapper.ToOptions(EditRequest());
        Assert.IsType<Qwen21EditOptions>(options);
        Assert.False(options is Qwen21EditExOptions);
    }

    [Fact]
    public void Margins_Map_To_Advanced_Options_On_The_Final_Canvas()
    {
        Qwen21EditExOptions advanced = Assert.IsType<Qwen21EditExOptions>(
            RequestMapper.ToOptions(EditRequest(r =>
            {
                // Width / Height are the FINAL canvas: the 32 px top margin and
                // 64 px left margin must still leave a >= 32x32 source rectangle.
                r.Width = 256;
                r.Height = 256;
                r.MarginTop = 32;
                r.MarginLeft = 64;
                r.MarginRight = 0;
                r.MarginBottom = 0;
            })));

        Assert.Equal(256, advanced.Width);
        Assert.Equal(256, advanced.Height);
        Assert.Equal(32u, advanced.MarginTop);
        Assert.Equal(64u, advanced.MarginLeft);
        Assert.Equal(0u, advanced.MarginRight);
        Assert.True(advanced.HasOutpainting);
        Assert.False(advanced.IsLanPaint);
        Assert.False(advanced.Rgba);
        Assert.Equal(0u, advanced.Flags);
        Assert.Equal(7ul, advanced.Seed);
        Assert.NotNull(advanced.Reference);
    }

    [Fact]
    public void LanPaint_And_Rgba_Upgrade_Without_Margins()
    {
        Qwen21EditExOptions advanced = Assert.IsType<Qwen21EditExOptions>(
            RequestMapper.ToOptions(EditRequest(r =>
            {
                r.ModelMode = 4;
                r.Rgba = 1;
            })));

        Assert.Equal(Qwen21EditMode.LanPaint10, advanced.ModelMode);
        Assert.Equal(4u, advanced.EffectiveModelMode);
        Assert.True(advanced.IsLanPaint);
        Assert.True(advanced.Rgba);
        Assert.False(advanced.HasOutpainting);
        Assert.NotNull(advanced.Mask);
    }

    [Fact]
    public void Explicit_Zero_Values_Are_Still_An_Advanced_Request()
    {
        // The browser only sends the envelope when something differs, but a
        // script may post explicit zeros. That must not silently downgrade to
        // the ordinary edit export (the caller asked for the ex ABI).
        GenerationOptions options = RequestMapper.ToOptions(
            EditRequest(r => r.Rgba = 0));
        Assert.IsType<Qwen21EditExOptions>(options);
    }

    [Fact]
    public void Invalid_Advanced_Fields_Fail_Before_Queueing()
    {
        // 32-pixel alignment and the residual 32x32 source rectangle.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => r.MarginTop = 16)));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => r.MarginTop = 64)));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => { r.MarginLeft = 64; r.MarginRight = 64; })));
        // Negative margins cannot be cast to uint.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => r.MarginTop = -32)));
        // Model mode whitelist and RGBA flag.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => r.ModelMode = 7)));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => r.ModelMode = 1)));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => r.Rgba = 2)));
        // LanPaint needs a mask and cannot be combined with margins.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => { r.ModelMode = 3; r.MaskImage = null; })));
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            EditRequest(r => { r.ModelMode = 3; r.MarginTop = 32; })));
        // Advanced fields without a reference image.
        Assert.Throws<ArgumentException>(() => RequestMapper.ToOptions(
            new GenerateRequest { Model = "qwen21", Prompt = "a fox", MarginTop = 32 }));
    }

    [Fact]
    public void Capability_Gating_Covers_The_Third_Increment()
    {
        GenerationOptions advanced = RequestMapper.ToOptions(
            EditRequest(r => r.MarginTop = 32));
        GenerationOptions ordinary = RequestMapper.ToOptions(EditRequest());

        Assert.Equal(WebUiCapabilities.AdvancedEditingUnavailable,
                     WebUiCapabilities.ErrorFor(advanced, true, true, false));
        Assert.Null(WebUiCapabilities.ErrorFor(advanced, true, true, true));
        Assert.Equal(WebUiCapabilities.EditingUnavailable,
                     WebUiCapabilities.ErrorFor(ordinary, false, true, true));
        Assert.Null(WebUiCapabilities.ErrorFor(ordinary, true, true, false));

        // The two-argument overload predates the increment and keeps its
        // original behaviour for basic edit / ControlNet jobs.
        Assert.Equal(WebUiCapabilities.EditingUnavailable,
                     WebUiCapabilities.ErrorFor(ordinary, false, true));
    }

    [Fact]
    public void Job_Badges_Derive_From_The_Advanced_Request()
    {
        (bool advanced, bool outpaint, bool rgba) = JobDto.AdvancedEditState(
            RequestMapper.ToOptions(EditRequest(r =>
            {
                r.MarginTop = 32;
                r.Rgba = 1;
            })));
        Assert.True(advanced);
        Assert.True(outpaint);
        Assert.True(rgba);

        (advanced, outpaint, rgba) = JobDto.AdvancedEditState(
            RequestMapper.ToOptions(EditRequest()));
        Assert.False(advanced);
        Assert.False(outpaint);
        Assert.False(rgba);
    }

    [Fact]
    public void Health_Record_Serialises_The_Advanced_Flag()
    {
        string json = JsonSerializer.Serialize(
            new WebUiEngineHealth { SupportsAdvancedEditing = true }, WebUiJson.Options);
        using JsonDocument document = JsonDocument.Parse(json);
        Assert.True(document.RootElement.GetProperty("supports_advanced_editing").GetBoolean());
        Assert.True(document.RootElement.TryGetProperty("supports_controlnet", out _));
    }

    // --------------------------------------------------------------- helpers

    private static byte[] MakePng(int width, int height, int colorType, byte[] pixels)
    {
        int samples = colorType == 2 ? 3 : 4;
        using MemoryStream raw = new();
        for (int y = 0; y < height; y++)
        {
            raw.WriteByte(0);
            raw.Write(pixels, y * width * samples, width * samples);
        }
        using MemoryStream zlib = new();
        using (ZLibStream compressor = new(zlib, CompressionLevel.Optimal, leaveOpen: true))
            compressor.Write(raw.ToArray());

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
