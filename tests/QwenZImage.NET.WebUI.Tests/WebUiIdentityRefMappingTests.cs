using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// Wire-level tests for the fifth additive ABI v1 increment: identity_refs
/// (Wan2GP image_refs). The mapper must keep identity conditions completely
/// separate from the img2img reference_image path and accept both base64 PNG
/// data URLs and local PNG paths.
/// </summary>
public class WebUiIdentityRefMappingTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static string PngDataUrl(int size = 2, byte fill = 0x55)
        => "data:image/png;base64," + Convert.ToBase64String(
            MakePng(size, size, 2, Enumerable.Repeat(fill, size * size * 3).ToArray()));

    private static GenerateRequest IdentityRequest(Action<GenerateRequest>? tweak = null)
    {
        GenerateRequest request = new()
        {
            Model = "qwen21-identity",
            Prompt = "same person, different clothes",
            Width = 64,
            Height = 64,
            IdentityRefs = [PngDataUrl(2, 0x55), PngDataUrl(2, 0xAA)],
            IdentityRefWeight = 85.0f,
            ReferenceEncodeMaxSide = 512u,
            Client = "webui"
        };
        tweak?.Invoke(request);
        return request;
    }

    [Fact]
    public void Snake_Case_Identity_Fields_Bind_To_The_Dto()
    {
        string json = """
            {
              "model": "qwen21-identity",
              "identity_refs": ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"],
              "identity_ref_weight": 85,
              "reference_encode_max_side": 512
            }
            """;
        GenerateRequest? dto = JsonSerializer.Deserialize<GenerateRequest>(json, WebUiJson.Options);

        Assert.NotNull(dto);
        Assert.Equal(2, dto.IdentityRefs?.Count);
        Assert.Equal(85.0f, dto.IdentityRefWeight);
        Assert.Equal(512u, dto.ReferenceEncodeMaxSide);
    }

    [Fact]
    public void Identity_Model_Decodes_Every_Reference_And_The_Two_Knobs()
    {
        Qwen21IdentityRefOptions options = Assert.IsType<Qwen21IdentityRefOptions>(
            RequestMapper.ToOptions(IdentityRequest()));

        Assert.Equal(GenerationModel.Qwen21, options.Model);
        Assert.Equal(2, options.References.Count);
        Assert.All(options.References, r =>
        {
            Assert.Equal(2, r.Width);
            Assert.Equal(2, r.Height);
            Assert.Equal(3, r.Channels);
        });
        Assert.NotEqual(options.References[0].Planar.Span[0],
                        options.References[1].Planar.Span[0]);
        Assert.Equal(85.0f, options.EffectiveIdentityRefWeight);
        Assert.Equal(512u, options.EffectiveReferenceEncodeMaxSide);
    }

    [Fact]
    public void Identity_Refs_On_Plain_Qwen21_Opt_Into_Identity_Mode()
    {
        GenerateRequest request = IdentityRequest(tweak: r => r.Model = "qwen21");
        Qwen21IdentityRefOptions options = Assert.IsType<Qwen21IdentityRefOptions>(
            RequestMapper.ToOptions(request));
        Assert.Equal(2, options.References.Count);
        Assert.All(options.References, r => Assert.Equal(2, r.Width));
    }

    [Fact]
    public void Identity_Refs_And_Reference_Image_Are_Mutually_Exclusive()
    {
        GenerateRequest request = IdentityRequest(tweak: r => r.ReferenceImage = PngDataUrl());
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(request));
        Assert.Contains("mutually exclusive", error.Message);
    }

    [Fact]
    public void Identity_Refs_And_ControlNet_Are_Mutually_Exclusive()
    {
        GenerateRequest request = IdentityRequest(tweak: r => r.ControlImage = PngDataUrl());
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(request));
        Assert.Contains("mutually exclusive", error.Message);
    }

    [Fact]
    public void Identity_Knobs_Without_References_Are_Rejected()
    {
        GenerateRequest request = new()
        {
            Model = "qwen21",
            Prompt = "a fox",
            Width = 64,
            Height = 64,
            IdentityRefWeight = 70.0f
        };
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => RequestMapper.ToOptions(request));
        Assert.Contains("require identity_refs", error.Message);
    }

    [Fact]
    public void Local_Png_Paths_Are_Accepted_For_Scripted_Callers()
    {
        string path = Path.Combine(Path.GetTempPath(),
            $"qzi-identity-{Guid.NewGuid():N}.png");
        File.WriteAllBytes(path, MakePng(2, 2, 2, new byte[2 * 2 * 3]));
        try
        {
            GenerateRequest request = IdentityRequest(tweak: r => r.IdentityRefs = [path]);
            Qwen21IdentityRefOptions options = Assert.IsType<Qwen21IdentityRefOptions>(
                RequestMapper.ToOptions(request));
            Assert.Single(options.References);
            Assert.Equal(2, options.References[0].Width);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Capability_Gate_Rejects_Identity_Jobs_On_Older_Libraries()
    {
        Qwen21IdentityRefOptions options = Assert.IsType<Qwen21IdentityRefOptions>(
            RequestMapper.ToOptions(IdentityRequest()));
        Assert.Equal(WebUiCapabilities.IdentityReferenceUnavailable,
            WebUiCapabilities.ErrorFor(options, true, true, true, true, false));
        Assert.Null(WebUiCapabilities.ErrorFor(options, true, true, true, true, true));
    }

    [Fact]
    public void Job_Dto_Exposes_Identity_Badge_State()
    {
        Qwen21IdentityRefOptions options = Assert.IsType<Qwen21IdentityRefOptions>(
            RequestMapper.ToOptions(IdentityRequest()));
        Assert.True(JobDto.IdentityRefState(options).IdentityRef);
        Assert.Equal(2, JobDto.IdentityRefState(options).Count);
        Assert.False(JobDto.IdentityRefState(new QwenImageOptions
        {
            Prompt = "a fox", Width = 64, Height = 64
        }).IdentityRef);
    }

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
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++)
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
        }
        return crc ^ 0xFFFFFFFFu;
    }
}
