using System.Buffers.Binary;
using System.IO.Compression;

namespace QwenZImage.NET.WebUI.Preprocessing;

/// <summary>
/// Minimal 8-bit RGB PNG encoder. The core library ships a PNG *decoder*
/// (QziRasterImage) but deliberately no encoder; the WebUI needs one to return
/// the preprocessed control map as a data URL the browser can preview and post
/// back. Uses only BCL types (ZLibStream + a CRC32 table), so no imaging
/// package is required.
/// </summary>
internal static class PngWriter
{
    private static readonly uint[] CrcTable = BuildCrcTable();

    /// <summary>Encodes interleaved RGB8 bytes as a complete PNG file.</summary>
    public static byte[] EncodeRgb(ReadOnlySpan<byte> rgb, int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        int expected = checked(width * height * 3);
        if (rgb.Length < expected)
            throw new ArgumentException($"RGB buffer is {rgb.Length} bytes; expected {expected}.",
                                        nameof(rgb));

        using var output = new MemoryStream();
        output.Write([137, 80, 78, 71, 13, 10, 26, 10]);

        Span<byte> ihdr = stackalloc byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr[..4], (uint)width);
        BinaryPrimitives.WriteUInt32BigEndian(ihdr[4..8], (uint)height);
        ihdr[8] = 8;  // bit depth
        ihdr[9] = 2;  // colour type: truecolour RGB
        ihdr[10] = 0; // compression
        ihdr[11] = 0; // filter
        ihdr[12] = 0; // no interlace
        WriteChunk(output, "IHDR"u8, ihdr);

        // Filter type 0 (None) on every scanline; zlib then does the work.
        byte[] raw = new byte[checked((width * 3 + 1) * height)];
        int stride = width * 3 + 1;
        for (int y = 0; y < height; y++)
        {
            int dst = y * stride;
            raw[dst] = 0;
            rgb.Slice(y * width * 3, width * 3).CopyTo(raw.AsSpan(dst + 1));
        }

        using (var compressed = new MemoryStream())
        {
            using (var zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
                zlib.Write(raw, 0, raw.Length);
            WriteChunk(output, "IDAT"u8, compressed.ToArray());
        }
        WriteChunk(output, "IEND"u8, ReadOnlySpan<byte>.Empty);
        return output.ToArray();
    }

    /// <summary>Encodes and wraps the result in a data URL for the browser.</summary>
    public static string ToDataUrl(ReadOnlySpan<byte> rgb, int width, int height)
        => "data:image/png;base64," + Convert.ToBase64String(EncodeRgb(rgb, width, height));

    private static void WriteChunk(Stream output, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(length, (uint)data.Length);
        output.Write(length);
        output.Write(type);
        output.Write(data);

        // CRC covers the chunk type followed by the chunk data.
        byte[] crcInput = new byte[4 + data.Length];
        type.CopyTo(crcInput);
        data.CopyTo(crcInput.AsSpan(4));
        Span<byte> crc = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crc, Crc32(crcInput));
        output.Write(crc);
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }

    private static uint[] BuildCrcTable()
    {
        uint[] table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }
}
