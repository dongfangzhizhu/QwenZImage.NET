using System.Buffers.Binary;
using System.IO.Compression;

namespace QwenZImage.NET;

/// <summary>
/// An 8-bit planar raster for Qwen-Image 2.1 image editing: three planes
/// <c>[R][G][B]</c>, each <c>Width * Height</c> bytes, for the reference image,
/// or one plane for an inpainting mask (255 = regenerate, 0 = keep). The native
/// ABI never takes ownership of the buffer; it is pinned only for the duration
/// of one synchronous generation call.
/// </summary>
/// <remarks>
/// PNG helpers decode 8-bit, non-interlaced RGB / RGBA / greyscale /
/// greyscale+alpha images with the standard ZLibStream, so callers do not need
/// System.Drawing or an imaging dependency. For the WebUI this means an HTML
/// canvas can upload a <c>data:image/png;base64,...</c> string directly.
/// </remarks>
public sealed class QziRasterImage
{
    /// <summary>Native limit on either reference side (qzi.c).</summary>
    public const int MaxSide = 4096;

    /// <summary>Native limit on the total number of reference pixels.</summary>
    public const long MaxPixels = 2048L * 2048L;

    private static ReadOnlySpan<byte> PngSignature => [137, 80, 78, 71, 13, 10, 26, 10];

    private readonly byte[] _bytes;

    private QziRasterImage(int width, int height, int channels, byte[] bytes)
    {
        Width = width;
        Height = height;
        Channels = channels;
        _bytes = bytes;
    }

    /// <summary>Raster width in pixels.</summary>
    public int Width { get; }

    /// <summary>Raster height in pixels.</summary>
    public int Height { get; }

    /// <summary>3 for a planar RGB reference, 1 for a mask.</summary>
    public int Channels { get; }

    /// <summary>Number of bytes in each channel plane.</summary>
    public int PlaneSize => Width * Height;

    /// <summary>The planar bytes, exposed read-only to callers.</summary>
    public ReadOnlyMemory<byte> Planar => _bytes;

    // The client lives in the same assembly and needs a pinnable array.
    internal byte[] Bytes => _bytes;

    /// <summary>Creates a raster from existing planar bytes (copied).</summary>
    public static QziRasterImage FromPlanar(ReadOnlySpan<byte> planar, int width,
                                            int height, int channels)
    {
        ValidateDimensions(width, height);
        if (channels is not (1 or 3))
            throw new ArgumentOutOfRangeException(nameof(channels),
                "Channels must be 3 (planar RGB) or 1 (mask).");
        long expected = (long)width * height * channels;
        if (planar.Length != expected)
            throw new ArgumentException(
                $"Planar buffer is {planar.Length} bytes; {width}x{height}x{channels} " +
                $"requires {expected}.", nameof(planar));
        return new QziRasterImage(width, height, channels, planar.ToArray());
    }

    /// <summary>Creates a planar RGB reference from interleaved RGB8 bytes.</summary>
    public static QziRasterImage FromInterleavedRgb(ReadOnlySpan<byte> rgb,
                                                    int width, int height)
    {
        ValidateDimensions(width, height);
        int plane = checked(width * height);
        if (rgb.Length != plane * 3)
            throw new ArgumentException(
                $"Interleaved RGB buffer is {rgb.Length} bytes; expected {plane * 3}.",
                nameof(rgb));
        byte[] planar = new byte[checked(plane * 3)];
        for (int i = 0; i < plane; i++)
        {
            planar[i] = rgb[i * 3];
            planar[plane + i] = rgb[i * 3 + 1];
            planar[2 * plane + i] = rgb[i * 3 + 2];
        }
        return new QziRasterImage(width, height, 3, planar);
    }

    /// <summary>Creates a one-channel mask from a solid value.</summary>
    public static QziRasterImage SolidMask(int width, int height, byte value)
    {
        ValidateDimensions(width, height);
        byte[] planar = new byte[checked(width * height)];
        if (value != 0) Array.Fill(planar, value);
        return new QziRasterImage(width, height, 1, planar);
    }

    /// <summary>Decodes an 8-bit PNG into a planar RGB reference.</summary>
    public static QziRasterImage FromPng(ReadOnlySpan<byte> png)
        => DecodePng(png, mask: false);

    /// <summary>
    /// Decodes an 8-bit PNG into a one-channel mask. The value is the RGB
    /// luminance scaled by alpha, so white brush strokes on a transparent
    /// canvas give 255 and untouched transparent pixels give 0.
    /// </summary>
    public static QziRasterImage FromPngMask(ReadOnlySpan<byte> png)
        => DecodePng(png, mask: true);

    /// <summary>
    /// Decodes a PNG supplied as base64 or as a
    /// <c>data:image/png;base64,...</c> data URL.
    /// </summary>
    public static QziRasterImage FromPngBase64(string base64, bool mask = false)
        => DecodePng(DecodeBase64(base64), mask);

    private static byte[] DecodeBase64(string base64)
    {
        ArgumentNullException.ThrowIfNull(base64);
        string payload = base64;
        if (payload.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            int comma = payload.IndexOf(',');
            if (comma < 0)
                throw new ArgumentException("Malformed data URL: no comma separator.",
                                            nameof(base64));
            payload = payload[(comma + 1)..];
        }
        try
        {
            return Convert.FromBase64String(payload);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException("The image payload is not valid base64 PNG data.",
                                        nameof(base64), ex);
        }
    }

    private static QziRasterImage DecodePng(ReadOnlySpan<byte> png, bool mask)
    {
        if (png.Length < 8 || !png[..8].SequenceEqual(PngSignature))
            throw new ArgumentException("Not a PNG file (bad signature).", nameof(png));

        int width = 0, height = 0, colorType = -1, bitDepth = 0;
        using MemoryStream compressed = new();
        int offset = 8;
        bool sawHeader = false, sawData = false;

        while (offset + 8 <= png.Length)
        {
            uint chunkLength = BinaryPrimitives.ReadUInt32BigEndian(png.Slice(offset, 4));
            if (chunkLength > int.MaxValue)
                throw new ArgumentException("PNG chunk is too large.", nameof(png));
            int length = (int)chunkLength;
            if (offset + 12 + length > png.Length)
                throw new ArgumentException("Truncated PNG chunk.", nameof(png));
            ReadOnlySpan<byte> type = png.Slice(offset + 4, 4);
            ReadOnlySpan<byte> data = png.Slice(offset + 8, length);
            offset += 12 + length; // 4 length + 4 type + data + 4 CRC

            if (type.SequenceEqual("IHDR"u8))
            {
                if (data.Length < 13)
                    throw new ArgumentException("Truncated PNG IHDR.", nameof(png));
                width = BinaryPrimitives.ReadInt32BigEndian(data[..4]);
                height = BinaryPrimitives.ReadInt32BigEndian(data.Slice(4, 4));
                bitDepth = data[8];
                colorType = data[9];
                byte compression = data[10], filter = data[11], interlace = data[12];
                ValidateDimensions(width, height);
                if (bitDepth != 8 || compression != 0 || filter != 0 || interlace != 0 ||
                    colorType is not (0 or 2 or 4 or 6))
                    throw new ArgumentException(
                        "Only non-interlaced 8-bit greyscale / RGB / RGBA PNGs are " +
                        "supported (no palette, 16-bit or Adam7 images).", nameof(png));
                sawHeader = true;
            }
            else if (type.SequenceEqual("IDAT"u8))
            {
                compressed.Write(data);
                sawData = true;
            }
            else if (type.SequenceEqual("IEND"u8))
            {
                break;
            }
        }

        if (!sawHeader || !sawData)
            throw new ArgumentException("PNG is missing IHDR or IDAT data.", nameof(png));

        int samplesPerPixel = colorType switch { 0 => 1, 2 => 3, 4 => 2, _ => 4 };
        int stride = checked(width * samplesPerPixel);
        int rawLength = checked(height * (stride + 1));
        byte[] raw = new byte[rawLength];
        compressed.Position = 0;
        using (ZLibStream zlib = new(compressed, CompressionMode.Decompress, leaveOpen: true))
        {
            int read = 0;
            while (read < raw.Length)
            {
                int count = zlib.Read(raw, read, raw.Length - read);
                if (count == 0) break;
                read += count;
            }
            if (read != raw.Length)
                throw new ArgumentException("Truncated PNG image data.", nameof(png));
        }

        byte[] pixels = Unfilter(raw, width, height, stride, samplesPerPixel);
        int plane = checked(width * height);
        int channels = mask ? 1 : 3;
        byte[] planar = new byte[checked(plane * channels)];

        for (int i = 0; i < plane; i++)
        {
            int r, g, b, a = 255;
            switch (colorType)
            {
                case 0:
                    r = g = b = pixels[i];
                    break;
                case 2:
                    r = pixels[i * 3]; g = pixels[i * 3 + 1]; b = pixels[i * 3 + 2];
                    break;
                case 4:
                    r = g = b = pixels[i * 2]; a = pixels[i * 2 + 1];
                    break;
                default:
                    r = pixels[i * 4]; g = pixels[i * 4 + 1]; b = pixels[i * 4 + 2];
                    a = pixels[i * 4 + 3];
                    break;
            }
            if (mask)
            {
                int luminance = (77 * r + 150 * g + 29 * b) >> 8;
                planar[i] = (byte)((luminance * a) / 255);
            }
            else
            {
                // Composite over white: the VAE always sees an OPAQUE reference,
                // so transparent pixels must not leak arbitrary RGB values.
                planar[i] = (byte)((r * a + 255 * (255 - a)) / 255);
                planar[plane + i] = (byte)((g * a + 255 * (255 - a)) / 255);
                planar[2 * plane + i] = (byte)((b * a + 255 * (255 - a)) / 255);
            }
        }
        return new QziRasterImage(width, height, channels, planar);
    }

    private static byte[] Unfilter(byte[] raw, int width, int height, int stride,
                                   int bytesPerPixel)
    {
        byte[] pixels = new byte[checked(height * stride)];
        byte[] previous = new byte[stride];
        byte[] current = new byte[stride];
        for (int y = 0; y < height; y++)
        {
            int start = y * (stride + 1);
            byte filter = raw[start];
            raw.AsSpan(start + 1, stride).CopyTo(current);
            for (int i = 0; i < stride; i++)
            {
                int left = i >= bytesPerPixel ? current[i - bytesPerPixel] : 0;
                int above = previous[i];
                int upperLeft = i >= bytesPerPixel ? previous[i - bytesPerPixel] : 0;
                int value = current[i];
                switch (filter)
                {
                    case 0: break;                              // None
                    case 1: value += left; break;               // Sub
                    case 2: value += above; break;              // Up
                    case 3: value += (left + above) / 2; break; // Average
                    case 4: value += Paeth(left, above, upperLeft); break;
                    default:
                        throw new ArgumentException(
                            $"Unsupported PNG row filter {filter}.", nameof(raw));
                }
                current[i] = (byte)value;
            }
            current.CopyTo(pixels, y * stride);
            (previous, current) = (current, previous);
        }
        return pixels;
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        return pb <= pc ? b : c;
    }

    private static void ValidateDimensions(int width, int height)
    {
        if (width <= 0 || height <= 0 || width > MaxSide || height > MaxSide ||
            (long)width * height > MaxPixels)
            throw new ArgumentException(
                $"Reference dimensions must be 1..{MaxSide} per side and at most " +
                "2048*2048 total pixels.");
    }
}
