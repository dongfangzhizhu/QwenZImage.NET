using System.Buffers;

namespace QwenZImage.NET.WebUI.Preprocessing;

/// <summary>
/// Dependency-free 8-bit image operators used to turn an ordinary uploaded
/// photograph into a Z-Image ControlNet guide map. The input is the core
/// library's planar RGB layout ([R plane][G plane][B plane]); every operator
/// returns a one-channel byte map. No System.Drawing, ImageSharp or OpenCV is
/// needed, so this works unchanged on Windows and Linux CI hosts.
/// </summary>
internal static class RasterOps
{
    /// <summary>Rec.601 luma, integer only (no per-pixel float allocations).</summary>
    public static byte[] ToGrayscale(ReadOnlySpan<byte> planar, int width, int height)
    {
        int plane = checked(width * height);
        if (planar.Length < plane * 3)
            throw new ArgumentException("Planar RGB buffer is too small.", nameof(planar));
        byte[] gray = new byte[plane];
        for (int i = 0; i < plane; i++)
        {
            int r = planar[i];
            int g = planar[plane + i];
            int b = planar[2 * plane + i];
            gray[i] = (byte)((r * 77 + g * 150 + b * 29) >> 8);
        }
        return gray;
    }

    /// <summary>Rec.601 luma computed directly from interleaved RGB8 bytes.</summary>
    public static byte[] ToGrayscaleInterleaved(ReadOnlySpan<byte> rgb, int width, int height)
    {
        int plane = checked(width * height);
        if (rgb.Length < plane * 3)
            throw new ArgumentException("Interleaved RGB buffer is too small.", nameof(rgb));
        byte[] gray = new byte[plane];
        for (int i = 0; i < plane; i++)
            gray[i] = (byte)((rgb[i * 3] * 77 + rgb[i * 3 + 1] * 150 +
                              rgb[i * 3 + 2] * 29) >> 8);
        return gray;
    }

    /// <summary>Separable 5-tap Gaussian approximation (1,4,6,4,1)/16.</summary>
    public static byte[] GaussianBlur(byte[] gray, int width, int height)
    {
        int plane = width * height;
        float[] tmp = new float[plane];
        byte[] result = new byte[plane];
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                int a = gray[row + Clamp(x - 2, width)];
                int b = gray[row + Clamp(x - 1, width)];
                int c = gray[row + x];
                int d = gray[row + Clamp(x + 1, width)];
                int e = gray[row + Clamp(x + 2, width)];
                tmp[row + x] = (a + 4 * b + 6 * c + 4 * d + e) / 16f;
            }
        }
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                int y0 = Clamp(y - 2, height) * width;
                int y1 = Clamp(y - 1, height) * width;
                int y3 = Clamp(y + 1, height) * width;
                int y4 = Clamp(y + 2, height) * width;
                float v = (tmp[y0 + x] + 4 * tmp[y1 + x] + 6 * tmp[row + x] +
                           4 * tmp[y3 + x] + tmp[y4 + x]) / 16f;
                result[row + x] = (byte)Math.Clamp((int)v, 0, 255);
            }
        }
        return result;
    }

    /// <summary>
    /// Classic Canny: Sobel gradients, non-maximum suppression and hysteresis.
    /// Returns 255 for an edge pixel and 0 for the background.
    /// </summary>
    public static byte[] Canny(byte[] gray, int width, int height,
                               float lowRatio = 0.15f, float highRatio = 0.35f)
    {
        byte[] blur = GaussianBlur(gray, width, height);
        int plane = width * height;
        float[] magnitude = new float[plane];
        byte[] direction = new byte[plane]; // 0 = E/W, 1 = NE/SW, 2 = N/S, 3 = NW/SE

        for (int y = 1; y < height - 1; y++)
        {
            int row = y * width;
            for (int x = 1; x < width - 1; x++)
            {
                int i = row + x;
                int tl = blur[i - width - 1], tc = blur[i - width], tr = blur[i - width + 1];
                int ml = blur[i - 1], mr = blur[i + 1];
                int bl = blur[i + width - 1], bc = blur[i + width], br = blur[i + width + 1];
                int gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
                int gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
                float mag = MathF.Sqrt(gx * (float)gx + gy * (float)gy);
                magnitude[i] = mag;
                float angle = MathF.Atan2(gy, gx) * 180f / MathF.PI;
                if (angle < 0) angle += 180f;
                direction[i] = angle switch
                {
                    < 22.5f or >= 157.5f => 0,
                    < 67.5f => 1,
                    < 112.5f => 2,
                    _ => 3
                };
            }
        }

        float max = 1f;
        foreach (float m in magnitude) if (m > max) max = m;
        // Relative thresholds keep the detector stable across photos with very
        // different contrast; users can override them from the API.
        float low = Math.Clamp(lowRatio, 0.01f, 0.99f) * max;
        float high = Math.Clamp(highRatio, lowRatio + 0.01f, 1f) * max;

        byte[] edges = new byte[plane];
        for (int y = 1; y < height - 1; y++)
        {
            int row = y * width;
            for (int x = 1; x < width - 1; x++)
            {
                int i = row + x;
                float m = magnitude[i];
                if (m < low) continue;
                float a, b;
                switch (direction[i])
                {
                    case 0: a = magnitude[i - 1]; b = magnitude[i + 1]; break;
                    case 1: a = magnitude[i - width + 1]; b = magnitude[i + width - 1]; break;
                    case 2: a = magnitude[i - width]; b = magnitude[i + width]; break;
                    default: a = magnitude[i - width - 1]; b = magnitude[i + width + 1]; break;
                }
                if (m < a || m < b) continue; // not a local maximum
                edges[i] = m >= high ? (byte)2 : (byte)1; // strong / weak
            }
        }

        // Hysteresis: every weak pixel connected (8-neighbourhood) to a strong
        // one becomes an edge. An explicit stack avoids recursion on big maps.
        int[] stack = ArrayPool<int>.Shared.Rent(plane);
        try
        {
            int top = 0;
            for (int i = 0; i < plane; i++)
                if (edges[i] == 2) stack[top++] = i;
            while (top > 0)
            {
                int i = stack[--top];
                int x = i % width, y = i / width;
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                    int n = ny * width + nx;
                    if (edges[n] != 1) continue;
                    edges[n] = 2;
                    if (top < stack.Length) stack[top++] = n;
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(stack);
        }

        for (int i = 0; i < plane; i++) edges[i] = edges[i] == 2 ? (byte)255 : (byte)0;
        return edges;
    }

    /// <summary>Zhang-Suen thinning: turns thick Canny ridges into 1px lines.</summary>
    public static byte[] Thin(byte[] binary, int width, int height)
    {
        byte[] map = (byte[])binary.Clone();
        int plane = width * height;
        bool changed = true;
        for (int pass = 0; changed && pass < 32; pass++)
        {
            changed = false;
            for (int step = 0; step < 2; step++)
            {
                var removed = new List<int>();
                for (int y = 1; y < height - 1; y++)
                for (int x = 1; x < width - 1; x++)
                {
                    int i = y * width + x;
                    if (map[i] == 0) continue;
                    int p2 = Bit(map[i - width]), p3 = Bit(map[i - width + 1]);
                    int p4 = Bit(map[i + 1]), p5 = Bit(map[i + width + 1]);
                    int p6 = Bit(map[i + width]), p7 = Bit(map[i + width - 1]);
                    int p8 = Bit(map[i - 1]), p9 = Bit(map[i - width - 1]);
                    int neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    if (neighbours < 2 || neighbours > 6) continue;
                    int transitions = 0;
                    int[] seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
                    for (int k = 0; k < 8; k++) if (seq[k] == 0 && seq[k + 1] == 1) transitions++;
                    if (transitions != 1) continue;
                    if (step == 0)
                    {
                        if (p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0) continue;
                    }
                    else
                    {
                        if (p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0) continue;
                    }
                    removed.Add(i);
                }
                foreach (int i in removed) { map[i] = 0; changed = true; }
            }
        }
        _ = plane;
        return map;
    }

    private static int Bit(byte v) => v == 0 ? 0 : 1;

    /// <summary>
    /// Deterministic relative-depth proxy. This is NOT Depth Anything V2: it
    /// combines a large-scale luminance prior (dark = distant), a vertical
    /// horizon cue and local edge density (textured regions tend to be nearer).
    /// It produces a usable monocular control map on hosts with no ONNX model;
    /// supplying a Depth Anything V2 ONNX file switches to the learned map.
    /// </summary>
    public static byte[] DepthProxy(byte[] gray, int width, int height)
    {
        int plane = width * height;
        byte[] blur = GaussianBlur(GaussianBlur(gray, width, height), width, height);
        byte[] edges = Canny(gray, width, height, 0.20f, 0.45f);
        byte[] result = new byte[plane];
        for (int y = 0; y < height; y++)
        {
            float vertical = 1f - (float)y / Math.Max(1, height - 1); // sky = far
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                int luminance = blur[i];
                // Coarse edge density in a 5x5 window.
                int localEdges = 0;
                for (int dy = -2; dy <= 2; dy += 2)
                for (int dx = -2; dx <= 2; dx += 2)
                {
                    int nx = Math.Clamp(x + dx, 0, width - 1);
                    int ny = Math.Clamp(y + dy, 0, height - 1);
                    if (edges[ny * width + nx] != 0) localEdges++;
                }
                float depth = 0.55f * (luminance / 255f) +
                              0.25f * (1f - vertical) +
                              0.20f * (localEdges / 9f);
                result[i] = (byte)Math.Clamp((int)(depth * 255f), 0, 255);
            }
        }
        return GaussianBlur(result, width, height);
    }

    /// <summary>Expands a one-channel map to interleaved RGB8, optionally inverted.</summary>
    public static byte[] ToInterleavedRgb(byte[] map, int width, int height, bool invert = false)
    {
        int plane = width * height;
        if (map.Length < plane) throw new ArgumentException("Map is too small.", nameof(map));
        byte[] rgb = new byte[checked(plane * 3)];
        for (int i = 0; i < plane; i++)
        {
            byte v = invert ? (byte)(255 - map[i]) : map[i];
            rgb[i * 3] = v;
            rgb[i * 3 + 1] = v;
            rgb[i * 3 + 2] = v;
        }
        return rgb;
    }

    /// <summary>
    /// Converts the core library's planar RGB raster into interleaved RGB8.
    /// </summary>
    public static byte[] PlanarToInterleaved(ReadOnlySpan<byte> planar, int width, int height)
    {
        int plane = checked(width * height);
        byte[] rgb = new byte[checked(plane * 3)];
        for (int i = 0; i < plane; i++)
        {
            rgb[i * 3] = planar[i];
            rgb[i * 3 + 1] = planar[plane + i];
            rgb[i * 3 + 2] = planar[2 * plane + i];
        }
        return rgb;
    }

    /// <summary>Bilinear resize of a one-channel map.</summary>
    public static byte[] Resize(byte[] src, int width, int height, int newWidth, int newHeight)
    {
        byte[] dst = new byte[checked(newWidth * newHeight)];
        for (int y = 0; y < newHeight; y++)
        {
            float sy = (y + 0.5f) * height / newHeight - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(sy), 0, height - 1);
            int y1 = Math.Clamp(y0 + 1, 0, height - 1);
            float fy = Math.Clamp(sy - y0, 0f, 1f);
            for (int x = 0; x < newWidth; x++)
            {
                float sx = (x + 0.5f) * width / newWidth - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(sx), 0, width - 1);
                int x1 = Math.Clamp(x0 + 1, 0, width - 1);
                float fx = Math.Clamp(sx - x0, 0f, 1f);
                float a = src[y0 * width + x0] * (1 - fx) + src[y0 * width + x1] * fx;
                float b = src[y1 * width + x0] * (1 - fx) + src[y1 * width + x1] * fx;
                dst[y * newWidth + x] = (byte)Math.Clamp((int)(a * (1 - fy) + b * fy), 0, 255);
            }
        }
        return dst;
    }

    /// <summary>Bilinear resize of interleaved RGB8.</summary>
    public static byte[] ResizeRgb(byte[] src, int width, int height, int newWidth, int newHeight)
    {
        byte[] dst = new byte[checked(newWidth * newHeight * 3)];
        for (int c = 0; c < 3; c++)
        {
            byte[] plane = new byte[width * height];
            for (int i = 0; i < plane.Length; i++) plane[i] = src[i * 3 + c];
            byte[] resized = Resize(plane, width, height, newWidth, newHeight);
            for (int i = 0; i < resized.Length; i++) dst[i * 3 + c] = resized[i];
        }
        return dst;
    }

    /// <summary>Draws a filled anti-alias-free disc into an interleaved RGB canvas.</summary>
    public static void FillDisc(byte[] rgb, int width, int height, float cx, float cy,
                                float radius, byte r, byte g, byte b)
    {
        int x0 = Math.Max(0, (int)(cx - radius - 1)), x1 = Math.Min(width - 1, (int)(cx + radius + 1));
        int y0 = Math.Max(0, (int)(cy - radius - 1)), y1 = Math.Min(height - 1, (int)(cy + radius + 1));
        float rr = radius * radius;
        for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++)
        {
            float dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy > rr) continue;
            int i = (y * width + x) * 3;
            rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b;
        }
    }

    /// <summary>Draws a thick line into an interleaved RGB canvas.</summary>
    public static void DrawLine(byte[] rgb, int width, int height, float x0, float y0,
                                float x1, float y1, float thickness, byte r, byte g, byte b)
    {
        float dx = x1 - x0, dy = y1 - y0;
        float length = MathF.Sqrt(dx * dx + dy * dy);
        if (length < 0.001f) { FillDisc(rgb, width, height, x0, y0, thickness, r, g, b); return; }
        int steps = Math.Max(1, (int)(length * 2));
        for (int s = 0; s <= steps; s++)
        {
            float t = (float)s / steps;
            FillDisc(rgb, width, height, x0 + dx * t, y0 + dy * t, thickness, r, g, b);
        }
    }

    private static int Clamp(int value, int max) => Math.Clamp(value, 0, max - 1);
}
