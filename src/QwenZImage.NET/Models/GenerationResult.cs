using System.Buffers;

namespace QwenZImage.NET;

/// <summary>
/// A completed generation. The PNG bytes live in an <see cref="ArrayPool{T}"/>
/// buffer: the native buffer is released immediately after one memcpy, so
/// there is no cross-boundary lifetime to manage. Dispose returns the buffer
/// to the pool (it is safe to skip, but recommended for hot loops).
/// </summary>
public sealed class GenerationResult : IDisposable
{
    private byte[]? _buffer;
    private readonly int _length;

    internal GenerationResult(byte[] buffer, int length, int width, int height,
                              ulong seed, GenerationModel model, TimeSpan elapsed)
    {
        _buffer = buffer;
        _length = length;
        Width = width;
        Height = height;
        Seed = seed;
        Model = model;
        Elapsed = elapsed;
    }

    /// <summary>Byte length of the PNG.</summary>
    public int Length => _length;

    /// <summary>Image width in pixels.</summary>
    public int Width { get; }

    /// <summary>Image height in pixels.</summary>
    public int Height { get; }

    /// <summary>Seed used for this generation.</summary>
    public ulong Seed { get; }

    /// <summary>Which generator produced the image.</summary>
    public GenerationModel Model { get; }

    /// <summary>Wall-clock generation time.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>PNG bytes as a memory slice over the pooled buffer.</summary>
    public ReadOnlyMemory<byte> Png
    {
        get
        {
            byte[] buffer = _buffer ?? throw new ObjectDisposedException(nameof(GenerationResult));
            return new ReadOnlyMemory<byte>(buffer, 0, _length);
        }
    }

    /// <summary>PNG bytes as a span (zero-copy).</summary>
    public ReadOnlySpan<byte> PngSpan
    {
        get
        {
            byte[] buffer = _buffer ?? throw new ObjectDisposedException(nameof(GenerationResult));
            return new ReadOnlySpan<byte>(buffer, 0, _length);
        }
    }

    /// <summary>Writes the PNG to a stream.</summary>
    public void CopyTo(Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        destination.Write(PngSpan);
    }

    /// <summary>Writes the PNG to a stream asynchronously.</summary>
    public ValueTask CopyToAsync(Stream destination, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination);
        return destination.WriteAsync(Png, cancellationToken);
    }

    /// <summary>Saves the PNG to a file.</summary>
    public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        await using FileStream stream = new(path, FileMode.Create, FileAccess.Write,
                                            FileShare.None, 64 * 1024, useAsync: true);
        await CopyToAsync(stream, cancellationToken);
    }

    /// <summary>Returns the pooled buffer. The <see cref="Png"/> memory is invalid afterwards.</summary>
    public void Dispose()
    {
        byte[]? buffer = Interlocked.Exchange(ref _buffer, null);
        if (buffer is not null) ArrayPool<byte>.Shared.Return(buffer);
    }
}
