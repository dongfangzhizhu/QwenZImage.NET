namespace QwenZImage.NET;

/// <summary>Engine configuration passed once at <see cref="QziClient"/> creation.</summary>
public sealed class QziClientOptions
{
    /// <summary>
    /// Absolute path to the qzi native library (qzi.dll / libqzi.so /
    /// libqzi.dylib). When unset, <c>QZI_NATIVE_LIBRARY</c> and the runtime's
    /// normal probing are used.
    /// </summary>
    public string? NativeLibraryPath { get; init; }

    /// <summary>Qwen-Image 2.1 model pack directory (dit_turbo/base, vae, text, tokenizer, loras).</summary>
    public string? Qwen21Directory { get; init; }

    /// <summary>Z-Image model pack directory (dit, text_encoder, vae, nunchaku packs).</summary>
    public string? ZImageDirectory { get; init; }

    /// <summary>Text-embedding cache directory.</summary>
    public string? CacheDirectory { get; init; }

    /// <summary>Compute device index; -1 selects automatically.</summary>
    public int DeviceId { get; init; } = -1;

    /// <summary>CPU thread count; 0 selects the engine default.</summary>
    public int ThreadCount { get; init; } = 0;
}
