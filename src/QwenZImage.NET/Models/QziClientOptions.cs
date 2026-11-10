namespace QwenZImage.NET;

/// <summary>Engine configuration passed once at <see cref="QziClient"/> creation.</summary>
public sealed class QziClientOptions
{
    /// <summary>
    /// Absolute path to the qzi native library (QwenZImageNative.dll / libQwenZImageNative.so /
    /// libQwenZImageNative.dylib). When unset, <c>QZI_NATIVE_LIBRARY</c> and the runtime's
    /// normal probing are used.
    /// </summary>
    public string? NativeLibraryPath { get; init; }

    /// <summary>Qwen-Image 2.1 model pack directory (dit_turbo/base, vae, text, tokenizer, loras).</summary>
    public string? Qwen21Directory { get; init; }

    /// <summary>Z-Image model pack directory (dit, text_encoder, vae, nunchaku packs).</summary>
    public string? ZImageDirectory { get; init; }

    /// <summary>Text-embedding cache directory.</summary>
    public string? CacheDirectory { get; init; }

    /// <summary>
    /// Directory holding Qwen-Image 2.1 LoRA adapters. The C ABI has no field
    /// for it, so the client scopes <c>QZI_QWEN21_LORA_DIR</c> around engine
    /// creation for the native backend. Code-level values win over the config
    /// file; the variable is restored afterwards, so clients can differ.
    /// </summary>
    public string? Qwen21LoraDirectory { get; init; }

    /// <summary>
    /// Directory holding Z-Image Nunchaku / SVDQuant int4 repacks. Scoped as
    /// <c>QZI_ZIMAGE_NUNCHAKU_DIR</c> around engine creation.
    /// </summary>
    public string? ZImageNunchakuDirectory { get; init; }

    /// <summary>
    /// Directory holding Z-Image Turbo Fun ControlNet Union weights. The C ABI
    /// has no field for it, so the client scopes <c>QZI_ZIMAGE_CONTROL_DIR</c>
    /// around engine creation for the native backend (the real engine reads it
    /// in its ControlNet init path). Code-level values win over the config file;
    /// the variable is restored afterwards, so clients can differ.
    /// </summary>
    public string? ZImageControlDirectory { get; init; }

    /// <summary>
    /// Path to <c>QwenZImage.config.json</c>. When set, it is scoped as the
    /// <c>QZI_CONFIG</c> environment variable around engine creation so the
    /// native backend reads the very same file (including platform overrides
    /// and any fields not explicitly set here).
    /// </summary>
    public string? ConfigPath { get; init; }

    /// <summary>Compute device index; -1 selects automatically.</summary>
    public int DeviceId { get; init; } = -1;

    /// <summary>CPU thread count; 0 selects the engine default.</summary>
    public int ThreadCount { get; init; } = 0;

    /// <summary>
    /// Reads <c>QwenZImage.config.json</c> (explicit path, <c>QZI_CONFIG</c>,
    /// next to the app binary, the working directory, then
    /// <c>~/.config/QwenZImage/config.json</c>) and turns it into options.
    /// <c>QZI_*</c> environment variables override individual values.
    /// </summary>
    public static QziClientOptions FromConfig(string? explicitPath = null)
        => QziConfig.Load(explicitPath).ToClientOptions();
}
