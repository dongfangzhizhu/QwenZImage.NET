using System.Text;

namespace QwenZImage.NET.Validation;

/// <summary>
/// Managed mirror of the engine's per-model validation rules (see
/// docs/PARAMETERS.md). Failing fast here avoids a multi-minute model load for
/// a request the engine would reject anyway. The native layer repeats the same
/// checks for callers that bypass this wrapper.
/// </summary>
public static class RequestValidator
{
    /// <summary>Maximum prompt length, in UTF-8 bytes.</summary>
    public const int MaxPromptBytes = 16384;

    /// <summary>Maximum output side length, in pixels.</summary>
    public const int MaxCanvasSide = 2048;

    /// <summary>Maximum output area, in pixels.</summary>
    public const long MaxCanvasPixels = 2048L * 2048L;

    /// <summary>Validates any request kind.</summary>
    /// <exception cref="ArgumentException">A rule was violated.</exception>
    public static void Validate(GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        switch (options)
        {
            case QwenImageOptions qwen: Validate(qwen); break;
            case ZImageOptions zimage: Validate(zimage); break;
            default: throw new ArgumentException(
                $"Unsupported request type {options.GetType().FullName}.", nameof(options));
        }
    }

    /// <summary>Validates a Qwen-Image 2.1 request.</summary>
    public static void Validate(QwenImageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidatePrompt(options.Prompt);
        ValidateCanvas(options.Width, options.Height, 32);

        bool turbo = options.Tier == QwenImageTier.Turbo;
        if (!Enum.IsDefined(options.Tier))
            throw new ArgumentException("Tier must be Turbo or Base.", nameof(options));

        if (turbo)
        {
            if (!string.IsNullOrEmpty(options.NegativePrompt))
                throw new ArgumentException(
                    "NegativePrompt is not supported on the turbo tier (no CFG branch).",
                    nameof(options));
            if (options.Steps is not null && options.Steps != 8)
                throw new ArgumentException(
                    "Turbo tier requires Steps == 8; other values are rejected, not ignored.",
                    nameof(options));
            if (options.Guidance is not null && options.Guidance != 1.0f)
                throw new ArgumentException(
                    "Turbo tier requires Guidance == 1.0.", nameof(options));
        }
        else
        {
            if (options.Steps is < 1 or > 50)
                throw new ArgumentException("Base tier Steps must be 1..50.", nameof(options));
            if (options.Guidance is not null &&
                (options.Guidance < 1.0f || options.Guidance > 20.0f))
                throw new ArgumentException("Base tier Guidance must be 1..20.", nameof(options));
        }

        if (options.LoraScale is not null && string.IsNullOrEmpty(options.Lora))
            throw new ArgumentException(
                "LoraScale requires Lora; otherwise the engine would silently drop it.",
                nameof(options));
        if (!string.IsNullOrEmpty(options.Lora))
        {
            string lora = options.Lora!;
            if (lora.Contains('/') || lora.Contains('\\') || lora.Contains("..", StringComparison.Ordinal))
                throw new ArgumentException(
                    "Lora must be a bare file name without path separators.", nameof(options));
            float scale = options.EffectiveLoraScale;
            if (scale < 0.05f || scale > 4.0f)
                throw new ArgumentException("LoraScale must be 0.05..4.", nameof(options));
        }
    }

    /// <summary>Validates a Z-Image request.</summary>
    public static void Validate(ZImageOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidatePrompt(options.Prompt);
        ValidateCanvas(options.Width, options.Height, 16);

        string? checkpoint = options.Checkpoint;
        if (string.IsNullOrEmpty(checkpoint)) return;
        if (checkpoint.Contains("fp4", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException(
                "svdq-fp4 packs are not supported (the engine only decodes int4).",
                nameof(options));
        if (checkpoint != "bf16" && checkpoint != "nunchaku" &&
            !checkpoint.StartsWith("nunchaku-", StringComparison.Ordinal))
            throw new ArgumentException(
                $"Unknown Z-Image checkpoint '{checkpoint}'. Use bf16, nunchaku, " +
                "or an id listed by the engine's pack scan.", nameof(options));
    }

    private static void ValidatePrompt(string? prompt)
    {
        if (string.IsNullOrEmpty(prompt))
            throw new ArgumentException("Prompt is required.", nameof(prompt));
        int bytes = Encoding.UTF8.GetByteCount(prompt);
        if (bytes > MaxPromptBytes)
            throw new ArgumentException(
                $"Prompt is {bytes} UTF-8 bytes; the limit is {MaxPromptBytes}.", nameof(prompt));
    }

    private static void ValidateCanvas(int width, int height, int multiple)
    {
        if (width <= 0 || height <= 0 || width % multiple != 0 || height % multiple != 0 ||
            width > MaxCanvasSide || height > MaxCanvasSide)
            throw new ArgumentException(
                $"Width and height must be multiples of {multiple} in 1..{MaxCanvasSide}.");
        if ((long)width * height > MaxCanvasPixels)
            throw new ArgumentException(
                $"Canvas {width}x{height} exceeds the 2048*2048 pixel limit.");
    }
}
