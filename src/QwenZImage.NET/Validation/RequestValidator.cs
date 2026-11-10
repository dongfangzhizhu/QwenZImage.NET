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

    /// <summary>Maximum LoRA adapter file name length, in UTF-8 bytes.</summary>
    public const int MaxLoraNameBytes = 255;

    /// <summary>
    /// WanGP <c>OUTPAINTING_ALIGNMENT</c>: outpainting margins and the residual
    /// source rectangle must be multiples of 32 output-canvas pixels.
    /// </summary>
    public const uint OutpaintAlignment = 32;

    /// <summary>Validates any request kind.</summary>
    /// <exception cref="ArgumentException">A rule was violated.</exception>
    public static void Validate(GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        switch (options)
        {
            // Derived options must be matched before their base record or their
            // extra rules would be skipped. Advanced editing derives from the
            // ordinary edit options, so it must come first.
            case Qwen21EditExOptions advanced: Validate(advanced); break;
            case Qwen21EditOptions edit: Validate(edit); break;
            case QwenImageOptions qwen: Validate(qwen); break;
            case ZImageControlOptions control: Validate(control); break;
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
                (!float.IsFinite(options.Guidance.Value) ||
                 options.Guidance < 1.0f || options.Guidance > 20.0f))
                throw new ArgumentException(
                    "Base tier Guidance must be finite and in 1..20.", nameof(options));
        }

        if (options.LoraScale is not null && string.IsNullOrEmpty(options.Lora))
            throw new ArgumentException(
                "LoraScale requires Lora; otherwise the engine would silently drop it.",
                nameof(options));
        if (!string.IsNullOrEmpty(options.Lora))
        {
            if (!IsBareLoraName(options.Lora!))
                throw new ArgumentException(
                    $"Lora must be a bare file name of at most {MaxLoraNameBytes} UTF-8 bytes " +
                    "(no path separators, ':', '..' or control characters).", nameof(options));
            float scale = options.EffectiveLoraScale;
            if (!float.IsFinite(scale) || scale < 0.05f || scale > 4.0f)
                throw new ArgumentException(
                    "LoraScale must be finite and in 0.05..4.", nameof(options));
        }
    }

    /// <summary>Validates a Qwen-Image 2.1 image-edit request.</summary>
    public static void Validate(Qwen21EditOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        Validate((QwenImageOptions)options);

        QziRasterImage? reference = options.Reference;
        if (reference is null)
            throw new ArgumentException("Reference is required for image editing.",
                                        nameof(options));
        if (reference.Channels != 3)
            throw new ArgumentException(
                "Reference must be a 3-channel planar RGB image.", nameof(options));

        if (options.Mask is { } mask)
        {
            if (mask.Channels != 1)
                throw new ArgumentException(
                    "Mask must be a single-channel 8-bit raster.", nameof(options));
            if (mask.Width != reference.Width || mask.Height != reference.Height)
                throw new ArgumentException(
                    "Mask dimensions must match the reference image, otherwise the " +
                    "painted region would be offset.", nameof(options));
        }

        if (options.Strength is { } strength &&
            (!float.IsFinite(strength) || strength < 0.05f || strength > 1.0f))
            throw new ArgumentException(
                "Strength must be finite and in 0.05..1.0 (or null for 0.75).",
                nameof(options));

        if (options.MaskingStrength is { } masking)
        {
            if (!float.IsFinite(masking) || masking < 0.05f || masking > 1.0f)
                throw new ArgumentException(
                    "MaskingStrength must be finite and in 0.05..1.0.",
                    nameof(options));
            if (options.Mask is null)
                throw new ArgumentException(
                    "MaskingStrength requires Mask; otherwise it would be silently dropped.",
                    nameof(options));
        }
    }

    /// <summary>
    /// Validates the third additive editing request. Mirrors
    /// <c>qzi_validate_qwen21_edit_ex()</c>: the ordinary edit rules apply to
    /// the embedded base request, then the Red Canvas margins, the LanPaint
    /// mode and the reserved flags are checked. Rejecting an explicit strength
    /// under LanPaint avoids silently overriding a caller's choice: WanGP (and
    /// the native layer) force both strengths to 1.0 in that mode.
    /// </summary>
    public static void Validate(Qwen21EditExOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        Validate((Qwen21EditOptions)options);

        uint[] margins = [options.MarginTop, options.MarginBottom,
                          options.MarginLeft, options.MarginRight];
        foreach (uint margin in margins)
            if (margin % OutpaintAlignment != 0)
                throw new ArgumentException(
                    $"Outpainting margins must be multiples of {OutpaintAlignment} pixels.",
                    nameof(options));

        long sourceWidth = (long)options.Width - options.MarginLeft - options.MarginRight;
        long sourceHeight = (long)options.Height - options.MarginTop - options.MarginBottom;
        if (sourceWidth < OutpaintAlignment || sourceHeight < OutpaintAlignment)
            throw new ArgumentException(
                $"Outpainting must leave at least a {OutpaintAlignment}x{OutpaintAlignment} " +
                "source rectangle inside the output canvas.", nameof(options));

        if (!Enum.IsDefined(options.ModelMode))
            throw new ArgumentException(
                "ModelMode must be MaskedDenoising (0) or LanPaint2/5/10/15 (2/3/4/5).",
                nameof(options));

        if (options.IsLanPaint)
        {
            if (options.Mask is null)
                throw new ArgumentException(
                    "LanPaint requires Mask; it refines a masked edit region.",
                    nameof(options));
            if (options.HasOutpainting)
                throw new ArgumentException(
                    "LanPaint and outpainting are mutually exclusive (WanGP disables " +
                    "LanPaint for Red Canvas outpainting).", nameof(options));
            if (options.Strength is { } strength && strength != 1.0f)
                throw new ArgumentException(
                    "LanPaint forces denoising strength to 1.0; leave Strength unset.",
                    nameof(options));
            if (options.MaskingStrength is { } masking && masking != 1.0f)
                throw new ArgumentException(
                    "LanPaint forces masking strength to 1.0; leave MaskingStrength unset.",
                    nameof(options));
        }

        if (options.Flags != 0)
            throw new ArgumentException("Flags is reserved and must be 0.", nameof(options));
    }

    /// <summary>
    /// Mirrors <c>qzi.c</c>'s <c>no_path_separators()</c>: the name is joined
    /// onto <c>qwen21_lora_dir</c> by the engine, so it must never be able to
    /// escape that directory or be truncated by a fixed-size buffer.
    /// </summary>
    private static bool IsBareLoraName(string name)
    {
        if (string.IsNullOrEmpty(name) ||
            Encoding.UTF8.GetByteCount(name) > MaxLoraNameBytes ||
            name.Contains('/') || name.Contains('\\') || name.Contains(':') ||
            name.Contains("..", StringComparison.Ordinal))
            return false;
        foreach (char ch in name)
            if (ch < ' ' || ch == '\u007f') return false;
        return true;
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

    /// <summary>
    /// Validates a Z-Image Turbo Fun ControlNet Union request. Mirrors
    /// <c>qzi_validate_zimage_control()</c>: the ordinary Z-Image rules apply to
    /// the base request, plus the control image, type, scale and reserved flags.
    /// </summary>
    public static void Validate(ZImageControlOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        Validate((ZImageOptions)options);

        QziRasterImage? control = options.ControlImage;
        if (control is null)
            throw new ArgumentException(
                "ControlImage is required for ControlNet generation.", nameof(options));
        if (control.Channels != 3)
            throw new ArgumentException(
                "ControlImage must be a 3-channel planar RGB image.", nameof(options));
        if (control.Width <= 0 || control.Height <= 0 ||
            control.Width > QziRasterImage.MaxSide ||
            control.Height > QziRasterImage.MaxSide ||
            (long)control.Width * control.Height > QziRasterImage.MaxPixels)
            throw new ArgumentException(
                $"ControlImage dimensions must be 1..{QziRasterImage.MaxSide} per side and " +
                "at most 2048*2048 total pixels.", nameof(options));

        string? type = options.EffectiveControlType;
        if (!ZImageControlOptions.IsValidControlType(type))
            throw new ArgumentException(
                "ControlType must be null/auto/pose/canny/depth/scribble/raw " +
                "(case-insensitive).", nameof(options));

        float scale = options.ControlScale;
        if (!float.IsFinite(scale) || (scale != 0.0f && (scale < 0.05f || scale > 2.0f)))
            throw new ArgumentException(
                "ControlScale must be 0 (engine default 0.65) or finite and in 0.05..2.0.",
                nameof(options));

        if (options.Flags != 0)
            throw new ArgumentException("Flags is reserved and must be 0.", nameof(options));
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
