using QwenZImage.NET;
using QwenZImage.NET.Validation;

namespace QwenZImage.NET.WebUI;

/// <summary>
/// Converts the browser's flat JSON envelope into the library's strongly typed
/// request and applies the same managed validation as <see cref="QziClient"/>.
/// Invalid envelopes fail with <see cref="ArgumentException"/> so the HTTP layer
/// can answer 400 before any minute-scale model loading happens.
/// </summary>
public static class RequestMapper
{
    // Mirrors ZImageControlOptions' canonical control types. WebUI keeps its
    // own copy so an unknown control_type fails with the same clear "unknown
    // value" style as an unknown qwen21 tier, before the library validator runs.
    private static readonly HashSet<string> CanonicalControlTypes = new(StringComparer.Ordinal)
    {
        "auto", "pose", "canny", "depth", "scribble", "raw"
    };

    public static GenerationOptions ToOptions(GenerateRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        string model = string.IsNullOrWhiteSpace(request.Model)
            ? "qwen21"
            : request.Model.Trim().ToLowerInvariant();

        // Both spellings are accepted: an explicit edit model, or a plain
        // qwen21 request that simply carries a reference image. Scripts that
        // already post "qwen21" can opt into editing by adding one field.
        bool editAlias = model is "qwen21-edit" or "qwen21_edit" or "qwen-image-edit";
        if (editAlias) model = "qwen21";
        bool wantsEdit = editAlias || !string.IsNullOrWhiteSpace(request.ReferenceImage);

        // Z-Image Turbo Fun ControlNet Union is the Z-Image counterpart of the
        // edit mode: an explicit "zimage-control" model, or a plain zimage
        // request that simply carries a control_image.
        bool controlAlias = model is "zimage-control" or "zimage_control" or "z-image-control";
        if (controlAlias) model = "zimage";
        bool wantsControl = controlAlias || !string.IsNullOrWhiteSpace(request.ControlImage);

        if (wantsEdit && wantsControl)
            throw new ArgumentException(
                "Image editing and Z-Image ControlNet are mutually exclusive features.");
        if (wantsEdit && model is not ("qwen21" or "qwen" or "qwen-image"))
            throw new ArgumentException(
                "Image editing is a Qwen-Image 2.1 feature; use model 'qwen21-edit'.");
        if (wantsControl && model is not ("zimage" or "z-image" or "z_image"))
            throw new ArgumentException(
                "Z-Image ControlNet is a Z-Image feature; use model 'zimage-control'.");
        if (!wantsEdit && HasEditParameters(request))
            throw new ArgumentException(
                "mask_image / strength / masking_strength and the advanced edit fields " +
                "(margin_*, model_mode, rgba) require reference_image.");
        if (!wantsControl && HasControlParameters(request))
            throw new ArgumentException(
                "control_type / control_scale require control_image.");

        GenerationOptions options = model switch
        {
            "qwen21" or "qwen" or "qwen-image" =>
                wantsEdit ? MapQwen21Edit(request) : MapQwen21(request),
            "zimage" or "z-image" or "z_image" =>
                wantsControl ? MapZImageControl(request) : MapZImage(request),
            _ => throw new ArgumentException(
                $"Unknown model '{request.Model}'. Use 'qwen21', 'qwen21-edit', " +
                "'zimage' or 'zimage-control'.")
        };

        RequestValidator.Validate(options);
        return options;
    }

    private static QwenImageOptions MapQwen21(GenerateRequest request)
    {
        QwenImageTier tier = request.Tier?.Trim().ToLowerInvariant() switch
        {
            null or "" or "turbo" => QwenImageTier.Turbo,
            "base" => QwenImageTier.Base,
            var other => throw new ArgumentException(
                $"Unknown qwen21 tier '{other}'. Use 'turbo' or 'base'.")
        };

        return new QwenImageOptions
        {
            Prompt = request.Prompt ?? string.Empty,
            Width = request.Width,
            Height = request.Height,
            Seed = request.Seed,
            Tier = tier,
            Steps = request.Steps,
            Guidance = request.Guidance,
            NegativePrompt = NullIfBlank(request.NegativePrompt),
            Lora = NullIfBlank(request.Lora),
            LoraScale = request.LoraScale
        };
    }

    /// <summary>
    /// Decodes the browser's PNG data URLs into planar rasters. Copying the
    /// common fields from <see cref="MapQwen21"/> keeps t2i and edit validation
    /// from drifting apart.
    /// </summary>
    private static GenerationOptions MapQwen21Edit(GenerateRequest request)
    {
        string reference = NullIfBlank(request.ReferenceImage)
            ?? throw new ArgumentException(
                "reference_image (a base64 PNG) is required for image editing.");
        if (!string.IsNullOrWhiteSpace(request.MaskImage) &&
            string.IsNullOrWhiteSpace(request.ReferenceImage))
            throw new ArgumentException("mask_image requires reference_image.");

        QwenImageOptions common = MapQwen21(request);
        QziRasterImage raster = QziRasterImage.FromPngBase64(reference);
        string? maskData = NullIfBlank(request.MaskImage);
        QziRasterImage? mask = maskData is null
            ? null
            : QziRasterImage.FromPngBase64(maskData, mask: true);

        // With no painted mask the whole image follows the denoising strength;
        // sending masking_strength in that case would be silently dropped.
        if (mask is null && request.MaskingStrength is not null)
            throw new ArgumentException(
                "masking_strength requires mask_image; without a mask there is no " +
                "region to restore from the reference.");
        float? masking = mask is null ? null : request.MaskingStrength;

        // No advanced field: keep the ordinary edit request on the second ABI
        // increment so older native libraries can still serve the job.
        if (!HasAdvancedParameters(request))
            return new Qwen21EditOptions
            {
                Prompt = common.Prompt,
                Width = common.Width,
                Height = common.Height,
                Seed = common.Seed,
                Tier = common.Tier,
                Steps = common.Steps,
                Guidance = common.Guidance,
                NegativePrompt = common.NegativePrompt,
                Lora = common.Lora,
                LoraScale = common.LoraScale,
                Reference = raster,
                Mask = mask,
                Strength = request.Strength,
                MaskingStrength = masking
            };

        Qwen21EditMode modelMode = request.ModelMode switch
        {
            null or 0 => Qwen21EditMode.MaskedDenoising,
            2 => Qwen21EditMode.LanPaint2,
            3 => Qwen21EditMode.LanPaint5,
            4 => Qwen21EditMode.LanPaint10,
            5 => Qwen21EditMode.LanPaint15,
            var other => throw new ArgumentException(
                $"Unknown model_mode '{other}'. Use 0 (masked denoising) or " +
                "2/3/4/5 (LanPaint 2/5/10/15).")
        };
        if (request.Rgba is not (null or 0 or 1))
            throw new ArgumentException("rgba must be 0 or 1.", nameof(request));

        // The library validator re-checks 32-pixel alignment, the residual
        // source rectangle, LanPaint's mask requirement and the mutual
        // exclusion with outpainting; the mapper only converts the JSON types.
        return new Qwen21EditExOptions
        {
            Prompt = common.Prompt,
            Width = common.Width,
            Height = common.Height,
            Seed = common.Seed,
            Tier = common.Tier,
            Steps = common.Steps,
            Guidance = common.Guidance,
            NegativePrompt = common.NegativePrompt,
            Lora = common.Lora,
            LoraScale = common.LoraScale,
            Reference = raster,
            Mask = mask,
            Strength = request.Strength,
            MaskingStrength = masking,
            MarginTop = Margin(request.MarginTop, "margin_top"),
            MarginBottom = Margin(request.MarginBottom, "margin_bottom"),
            MarginLeft = Margin(request.MarginLeft, "margin_left"),
            MarginRight = Margin(request.MarginRight, "margin_right"),
            ModelMode = modelMode,
            Rgba = request.Rgba == 1,
            Flags = 0
        };
    }

    private static uint Margin(int? value, string name)
    {
        if (value is null) return 0u;
        if (value < 0) throw new ArgumentException($"{name} must be non-negative.");
        return (uint)value.Value;
    }

    private static bool HasAdvancedParameters(GenerateRequest request)
        => request.MarginTop is not null || request.MarginBottom is not null ||
           request.MarginLeft is not null || request.MarginRight is not null ||
           request.ModelMode is not null || request.Rgba is not null;

    private static bool HasEditParameters(GenerateRequest request)
        => !string.IsNullOrWhiteSpace(request.MaskImage) ||
           request.Strength is not null || request.MaskingStrength is not null ||
           HasAdvancedParameters(request);

    private static ZImageOptions MapZImage(GenerateRequest request)
        => new()
        {
            Prompt = request.Prompt ?? string.Empty,
            Width = request.Width,
            Height = request.Height,
            Seed = request.Seed,
            Checkpoint = NullIfBlank(request.Checkpoint) ?? "bf16"
        };

    /// <summary>
    /// Decodes the browser's control-image PNG data URL into a planar RGB
    /// raster. Copying the common fields from <see cref="MapZImage"/> keeps t2i
    /// and ControlNet validation from drifting apart. The control image must
    /// already be preprocessed upstream: this library never estimates edges,
    /// poses or depth. The library validator still checks the type whitelist,
    /// scale bounds, flags and raster dimensions.
    /// </summary>
    private static ZImageControlOptions MapZImageControl(GenerateRequest request)
    {
        string controlData = NullIfBlank(request.ControlImage)
            ?? throw new ArgumentException(
                "control_image (a base64 PNG) is required for Z-Image ControlNet.");

        ZImageOptions common = MapZImage(request);
        QziRasterImage raster = QziRasterImage.FromPngBase64(controlData);

        string? type = NormalizeControlType(request.ControlType);
        if (type is not null && !CanonicalControlTypes.Contains(type))
            throw new ArgumentException(
                $"Unknown control_type '{request.ControlType}'. Use auto, pose, " +
                "canny, depth, scribble or raw.");

        return new ZImageControlOptions
        {
            Prompt = common.Prompt,
            Width = common.Width,
            Height = common.Height,
            Seed = common.Seed,
            Checkpoint = common.Checkpoint,
            ControlImage = raster,
            ControlType = type,
            // null (or an explicit 0) is the ABI sentinel for the engine
            // default 0.65; other values are validated in 0.05..2.0.
            ControlScale = request.ControlScale ?? 0.0f,
            Flags = 0
        };
    }

    private static bool HasControlParameters(GenerateRequest request)
        => !string.IsNullOrWhiteSpace(request.ControlType) ||
           request.ControlScale is not null;

    private static string? NormalizeControlType(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim().ToLowerInvariant();

    private static string? NullIfBlank(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
}
