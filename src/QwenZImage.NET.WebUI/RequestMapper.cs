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
    public static GenerationOptions ToOptions(GenerateRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        string model = string.IsNullOrWhiteSpace(request.Model)
            ? "qwen21"
            : request.Model.Trim().ToLowerInvariant();

        GenerationOptions options = model switch
        {
            "qwen21" or "qwen" or "qwen-image" => MapQwen21(request),
            "zimage" or "z-image" or "z_image" => MapZImage(request),
            _ => throw new ArgumentException(
                $"Unknown model '{request.Model}'. Use 'qwen21' or 'zimage'.")
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

    private static ZImageOptions MapZImage(GenerateRequest request)
        => new()
        {
            Prompt = request.Prompt ?? string.Empty,
            Width = request.Width,
            Height = request.Height,
            Seed = request.Seed,
            Checkpoint = NullIfBlank(request.Checkpoint) ?? "bf16"
        };

    private static string? NullIfBlank(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
}
