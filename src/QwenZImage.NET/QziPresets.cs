namespace QwenZImage.NET;

/// <summary>A named, engine-valid generation preset.</summary>
/// <param name="Label">UI label.</param>
/// <param name="Options">The request the preset expands to.</param>
public sealed record GenerationPreset(string Label, GenerationOptions Options);

/// <summary>
/// Resolution / sampler presets held on the managed side, all guaranteed to
/// satisfy the engine's validation rules (see docs/PARAMETERS.md). This is the
/// .NET replacement for the former Go <c>/api/presets</c> table.
/// </summary>
public static class QziPresets
{
    /// <summary>Qwen-Image 2.1 presets.</summary>
    public static IReadOnlyList<GenerationPreset> Qwen21 { get; } =
    [
        new("1024x1024 Turbo 8 steps", new QwenImageOptions
        {
            Prompt = "", Width = 1024, Height = 1024, Tier = QwenImageTier.Turbo
        }),
        new("512x512 Turbo 8 steps", new QwenImageOptions
        {
            Prompt = "", Width = 512, Height = 512, Tier = QwenImageTier.Turbo
        }),
        new("1024x1024 Base 40 steps", new QwenImageOptions
        {
            Prompt = "", Width = 1024, Height = 1024, Tier = QwenImageTier.Base,
            Steps = 40, Guidance = 4.0f
        }),
        new("512x512 DMD2 2 steps", new QwenImageOptions
        {
            Prompt = "", Width = 512, Height = 512, Tier = QwenImageTier.Base,
            Steps = 2, Guidance = 1.0f, Lora = "dmd2.safetensors", LoraScale = 1.0f
        }),
        new("1024x1024 DMD2 2 steps", new QwenImageOptions
        {
            Prompt = "", Width = 1024, Height = 1024, Tier = QwenImageTier.Base,
            Steps = 2, Guidance = 1.0f, Lora = "dmd2.safetensors", LoraScale = 1.0f
        })
    ];

    /// <summary>Z-Image presets.</summary>
    public static IReadOnlyList<GenerationPreset> ZImage { get; } =
    [
        new("1024x1024 bf16", new ZImageOptions
        {
            Prompt = "", Width = 1024, Height = 1024, Checkpoint = "bf16"
        }),
        new("1024x1024 Nunchaku int4", new ZImageOptions
        {
            Prompt = "", Width = 1024, Height = 1024, Checkpoint = "nunchaku"
        }),
        new("512x512 bf16", new ZImageOptions
        {
            Prompt = "", Width = 512, Height = 512, Checkpoint = "bf16"
        }),
        new("1536x1536 bf16", new ZImageOptions
        {
            Prompt = "", Width = 1536, Height = 1536, Checkpoint = "bf16"
        })
    ];

    /// <summary>All presets in one list.</summary>
    public static IEnumerable<GenerationPreset> All => Qwen21.Concat(ZImage);

    /// <summary>
    /// Returns a copy of a preset with a real prompt. Presets intentionally
    /// carry an empty prompt; they are templates, not runnable requests.
    /// </summary>
    public static GenerationOptions WithPrompt(GenerationPreset preset, string prompt)
        => preset.Options switch
        {
            QwenImageOptions qwen => qwen with { Prompt = prompt },
            ZImageOptions zimage => zimage with { Prompt = prompt },
            _ => throw new ArgumentOutOfRangeException(nameof(preset))
        };
}
