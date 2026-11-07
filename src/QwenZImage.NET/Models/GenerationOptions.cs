namespace QwenZImage.NET;

/// <summary>Qwen-Image 2.1 sampling tiers.</summary>
public enum QwenImageTier
{
    /// <summary>Distilled 8-step, guidance 1.0, no CFG / negative prompt.</summary>
    Turbo = 0,
    /// <summary>Full sampler, 1..50 steps, guidance 1..20, optional negative prompt.</summary>
    Base = 1
}

/// <summary>Common fields of every text-to-image request.</summary>
public abstract record GenerationOptions
{
    /// <summary>Non-empty UTF-8 prompt, at most 16384 bytes.</summary>
    public required string Prompt { get; init; }

    /// <summary>Output width. Qwen21: multiple of 32; ZImage: multiple of 16.</summary>
    public int Width { get; init; } = 1024;

    /// <summary>Output height. Qwen21: multiple of 32; ZImage: multiple of 16.</summary>
    public int Height { get; init; } = 1024;

    /// <summary>Deterministic noise seed.</summary>
    public ulong Seed { get; init; } = 42;

    /// <summary>Which native generator this request targets.</summary>
    public abstract GenerationModel Model { get; }
}

/// <summary>
/// Qwen-Image 2.1 request. The turbo tier fixes steps to 8 and guidance to 1.0;
/// setting either to another value is rejected locally rather than being
/// silently ignored by the sampler.
/// </summary>
public sealed record QwenImageOptions : GenerationOptions
{
    /// <summary>Sampling tier. Defaults to <see cref="QwenImageTier.Turbo"/>.</summary>
    public QwenImageTier Tier { get; init; } = QwenImageTier.Turbo;

    /// <summary>Turbo: 8 (or null). Base: 1..50 (null = 40).</summary>
    public int? Steps { get; init; }

    /// <summary>Turbo: 1.0 (or null). Base: 1..20 (null = 4.0).</summary>
    public float? Guidance { get; init; }

    /// <summary>Base tier only. Turbo has no CFG branch.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>
    /// Bare adapter file name (≤ 255 UTF-8 bytes) resolved inside the engine's
    /// <c>qwen21_lora_dir</c>. Works on both tiers; changing the adapter or its
    /// scale forces a DiT reload. Forbidden: path separators, ':', '..' and
    /// control characters.
    /// </summary>
    public string? Lora { get; init; }

    /// <summary>0.05..4, defaults to 1.0. Requires <see cref="Lora"/>.</summary>
    public float? LoraScale { get; init; }

    /// <inheritdoc/>
    public override GenerationModel Model => GenerationModel.Qwen21;

    /// <summary>The step count actually sent to the engine.</summary>
    public int EffectiveSteps => Steps ?? (Tier == QwenImageTier.Turbo ? 8 : 40);

    /// <summary>The guidance scale actually sent to the engine.</summary>
    public float EffectiveGuidance => Guidance ?? (Tier == QwenImageTier.Turbo ? 1.0f : 4.0f);

    /// <summary>
    /// The effective LoRA strength (1.0 when <see cref="LoraScale"/> is null).
    /// Without a <see cref="Lora"/> the client sends the ABI sentinel 0.0f
    /// instead, which the native layer interprets as this same default.
    /// </summary>
    public float EffectiveLoraScale => LoraScale ?? 1.0f;

    /// <summary>The tier name used by the C ABI.</summary>
    public string TierName => Tier == QwenImageTier.Turbo ? "turbo" : "base";
}

/// <summary>
/// Z-Image (Turbo / Nunchaku) request. Turbo runs a fixed 8-step schedule with
/// no CFG branch, so there are no steps, guidance or negative-prompt fields.
/// </summary>
public sealed record ZImageOptions : GenerationOptions
{
    /// <summary>
    /// <c>null</c> or <c>"bf16"</c> (default), <c>"nunchaku"</c>, or a pack id
    /// such as <c>"nunchaku-r256"</c>. <c>svdq-fp4</c> packs are not supported
    /// by the int4-only engine.
    /// </summary>
    public string? Checkpoint { get; init; } = "bf16";

    /// <inheritdoc/>
    public override GenerationModel Model => GenerationModel.ZImage;
}
