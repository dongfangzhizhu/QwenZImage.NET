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
public record QwenImageOptions : GenerationOptions
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
/// Qwen-Image 2.1 image editing (img2img / masked inpainting) on the SAME 7B
/// weights as <see cref="QwenImageOptions"/>. The reference is VAE-encoded and
/// blended into the noise at WanGP's denoising strength; an optional mask pins
/// the unmasked latent cells to the reference trajectory.
/// </summary>
public record Qwen21EditOptions : QwenImageOptions
{
    /// <summary>Required planar RGB8 reference image.</summary>
    public required QziRasterImage Reference { get; init; }

    /// <summary>
    /// Optional one-channel mask at the reference's dimensions. White (255)
    /// pixels are regenerated from the prompt, black (0) pixels keep the
    /// reference. Resized to the output canvas with the same cover fit.
    /// </summary>
    public QziRasterImage? Mask { get; init; }

    /// <summary>
    /// WanGP's denoising strength, 0.05..1.0. 1.0 ignores the reference
    /// (equivalent to text-to-image); smaller values stay closer to it.
    /// <c>null</c> uses the engine default 0.75.
    /// </summary>
    public float? Strength { get; init; }

    /// <summary>
    /// Fraction of the remaining schedule during which unmasked cells are
    /// restored, 0.05..1.0. Requires <see cref="Mask"/>. <c>null</c> uses 1.0
    /// when a mask is present, otherwise 0 (no masking).
    /// </summary>
    public float? MaskingStrength { get; init; }

    /// <summary>The strength actually sent to the engine (never the 0 sentinel).</summary>
    public float EffectiveStrength => Strength ?? 0.75f;

    /// <summary>The masking strength actually sent (0 without a mask).</summary>
    public float EffectiveMaskingStrength =>
        Mask is null ? 0.0f : (MaskingStrength ?? 1.0f);
}

/// <summary>
/// WanGP <c>qwen21_handler</c> model modes. Mode 0 is ordinary masked
/// denoising; the four nonzero values select training-free LanPaint with
/// NSteps = 2 / 5 / 10 / 15 inner Langevin iterations before each sampler
/// step. The numeric values are the WanGP combo-box indices, NOT the iteration
/// counts, and are passed through the C ABI unchanged.
/// </summary>
public enum Qwen21EditMode : uint
{
    /// <summary>Ordinary masked denoising (WanGP "Masked Denoising").</summary>
    MaskedDenoising = 0,

    /// <summary>LanPaint, 2 inner iterations (WanGP mode 2).</summary>
    LanPaint2 = 2,

    /// <summary>LanPaint, 5 inner iterations (WanGP mode 3).</summary>
    LanPaint5 = 3,

    /// <summary>LanPaint, 10 inner iterations (WanGP mode 4).</summary>
    LanPaint10 = 4,

    /// <summary>LanPaint, 15 inner iterations (WanGP mode 5).</summary>
    LanPaint15 = 5
}

/// <summary>
/// Advanced Qwen-Image 2.1 editing, additive over <see cref="Qwen21EditOptions"/>
/// and using the SAME 7B weights (no extra checkpoint). Three WanGP features
/// are exposed:
/// <list type="number">
///   <item>Red Canvas outpainting: the four margins extend the output canvas;
///   the reference is cover-fitted into the inner source rectangle and the
///   red margins are fully regenerated.</item>
///   <item>LanPaint model modes (see <see cref="Qwen21EditMode"/>), which force
///   denoising / masking strengths to 1.0 and require a mask.</item>
///   <item>RGBA output, keeping the VAE decoder's generated 4th channel.</item>
/// </list>
/// </summary>
/// <remarks>
/// <c>Width</c> and <c>Height</c> describe the FINAL canvas
/// including the margins. Every margin must be a multiple of 32
/// (<c>RequestValidator.OutpaintAlignment</c>) and the inner source rectangle
/// must remain at least 32x32. Outpainting and LanPaint are mutually
/// exclusive, matching WanGP.
/// </remarks>
public sealed record Qwen21EditExOptions : Qwen21EditOptions
{
    /// <summary>Output-canvas pixels added above the source rectangle.</summary>
    public uint MarginTop { get; init; }

    /// <summary>Output-canvas pixels added below the source rectangle.</summary>
    public uint MarginBottom { get; init; }

    /// <summary>Output-canvas pixels added left of the source rectangle.</summary>
    public uint MarginLeft { get; init; }

    /// <summary>Output-canvas pixels added right of the source rectangle.</summary>
    public uint MarginRight { get; init; }

    /// <summary>Masked denoising or one of the LanPaint modes.</summary>
    public Qwen21EditMode ModelMode { get; init; } = Qwen21EditMode.MaskedDenoising;

    /// <summary>
    /// <see langword="true"/> writes an RGBA PNG (PNG colour type 6) using the
    /// VAE decoder's generated alpha; <see langword="false"/> returns RGB.
    /// </summary>
    public bool Rgba { get; init; }

    /// <summary>Reserved for the C ABI; must be 0.</summary>
    public uint Flags { get; init; }

    /// <summary>True when at least one outpainting margin is nonzero.</summary>
    public bool HasOutpainting =>
        (MarginTop | MarginBottom | MarginLeft | MarginRight) != 0;

    /// <summary>True when a LanPaint model mode is selected.</summary>
    public bool IsLanPaint => ModelMode != Qwen21EditMode.MaskedDenoising;

    /// <summary>The numeric WanGP model mode actually sent over the C ABI.</summary>
    public uint EffectiveModelMode => (uint)ModelMode;
}

/// <summary>
/// Z-Image (Turbo / Nunchaku) request. Turbo runs a fixed 8-step schedule with
/// no CFG branch, so there are no steps, guidance or negative-prompt fields.
/// </summary>
public record ZImageOptions : GenerationOptions
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

/// <summary>
/// Z-Image Turbo Fun ControlNet Union request (WanGP z_image_control2_1).
/// Reuses the ordinary <see cref="ZImageOptions"/> fields (prompt, checkpoint,
/// width, height, seed) and adds a planar RGB control image plus the Union
/// control metadata. The control image is descriptive metadata only: this
/// wrapper does NOT run an edge / pose / depth estimator.
/// </summary>
/// <remarks>
/// The control image is VAE-encoded to a 16-channel 1/8 latent and combined
/// with the DiT's control context upstream. <see cref="ControlType"/> is
/// normalised to a lowercase canonical value before it is sent across the ABI.
/// </remarks>
public sealed record ZImageControlOptions : ZImageOptions
{
    /// <summary>Canonical (lowercase) control types accepted by the engine.</summary>
    private static readonly HashSet<string> CanonicalControlTypes = new(StringComparer.Ordinal)
    {
        "auto", "pose", "canny", "depth", "scribble", "raw"
    };

    /// <summary>Required planar RGB8 control image (<c>[3][cw*ch]</c>).</summary>
    public required QziRasterImage ControlImage { get; init; }

    /// <summary>
    /// Descriptive control type: <c>null</c> or one of <c>auto</c>, <c>pose</c>,
    /// <c>canny</c>, <c>depth</c>, <c>scribble</c>, <c>raw</c> (case-insensitive).
    /// <c>raw</c> means the caller already supplies a Z-Image-format control image.
    /// </summary>
    public string? ControlType { get; init; }

    /// <summary>
    /// WanGP's control context scale. <c>0</c> is the ABI sentinel for the engine
    /// default (0.65); any other value must be finite and lie in 0.05..2.0.
    /// </summary>
    public float ControlScale { get; init; }

    /// <summary>Reserved for the C ABI; must be 0.</summary>
    public uint Flags { get; init; }

    /// <summary>
    /// The lowercase canonical control type actually sent to the engine, or
    /// <c>null</c> when <see cref="ControlType"/> is absent. Validation rejects
    /// unknown values before the request is built, so the value is always one
    /// of <see cref="CanonicalControlTypes"/> on the generation path.
    /// </summary>
    public string? EffectiveControlType => NormalizeControlType(ControlType);

    /// <summary>
    /// Normalises a control type to its lowercase canonical form. Returns
    /// <c>null</c> for a null / blank value; an unknown non-blank value is
    /// returned lowercased so that <see cref="IsValidControlType"/> can reject it.
    /// </summary>
    internal static string? NormalizeControlType(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim().ToLowerInvariant();

    /// <summary>
    /// True when a <see cref="NormalizeControlType"/> result is accepted by the
    /// C ABI (<c>null</c> or a canonical control type).
    /// </summary>
    internal static bool IsValidControlType(string? normalized)
        => normalized is null || CanonicalControlTypes.Contains(normalized);
}
