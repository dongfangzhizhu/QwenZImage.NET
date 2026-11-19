using System.Text.Json.Serialization;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>Normalised health snapshot for the WebUI's <c>/api/health</c>.</summary>
public sealed record WebUiEngineHealth
{
    public string ResidentModel { get; init; } = "none";
    public bool Busy { get; init; }
    public int QueueDepth { get; init; }
    public int DeviceId { get; init; }
    public long FreeVramMib { get; init; }
    public long TotalVramMib { get; init; }
    public ulong Served { get; init; }
    public ulong Rejected { get; init; }
    public double UptimeSeconds { get; init; }

    /// <summary>True only for the deterministic noise-only reference build.</summary>
    public bool IsReferenceBackend { get; init; }

    /// <summary>Human-readable engine identifier shown in the UI pill.</summary>
    public string Backend { get; init; } = "unknown";

    /// <summary>
    /// Whether the loaded native library exports the additive ABI v1 Z-Image
    /// Turbo Fun ControlNet Union surface. Mirrors the top-level
    /// <c>supports_controlnet</c> flag in <c>/api/health</c>. The explicit name
    /// keeps the record's serialised form identical to the endpoint flag (the
    /// default snake_case policy would split "ControlNet" into control_net).
    /// </summary>
    [JsonPropertyName("supports_controlnet")]
    public bool SupportsControlNet { get; init; }

    /// <summary>
    /// Whether the loaded native library exports the third additive ABI v1
    /// increment (advanced Qwen-Image 2.1 editing). Mirrors the top-level
    /// <c>supports_advanced_editing</c> flag in <c>/api/health</c>.
    /// </summary>
    [JsonPropertyName("supports_advanced_editing")]
    public bool SupportsAdvancedEditing { get; init; }

    /// <summary>
    /// Whether the loaded native library exports the fourth additive ABI v1
    /// increment (Z-Image Union 2.x inpaint). Mirrors the top-level
    /// <c>supports_inpaint</c> flag in <c>/api/health</c>.
    /// </summary>
    [JsonPropertyName("supports_inpaint")]
    public bool SupportsInpaint { get; init; }

    /// <summary>
    /// Whether the loaded native library exports the fifth additive ABI v1
    /// increment (Qwen-Image 2.1 identity references / Wan2GP image_refs).
    /// Mirrors the top-level <c>supports_identity_reference</c> flag in
    /// <c>/api/health</c>.
    /// </summary>
    [JsonPropertyName("supports_identity_reference")]
    public bool SupportsIdentityReference { get; init; }

    /// <summary>
    /// Whether the loaded native library exports the sixth additive ABI v1
    /// increment (low-VRAM / memory profile). Mirrors the top-level
    /// <c>supports_low_vram</c> flag in <c>/api/health</c>.
    /// </summary>
    [JsonPropertyName("supports_low_vram")]
    public bool SupportsLowVram { get; init; }

    /// <summary>Active memory profile (auto/high/balanced/low/lowest) or null.</summary>
    public string? MemoryProfile { get; init; }

    /// <summary>Resolved pool release threshold in MiB, or null when unsupported.</summary>
    public int? PoolReleaseMib { get; init; }

    /// <summary>Resolved reference VAE encode cap in px, or null when unsupported.</summary>
    public int? ReferenceEncodeCap { get; init; }
}

/// <summary>
/// Capability gating for additive ABI v1 features. Kept as pure functions so
/// the WebUI can fail fast on edit / ControlNet / inpaint jobs and the behaviour
/// is unit-testable without constructing a native engine. Text-to-image requests
/// never need an optional feature.
/// </summary>
public static class WebUiCapabilities
{
    /// <summary>400 message for an image-edit job on a library without editing.</summary>
    public const string EditingUnavailable =
        "The loaded native library does not implement Qwen-Image 2.1 image " +
        "editing. Rebuild QwenZImageNative from the matching source tree; " +
        "text-to-image generation is unaffected.";

    /// <summary>400 message for a ControlNet job on a library without ControlNet.</summary>
    public const string ControlNetUnavailable =
        "The loaded native library does not implement Z-Image Turbo Fun " +
        "ControlNet Union. Rebuild QwenZImageNative from the matching source tree; " +
        "text-to-image generation is unaffected.";

    /// <summary>400 message for an advanced-edit job on an older library.</summary>
    public const string AdvancedEditingUnavailable =
        "The loaded native library does not implement advanced Qwen-Image 2.1 " +
        "editing (outpainting / LanPaint / RGBA). Rebuild QwenZImageNative from " +
        "the matching source tree; ordinary image editing and text-to-image " +
        "generation are unaffected.";

    /// <summary>400 message for an inpaint job on a library without inpaint.</summary>
    public const string InpaintUnavailable =
        "The loaded native library does not implement Z-Image Union 2.x " +
        "inpaint. Rebuild QwenZImageNative from the matching source tree; " +
        "text-to-image generation is unaffected.";

    /// <summary>400 message for an identity-reference job on an older library.</summary>
    public const string IdentityReferenceUnavailable =
        "The loaded native library does not implement Qwen-Image 2.1 identity " +
        "references (image_refs). Rebuild QwenZImageNative from the matching " +
        "source tree; ordinary editing and text-to-image are unaffected.";

    /// <summary>
    /// Two-argument overload kept for callers that predate the third additive
    /// increment; it assumes the advanced capability follows the basic one.
    /// </summary>
    public static string? ErrorFor(GenerationOptions options,
                                   bool supportsImageEditing,
                                   bool supportsControlNet)
        => ErrorFor(options, supportsImageEditing, supportsControlNet,
                    supportsImageEditing);

    /// <summary>
    /// Four-argument overload kept for callers that predate the fourth additive
    /// increment; it assumes the inpaint capability follows ControlNet.
    /// </summary>
    public static string? ErrorFor(GenerationOptions options,
                                   bool supportsImageEditing,
                                   bool supportsControlNet,
                                   bool supportsAdvancedEditing)
        => ErrorFor(options, supportsImageEditing, supportsControlNet,
                    supportsAdvancedEditing, supportsControlNet, supportsImageEditing);

    /// <summary>
    /// Five-argument overload kept for callers that predate the fifth additive
    /// increment; it assumes identity references follow image editing.
    /// </summary>
    public static string? ErrorFor(GenerationOptions options,
                                   bool supportsImageEditing,
                                   bool supportsControlNet,
                                   bool supportsAdvancedEditing,
                                   bool supportsInpaint)
        => ErrorFor(options, supportsImageEditing, supportsControlNet,
                    supportsAdvancedEditing, supportsInpaint, supportsImageEditing);

    /// <summary>
    /// Returns a 400-style error when <paramref name="options"/> needs a
    /// feature the loaded library does not provide, otherwise <c>null</c>.
    /// Editing, ControlNet, advanced editing, inpaint and identity references
    /// are additive in ABI v1: an older library still serves text-to-image but
    /// lacks the matching export. Rejecting here means the operator sees a 400
    /// instead of a queued job failing minutes later (or silently degrading to
    /// text-to-image / the noise backend). The HTTP endpoint always supplies
    /// the real capability probe results.
    /// </summary>
    public static string? ErrorFor(GenerationOptions options,
                                   bool supportsImageEditing,
                                   bool supportsControlNet,
                                   bool supportsAdvancedEditing,
                                   bool supportsInpaint,
                                   bool supportsIdentityReference)
        => options switch
        {
            // Inpaint derives from ControlNet, so it must be matched first.
            ZImageInpaintOptions when !supportsInpaint => InpaintUnavailable,
            Qwen21IdentityRefOptions when !supportsIdentityReference =>
                IdentityReferenceUnavailable,
            Qwen21EditExOptions when !supportsAdvancedEditing => AdvancedEditingUnavailable,
            Qwen21EditOptions when !supportsImageEditing => EditingUnavailable,
            ZImageControlOptions when !supportsControlNet => ControlNetUnavailable,
            _ => null
        };
}

/// <summary>
/// The one engine behind the manual-test WebUI: the in-process C ABI through
/// <see cref="QziClient"/>, configured by <c>QwenZImage.config.json</c>. There
/// is no HTTP hop, no secondary service and no in-process transport switch:
/// the managed assembly P/Invokes QwenZImageNative directly.
/// </summary>
public sealed class WebUiEngine : IQziExecutor, IDisposable
{
    private readonly QziClient _client;

    public WebUiEngine(QziClientOptions options) => _client = new QziClient(options);

    /// <summary>Native library actually loaded, for /api/health diagnostics.</summary>
    public string? NativeLibraryPath => _client.NativeLibraryPath;

    /// <summary>
    /// Whether the loaded native library exports the additive ABI v1 image-edit
    /// surface. Exposed so the WebUI can fail fast on edit jobs (and disable the
    /// edit mode) instead of letting a queued job fail minutes later.
    /// </summary>
    public bool SupportsImageEditing => _client.SupportsImageEditing;

    /// <summary>
    /// Whether the loaded native library exports the additive ABI v1 Z-Image
    /// Turbo Fun ControlNet Union surface. Exposed so the WebUI can fail fast
    /// on ControlNet jobs (and disable the ControlNet mode) instead of silently
    /// degrading to text-to-image or the noise backend.
    /// </summary>
    public bool SupportsControlNet => _client.SupportsControlNet;

    /// <summary>
    /// Whether the loaded native library exports the third additive ABI v1
    /// increment (advanced editing). Exposed so the WebUI can reject advanced
    /// jobs with a 400 instead of letting them fail minutes later.
    /// </summary>
    public bool SupportsAdvancedEditing => _client.SupportsAdvancedEditing;

    /// <summary>
    /// Whether the loaded native library exports the fourth additive ABI v1
    /// increment (Z-Image Union 2.x inpaint). Exposed so the WebUI can reject
    /// inpaint jobs with a 400 instead of letting them fail minutes later.
    /// </summary>
    public bool SupportsInpaint => _client.SupportsInpaint;

    /// <summary>
    /// Whether the loaded native library exports the fifth additive ABI v1
    /// increment (Qwen-Image identity references / Wan2GP image_refs).
    /// </summary>
    public bool SupportsIdentityReference => _client.SupportsIdentityReference;

    /// <summary>
    /// Whether the loaded native library exports the low-VRAM / memory profile
    /// control surface. Exposed so the WebUI can offer the profile selector only
    /// when the engine can honour it.
    /// </summary>
    public bool SupportsLowVram => _client.SupportsLowVram;

    /// <summary>Reads the resolved low-VRAM budget and live device VRAM.</summary>
    public QziMemoryPlan GetMemoryPlan() => _client.GetMemoryPlan();

    /// <summary>Sets the low-VRAM memory profile for subsequent generations.</summary>
    public void SetMemoryProfile(QziMemoryProfile profile)
        => _client.SetMemoryProfile(profile);

    public GenerationResult Generate(GenerationOptions options,
                                     IProgress<GenerationProgress>? progress = null,
                                     CancellationToken cancellationToken = default)
        => options switch
        {
            // Derived options must go to their dedicated calls, not the base
            // Z-Image path (which would ignore the control image / mask).
            // Inpaint derives from ControlNet, so it is matched first.
            ZImageInpaintOptions inpaint =>
                _client.GenerateZImageInpaint(inpaint, progress, cancellationToken),
            ZImageControlOptions control =>
                _client.GenerateZImageControl(control, progress, cancellationToken),
            Qwen21IdentityRefOptions identity =>
                _client.GenerateQwen21IdentityRef(identity, progress, cancellationToken),
            _ => _client.Generate(options, progress, cancellationToken)
        };

    public WebUiEngineHealth GetHealth()
    {
        EngineHealth h = _client.GetHealth();
        string? memoryProfile = null;
        int? poolReleaseMib = null;
        int? referenceEncodeCap = null;
        if (SupportsLowVram)
        {
            try
            {
                QziMemoryPlan plan = _client.GetMemoryPlan();
                memoryProfile = plan.Profile.ToString().ToLowerInvariant();
                poolReleaseMib = plan.PoolReleaseMib;
                referenceEncodeCap = plan.ReferenceEncodeCap;
            }
            catch (Exception)
            {
                // Health must never fail because an optional probe did.
            }
        }
        return new WebUiEngineHealth
        {
            ResidentModel = h.ResidentModel.ToString().ToLowerInvariant(),
            Busy = h.Busy,
            QueueDepth = h.QueueDepth,
            DeviceId = h.DeviceId,
            FreeVramMib = h.FreeVramMib,
            TotalVramMib = h.TotalVramMib,
            Served = h.Served,
            Rejected = h.Rejected,
            UptimeSeconds = h.Uptime.TotalSeconds,
            IsReferenceBackend = h.IsReferenceBackend,
            Backend = h.IsReferenceBackend ? "reference (noise)" : "QwenZImageNative",
            SupportsControlNet = SupportsControlNet,
            SupportsAdvancedEditing = SupportsAdvancedEditing,
            SupportsInpaint = SupportsInpaint,
            SupportsIdentityReference = SupportsIdentityReference,
            SupportsLowVram = SupportsLowVram,
            MemoryProfile = memoryProfile,
            PoolReleaseMib = poolReleaseMib,
            ReferenceEncodeCap = referenceEncodeCap
        };
    }

    public void Unload() => _client.Unload();

    public void Dispose() => _client.Dispose();
}
