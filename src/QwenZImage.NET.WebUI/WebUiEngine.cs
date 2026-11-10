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
}

/// <summary>
/// Capability gating for additive ABI v1 features. Kept as pure functions so
/// the WebUI can fail fast on edit / ControlNet jobs and the behaviour is
/// unit-testable without constructing a native engine. Text-to-image requests
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
    /// Returns a 400-style error when <paramref name="options"/> needs a
    /// feature the loaded library does not provide, otherwise <c>null</c>.
    /// Editing and ControlNet are additive in ABI v1: an older library still
    /// serves text-to-image but lacks the matching export. Rejecting here means
    /// the operator sees a 400 instead of a queued job failing minutes later
    /// (or silently degrading to ordinary t2i / the noise backend).
    /// </summary>
    public static string? ErrorFor(GenerationOptions options,
                                   bool supportsImageEditing,
                                   bool supportsControlNet,
                                   bool supportsAdvancedEditing)
        => options switch
        {
            // Advanced editing derives from the ordinary edit options, so it
            // must be matched first.
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

    public GenerationResult Generate(GenerationOptions options,
                                     IProgress<GenerationProgress>? progress = null,
                                     CancellationToken cancellationToken = default)
        => options switch
        {
            // Derived ControlNet options must go to the dedicated ControlNet
            // call, not the base Z-Image path (which would ignore the control
            // image). All other requests keep the client's own dispatch.
            ZImageControlOptions control =>
                _client.GenerateZImageControl(control, progress, cancellationToken),
            _ => _client.Generate(options, progress, cancellationToken)
        };

    public WebUiEngineHealth GetHealth()
    {
        EngineHealth h = _client.GetHealth();
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
            SupportsAdvancedEditing = SupportsAdvancedEditing
        };
    }

    public void Unload() => _client.Unload();

    public void Dispose() => _client.Dispose();
}
