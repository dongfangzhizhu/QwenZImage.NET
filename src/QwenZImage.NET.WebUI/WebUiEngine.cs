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

    public GenerationResult Generate(GenerationOptions options,
                                     IProgress<GenerationProgress>? progress = null,
                                     CancellationToken cancellationToken = default)
        => _client.Generate(options, progress, cancellationToken);

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
            Backend = h.IsReferenceBackend ? "reference (noise)" : "QwenZImageNative"
        };
    }

    public void Unload() => _client.Unload();

    public void Dispose() => _client.Dispose();
}
