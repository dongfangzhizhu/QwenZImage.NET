using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>
/// Transport-independent view of the engine behind the manual-test WebUI.
/// Both the in-process native P/Invoke client and the optional remote
/// <c>h3 serve</c> adapter implement it, so the queue, SSE stream and health
/// endpoint do not need to know where the PNG actually came from.
/// </summary>
public interface IWebUiEngine : IQziExecutor, IDisposable
{
    WebUiEngineHealth GetHealth();
    void Unload();
}

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

    /// <summary>True only for the bundled deterministic noise backend.</summary>
    public bool IsReferenceBackend { get; init; }

    /// <summary>Human-readable engine identifier shown in the UI pill.</summary>
    public string Backend { get; init; } = "unknown";
}
