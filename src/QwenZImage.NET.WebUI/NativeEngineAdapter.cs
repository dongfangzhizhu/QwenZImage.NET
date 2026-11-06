using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>
/// Default transport: the in-process C ABI through <see cref="QziClient"/>.
/// With the library bundled as-is this is the deterministic reference backend,
/// i.e. it produces pseudo-random noise rather than a real image.
/// </summary>
internal sealed class NativeEngineAdapter : IWebUiEngine
{
    private readonly QziClient _client;

    public NativeEngineAdapter(WebUiConfig config)
    {
        _client = new QziClient(new QziClientOptions
        {
            NativeLibraryPath = config.NativeLibraryPath,
            Qwen21Directory = config.ModelDirectory,
            ZImageDirectory = config.ModelDirectory,
            CacheDirectory = config.CacheDirectory,
            DeviceId = config.DeviceId
        });
    }

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
            Backend = h.IsReferenceBackend ? "qzi-reference" : "qzi-native"
        };
    }

    public void Unload() => _client.Unload();

    public void Dispose() => _client.Dispose();
}
