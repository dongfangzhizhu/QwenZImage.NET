using System.Collections.Concurrent;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>Configuration read from environment variables / CLI, all optional.</summary>
public sealed class WebUiConfig
{
    public string Urls { get; init; } = "http://127.0.0.1:7860";
    public string OutputDirectory { get; init; } = "";
    public string ModelDirectory { get; init; } = "";
    public string CacheDirectory { get; init; } = "";
    public string? NativeLibraryPath { get; init; }
    public int DeviceId { get; init; } = -1;

    /// <summary>
    /// Base URL of a real <c>h3 serve</c> instance. When set, the WebUI uses the
    /// HTTP transport instead of the bundled native reference backend, and the
    /// bundled noise-only library is never loaded.
    /// </summary>
    public string? EngineUrl { get; init; }

    /// <summary>Whole-request deadline for the HTTP transport, in minutes.</summary>
    public double EngineTimeoutMinutes { get; init; } = 30;

    /// <summary>Poll interval for the engine's optional /progress endpoint.</summary>
    public int EngineProgressPollMs { get; init; } = 500;

    public static WebUiConfig FromEnvironment(string contentRoot, string[] args)
    {
        string output = Environment.GetEnvironmentVariable("QZI_WEBUI_OUTPUT") ?? "";
        string model = Environment.GetEnvironmentVariable("QZI_MODEL_DIR") ?? "";
        string cache = Environment.GetEnvironmentVariable("QZI_MODEL_CACHE") ?? "";
        string urls = Environment.GetEnvironmentVariable("QZI_WEBUI_URLS") ?? "";
        string? engine = Environment.GetEnvironmentVariable("QZI_ENGINE_URL");

        int device = -1;
        string? deviceText = Environment.GetEnvironmentVariable("QZI_DEVICE_ID");
        if (!string.IsNullOrWhiteSpace(deviceText) && int.TryParse(deviceText, out int parsed))
            device = parsed;

        double timeout = 30;
        string? timeoutText = Environment.GetEnvironmentVariable("QZI_ENGINE_TIMEOUT_MIN");
        if (double.TryParse(timeoutText, out double parsedTimeout) && parsedTimeout > 0)
            timeout = parsedTimeout;

        int pollMs = 500;
        string? pollText = Environment.GetEnvironmentVariable("QZI_ENGINE_PROGRESS_MS");
        if (int.TryParse(pollText, out int parsedPoll) && parsedPoll >= 100)
            pollMs = parsedPoll;

        // `--engine URL` mirrors the original Go WebUI's --engine flag and wins
        // over the environment variable. It is intentionally the only extra CLI
        // switch; ASP.NET Core ignores unknown command-line configuration keys.
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], "--engine", StringComparison.Ordinal))
                engine = args[i + 1];

        return new WebUiConfig
        {
            Urls = string.IsNullOrWhiteSpace(urls) ? "http://127.0.0.1:7860" : urls,
            OutputDirectory = string.IsNullOrWhiteSpace(output)
                ? Path.Combine(contentRoot, "outputs") : output,
            ModelDirectory = string.IsNullOrWhiteSpace(model) ? contentRoot : model,
            CacheDirectory = string.IsNullOrWhiteSpace(cache)
                ? Path.Combine(Path.GetTempPath(), "qzi-text-cache") : cache,
            NativeLibraryPath = Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY"),
            DeviceId = device,
            EngineUrl = string.IsNullOrWhiteSpace(engine) ? null : engine.Trim(),
            EngineTimeoutMinutes = timeout,
            EngineProgressPollMs = pollMs
        };
    }

    /// <summary>Wire-level transport name, surfaced in /api/health.</summary>
    public string Transport => EngineUrl is null ? "native-pinvoke" : "h3-serve-http";
}

/// <summary>
/// Owns the single engine, the serial job queue and the output gallery.
///
/// Two transports are supported:
///  * default — in-process native P/Invoke (<see cref="QziClient"/>), which with
///    the bundled library is the deterministic noise-only reference backend;
///  * <c>QZI_ENGINE_URL</c> — an HTTP adapter for the real C/CUDA <c>h3 serve</c>,
///    giving real Qwen-Image 2.1 / Z-Image output. WebUI-only, not in NuGet.
///
/// If native initialisation fails the WebUI still starts and serves a clear
/// error in /api/health.
/// </summary>
public sealed class WebUiRuntime : IDisposable
{
    private readonly ConcurrentDictionary<long, string> _outputNames = new();
    private readonly ConcurrentDictionary<long, string?> _clients = new();

    public WebUiRuntime(WebUiConfig config)
    {
        Config = config;
        StartedAt = DateTimeOffset.UtcNow;
        Outputs = new OutputStore(config.OutputDirectory);

        try
        {
            Engine = config.EngineUrl is null
                ? new NativeEngineAdapter(config)
                : new H3ServeExecutor(config.EngineUrl,
                                      TimeSpan.FromMinutes(config.EngineTimeoutMinutes),
                                      config.EngineProgressPollMs);

            Queue = new QziGenerationQueue(Engine, async (job, result, token) =>
            {
                string name = await Outputs.SaveAsync(job.Id, result, token);
                _outputNames[job.Id] = name;
            });
        }
        catch (Exception ex)
        {
            InitError = $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    public WebUiConfig Config { get; }
    public OutputStore Outputs { get; }
    public IWebUiEngine? Engine { get; }
    public QziGenerationQueue? Queue { get; }
    public string? InitError { get; }
    public DateTimeOffset StartedAt { get; }

    public string? OutputNameFor(long jobId)
        => _outputNames.TryGetValue(jobId, out string? name) ? name : null;

    public void SetClient(long jobId, string? client)
    {
        if (!string.IsNullOrWhiteSpace(client)) _clients[jobId] = client;
    }

    public string? ClientFor(long jobId)
        => _clients.TryGetValue(jobId, out string? client) ? client : null;

    public void Dispose()
    {
        Queue?.Dispose();
        Engine?.Dispose();
        GC.SuppressFinalize(this);
    }
}
