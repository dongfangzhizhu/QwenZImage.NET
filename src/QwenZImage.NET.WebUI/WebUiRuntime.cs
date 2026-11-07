using System.Collections.Concurrent;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>Configuration read from <c>QwenZImage.config.json</c> plus optional overrides.</summary>
public sealed class WebUiConfig
{
    public string Urls { get; init; } = "http://127.0.0.1:7860";
    public string OutputDirectory { get; init; } = "";

    /// <summary>
    /// Fully resolved options handed to <see cref="QziClient"/>. Model pack
    /// directories come from the config file; the <c>QZI_*</c> environment
    /// variables and the CLI switches below override individual fields.
    /// </summary>
    public QziClientOptions ClientOptions { get; init; } = new();

    /// <summary>Config file actually used, for the health endpoint.</summary>
    public string? ConfigPath => ClientOptions.ConfigPath;

    /// <summary>Primary model pack directory, for the health endpoint.</summary>
    public string ModelDirectory => ClientOptions.Qwen21Directory ?? "";

    public static WebUiConfig FromEnvironment(string contentRoot, string[] args)
    {
        string output = Environment.GetEnvironmentVariable("QZI_WEBUI_OUTPUT") ?? "";
        string urls = Environment.GetEnvironmentVariable("QZI_WEBUI_URLS") ?? "";
        string? model = NullIfBlank(Environment.GetEnvironmentVariable("QZI_MODEL_DIR"));
        string? cache = NullIfBlank(Environment.GetEnvironmentVariable("QZI_MODEL_CACHE"));
        string? native = NullIfBlank(Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY"));
        string? configPath = NullIfBlank(Environment.GetEnvironmentVariable(QziConfig.EnvironmentVariable));

        int? device = null;
        string? deviceText = Environment.GetEnvironmentVariable("QZI_DEVICE_ID");
        if (int.TryParse(deviceText, out int parsedDevice)) device = parsedDevice;

        // Explicit CLI switches win over the environment. ASP.NET Core ignores
        // unknown command-line configuration keys, so these are harmless there.
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], "--config", StringComparison.Ordinal))
                configPath = args[i + 1];
            else if (string.Equals(args[i], "--models", StringComparison.Ordinal))
                model = args[i + 1];
        }

        QziClientOptions baseline = QziClientOptions.FromConfig(configPath);
        QziClientOptions options = new()
        {
            NativeLibraryPath = native ?? baseline.NativeLibraryPath,
            Qwen21Directory = model ?? baseline.Qwen21Directory,
            ZImageDirectory = model ?? baseline.ZImageDirectory,
            CacheDirectory = cache ?? baseline.CacheDirectory,
            Qwen21LoraDirectory = baseline.Qwen21LoraDirectory,
            ZImageNunchakuDirectory = baseline.ZImageNunchakuDirectory,
            DeviceId = device ?? baseline.DeviceId,
            ThreadCount = baseline.ThreadCount,
            ConfigPath = baseline.ConfigPath
        };

        return new WebUiConfig
        {
            Urls = string.IsNullOrWhiteSpace(urls) ? "http://127.0.0.1:7860" : urls,
            OutputDirectory = string.IsNullOrWhiteSpace(output)
                ? Path.Combine(contentRoot, "outputs") : output,
            ClientOptions = options
        };
    }

    private static string? NullIfBlank(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value;
}

/// <summary>
/// Owns the single in-process native engine, the serial job queue and the
/// output gallery. If native initialisation fails the WebUI still starts and
/// serves a clear error in /api/health.
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
            Engine = new WebUiEngine(config.ClientOptions);
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
    public WebUiEngine? Engine { get; }
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
