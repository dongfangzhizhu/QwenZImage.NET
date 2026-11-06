using System.Collections.Concurrent;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>Configuration read from environment variables, all optional.</summary>
public sealed class WebUiConfig
{
    public string Urls { get; init; } = "http://127.0.0.1:7860";
    public string OutputDirectory { get; init; } = "";
    public string ModelDirectory { get; init; } = "";
    public string CacheDirectory { get; init; } = "";
    public string? NativeLibraryPath { get; init; }
    public int DeviceId { get; init; } = -1;

    public static WebUiConfig FromEnvironment(string contentRoot)
    {
        string output = Environment.GetEnvironmentVariable("QZI_WEBUI_OUTPUT") ?? "";
        string model = Environment.GetEnvironmentVariable("QZI_MODEL_DIR") ?? "";
        string cache = Environment.GetEnvironmentVariable("QZI_MODEL_CACHE") ?? "";
        string urls = Environment.GetEnvironmentVariable("QZI_WEBUI_URLS") ?? "";
        int device = -1;
        string? deviceText = Environment.GetEnvironmentVariable("QZI_DEVICE_ID");
        if (!string.IsNullOrWhiteSpace(deviceText) && int.TryParse(deviceText, out int parsed))
            device = parsed;

        return new WebUiConfig
        {
            Urls = string.IsNullOrWhiteSpace(urls) ? "http://127.0.0.1:7860" : urls,
            OutputDirectory = string.IsNullOrWhiteSpace(output)
                ? Path.Combine(contentRoot, "outputs") : output,
            ModelDirectory = string.IsNullOrWhiteSpace(model) ? contentRoot : model,
            CacheDirectory = string.IsNullOrWhiteSpace(cache)
                ? Path.Combine(Path.GetTempPath(), "qzi-text-cache") : cache,
            NativeLibraryPath = Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY"),
            DeviceId = device
        };
    }
}

/// <summary>
/// Owns the single native engine, the serial job queue and the output gallery.
/// If the native library cannot be loaded the WebUI still starts and serves a
/// clear error in <c>/api/health</c> — useful when manually probing a machine
/// that has not built the reference library yet.
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
            Client = new QziClient(new QziClientOptions
            {
                NativeLibraryPath = config.NativeLibraryPath,
                Qwen21Directory = config.ModelDirectory,
                ZImageDirectory = config.ModelDirectory,
                CacheDirectory = config.CacheDirectory,
                DeviceId = config.DeviceId
            });

            Queue = new QziGenerationQueue(Client, async (job, result, token) =>
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
    public QziClient? Client { get; }
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
        Client?.Dispose();
        GC.SuppressFinalize(this);
    }
}
