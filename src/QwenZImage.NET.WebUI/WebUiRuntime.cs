using System.Collections.Concurrent;
using QwenZImage.NET;
using QwenZImage.NET.WebUI.ModelDownloads;
using QwenZImage.NET.WebUI.Preprocessing;

namespace QwenZImage.NET.WebUI;

/// <summary>Configuration read from <c>QwenZImage.config.json</c> plus optional overrides.</summary>
public sealed class WebUiConfig
{
    public string Urls { get; init; } = "http://127.0.0.1:7860";
    public string OutputDirectory { get; init; } = "";

    /// <summary>
    /// Directory the server was launched from. Used to walk upwards when
    /// looking for <c>preprocess_dir</c> in QwenZImage.config.json.
    /// </summary>
    public string LaunchDirectory { get; init; } = "";

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

    /// <summary>How the native backend was selected, surfaced by /api/health.</summary>
    public string NativeLibrarySource { get; init; } = "bundled reference runtime";

    /// <summary>
    /// True when <c>--models</c> / <c>QZI_MODEL_DIR</c> replaced the config
    /// file's model directories. The config writer warns about this, because
    /// editing the file alone would then have no effect.
    /// </summary>
    public bool ModelOverrideActive { get; init; }

    /// <summary>
    /// When true, the app refuses to serve generation jobs with the
    /// deterministic noise backend instead of silently returning meaningless
    /// images. Set by the one-click CUDA launcher scripts.
    /// </summary>
    public bool RequireRealBackend { get; init; }

    public static WebUiConfig FromEnvironment(string contentRoot, string[] args)
    {
        string output = Environment.GetEnvironmentVariable("QZI_WEBUI_OUTPUT") ?? "";
        string urls = Environment.GetEnvironmentVariable("QZI_WEBUI_URLS") ?? "";
        string? model = NullIfBlank(Environment.GetEnvironmentVariable("QZI_MODEL_DIR"));
        string? cache = NullIfBlank(Environment.GetEnvironmentVariable("QZI_MODEL_CACHE"));
        string? native = NullIfBlank(Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY"));
        string? nativePreference = NullIfBlank(Environment.GetEnvironmentVariable("QZI_WEBUI_NATIVE"));
        bool requireReal = string.Equals(
            NullIfBlank(Environment.GetEnvironmentVariable("QZI_REQUIRE_REAL_BACKEND")),
            "1", StringComparison.Ordinal);
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

        // Resolve the native backend. Priority:
        //   1. QZI_NATIVE_LIBRARY (explicit; NativeLibraryLoader fails fast)
        //   2. QZI_WEBUI_NATIVE=<path>, or "reference" to force the stub
        //   3. a locally built -cuda artefact (development checkout only)
        //   4. the bundled reference runtime (CI / NuGet / GPU-free hosts)
        // Step 3 is what makes `dotnet run` produce real images on a machine
        // that has built the CUDA engine, regardless of which flavour was most
        // recently staged into artifacts/runtimes/<rid>/native for packaging.
        string nativeSource;
        if (!string.IsNullOrWhiteSpace(native))
        {
            nativeSource = "QZI_NATIVE_LIBRARY";
        }
        else if (string.Equals(nativePreference, "reference", StringComparison.OrdinalIgnoreCase))
        {
            nativeSource = "bundled reference runtime (QZI_WEBUI_NATIVE=reference)";
        }
        else if (!string.IsNullOrWhiteSpace(nativePreference) &&
                 !string.Equals(nativePreference, "auto", StringComparison.OrdinalIgnoreCase))
        {
            native = nativePreference;
            nativeSource = "QZI_WEBUI_NATIVE";
        }
        else
        {
            native = FindCudaNative(contentRoot);
            nativeSource = native is null
                ? "bundled reference runtime (no local CUDA build found)"
                : "local CUDA build (auto-detected)";
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
            ZImageControlDirectory = baseline.ZImageControlDirectory,
            DeviceId = device ?? baseline.DeviceId,
            ThreadCount = baseline.ThreadCount,
            ConfigPath = baseline.ConfigPath
        };

        return new WebUiConfig
        {
            Urls = string.IsNullOrWhiteSpace(urls) ? "http://127.0.0.1:7860" : urls,
            OutputDirectory = string.IsNullOrWhiteSpace(output)
                ? Path.Combine(contentRoot, "outputs") : output,
            LaunchDirectory = Path.GetFullPath(contentRoot),
            ClientOptions = options,
            NativeLibrarySource = nativeSource,
            ModelOverrideActive = model is not null,
            RequireRealBackend = requireReal
        };
    }

    /// <summary>
    /// Looks for a real `-cuda` build in the development checkout. The NuGet
    /// package and CI never contain such a directory, so those hosts keep the
    /// bundled reference backend unchanged. Searching up from the WebUI's
    /// content root (rather than AppContext.BaseDirectory) keeps unit tests
    /// running from a temp directory out of this path entirely.
    /// </summary>
    private static string? FindCudaNative(string contentRoot)
    {
        string name = OperatingSystem.IsWindows() ? "QwenZImageNative.dll"
                    : OperatingSystem.IsMacOS() ? "libQwenZImageNative.dylib"
                    : "libQwenZImageNative.so";

        DirectoryInfo? dir = new(Path.GetFullPath(contentRoot));
        for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            string tree = Path.Combine(dir.FullName, "artifacts", "native");
            if (!Directory.Exists(tree)) continue;
            foreach (string platform in Directory.EnumerateDirectories(tree))
            {
                if (!Path.GetFileName(platform).EndsWith("-cuda",
                        StringComparison.OrdinalIgnoreCase)) continue;
                string candidate = Path.Combine(platform, name);
                if (File.Exists(candidate)) return Path.GetFullPath(candidate);
            }
        }
        return null;
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
        Preprocessor = new ControlImagePreprocessor(
            PreprocessAssets.Discover(config.ConfigPath, config.LaunchDirectory));

        // Model downloads live entirely in the host layer. The core library /
        // C ABI keeps loading weights from disk exactly as before; the WebUI
        // merely fetches WanGP's Hugging Face files into those directories.
        Downloads = new ModelDownloadService();
        Models = new ModelCatalog(config, Preprocessor.Assets.Root, Downloads,
            Environment.GetEnvironmentVariable(ModelManifest.MirrorEnvironmentVariable));

        try
        {
            WebUiEngine engine = new(config.ClientOptions);
            if (config.RequireRealBackend && engine.GetHealth().IsReferenceBackend)
            {
                // Never let a caller who explicitly asked for real inference
                // receive deterministic noise. /api/health reports ok=false and
                // /api/jobs returns 503 until the backend is fixed.
                engine.Dispose();
                InitError =
                    "QZI_REQUIRE_REAL_BACKEND=1 but the deterministic reference " +
                    "backend is loaded: it emits noise, not model output. Point " +
                    "QZI_NATIVE_LIBRARY at a -cuda build (or run " +
                    "scripts/start-webui-cuda.ps1 / .sh).";
            }
            else
            {
                Engine = engine;
                Queue = new QziGenerationQueue(Engine, async (job, result, token) =>
                {
                    string name = await Outputs.SaveAsync(job.Id, result, token);
                    _outputNames[job.Id] = name;
                });
            }
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

    /// <summary>
    /// Application-layer ControlNet guide preprocessing. Unlike the native
    /// engine this is always available, even when model weights or a CUDA
    /// build are missing, so uploads can still be prepared and previewed.
    /// </summary>
    public ControlImagePreprocessor Preprocessor { get; }

    /// <summary>
    /// Serial, resumable weight downloader (host layer only). Works even when
    /// the native engine failed to initialise or no CUDA build is present.
    /// </summary>
    public ModelDownloadService Downloads { get; }

    /// <summary>Catalogue bound to the effective config paths.</summary>
    public ModelCatalog Models { get; }

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
        Downloads.Dispose();
        Preprocessor.Dispose();
        GC.SuppressFinalize(this);
    }
}
