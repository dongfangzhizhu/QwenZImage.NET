using System.Buffers;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>
/// Optional WebUI-only transport that talks to the real C/CUDA engine through
/// its <c>h3 serve</c> JSON/HTTP API (documented in h3-spark.c/docs/PROTOCOL.md).
///
/// This is deliberately NOT part of the NuGet library: <c>QwenZImage.NET</c>
/// itself stays a pure Native P/Invoke wrapper. The manual-test WebUI may be
/// pointed at a real engine with <c>QZI_ENGINE_URL</c> so its job queue, SSE
/// stream and output gallery exercise genuine Qwen-Image 2.1 / Z-Image output
/// instead of the bundled deterministic reference backend.
///
/// No Go code is involved; this is a small <see cref="HttpClient"/> adapter
/// that implements the same <see cref="IQziExecutor"/> surface as QziClient.
/// </summary>
public sealed class H3ServeExecutor : IWebUiEngine
{
    private static readonly byte[] PngSignature =
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    // /generate is allowed to run for minutes, so HttpClient.Timeout is
    // infinite. Health / unload / progress are control-plane calls: they must
    // never hang the WebUI when the engine is wedged or unreachable.
    private static readonly TimeSpan ControlPlaneTimeout = TimeSpan.FromSeconds(5);

    private readonly HttpClient _http;
    private readonly string _baseUrl;
    private readonly TimeSpan _timeout;
    private readonly int _progressPollMs;

    /// <param name="engineUrl">Base URL of a running <c>h3 serve</c>, e.g. http://127.0.0.1:8411.</param>
    /// <param name="timeout">Whole-request deadline; generations can take many minutes.</param>
    /// <param name="progressPollMs">Interval for polling the optional <c>GET /progress</c> endpoint.</param>
    /// <param name="handler">Injectable for unit tests; defaults to real sockets.</param>
    public H3ServeExecutor(string engineUrl, TimeSpan timeout, int progressPollMs = 500,
                           HttpMessageHandler? handler = null)
    {
        if (string.IsNullOrWhiteSpace(engineUrl))
            throw new ArgumentException("QZI_ENGINE_URL is empty.", nameof(engineUrl));
        if (!Uri.TryCreate(engineUrl, UriKind.Absolute, out Uri? uri) ||
            (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            throw new ArgumentException(
                "QZI_ENGINE_URL must be an absolute http:// or https:// URL.", nameof(engineUrl));

        _baseUrl = engineUrl;
        _timeout = timeout > TimeSpan.Zero ? timeout : TimeSpan.FromMinutes(30);
        _progressPollMs = Math.Max(100, progressPollMs);
        _http = new HttpClient(handler ?? new SocketsHttpHandler(), disposeHandler: true)
        {
            BaseAddress = new Uri(engineUrl.TrimEnd('/') + "/", UriKind.Absolute),
            // We own the deadline with a linked CTS, because HttpClient.Timeout
            // only covers the send when ResponseHeadersRead is used.
            Timeout = Timeout.InfiniteTimeSpan
        };
    }

    public GenerationResult Generate(GenerationOptions options,
                                     IProgress<GenerationProgress>? progress = null,
                                     CancellationToken cancellationToken = default)
        => GenerateCoreAsync(options, progress, cancellationToken).GetAwaiter().GetResult();

    private async Task<GenerationResult> GenerateCoreAsync(
        GenerationOptions options, IProgress<GenerationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        byte[] body = BuildRequestBody(options);

        using CancellationTokenSource deadline =
            CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(_timeout);
        CancellationToken token = deadline.Token;

        using CancellationTokenSource pollStop =
            CancellationTokenSource.CreateLinkedTokenSource(token);
        Task? poll = progress is null ? null : PollProgressAsync(progress, pollStop.Token);

        Stopwatch clock = Stopwatch.StartNew();
        try
        {
            using ByteArrayContent content = new(body);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using HttpResponseMessage response = await _http
                .PostAsync("generate", content, token).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
                throw await EngineErrorAsync(response, token).ConfigureAwait(false);

            byte[] png = await response.Content.ReadAsByteArrayAsync(token).ConfigureAwait(false);
            clock.Stop();

            if (png.Length < PngSignature.Length ||
                !png.AsSpan(0, PngSignature.Length).SequenceEqual(PngSignature))
                throw new QziException(QziStatus.EngineFailure,
                    $"h3 serve returned {png.Length} bytes with no PNG signature " +
                    $"(Content-Type: {response.Content.Headers.ContentType}). " +
                    "Check that the request model is qwen21 or zimage.");

            return GenerationResult.FromPng(png, options.Width, options.Height,
                                            options.Seed, options.Model, clock.Elapsed);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw; // caller cancelled -> the queue marks the job "cancelled"
        }
        catch (OperationCanceledException)
        {
            throw new QziException(QziStatus.EngineFailure,
                $"h3 serve request timed out after {_timeout.TotalMinutes:0.#} minutes " +
                $"(tune QZI_ENGINE_TIMEOUT_MIN). Engine: {_baseUrl}");
        }
        catch (HttpRequestException ex)
        {
            throw new QziException(QziStatus.NotConfigured,
                $"engine unreachable at {_baseUrl}: {ex.Message}", ex);
        }
        finally
        {
            pollStop.Cancel();
            if (poll is not null)
            {
                try { await poll.ConfigureAwait(false); }
                catch { /* progress is best effort, never fails a generation */ }
            }
        }
    }

    /// <summary>
    /// Serialises the typed request into the exact <c>POST /generate</c> envelope
    /// the C engine expects. Forbidden/unknown fields are never emitted: qwen21
    /// turbo must send steps=8 and guidance=1.0 and must NOT send
    /// negative_prompt, while zimage must NOT send steps/guidance/frames/pdd8.
    /// </summary>
    public static byte[] BuildRequestBody(GenerationOptions options)
    {
        ArrayBufferWriter<byte> buffer = new();
        using (Utf8JsonWriter w = new(buffer))
        {
            w.WriteStartObject();
            w.WriteString("prompt", options.Prompt);
            w.WriteNumber("width", options.Width);
            w.WriteNumber("height", options.Height);
            w.WriteNumber("seed", options.Seed);

            switch (options)
            {
                case QwenImageOptions q:
                    w.WriteString("model", "qwen21");
                    w.WriteString("tier", q.TierName);
                    if (q.Tier == QwenImageTier.Turbo)
                    {
                        // The engine rejects any other value with 400 instead
                        // of silently rewriting it.
                        w.WriteNumber("steps", 8);
                        w.WritePropertyName("guidance");
                        w.WriteRawValue("1.0", skipInputValidation: true);
                    }
                    else
                    {
                        w.WriteNumber("steps", q.EffectiveSteps);
                        w.WriteNumber("guidance", (double)q.EffectiveGuidance);
                        if (!string.IsNullOrWhiteSpace(q.NegativePrompt))
                            w.WriteString("negative_prompt", q.NegativePrompt);
                    }
                    if (!string.IsNullOrWhiteSpace(q.Lora))
                    {
                        w.WriteString("lora", q.Lora);
                        w.WriteNumber("lora_scale", (double)q.EffectiveLoraScale);
                    }
                    break;

                case ZImageOptions z:
                    w.WriteString("model", "zimage");
                    if (!string.IsNullOrWhiteSpace(z.Checkpoint))
                        w.WriteString("checkpoint", z.Checkpoint);
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(options),
                        "Only QwenImageOptions and ZImageOptions are supported.");
            }

            w.WriteEndObject();
        }
        return buffer.WrittenSpan.ToArray();
    }

    private async Task PollProgressAsync(IProgress<GenerationProgress> progress,
                                         CancellationToken token)
    {
        using PeriodicTimer timer = new(TimeSpan.FromMilliseconds(_progressPollMs));
        while (!token.IsCancellationRequested)
        {
            try
            {
                if (!await timer.WaitForNextTickAsync(token).ConfigureAwait(false)) return;
                using CancellationTokenSource request =
                    CancellationTokenSource.CreateLinkedTokenSource(token);
                request.CancelAfter(ControlPlaneTimeout);
                using JsonDocument doc = await GetJsonAsync("progress", request.Token)
                                                 .ConfigureAwait(false);
                JsonElement root = doc.RootElement;
                if (!root.TryGetProperty("active", out JsonElement active) ||
                    active.ValueKind != JsonValueKind.True)
                    continue;

                string phase = Str(root, "phase") ?? "";
                int completed = (int)Num(root, "completed");
                int total = Math.Max(1, (int)Num(root, "total"));
                double elapsed = Dbl(root, "elapsed_s");
                progress.Report(new GenerationProgress(MapPhase(phase), completed, total,
                                                       TimeSpan.FromSeconds(elapsed)));
            }
            catch (OperationCanceledException) { return; }
            catch (Exception) { /* /progress is optional on older engine builds */ }
        }
    }

    private static GenerationPhase MapPhase(string phase)
    {
        if (phase.StartsWith("model", StringComparison.OrdinalIgnoreCase) ||
            phase.Contains("load", StringComparison.OrdinalIgnoreCase))
            return GenerationPhase.ModelLoad;
        if (phase.StartsWith("text", StringComparison.OrdinalIgnoreCase) ||
            phase.StartsWith("token", StringComparison.OrdinalIgnoreCase))
            return GenerationPhase.Text;
        if (phase.Contains("dit", StringComparison.OrdinalIgnoreCase) ||
            phase.Contains("denoise", StringComparison.OrdinalIgnoreCase))
            return GenerationPhase.Dit;
        if (phase.Contains("vae", StringComparison.OrdinalIgnoreCase))
            return GenerationPhase.Vae;
        if (phase.Contains("png", StringComparison.OrdinalIgnoreCase) ||
            phase.Contains("encode", StringComparison.OrdinalIgnoreCase))
            return GenerationPhase.Encode;
        return GenerationPhase.Unknown;
    }

    public WebUiEngineHealth GetHealth()
    {
        using CancellationTokenSource cts = new(ControlPlaneTimeout);
        try
        {
            return GetHealthAsync(cts.Token).GetAwaiter().GetResult();
        }
        catch (OperationCanceledException) when (cts.IsCancellationRequested)
        {
            throw new QziException(QziStatus.EngineFailure,
                $"h3 serve /health did not respond within " +
                $"{ControlPlaneTimeout.TotalSeconds:0}s ({_baseUrl})");
        }
        catch (HttpRequestException ex)
        {
            throw new QziException(QziStatus.NotConfigured,
                $"engine unreachable at {_baseUrl}: {ex.Message}", ex);
        }
    }

    private async Task<WebUiEngineHealth> GetHealthAsync(CancellationToken token)
    {
        using JsonDocument doc = await GetJsonAsync("health", token).ConfigureAwait(false);
        JsonElement r = doc.RootElement;
        long free = 0, total = 0;
        if (r.TryGetProperty("vram", out JsonElement vram) &&
            vram.ValueKind == JsonValueKind.Object)
        {
            free = Num(vram, "free_mib");
            total = Num(vram, "total_mib");
        }

        return new WebUiEngineHealth
        {
            ResidentModel = Str(r, "resident_model") ?? "none",
            Busy = Bool(r, "busy"),
            QueueDepth = (int)Num(r, "queue_depth"),
            Served = (ulong)Math.Max(0, Num(r, "served")),
            Rejected = (ulong)Math.Max(0, Num(r, "rejected")),
            UptimeSeconds = Dbl(r, "uptime_s"),
            FreeVramMib = free,
            TotalVramMib = total,
            IsReferenceBackend = false, // a real h3 serve owns the CUDA weights
            Backend = "h3-serve"
        };
    }

    public void Unload()
    {
        using CancellationTokenSource cts = new(ControlPlaneTimeout);
        try
        {
            UnloadAsync(cts.Token).GetAwaiter().GetResult();
        }
        catch (OperationCanceledException) when (cts.IsCancellationRequested)
        {
            throw new QziException(QziStatus.EngineFailure,
                $"h3 serve /unload did not respond within " +
                $"{ControlPlaneTimeout.TotalSeconds:0}s ({_baseUrl})");
        }
        catch (HttpRequestException ex)
        {
            throw new QziException(QziStatus.NotConfigured,
                $"engine unreachable at {_baseUrl}: {ex.Message}", ex);
        }
    }

    private async Task UnloadAsync(CancellationToken token)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, "unload");
        using HttpResponseMessage response = await _http
            .SendAsync(request, token).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw await EngineErrorAsync(response, token).ConfigureAwait(false);
    }

    private async Task<JsonDocument> GetJsonAsync(string path, CancellationToken token)
    {
        using HttpResponseMessage response = await _http
            .GetAsync(path, token).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw await EngineErrorAsync(response, token).ConfigureAwait(false);
        await using Stream stream = await response.Content
            .ReadAsStreamAsync(token).ConfigureAwait(false);
        return await JsonDocument.ParseAsync(stream, cancellationToken: token)
                                 .ConfigureAwait(false);
    }

    private static async Task<QziException> EngineErrorAsync(HttpResponseMessage response,
                                                             CancellationToken token)
    {
        string raw = await response.Content.ReadAsStringAsync(token).ConfigureAwait(false);
        string message = ExtractError(raw);
        QziStatus status = MapStatus((int)response.StatusCode, raw);
        return new QziException(status,
            $"h3 serve HTTP {(int)response.StatusCode}: {message}");
    }

    private static string ExtractError(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return "(empty response body)";
        try
        {
            using JsonDocument doc = JsonDocument.Parse(raw);
            if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                doc.RootElement.TryGetProperty("error", out JsonElement error) &&
                error.ValueKind == JsonValueKind.String)
                return error.GetString() ?? raw;
        }
        catch (JsonException) { /* not a JSON error envelope */ }
        return raw.Length > 512 ? raw[..512] + "..." : raw;
    }

    private static QziStatus MapStatus(int httpStatus, string raw) => httpStatus switch
    {
        400 or 413 or 431 => QziStatus.InvalidArgument,
        404 or 405 => QziStatus.Unsupported,
        503 when raw.Contains("\"busy\":true", StringComparison.OrdinalIgnoreCase) ||
                   raw.Contains("\"busy\": true", StringComparison.OrdinalIgnoreCase)
            => QziStatus.Busy,
        503 => QziStatus.NotConfigured, // missing model pack / unsupported tier
        500 => QziStatus.EngineFailure,
        _ => QziStatus.EngineFailure
    };

    private static string? Str(JsonElement root, string name)
        => root.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() : null;

    private static long Num(JsonElement root, string name)
        => root.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number
           && v.TryGetInt64(out long n) ? n : 0;

    private static double Dbl(JsonElement root, string name)
        => root.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number
           && v.TryGetDouble(out double d) ? d : 0;

    private static bool Bool(JsonElement root, string name)
        => root.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.True;

    public void Dispose() => _http.Dispose();
}
