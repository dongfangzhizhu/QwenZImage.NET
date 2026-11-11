using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using Microsoft.AspNetCore.Http.Features;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;

// `dotnet run` uses the project directory (where wwwroot lives); a published
// deployment may be launched from any working directory, so fall back to the
// app base directory when the cwd has no wwwroot. User data (models, outputs)
// stays relative to the launch directory, independently of the content root.
string launchDirectory = Directory.GetCurrentDirectory();
bool cwdHasWebRoot = Directory.Exists(Path.Combine(launchDirectory, "wwwroot"));
bool appHasWebRoot = Directory.Exists(Path.Combine(AppContext.BaseDirectory, "wwwroot"));
WebApplicationBuilder builder = (!cwdHasWebRoot && appHasWebRoot)
    ? WebApplication.CreateBuilder(new WebApplicationOptions
    {
        Args = args,
        ContentRootPath = AppContext.BaseDirectory
    })
    : WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddSimpleConsole(options =>
{
    options.SingleLine = true;
    options.TimestampFormat = "HH:mm:ss ";
});

// Web assets still resolve from ContentRootPath/wwwroot, but user data (model
// weights, cache, outputs) is relative to the directory the server was
// launched from — that is what a manual-test operator expects.
WebUiConfig config = WebUiConfig.FromEnvironment(launchDirectory, args);
// An explicit --urls / ASPNETCORE_URLS always wins over the default.
if (string.IsNullOrWhiteSpace(builder.Configuration["urls"]))
    builder.WebHost.UseUrls(config.Urls);

// The Go WebUI used snake_case JSON. Keep that wire format so the manual-test
// UI (and any existing scripts) can talk to this server without translation.
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = WebUiJson.Options.PropertyNamingPolicy;
    options.SerializerOptions.PropertyNameCaseInsensitive = WebUiJson.Options.PropertyNameCaseInsensitive;
    options.SerializerOptions.DefaultIgnoreCondition = WebUiJson.Options.DefaultIgnoreCondition;
    options.SerializerOptions.Encoder = WebUiJson.Options.Encoder;
});

WebUiRuntime runtime = new(config);
builder.Services.AddSingleton(runtime);

WebApplication app = builder.Build();
app.Lifetime.ApplicationStopped.Register(runtime.Dispose);

// Unhandled errors become a JSON envelope instead of an HTML developer page.
// SSE handlers have already started the response by then, so they are skipped.
app.Use(async (context, next) =>
{
    try
    {
        await next();
    }
    catch (Exception ex) when (!context.Response.HasStarted)
    {
        app.Logger.LogError(ex, "Unhandled WebUI request error");
        context.Response.Clear();
        await Results.Json(new ErrorResponse("internal error", ex.Message), WebUiJson.Options,
                           statusCode: StatusCodes.Status500InternalServerError)
                     .ExecuteAsync(context);
    }
});

app.UseDefaultFiles();
app.UseStaticFiles();

// ------------------------------------------------------------------ health

app.MapGet("/api/health", (WebUiRuntime rt) =>
{
    object? engine = null;
    bool ok = rt.Engine is not null && rt.Queue is not null;
    string? error = rt.InitError;
    bool editing = false;
    try { editing = rt.Engine?.SupportsImageEditing == true; }
    catch (Exception ex) { app.Logger.LogDebug(ex, "Could not probe image-edit capability"); }
    bool controlnet = false;
    try { controlnet = rt.Engine?.SupportsControlNet == true; }
    catch (Exception ex) { app.Logger.LogDebug(ex, "Could not probe ControlNet capability"); }
    bool advancedEditing = false;
    try { advancedEditing = rt.Engine?.SupportsAdvancedEditing == true; }
    catch (Exception ex) { app.Logger.LogDebug(ex, "Could not probe advanced-edit capability"); }
    bool inpaint = false;
    try { inpaint = rt.Engine?.SupportsInpaint == true; }
    catch (Exception ex) { app.Logger.LogDebug(ex, "Could not probe inpaint capability"); }

    if (ok)
    {
        try
        {
            WebUiEngineHealth h = rt.Engine!.GetHealth();
            engine = new
            {
                backend = h.Backend,
                resident_model = h.ResidentModel,
                busy = h.Busy,
                queue_depth = h.QueueDepth,
                device_id = h.DeviceId,
                free_vram_mib = h.FreeVramMib,
                total_vram_mib = h.TotalVramMib,
                served = h.Served,
                rejected = h.Rejected,
                uptime_s = h.UptimeSeconds,
                reference_backend = h.IsReferenceBackend
            };
        }
        catch (Exception ex)
        {
            ok = false;
            error = $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    return Json(new
    {
        service = "QwenZImage.NET.WebUI",
        ok,
        error,
        out_dir = rt.Outputs.Root,
        model_dir = rt.Config.ModelDirectory,
        transport = "native-pinvoke",
        config_path = rt.Config.ConfigPath,
        native_library = rt.Engine?.NativeLibraryPath,
        native_library_source = rt.Config.NativeLibrarySource,
        supports_image_editing = editing,
        supports_controlnet = controlnet,
        supports_advanced_editing = advancedEditing,
        supports_inpaint = inpaint,
        uptime_s = (DateTimeOffset.UtcNow - rt.StartedAt).TotalSeconds,
        engine
    });
});

// ----------------------------------------------------------------- presets

app.MapGet("/api/presets", () =>
    Json(new { presets = QziPresets.All.Select(PresetDto.From).ToArray() }));

// -------------------------------------------------------------------- jobs

app.MapPost("/api/jobs", async (HttpContext context, WebUiRuntime rt) =>
{
    if (rt.Queue is null) return Unavailable(rt);

    IHttpMaxRequestBodySizeFeature? maxBody = context.Features.Get<IHttpMaxRequestBodySizeFeature>();
    if (maxBody is { IsReadOnly: false }) maxBody.MaxRequestBodySize = 64L << 20;

    GenerateRequest? body;
    try
    {
        body = await context.Request.ReadFromJsonAsync<GenerateRequest>(
            WebUiJson.Options, context.RequestAborted);
    }
    catch (BadHttpRequestException)
    {
        return Json(new ErrorResponse("request body too large"),
                    StatusCodes.Status413PayloadTooLarge);
    }
    catch (JsonException ex)
    {
        return BadRequest("invalid JSON body: " + ex.Message);
    }

    if (body is null) return BadRequest("request body must be a JSON object");

    GenerationOptions options;
    QziJob job;
    try
    {
        options = RequestMapper.ToOptions(body);
        // Editing, ControlNet and inpaint are additive in ABI v1: an older
        // native library still serves text-to-image but lacks the matching
        // export. Reject those modes here so the operator sees a 400 instead
        // of a queued job failing later (or silently degrading to t2i / noise).
        string? capabilityError = WebUiCapabilities.ErrorFor(options,
            rt.Engine?.SupportsImageEditing == true,
            rt.Engine?.SupportsControlNet == true,
            rt.Engine?.SupportsAdvancedEditing == true,
            rt.Engine?.SupportsInpaint == true);
        if (capabilityError is not null) return BadRequest(capabilityError);
        job = rt.Queue.Submit(options);
    }
    catch (ArgumentException ex)
    {
        return BadRequest(ex.Message);
    }
    catch (InvalidOperationException ex)
    {
        return Json(new ErrorResponse("generation queue is closed", ex.Message),
                    StatusCodes.Status503ServiceUnavailable);
    }

    string? client = body.Client?.Trim();
    if (client is { Length: > 128 }) client = client[..128];
    rt.SetClient(job.Id, client);
    return Json(JobDto.From(job, rt), statusCode: StatusCodes.Status202Accepted);
});

app.MapGet("/api/jobs", (HttpRequest request, WebUiRuntime rt) =>
{
    if (rt.Queue is null) return Json(new { jobs = Array.Empty<JobDto>() });
    string? client = request.Query["client"].FirstOrDefault();
    IEnumerable<QziJob> jobs = rt.Queue.ListJobs();
    if (!string.IsNullOrWhiteSpace(client))
        jobs = jobs.Where(j => string.Equals(rt.ClientFor(j.Id), client, StringComparison.Ordinal));
    return Json(new { jobs = jobs.Select(j => JobDto.From(j, rt)).ToArray() });
});

app.MapGet("/api/jobs/{id:long}", (long id, WebUiRuntime rt) =>
    rt.Queue?.GetJob(id) is { } job
        ? Json(JobDto.From(job, rt))
        : NotFound("no such job"));

app.MapPost("/api/jobs/{id:long}/cancel", (long id, WebUiRuntime rt) =>
{
    if (rt.Queue is null) return Unavailable(rt);
    QziJob? job = rt.Queue.GetJob(id);
    if (job is null) return NotFound("no such job");
    if (job.IsTerminal)
        return Conflict($"job {id} is already {job.State.ToString().ToLowerInvariant()}");
    if (!rt.Queue.Cancel(id)) return Conflict($"job {id} could not be cancelled");
    return Json(JobDto.From(job, rt));
});

app.MapPost("/api/jobs/{id:long}/retry", (long id, WebUiRuntime rt) =>
{
    if (rt.Queue is null) return Unavailable(rt);
    QziJob? original = rt.Queue.GetJob(id);
    if (original is null) return NotFound("no such job");
    if (!original.IsTerminal) return Conflict($"job {id} is still {original.State.ToString().ToLowerInvariant()}");
    QziJob? retry = rt.Queue.Retry(id);
    if (retry is null) return Conflict($"job {id} could not be retried");
    rt.SetClient(retry.Id, rt.ClientFor(id));
    return Json(JobDto.From(retry, rt), statusCode: StatusCodes.Status202Accepted);
});

// ------------------------------------------------------------------ unload

IResult UnloadEngine(WebUiRuntime rt)
{
    if (rt.Engine is null) return Unavailable(rt);
    try
    {
        rt.Engine.Unload();
        return Json(new { released = true });
    }
    catch (Exception ex)
    {
        return Json(new ErrorResponse("unload failed", ex.Message),
                    StatusCodes.Status500InternalServerError);
    }
}

app.MapGet("/api/unload", (WebUiRuntime rt) => UnloadEngine(rt));
app.MapPost("/api/unload", (WebUiRuntime rt) => UnloadEngine(rt));

// ----------------------------------------------------------------- outputs

app.MapGet("/api/outputs", (WebUiRuntime rt) =>
    Json(new { outputs = rt.Outputs.List() }));

app.MapGet("/api/outputs/{name}", (string name, WebUiRuntime rt) =>
{
    if (!OutputStore.IsSafeName(name)) return BadRequest("invalid output name");
    string path = rt.Outputs.Resolve(name);
    if (!File.Exists(path)) return NotFound("no such output file");
    return Results.File(path, contentType: OutputStore.ContentTypeFor(name),
                        enableRangeProcessing: true);
});

app.MapDelete("/api/outputs/{name}", (string name, WebUiRuntime rt) =>
{
    if (!rt.Outputs.TryTrash(name, out string error))
        return BadRequest(error);
    return Json(new { deleted = name, trashed = name });
});

app.MapDelete("/api/outputs", async (HttpContext context, WebUiRuntime rt) =>
{
    List<string>? names = await ReadNamesAsync(context);
    if (names is null) return BadRequest("body must be a JSON object with a names array");
    if (names.Count == 0) return BadRequest("names must not be empty");

    var deleted = new List<string>();
    var failed = new List<object>();
    foreach (string name in names)
    {
        if (rt.Outputs.TryTrash(name, out string error)) deleted.Add(name);
        else failed.Add(new { name, error });
    }
    return Json(new { deleted, failed });
});

app.MapPost("/api/outputs/restore", async (HttpContext context, WebUiRuntime rt) =>
{
    List<string>? names = await ReadNamesAsync(context);
    if (names is null) return BadRequest("body must be a JSON object with a names array");
    if (names.Count == 0) return BadRequest("names must not be empty");

    var restored = new List<object>();
    var failed = new List<object>();
    foreach (string name in names)
    {
        if (rt.Outputs.TryRestore(name, out string restoredName, out string error))
            restored.Add(new { name, restored_as = restoredName });
        else
            failed.Add(new { name, error });
    }
    return Json(new { restored, failed });
});

// -------------------------------------------------------------------- SSE

// One merged stream: job snapshots (and their stage-level progress twins) plus
// a 15-second heartbeat. A single writer owns the response body; the queue
// watcher and the timer only write to an in-memory channel.
app.MapGet("/api/events", async (HttpContext context, WebUiRuntime rt) =>
{
    context.Response.Headers.ContentType = "text/event-stream";
    context.Response.Headers.CacheControl = "no-store";
    context.Response.Headers.Connection = "keep-alive";
    context.Response.Headers["X-Accel-Buffering"] = "no";

    CancellationToken abort = context.RequestAborted;
    await context.Response.WriteAsync("retry: 3000\n\n", abort);
    await context.Response.Body.FlushAsync(abort);

    Channel<SseEvent> events = Channel.CreateUnbounded<SseEvent>(
        new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
    using CancellationTokenSource linked = CancellationTokenSource.CreateLinkedTokenSource(abort);
    CancellationToken token = linked.Token;

    _ = Task.Run(async () =>
    {
        if (rt.Queue is null)
        {
            events.Writer.TryWrite(new SseEvent("error", JsonSerializer.Serialize(
                new ErrorResponse("generation engine unavailable", rt.InitError), WebUiJson.Options)));
            return; // keep the channel open so heartbeats still flow
        }

        try
        {
            await foreach (QziJob job in rt.Queue.WatchAsync(token))
            {
                string data = JsonSerializer.Serialize(JobDto.From(job, rt), WebUiJson.Options);
                events.Writer.TryWrite(new SseEvent("job", data));

                GenerationProgress p = job.LastProgress;
                if (job.State == QziJobState.Running && p.Total > 0)
                {
                    string progress = JsonSerializer.Serialize(new
                    {
                        job_id = job.Id,
                        client = rt.ClientFor(job.Id),
                        phase = p.Phase.ToString(),
                        completed = p.Completed,
                        total = p.Total,
                        fraction = p.Fraction,
                        elapsed_ms = p.Elapsed.TotalMilliseconds
                    }, WebUiJson.Options);
                    events.Writer.TryWrite(new SseEvent("progress", progress));
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            app.Logger.LogError(ex, "SSE queue watcher stopped");
        }
        finally
        {
            events.Writer.TryComplete();
        }
    }, token);

    _ = Task.Run(async () =>
    {
        using PeriodicTimer timer = new(TimeSpan.FromSeconds(15));
        try
        {
            while (await timer.WaitForNextTickAsync(token))
                events.Writer.TryWrite(new SseEvent(null, string.Empty));
        }
        catch (OperationCanceledException) { }
    }, token);

    try
    {
        await foreach (SseEvent item in events.Reader.ReadAllAsync(token))
        {
            if (item.Event is null)
                await context.Response.WriteAsync(": ping\n\n", token);
            else
                await context.Response.WriteAsync(
                    $"event: {item.Event}\ndata: {item.Data}\n\n", token);
            await context.Response.Body.FlushAsync(token);
        }
    }
    catch (OperationCanceledException) { }
    catch (Exception ex) when (ex is not OperationCanceledException)
    {
        app.Logger.LogDebug(ex, "SSE client disconnected");
    }
    finally
    {
        linked.Cancel();
    }
});

app.MapFallbackToFile("index.html");

app.Lifetime.ApplicationStarted.Register(() =>
{
    app.Logger.LogInformation("QwenZImage.NET WebUI listening on {Urls}",
                              string.Join(", ", app.Urls));
    app.Logger.LogInformation("Output directory: {OutputDirectory}", runtime.Outputs.Root);
    app.Logger.LogInformation("Config file: {ConfigPath}",
                              runtime.Config.ConfigPath ?? "(none - using defaults)");
    app.Logger.LogInformation("Model pack: {ModelDirectory}", runtime.Config.ModelDirectory);
    app.Logger.LogInformation("Transport: in-process P/Invoke into QwenZImageNative");
    try
    {
        if (runtime.Engine?.GetHealth().IsReferenceBackend == true)
            app.Logger.LogWarning(
                "The loaded native library is the deterministic REFERENCE backend: it emits " +
                "noise. Build with -DQZI_WITH_CUDA_ENGINE=ON and point " +
                "QwenZImage.config.json at the model packs for real images.");
    }
    catch (Exception ex)
    {
        app.Logger.LogWarning("Could not query the native backend: {Error}", ex.Message);
    }
    if (runtime.InitError is not null)
        app.Logger.LogWarning("Engine unavailable: {Error}", runtime.InitError);
});

await app.RunAsync();

// ------------------------------------------------------------------ helpers

static IResult Json(object payload, int statusCode = StatusCodes.Status200OK)
    => Results.Json(payload, WebUiJson.Options, statusCode: statusCode);

static IResult BadRequest(string message)
    => Json(new ErrorResponse(message), StatusCodes.Status400BadRequest);

static IResult NotFound(string message)
    => Json(new ErrorResponse(message), StatusCodes.Status404NotFound);

static IResult Conflict(string message)
    => Json(new ErrorResponse(message), StatusCodes.Status409Conflict);

static IResult Unavailable(WebUiRuntime rt)
    => Json(new ErrorResponse("generation engine unavailable", rt.InitError),
            StatusCodes.Status503ServiceUnavailable);

static async Task<List<string>?> ReadNamesAsync(HttpContext context)
{
    try
    {
        NamesRequest? body = await context.Request.ReadFromJsonAsync<NamesRequest>(
            WebUiJson.Options, context.RequestAborted);
        return body?.Names;
    }
    catch (JsonException)
    {
        return null;
    }
    catch (BadHttpRequestException)
    {
        return null;
    }
}

// -------------------------------------------------------------- shared types

internal sealed record SseEvent(string? Event, string Data);

/// <summary>
/// One JSON configuration used for both Minimal API results and SSE payloads,
/// so every endpoint emits the same snake_case wire format.
/// </summary>
internal static class WebUiJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };
}
