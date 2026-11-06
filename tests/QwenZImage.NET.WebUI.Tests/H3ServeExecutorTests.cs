using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// Protocol-level tests for the optional h3 serve transport. A stub
/// HttpMessageHandler plays the C engine, so these run on CI without CUDA,
/// model weights or a listening socket.
/// </summary>
public sealed class H3ServeExecutorTests
{
    private const string EngineUrl = "http://127.0.0.1:8411";

    private static readonly byte[] FakePng =
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02, 0x03];

    [Fact]
    public void Qwen21_Turbo_Emits_The_Exact_Engine_Envelope_And_Returns_Real_Bytes()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(PngResponse(FakePng)));
        using H3ServeExecutor executor = NewExecutor(handler);

        using GenerationResult result = executor.Generate(new QwenImageOptions
        {
            Prompt = "a cat", Width = 1024, Height = 1024, Seed = 7
        });

        Assert.Equal(FakePng, result.PngSpan.ToArray());
        Assert.Equal(GenerationModel.Qwen21, result.Model);
        Assert.Equal(7ul, result.Seed);
        Assert.Equal(1024, result.Width);

        string body = Assert.Single(handler.Bodies);
        Assert.Contains("\"guidance\":1.0", body); // literal 1.0, not 1
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement r = doc.RootElement;
        Assert.Equal("qwen21", r.GetProperty("model").GetString());
        Assert.Equal("turbo", r.GetProperty("tier").GetString());
        Assert.Equal("a cat", r.GetProperty("prompt").GetString());
        Assert.Equal(8, r.GetProperty("steps").GetInt32());
        Assert.Equal(1.0, r.GetProperty("guidance").GetDouble());
        Assert.Equal(1024, r.GetProperty("width").GetInt32());
        Assert.Equal(7ul, r.GetProperty("seed").GetUInt64());
        Assert.False(r.TryGetProperty("negative_prompt", out _)); // turbo has no CFG
        Assert.False(r.TryGetProperty("frames", out _));
        Assert.False(r.TryGetProperty("pdd8", out _));
        Assert.Equal("/generate", handler.Paths[0]);
    }

    [Fact]
    public void Qwen21_Base_Emits_Steps_Guidance_Negative_Prompt_And_Lora()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(PngResponse(FakePng)));
        using H3ServeExecutor executor = NewExecutor(handler);

        using GenerationResult _ = executor.Generate(new QwenImageOptions
        {
            Prompt = "a dog", Tier = QwenImageTier.Base, Steps = 40, Guidance = 4.0f,
            NegativePrompt = "blurry", Lora = "dmd2.safetensors", LoraScale = 1.0f
        });

        using JsonDocument doc = JsonDocument.Parse(handler.Bodies[0]);
        JsonElement r = doc.RootElement;
        Assert.Equal("base", r.GetProperty("tier").GetString());
        Assert.Equal(40, r.GetProperty("steps").GetInt32());
        Assert.Equal(4.0, r.GetProperty("guidance").GetDouble());
        Assert.Equal("blurry", r.GetProperty("negative_prompt").GetString());
        Assert.Equal("dmd2.safetensors", r.GetProperty("lora").GetString());
        Assert.Equal(1.0, r.GetProperty("lora_scale").GetDouble());
    }

    [Fact]
    public void ZImage_Omits_Every_Field_The_C_Engine_Rejects()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(PngResponse(FakePng)));
        using H3ServeExecutor executor = NewExecutor(handler);

        using GenerationResult result = executor.Generate(new ZImageOptions
        {
            Prompt = "a lake", Width = 1024, Height = 1024,
            Checkpoint = "nunchaku-r256"
        });

        Assert.Equal(GenerationModel.ZImage, result.Model);
        using JsonDocument doc = JsonDocument.Parse(handler.Bodies[0]);
        JsonElement r = doc.RootElement;
        Assert.Equal("zimage", r.GetProperty("model").GetString());
        Assert.Equal("nunchaku-r256", r.GetProperty("checkpoint").GetString());
        foreach (string forbidden in new[]
                 { "steps", "guidance", "frames", "pdd8", "negative_prompt", "lora", "lora_scale", "tier" })
            Assert.False(r.TryGetProperty(forbidden, out _), $"'{forbidden}' must not be sent for zimage");
    }

    [Fact]
    public void NonPng_Success_Body_Is_Rejected()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(
            BytesResponse(HttpStatusCode.OK, [0x00, 0x01, 0x02, 0x03], "video/mp4")));
        using H3ServeExecutor executor = NewExecutor(handler);

        QziException ex = Assert.Throws<QziException>(() =>
            executor.Generate(new ZImageOptions { Prompt = "x" }));

        Assert.Equal(QziStatus.EngineFailure, ex.Status);
        Assert.Contains("no PNG signature", ex.NativeMessage);
    }

    [Theory]
    [InlineData(400, "{\"error\":\"prompt is required\",\"status\":400}", QziStatus.InvalidArgument)]
    [InlineData(503, "{\"error\":\"busy\",\"busy\":true,\"queue_depth\":1}", QziStatus.Busy)]
    [InlineData(503, "{\"error\":\"model directory is missing\",\"status\":503}", QziStatus.NotConfigured)]
    [InlineData(500, "{\"error\":\"CUDA OOM\",\"status\":500}", QziStatus.EngineFailure)]
    public void Engine_Errors_Are_Mapped_To_QziStatus(int statusCode, string body, QziStatus expected)
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(
            JsonResponse((HttpStatusCode)statusCode, body)));
        using H3ServeExecutor executor = NewExecutor(handler);

        QziException ex = Assert.Throws<QziException>(() =>
            executor.Generate(new QwenImageOptions { Prompt = "x" }));

        Assert.Equal(expected, ex.Status);
        Assert.Contains($"h3 serve HTTP {statusCode}", ex.NativeMessage);
        if (statusCode == 400)
            Assert.Contains("prompt is required", ex.NativeMessage);
        if (statusCode == 500)
            Assert.Contains("CUDA OOM", ex.NativeMessage);
    }

    [Fact]
    public void Health_Reports_A_Real_Engine_Not_The_Reference_Backend()
    {
        const string health = """
            {"ready":true,"service":"h3-serve","resident_model":"qwen21",
             "busy":false,"queue_depth":0,"served":3,"rejected":1,"uptime_s":123.5,
             "vram":{"known":true,"free_mib":8192,"total_mib":24576,"used_mib":16384}}
            """;
        StubHandler handler = new((_, _, _) => Task.FromResult(JsonResponse(HttpStatusCode.OK, health)));
        using H3ServeExecutor executor = NewExecutor(handler);

        WebUiEngineHealth h = executor.GetHealth();

        Assert.False(h.IsReferenceBackend);
        Assert.Equal("h3-serve", h.Backend);
        Assert.Equal("qwen21", h.ResidentModel);
        Assert.Equal(3ul, h.Served);
        Assert.Equal(1ul, h.Rejected);
        Assert.Equal(8192, h.FreeVramMib);
        Assert.Equal(24576, h.TotalVramMib);
        Assert.Equal(123.5, h.UptimeSeconds);
        Assert.Equal("/health", handler.Paths[0]);
    }

    [Fact]
    public void Unload_Posts_To_The_Engine()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(
            JsonResponse(HttpStatusCode.OK, "{\"released\":{\"dit\":false}}")));
        using H3ServeExecutor executor = NewExecutor(handler);

        executor.Unload();

        Assert.Equal("/unload", handler.Paths[0]);
        Assert.Equal("POST", handler.Methods[0]);
    }

    [Fact]
    public void GenerationResult_FromPng_Rejects_Non_Png_Bytes()
        => Assert.Throws<ArgumentException>(() =>
            GenerationResult.FromPng([1, 2, 3], 64, 64, 0,
                                     GenerationModel.ZImage, TimeSpan.Zero));

    [Fact]
    public void Invalid_Engine_Url_Is_Rejected_Up_Front()
    {
        Assert.Throws<ArgumentException>(() => new H3ServeExecutor("ftp://x", TimeSpan.FromSeconds(1)));
        Assert.Throws<ArgumentException>(() => new H3ServeExecutor("", TimeSpan.FromSeconds(1)));
    }

    [Fact]
    public void Qwen21_Base_Without_Lora_Omits_Lora_Fields()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(PngResponse(FakePng)));
        using H3ServeExecutor executor = NewExecutor(handler);

        using GenerationResult ignored = executor.Generate(new QwenImageOptions
        {
            Prompt = "a bird", Tier = QwenImageTier.Base, Steps = 20, Guidance = 3.5f
        });

        using JsonDocument doc = JsonDocument.Parse(handler.Bodies[0]);
        JsonElement r = doc.RootElement;
        Assert.False(r.TryGetProperty("lora", out _));
        Assert.False(r.TryGetProperty("lora_scale", out _));
        Assert.False(r.TryGetProperty("negative_prompt", out _));
    }

    [Fact]
    public void ZImage_Checkpoint_Defaults_To_Bf16_And_Empty_Is_Omitted()
    {
        StubHandler handler = new((_, _, _) => Task.FromResult(PngResponse(FakePng)));
        using H3ServeExecutor executor = NewExecutor(handler);

        using GenerationResult first = executor.Generate(
            new ZImageOptions { Prompt = "a river" });
        using GenerationResult second = executor.Generate(
            new ZImageOptions { Prompt = "a river", Checkpoint = "" });

        using JsonDocument defaultBody = JsonDocument.Parse(handler.Bodies[0]);
        Assert.Equal("bf16", defaultBody.RootElement.GetProperty("checkpoint").GetString());
        using JsonDocument emptyBody = JsonDocument.Parse(handler.Bodies[1]);
        Assert.False(emptyBody.RootElement.TryGetProperty("checkpoint", out _));
    }

    [Fact]
    public void Health_Maps_Unreachable_Engine_To_NotConfigured()
    {
        StubHandler handler = new((_, _, _) =>
            throw new HttpRequestException("connection refused"));
        using H3ServeExecutor executor = NewExecutor(handler);

        QziException ex = Assert.Throws<QziException>(() => executor.GetHealth());

        Assert.Equal(QziStatus.NotConfigured, ex.Status);
        Assert.Contains("engine unreachable", ex.NativeMessage);
    }

    [Fact]
    public void Generate_Propagates_Caller_Cancellation()
    {
        // A cancelled caller token must surface as cancellation (so the queue
        // marks the job "cancelled"), never as a generic engine failure/timeout.
        StubHandler handler = new((_, _, token) =>
            Task.FromCanceled<HttpResponseMessage>(token));
        using H3ServeExecutor executor = NewExecutor(handler);
        using CancellationTokenSource cts = new();
        cts.Cancel();

        Assert.ThrowsAny<OperationCanceledException>(() =>
            executor.Generate(new ZImageOptions { Prompt = "x" },
                              cancellationToken: cts.Token));
    }

    // -------------------------------------------------------------- helpers

    private static H3ServeExecutor NewExecutor(StubHandler handler)
        => new(EngineUrl, TimeSpan.FromSeconds(10), handler: handler);

    private static HttpResponseMessage PngResponse(byte[] bytes)
        => BytesResponse(HttpStatusCode.OK, bytes, "image/png");

    private static HttpResponseMessage BytesResponse(HttpStatusCode status, byte[] bytes,
                                                      string contentType)
    {
        HttpResponseMessage response = new(status) { Content = new ByteArrayContent(bytes) };
        response.Content.Headers.ContentType = new MediaTypeHeaderValue(contentType);
        return response;
    }

    private static HttpResponseMessage JsonResponse(HttpStatusCode status, string json)
    {
        HttpResponseMessage response = new(status)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json")
        };
        return response;
    }

    private sealed class StubHandler(
        Func<HttpRequestMessage, string, CancellationToken,
             Task<HttpResponseMessage>> respond) : HttpMessageHandler
    {
        public List<string> Bodies { get; } = [];
        public List<string> Paths { get; } = [];
        public List<string> Methods { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string body = request.Content is null
                ? "" : await request.Content.ReadAsStringAsync(cancellationToken);
            Bodies.Add(body);
            Paths.Add(request.RequestUri?.AbsolutePath ?? "");
            Methods.Add(request.Method.Method);
            return await respond(request, body, cancellationToken);
        }
    }
}
