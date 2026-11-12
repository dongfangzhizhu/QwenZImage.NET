using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using QwenZImage.NET.WebUI;
using QwenZImage.NET.WebUI.ModelDownloads;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// Model downloads are a host-layer feature: the catalogue is static (URLs
/// transcribed from WanGP), the HTTP client is injectable, and the config
/// writer is a dry run until explicitly applied. These tests exercise the
/// whole state machine without network access or real weights.
/// </summary>
public class ModelDownloadsTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(),
        $"qzi-models-{Guid.NewGuid():N}");
    private readonly string _configPath;

    public ModelDownloadsTests()
    {
        Directory.CreateDirectory(_dir);
        _configPath = Path.Combine(_dir, QziConfig.FileName);
        File.WriteAllText(_configPath, JsonSerializer.Serialize(new
        {
            _comment = "keep me",
            device_id = 7,
            windows = new Dictionary<string, object>
            {
                ["qwen21_dir"] = Path.Combine(_dir, "qwen21"),
                ["text_cache_dir"] = "/keep/cache"
            },
            linux = new Dictionary<string, object> { ["qwen21_dir"] = "/keep/linux" }
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    // ------------------------------------------------------------- manifest

    [Fact]
    public void Every_Url_Is_A_WanGP_HuggingFace_Resolve_Main_Link()
    {
        IReadOnlyList<ModelGroup> groups = ModelManifest.Groups;
        Assert.Equal(6, groups.Count);
        Assert.Equal(groups.Count, groups.Select(g => g.Id).Distinct().Count());

        ModelFile[] files = groups.SelectMany(g => g.Files).ToArray();
        Assert.Equal(files.Length, files.Select(f => f.Id).Distinct().Count());
        Assert.Equal(files.Length, ModelManifest.FilesById.Count);

        foreach (ModelFile file in files)
        {
            Assert.False(string.IsNullOrWhiteSpace(file.RelativePath));
            Assert.NotEmpty(file.Variants);
            foreach (ModelVariant variant in file.Variants)
            {
                Assert.StartsWith(ModelManifest.HuggingFacePrefix, variant.Url);
                Assert.Contains("/resolve/main/", variant.Url);
                Assert.True(variant.Size > 0, $"{file.Id}/{variant.Id} has no size");
                Assert.Equal(variant.Id, ModelCatalog.PickVariant(file, variant.Id).Id);
            }
        }
        Assert.All(groups, g => Assert.Contains(g.ConfigKey,
            new[] { "qwen21_dir", "qwen21_lora_dir", "zimage_dir",
                    "zimage_nunchaku_dir", "zimage_control_dir", "preprocess_dir" }));
    }

    [Fact]
    public void Mirror_Rewrites_Only_The_Fixed_HuggingFace_Prefix()
    {
        const string url = "https://huggingface.co/DeepBeepMeep/Z-Image/resolve/main/a.safetensors";
        Assert.Equal("https://hf-mirror.com/DeepBeepMeep/Z-Image/resolve/main/a.safetensors",
            ModelManifest.ResolveUrl(url, "https://hf-mirror.com/"));
        Assert.Equal(url, ModelManifest.ResolveUrl(url, null));
        Assert.Throws<ArgumentException>(() => ModelManifest.ResolveUrl(url, "file:///tmp"));
        Assert.Throws<ArgumentException>(() => ModelManifest.ResolveUrl(url, "not a url"));
    }

    [Fact]
    public void Manifest_Paths_Reject_Traversal_And_Absolute_Paths()
    {
        Assert.Throws<ArgumentException>(() => ModelManifest.CombineRelative(_dir, "../escape"));
        Assert.Throws<ArgumentException>(() => ModelManifest.CombineRelative(_dir, "a/../../escape"));
        Assert.Throws<ArgumentException>(() => ModelManifest.CombineRelative(_dir, Path.Combine(_dir, "abs")));
        Assert.Equal(Path.Combine(_dir, "pose", "yolox_l.onnx"),
            ModelManifest.CombineRelative(_dir, "pose/yolox_l.onnx"));
    }

    // -------------------------------------------------------------- catalog

    private (WebUiConfig Config, ModelDownloadService Downloads, ModelCatalog Catalog) BuildCatalog(
        string? mirror = null)
    {
        WebUiConfig config = WebUiConfig.FromEnvironment(_dir, ["--config", _configPath]);
        ModelDownloadService downloads = new(new HttpClient(new RangeHandler([1, 2, 3])));
        ModelCatalog catalog = new(config, null, downloads, mirror);
        return (config, downloads, catalog);
    }

    [Fact]
    public void Snapshot_Lists_Every_Group_With_Resumable_Manual_Commands()
    {
        (_, ModelDownloadService downloads, ModelCatalog catalog) = BuildCatalog();
        using (downloads)
        {
            ModelCatalogSnapshot snapshot = catalog.Snapshot();
            Assert.Equal(6, snapshot.Groups.Count);
            Assert.Equal(_configPath, snapshot.ConfigPath);
            Assert.Equal(_configPath, snapshot.ConfigTargetPath);
            Assert.True(snapshot.ConfigExists);
            Assert.Contains("qwen21_dir", snapshot.ConfigExample);
            Assert.Equal(ModelManifest.DefaultVariant, snapshot.DefaultVariant);

            ModelFileStatus[] files = snapshot.Groups.SelectMany(g => g.Files).ToArray();
            Assert.All(files, f => Assert.Equal("missing", f.Status));
            Assert.All(files, f => Assert.NotEmpty(f.Variants));
            Assert.All(files.SelectMany(f => f.Variants), v =>
            {
                Assert.Contains("curl -L -C -", v.CurlCommand);
                Assert.Contains(v.Url, v.CurlCommand);
                Assert.StartsWith(ModelManifest.HuggingFacePrefix, v.Url);
            });

            string guide = catalog.BuildManualGuide();
            Assert.Contains("curl -L -C -", guide);
            Assert.Contains("## 2. Point the config file", guide);
            Assert.Contains("Z-Image-Turbo-Fun-Controlnet-Union2.1", guide);
        }
    }

    [Fact]
    public void Empty_Download_Request_Expands_To_All_Missing_Required_Files()
    {
        (_, ModelDownloadService downloads, ModelCatalog catalog) = BuildCatalog();
        using (downloads)
        {
            int required = ModelManifest.Groups.Where(g => !g.Optional)
                .SelectMany(g => g.Files).Count(f => f.Required);
            Assert.Equal(10, required);   // 6 Qwen21 + 4 Z-Image core slots
            Assert.Equal(required, catalog.MissingRequiredIds().Count);

            (IReadOnlyList<ModelDownloadJob> jobs, IReadOnlyList<string> skipped) =
                catalog.CreateJobs(null, null, false);
            Assert.Equal(required, jobs.Count);
            Assert.Empty(skipped);
            Assert.All(jobs, j =>
            {
                (_, ModelFile file) = catalog.Find(j.FileId);
                // Multi-variant checkpoints take int8; single-variant assets
                // (VAE, configs, tokenizer) fall back to their only variant.
                Assert.Equal(ModelCatalog.PickVariant(file, ModelManifest.DefaultVariant).Id,
                             j.VariantId);
                Assert.StartsWith(ModelManifest.HuggingFacePrefix, j.Url);
            });

            Assert.True(catalog.MissingAllIds().Count > catalog.MissingRequiredIds().Count);
            Assert.Equal(catalog.MissingAllIds().Count,
                catalog.CreateJobs(null, null, false, all: true).Jobs.Count);
        }
    }

    [Fact]
    public void Variant_Selection_And_Unknown_Ids_Are_Handled()
    {
        (_, ModelDownloadService downloads, ModelCatalog catalog) = BuildCatalog();
        using (downloads)
        {
            (IReadOnlyList<ModelDownloadJob> jobs, _) =
                catalog.CreateJobs(["qwen21.dit.base"], "bf16", false);
            ModelDownloadJob job = Assert.Single(jobs);
            Assert.EndsWith("qwen_image_21_7B_bf16.safetensors", job.Url);
            Assert.EndsWith("dit_base.safetensors", job.TargetPath);
            Assert.Equal("bf16", job.VariantId);

            // A single-variant file falls back to its only variant.
            (IReadOnlyList<ModelDownloadJob> fallback, _) =
                catalog.CreateJobs(["qwen21.vae"], "bf16", false);
            Assert.Equal("default", Assert.Single(fallback).VariantId);

            Assert.Throws<ArgumentException>(() => catalog.CreateJobs(["not.a.file"], null, false));
        }
    }

    [Fact]
    public void Installed_Files_Are_Skipped_Unless_Forced()
    {
        (_, ModelDownloadService downloads, ModelCatalog catalog) = BuildCatalog();
        using (downloads)
        {
            (ModelGroup group, ModelFile file) = catalog.Find("qwen21.vae");
            string target = catalog.ResolveTarget(group, file, file.Variants[0]);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.WriteAllBytes(target, [1, 2, 3]);

            (IReadOnlyList<ModelDownloadJob> skippedJobs, IReadOnlyList<string> skipped) =
                catalog.CreateJobs(["qwen21.vae"], null, false);
            Assert.Empty(skippedJobs);
            Assert.Equal(["qwen21.vae"], skipped);

            (IReadOnlyList<ModelDownloadJob> forced, _) =
                catalog.CreateJobs(["qwen21.vae"], null, true);
            Assert.Single(forced);
            Assert.Equal("installed", catalog.Snapshot().Groups[0].Files
                .Single(f => f.Id == "qwen21.vae").Status);
        }
    }

    [Fact]
    public void Mirror_Appears_In_Snapshot_And_Curl_Commands()
    {
        (_, ModelDownloadService downloads, ModelCatalog catalog) =
            BuildCatalog("https://hf-mirror.com");
        using (downloads)
        {
            ModelCatalogSnapshot snapshot = catalog.Snapshot();
            Assert.Equal("https://hf-mirror.com", snapshot.Mirror);
            Assert.All(snapshot.Groups.SelectMany(g => g.Files).SelectMany(f => f.Variants),
                v => Assert.StartsWith("https://hf-mirror.com/", v.Url));
            // A mirror must never hide the exact WanGP address from the manual guide.
            Assert.All(snapshot.Groups.SelectMany(g => g.Files).SelectMany(f => f.Variants),
                v => Assert.StartsWith(ModelManifest.HuggingFacePrefix, v.SourceUrl));
            string guide = catalog.BuildManualGuide();
            Assert.Contains("https://hf-mirror.com/", guide);
            Assert.Contains("WanGP source URL", guide);
            Assert.Contains(
                "https://huggingface.co/DeepBeepMeep/Qwen_image_2/resolve/main/", guide);
        }
    }

    // -------------------------------------------------------- config writer

    [Fact]
    public void Config_Preview_Is_Read_Only_And_Apply_Preserves_Other_Keys()
    {
        (WebUiConfig config, ModelDownloadService downloads, ModelCatalog catalog) = BuildCatalog();
        using (downloads)
        {
            string before = File.ReadAllText(_configPath);
            ModelConfigApplyResult preview = ModelConfigWriter.Run(config, catalog, apply: false);
            Assert.False(preview.Applied);
            Assert.Null(preview.BackupPath);
            Assert.Equal(before, File.ReadAllText(_configPath));
            Assert.Equal(6, preview.Keys.Count);
            Assert.True(preview.RestartRequired);

            ModelConfigApplyResult applied = ModelConfigWriter.Run(config, catalog, apply: true);
            Assert.True(applied.Applied);
            Assert.NotNull(applied.BackupPath);
            Assert.True(File.Exists(applied.BackupPath));
            Assert.Equal(before, File.ReadAllText(applied.BackupPath));
            Assert.Contains("Updated", applied.Message);

            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(_configPath));
            JsonElement root = doc.RootElement;
            Assert.Equal("keep me", root.GetProperty("_comment").GetString());
            Assert.Equal(7, root.GetProperty("device_id").GetInt32());
            Assert.Equal("/keep/linux", root.GetProperty("linux").GetProperty("qwen21_dir").GetString());
            JsonElement section = root.GetProperty(catalog.ConfigPlatform);
            Assert.Equal("/keep/cache", section.GetProperty("text_cache_dir").GetString());
            foreach ((string key, string value) in applied.Keys)
                Assert.Equal(value, section.GetProperty(key).GetString());
            Assert.Equal(applied.MergedJson.TrimEnd(), File.ReadAllText(_configPath).TrimEnd());
        }
    }

    // ------------------------------------------------------- downloader core

    private static byte[] Bytes(int count)
    {
        byte[] data = new byte[count];
        for (int i = 0; i < count; i++) data[i] = (byte)(i % 251);
        return data;
    }

    private static ModelDownloadJob Job(byte[] data, string target, string id = "test.file")
        => new(id, "test-group", "default", "test file",
               "https://huggingface.co/DeepBeepMeep/Z-Image/resolve/main/test.bin",
               target, data.Length);

    private static async Task WaitUntil(Func<bool> condition, string description,
                                        int timeoutMs = 15000)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (clock.ElapsedMilliseconds < timeoutMs)
        {
            if (condition()) return;
            await Task.Delay(20);
        }
        Assert.Fail($"Timed out waiting for {description}");
    }

    private static async Task WaitForStatus(ModelDownloadService service, string status)
        => await WaitUntil(() => service.Snapshot().Values.Any(s => s.Status == status),
                           $"status '{status}'");

    /// <summary>
    /// Pause() flips the visible state immediately while the worker is still
    /// unwinding its FileStream, so wait for the .qzi-part length to settle
    /// before treating it as the resume offset.
    /// </summary>
    private static async Task<long> WaitForStablePart(string part)
    {
        long last = -1;
        int stable = 0;
        Stopwatch clock = Stopwatch.StartNew();
        while (clock.ElapsedMilliseconds < 10000)
        {
            long current = File.Exists(part) ? new FileInfo(part).Length : 0;
            stable = current == last ? stable + 1 : 0;
            if (stable >= 3 && current > 0) return current;
            last = current;
            await Task.Delay(40);
        }
        Assert.Fail("partial file size never stabilised after pause");
        return 0;
    }

    [Fact]
    public async Task Full_Download_Completes_And_Removes_The_Part_File()
    {
        byte[] data = Bytes(8192);
        RangeHandler handler = new(data);
        using HttpClient client = new(handler);
        using ModelDownloadService service = new(client);
        string target = Path.Combine(_dir, "full.bin");

        Assert.Equal(1, service.Enqueue([Job(data, target)]));
        await WaitForStatus(service, "done");

        Assert.Equal(data, File.ReadAllBytes(target));
        Assert.False(File.Exists(target + ModelDownloadService.PartSuffix));
        Assert.False(File.Exists(target + ModelDownloadService.ETagSuffix));
        Assert.Equal([0L], handler.Offsets);
        ModelDownloadState state = Assert.Single(service.Snapshot()).Value;
        Assert.Equal(data.Length, state.Bytes);
        Assert.Equal(data.Length, state.Total);
    }

    [Fact]
    public async Task Interrupted_Download_Resumes_With_Range_And_IfRange()
    {
        byte[] data = Bytes(16384);
        RangeHandler handler = new(data);
        using HttpClient client = new(handler);
        using ModelDownloadService service = new(client);
        string target = Path.Combine(_dir, "resume.bin");
        File.WriteAllBytes(target + ModelDownloadService.PartSuffix, data[..4096]);
        File.WriteAllText(target + ModelDownloadService.ETagSuffix, "\"v1\"");

        Assert.Equal(1, service.Enqueue([Job(data, target)]));
        await WaitForStatus(service, "done");

        Assert.Equal(4096L, handler.Offsets[0]);
        Assert.Equal("\"v1\"", handler.IfRanges[0]);
        Assert.Equal(data, File.ReadAllBytes(target));
        ModelDownloadState state = Assert.Single(service.Snapshot()).Value;
        Assert.Equal(4096, state.ResumeFrom);
        Assert.Equal(data.Length, state.Total);
    }

    [Fact]
    public async Task A_Server_Ignoring_Range_Restarts_Instead_Of_Appending()
    {
        byte[] data = Bytes(8192);
        RangeHandler handler = new(data) { IgnoreRange = true };
        using HttpClient client = new(handler);
        using ModelDownloadService service = new(client);
        string target = Path.Combine(_dir, "ignored.bin");
        File.WriteAllBytes(target + ModelDownloadService.PartSuffix, [9, 9, 9, 9]);

        Assert.Equal(1, service.Enqueue([Job(data, target)]));
        await WaitForStatus(service, "done");

        Assert.Equal(data, File.ReadAllBytes(target));   // not prefix + full body
        Assert.Equal(2, handler.Offsets.Count);
    }

    [Fact]
    public async Task A_Truncated_Resume_Fails_And_Keeps_The_Partial_For_Retry()
    {
        byte[] data = Bytes(8192);
        RangeHandler handler = new(data) { TruncateResume = true };
        using HttpClient client = new(handler);
        using ModelDownloadService service = new(client);
        string target = Path.Combine(_dir, "short.bin");
        File.WriteAllBytes(target + ModelDownloadService.PartSuffix, data[..1024]);

        Assert.Equal(1, service.Enqueue([Job(data, target)]));
        await WaitForStatus(service, "failed");

        ModelDownloadState state = Assert.Single(service.Snapshot()).Value;
        Assert.Contains("expected", state.Error);
        Assert.True(File.Exists(target + ModelDownloadService.PartSuffix));
        Assert.False(File.Exists(target));
    }

    [Fact]
    public async Task Pause_Resume_And_Reset_Preserve_Or_Discard_Partial_Bytes()
    {
        byte[] data = Bytes(256 * 1024);
        RangeHandler handler = new(data) { Throttle = true };
        using HttpClient client = new(handler);
        using ModelDownloadService service = new(client);
        string target = Path.Combine(_dir, "paused.bin");
        string part = target + ModelDownloadService.PartSuffix;

        service.Enqueue([Job(data, target)]);
        await WaitUntil(() => File.Exists(part) && new FileInfo(part).Length >= 4096,
                        "the first throttled chunk");
        service.Pause();
        await WaitForStatus(service, "paused");
        long pausedBytes = await WaitForStablePart(part);
        Assert.InRange(pausedBytes, 1, data.Length - 1);

        service.Resume();
        await WaitForStatus(service, "done");
        Assert.Equal(data, File.ReadAllBytes(target));
        Assert.Contains(handler.Offsets, offset => offset == pausedBytes);

        // Reset only removes .qzi-part / .qzi-etag, never a completed weight.
        service.Reset("test.file");
        Assert.False(File.Exists(part));
        Assert.True(File.Exists(target));
        Assert.Equal(data, File.ReadAllBytes(target));
    }

    [Fact]
    public async Task Reset_While_Paused_Does_Not_Resurrect_An_Abandoned_Job()
    {
        byte[] data = Bytes(128 * 1024);
        RangeHandler handler = new(data) { Throttle = true };
        using HttpClient client = new(handler);
        using ModelDownloadService service = new(client);
        string target = Path.Combine(_dir, "abandoned.bin");
        string part = target + ModelDownloadService.PartSuffix;

        service.Enqueue([Job(data, target)]);
        await WaitUntil(() => File.Exists(part) && new FileInfo(part).Length >= 4096,
                        "the first throttled chunk");
        service.Pause();
        await WaitForStatus(service, "paused");
        int offsetsAtPause = handler.Offsets.Count;

        Assert.Equal(1, service.Reset("test.file"));
        service.Resume();
        await Task.Delay(400);

        Assert.Empty(service.Snapshot());
        Assert.Equal(0, service.QueueLength);
        Assert.False(File.Exists(part));
        Assert.Equal(offsetsAtPause, handler.Offsets.Count);
    }

    // ------------------------------------------------------------- fakes

    /// <summary>In-process HTTP server with real Range / Content-Range / ETag semantics.</summary>
    private sealed class RangeHandler : HttpMessageHandler
    {
        private readonly byte[] _data;
        public RangeHandler(byte[] data) => _data = data;

        public bool IgnoreRange { get; init; }
        public bool TruncateResume { get; init; }
        public bool Throttle { get; init; }
        public List<long> Offsets { get; } = [];
        public List<string?> IfRanges { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            long? from = request.Headers.Range?.Ranges.FirstOrDefault()?.From;
            Offsets.Add(from ?? 0);
            IfRanges.Add(request.Headers.TryGetValues("If-Range", out IEnumerable<string>? values)
                ? values.FirstOrDefault() : null);

            bool partial = !IgnoreRange && from is > 0;
            long start = partial ? from!.Value : 0;
            int count = (int)(_data.Length - start);
            if (TruncateResume && partial) count = Math.Max(0, count - 3);

            HttpContent content = Throttle
                ? new StreamContent(new ThrottledStream(_data, (int)start, count))
                : new ByteArrayContent(_data, (int)start, count);
            HttpResponseMessage response = new(
                partial ? HttpStatusCode.PartialContent : HttpStatusCode.OK)
            {
                Content = content
            };
            response.Headers.ETag = new EntityTagHeaderValue("\"v1\"");
            if (partial)
                response.Content.Headers.ContentRange =
                    new ContentRangeHeaderValue(start, _data.Length - 1, _data.Length);
            return Task.FromResult(response);
        }
    }

    /// <summary>A seekable stream that yields small chunks slowly, so a pause lands mid-file.</summary>
    private sealed class ThrottledStream : Stream
    {
        private readonly MemoryStream _inner;

        public ThrottledStream(byte[] data, int offset, int count)
            => _inner = new MemoryStream(data, offset, count, writable: false, publiclyVisible: true);

        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => _inner.Length;
        public override long Position { get => _inner.Position; set => _inner.Position = value; }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
            => _inner.Read(buffer, offset, Math.Min(count, 4096));

        public override async ValueTask<int> ReadAsync(
            Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            await Task.Delay(12, cancellationToken);
            return await _inner.ReadAsync(buffer[..Math.Min(buffer.Length, 4096)], cancellationToken);
        }

        public override long Seek(long offset, SeekOrigin origin) => _inner.Seek(offset, origin);
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
