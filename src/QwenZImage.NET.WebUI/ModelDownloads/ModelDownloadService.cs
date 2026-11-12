using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;

namespace QwenZImage.NET.WebUI.ModelDownloads;

/// <summary>One queued file transfer. URLs and paths are resolved by ModelCatalog, never by the browser.</summary>
public sealed record ModelDownloadJob(
    string FileId,
    string GroupId,
    string VariantId,
    string Label,
    string Url,
    string TargetPath,
    long ExpectedSize);

/// <summary>Immutable view of one file's transfer state, for the JSON API.</summary>
public sealed record ModelDownloadState(
    string Status,
    string? VariantId,
    string? Url,
    string? TargetPath,
    long Bytes,
    long Total,
    long ResumeFrom,
    double SpeedBps,
    string? Error);

/// <summary>
/// Serial, resumable downloader. Every transfer writes to
/// <c>&lt;target&gt;.qzi-part</c> and resumes from that file's length with an
/// HTTP <c>Range: bytes=N-</c> request (plus <c>If-Range</c> when the server
/// supplied an ETag). Pausing, restarting the process or a network failure
/// therefore never discards already-fetched bytes. A completed file is
/// size-checked and atomically moved into place.
/// </summary>
/// <remarks>
/// Only URLs from <see cref="ModelManifest"/> are ever requested. The optional
/// <see cref="HttpClient"/> injection exists so xUnit tests can serve Range
/// responses without network access.
/// </remarks>
public sealed class ModelDownloadService : IDisposable
{
    /// <summary>Suffix of the temporary partial file.</summary>
    public const string PartSuffix = ".qzi-part";

    /// <summary>Suffix of the file holding the ETag used for If-Range.</summary>
    public const string ETagSuffix = ".qzi-etag";

    private readonly object _sync = new();
    private readonly Dictionary<string, MutableState> _states = new(StringComparer.Ordinal);
    private readonly List<ModelDownloadJob> _pending = [];
    private readonly CancellationTokenSource _shutdown = new();
    private readonly HttpClient _client;
    private readonly bool _ownsClient;

    private ModelDownloadJob? _current;
    private CancellationTokenSource? _currentCts;
    private Task? _worker;
    private bool _paused;
    private bool _disposed;

    // Bumped by Reset(). A worker that started a job before the reset must not
    // put that abandoned job back into the queue when its cancellation lands.
    private long _generation;

    // Targets whose partial files were deleted by Reset() while the worker
    // still owned an open FileStream. On Windows File.Delete() cannot remove
    // an open file (and the worker may recreate it), so the worker performs a
    // second deletion in its finally block, after the stream is disposed.
    private readonly HashSet<string> _discardParts = new(StringComparer.Ordinal);

    public ModelDownloadService(HttpClient? client = null)
    {
        if (client is not null)
        {
            _client = client;
            _ownsClient = false;
        }
        else
        {
            _client = new HttpClient(new HttpClientHandler
            {
                AllowAutoRedirect = true,
                MaxConnectionsPerServer = 4
            })
            {
                // A 14 GB BF16 checkpoint can take a long time on a slow link;
                // cancellation is explicit, not a surprise timeout mid-file.
                Timeout = TimeSpan.FromHours(12)
            };
            _client.DefaultRequestHeaders.UserAgent.ParseAdd("QwenZImage.NET.WebUI/1.0");
            _ownsClient = true;
        }
    }

    /// <summary>True while transfers are held; partial files are kept for resume.</summary>
    public bool Paused { get { lock (_sync) return _paused; } }

    /// <summary>Number of jobs waiting behind the active transfer.</summary>
    public int QueueLength { get { lock (_sync) return _pending.Count + (_current is null ? 0 : 1); } }

    /// <summary>
    /// Adds jobs to the queue. A file already queued, downloading or done is
    /// skipped, so a double-click cannot start two writers for one target.
    /// </summary>
    public int Enqueue(IEnumerable<ModelDownloadJob> jobs)
    {
        int added = 0;
        lock (_sync)
        {
            ThrowIfDisposed();
            foreach (ModelDownloadJob job in jobs)
            {
                if (_current?.FileId == job.FileId) continue;
                if (_pending.Any(p => p.FileId == job.FileId)) continue;
                // A "done" / "failed" / "paused" state may be re-queued (for
                // example with force=true), but never an active transfer.
                if (_states.TryGetValue(job.FileId, out MutableState? state) &&
                    state.Status == "downloading") continue;

                _pending.Add(job);
                _states[job.FileId] = MutableState.Queued(job);
                added++;
            }
            EnsureWorker();
        }
        return added;
    }

    /// <summary>Suspends the active transfer (partial bytes are retained).</summary>
    public void Pause()
    {
        lock (_sync)
        {
            _paused = true;
            _currentCts?.Cancel();
            foreach (MutableState state in _states.Values)
                if (state.Status is "queued" or "downloading") state.Status = "paused";
        }
    }

    /// <summary>Continues paused / queued transfers from their partial files.</summary>
    public void Resume()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            _paused = false;
            foreach (MutableState state in _states.Values)
                if (state.Status == "paused") state.Status = "queued";
            EnsureWorker();
        }
    }

    /// <summary>
    /// Clears a failed / paused transfer and deletes its partial file so the
    /// next download starts from zero. Pass <c>null</c> to reset everything.
    /// </summary>
    public int Reset(string? fileId)
    {
        lock (_sync)
        {
            _generation++;
            bool abandonCurrent = _current is not null &&
                                  (fileId is null || _current.FileId == fileId);
            if (abandonCurrent) _currentCts?.Cancel();
            if (fileId is null) _pending.Clear();
            else _pending.RemoveAll(p => p.FileId == fileId);

            string[] ids = fileId is null
                ? _states.Keys.ToArray()
                : _states.ContainsKey(fileId) ? [fileId] : [];
            foreach (string id in ids)
            {
                string? target = _states[id].TargetPath;
                DeletePart(target);
                _states.Remove(id);
                // The active transfer may still hold (or be about to open) its
                // FileStream, so schedule a second deletion after it unwinds.
                if (abandonCurrent && !string.IsNullOrWhiteSpace(target) &&
                    _current is not null && _current.FileId == id)
                    _discardParts.Add(target);
            }
            return ids.Length;
        }
    }

    /// <summary>Thread-safe view for <c>GET /api/models</c>.</summary>
    public IReadOnlyDictionary<string, ModelDownloadState> Snapshot()
    {
        lock (_sync)
        {
            return _states.ToDictionary(
                pair => pair.Key,
                pair => new ModelDownloadState(
                    pair.Value.Status, pair.Value.VariantId, pair.Value.Url,
                    pair.Value.TargetPath, pair.Value.Bytes, pair.Value.Total,
                    pair.Value.ResumeFrom, pair.Value.SpeedBps, pair.Value.Error),
                StringComparer.Ordinal);
        }
    }

    public void Dispose()
    {
        Task? worker;
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            _shutdown.Cancel();
            try { _currentCts?.Cancel(); }
            catch (ObjectDisposedException) { /* the worker already finished */ }
            worker = _worker;
        }
        // Let the worker observe the shutdown token before the CTS / HttpClient
        // are disposed; otherwise its Task.Delay(..., _shutdown.Token) would
        // touch a disposed source. The wait is bounded so shutdown never hangs
        // on a slow network read.
        try { worker?.Wait(TimeSpan.FromSeconds(2)); }
        catch (AggregateException) { }
        _shutdown.Dispose();
        if (_ownsClient) _client.Dispose();
        GC.SuppressFinalize(this);
    }

    // ------------------------------------------------------------- internals

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ModelDownloadService));
    }

    private void EnsureWorker()
    {
        // Called under _sync.
        if (_worker is { IsCompleted: false }) return;
        _worker = Task.Run(WorkerAsync);
    }

    private async Task WorkerAsync()
    {
        while (true)
        {
            ModelDownloadJob job;
            CancellationTokenSource cts;
            long generation;
            lock (_sync)
            {
                generation = _generation;
                if (_disposed || (_pending.Count == 0 && _current is null))
                {
                    _worker = null;
                    return;
                }
                if (_paused || _pending.Count == 0)
                {
                    // Keep one idle worker alive while paused so Resume() is
                    // race-free; Dispose() cancels the shutdown token.
                    job = null!;
                    cts = null!;
                }
                else
                {
                    job = _pending[0];
                    _pending.RemoveAt(0);
                    _current = job;
                    cts = CancellationTokenSource.CreateLinkedTokenSource(_shutdown.Token);
                    _currentCts = cts;
                    if (_states.TryGetValue(job.FileId, out MutableState? state))
                    {
                        state.Status = "downloading";
                        state.Error = null;
                    }
                }
            }

            if (job is null)
            {
                try { await Task.Delay(120, _shutdown.Token).ConfigureAwait(false); }
                catch (OperationCanceledException)
                {
                    lock (_sync) _worker = null;
                    return;
                }
                continue;
            }

            try
            {
                await DownloadOneAsync(job, cts.Token).ConfigureAwait(false);
                lock (_sync)
                {
                    if (_states.TryGetValue(job.FileId, out MutableState? done))
                    {
                        done.Status = "done";
                        done.Bytes = done.Total;
                        done.SpeedBps = 0;
                        done.Error = null;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                bool requeue;
                lock (_sync)
                {
                    // A Reset() that happened while this transfer was being
                    // cancelled removes its state and bumps the generation:
                    // never resurrect an abandoned job into the pending list.
                    requeue = !_disposed && _paused && generation == _generation &&
                              _states.ContainsKey(job.FileId);
                    if (requeue)
                    {
                        _pending.Insert(0, job);
                        if (_states.TryGetValue(job.FileId, out MutableState? held))
                        {
                            held.Status = "paused";
                            held.SpeedBps = 0;
                        }
                    }
                    else if (_states.TryGetValue(job.FileId, out MutableState? idle))
                    {
                        idle.Status = "idle";
                    }
                }
                if (!requeue && _disposed)
                {
                    lock (_sync) { _current = null; _worker = null; }
                    cts.Dispose();
                    return;
                }
            }
            catch (Exception ex)
            {
                lock (_sync)
                {
                    if (_states.TryGetValue(job.FileId, out MutableState? failed))
                    {
                        failed.Status = "failed";
                        failed.Error = ex.Message;
                        failed.SpeedBps = 0;
                    }
                }
            }
            finally
            {
                cts.Dispose();
                bool discard;
                lock (_sync)
                {
                    discard = generation != _generation &&
                              _discardParts.Remove(job.TargetPath);
                    _current = null;
                    _currentCts = null;
                }
                // FileStream is closed by now: this second deletion actually
                // succeeds on Windows and cannot be undone by a late write.
                if (discard) DeletePart(job.TargetPath);
            }
        }
    }

    private async Task DownloadOneAsync(ModelDownloadJob job, CancellationToken token)
    {
        string directory = Path.GetDirectoryName(job.TargetPath)
                           ?? throw new InvalidDataException($"No directory for {job.TargetPath}");
        Directory.CreateDirectory(directory);

        string part = job.TargetPath + PartSuffix;
        string etagFile = job.TargetPath + ETagSuffix;
        long offset = File.Exists(part) ? new FileInfo(part).Length : 0;
        string? etag = File.Exists(etagFile) ? File.ReadAllText(etagFile) : null;
        if (offset == 0 && !string.IsNullOrWhiteSpace(etag)) File.Delete(etagFile);

        using HttpResponseMessage response = await OpenAsync(job.Url, offset, etag, token)
                                                       .ConfigureAwait(false);
        ValidateStatus(response, job.Label);

        // If the server ignored / rejected our Range request, restart cleanly
        // instead of appending a full body to an existing prefix.
        if (offset > 0 && (response.StatusCode != HttpStatusCode.PartialContent ||
                           response.Content.Headers.ContentRange?.From != offset))
        {
            offset = 0;
            response.Dispose();
            using HttpResponseMessage fresh = await OpenAsync(job.Url, 0, null, token)
                                                          .ConfigureAwait(false);
            ValidateStatus(fresh, job.Label, "while restarting ");
            long freshTotal = ResolveTotal(fresh, job, 0);
            await CopyAsync(fresh, job, part, etagFile, 0, freshTotal, token)
                .ConfigureAwait(false);
            VerifyLength(part, freshTotal, job.Label);
        }
        else
        {
            long total = ResolveTotal(response, job, offset);
            await CopyAsync(response, job, part, etagFile, offset, total, token)
                .ConfigureAwait(false);
            VerifyLength(part, total, job.Label);
        }

        File.Move(part, job.TargetPath, overwrite: true);
        if (File.Exists(etagFile)) File.Delete(etagFile);
    }

    private static void ValidateStatus(HttpResponseMessage response, string label,
                                       string where = "while downloading ")
    {
        if (response.StatusCode is not (HttpStatusCode.OK or HttpStatusCode.PartialContent))
            throw new HttpRequestException(
                $"{(int)response.StatusCode} {response.StatusCode} {where}{label}");
    }

    /// <summary>
    /// Prefer the server's own length over the catalogue's recorded size.
    /// WanGP links follow <c>resolve/main</c> without a pinned commit, so an
    /// upstream re-upload can change a checkpoint's size; the manifest value
    /// is then only a progress-display hint, not a reason to reject a complete
    /// and otherwise valid transfer.
    /// </summary>
    private static long ResolveTotal(HttpResponseMessage response, ModelDownloadJob job,
                                     long offset)
    {
        ContentRangeHeaderValue? range = response.Content.Headers.ContentRange;
        if (range?.To is { } last) return last + 1;   // Content-Range To is inclusive
        if (response.Content.Headers.ContentLength is { } contentLength)
            return offset + contentLength;
        return job.ExpectedSize;
    }

    private static void VerifyLength(string part, long total, string label)
    {
        long length = new FileInfo(part).Length;
        if (total > 0 && length != total)
            throw new InvalidDataException(
                $"{label}: expected {total} bytes but the download has {length}. " +
                "Use reset to discard the partial file and try again.");
    }

    private async Task<HttpResponseMessage> OpenAsync(string url, long offset, string? etag,
                                                      CancellationToken token)
    {
        using HttpRequestMessage request = new(HttpMethod.Get, url);
        if (offset > 0)
        {
            request.Headers.Range = new RangeHeaderValue(offset, null);
            if (!string.IsNullOrWhiteSpace(etag))
                request.Headers.TryAddWithoutValidation("If-Range", etag);
        }
        return await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token)
                            .ConfigureAwait(false);
    }

    private async Task CopyAsync(HttpResponseMessage response, ModelDownloadJob job,
                                 string part, string etagFile, long offset, long total,
                                 CancellationToken token)
    {
        string? serverETag = response.Headers.ETag?.Tag;
        if (!string.IsNullOrWhiteSpace(serverETag))
            await File.WriteAllTextAsync(etagFile, serverETag, token).ConfigureAwait(false);

        lock (_sync)
        {
            if (_states.TryGetValue(job.FileId, out MutableState? begin))
            {
                begin.ResumeFrom = offset;
                begin.Bytes = offset;
                begin.Total = total;
            }
        }

        await using Stream source = await response.Content.ReadAsStreamAsync(token)
                                                   .ConfigureAwait(false);
        await using FileStream target = new(part, FileMode.OpenOrCreate, FileAccess.Write,
                                            FileShare.Read, 1 << 16, useAsync: true);
        target.SetLength(offset);
        target.Seek(offset, SeekOrigin.Begin);

        byte[] buffer = new byte[1 << 16];
        Stopwatch clock = Stopwatch.StartNew();
        long lastBytes = offset;
        double lastSeconds = 0;
        long written = 0;
        while (true)
        {
            int read = await source.ReadAsync(buffer, token).ConfigureAwait(false);
            if (read == 0) break;
            await target.WriteAsync(buffer.AsMemory(0, read), token).ConfigureAwait(false);
            written += read;

            double seconds = clock.Elapsed.TotalSeconds;
            if (seconds - lastSeconds >= 0.2)
            {
                double speed = (written - (lastBytes - offset)) / (seconds - lastSeconds);
                lock (_sync)
                {
                    if (_states.TryGetValue(job.FileId, out MutableState? progress))
                    {
                        progress.Bytes = offset + written;
                        progress.SpeedBps = speed;
                    }
                }
                lastBytes = offset + written;
                lastSeconds = seconds;
            }
        }
        await target.FlushAsync(token).ConfigureAwait(false);

        lock (_sync)
        {
            if (_states.TryGetValue(job.FileId, out MutableState? finish))
            {
                finish.Bytes = offset + written;
                finish.Total = total;
                finish.SpeedBps = written / Math.Max(clock.Elapsed.TotalSeconds, 0.001);
            }
        }
    }

    private static void DeletePart(string? targetPath)
    {
        if (string.IsNullOrWhiteSpace(targetPath)) return;
        foreach (string suffix in new[] { PartSuffix, ETagSuffix })
        {
            try { if (File.Exists(targetPath + suffix)) File.Delete(targetPath + suffix); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    private sealed class MutableState
    {
        public string Status = "queued";
        public string? VariantId;
        public string? Url;
        public string? TargetPath;
        public long Bytes;
        public long Total;
        public long ResumeFrom;
        public double SpeedBps;
        public string? Error;

        public static MutableState Queued(ModelDownloadJob job) => new()
        {
            Status = "queued",
            VariantId = job.VariantId,
            Url = job.Url,
            TargetPath = job.TargetPath,
            Total = job.ExpectedSize
        };
    }
}
