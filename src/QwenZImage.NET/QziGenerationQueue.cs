using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using QwenZImage.NET.Validation;

namespace QwenZImage.NET;

/// <summary>
/// Strictly serial job dispatcher implemented entirely in .NET 10. The native
/// engine owns one GPU and one resident model, so concurrent generations are
/// refused with <see cref="QziStatus.Busy"/>: this queue replaces the former Go
/// WebUI queue (queued/running/done/error/cancelled, cancel-queued, retry,
/// state-change events) without any Go process or HTTP hop.
/// </summary>
public sealed class QziGenerationQueue : IDisposable
{
    private readonly IQziExecutor _executor;
    private readonly Func<QziJob, GenerationResult, CancellationToken, Task>? _outputSink;
    private readonly Channel<QziJob> _queue =
        Channel.CreateUnbounded<QziJob>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false
        });
    private readonly ConcurrentDictionary<long, QziJob> _jobs = new();
    private readonly ConcurrentDictionary<int, Channel<QziJob>> _subscribers = new();
    private readonly Task _dispatcher;
    private long _nextId;
    private int _subscriberKey;
    private int _disposed;

    /// <summary>Raised on every job state transition (and on progress ticks).</summary>
    public event EventHandler<QziJob>? JobChanged;

    /// <summary>The dispatcher task; completes after the queue is drained and closed.</summary>
    public Task Completion => _dispatcher;

    /// <summary>Creates the queue and starts its single dispatcher.</summary>
    /// <param name="executor">Usually a <see cref="QziClient"/>; injectable for tests.</param>
    /// <param name="outputSink">
    /// Optional async sink that receives the PNG before it is disposed, e.g. to
    /// write <c>&lt;timestamp&gt;-job&lt;id&gt;.png</c> into an output directory.
    /// </param>
    public QziGenerationQueue(
        IQziExecutor executor,
        Func<QziJob, GenerationResult, CancellationToken, Task>? outputSink = null)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _outputSink = outputSink;
        _dispatcher = Task.Run(RunDispatcherAsync);
    }

    /// <summary>Enqueues a generation and returns immediately.</summary>
    public QziJob Submit(GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) == 1, this);
        return Enqueue(options, retryOf: null);
    }

    /// <summary>
    /// Cancels a queued job (the engine is never called). A running job is
    /// cancelled through the native cancellation callback, which the engine
    /// polls between phases and denoise steps.
    /// </summary>
    public bool Cancel(long id)
    {
        if (!_jobs.TryGetValue(id, out QziJob? job)) return false;
        if (job.IsTerminal) return false;
        if (job.State == QziJobState.Queued)
        {
            job.State = QziJobState.Cancelled;
            job.FinishedAt = DateTimeOffset.UtcNow;
            job.Cancellation?.Cancel();
            Publish(job);
            return true;
        }
        job.Cancellation?.Cancel();
        return true;
    }

    /// <summary>
    /// Re-submits a terminal job with the exact same request object. Queued and
    /// running jobs are refused (they would double-dispatch the same work).
    /// </summary>
    public QziJob? Retry(long id)
    {
        if (!_jobs.TryGetValue(id, out QziJob? original)) return null;
        if (!original.IsTerminal) return null;
        return Enqueue(original.Request, retryOf: id);
    }

    /// <summary>Looks up a job by id.</summary>
    public QziJob? GetJob(long id) => _jobs.TryGetValue(id, out QziJob? job) ? job : null;

    /// <summary>Returns all jobs, newest first.</summary>
    public IReadOnlyList<QziJob> ListJobs() => _jobs.Values.OrderByDescending(j => j.Id).ToList();

    /// <summary>
    /// Streams job state snapshots to one subscriber (replaces SSE). The
    /// subscription is registered eagerly, so no events are lost between
    /// calling this method and starting enumeration.
    /// </summary>
    public IAsyncEnumerable<QziJob> WatchAsync(CancellationToken cancellationToken = default)
    {
        int key = Interlocked.Increment(ref _subscriberKey);
        Channel<QziJob> channel = Channel.CreateUnbounded<QziJob>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        _subscribers[key] = channel;
        return DrainAsync(key, channel, cancellationToken);
    }

    private async IAsyncEnumerable<QziJob> DrainAsync(
        int key, Channel<QziJob> channel,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        try
        {
            await foreach (QziJob job in channel.Reader.ReadAllAsync(cancellationToken))
                yield return job;
        }
        finally
        {
            _subscribers.TryRemove(key, out _);
            channel.Writer.TryComplete();
        }
    }

    /// <summary>Waits until every job has reached a terminal state.</summary>
    public async Task WaitForIdleAsync(CancellationToken cancellationToken = default)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_queue.Reader.TryPeek(out _))
            {
                await Task.Delay(5, cancellationToken);
                continue;
            }
            if (_jobs.Values.All(j => j.IsTerminal)) return;
            await Task.Delay(5, cancellationToken);
        }
    }

    /// <summary>Stops accepting new jobs.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 1) return;
        _queue.Writer.TryComplete();
        foreach (Channel<QziJob> channel in _subscribers.Values)
            channel.Writer.TryComplete();
        GC.SuppressFinalize(this);
    }

    private QziJob Enqueue(GenerationOptions options, long? retryOf)
    {
        long id = Interlocked.Increment(ref _nextId);
        QziJob job = new(id, options, retryOf) { Cancellation = new CancellationTokenSource() };
        _jobs[id] = job;
        if (!_queue.Writer.TryWrite(job))
            throw new InvalidOperationException("The generation queue is closed.");
        Publish(job);
        return job;
    }

    private async Task RunDispatcherAsync()
    {
        await foreach (QziJob job in _queue.Reader.ReadAllAsync())
        {
            if (job.State == QziJobState.Cancelled)
            {
                Publish(job);
                continue; // the engine is never called for a cancelled queued job
            }

            job.State = QziJobState.Running;
            job.StartedAt = DateTimeOffset.UtcNow;
            Publish(job);

            CancellationToken token = job.Cancellation?.Token ?? CancellationToken.None;
            IProgress<GenerationProgress> progress = new SynchronousProgress<GenerationProgress>(
                p => { job.LastProgress = p; Publish(job); });

            try
            {
                using GenerationResult result = await Task.Run(
                    () => _executor.Generate(job.Request, progress, token), token);
                if (_outputSink is not null)
                    await _outputSink(job, result, token);
                job.OutputLength = result.Length;
                job.Elapsed = result.Elapsed;
                job.State = QziJobState.Done;
            }
            catch (OperationCanceledException)
            {
                job.State = QziJobState.Cancelled;
            }
            catch (QziException ex)
            {
                job.State = QziJobState.Error;
                job.Error = ex.Message;
                job.ErrorStatus = ex.Status;
            }
            catch (Exception ex)
            {
                job.State = QziJobState.Error;
                job.Error = ex.Message;
            }
            finally
            {
                job.FinishedAt = DateTimeOffset.UtcNow;
                Publish(job);
            }
        }
    }

    private void Publish(QziJob job)
    {
        JobChanged?.Invoke(this, job);
        foreach (Channel<QziJob> channel in _subscribers.Values)
            channel.Writer.TryWrite(job);
    }

    private sealed class SynchronousProgress<T>(Action<T> action) : IProgress<T>
    {
        public void Report(T value) => action(value);
    }
}
