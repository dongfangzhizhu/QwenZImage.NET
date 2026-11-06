namespace QwenZImage.NET;

/// <summary>Job lifecycle states (mirrors the former Go WebUI state machine).</summary>
public enum QziJobState
{
    /// <summary>Waiting in the serial queue.</summary>
    Queued = 0,
    /// <summary>Generation is running on the engine.</summary>
    Running = 1,
    /// <summary>Generation succeeded.</summary>
    Done = 2,
    /// <summary>Generation failed.</summary>
    Error = 3,
    /// <summary>Cancelled before finishing.</summary>
    Cancelled = 4
}

/// <summary>One queued generation and its outcome.</summary>
public sealed class QziJob
{
    internal QziJob(long id, GenerationOptions request, long? retryOf)
    {
        Id = id;
        Request = request;
        RetryOf = retryOf;
        CreatedAt = DateTimeOffset.UtcNow;
        State = QziJobState.Queued;
    }

    /// <summary>Monotonic job id.</summary>
    public long Id { get; }

    /// <summary>The request, kept verbatim for retry.</summary>
    public GenerationOptions Request { get; }

    /// <summary>Current state.</summary>
    public QziJobState State { get; internal set; }

    /// <summary>Error text for a failed job.</summary>
    public string? Error { get; internal set; }

    /// <summary>Native status for a failed job, when available.</summary>
    public QziStatus? ErrorStatus { get; internal set; }

    /// <summary>PNG byte length of a successful job.</summary>
    public int? OutputLength { get; internal set; }

    /// <summary>Last progress snapshot.</summary>
    public GenerationProgress LastProgress { get; internal set; }

    /// <summary>Generation wall-clock time.</summary>
    public TimeSpan Elapsed { get; internal set; }

    /// <summary>When the job was submitted.</summary>
    public DateTimeOffset CreatedAt { get; }

    /// <summary>When generation started.</summary>
    public DateTimeOffset? StartedAt { get; internal set; }

    /// <summary>When the job reached a terminal state.</summary>
    public DateTimeOffset? FinishedAt { get; internal set; }

    /// <summary>Original job id when this job is a retry.</summary>
    public long? RetryOf { get; }

    /// <summary>Client marker, ignored by the engine, useful for correlation.</summary>
    public string? Client { get; init; }

    internal CancellationTokenSource? Cancellation { get; set; }

    /// <summary>True for Done, Error or Cancelled.</summary>
    public bool IsTerminal => State is QziJobState.Done or QziJobState.Error or QziJobState.Cancelled;
}
