using Xunit;

namespace QwenZImage.NET.Tests;

public class QueueTests
{
    private static QwenImageOptions Request(string prompt = "a fox") =>
        new() { Prompt = prompt, Width = 32, Height = 32 };

    [Fact]
    public async Task Jobs_Run_Serially_And_All_Complete()
    {
        var executor = new RecordingExecutor();
        using var queue = new QziGenerationQueue(executor);

        for (int i = 0; i < 6; i++) queue.Submit(Request($"fox {i}"));

        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(1, executor.MaxConcurrency);
        Assert.Equal(6, executor.Calls.Count);
        Assert.All(queue.ListJobs(), j => Assert.Equal(QziJobState.Done, j.State));
        Assert.All(queue.ListJobs(), j => Assert.NotNull(j.OutputLength));
        Assert.Contains(queue.ListJobs(), j => j.LastProgress.Phase == GenerationPhase.Dit);
    }

    [Fact]
    public async Task Cancelling_A_Queued_Job_Never_Calls_The_Engine()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var executor = new RecordingExecutor
        {
            Handler = (options, _, _) =>
            {
                entered.TrySetResult();
                release.Task.GetAwaiter().GetResult();
                return RecordingExecutor.MakeResult(options);
            }
        };
        using var queue = new QziGenerationQueue(executor);

        QziJob first = queue.Submit(Request("first"));
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        QziJob second = queue.Submit(Request("cancelled"));

        Assert.True(queue.Cancel(second.Id));
        release.TrySetResult();
        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(QziJobState.Done, first.State);
        Assert.Equal(QziJobState.Cancelled, second.State);
        Assert.DoesNotContain(executor.Calls, o => o.Prompt == "cancelled");
        Assert.Single(executor.Calls);
    }

    [Fact]
    public async Task Retry_Reuses_The_Exact_Request_And_Records_RetryOf()
    {
        var executor = new RecordingExecutor();
        using var queue = new QziGenerationQueue(executor);
        QziJob original = queue.Submit(Request());
        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));

        QziJob? retry = queue.Retry(original.Id);
        Assert.NotNull(retry);
        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(original.Id, retry!.RetryOf);
        Assert.Same(original.Request, retry.Request);
        Assert.Equal(2, executor.Calls.Count);
        Assert.Null(queue.Retry(long.MaxValue));
    }

    [Fact]
    public async Task Retry_Of_A_Queued_Job_Is_Refused()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var executor = new RecordingExecutor
        {
            Handler = (options, _, _) =>
            {
                entered.TrySetResult();
                release.Task.GetAwaiter().GetResult();
                return RecordingExecutor.MakeResult(options);
            }
        };
        using var queue = new QziGenerationQueue(executor);
        QziJob first = queue.Submit(Request());
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Null(queue.Retry(first.Id)); // running jobs must not be double-dispatched
        release.TrySetResult();
        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task Native_Errors_Are_Captured_On_The_Job()
    {
        var executor = new RecordingExecutor
        {
            Handler = (_, _, _) => throw new QziException(QziStatus.Busy, "device is busy")
        };
        using var queue = new QziGenerationQueue(executor);
        QziJob job = queue.Submit(Request());
        await queue.WaitForIdleAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(QziJobState.Error, job.State);
        Assert.Equal(QziStatus.Busy, job.ErrorStatus);
        Assert.Contains("device is busy", job.Error!);
        Assert.NotNull(job.FinishedAt);
    }

    [Fact]
    public async Task WatchAsync_Streams_State_Transitions()
    {
        var executor = new RecordingExecutor();
        using var queue = new QziGenerationQueue(executor);
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(5));
        IAsyncEnumerable<QziJob> watcher = queue.WatchAsync(cts.Token);

        List<QziJobState> states = [];
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);
        Task reader = Task.Run(async () =>
        {
            await foreach (QziJob job in watcher.WithCancellation(linked.Token))
            {
                states.Add(job.State);
                if (job.State == QziJobState.Done) break;
            }
        }, linked.Token);

        queue.Submit(Request());
        await reader.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Contains(QziJobState.Done, states);
    }
}
