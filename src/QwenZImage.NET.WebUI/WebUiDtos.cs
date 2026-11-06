using System.Text.Json.Serialization;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>Browser-submitted generation request (JSON body of POST /api/jobs).</summary>
public sealed class GenerateRequest
{
    public string? Model { get; set; }
    public string? Prompt { get; set; }
    public int Width { get; set; } = 1024;
    public int Height { get; set; } = 1024;
    public ulong Seed { get; set; } = 42;
    public string? Tier { get; set; }
    public int? Steps { get; set; }
    public float? Guidance { get; set; }
    public string? NegativePrompt { get; set; }
    public string? Lora { get; set; }
    public float? LoraScale { get; set; }
    public string? Checkpoint { get; set; }
    public string? Client { get; set; }
}

/// <summary>Batch delete / restore body.</summary>
public sealed class NamesRequest
{
    public List<string> Names { get; set; } = [];
}

/// <summary>Flat, JSON-safe view of a <see cref="QziJob"/> (no request object, no CTS).</summary>
public sealed class JobDto
{
    public long Id { get; init; }
    public string State { get; init; } = "queued";
    public string Model { get; init; } = "none";
    public string Prompt { get; init; } = "";
    public int Width { get; init; }
    public int Height { get; init; }
    public ulong Seed { get; init; }
    public string? Error { get; init; }
    public string? ErrorStatus { get; init; }
    public int? OutputLength { get; init; }
    public string? OutputName { get; init; }
    public string? OutputUrl { get; init; }
    public string? Phase { get; init; }
    public int ProgressCompleted { get; init; }
    public int ProgressTotal { get; init; }
    public double ProgressFraction { get; init; }
    public double ElapsedMs { get; init; }
    public DateTimeOffset CreatedAt { get; init; }
    public DateTimeOffset? StartedAt { get; init; }
    public DateTimeOffset? FinishedAt { get; init; }
    public long? RetryOf { get; init; }
    public string? Client { get; init; }

    public static JobDto From(QziJob job, WebUiRuntime runtime)
    {
        GenerationOptions request = job.Request;
        string? output = runtime.OutputNameFor(job.Id);
        GenerationProgress p = job.LastProgress;
        return new JobDto
        {
            Id = job.Id,
            State = job.State.ToString().ToLowerInvariant(),
            Model = request.Model switch
            {
                GenerationModel.Qwen21 => "qwen21",
                GenerationModel.ZImage => "zimage",
                _ => "none"
            },
            Prompt = request.Prompt,
            Width = request.Width,
            Height = request.Height,
            Seed = request.Seed,
            Error = job.Error,
            ErrorStatus = job.ErrorStatus?.ToString(),
            OutputLength = job.OutputLength,
            OutputName = output,
            OutputUrl = output is null ? null : "/api/outputs/" + Uri.EscapeDataString(output),
            Phase = p.Total > 0 ? p.Phase.ToString() : null,
            ProgressCompleted = p.Completed,
            ProgressTotal = p.Total,
            ProgressFraction = p.Fraction,
            ElapsedMs = job.Elapsed.TotalMilliseconds,
            CreatedAt = job.CreatedAt,
            StartedAt = job.StartedAt,
            FinishedAt = job.FinishedAt,
            RetryOf = job.RetryOf,
            Client = runtime.ClientFor(job.Id)
        };
    }
}

/// <summary>A preset as the UI needs it: flat values, no polymorphic request object.</summary>
public sealed class PresetDto
{
    public string Label { get; init; } = "";
    public string Model { get; init; } = "";
    public int Width { get; init; }
    public int Height { get; init; }
    public string? Tier { get; init; }
    public int? Steps { get; init; }
    public float? Guidance { get; init; }
    public string? Lora { get; init; }
    public float? LoraScale { get; init; }
    public string? Checkpoint { get; init; }

    public static PresetDto From(GenerationPreset preset)
        => preset.Options switch
        {
            QwenImageOptions q => new PresetDto
            {
                Label = preset.Label, Model = "qwen21", Width = q.Width, Height = q.Height,
                Tier = q.TierName, Steps = q.Steps, Guidance = q.Guidance,
                Lora = q.Lora, LoraScale = q.LoraScale
            },
            ZImageOptions z => new PresetDto
            {
                Label = preset.Label, Model = "zimage", Width = z.Width, Height = z.Height,
                Checkpoint = z.Checkpoint
            },
            _ => throw new ArgumentOutOfRangeException(nameof(preset))
        };
}

/// <summary>One file in the output gallery.</summary>
public sealed class OutputEntryDto
{
    public string Name { get; init; } = "";
    public long Size { get; init; }
    public DateTimeOffset ModTime { get; init; }
    public string Url { get; init; } = "";
    public string ContentType { get; init; } = "application/octet-stream";
}

/// <summary>Common error envelope.</summary>
public sealed record ErrorResponse(string Error, [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Detail = null);
