namespace QwenZImage.NET;

/// <summary>
/// One progress snapshot from the native generation thread. Delivered as a
/// value type with no string allocation: phase names are compared as UTF-8
/// spans in the unmanaged callback.
/// </summary>
public readonly record struct GenerationProgress(
    GenerationPhase Phase,
    int Completed,
    int Total,
    TimeSpan Elapsed)
{
    /// <summary>Fraction of the current stage, in [0, 1].</summary>
    public double Fraction => Total > 0 ? Math.Clamp((double)Completed / Total, 0.0, 1.0) : 0.0;

    /// <inheritdoc/>
    public override string ToString()
        => $"{Phase} {Completed}/{Total} ({Elapsed.TotalSeconds:F2}s)";
}

/// <summary>Native engine state snapshot.</summary>
public readonly record struct EngineHealth(
    GenerationModel ResidentModel,
    bool Busy,
    int QueueDepth,
    int DeviceId,
    long FreeVramMib,
    long TotalVramMib,
    ulong Served,
    ulong Rejected,
    TimeSpan Uptime,
    bool IsReferenceBackend);

/// <summary>Model families supported by this package.</summary>
public enum GenerationModel
{
    /// <summary>No model is resident.</summary>
    None = 0,
    /// <summary>Qwen-Image 2.1.</summary>
    Qwen21 = 1,
    /// <summary>Z-Image.</summary>
    ZImage = 2
}
