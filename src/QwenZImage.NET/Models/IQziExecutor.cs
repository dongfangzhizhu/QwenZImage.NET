namespace QwenZImage.NET;

/// <summary>
/// The minimal synchronous generation surface. Implemented by
/// <see cref="QziClient"/> and injectable into <see cref="QziGenerationQueue"/>
/// so callers can unit-test job scheduling without a GPU or a native library.
/// </summary>
public interface IQziExecutor
{
    /// <summary>Runs one generation synchronously (may take minutes).</summary>
    GenerationResult Generate(GenerationOptions options,
                              IProgress<GenerationProgress>? progress = null,
                              CancellationToken cancellationToken = default);
}
