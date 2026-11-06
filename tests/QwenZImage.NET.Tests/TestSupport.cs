using System.Buffers;
using System.Collections.Concurrent;

namespace QwenZImage.NET.Tests;

internal static class NativeTestLibrary
{
    /// <summary>
    /// Locates the reference qzi library built into artifacts/native by CMake.
    /// CI builds it before dotnet test (see .github/workflows/ci.yml); locally
    /// run scripts/build-native.sh or scripts/build-native.ps1 first.
    /// </summary>
    internal static QziClientOptions Options()
    {
        string? path = Find();
        if (path is null)
            throw new FileNotFoundException(
                "qzi native library not found. Build it first: " +
                "cmake -S native -B native/build -G Ninja && cmake --build native/build");
        return new QziClientOptions
        {
            NativeLibraryPath = path,
            Qwen21Directory = ".",
            ZImageDirectory = ".",
            CacheDirectory = "."
        };
    }

    internal static bool IsAvailable => Find() is not null;

    private static string? Find()
    {
        string name = OperatingSystem.IsWindows() ? "qzi.dll"
                    : OperatingSystem.IsMacOS() ? "libqzi.dylib" : "libqzi.so";
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            // Prefer the staged NuGet RID layout: CI downloads only that tree,
            // and it is exactly what ends up in the .nupkg.
            foreach (string tree in new[]
                     {
                         Path.Combine(dir.FullName, "artifacts", "runtimes"),
                         Path.Combine(dir.FullName, "artifacts", "native")
                     })
            {
                if (!Directory.Exists(tree)) continue;
                string? found = Directory.EnumerateFiles(tree, name, SearchOption.AllDirectories)
                                         .FirstOrDefault();
                if (found is not null) return found;
            }
        }
        string? env = Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY");
        return !string.IsNullOrEmpty(env) && File.Exists(env) ? env : null;
    }
}

/// <summary>Deterministic IQziExecutor used by the queue tests (no GPU needed).</summary>
internal sealed class RecordingExecutor : IQziExecutor
{
    private int _running;
    private int _maxConcurrency;

    public ConcurrentBag<GenerationOptions> Calls { get; } = new();
    public int MaxConcurrency => Volatile.Read(ref _maxConcurrency);
    public Func<GenerationOptions, IProgress<GenerationProgress>?, CancellationToken, GenerationResult>? Handler { get; set; }

    public GenerationResult Generate(GenerationOptions options,
                                     IProgress<GenerationProgress>? progress = null,
                                     CancellationToken cancellationToken = default)
    {
        int now = Interlocked.Increment(ref _running);
        int observed;
        do
        {
            observed = Volatile.Read(ref _maxConcurrency);
            if (now <= observed) break;
        } while (Interlocked.CompareExchange(ref _maxConcurrency, now, observed) != observed);

        try
        {
            Calls.Add(options);
            progress?.Report(new GenerationProgress(GenerationPhase.Dit, 1, 8, TimeSpan.FromMilliseconds(1)));
            cancellationToken.ThrowIfCancellationRequested();
            return Handler is not null
                ? Handler(options, progress, cancellationToken)
                : MakeResult(options);
        }
        finally
        {
            Interlocked.Decrement(ref _running);
        }
    }

    internal static GenerationResult MakeResult(GenerationOptions options, byte fill = 0x5A)
    {
        byte[] buffer = ArrayPool<byte>.Shared.Rent(64);
        Array.Fill(buffer, fill, 0, 64);
        buffer[0] = 137; buffer[1] = 80; buffer[2] = 78; buffer[3] = 71;
        return new GenerationResult(buffer, 64, options.Width, options.Height,
                                    options.Seed, options.Model, TimeSpan.FromMilliseconds(1));
    }
}
