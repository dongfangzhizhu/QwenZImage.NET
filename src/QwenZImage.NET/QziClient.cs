using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using QwenZImage.NET.Native;
using QwenZImage.NET.Validation;

namespace QwenZImage.NET;

/// <summary>
/// High-performance .NET 10 wrapper over the native qzi C ABI. One instance
/// owns one native engine (one GPU, one resident model). Generation calls are
/// synchronous at the native layer and single-flight; use
/// <see cref="QziGenerationQueue"/> to serialise many callers.
/// </summary>
/// <remarks>
/// The hot path uses source-generated P/Invoke with blittable structs, pinned
/// pooled UTF-8 buffers, static unmanaged callbacks and a single memcpy out of
/// native memory. No reflection-based marshalling, no JSON, no HTTP, no Go.
/// </remarks>
public sealed unsafe class QziClient : IDisposable, IQziExecutor
{
    private readonly QziEngineHandle _handle;
    private bool _disposed;

    /// <summary>Path of the native library actually loaded, when known.</summary>
    public string? NativeLibraryPath { get; }

    /// <summary>ABI version reported by the loaded native library.</summary>
    public uint AbiVersion => NativeMethods.AbiVersion();

    /// <summary>
    /// Creates a native engine instance. With no options,
    /// <c>QwenZImage.config.json</c> is loaded automatically and its model
    /// directories are passed straight into the native backend; an explicit
    /// <see cref="QziClientOptions"/> wins over the file.
    /// </summary>
    /// <exception cref="DllNotFoundException">No QwenZImageNative library could be resolved.</exception>
    /// <exception cref="QziException">The native library rejected the configuration.</exception>
    /// <exception cref="InvalidDataException">An invalid config file was found.</exception>
    public QziClient(QziClientOptions? options = null)
    {
        options ??= QziClientOptions.FromConfig();

        // The ABI carries the three core directories. QZI_CONFIG tells the
        // native backend which file to read for the fields the ABI does not
        // carry (Nunchaku packs, LoRA directory, platform overrides), so both
        // sides always agree on the same configuration.
        if (!string.IsNullOrWhiteSpace(options.ConfigPath))
            Environment.SetEnvironmentVariable(QziConfig.EnvironmentVariable,
                                               options.ConfigPath);

        NativeLibraryLoader.Configure(options.NativeLibraryPath);
        AbiGuard.Verify();

        using var qwenDir = new Utf8PinnedString(options.Qwen21Directory);
        using var zimageDir = new Utf8PinnedString(options.ZImageDirectory);
        using var cacheDir = new Utf8PinnedString(options.CacheDirectory);

        QziEngineConfig config = new()
        {
            AbiVersion = NativeMethods.ExpectedAbiVersion,
            Qwen21Dir = qwenDir.Pointer,
            ZimageDir = zimageDir.Pointer,
            CacheDir = cacheDir.Pointer,
            DeviceId = options.DeviceId,
            ThreadCount = options.ThreadCount,
            Flags = 0
        };

        byte* error = stackalloc byte[NativeMethods.ErrorCapacity];
        new Span<byte>(error, NativeMethods.ErrorCapacity).Clear();
        IntPtr raw = IntPtr.Zero;
        int status = NativeMethods.EngineCreate(&config, &raw, error,
                                                NativeMethods.ErrorCapacity);
        if (status != 0 || raw == IntPtr.Zero)
            throw NativeError.ToException(status, error);

        _handle = new QziEngineHandle(raw);
        NativeLibraryPath = NativeLibraryLoader.ResolvedPath;
    }

    // ---------------------------------------------------------------- public

    /// <summary>Reads the engine's health / counters snapshot.</summary>
    public EngineHealth GetHealth()
    {
        ThrowIfDisposed();
        QziHealth native = default;
        byte* error = stackalloc byte[NativeMethods.ErrorCapacity];
        new Span<byte>(error, NativeMethods.ErrorCapacity).Clear();
        int status = NativeMethods.EngineHealth(_handle.DangerousGetHandle(),
                                                &native, error,
                                                NativeMethods.ErrorCapacity);
        if (status != 0) throw NativeError.ToException(status, error);
        return new EngineHealth(
            (GenerationModel)native.ResidentModel,
            native.Busy != 0,
            native.QueueDepth,
            native.DeviceId,
            native.FreeVramMib,
            native.TotalVramMib,
            native.Served,
            native.Rejected,
            TimeSpan.FromSeconds(native.UptimeSeconds),
            native.ReferenceBackend != 0);
    }

    /// <summary>
    /// Releases re-creatable resident state (text tower / caches). The DiT stays
    /// resident because reloading it costs minutes. Refused while busy.
    /// </summary>
    public void Unload()
    {
        ThrowIfDisposed();
        byte* error = stackalloc byte[NativeMethods.ErrorCapacity];
        new Span<byte>(error, NativeMethods.ErrorCapacity).Clear();
        int status = NativeMethods.EngineUnload(_handle.DangerousGetHandle(),
                                                error, NativeMethods.ErrorCapacity);
        if (status != 0) throw NativeError.ToException(status, error);
    }

    /// <summary>Generates a Qwen-Image 2.1 PNG (synchronous, may take minutes).</summary>
    public GenerationResult GenerateQwen21(QwenImageOptions options,
                                           IProgress<GenerationProgress>? progress = null,
                                           CancellationToken cancellationToken = default)
    {
        QziImage image = CallQwen21(options, progress, cancellationToken, out TimeSpan elapsed);
        try
        {
            int length = checked((int)image.Size);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(length);
            new ReadOnlySpan<byte>(image.Data, length).CopyTo(buffer);
            return new GenerationResult(buffer, length, (int)image.Width,
                                        (int)image.Height, options.Seed,
                                        GenerationModel.Qwen21, elapsed);
        }
        finally
        {
            ReleaseImage(ref image);
        }
    }

    /// <summary>Generates a Z-Image PNG (synchronous, may take minutes).</summary>
    public GenerationResult GenerateZImage(ZImageOptions options,
                                           IProgress<GenerationProgress>? progress = null,
                                           CancellationToken cancellationToken = default)
    {
        QziImage image = CallZImage(options, progress, cancellationToken, out TimeSpan elapsed);
        try
        {
            int length = checked((int)image.Size);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(length);
            new ReadOnlySpan<byte>(image.Data, length).CopyTo(buffer);
            return new GenerationResult(buffer, length, (int)image.Width,
                                        (int)image.Height, options.Seed,
                                        GenerationModel.ZImage, elapsed);
        }
        finally
        {
            ReleaseImage(ref image);
        }
    }

    /// <summary>Dispatches either request kind.</summary>
    public GenerationResult Generate(GenerationOptions options,
                                     IProgress<GenerationProgress>? progress = null,
                                     CancellationToken cancellationToken = default)
        => options switch
        {
            QwenImageOptions qwen => GenerateQwen21(qwen, progress, cancellationToken),
            ZImageOptions zimage => GenerateZImage(zimage, progress, cancellationToken),
            null => throw new ArgumentNullException(nameof(options)),
            _ => throw new ArgumentException(
                $"Unsupported request type {options.GetType().FullName}.", nameof(options))
        };

    /// <summary>
    /// Zero-copy write path: streams the PNG straight out of native memory into
    /// <paramref name="destination"/> without materialising a managed copy.
    /// </summary>
    public void WriteTo(Stream destination, GenerationOptions options,
                        IProgress<GenerationProgress>? progress = null,
                        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination);
        QziImage image;
        switch (options)
        {
            case QwenImageOptions qwen:
                image = CallQwen21(qwen, progress, cancellationToken, out _);
                break;
            case ZImageOptions zimage:
                image = CallZImage(zimage, progress, cancellationToken, out _);
                break;
            case null:
                throw new ArgumentNullException(nameof(options));
            default:
                throw new ArgumentException(
                    $"Unsupported request type {options.GetType().FullName}.", nameof(options));
        }
        try
        {
            using var source = new UnmanagedMemoryStream(image.Data, checked((long)image.Size));
            source.CopyTo(destination, 64 * 1024);
        }
        finally
        {
            ReleaseImage(ref image);
        }
    }

    /// <summary>Async wrapper over the synchronous native call.</summary>
    public Task<GenerationResult> GenerateAsync(
        GenerationOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => Generate(options, progress, cancellationToken), cancellationToken);

    /// <summary>Async wrapper over the synchronous native call.</summary>
    public Task<GenerationResult> GenerateQwen21Async(
        QwenImageOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateQwen21(options, progress, cancellationToken), cancellationToken);

    /// <summary>Async wrapper over the synchronous native call.</summary>
    public Task<GenerationResult> GenerateZImageAsync(
        ZImageOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateZImage(options, progress, cancellationToken), cancellationToken);

    /// <summary>Async wrapper over the zero-copy <see cref="WriteTo"/> path.</summary>
    public Task WriteToAsync(Stream destination, GenerationOptions options,
                             IProgress<GenerationProgress>? progress = null,
                             CancellationToken cancellationToken = default)
        => Task.Run(() => WriteTo(destination, options, progress, cancellationToken),
                    cancellationToken);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    // --------------------------------------------------------------- private

    private QziImage CallQwen21(QwenImageOptions options,
                                IProgress<GenerationProgress>? progress,
                                CancellationToken cancellationToken,
                                out TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        using var prompt = new Utf8PinnedString(options.Prompt);
        using var negative = new Utf8PinnedString(options.NegativePrompt);
        using var tier = new Utf8PinnedString(options.TierName);
        using var lora = new Utf8PinnedString(options.Lora);
        using var callbacks = new CallbackScope(progress, cancellationToken);

        QziQwen21Request request = new()
        {
            Prompt = prompt.Pointer,
            NegativePrompt = negative.Pointer,
            Tier = tier.Pointer,
            Lora = lora.Pointer,
            Width = options.Width,
            Height = options.Height,
            Steps = options.EffectiveSteps,
            Guidance = options.EffectiveGuidance,
            LoraScale = options.EffectiveLoraScale,
            Seed = options.Seed,
            Flags = 0
        };

        return Invoke(&request, null, callbacks.Callbacks,
                      GenerationModel.Qwen21, cancellationToken, out elapsed);
    }

    private QziImage CallZImage(ZImageOptions options,
                                IProgress<GenerationProgress>? progress,
                                CancellationToken cancellationToken,
                                out TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        using var prompt = new Utf8PinnedString(options.Prompt);
        using var checkpoint = new Utf8PinnedString(options.Checkpoint);
        using var callbacks = new CallbackScope(progress, cancellationToken);

        QziZImageRequest request = new()
        {
            Prompt = prompt.Pointer,
            Checkpoint = checkpoint.Pointer,
            Width = options.Width,
            Height = options.Height,
            Seed = options.Seed,
            Flags = 0
        };

        return Invoke(null, &request, callbacks.Callbacks,
                      GenerationModel.ZImage, cancellationToken, out elapsed);
    }

    private QziImage Invoke(QziQwen21Request* qwen, QziZImageRequest* zimage,
                            QziCallbacks callbacks, GenerationModel model,
                            CancellationToken cancellationToken, out TimeSpan elapsed)
    {
        // Exactly one of the two request pointers is non-null.
        QziImage image = default;
        IntPtr engine = _handle.DangerousGetHandle();
        byte* error = stackalloc byte[NativeMethods.ErrorCapacity];
        new Span<byte>(error, NativeMethods.ErrorCapacity).Clear();

        bool added = false;
        _handle.DangerousAddRef(ref added);
        long start = Stopwatch.GetTimestamp();
        try
        {
            int status;
            if (model == GenerationModel.Qwen21)
                status = NativeMethods.GenerateQwen21(engine, qwen, &callbacks, &image,
                                                      error, NativeMethods.ErrorCapacity);
            else
                status = NativeMethods.GenerateZImage(engine, zimage, &callbacks, &image,
                                                      error, NativeMethods.ErrorCapacity);

            elapsed = Stopwatch.GetElapsedTime(start);
            if (status != 0)
            {
                if (image.Data is not null) NativeMethods.ImageRelease(engine, &image);
                if (status == (int)QziStatus.Cancelled && cancellationToken.IsCancellationRequested)
                    throw new OperationCanceledException(cancellationToken);
                throw NativeError.ToException(status, error);
            }
            return image;
        }
        finally
        {
            if (added) _handle.DangerousRelease();
        }
    }

    private void ReleaseImage(ref QziImage image)
    {
        if (image.Data is null) return;
        bool added = false;
        _handle.DangerousAddRef(ref added);
        try
        {
            fixed (QziImage* pointer = &image)
            {
                NativeMethods.ImageRelease(_handle.DangerousGetHandle(), pointer);
            }
        }
        finally
        {
            if (added) _handle.DangerousRelease();
        }
        image = default;
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(_disposed, this);
}
