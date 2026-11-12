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

    // Guards the process-wide QZI_* variables while one engine is being created.
    private static readonly object EngineCreateGate = new();

    /// <summary>Path of the native library actually loaded, when known.</summary>
    public string? NativeLibraryPath { get; }

    /// <summary>ABI version reported by the loaded native library.</summary>
    public uint AbiVersion => NativeMethods.AbiVersion();

    /// <summary>
    /// Whether the loaded native library implements Qwen-Image 2.1 image
    /// editing. Editing is additive in ABI v1: an older library keeps serving
    /// text-to-image but does not export qzi_generate_qwen21_edit(), so the
    /// wrapper probes the exports instead of assuming the feature exists.
    /// </summary>
    public bool SupportsImageEditing => AbiGuard.Qwen21EditSupported;

    /// <summary>
    /// Whether the loaded native library implements Z-Image Turbo Fun ControlNet
    /// Union. Like editing this is additive in ABI v1: an older library keeps
    /// serving text-to-image but does not export qzi_generate_zimage_control(),
    /// so the wrapper probes the exports instead of assuming the feature exists.
    /// </summary>
    public bool SupportsControlNet => AbiGuard.ZImageControlSupported;

    /// <summary>
    /// Whether the loaded native library implements the third additive ABI v1
    /// increment: advanced Qwen-Image 2.1 editing (Red Canvas outpainting,
    /// LanPaint model modes and RGBA output). Older libraries keep serving
    /// ordinary image editing and text-to-image; the wrapper probes the exports
    /// instead of assuming the feature exists.
    /// </summary>
    public bool SupportsAdvancedEditing => AbiGuard.Qwen21EditAdvSupported;

    /// <summary>
    /// Whether the loaded native library implements the fourth additive ABI v1
    /// increment: Z-Image Union 2.x mask inpainting. Inpainting embeds the
    /// ControlNet request, so it is available only when
    /// <see cref="SupportsControlNet"/> is also true; an older v1 library that
    /// does not export qzi_generate_zimage_inpaint() keeps serving ControlNet
    /// and text-to-image generation.
    /// </summary>
    public bool SupportsInpaint => AbiGuard.ZImageInpaintSupported;

    /// <summary>
    /// Creates a native engine instance. With no options,
    /// <c>QwenZImage.config.json</c> is loaded automatically and its model
    /// directories are passed straight into the native backend; an explicit
    /// <see cref="QziClientOptions"/> wins over the file.
    /// </summary>
    /// <remarks>
    /// The three <c>QZI_*</c> variables carrying fields the C ABI does not have
    /// are scoped to this single <c>qzi_engine_create</c> call and restored
    /// immediately afterwards. Each engine therefore snapshots its own LoRA /
    /// Nunchaku directories and config path, so clients with different values
    /// can coexist in one process without clobbering one another.
    /// </remarks>
    /// <exception cref="DllNotFoundException">No QwenZImageNative library could be resolved.</exception>
    /// <exception cref="QziException">The native library rejected the configuration.</exception>
    /// <exception cref="InvalidDataException">An invalid config file was found.</exception>
    public QziClient(QziClientOptions? options = null)
    {
        // QZI_* variables are process-wide while qzi_engine_create() runs and
        // its backend init() snapshots them. Serialise construction so those
        // scoped values cannot interleave with a concurrent client (including
        // the FromConfig() lookup, which reads the same variables).
        lock (EngineCreateGate)
        {
            options ??= QziClientOptions.FromConfig();
            using var scope = new NativeConfigScope(options);

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

    /// <summary>
    /// Edits a reference image with Qwen-Image 2.1 (synchronous, may take
    /// minutes). Requires a native library that implements the additive ABI v1
    /// editing surface; see <see cref="SupportsImageEditing"/>.
    /// </summary>
    public GenerationResult GenerateQwen21Edit(Qwen21EditOptions options,
                                               IProgress<GenerationProgress>? progress = null,
                                               CancellationToken cancellationToken = default)
    {
        QziImage image = CallQwen21Edit(options, progress, cancellationToken,
                                        out TimeSpan elapsed);
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

    /// <summary>
    /// Advanced Qwen-Image 2.1 editing: Red Canvas outpainting, LanPaint model
    /// modes and/or RGBA output (synchronous, may take minutes). Requires a
    /// native library implementing the third additive ABI v1 increment; see
    /// <see cref="SupportsAdvancedEditing"/>. With all advanced knobs at their
    /// defaults the sampling schedule matches <see cref="GenerateQwen21Edit"/>;
    /// the reference backend still tags the advanced call so both paths stay
    /// distinguishable in GPU-less tests.
    /// </summary>
    public GenerationResult GenerateQwen21EditEx(Qwen21EditExOptions options,
                                                  IProgress<GenerationProgress>? progress = null,
                                                  CancellationToken cancellationToken = default)
    {
        QziImage image = CallQwen21EditEx(options, progress, cancellationToken,
                                          out TimeSpan elapsed);
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

    /// <summary>
    /// Generates a Z-Image Turbo Fun ControlNet Union PNG (synchronous, may take
    /// minutes). Requires a native library that implements the additive ABI v1
    /// ControlNet surface; see <see cref="SupportsControlNet"/>.
    /// </summary>
    public GenerationResult GenerateZImageControl(ZImageControlOptions options,
                                                   IProgress<GenerationProgress>? progress = null,
                                                   CancellationToken cancellationToken = default)
    {
        QziImage image = CallZImageControl(options, progress, cancellationToken,
                                            out TimeSpan elapsed);
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

    /// <summary>
    /// Generates a Z-Image Union 2.x inpaint PNG (masked regeneration on the
    /// ControlNet Union weights; synchronous, may take minutes). WHITE (255)
    /// mask pixels are repainted from the prompt, BLACK (0) pixels keep the
    /// control image. Requires a native library that implements the fourth
    /// additive ABI v1 increment; see <see cref="SupportsInpaint"/>. Only Union
    /// v2 / v2.1 weights are supported by the real backend.
    /// </summary>
    public GenerationResult GenerateZImageInpaint(ZImageInpaintOptions options,
                                                   IProgress<GenerationProgress>? progress = null,
                                                   CancellationToken cancellationToken = default)
    {
        QziImage image = CallZImageInpaint(options, progress, cancellationToken,
                                            out TimeSpan elapsed);
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
            // Derived options must be matched before their base record.
            Qwen21EditExOptions advanced => GenerateQwen21EditEx(advanced, progress, cancellationToken),
            Qwen21EditOptions edit => GenerateQwen21Edit(edit, progress, cancellationToken),
            QwenImageOptions qwen => GenerateQwen21(qwen, progress, cancellationToken),
            // Inpaint derives from the ControlNet options, so it must come first.
            ZImageInpaintOptions inpaint => GenerateZImageInpaint(inpaint, progress, cancellationToken),
            ZImageControlOptions control => GenerateZImageControl(control, progress, cancellationToken),
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
            case Qwen21EditExOptions advanced:
                image = CallQwen21EditEx(advanced, progress, cancellationToken, out _);
                break;
            case Qwen21EditOptions edit:
                image = CallQwen21Edit(edit, progress, cancellationToken, out _);
                break;
            case QwenImageOptions qwen:
                image = CallQwen21(qwen, progress, cancellationToken, out _);
                break;
            case ZImageInpaintOptions inpaint:
                image = CallZImageInpaint(inpaint, progress, cancellationToken, out _);
                break;
            case ZImageControlOptions control:
                image = CallZImageControl(control, progress, cancellationToken, out _);
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

    /// <summary>Async wrapper over the synchronous image-edit call.</summary>
    public Task<GenerationResult> GenerateQwen21EditAsync(
        Qwen21EditOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateQwen21Edit(options, progress, cancellationToken),
                    cancellationToken);

    /// <summary>Async wrapper over the synchronous advanced image-edit call.</summary>
    public Task<GenerationResult> GenerateQwen21EditExAsync(
        Qwen21EditExOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateQwen21EditEx(options, progress, cancellationToken),
                    cancellationToken);

    /// <summary>Async wrapper over the synchronous native call.</summary>
    public Task<GenerationResult> GenerateZImageAsync(
        ZImageOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateZImage(options, progress, cancellationToken), cancellationToken);

    /// <summary>Async wrapper over the synchronous ControlNet call.</summary>
    public Task<GenerationResult> GenerateZImageControlAsync(
        ZImageControlOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateZImageControl(options, progress, cancellationToken),
                    cancellationToken);

    /// <summary>Async wrapper over the synchronous Z-Image inpaint call.</summary>
    public Task<GenerationResult> GenerateZImageInpaintAsync(
        ZImageInpaintOptions options,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateZImageInpaint(options, progress, cancellationToken),
                    cancellationToken);

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

        QziQwen21Request request = BuildQwen21Request(options, prompt, negative,
                                                       tier, lora);

        return Invoke(&request, null, null, null, null, null, callbacks,
                      cancellationToken, out elapsed);
    }

    private QziImage CallQwen21Edit(Qwen21EditOptions options,
                                    IProgress<GenerationProgress>? progress,
                                    CancellationToken cancellationToken,
                                    out TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ThrowIfDisposed();
        if (!AbiGuard.Qwen21EditSupported)
            throw new QziException(QziStatus.Unsupported,
                "The loaded native library does not implement Qwen-Image 2.1 image " +
                "editing (qzi_generate_qwen21_edit is not exported). Rebuild " +
                "QwenZImageNative from the matching source tree; text-to-image " +
                "generation is unaffected.");
        cancellationToken.ThrowIfCancellationRequested();

        using var prompt = new Utf8PinnedString(options.Prompt);
        using var negative = new Utf8PinnedString(options.NegativePrompt);
        using var tier = new Utf8PinnedString(options.TierName);
        using var lora = new Utf8PinnedString(options.Lora);
        using var callbacks = new CallbackScope(progress, cancellationToken);

        QziQwen21Request baseRequest = BuildQwen21Request(options, prompt, negative,
                                                           tier, lora);
        QziRasterImage reference = options.Reference;
        QziRasterImage? mask = options.Mask;
        // A non-null, non-empty stand-in keeps `fixed` well defined when there
        // is no mask; the pointer itself is never passed in that case.
        byte[] maskBytes = mask?.Bytes ?? [0];
        fixed (byte* referencePtr = reference.Bytes)
        fixed (byte* maskPtr = maskBytes)
        {
            QziQwen21EditRequest request = new()
            {
                Base = baseRequest,
                ReferenceRgb = referencePtr,
                ReferenceWidth = (uint)reference.Width,
                ReferenceHeight = (uint)reference.Height,
                Mask = mask is null ? null : maskPtr,
                Strength = options.EffectiveStrength,
                MaskingStrength = options.EffectiveMaskingStrength,
                Flags = 0
            };
            // The native call is synchronous, so both pins stay alive until
            // Invoke returns and the caller owns the resulting PNG buffer.
            return Invoke(null, null, &request, null, null, null, callbacks,
                          cancellationToken, out elapsed);
        }
    }

    private QziImage CallQwen21EditEx(Qwen21EditExOptions options,
                                      IProgress<GenerationProgress>? progress,
                                      CancellationToken cancellationToken,
                                      out TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ThrowIfDisposed();
        if (!AbiGuard.Qwen21EditAdvSupported)
            throw new QziException(QziStatus.Unsupported,
                "The loaded native library does not implement advanced Qwen-Image 2.1 " +
                "editing (qzi_generate_qwen21_edit_ex is not exported). Rebuild " +
                "QwenZImageNative from the matching source tree; ordinary image " +
                "editing and text-to-image generation are unaffected.");
        cancellationToken.ThrowIfCancellationRequested();

        using var prompt = new Utf8PinnedString(options.Prompt);
        using var negative = new Utf8PinnedString(options.NegativePrompt);
        using var tier = new Utf8PinnedString(options.TierName);
        using var lora = new Utf8PinnedString(options.Lora);
        using var callbacks = new CallbackScope(progress, cancellationToken);

        QziQwen21Request baseRequest = BuildQwen21Request(options, prompt, negative,
                                                           tier, lora);
        QziRasterImage reference = options.Reference;
        QziRasterImage? mask = options.Mask;
        // A non-null, non-empty stand-in keeps `fixed` well defined when there
        // is no mask; the pointer itself is never passed in that case.
        byte[] maskBytes = mask?.Bytes ?? [0];
        // The native layer normalises LanPaint to full strength; sending the
        // same values keeps the managed and C views of the request identical.
        float strength = options.IsLanPaint ? 1.0f : options.EffectiveStrength;
        float maskingStrength = options.IsLanPaint ? 1.0f
                                                   : options.EffectiveMaskingStrength;
        fixed (byte* referencePtr = reference.Bytes)
        fixed (byte* maskPtr = maskBytes)
        {
            QziQwen21EditRequest editRequest = new()
            {
                Base = baseRequest,
                ReferenceRgb = referencePtr,
                ReferenceWidth = (uint)reference.Width,
                ReferenceHeight = (uint)reference.Height,
                Mask = mask is null ? null : maskPtr,
                Strength = strength,
                MaskingStrength = maskingStrength,
                Flags = 0
            };
            QziQwen21EditExRequest request = new()
            {
                Base = editRequest,
                MarginTop = options.MarginTop,
                MarginBottom = options.MarginBottom,
                MarginLeft = options.MarginLeft,
                MarginRight = options.MarginRight,
                ModelMode = options.EffectiveModelMode,
                Rgba = options.Rgba ? 1u : 0u,
                Flags = options.Flags
            };
            // The native call is synchronous, so both pins stay alive until
            // Invoke returns and the caller owns the resulting PNG buffer.
            return Invoke(null, null, null, null, &request, null, callbacks,
                          cancellationToken, out elapsed);
        }
    }

    private static QziQwen21Request BuildQwen21Request(
        QwenImageOptions options,
        Utf8PinnedString prompt, Utf8PinnedString negative,
        Utf8PinnedString tier, Utf8PinnedString lora)
        => new()
        {
            Prompt = prompt.Pointer,
            NegativePrompt = negative.Pointer,
            Tier = tier.Pointer,
            Lora = lora.Pointer,
            Width = options.Width,
            Height = options.Height,
            Steps = options.EffectiveSteps,
            Guidance = options.EffectiveGuidance,
            // 0.0f is the ABI sentinel for "no adapter, default strength".
            // Sending 1.0f without a lora would be rejected as a silently
            // dropped parameter by qzi_validate_qwen21().
            LoraScale = string.IsNullOrEmpty(options.Lora) ? 0.0f : options.EffectiveLoraScale,
            Seed = options.Seed,
            Flags = 0
        };

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

        return Invoke(null, &request, null, null, null, null, callbacks,
                      cancellationToken, out elapsed);
    }

    private QziImage CallZImageControl(ZImageControlOptions options,
                                       IProgress<GenerationProgress>? progress,
                                       CancellationToken cancellationToken,
                                       out TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ThrowIfDisposed();
        if (!AbiGuard.ZImageControlSupported)
            throw new QziException(QziStatus.Unsupported,
                "The loaded native library does not implement Z-Image Turbo Fun " +
                "ControlNet Union (qzi_generate_zimage_control is not exported). " +
                "Rebuild QwenZImageNative from the matching source tree; text-to-image " +
                "generation is unaffected.");
        cancellationToken.ThrowIfCancellationRequested();

        using var prompt = new Utf8PinnedString(options.Prompt);
        using var checkpoint = new Utf8PinnedString(options.Checkpoint);
        // null ControlType becomes a NULL pointer; a non-null one is already
        // the lowercase canonical value produced by EffectiveControlType.
        using var controlType = new Utf8PinnedString(options.EffectiveControlType);
        using var callbacks = new CallbackScope(progress, cancellationToken);

        QziZImageRequest baseRequest = new()
        {
            Prompt = prompt.Pointer,
            Checkpoint = checkpoint.Pointer,
            Width = options.Width,
            Height = options.Height,
            Seed = options.Seed,
            Flags = 0
        };
        QziRasterImage control = options.ControlImage;
        fixed (byte* controlPtr = control.Bytes)
        {
            QziZImageControlRequest request = new()
            {
                Base = baseRequest,
                ControlRgb = controlPtr,
                ControlWidth = (uint)control.Width,
                ControlHeight = (uint)control.Height,
                ControlType = controlType.Pointer,
                ControlScale = options.ControlScale,
                Flags = options.Flags
            };
            // The native call is synchronous, so the pin stays alive until
            // Invoke returns and the caller owns the resulting PNG buffer.
            return Invoke(null, null, null, &request, null, null, callbacks,
                          cancellationToken, out elapsed);
        }
    }

    private QziImage CallZImageInpaint(ZImageInpaintOptions options,
                                       IProgress<GenerationProgress>? progress,
                                       CancellationToken cancellationToken,
                                       out TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(options);
        RequestValidator.Validate(options);
        ThrowIfDisposed();
        if (!AbiGuard.ZImageInpaintSupported)
            throw new QziException(QziStatus.Unsupported,
                "The loaded native library does not implement Z-Image Union 2.x " +
                "inpainting (qzi_generate_zimage_inpaint is not exported, or the " +
                "ControlNet capability is missing). Rebuild QwenZImageNative from " +
                "the matching source tree; ControlNet and text-to-image generation " +
                "are unaffected.");
        cancellationToken.ThrowIfCancellationRequested();

        using var prompt = new Utf8PinnedString(options.Prompt);
        using var checkpoint = new Utf8PinnedString(options.Checkpoint);
        // null ControlType becomes a NULL pointer; a non-null one is already
        // the lowercase canonical value produced by EffectiveControlType.
        using var controlType = new Utf8PinnedString(options.EffectiveControlType);
        using var callbacks = new CallbackScope(progress, cancellationToken);

        QziZImageRequest baseRequest = new()
        {
            Prompt = prompt.Pointer,
            Checkpoint = checkpoint.Pointer,
            Width = options.Width,
            Height = options.Height,
            Seed = options.Seed,
            Flags = 0
        };
        QziRasterImage control = options.ControlImage;
        QziRasterImage mask = options.Mask;
        // Both buffers are pinned exactly like the reference/mask pair in
        // CallQwen21Edit: the native call is synchronous, so the pins stay
        // alive until Invoke returns and the caller owns the PNG buffer.
        fixed (byte* controlPtr = control.Bytes)
        fixed (byte* maskPtr = mask.Bytes)
        {
            QziZImageControlRequest controlRequest = new()
            {
                Base = baseRequest,
                ControlRgb = controlPtr,
                ControlWidth = (uint)control.Width,
                ControlHeight = (uint)control.Height,
                ControlType = controlType.Pointer,
                ControlScale = options.ControlScale,
                Flags = options.Flags
            };
            QziZImageInpaintRequest request = new()
            {
                Base = controlRequest,
                Mask = (IntPtr)maskPtr,
                MaskWidth = (uint)mask.Width,
                MaskHeight = (uint)mask.Height,
                // Reserved for the C ABI; the reusable control flags above
                // carry options.Flags and the validator keeps both at zero.
                Flags = 0
            };
            return Invoke(null, null, null, null, null, &request, callbacks,
                          cancellationToken, out elapsed);
        }
    }

    private QziImage Invoke(QziQwen21Request* qwen, QziZImageRequest* zimage,
                            QziQwen21EditRequest* edit, QziZImageControlRequest* control,
                            QziQwen21EditExRequest* editEx, QziZImageInpaintRequest* inpaint,
                            CallbackScope scope,
                            CancellationToken cancellationToken, out TimeSpan elapsed)
    {
        // Exactly one of the six request pointers is non-null. The callbacks
        // struct is a local copy whose address stays valid for the whole
        // synchronous call; the owning scope keeps the live-token entry alive.
        QziCallbacks callbacks = scope.Callbacks;
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
            if (editEx is not null)
                status = NativeMethods.GenerateQwen21EditEx(engine, editEx, &callbacks, &image,
                                                            error, NativeMethods.ErrorCapacity);
            else if (edit is not null)
                status = NativeMethods.GenerateQwen21Edit(engine, edit, &callbacks, &image,
                                                          error, NativeMethods.ErrorCapacity);
            else if (inpaint is not null)
                status = NativeMethods.GenerateZImageInpaint(engine, inpaint, &callbacks, &image,
                                                             error, NativeMethods.ErrorCapacity);
            else if (control is not null)
                status = NativeMethods.GenerateZImageControl(engine, control, &callbacks, &image,
                                                             error, NativeMethods.ErrorCapacity);
            else if (qwen is not null)
                status = NativeMethods.GenerateQwen21(engine, qwen, &callbacks, &image,
                                                      error, NativeMethods.ErrorCapacity);
            else
                status = NativeMethods.GenerateZImage(engine, zimage, &callbacks, &image,
                                                      error, NativeMethods.ErrorCapacity);

            elapsed = Stopwatch.GetElapsedTime(start);

            // A progress / cancellation callback failed while native frames
            // were on the stack. CallbackScope contained it at the unmanaged
            // boundary and asked the engine to stop; rethrow here, where a
            // normal managed catch (or the WebUI job queue) can report it.
            // Letting such an exception unwind through CUDA / ggml frames is
            // what produced the fatal System.ExecutionEngineException
            // (HResult 0x80131506) at this call site.
            if (scope.TakeFault() is { } fault)
            {
                if (image.Data is not null) NativeMethods.ImageRelease(engine, &image);
                throw new InvalidOperationException(
                    "A generation progress callback threw an exception while the native " +
                    "engine was running. The failure was contained at the unmanaged " +
                    "callback boundary and the generation was stopped. Inspect the inner " +
                    "exception (typically an IProgress<T>.Report handler or an SSE " +
                    "subscriber) for the real cause.", fault);
            }

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
