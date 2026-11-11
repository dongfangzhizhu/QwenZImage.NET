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

    /// <summary>
    /// Qwen-Image 2.1 image editing only: a base64 PNG (optionally a
    /// <c>data:image/png;base64,...</c> URL) carrying the reference image.
    /// The browser canvas re-encodes uploads so JPEG / WebP also work.
    /// </summary>
    public string? ReferenceImage { get; set; }

    /// <summary>
    /// Optional single-channel brush mask at the reference resolution. White
    /// pixels are regenerated, black pixels follow the reference trajectory.
    /// Reused by Z-Image Union 2.x inpaint (white = repaint, black = keep),
    /// where it is required and paired with <see cref="ControlImage"/>.
    /// </summary>
    public string? MaskImage { get; set; }

    /// <summary>Denoising strength 0.05..1.0 (default 0.75).</summary>
    public float? Strength { get; set; }

    /// <summary>
    /// Fraction of the denoise steps during which unmasked pixels are restored
    /// from the reference. Requires <see cref="MaskImage"/>.
    /// </summary>
    public float? MaskingStrength { get; set; }

    /// <summary>
    /// Advanced WanGP editing only. These four margins enlarge the OUTPUT
    /// canvas (the reference is cover-fitted into the remaining inner
    /// rectangle); each must be a multiple of 32 and all zero means "no
    /// outpainting". Sending any advanced field upgrades a qwen21-edit job to
    /// the third additive ABI v1 increment.
    /// </summary>
    public int? MarginTop { get; set; }
    public int? MarginBottom { get; set; }
    public int? MarginLeft { get; set; }
    public int? MarginRight { get; set; }

    /// <summary>
    /// WanGP model mode: 0 = masked denoising, 2/3/4/5 = LanPaint with
    /// 2/5/10/15 inner iterations. LanPaint requires a mask and is mutually
    /// exclusive with outpainting.
    /// </summary>
    public int? ModelMode { get; set; }

    /// <summary>1 writes an RGBA PNG using the VAE's generated alpha; 0 (default) writes RGB.</summary>
    public int? Rgba { get; set; }

    /// <summary>
    /// Z-Image Turbo Fun ControlNet Union only: a base64 PNG (optionally a
    /// <c>data:image/png;base64,...</c> URL) carrying the preprocessed control
    /// image. The browser canvas re-encodes uploads so JPEG / WebP also work.
    /// This library does NOT run edge / pose / depth estimation: the caller
    /// supplies an already-preprocessed control image ("raw" semantics).
    /// </summary>
    public string? ControlImage { get; set; }

    /// <summary>
    /// Descriptive control type: <c>auto</c> / <c>pose</c> / <c>canny</c> /
    /// <c>depth</c> / <c>scribble</c> / <c>raw</c> (case-insensitive).
    /// <c>raw</c> means the control image is already in Z-Image control format.
    /// </summary>
    public string? ControlType { get; set; }

    /// <summary>Control context scale. 0 uses the engine default 0.65, else 0.05..2.0.</summary>
    public float? ControlScale { get; set; }
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
    public bool Edit { get; init; }
    public bool HasMask { get; init; }
    public bool AdvancedEdit { get; init; }
    public bool Outpaint { get; init; }
    public bool Rgba { get; init; }
    public bool Control { get; init; }
    public bool HasControl { get; init; }
    /// <summary>Z-Image Union 2.x inpaint job (a control request with a mask).</summary>
    public bool Inpaint { get; init; }
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
        (bool control, bool hasControl) = ControlState(request);
        (bool advanced, bool outpaint, bool rgba) = AdvancedEditState(request);
        (bool inpaint, bool hasInpaintMask) = InpaintState(request);
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
            // Qwen21EditOptions derives from QwenImageOptions, so the model id
            // alone cannot tell an edit job apart in the gallery / job list.
            Edit = request is Qwen21EditOptions,
            HasMask = request is Qwen21EditOptions { Mask: not null } || hasInpaintMask,
            AdvancedEdit = advanced,
            Outpaint = outpaint,
            Rgba = rgba,
            Control = control,
            HasControl = hasControl,
            Inpaint = inpaint,
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

    /// <summary>
    /// ControlNet state derived from a request. Kept as a pure function so the
    /// job-list / gallery flags are unit-testable without a native engine;
    /// <see cref="From"/> uses it for both properties.
    /// </summary>
    internal static (bool Control, bool HasControl) ControlState(GenerationOptions request)
        => request switch
        {
            // ZImageInpaintOptions derives from ZImageControlOptions, so the
            // derived record must be matched first. Inpaint still carries a
            // control image, so it keeps the ControlNet badges as well.
            ZImageInpaintOptions inpaint => (true, inpaint.ControlImage is not null),
            ZImageControlOptions control => (true, control.ControlImage is not null),
            _ => (false, false)
        };

    /// <summary>
    /// Inpaint state derived from a request. Pure function so the gallery
    /// badges are unit-testable without a native engine. Returns
    /// <c>(Inpaint, HasMask)</c>; the mask is required on an inpaint request.
    /// </summary>
    internal static (bool Inpaint, bool HasMask) InpaintState(GenerationOptions request)
        => request is ZImageInpaintOptions inpaint
            ? (true, inpaint.Mask is not null)
            : (false, false);

    /// <summary>
    /// Advanced-edit state derived from a request. Pure function so the gallery
    /// badges are unit-testable without a native engine.
    /// </summary>
    internal static (bool Advanced, bool Outpaint, bool Rgba) AdvancedEditState(
        GenerationOptions request)
        => request is Qwen21EditExOptions advanced
            ? (true, advanced.HasOutpainting, advanced.Rgba)
            : (false, false, false);
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
