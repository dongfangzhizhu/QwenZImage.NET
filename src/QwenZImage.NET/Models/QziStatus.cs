namespace QwenZImage.NET;

/// <summary>Native <c>qzi_status</c> codes (numeric values are part of the ABI).</summary>
public enum QziStatus
{
    /// <summary>Success.</summary>
    Ok = 0,
    /// <summary>Malformed request or failed validation.</summary>
    InvalidArgument = 1,
    /// <summary>Model pack is missing or the engine has not been created.</summary>
    NotConfigured = 2,
    /// <summary>Unsupported tier, checkpoint or feature.</summary>
    Unsupported = 3,
    /// <summary>Another generation owns the device (single-flight).</summary>
    Busy = 4,
    /// <summary>Cancelled by the caller's token.</summary>
    Cancelled = 5,
    /// <summary>Native allocation failed.</summary>
    OutOfMemory = 6,
    /// <summary>Model load, denoise, VAE or device error.</summary>
    EngineFailure = 7,
    /// <summary>Internal error in the native control plane.</summary>
    InternalError = 8
}

/// <summary>Generation stages reported by the engine.</summary>
public enum GenerationPhase
{
    /// <summary>No stage reported yet.</summary>
    None = 0,
    /// <summary>Switching the resident model / loading DiT weights.</summary>
    ModelLoad = 1,
    /// <summary>Text encoding.</summary>
    Text = 2,
    /// <summary>Denoise loop (<c>Completed</c>/<c>Total</c> = step/steps).</summary>
    Dit = 3,
    /// <summary>VAE decode.</summary>
    Vae = 4,
    /// <summary>PNG encoding.</summary>
    Encode = 5,
    /// <summary>A stage name the wrapper does not recognise (still surfaced).</summary>
    Unknown = 6
}
