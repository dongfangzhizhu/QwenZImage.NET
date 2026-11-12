namespace QwenZImage.NET.WebUI.ModelDownloads;

/// <summary>Body of <c>POST /api/models/download</c>.</summary>
public sealed class ModelDownloadRequest
{
    /// <summary>File ids from <c>GET /api/models</c>. Empty means all missing required files.</summary>
    public List<string>? Files { get; set; }

    /// <summary>Queue every missing file, including optional LoRAs and ONNX assets.</summary>
    public bool All { get; set; }

    /// <summary>Variant id (<c>int8</c> / <c>bf16</c>); defaults to <c>int8</c>.</summary>
    public string? Variant { get; set; }

    /// <summary>Re-download files that are already present.</summary>
    public bool Force { get; set; }
}

/// <summary>Body of <c>POST /api/models/config</c>. A dry run is the default.</summary>
public sealed class ModelConfigRequest
{
    /// <summary>Actually merge the keys into the config file (with a backup).</summary>
    public bool Apply { get; set; }
}

/// <summary>Body of <c>POST /api/models/control</c>.</summary>
public sealed class ModelControlRequest
{
    /// <summary><c>pause</c>, <c>resume</c>, <c>reset</c> or <c>reset-all</c>.</summary>
    public string? Action { get; set; }

    /// <summary>Single file id for <c>reset</c>; omit for <c>reset-all</c>.</summary>
    public string? File { get; set; }
}
