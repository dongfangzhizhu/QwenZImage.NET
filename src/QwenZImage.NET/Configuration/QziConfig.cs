using System.Text.Json;

namespace QwenZImage.NET;

/// <summary>
/// Model-pack locations read from <c>QwenZImage.config.json</c>, the single
/// source of truth for both the managed wrapper and the native backend.
/// </summary>
/// <remarks>
/// Layout mirrors the original engine's config file: top-level keys apply to
/// every platform, and a <c>windows</c> / <c>linux</c> / <c>osx</c> object
/// overrides them on that OS. Values may start with <c>~</c>. Unknown keys are
/// rejected rather than ignored, so a typo cannot look like "the setting does
/// not work". Keys beginning with <c>_</c> are documentation comments.
///
/// Resolution order (highest first):
/// <list type="number">
///   <item>values set directly in code on <see cref="QziClientOptions"/></item>
///   <item><c>QZI_*</c> environment overrides</item>
///   <item>the platform section of the config file</item>
///   <item>the top-level keys of the config file</item>
/// </list>
///
/// File search order: an explicit path (<c>--config</c> or
/// <c>QZI_CONFIG</c>), then next to the app binary, then the current
/// directory, then a bounded walk up both parent chains (so a config file at
/// the solution root is found when <c>dotnet run</c> starts the app from the
/// project directory), then <c>~/.config/QwenZImage/config.json</c>. When no
/// file is found an empty configuration is returned, leaving the native
/// backend's own defaults in place.
/// </remarks>
public sealed record QziConfig
{
    /// <summary>Default config file name.</summary>
    public const string FileName = "QwenZImage.config.json";

    /// <summary>Environment variable naming an explicit config file.</summary>
    public const string EnvironmentVariable = "QZI_CONFIG";

    /// <summary>Environment variable overriding <c>qwen21_dir</c>.</summary>
    public const string Qwen21DirectoryVariable = "QZI_QWEN21_DIR";

    /// <summary>Environment variable overriding <c>zimage_dir</c>.</summary>
    public const string ZImageDirectoryVariable = "QZI_ZIMAGE_DIR";

    /// <summary>Environment variable carrying the Qwen21 LoRA directory to the native backend.</summary>
    public const string LoraDirectoryVariable = "QZI_QWEN21_LORA_DIR";

    /// <summary>Environment variable carrying the Z-Image Nunchaku directory to the native backend.</summary>
    public const string NunchakuDirectoryVariable = "QZI_ZIMAGE_NUNCHAKU_DIR";

    /// <summary>Environment variable carrying the Z-Image ControlNet Union weights directory to the native backend.</summary>
    public const string ZImageControlDirectoryVariable = "QZI_ZIMAGE_CONTROL_DIR";

    /// <summary>Environment variable overriding <c>text_cache_dir</c>.</summary>
    public const string ModelCacheVariable = "QZI_MODEL_CACHE";

    /// <summary>Environment variable naming the native library to load.</summary>
    public const string NativeLibraryVariable = "QZI_NATIVE_LIBRARY";

    /// <summary>Environment variable overriding <c>device_id</c>.</summary>
    public const string DeviceIdVariable = "QZI_DEVICE_ID";

    private static readonly HashSet<string> KnownKeys = new(StringComparer.Ordinal)
    {
        "qwen21_dir", "zimage_dir", "zimage_nunchaku_dir", "zimage_control_dir",
        "qwen21_lora_dir", "text_cache_dir", "device_id", "thread_count", "native_library"
    };

    private static readonly HashSet<string> PlatformKeys = new(StringComparer.Ordinal)
    {
        "windows", "linux", "osx", "macos"
    };

    /// <summary>Qwen-Image 2.1 pack (dit_turbo/base, vae, text, tokenizer, loras).</summary>
    public string? Qwen21Directory { get; init; }

    /// <summary>Z-Image pack (dit, text_encoder, vae).</summary>
    public string? ZImageDirectory { get; init; }

    /// <summary>SVDQuant int4 DiT repacks for Z-Image; empty means unavailable.</summary>
    public string? ZImageNunchakuDirectory { get; init; }

    /// <summary>Z-Image Turbo Fun ControlNet Union weights directory.</summary>
    public string? ZImageControlDirectory { get; init; }

    /// <summary>Directory holding Qwen-Image 2.1 LoRA adapters (bare file names).</summary>
    public string? Qwen21LoraDirectory { get; init; }

    /// <summary>Text-embedding cache directory.</summary>
    public string? TextCacheDirectory { get; init; }

    /// <summary>Compute device index; -1 selects automatically.</summary>
    public int DeviceId { get; init; } = -1;

    /// <summary>CPU thread count; 0 selects the engine default.</summary>
    public int ThreadCount { get; init; }

    /// <summary>Explicit path to the native backend library.</summary>
    public string? NativeLibraryPath { get; init; }

    /// <summary>Absolute path of the file the values came from, or null.</summary>
    public string? SourcePath { get; init; }

    /// <summary>True when no config file was found.</summary>
    public bool IsEmpty => SourcePath is null;

    /// <summary>
    /// Loads the first config file found. An explicit path that does not exist
    /// is an error; a missing optional file is not.
    /// </summary>
    /// <exception cref="FileNotFoundException">An explicit path was given but is missing.</exception>
    /// <exception cref="InvalidDataException">Malformed JSON or an unknown key.</exception>
    public static QziConfig Load(string? explicitPath = null)
    {
        string? path = ResolvePath(explicitPath);
        // Environment overrides must apply even when no config file exists:
        // a container that only sets QZI_* still has to reach the engine.
        if (path is null) return new QziConfig().ApplyEnvironmentOverrides();

        using JsonDocument doc = Parse(path);
        JsonElement root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"{path}: the root must be a JSON object.");

        Dictionary<string, JsonElement> values = ReadSection(root, path, "top level");

        string? platform = PlatformSection();
        if (platform is not null && root.TryGetProperty(platform, out JsonElement section))
        {
            if (section.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"{path}: '{platform}' must be a JSON object.");
            foreach ((string key, JsonElement value) in ReadSection(section, path, platform))
                values[key] = value;
        }

        QziConfig config = new()
        {
            Qwen21Directory = Expand(Str(values, "qwen21_dir", path)),
            ZImageDirectory = Expand(Str(values, "zimage_dir", path)),
            ZImageNunchakuDirectory = Expand(Str(values, "zimage_nunchaku_dir", path)),
            ZImageControlDirectory = Expand(Str(values, "zimage_control_dir", path)),
            Qwen21LoraDirectory = Expand(Str(values, "qwen21_lora_dir", path)),
            TextCacheDirectory = Expand(Str(values, "text_cache_dir", path)),
            NativeLibraryPath = Expand(Str(values, "native_library", path)),
            DeviceId = Int(values, "device_id", path, -1),
            ThreadCount = Int(values, "thread_count", path, 0),
            SourcePath = path
        };

        return config.ApplyEnvironmentOverrides();
    }

    /// <summary>Paths that would be tried, in order, without touching the disk.</summary>
    public static IReadOnlyList<string> SearchPaths()
    {
        List<string> paths = [];
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);

        void Add(string path)
        {
            if (seen.Add(path)) paths.Add(path);
        }

        string binaryDirectory = AppContext.BaseDirectory;
        string workingDirectory = Directory.GetCurrentDirectory();
        Add(Path.Combine(binaryDirectory, FileName));
        Add(Path.Combine(workingDirectory, FileName));

        // `dotnet run` makes the *project* directory the working directory, so
        // a solution-root config file is several levels up. Walk both chains a
        // bounded number of steps; never scan the whole drive.
        foreach (string start in new[] { binaryDirectory, workingDirectory })
        {
            DirectoryInfo? dir = new(start);
            for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
                Add(Path.Combine(dir.FullName, FileName));
        }

        Add(Path.Combine(HomeDirectory(), ".config", "QwenZImage", "config.json"));
        return paths;
    }

    /// <summary>Converts this configuration into the options object used by <see cref="QziClient"/>.</summary>
    public QziClientOptions ToClientOptions() => new()
    {
        Qwen21Directory = Qwen21Directory,
        ZImageDirectory = ZImageDirectory,
        CacheDirectory = TextCacheDirectory,
        Qwen21LoraDirectory = Qwen21LoraDirectory,
        ZImageNunchakuDirectory = ZImageNunchakuDirectory,
        ZImageControlDirectory = ZImageControlDirectory,
        DeviceId = DeviceId,
        ThreadCount = ThreadCount,
        NativeLibraryPath = NativeLibraryPath,
        ConfigPath = SourcePath
    };

    /// <summary>
    /// Produces a copy with a single field overridden, for callers that want
    /// "config file plus one explicit flag" without mutating shared state.
    /// </summary>
    public QziConfig With(string key, string? value)
    {
        if (!KnownKeys.Contains(key))
            throw new ArgumentException($"Unknown config key '{key}'.", nameof(key));
        string? expanded = Expand(value);
        return key switch
        {
            "qwen21_dir" => this with { Qwen21Directory = expanded },
            "zimage_dir" => this with { ZImageDirectory = expanded },
            "zimage_nunchaku_dir" => this with { ZImageNunchakuDirectory = expanded },
            "zimage_control_dir" => this with { ZImageControlDirectory = expanded },
            "qwen21_lora_dir" => this with { Qwen21LoraDirectory = expanded },
            "text_cache_dir" => this with { TextCacheDirectory = expanded },
            "native_library" => this with { NativeLibraryPath = expanded },
            "device_id" => this with { DeviceId = ParseInt(key, value, -1) },
            "thread_count" => this with { ThreadCount = ParseInt(key, value, 0) },
            _ => this
        };
    }

    // -------------------------------------------------------------- internals

    private static string? ResolvePath(string? explicitPath)
    {
        string? chosen = explicitPath;
        if (string.IsNullOrWhiteSpace(chosen))
            chosen = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(chosen))
        {
            string full = Path.GetFullPath(Expand(chosen) ?? chosen);
            if (!File.Exists(full))
                throw new FileNotFoundException($"Config file not found: {full}", full);
            return full;
        }

        foreach (string candidate in SearchPaths())
            if (File.Exists(candidate)) return Path.GetFullPath(candidate);
        return null;
    }

    private static JsonDocument Parse(string path)
    {
        try
        {
            return JsonDocument.Parse(File.ReadAllBytes(path),
                new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip,
                                          AllowTrailingCommas = true });
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"{path}: invalid JSON: {ex.Message}", ex);
        }
    }

    private static Dictionary<string, JsonElement> ReadSection(
        JsonElement section, string path, string where)
    {
        Dictionary<string, JsonElement> result = new(StringComparer.Ordinal);
        foreach (JsonProperty property in section.EnumerateObject())
        {
            string key = property.Name;
            if (key.StartsWith('_')) continue;              // documentation comments
            if (PlatformKeys.Contains(key)) continue;       // handled by the caller
            if (!KnownKeys.Contains(key))
                throw new InvalidDataException(
                    $"{path}: unknown config key '{key}' in {where}. " +
                    $"Known keys: {string.Join(", ", KnownKeys.OrderBy(k => k))}.");
            result[key] = property.Value;
        }
        return result;
    }

    private QziConfig ApplyEnvironmentOverrides()
    {
        QziConfig result = this;
        result = Env(result, Qwen21DirectoryVariable, c => c with { Qwen21Directory = Expand(Environment.GetEnvironmentVariable(Qwen21DirectoryVariable)) });
        result = Env(result, ZImageDirectoryVariable, c => c with { ZImageDirectory = Expand(Environment.GetEnvironmentVariable(ZImageDirectoryVariable)) });
        result = Env(result, NunchakuDirectoryVariable, c => c with { ZImageNunchakuDirectory = Expand(Environment.GetEnvironmentVariable(NunchakuDirectoryVariable)) });
        result = Env(result, ZImageControlDirectoryVariable, c => c with { ZImageControlDirectory = Expand(Environment.GetEnvironmentVariable(ZImageControlDirectoryVariable)) });
        result = Env(result, LoraDirectoryVariable, c => c with { Qwen21LoraDirectory = Expand(Environment.GetEnvironmentVariable(LoraDirectoryVariable)) });
        result = Env(result, ModelCacheVariable, c => c with { TextCacheDirectory = Expand(Environment.GetEnvironmentVariable(ModelCacheVariable)) });
        result = Env(result, NativeLibraryVariable, c => c with { NativeLibraryPath = Expand(Environment.GetEnvironmentVariable(NativeLibraryVariable)) });
        if (int.TryParse(Environment.GetEnvironmentVariable(DeviceIdVariable), out int device))
            result = result with { DeviceId = device };
        return result;
    }

    private static QziConfig Env(QziConfig config, string name, Func<QziConfig, QziConfig> apply)
        => string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(name))
            ? config : apply(config);

    private static string? PlatformSection()
        => OperatingSystem.IsWindows() ? "windows"
         : OperatingSystem.IsLinux() ? "linux"
         : OperatingSystem.IsMacOS() ? "osx" : null;

    private static string HomeDirectory()
        => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>Expands a leading <c>~</c> / <c>~/</c> / <c>~\</c>.</summary>
    public static string? Expand(string? value)
    {
        if (string.IsNullOrEmpty(value)) return value;
        if (value == "~") return HomeDirectory();
        if (value.StartsWith("~/", StringComparison.Ordinal) ||
            value.StartsWith("~\\", StringComparison.Ordinal))
            return Path.Combine(HomeDirectory(), value[2..]);
        return value;
    }

    private static string? Str(Dictionary<string, JsonElement> values, string key, string path)
    {
        if (!values.TryGetValue(key, out JsonElement element)) return null;
        if (element.ValueKind == JsonValueKind.Null) return null;
        if (element.ValueKind != JsonValueKind.String)
            throw new InvalidDataException($"{path}: '{key}' must be a string.");
        return element.GetString();
    }

    private static int Int(Dictionary<string, JsonElement> values, string key, string path, int fallback)
    {
        if (!values.TryGetValue(key, out JsonElement element)) return fallback;
        if (element.ValueKind == JsonValueKind.Null) return fallback;
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt32(out int parsed))
            throw new InvalidDataException($"{path}: '{key}' must be an integer.");
        return parsed;
    }

    private static int ParseInt(string key, string? value, int fallback)
        => int.TryParse(value, out int parsed)
            ? parsed
            : throw new ArgumentException($"'{key}' must be an integer.", nameof(value));
}
