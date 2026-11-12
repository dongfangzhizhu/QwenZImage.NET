using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace QwenZImage.NET.WebUI.ModelDownloads;

/// <summary>Result of a dry run or a real write to <c>QwenZImage.config.json</c>.</summary>
/// <param name="Applied">False for a preview: nothing was written.</param>
/// <param name="ConfigPath">Absolute path of the config file that is (or would be) written.</param>
/// <param name="Created">True when no config file existed and a new one would be created.</param>
/// <param name="BackupPath">Copy of the previous file, made before an applied write.</param>
/// <param name="Keys">The resolved model paths merged into the platform section.</param>
/// <param name="ActiveOverrides">Environment / CLI overrides that win over the file.</param>
/// <param name="MergedJson">The exact JSON that was (or would be) written.</param>
/// <param name="RestartRequired">The native engine only reads its config at startup.</param>
public sealed record ModelConfigApplyResult(
    bool Applied,
    string ConfigPath,
    bool Created,
    string? BackupPath,
    IReadOnlyDictionary<string, string> Keys,
    IReadOnlyList<string> ActiveOverrides,
    string MergedJson,
    bool RestartRequired)
{
    /// <summary>Operator-facing explanation, shown next to the Apply button.</summary>
    public string Message => Applied
        ? (ActiveOverrides.Count == 0
            ? $"Updated {ConfigPath}. Restart the WebUI so the native engine re-reads it."
            : $"Updated {ConfigPath}, but the active overrides below still win. Unset them and restart the WebUI.")
        : "Preview only — nothing was written. POST with {\"apply\":true} to merge these keys.";
}

/// <summary>
/// Safely merges the downloader's resolved model directories into the host's
/// <c>QwenZImage.config.json</c>. Only the six keys owned by the manifest are
/// touched, and only inside the current platform section; every other key,
/// comment and platform section is preserved. The path is always chosen by the
/// server (never by the browser), a timestamped backup is made first, and the
/// write is atomic.
/// </summary>
public static class ModelConfigWriter
{
    private static readonly JsonDocumentOptions ParseOptions = new()
    {
        CommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    private static readonly JsonSerializerOptions WriteOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    /// <summary>
    /// The file the running WebUI loaded, or — when it loaded no file at all —
    /// a new one in the launch directory (which the config search order checks
    /// on the next start).
    /// </summary>
    public static string TargetPath(WebUiConfig config)
        => Path.GetFullPath(config.ConfigPath ??
                            Path.Combine(config.LaunchDirectory, QziConfig.FileName));

    /// <summary>Environment variables and CLI switches that outrank the file.</summary>
    public static IReadOnlyList<string> ActiveOverrides(WebUiConfig config)
    {
        List<string> active = [];
        foreach (ModelGroup group in ModelManifest.Groups)
        {
            string? value = Environment.GetEnvironmentVariable(group.EnvironmentVariable);
            if (!string.IsNullOrWhiteSpace(value))
                active.Add($"{group.EnvironmentVariable} overrides {group.ConfigKey}");
        }
        if (config.ModelOverrideActive)
            active.Add("--models / QZI_MODEL_DIR overrides qwen21_dir and zimage_dir");
        return active;
    }

    /// <summary>Builds the merged JSON without touching the disk.</summary>
    public static string MergedJson(WebUiConfig config, ModelCatalog catalog, string path)
    {
        JsonObject root;
        if (File.Exists(path))
        {
            string existing = File.ReadAllText(path);
            JsonNode? parsed = JsonNode.Parse(existing, nodeOptions: null, ParseOptions);
            if (parsed is not JsonObject obj)
                throw new InvalidDataException($"{path}: the root must be a JSON object.");
            root = obj;
        }
        else
        {
            root = new JsonObject();
        }

        string platform = catalog.ConfigPlatform;
        JsonObject section;
        if (root.TryGetPropertyValue(platform, out JsonNode? existingSection) &&
            existingSection is JsonObject typed)
        {
            section = typed;
        }
        else
        {
            section = new JsonObject();
            root[platform] = section;
        }

        foreach (ModelGroup group in ModelManifest.Groups)
            section[group.ConfigKey] = JsonValue.Create(Slash(catalog.BasePathFor(group)));

        return root.ToJsonString(WriteOptions) + Environment.NewLine;
    }

    /// <summary>Resolved values for every config key owned by the manifest.</summary>
    public static IReadOnlyDictionary<string, string> ResolvedKeys(ModelCatalog catalog)
    {
        Dictionary<string, string> keys = new(StringComparer.Ordinal);
        foreach (ModelGroup group in ModelManifest.Groups)
            keys[group.ConfigKey] = Slash(catalog.BasePathFor(group));
        return keys;
    }

    /// <summary>
    /// Previews (<paramref name="apply"/>=false) or performs the merge. A real
    /// write is guarded by a backup and an atomic replace; a failure leaves the
    /// original file untouched.
    /// </summary>
    public static ModelConfigApplyResult Run(WebUiConfig config, ModelCatalog catalog, bool apply)
    {
        string path = TargetPath(config);
        bool created = !File.Exists(path);
        IReadOnlyDictionary<string, string> keys = ResolvedKeys(catalog);
        IReadOnlyList<string> overrides = ActiveOverrides(config);
        string merged = MergedJson(config, catalog, path);

        if (!apply)
            return new ModelConfigApplyResult(false, path, created, null,
                                              keys, overrides, merged, true);

        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory)) Directory.CreateDirectory(directory);

        string? backup = null;
        if (!created)
        {
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss",
                                                    System.Globalization.CultureInfo.InvariantCulture);
            string candidate = $"{path}.bak.{stamp}";
            for (int i = 1; File.Exists(candidate); i++)
                candidate = $"{path}.bak.{stamp}-{i}";
            File.Copy(path, candidate, overwrite: false);
            backup = candidate;
        }

        // Write to a sibling temp file, then replace. A crash mid-write can
        // never leave a half-written config behind.
        string temporary = path + ".qzi-tmp";
        try
        {
            File.WriteAllText(temporary, merged, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            // Validate our own output before it replaces the live config.
            using JsonDocument check = JsonDocument.Parse(File.ReadAllBytes(temporary), ParseOptions);
            if (check.RootElement.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("merged config root is not a JSON object");
            File.Move(temporary, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }

        return new ModelConfigApplyResult(true, path, created, backup,
                                          keys, overrides, merged, true);
    }

    private static string Slash(string path) => path.Replace('\\', '/');
}
