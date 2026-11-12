using System.Text.Json;

namespace QwenZImage.NET.WebUI.ModelDownloads;

/// <summary>
/// One choice for a logical model file, with its resolved local path.
/// <see cref="SourceUrl"/> is always the canonical WanGP Hugging Face address,
/// while <see cref="Url"/> is the effective URL after an optional mirror rewrite.
/// </summary>
public sealed record ModelVariantStatus(
    string Id, string Label, string Url, string SourceUrl, long Size,
    string TargetPath, string CurlCommand, bool Recommended);

/// <summary>Current state of one logical model file.</summary>
public sealed record ModelFileStatus(
    string Id, string Name, string Description, bool Required, string? Note,
    string Status, string TargetPath, long InstalledBytes, string? InstalledVariant,
    long DownloadedBytes, long TotalBytes, long ResumeFrom, double SpeedBps,
    string? Error, IReadOnlyList<ModelVariantStatus> Variants);

/// <summary>One <c>QwenZImage.config.json</c> key and all files under it.</summary>
public sealed record ModelGroupStatus(
    string Id, string Name, string ConfigKey, string EnvironmentVariable,
    string ResolvedPath, string TargetDirectory, string PathSource, bool Configured,
    string Description, bool Optional, string? Note,
    IReadOnlyList<ModelFileStatus> Files);

/// <summary>Complete JSON view served by <c>GET /api/models</c>.</summary>
public sealed record ModelCatalogSnapshot(
    string? ConfigPath, string ConfigPlatform, string ConfigExample,
    string ConfigTargetPath, bool ConfigExists, IReadOnlyList<string> ConfigOverrides,
    string? Mirror, string DefaultVariant, bool Paused, int QueueLength,
    long TotalBytes, long InstalledBytes, IReadOnlyList<ModelGroupStatus> Groups);

/// <summary>
/// Binds the static WanGP catalogue to the host's actual configuration. It
/// resolves destination paths, detects already-present weights (including
/// WanGP's upstream file names), builds the manual-download guide and creates
/// download jobs. No file is fetched or written here.
/// </summary>
public sealed class ModelCatalog
{
    private readonly WebUiConfig _config;
    private readonly ModelDownloadService _downloads;
    private readonly string? _mirror;
    private readonly string _defaultRoot;
    private readonly IReadOnlyDictionary<string, ResolvedRoot> _roots;

    public ModelCatalog(WebUiConfig config, string? preprocessDirectory,
                        ModelDownloadService downloads, string? mirror = null)
    {
        _config = config;
        _downloads = downloads;
        _mirror = string.IsNullOrWhiteSpace(mirror) ? null : mirror.Trim();
        _defaultRoot = Path.GetFullPath(config.LaunchDirectory);

        QziClientOptions options = config.ClientOptions;
        Dictionary<string, ResolvedRoot> roots = new(StringComparer.Ordinal);
        roots["qwen21_dir"] = Resolve(config, options.Qwen21Directory, QziConfig.Qwen21DirectoryVariable);
        roots["qwen21_lora_dir"] = Resolve(config, options.Qwen21LoraDirectory, QziConfig.LoraDirectoryVariable);
        roots["zimage_dir"] = Resolve(config, options.ZImageDirectory, QziConfig.ZImageDirectoryVariable);
        roots["zimage_nunchaku_dir"] = Resolve(config, options.ZImageNunchakuDirectory, QziConfig.NunchakuDirectoryVariable);
        roots["zimage_control_dir"] = Resolve(config, options.ZImageControlDirectory, QziConfig.ZImageControlDirectoryVariable);
        roots["preprocess_dir"] = Resolve(config, preprocessDirectory, QziConfig.PreprocessDirectoryVariable);
        _roots = roots;
    }

    /// <summary>Platform section name used in config examples.</summary>
    public string ConfigPlatform { get; } = OperatingSystem.IsWindows() ? "windows"
                                         : OperatingSystem.IsLinux() ? "linux" : "osx";

    /// <summary>Optional mirror base URL (only the HF prefix is ever rewritten).</summary>
    public string? Mirror => _mirror;

    private ResolvedRoot Resolve(WebUiConfig config, string? configured, string variable)
    {
        string? env = Environment.GetEnvironmentVariable(variable);
        string raw = !string.IsNullOrWhiteSpace(env) ? env! : configured ?? "";
        if (string.IsNullOrWhiteSpace(raw))
        {
            return new ResolvedRoot("", "default (not configured)", false);
        }
        string expanded = QziConfig.Expand(raw.Trim()) ?? raw.Trim();
        string full = Path.GetFullPath(expanded);
        string source = !string.IsNullOrWhiteSpace(env) ? "environment variable"
                      : "config file / CLI override";
        return new ResolvedRoot(full, source, true);
    }

    /// <summary>Directory (or explicit .safetensors path) for a config key.</summary>
    public string BasePathFor(ModelGroup group)
    {
        ResolvedRoot root = _roots[group.ConfigKey];
        return root.Configured
            ? root.Path
            : ModelManifest.CombineRelative(_defaultRoot, group.DefaultSubdirectory);
    }

    /// <summary>Absolute destination for one concrete variant.</summary>
    public string ResolveTarget(ModelGroup group, ModelFile file, ModelVariant variant)
    {
        string root = BasePathFor(group);
        if (group.PathKind == ModelPathKind.FileOrDirectory &&
            root.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            return root;                        // config points at the exact file

        string relative = variant.RelativePath ?? file.RelativePath;
        return ModelManifest.CombineRelative(root, relative);
    }

    private IEnumerable<string> CandidatePaths(ModelGroup group, ModelFile file)
    {
        string root = BasePathFor(group);
        bool exactFile = group.PathKind == ModelPathKind.FileOrDirectory &&
                         root.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);
        if (exactFile)
        {
            yield return root;
            yield break;
        }
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        IEnumerable<string> relatives = new[] { file.RelativePath }
            .Concat(file.Aliases ?? [])
            .Concat(file.Variants.Where(v => v.RelativePath is not null)
                                 .Select(v => v.RelativePath!));
        foreach (string relative in relatives)
        {
            string path = ModelManifest.CombineRelative(root, relative);
            if (seen.Add(path)) yield return path;
        }
    }

    private static (string? Path, long Bytes) FirstInstalled(IEnumerable<string> candidates)
    {
        foreach (string path in candidates)
        {
            try
            {
                FileInfo info = new(path);
                if (info.Exists && info.Length > 0) return (path, info.Length);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
        return (null, 0);
    }

    private static (string? Path, long Bytes) FirstPartial(IEnumerable<string> candidates)
    {
        foreach (string path in candidates)
        {
            string part = path + ModelDownloadService.PartSuffix;
            try
            {
                FileInfo info = new(part);
                if (info.Exists && info.Length > 0) return (part, info.Length);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
        return (null, 0);
    }

    /// <summary>Full snapshot for the API and the manual-download page.</summary>
    public ModelCatalogSnapshot Snapshot()
    {
        IReadOnlyDictionary<string, ModelDownloadState> states = _downloads.Snapshot();
        List<ModelGroupStatus> groups = [];
        long totalBytes = 0, installedBytes = 0;

        foreach (ModelGroup group in ModelManifest.Groups)
        {
            string root = BasePathFor(group);
            ResolvedRoot resolved = _roots[group.ConfigKey];
            string directory = group.PathKind == ModelPathKind.FileOrDirectory &&
                               root.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                ? Path.GetDirectoryName(root) ?? root
                : root;
            List<ModelFileStatus> files = [];

            foreach (ModelFile file in group.Files)
            {
                string[] candidates = CandidatePaths(group, file).ToArray();
                (string? installed, long installedLength) = FirstInstalled(candidates);
                (string? partial, long partialLength) = FirstPartial(candidates);
                ModelDownloadState? state = states.GetValueOrDefault(file.Id);
                ModelVariant recommended = PickVariant(file, ModelManifest.DefaultVariant);
                string target = ResolveTarget(group, file, recommended);

                string status = state?.Status switch
                {
                    "downloading" => "downloading",
                    "queued" => "queued",
                    "paused" => "paused",
                    "done" => "installed",
                    "failed" => "failed",
                    _ when installed is not null => "installed",
                    _ when partial is not null => "partial",
                    _ => "missing"
                };

                string? installedVariant = null;
                if (installed is not null)
                {
                    ModelVariant? match = file.Variants.FirstOrDefault(
                        v => string.Equals(ResolveTarget(group, file, v), installed,
                                           StringComparison.OrdinalIgnoreCase));
                    installedVariant = match?.Id ?? "existing file";
                }

                totalBytes += file.Variants.First().Size;
                if (installed is not null) installedBytes += installedLength;
                else if (partial is not null) installedBytes += partialLength;

                List<ModelVariantStatus> variants = [];
                foreach (ModelVariant variant in file.Variants)
                {
                    string variantPath = ResolveTarget(group, file, variant);
                    variants.Add(new ModelVariantStatus(
                        variant.Id, variant.Label,
                        ModelManifest.ResolveUrl(variant.Url, _mirror), variant.Url,
                        variant.Size, variantPath,
                        CurlCommand(variantPath, variant.Url, variant.Size),
                        string.Equals(variant.Id, recommended.Id, StringComparison.Ordinal)));
                }

                files.Add(new ModelFileStatus(
                    file.Id, file.Name, file.Description, file.Required, file.Note,
                    status, installed ?? target, installedLength, installedVariant,
                    state?.Bytes ?? partialLength, state?.Total ?? recommended.Size,
                    state?.ResumeFrom ?? partialLength, state?.SpeedBps ?? 0,
                    state?.Error, variants));
            }

            groups.Add(new ModelGroupStatus(
                group.Id, group.Name, group.ConfigKey, group.EnvironmentVariable,
                root, directory, resolved.Source, resolved.Configured,
                group.Description, group.Optional, group.Note, files));
        }

        string configTarget = ModelConfigWriter.TargetPath(_config);
        return new ModelCatalogSnapshot(
            _config.ConfigPath, ConfigPlatform, ConfigExample(),
            configTarget, File.Exists(configTarget), ModelConfigWriter.ActiveOverrides(_config),
            _mirror, ModelManifest.DefaultVariant, _downloads.Paused,
            _downloads.QueueLength, totalBytes, installedBytes, groups);
    }

    /// <summary>
    /// Ids of files that still need fetching: non-optional, required files whose
    /// status is missing / partial / failed. An empty <c>files</c> array in
    /// <c>POST /api/models/download</c> expands to exactly this list.
    /// </summary>
    public IReadOnlyList<string> MissingRequiredIds()
        => MissingIds(includeOptional: false);

    /// <summary>Every missing / partial / failed file, including optional groups.</summary>
    public IReadOnlyList<string> MissingAllIds()
        => MissingIds(includeOptional: true);

    private IReadOnlyList<string> MissingIds(bool includeOptional)
    {
        ModelCatalogSnapshot snapshot = Snapshot();
        return snapshot.Groups
            .Where(group => includeOptional || !group.Optional)
            .SelectMany(group => group.Files)
            .Where(file => (includeOptional || file.Required) &&
                           file.Status is "missing" or "partial" or "failed")
            .Select(file => file.Id)
            .ToArray();
    }

    /// <summary>
    /// Turns a list of file ids into jobs. An empty / missing list expands to
    /// every missing required file (or every missing file when
    /// <paramref name="all"/> is set), which is what the "Download all" button
    /// relies on. Installed files are skipped unless <paramref name="force"/>
    /// is set. Unknown ids are rejected with an actionable message.
    /// </summary>
    public (IReadOnlyList<ModelDownloadJob> Jobs, IReadOnlyList<string> Skipped) CreateJobs(
        IEnumerable<string>? ids, string? variantId, bool force, bool all = false)
    {
        string wanted = string.IsNullOrWhiteSpace(variantId)
            ? ModelManifest.DefaultVariant : variantId!.Trim();
        IReadOnlyDictionary<string, ModelDownloadState> states = _downloads.Snapshot();
        List<ModelDownloadJob> jobs = [];
        List<string> skipped = [];

        IEnumerable<string> chosen = ids?.Where(id => !string.IsNullOrWhiteSpace(id))
                                         .Distinct(StringComparer.Ordinal)
                                    ?? [];
        if (all || !chosen.Any())
            chosen = all ? MissingAllIds() : MissingRequiredIds();

        foreach (string id in chosen)
        {
            (ModelGroup group, ModelFile file) = Find(id);
            ModelVariant variant = PickVariant(file, wanted);
            string target = ResolveTarget(group, file, variant);
            if (!force && FirstInstalled(CandidatePaths(group, file)).Path is not null)
            {
                skipped.Add(id);
                continue;
            }
            if (states.TryGetValue(id, out ModelDownloadState? state) &&
                state.Status is "queued" or "downloading" or "paused")
            {
                skipped.Add(id);
                continue;
            }
            jobs.Add(new ModelDownloadJob(
                id, group.Id, variant.Id, file.Name,
                ModelManifest.ResolveUrl(variant.Url, _mirror), target, variant.Size));
        }
        return (jobs, skipped);
    }

    /// <summary>Looks up a file id, returning its owning group.</summary>
    public (ModelGroup Group, ModelFile File) Find(string id)
    {
        foreach (ModelGroup group in ModelManifest.Groups)
            foreach (ModelFile file in group.Files)
                if (string.Equals(file.Id, id, StringComparison.Ordinal)) return (group, file);
        throw new ArgumentException(
            $"Unknown model file '{id}'. Valid ids: {string.Join(", ", ModelManifest.FilesById.Keys)}.",
            nameof(id));
    }

    /// <summary>Picks the requested variant, falling back to the first (recommended) one.</summary>
    public static ModelVariant PickVariant(ModelFile file, string? variantId)
        => string.IsNullOrWhiteSpace(variantId)
            ? file.Variants[0]
            : file.Variants.FirstOrDefault(
                  v => string.Equals(v.Id, variantId, StringComparison.OrdinalIgnoreCase))
              ?? file.Variants[0];

    /// <summary>A copy/paste-able resumable command for manual download.</summary>
    public string CurlCommand(string targetPath, string url, long size)
    {
        string resolved = ModelManifest.ResolveUrl(url, _mirror);
        string quotedPath = '"' + targetPath.Replace('\\', '/') + '"';
        string directory = (Path.GetDirectoryName(targetPath) ?? targetPath).Replace('\\', '/');
        string mkdir = OperatingSystem.IsWindows()
            ? $"powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '{directory}' | Out-Null\""
            : $"mkdir -p \"{directory}\"";
        return $"{mkdir}\ncurl -L -C - --retry 3 --fail -o {quotedPath} \"{resolved}\"\n" +
               $"# expected size: {size} bytes";
    }

    /// <summary>
    /// A ready-to-paste config fragment with the resolved destination of every
    /// key. The file is never modified by the server: the operator decides when
    /// to apply it (and restarts the WebUI so the native engine re-reads it).
    /// </summary>
    public string ConfigExample()
    {
        Dictionary<string, object?> section = new(StringComparer.Ordinal);
        foreach (ModelGroup group in ModelManifest.Groups)
        {
            string root = BasePathFor(group);
            section[group.ConfigKey] = Slash(root);
        }
        Dictionary<string, object?> document = new(StringComparer.Ordinal)
        {
            ["_comment"] = "Model weights downloaded by the WebUI. Paste this platform section into QwenZImage.config.json, then restart the WebUI.",
            ["device_id"] = -1,
            ["thread_count"] = 0,
            [ConfigPlatform] = section
        };
        return JsonSerializer.Serialize(document, new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>Markdown guide with every URL, destination, command and config step.</summary>
    public string BuildManualGuide()
    {
        ModelCatalogSnapshot snapshot = Snapshot();
        List<string> lines =
        [
            "# QwenZImage.NET — manual model download",
            "",
            "All links below are the WanGP Hugging Face `resolve/main` URLs used by",
            "`D:/AI/Wan2GP` (`defaults/*.json`, the model handlers and `preprocessing/*/assets.py`).",
            "When `QZI_HF_MIRROR` is set, each entry lists **both** the canonical WanGP",
            "source URL and the effective mirror URL used by the copy/paste command.",
            "This project does **not** bundle or redistribute weights; check each upstream",
            "repository's licence before downloading or redistributing them.",
            "",
            $"- Active config file: `{snapshot.ConfigPath ?? "(none found — using defaults)"}`",
            $"- Platform section: `{snapshot.ConfigPlatform}`",
            $"- Mirror: `{snapshot.Mirror ?? "(none — direct huggingface.co)"}`",
            "- Recommended variant: `int8` (smaller quanto / ConvRot repacks)",
            "",
            "## 1. Download the files",
            "",
            "Each command is resumable: `curl -C -` continues an interrupted transfer,",
            "and `curl -L` follows the Hugging Face CDN redirect. You can also paste the",
            "URL into a browser / download manager and rename the file to the exact",
            "target name shown below."
        ];

        foreach (ModelGroupStatus group in snapshot.Groups)
        {
            lines.Add("");
            lines.Add($"### {group.Name}");
            lines.Add("");
            lines.Add($"- Config key: `{group.ConfigKey}` (env `{group.EnvironmentVariable}`)");
            lines.Add($"- Destination: `{group.ResolvedPath}` — {group.PathSource}");
            if (!group.Configured)
                lines.Add("- **Not configured yet**: add this key to the config example in step 2.");
            lines.Add($"- {group.Description}");
            if (group.Note is not null) lines.Add($"- Source: {group.Note}");
            lines.Add("");

            foreach (ModelFileStatus file in group.Files)
            {
                lines.Add($"#### {file.Name} `{file.Id}` — {(file.Required ? "required" : "optional")}");
                lines.Add("");
                lines.Add(file.Description);
                if (file.Note is not null) { lines.Add(""); lines.Add("> " + file.Note); }
                lines.Add("");
                foreach (ModelVariantStatus variant in file.Variants)
                {
                    lines.Add($"- **{variant.Label}** (`{variant.Id}`) — {variant.Size} bytes" +
                              (variant.Recommended ? " ← recommended" : ""));
                    if (!string.Equals(variant.SourceUrl, variant.Url, StringComparison.Ordinal))
                    {
                        // The canonical WanGP address must stay visible even when
                        // QZI_HF_MIRROR rewrites the effective download URL.
                        lines.Add($"  - WanGP source URL: <{variant.SourceUrl}>");
                        lines.Add($"  - Effective URL (mirror): <{variant.Url}>");
                    }
                    else
                    {
                        lines.Add($"  - URL: <{variant.Url}>");
                    }
                    lines.Add($"  - Save as: `{variant.TargetPath}`");
                    lines.Add("  - Command:");
                    lines.Add("");
                    lines.Add("    ```sh");
                    foreach (string commandLine in variant.CurlCommand.Split('\n'))
                        lines.Add("    " + commandLine);
                    lines.Add("    ```");
                }
                lines.Add("");
            }
        }

        lines.AddRange(
        [
            "## 2. Point the config file at the downloaded files",
            "",
            $"Open `{snapshot.ConfigPath ?? "QwenZImage.config.json"}` and merge the",
            $"`{snapshot.ConfigPlatform}` section below (keep your other keys). The values",
            "are the exact destinations used above. Alternatively export the matching",
            "`QZI_*` environment variables, which take precedence over the file.",
            "",
            "```json",
            snapshot.ConfigExample,
            "```",
            "",
            "- `zimage_control_dir` may name **one** `.safetensors` file or a directory;",
            "  when it is a directory the engine matches the",
            "  `Z-Image-Turbo-Fun-Controlnet-Union2.1_*.safetensors` prefix.",
            "- `preprocess_dir` is WebUI-only: the native engine ignores it.",
            "- After editing the file, **restart the WebUI** — the native engine reads",
            "  its config once at `qzi_engine_create()`.",
            "",
            "## 3. Verify",
            "",
            "- `GET /api/models` reports `installed` for each file with a non-empty target.",
            "- `GET /api/health` reports the resident model / backend once a real CUDA",
            "  native library is selected; the bundled reference backend emits noise.",
            "- Inpaint needs no extra checkpoint: it reuses ControlNet Union 2.x weights.",
            ""
        ]);
        return string.Join('\n', lines);
    }

    private static string Slash(string path) => path.Replace('\\', '/');

    private sealed record ResolvedRoot(string Path, string Source, bool Configured);
}
