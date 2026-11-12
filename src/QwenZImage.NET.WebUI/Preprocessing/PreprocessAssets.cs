using System.Text.Json;

namespace QwenZImage.NET.WebUI.Preprocessing;

/// <summary>
/// Locates optional WanGP preprocessor assets. The WebUI always ships the
/// dependency-free managed operators; when these ONNX models are present it
/// switches to the learned estimators used by WanGP itself (YOLOX + DWPose for
/// `pose`, a Depth Anything V2 ONNX export for `depth`, and an ONNX export of
/// `netG_A_latest.pth` for `canny` / `scribble`).
/// </summary>
/// <remarks>
/// Resolution order for the asset root:
/// <list type="number">
/// <item><c>QZI_PREPROCESS_DIR</c> environment variable.</item>
/// <item><c>preprocess_dir</c> in <c>QwenZImage.config.json</c> (top-level or
/// the platform-specific <c>windows</c> / <c>linux</c> object).</item>
/// </list>
/// WanGP's own checkpoint tree uses <c>&lt;root&gt;/pose/*.onnx</c>,
/// <c>&lt;root&gt;/depth/*.onnx</c> and <c>&lt;root&gt;/scribble/*.onnx</c>.
/// </remarks>
public sealed class PreprocessAssets
{
    /// <summary>
    /// Environment override for the preprocessor asset directory. Shared with
    /// the core config parser so a typo cannot make the two disagree.
    /// </summary>
    public const string DirectoryEnvironmentVariable = QziConfig.PreprocessDirectoryVariable;

    private PreprocessAssets(string? root, string? detector, string? pose,
                             string? depth, string? contour)
    {
        Root = root;
        PoseDetectorPath = detector;
        PoseModelPath = pose;
        DepthModelPath = depth;
        ContourModelPath = contour;
    }

    /// <summary>
    /// An empty asset set: every operator uses its dependency-free managed
    /// implementation. Also the deterministic baseline for unit tests.
    /// </summary>
    public static PreprocessAssets None { get; } = new(null, null, null, null, null);

    public string? Root { get; }
    public string? PoseDetectorPath { get; }
    public string? PoseModelPath { get; }
    public string? DepthModelPath { get; }
    public string? ContourModelPath { get; }

    public bool PoseAvailable => PoseDetectorPath is not null && PoseModelPath is not null;
    public bool DepthAvailable => DepthModelPath is not null;
    public bool ContourAvailable => ContourModelPath is not null;

    public static PreprocessAssets Discover(string? configPath, string contentRoot)
    {
        string? root = Environment.GetEnvironmentVariable(DirectoryEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(root)) root = ReadConfigDirectory(configPath, contentRoot);
        if (string.IsNullOrWhiteSpace(root)) return new PreprocessAssets(null, null, null, null, null);

        root = Path.GetFullPath(ExpandHome(root));
        if (!Directory.Exists(root)) return new PreprocessAssets(root, null, null, null, null);

        string? detector = Find(root, "pose", "yolox_l.onnx") ?? FindAny(root, "yolox*.onnx");
        string? pose = Find(root, "pose", "dw-ll_ucoco_384.onnx") ?? FindAny(root, "dw-ll*.onnx");
        string? depth = FindAny(root, "depth*.onnx") ?? FindAny(root, "depth_anything*.onnx");
        string? contour = FindAny(root, "netG_A_latest.onnx") ?? FindAny(root, "contour*.onnx");
        return new PreprocessAssets(root, detector, pose, depth, contour);
    }

    /// <summary>JSON-safe capability description for the health endpoint.</summary>
    public object Describe() => new
    {
        asset_dir = Root,
        onnx_runtime = true,
        learned_pose = PoseAvailable,
        learned_depth = DepthAvailable,
        learned_contour = ContourAvailable,
        pose_detector = PoseDetectorPath,
        pose_model = PoseModelPath,
        depth_model = DepthModelPath,
        contour_model = ContourModelPath
    };

    private static string? Find(string root, string folder, string file)
    {
        string candidate = Path.Combine(root, folder, file);
        return File.Exists(candidate) ? candidate : null;
    }

    private static string? FindAny(string root, string pattern)
    {
        // A WanGP ckpts tree can contain protected / transient subdirectories.
        // Missing or unreadable assets must degrade to the managed operators,
        // never take the whole WebUI down.
        try
        {
            return Directory.EnumerateFiles(root, pattern, SearchOption.AllDirectories)
                            .FirstOrDefault();
        }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
    }

    private static string? ReadConfigDirectory(string? configPath, string contentRoot)
    {
        string? file = configPath;
        if (string.IsNullOrWhiteSpace(file))
        {
            DirectoryInfo? dir = new(contentRoot);
            for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
            {
                string candidate = Path.Combine(dir.FullName, "QwenZImage.config.json");
                if (File.Exists(candidate)) { file = candidate; break; }
            }
        }
        if (string.IsNullOrWhiteSpace(file) || !File.Exists(file)) return null;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(file));
            JsonElement root = doc.RootElement;
            string platform = OperatingSystem.IsWindows() ? "windows"
                            : OperatingSystem.IsLinux() ? "linux" : "osx";
            if (root.TryGetProperty(platform, out JsonElement scoped) &&
                scoped.ValueKind == JsonValueKind.Object &&
                scoped.TryGetProperty("preprocess_dir", out JsonElement scopedValue) &&
                scopedValue.ValueKind == JsonValueKind.String)
                return scopedValue.GetString();
            if (root.TryGetProperty("preprocess_dir", out JsonElement global) &&
                global.ValueKind == JsonValueKind.String)
                return global.GetString();
        }
        catch (JsonException) { /* a malformed config must not break preprocessing */ }
        catch (IOException) { }
        return null;
    }

    private static string ExpandHome(string path)
    {
        if (!path.StartsWith('~')) return path;
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, path.TrimStart('~').TrimStart('/', '\\'));
    }
}
