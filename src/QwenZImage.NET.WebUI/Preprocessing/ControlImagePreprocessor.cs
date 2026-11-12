using QwenZImage.NET;
using QwenZImage.NET.WebUI;

namespace QwenZImage.NET.WebUI.Preprocessing;

/// <summary>Mirrors WanGP's guide preprocessing choices.</summary>
public enum ControlPreprocessMode
{
    /// <summary>Derive the operator from <c>control_type</c>; <c>auto</c>/<c>raw</c> means WanGP "V" (no transform).</summary>
    Auto,
    /// <summary>WanGP "V": Use Z-Image Raw Format. The uploaded pixels are kept.</summary>
    Raw,
    /// <summary>WanGP "C": plain greyscale guide.</summary>
    Gray,
    /// <summary>WanGP "E"/"EV": edge / contour guide.</summary>
    Canny,
    /// <summary>WanGP "S": pencil-scribble guide.</summary>
    Scribble,
    /// <summary>WanGP "D"/"DV": monocular depth guide.</summary>
    Depth,
    /// <summary>WanGP "P"/"PV": DWPose skeleton guide.</summary>
    Pose
}

/// <summary>Result of one preprocessing pass, ready to post as <c>control_image</c>.</summary>
public sealed class ControlPreprocessResult
{
    public required string ControlImage { get; init; }
    public required string Mode { get; init; }
    public required string ControlType { get; init; }
    public required string Provider { get; init; }
    public bool Learned { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
    public string? Warning { get; init; }
}

/// <summary>
/// Turns a browser upload (an ordinary photograph) into a Z-Image ControlNet
/// guide map. The core <c>QwenZImage.NET</c> library deliberately keeps the C
/// ABI's raw-raster contract; preprocessing is an application-layer concern and
/// therefore lives in the WebUI, where third-party imaging dependencies are
/// allowed. All six operators have dependency-free managed implementations;
/// when WanGP ONNX assets are configured the learned estimators replace them.
/// </summary>
public sealed class ControlImagePreprocessor : IDisposable
{
    public static IReadOnlyList<string> Modes { get; } =
        ["auto", "raw", "gray", "canny", "scribble", "depth", "pose"];

    private readonly PreprocessAssets _assets;
    private readonly object _gate = new();
    private PoseEstimator? _poseEstimator;
    private DenseMapEstimator? _depthEstimator;
    private DenseMapEstimator? _contourEstimator;

    public ControlImagePreprocessor(PreprocessAssets assets)
        => _assets = assets ?? throw new ArgumentNullException(nameof(assets));

    public PreprocessAssets Assets => _assets;

    public static bool TryParse(string? value, out ControlPreprocessMode mode)
    {
        switch (string.IsNullOrWhiteSpace(value) ? "auto" : value.Trim().ToLowerInvariant())
        {
            case "auto": mode = ControlPreprocessMode.Auto; return true;
            case "raw" or "v" or "none" or "identity": mode = ControlPreprocessMode.Raw; return true;
            case "gray" or "grey" or "c": mode = ControlPreprocessMode.Gray; return true;
            case "canny" or "edge" or "e" or "ev": mode = ControlPreprocessMode.Canny; return true;
            case "scribble" or "s": mode = ControlPreprocessMode.Scribble; return true;
            case "depth" or "d" or "dv": mode = ControlPreprocessMode.Depth; return true;
            case "pose" or "p" or "pv": mode = ControlPreprocessMode.Pose; return true;
            default: mode = default; return false;
        }
    }

    /// <summary>
    /// Resolves <c>mode=auto</c> from the descriptive <c>control_type</c>.
    /// Both canonical names and WanGP's guide letters (with or without the
    /// trailing "V") are accepted, so a WanGP-style configuration maps 1:1.
    /// </summary>
    public static ControlPreprocessMode FromControlType(string? controlType)
        => (controlType ?? "").Trim().ToLowerInvariant() switch
        {
            "canny" or "e" or "ev" => ControlPreprocessMode.Canny,
            "scribble" or "s" => ControlPreprocessMode.Scribble,
            "depth" or "d" or "dv" => ControlPreprocessMode.Depth,
            "pose" or "p" or "pv" => ControlPreprocessMode.Pose,
            "gray" or "grey" or "c" => ControlPreprocessMode.Gray,
            _ => ControlPreprocessMode.Raw
        };

    public static string ControlTypeFor(ControlPreprocessMode mode) => mode switch
    {
        ControlPreprocessMode.Canny => "canny",
        ControlPreprocessMode.Scribble => "scribble",
        ControlPreprocessMode.Depth => "depth",
        ControlPreprocessMode.Pose => "pose",
        _ => "raw"
    };

    /// <summary>Health-endpoint capability description.</summary>
    public object Describe() => new
    {
        available = true,
        modes = Modes,
        learned = _assets.Describe()
    };

    /// <summary>
    /// Applies the selected operator and returns a PNG data URL that can be
    /// previewed in the browser and posted back as <c>control_image</c>.
    /// </summary>
    public ControlPreprocessResult Process(string pngBase64, ControlPreprocessMode mode,
                                           string? controlType = null,
                                           float lowThreshold = 0.15f,
                                           float highThreshold = 0.35f,
                                           int maxSide = 1024)
    {
        if (string.IsNullOrWhiteSpace(pngBase64))
            throw new ArgumentException("image (a base64 PNG or data URL) is required.");
        if (mode == ControlPreprocessMode.Auto) mode = FromControlType(controlType);
        if (lowThreshold <= 0 || highThreshold <= lowThreshold || highThreshold > 1f)
            throw new ArgumentException(
                "low_threshold must be > 0 and high_threshold must be greater than it and <= 1.");
        if (maxSide < 64 || maxSide > QziRasterImage.MaxSide)
            throw new ArgumentException($"max_side must be between 64 and {QziRasterImage.MaxSide}.");

        QziRasterImage raster;
        try
        {
            raster = QziRasterImage.FromPngBase64(pngBase64);
        }
        catch (Exception ex) when (ex is ArgumentException or FormatException or InvalidDataException)
        {
            // The native decoder caps a control raster at 4096 per side and
            // 2048^2 pixels. Browser uploads are downscaled before this call;
            // give direct API callers an actionable message instead of a raw
            // PNG decoder / dimension error.
            throw new ArgumentException(
                "image must be an 8-bit PNG (base64 or a data:image/png;base64,... URL) " +
                $"no larger than {QziRasterImage.MaxSide}px per side and " +
                $"{QziRasterImage.MaxPixels} pixels total. {ex.Message}", ex);
        }
        byte[] rgb = RasterOps.PlanarToInterleaved(raster.Planar.Span, raster.Width, raster.Height);
        int width = raster.Width, height = raster.Height;

        // The native control raster is capped at 4096 per side / 2048^2 pixels.
        double factor = Math.Min(1.0, (double)maxSide / Math.Max(width, height));
        if ((double)width * height > QziRasterImage.MaxPixels)
            factor = Math.Min(factor, Math.Sqrt((double)QziRasterImage.MaxPixels / (width * height)));
        if (factor < 1.0)
        {
            int newWidth = Math.Max(1, (int)(width * factor));
            int newHeight = Math.Max(1, (int)(height * factor));
            rgb = RasterOps.ResizeRgb(rgb, width, height, newWidth, newHeight);
            width = newWidth; height = newHeight;
        }

        byte[] grey = RasterOps.ToGrayscaleInterleaved(rgb, width, height);

        byte[] output;
        string provider;
        bool learned = false;
        string? warning = null;

        switch (mode)
        {
            case ControlPreprocessMode.Raw:
                output = rgb;
                provider = "raw";
                break;

            case ControlPreprocessMode.Gray:
                output = RasterOps.ToInterleavedRgb(grey, width, height);
                provider = "managed-gray";
                break;

            case ControlPreprocessMode.Canny:
                (output, provider, learned, warning) = ContourMap(
                    rgb, grey, width, height, lowThreshold, highThreshold,
                    invert: true, binarise: true, scribble: false);
                break;

            case ControlPreprocessMode.Scribble:
                (output, provider, learned, warning) = ContourMap(
                    rgb, grey, width, height, lowThreshold, highThreshold,
                    invert: false, binarise: false, scribble: true);
                break;

            case ControlPreprocessMode.Depth:
                try
                {
                    if (_assets.DepthAvailable)
                    {
                        output = RasterOps.ToInterleavedRgb(
                            Depth().Estimate(rgb, width, height), width, height);
                        provider = "onnx-depth-anything-v2";
                        learned = true;
                    }
                    else throw new FileNotFoundException("no depth ONNX model");
                }
                catch (Exception ex) when (ex is not ArgumentException)
                {
                    output = RasterOps.ToInterleavedRgb(
                        RasterOps.DepthProxy(grey, width, height), width, height);
                    provider = "managed-depth-proxy";
                    warning = _assets.DepthAvailable
                        ? $"The depth ONNX estimator failed ({ex.Message}); used the deterministic proxy instead."
                        : "No Depth Anything V2 ONNX model is configured; used the deterministic relative-depth proxy. " +
                          "Place one under <preprocess_dir>/depth/*.onnx to use the WanGP learned map.";
                }
                break;

            case ControlPreprocessMode.Pose:
                if (!_assets.PoseAvailable)
                    throw new ArgumentException(
                        "Pose preprocessing needs WanGP's ONNX pose models. Set QZI_PREPROCESS_DIR " +
                        "(or preprocess_dir in QwenZImage.config.json) to the ckpts directory containing " +
                        "pose/yolox_l.onnx and pose/dw-ll_ucoco_384.onnx. Choose canny / scribble / " +
                        "depth / gray / raw for a dependency-free guide.");
                output = Pose().Estimate(rgb, width, height);
                provider = "onnx-wan2gp-dwpose";
                learned = true;
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }

        return new ControlPreprocessResult
        {
            ControlImage = PngWriter.ToDataUrl(output, width, height),
            Mode = mode.ToString().ToLowerInvariant(),
            ControlType = ControlTypeFor(mode),
            Provider = provider,
            Learned = learned,
            Width = width,
            Height = height,
            Warning = warning
        };
    }

    private (byte[] Rgb, string Provider, bool Learned, string? Warning) ContourMap(
        byte[] rgb, byte[] grey, int width, int height, float low, float high,
        bool invert, bool binarise, bool scribble)
    {
        try
        {
            if (_assets.ContourAvailable)
            {
                byte[] map = Contour().Estimate(rgb, width, height, invert, binarise);
                return (RasterOps.ToInterleavedRgb(map, width, height),
                        scribble ? "onnx-wan2gp-scribble" : "onnx-wan2gp-contour", true, null);
            }
        }
        catch (Exception ex) when (ex is not ArgumentException)
        {
            return (ManagedContour(grey, width, height, low, high, scribble),
                    scribble ? "managed-scribble" : "managed-canny", false,
                    $"The contour ONNX estimator failed ({ex.Message}); used the classic Canny operator instead.");
        }
        return (ManagedContour(grey, width, height, low, high, scribble),
                scribble ? "managed-scribble" : "managed-canny", false, null);
    }

    private static byte[] ManagedContour(byte[] grey, int width, int height,
                                         float low, float high, bool scribble)
    {
        byte[] edges = RasterOps.Canny(grey, width, height, low, high);
        if (!scribble) return RasterOps.ToInterleavedRgb(edges, width, height);
        // WanGP's scribble is a thin pencil line on white paper; canny is a
        // white line on black, so thin + invert reproduces that polarity.
        byte[] lines = RasterOps.Thin(edges, width, height);
        return RasterOps.ToInterleavedRgb(lines, width, height, invert: true);
    }

    private PoseEstimator Pose()
    {
        lock (_gate)
            return _poseEstimator ??= new PoseEstimator(_assets.PoseDetectorPath!,
                                                        _assets.PoseModelPath!);
    }

    private DenseMapEstimator Depth()
    {
        lock (_gate)
            return _depthEstimator ??= new DenseMapEstimator(_assets.DepthModelPath!,
                                                             imageNetNormalisation: true);
    }

    private DenseMapEstimator Contour()
    {
        lock (_gate)
            return _contourEstimator ??= new DenseMapEstimator(_assets.ContourModelPath!,
                                                               imageNetNormalisation: false,
                                                               defaultSize: 512);
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _poseEstimator?.Dispose();
            _depthEstimator?.Dispose();
            _contourEstimator?.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
