using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using QwenZImage.NET.WebUI.Preprocessing;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// The WebUI owns ControlNet guide preprocessing while the NuGet library keeps
/// the C ABI's raw planar-RGB contract. These tests pin down the operator
/// mapping, the BCL-only PNG encoder/decoder round trip and the API wire
/// format, so an ordinary browser upload can never silently be consumed
/// "as-is" again.
/// </summary>
public class ControlImagePreprocessingTests
{
    private const int Width = 96;
    private const int Height = 64;

    // A flat two-tone image with one strong vertical edge at Width / 2. Easy
    // to reason about for Canny / scribble / gray assertions.
    private static byte[] SyntheticRgb(int width = Width, int height = Height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int i = (y * width + x) * 3;
            bool right = x >= width / 2;
            rgb[i] = (byte)(right ? 230 : 25);
            rgb[i + 1] = (byte)(right ? 40 : 90);
            rgb[i + 2] = (byte)(right ? 40 : 200);
        }
        return rgb;
    }

    private static string SyntheticPng(int width = Width, int height = Height)
        => PngWriter.ToDataUrl(SyntheticRgb(width, height), width, height);

    private static ControlImagePreprocessor Managed()
        => new(PreprocessAssets.None);

    // ------------------------------------------------------------- mode table

    [Fact]
    public void All_Seven_Modes_Are_Advertised_And_Parseable()
    {
        string[] expected = ["auto", "raw", "gray", "canny", "scribble", "depth", "pose"];
        Assert.Equal(expected, ControlImagePreprocessor.Modes);
        foreach (string mode in expected)
            Assert.True(ControlImagePreprocessor.TryParse(mode, out ControlPreprocessMode parsed),
                        $"'{mode}' should parse");
        Assert.False(ControlImagePreprocessor.TryParse("nonsense", out _));
        Assert.True(ControlImagePreprocessor.TryParse(null, out ControlPreprocessMode blank));
        Assert.Equal(ControlPreprocessMode.Auto, blank);
    }

    [Theory]
    [InlineData("V", ControlPreprocessMode.Raw)]
    [InlineData("identity", ControlPreprocessMode.Raw)]
    [InlineData("C", ControlPreprocessMode.Gray)]
    [InlineData("E", ControlPreprocessMode.Canny)]
    [InlineData("EV", ControlPreprocessMode.Canny)]
    [InlineData("S", ControlPreprocessMode.Scribble)]
    [InlineData("DV", ControlPreprocessMode.Depth)]
    [InlineData("P", ControlPreprocessMode.Pose)]
    [InlineData("PV", ControlPreprocessMode.Pose)]
    public void WanGP_Guide_Letters_Map_To_Operators(string value, ControlPreprocessMode expected)
    {
        Assert.True(ControlImagePreprocessor.TryParse(value, out ControlPreprocessMode mode));
        Assert.Equal(expected, mode);
        Assert.Equal(expected, ControlImagePreprocessor.FromControlType(value));
    }

    [Theory]
    [InlineData(ControlPreprocessMode.Canny, "canny")]
    [InlineData(ControlPreprocessMode.Scribble, "scribble")]
    [InlineData(ControlPreprocessMode.Depth, "depth")]
    [InlineData(ControlPreprocessMode.Pose, "pose")]
    [InlineData(ControlPreprocessMode.Raw, "raw")]
    [InlineData(ControlPreprocessMode.Gray, "raw")]
    public void Recommended_Control_Type_Stays_Inside_The_Canonical_Whitelist(
        ControlPreprocessMode mode, string expected)
    {
        // "gray" has no canonical native control type, so it is posted as raw.
        string type = ControlImagePreprocessor.ControlTypeFor(mode);
        Assert.Equal(expected, type);
        Assert.Contains(type, new[] { "auto", "pose", "canny", "depth", "scribble", "raw" });
    }

    // ------------------------------------------------------------ PNG encoder

    [Fact]
    public void PngWriter_RoundTrips_Through_The_Core_Decoder()
    {
        byte[] rgb = SyntheticRgb(37, 23); // deliberately non-round dimensions
        string url = PngWriter.ToDataUrl(rgb, 37, 23);

        Assert.StartsWith("data:image/png;base64,", url);
        QziRasterImage raster = QziRasterImage.FromPngBase64(url);
        Assert.Equal(37, raster.Width);
        Assert.Equal(23, raster.Height);
        Assert.Equal(rgb, RasterOps.PlanarToInterleaved(raster.Planar.Span, 37, 23));
    }

    // -------------------------------------------------------------- operators

    [Theory]
    [InlineData("raw", "raw", "raw")]
    [InlineData("gray", "managed-gray", "raw")]
    [InlineData("canny", "managed-canny", "canny")]
    [InlineData("scribble", "managed-scribble", "scribble")]
    [InlineData("depth", "managed-depth-proxy", "depth")]
    public void Every_Dependency_Free_Mode_Produces_A_Decodable_Guide(
        string modeName, string provider, string controlType)
    {
        Assert.True(ControlImagePreprocessor.TryParse(modeName, out ControlPreprocessMode mode));
        using ControlImagePreprocessor preprocessor = Managed();

        ControlPreprocessResult result = preprocessor.Process(
            SyntheticPng(), mode, maxSide: 256);

        Assert.Equal(modeName, result.Mode);
        Assert.Equal(provider, result.Provider);
        Assert.Equal(controlType, result.ControlType);
        Assert.False(result.Learned);
        Assert.Equal(Width, result.Width);
        Assert.Equal(Height, result.Height);
        Assert.StartsWith("data:image/png;base64,", result.ControlImage);

        QziRasterImage guide = QziRasterImage.FromPngBase64(result.ControlImage);
        Assert.Equal(Width, guide.Width);
        Assert.Equal(Height, guide.Height);
        byte[] rgb = RasterOps.PlanarToInterleaved(guide.Planar.Span, Width, Height);

        switch (modeName)
        {
            case "raw":
                Assert.Equal(SyntheticRgb(), rgb);
                Assert.Null(result.Warning);
                break;
            case "gray":
                for (int i = 0; i < Width * Height; i++)
                    Assert.True(rgb[i * 3] == rgb[i * 3 + 1] && rgb[i * 3] == rgb[i * 3 + 2],
                                "gray guide must be an achromatic RGB triple");
                break;
            case "canny":
            case "scribble":
                for (int i = 0; i < rgb.Length; i++)
                    Assert.True(rgb[i] is 0 or 255, "contour guides are binary");
                Assert.Contains((byte)255, rgb); // the synthetic edge is detected
                break;
            case "depth":
                Assert.NotNull(result.Warning);
                Assert.Contains("ONNX", result.Warning);
                Assert.True(rgb.Distinct().Count() > 4, "depth proxy must be a continuous map");
                break;
        }
    }

    [Fact]
    public void Canny_Localises_The_Synthetic_Edge_And_Thin_Shrinks_It()
    {
        byte[] gray = RasterOps.ToGrayscaleInterleaved(SyntheticRgb(), Width, Height);
        byte[] edges = RasterOps.Canny(gray, Width, Height);

        int nearEdge = 0, farFromEdge = 0;
        for (int y = 2; y < Height - 2; y++)
        for (int x = 0; x < Width; x++)
        {
            if (edges[y * Width + x] == 0) continue;
            if (Math.Abs(x - Width / 2) <= 2) nearEdge++;
            else farFromEdge++;
        }
        Assert.True(nearEdge > 0, "Canny must find the vertical step edge");
        Assert.True(farFromEdge <= nearEdge, "flat regions must not produce stray edges");
        Assert.All(edges, v => Assert.True(v is 0 or 255));

        byte[] thin = RasterOps.Thin(edges, Width, Height);
        Assert.True(thin.Count(v => v != 0) <= edges.Count(v => v != 0));
        Assert.Contains((byte)255, thin);
    }

    [Fact]
    public void Auto_Mode_Resolves_From_Control_Type()
    {
        using ControlImagePreprocessor preprocessor = Managed();

        // "auto" must not be a synonym for "do nothing": it picks the operator
        // matching the declared control type, including WanGP letter aliases.
        Assert.Throws<ArgumentException>(() => preprocessor.Process(SyntheticPng(),
            ControlPreprocessMode.Auto, controlType: "pose"));

        ControlPreprocessResult scribble = preprocessor.Process(SyntheticPng(),
            ControlPreprocessMode.Auto, controlType: "S");
        Assert.Equal("scribble", scribble.Mode);
        Assert.Equal("managed-scribble", scribble.Provider);

        ControlPreprocessResult untouched = preprocessor.Process(SyntheticPng(),
            ControlPreprocessMode.Auto, controlType: "auto");
        Assert.Equal("raw", untouched.Mode);
        Assert.Equal("raw", untouched.Provider);
    }

    [Fact]
    public void Auto_Mode_Follows_A_Canny_Control_Type()
    {
        using ControlImagePreprocessor preprocessor = Managed();
        ControlPreprocessResult result = preprocessor.Process(SyntheticPng(),
            ControlPreprocessMode.Auto, controlType: "canny");

        Assert.Equal("canny", result.Mode);
        Assert.Equal("canny", result.ControlType);
        Assert.Equal("managed-canny", result.Provider);
    }

    [Fact]
    public void Pose_Without_WanGP_Assets_Fails_With_Configuration_Guidance()
    {
        using ControlImagePreprocessor preprocessor = Managed();
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => preprocessor.Process(SyntheticPng(), ControlPreprocessMode.Pose));

        Assert.Contains("QZI_PREPROCESS_DIR", error.Message);
        Assert.Contains("yolox_l.onnx", error.Message);
        Assert.Contains("dw-ll_ucoco_384.onnx", error.Message);
    }

    [Fact]
    public void Invalid_Arguments_Are_Rejected_Before_Any_Work()
    {
        using ControlImagePreprocessor preprocessor = Managed();
        Assert.Throws<ArgumentException>(
            () => preprocessor.Process("", ControlPreprocessMode.Raw));
        Assert.Throws<ArgumentException>(
            () => preprocessor.Process(SyntheticPng(), ControlPreprocessMode.Canny,
                                       lowThreshold: 0.9f, highThreshold: 0.1f));
        Assert.Throws<ArgumentException>(
            () => preprocessor.Process(SyntheticPng(), ControlPreprocessMode.Canny,
                                       maxSide: 8));
    }

    // ------------------------------------------------------- asset discovery

    [Fact]
    public void WanGP_Checkpoint_Layout_Is_Discovered_From_The_Config_File()
    {
        string root = Path.Combine(Path.GetTempPath(), $"qzi-assets-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(Path.Combine(root, "pose"));
            Directory.CreateDirectory(Path.Combine(root, "depth"));
            File.WriteAllText(Path.Combine(root, "pose", "yolox_l.onnx"), "placeholder");
            File.WriteAllText(Path.Combine(root, "pose", "dw-ll_ucoco_384.onnx"), "placeholder");
            File.WriteAllText(Path.Combine(root, "depth", "depth_anything_v2.onnx"), "placeholder");

            string config = Path.Combine(root, "QwenZImage.config.json");
            File.WriteAllText(config, JsonSerializer.Serialize(new { preprocess_dir = root }));

            PreprocessAssets assets = PreprocessAssets.Discover(config, root);
            Assert.Equal(Path.GetFullPath(root), assets.Root);
            Assert.True(assets.PoseAvailable);
            Assert.True(assets.DepthAvailable);
            Assert.False(assets.ContourAvailable);
            Assert.EndsWith("yolox_l.onnx", assets.PoseDetectorPath);
            Assert.EndsWith("dw-ll_ucoco_384.onnx", assets.PoseModelPath);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Missing_Assets_Degrade_To_The_Managed_Operators()
    {
        PreprocessAssets assets = PreprocessAssets.None;
        Assert.Null(assets.Root);
        Assert.False(assets.PoseAvailable);
        Assert.False(assets.DepthAvailable);
        Assert.False(assets.ContourAvailable);
    }

    // ------------------------------------------------------------- wire format

    [Fact]
    public void Result_Uses_The_Snake_Case_API_Wire_Format()
    {
        ControlPreprocessResult result = new()
        {
            ControlImage = "data:image/png;base64,AAAA",
            Mode = "canny",
            ControlType = "canny",
            Provider = "managed-canny",
            Learned = false,
            Width = Width,
            Height = Height
        };

        string json = JsonSerializer.Serialize(result, WebUiJson.Options);
        Assert.Contains("\"control_image\"", json);
        Assert.Contains("\"control_type\"", json);
        Assert.DoesNotContain("ControlImage", json);

        string requestJson = JsonSerializer.Serialize(
            new PreprocessRequest { Image = "x", Mode = "pose", MaxSide = 1024 },
            WebUiJson.Options);
        Assert.Contains("\"max_side\":1024", requestJson);
    }

    [Fact]
    public void Preprocess_Request_Binds_Snake_Case_Properties()
    {
        const string json = """
            { "image": "data:image/png;base64,AAAA", "mode": "pose",
              "control_type": "auto", "low_threshold": 0.1, "high_threshold": 0.4,
              "max_side": 768 }
            """;
        PreprocessRequest? request = JsonSerializer.Deserialize<PreprocessRequest>(
            json, WebUiJson.Options);

        Assert.NotNull(request);
        Assert.Equal("pose", request.Mode);
        Assert.Equal(0.1f, request.LowThreshold);
        Assert.Equal(0.4f, request.HighThreshold);
        Assert.Equal(768, request.MaxSide);
        Assert.StartsWith("data:image/png;base64,", request.Image);
    }

    [Fact]
    public void WebUi_Config_Exposes_The_Launch_Directory()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"qzi-launch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            WebUiConfig config = WebUiConfig.FromEnvironment(dir, []);
            Assert.Equal(Path.GetFullPath(dir), config.LaunchDirectory);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
