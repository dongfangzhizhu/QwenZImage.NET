using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace QwenZImage.NET.WebUI.Preprocessing;

/// <summary>
/// Generic single-input / single-output ONNX dense-map estimator. It covers the
/// two learned WanGP preprocessors that are not pose estimators:
/// <list type="bullet">
/// <item>Depth Anything V2 ONNX export -> a relative depth map.</item>
/// <item>ContourInference (<c>netG_A_latest.pth</c> exported to ONNX) -> the
/// learned line map used by WanGP's canny / scribble annotators.</item>
/// </list>
/// The output is min/max normalised to 8-bit, resized back to the source size,
/// and optionally inverted or binarised to match each WanGP annotator.
/// </summary>
internal sealed class DenseMapEstimator : IDisposable
{
    private static readonly float[] ImageNetMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] ImageNetStd = [0.229f, 0.224f, 0.225f];

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _height;
    private readonly int _width;
    private readonly bool _imageNetNormalisation;

    public DenseMapEstimator(string path, bool imageNetNormalisation, int defaultSize = 518)
    {
        _session = new InferenceSession(path);
        _inputName = _session.InputMetadata.Keys.First();
        int[] dims = _session.InputMetadata[_inputName].Dimensions;
        _height = dims.Length >= 4 && dims[2] > 0 ? dims[2] : defaultSize;
        _width = dims.Length >= 4 && dims[3] > 0 ? dims[3] : defaultSize;
        _imageNetNormalisation = imageNetNormalisation;
    }

    /// <summary>Runs the model and returns a one-channel 8-bit map.</summary>
    public byte[] Estimate(byte[] interleavedRgb, int width, int height,
                           bool invert = false, bool binarise = false)
    {
        byte[] resized = RasterOps.ResizeRgb(interleavedRgb, width, height, _width, _height);
        int plane = _width * _height;
        float[] input = new float[checked(3 * plane)];
        for (int i = 0; i < plane; i++)
        for (int c = 0; c < 3; c++)
        {
            float value = resized[i * 3 + c] / 255f;
            if (_imageNetNormalisation) value = (value - ImageNetMean[c]) / ImageNetStd[c];
            input[c * plane + i] = value;
        }

        using var results = _session.Run([
            NamedOnnxValue.CreateFromTensor(_inputName,
                new DenseTensor<float>(input, [1, 3, _height, _width]))
        ]);
        var first = results.First();
        float[] raw = first.AsEnumerable<float>().ToArray();
        int[] dims = first.AsTensor<float>().Dimensions.ToArray();
        int outHeight = dims.Length >= 2 ? dims[^2] : _height;
        int outWidth = dims.Length >= 2 ? dims[^1] : _width;
        if (outHeight <= 0 || outWidth <= 0 || raw.Length < outHeight * outWidth)
            throw new InvalidOperationException(
                $"Unexpected ONNX dense-map output shape [{string.Join(",", dims)}].");

        int outPlane = outHeight * outWidth;
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < outPlane; i++)
        {
            float v = raw[i];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float scale = max > min ? 255f / (max - min) : 0f;
        byte[] map = new byte[outPlane];
        for (int i = 0; i < outPlane; i++)
        {
            float v = scale == 0f ? 127.5f : (raw[i] - min) * scale;
            if (binarise) v = v > 127.5f ? 255f : 0f;
            if (invert) v = 255f - v;
            map[i] = (byte)Math.Clamp((int)v, 0, 255);
        }
        if (outWidth == width && outHeight == height) return map;
        return RasterOps.Resize(map, outWidth, outHeight, width, height);
    }

    public void Dispose() => _session.Dispose();
}
