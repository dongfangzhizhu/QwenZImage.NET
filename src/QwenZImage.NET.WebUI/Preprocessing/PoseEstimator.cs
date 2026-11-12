using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace QwenZImage.NET.WebUI.Preprocessing;

/// <summary>
/// A direct C# port of WanGP's <c>preprocessing/dwpose</c> ONNX pipeline:
/// YOLOX-L person detection followed by DWPose (RTMPose SimCC) whole-body
/// keypoints, remapped to the OpenPose 18-point layout and painted as a
/// coloured skeleton on a black canvas. Sessions are created lazily and kept
/// for the lifetime of the WebUI process; nothing is loaded when <c>pose</c>
/// is never requested.
/// </summary>
internal sealed class PoseEstimator : IDisposable
{
    private const int DetectorSize = 640;
    private const float PersonScoreThreshold = 0.3f;
    private const float NmsThreshold = 0.45f;
    private const float CandidateScoreThreshold = 0.1f;
    private const float SimccSplitRatio = 2.0f;

    // ImageNet RGB normalisation used by RTMPose (onnxpose.py).
    private static readonly float[] Mean = [123.675f, 116.28f, 103.53f];
    private static readonly float[] Std = [58.395f, 57.12f, 57.375f];

    // OpenPose limb / colour tables from WanGP preprocessing/dwpose/util.py.
    private static readonly int[,] Limbs =
    {
        { 2, 3 }, { 2, 6 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 7, 8 },
        { 2, 9 }, { 9, 10 }, { 10, 11 }, { 2, 12 }, { 12, 13 }, { 13, 14 },
        { 2, 1 }, { 1, 15 }, { 15, 17 }, { 1, 16 }, { 16, 18 }, { 3, 17 }, { 6, 18 }
    };

    private static readonly byte[,] Colours =
    {
        { 255, 0, 0 }, { 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
        { 170, 255, 0 }, { 85, 255, 0 }, { 0, 255, 0 }, { 0, 255, 85 },
        { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 85, 255 },
        { 0, 0, 255 }, { 85, 0, 255 }, { 170, 0, 255 }, { 255, 0, 255 },
        { 255, 0, 170 }, { 255, 0, 85 }
    };

    // WanGP wholebody.py: mmpose 17-body + neck -> OpenPose 18-point order.
    private static readonly int[] MmPoseIndex = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3];
    private static readonly int[] OpenPoseIndex = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17];

    private readonly InferenceSession _detector;
    private readonly InferenceSession _pose;
    private readonly string _detectorInputName;
    private readonly string _poseInputName;
    private readonly int _poseHeight;
    private readonly int _poseWidth;

    public PoseEstimator(string detectorPath, string poseModelPath)
    {
        _detector = new InferenceSession(detectorPath);
        _pose = new InferenceSession(poseModelPath);
        _detectorInputName = _detector.InputMetadata.Keys.First();
        _poseInputName = _pose.InputMetadata.Keys.First();
        int[] dims = _pose.InputMetadata[_poseInputName].Dimensions;
        // [N, 3, H, W]; WanGP's dw-ll_ucoco_384.onnx is [N,3,384,288].
        _poseHeight = dims[2] > 0 ? dims[2] : 384;
        _poseWidth = dims[3] > 0 ? dims[3] : 288;
    }

    /// <summary>
    /// Returns an interleaved RGB8 canvas of the same size as the input, with
    /// an OpenPose-style skeleton drawn on black. An image with no detectable
    /// person produces an all-black guide (WanGP behaves the same way).
    /// </summary>
    public byte[] Estimate(byte[] interleavedRgb, int width, int height)
    {
        byte[] canvas = new byte[checked(width * height * 3)];
        var skeletons = new List<(float X, float Y, float Score)[]>();
        List<float[]> boxes = Detect(interleavedRgb, width, height);
        if (boxes.Count == 0) boxes.Add([0f, 0f, width, height]);

        // Build the batched top-down affine crops ([N,3,H,W]).
        int cropPixels = _poseWidth * _poseHeight;
        float[] input = new float[checked(boxes.Count * 3 * cropPixels)];
        float[] centers = new float[boxes.Count * 2];
        float[] scales = new float[boxes.Count * 2];

        for (int b = 0; b < boxes.Count; b++)
        {
            float[] box = boxes[b];
            float cx = (box[0] + box[2]) * 0.5f;
            float cy = (box[1] + box[3]) * 0.5f;
            float sw = (box[2] - box[0]) * 1.25f;
            float sh = (box[3] - box[1]) * 1.25f;
            FixAspectRatio(ref sw, ref sh, (float)_poseWidth / _poseHeight);
            centers[b * 2] = cx; centers[b * 2 + 1] = cy;
            scales[b * 2] = sw; scales[b * 2 + 1] = sh;
            WarpCrop(interleavedRgb, width, height, cx, cy, sw, sh,
                     input, b * 3 * cropPixels);
        }

        using var results = _pose.Run([
            NamedOnnxValue.CreateFromTensor(_poseInputName,
                new DenseTensor<float>(input, [boxes.Count, 3, _poseHeight, _poseWidth]))
        ]);
        var outputs = results.ToList();
        if (outputs.Count < 2)
            throw new InvalidOperationException("The DWPose ONNX model must expose simcc_x and simcc_y outputs.");
        int[] xDims = outputs[0].AsTensor<float>().Dimensions.ToArray();
        int[] yDims = outputs[1].AsTensor<float>().Dimensions.ToArray();
        float[] xData = outputs[0].AsEnumerable<float>().ToArray();
        float[] yData = outputs[1].AsEnumerable<float>().ToArray();
        int keypoints = xDims.Length == 3 ? xDims[1] : 133;
        int widthBins = xDims[^1];
        int heightBins = yDims[^1];

        for (int b = 0; b < boxes.Count; b++)
        {
            float cx = centers[b * 2], cy = centers[b * 2 + 1];
            float sw = scales[b * 2], sh = scales[b * 2 + 1];
            var points = new (float X, float Y, float Score)[18];
            // Body keypoints 0..16 in mmpose order.
            for (int k = 0; k < 17 && k < keypoints; k++)
            {
                int xBase = b * keypoints * widthBins + k * widthBins;
                int yBase = b * keypoints * heightBins + k * heightBins;
                ArgMax(xData.AsSpan(xBase, widthBins), out int xLoc, out float xMax);
                ArgMax(yData.AsSpan(yBase, heightBins), out int yLoc, out float yMax);
                float score = MathF.Min(xMax, yMax);
                float px = xLoc / SimccSplitRatio;
                float py = yLoc / SimccSplitRatio;
                points[k] = (px / _poseWidth * sw + cx - sw * 0.5f,
                             py / _poseHeight * sh + cy - sh * 0.5f,
                             score);
            }
            // Neck = mean of shoulders 5/6, inserted at index 17.
            points[17] = ((points[5].X + points[6].X) * 0.5f,
                          (points[5].Y + points[6].Y) * 0.5f,
                          points[5].Score > 0.3f && points[6].Score > 0.3f ? 1f : 0f);

            // Permute to OpenPose order using a snapshot of the source array.
            var openPose = new (float X, float Y, float Score)[18];
            Array.Copy(points, openPose, 18);
            for (int i = 0; i < MmPoseIndex.Length; i++)
                openPose[OpenPoseIndex[i]] = points[MmPoseIndex[i]];

            skeletons.Add(openPose);
            DrawLimbs(canvas, width, height, openPose);
        }

        // WanGP dims the limb canvas by 0.6 before drawing brighter joints.
        for (int i = 0; i < canvas.Length; i++) canvas[i] = (byte)(canvas[i] * 0.6f);
        float thickness = Math.Max(2f, Math.Min(width, height) / 256f * 4f);
        foreach (var points in skeletons)
            for (int i = 0; i < 18; i++)
            {
                if (points[i].Score <= 0.3f) continue;
                RasterOps.FillDisc(canvas, width, height, points[i].X, points[i].Y,
                                   thickness, Colours[i, 0], Colours[i, 1], Colours[i, 2]);
            }
        return canvas;
    }

    private static void DrawLimbs(byte[] canvas, int width, int height,
                                  (float X, float Y, float Score)[] points)
    {
        float thickness = Math.Max(2f, Math.Min(width, height) / 256f * 4f);
        int limbs = Math.Min(17, Limbs.GetLength(0));
        for (int i = 0; i < limbs; i++)
        {
            int a = Limbs[i, 0] - 1, b = Limbs[i, 1] - 1;
            if (a < 0 || b < 0 || a >= 18 || b >= 18) continue;
            if (points[a].Score <= 0.3f || points[b].Score <= 0.3f) continue;
            RasterOps.DrawLine(canvas, width, height, points[a].X, points[a].Y,
                               points[b].X, points[b].Y, thickness / 2f,
                               Colours[i, 0], Colours[i, 1], Colours[i, 2]);
        }
    }

    private void WarpCrop(byte[] rgb, int width, int height,
                          float cx, float cy, float sw, float sh,
                          float[] destination, int offset)
    {
        int plane = _poseWidth * _poseHeight;
        for (int y = 0; y < _poseHeight; y++)
        {
            float sy = cy + (y - _poseHeight * 0.5f) * (sh / _poseHeight);
            for (int x = 0; x < _poseWidth; x++)
            {
                float sx = cx + (x - _poseWidth * 0.5f) * (sw / _poseWidth);
                for (int c = 0; c < 3; c++)
                {
                    float value = Sample(rgb, width, height, sx, sy, c);
                    destination[offset + c * plane + y * _poseWidth + x] =
                        (value - Mean[c]) / Std[c];
                }
            }
        }
    }

    private static float Sample(byte[] rgb, int width, int height, float x, float y, int channel)
    {
        int x0 = Math.Clamp((int)MathF.Floor(x), 0, width - 1);
        int y0 = Math.Clamp((int)MathF.Floor(y), 0, height - 1);
        int x1 = Math.Clamp(x0 + 1, 0, width - 1);
        int y1 = Math.Clamp(y0 + 1, 0, height - 1);
        float fx = Math.Clamp(x - x0, 0f, 1f), fy = Math.Clamp(y - y0, 0f, 1f);
        float a = rgb[(y0 * width + x0) * 3 + channel] * (1 - fx) +
                  rgb[(y0 * width + x1) * 3 + channel] * fx;
        float b = rgb[(y1 * width + x0) * 3 + channel] * (1 - fx) +
                  rgb[(y1 * width + x1) * 3 + channel] * fx;
        float v = a * (1 - fy) + b * fy;
        return Math.Clamp(v, 0f, 255f);
    }

    private static void FixAspectRatio(ref float w, ref float h, float aspect)
    {
        if (w > h * aspect) h = w / aspect;
        else w = h * aspect;
    }

    private static void ArgMax(ReadOnlySpan<float> values, out int index, out float max)
    {
        index = 0;
        max = values.Length > 0 ? values[0] : 0f;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) { max = values[i]; index = i; }
    }

    /// <summary>YOLOX-L detection ported from onnxdet.py (letterbox + NMS).</summary>
    private List<float[]> Detect(byte[] rgb, int width, int height)
    {
        float ratio = MathF.Min(DetectorSize / (float)height, DetectorSize / (float)width);
        int newWidth = Math.Max(1, (int)(width * ratio));
        int newHeight = Math.Max(1, (int)(height * ratio));
        byte[] resized = RasterOps.ResizeRgb(rgb, width, height, newWidth, newHeight);

        int plane = DetectorSize * DetectorSize;
        float[] input = new float[checked(3 * plane)];
        Array.Fill(input, 114f);
        for (int y = 0; y < newHeight; y++)
        for (int x = 0; x < newWidth; x++)
        for (int c = 0; c < 3; c++)
            input[c * plane + y * DetectorSize + x] = resized[(y * newWidth + x) * 3 + c];

        using var results = _detector.Run([
            NamedOnnxValue.CreateFromTensor(_detectorInputName,
                new DenseTensor<float>(input, [1, 3, DetectorSize, DetectorSize]))
        ]);
        float[] output = results.First().AsEnumerable<float>().ToArray();

        var candidates = new List<(float X1, float Y1, float X2, float Y2, float Score)>();
        int anchor = 0;
        foreach (int stride in new[] { 8, 16, 32 })
        {
            int cells = DetectorSize / stride;
            for (int gy = 0; gy < cells; gy++)
            for (int gx = 0; gx < cells; gx++, anchor++)
            {
                int i = anchor * 85;
                if (i + 85 > output.Length) break;
                float obj = output[i + 4];
                float score = obj * output[i + 5]; // COCO class 0 = person
                if (score <= CandidateScoreThreshold) continue;
                float cx = (output[i] + gx) * stride;
                float cy = (output[i + 1] + gy) * stride;
                float bw = MathF.Exp(output[i + 2]) * stride;
                float bh = MathF.Exp(output[i + 3]) * stride;
                candidates.Add(((cx - bw * 0.5f) / ratio, (cy - bh * 0.5f) / ratio,
                                (cx + bw * 0.5f) / ratio, (cy + bh * 0.5f) / ratio, score));
            }
        }

        // Greedy single-class NMS, identical to onnxdet.nms / multiclass_nms
        // restricted to the only class we keep (person).
        candidates.Sort(static (a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<float[]>();
        var suppressed = new bool[candidates.Count];
        for (int i = 0; i < candidates.Count; i++)
        {
            if (suppressed[i] || candidates[i].Score <= PersonScoreThreshold) continue;
            var box = candidates[i];
            kept.Add([box.X1, box.Y1, box.X2, box.Y2]);
            float area = MathF.Max(0, box.X2 - box.X1 + 1) * MathF.Max(0, box.Y2 - box.Y1 + 1);
            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (suppressed[j]) continue;
                var other = candidates[j];
                float iw = MathF.Max(0, MathF.Min(box.X2, other.X2) - MathF.Max(box.X1, other.X1) + 1);
                float ih = MathF.Max(0, MathF.Min(box.Y2, other.Y2) - MathF.Max(box.Y1, other.Y1) + 1);
                float inter = iw * ih;
                float otherArea = MathF.Max(0, other.X2 - other.X1 + 1) *
                                  MathF.Max(0, other.Y2 - other.Y1 + 1);
                float union = area + otherArea - inter;
                if (union > 0 && inter / union > NmsThreshold) suppressed[j] = true;
            }
        }
        return kept;
    }

    public void Dispose()
    {
        _pose.Dispose();
        _detector.Dispose();
    }
}
