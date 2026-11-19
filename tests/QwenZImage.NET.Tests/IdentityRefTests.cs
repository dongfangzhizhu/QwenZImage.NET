using QwenZImage.NET.Validation;
using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Fifth additive ABI v1 increment: identity references (Wan2GP image_refs).
/// Covers capability probing, the deterministic native round trip, phase
/// reporting and the managed validation mirror. No GPU or model weights needed.
/// </summary>
public class IdentityRefTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static QziRasterImage SolidRgb(int size, byte value)
        => QziRasterImage.FromPlanar(
            Enumerable.Repeat(value, size * size * 3).ToArray(), size, size, 3);

    private static Qwen21IdentityRefOptions Options(int count = 2, byte value = 0x33)
        => new()
        {
            Prompt = "same person, different clothes",
            Width = 64,
            Height = 64,
            Seed = 42,
            References = Enumerable.Range(0, count)
                                   .Select(_ => SolidRgb(32, value)).ToArray(),
            IdentityRefWeight = 70.0f,
            ReferenceEncodeMaxSide = 768u
        };

    [Fact]
    public void IdentityReference_Capability_Is_Detected_On_The_Matching_Library()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Assert.True(client.SupportsIdentityReference);
    }

    [Fact]
    public void IdentityRef_Generates_A_Png_With_One_Encode_Stage_Per_Reference()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        List<GenerationProgress> progress = [];
        var collector = new CollectingProgress<GenerationProgress>(progress);

        using GenerationResult result = client.GenerateQwen21IdentityRef(
            Options(count: 3), collector);

        Assert.Equal(PngSignature, result.PngSpan[..8].ToArray());
        Assert.Equal(64, result.Width);
        Assert.Equal(64, result.Height);
        Assert.Equal(42ul, result.Seed);
        Assert.Equal(GenerationModel.Qwen21, result.Model);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Text);
        Assert.Contains(progress, p => p.Phase == GenerationPhase.Vae);
        // Each reference VAE encode emits a start and an end progress event.
        Assert.Equal(6, progress.Count(p => p.Phase == GenerationPhase.ImageEncode));
        Assert.Equal(8, progress.Count(p => p.Phase == GenerationPhase.Dit));
    }

    [Fact]
    public void IdentityRef_Is_Deterministic_And_Every_Knob_Changes_The_Output()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Qwen21IdentityRefOptions baseline = Options();

        using GenerationResult a = client.GenerateQwen21IdentityRef(baseline);
        using GenerationResult b = client.GenerateQwen21IdentityRef(baseline);
        Assert.Equal(a.PngSpan.ToArray(), b.PngSpan.ToArray());

        using GenerationResult otherPixels = client.GenerateQwen21IdentityRef(
            baseline with { References = [SolidRgb(32, 0x33), SolidRgb(32, 0xC0)] });
        Assert.NotEqual(a.PngSpan.ToArray(), otherPixels.PngSpan.ToArray());

        using GenerationResult otherWeight = client.GenerateQwen21IdentityRef(
            baseline with { IdentityRefWeight = 90.0f });
        Assert.NotEqual(a.PngSpan.ToArray(), otherWeight.PngSpan.ToArray());

        using GenerationResult otherCap = client.GenerateQwen21IdentityRef(
            baseline with { ReferenceEncodeMaxSide = 512u });
        Assert.NotEqual(a.PngSpan.ToArray(), otherCap.PngSpan.ToArray());

        using GenerationResult oneReference = client.GenerateQwen21IdentityRef(
            Options(count: 1));
        Assert.NotEqual(a.PngSpan.ToArray(), oneReference.PngSpan.ToArray());
    }

    [Fact]
    public void Generate_Dispatcher_WriteTo_And_Async_All_Accept_Identity_Options()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Qwen21IdentityRefOptions options = Options();

        using GenerationResult direct = client.Generate(options);
        using GenerationResult typed = client.GenerateQwen21IdentityRef(options);
        Assert.Equal(direct.PngSpan.ToArray(), typed.PngSpan.ToArray());

        using MemoryStream stream = new();
        client.WriteTo(stream, options);
        Assert.Equal(direct.PngSpan.ToArray(), stream.ToArray());
    }

    [Fact]
    public async Task Async_IdentityRef_Matches_The_Synchronous_Call()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Qwen21IdentityRefOptions options = Options();
        using GenerationResult sync = client.GenerateQwen21IdentityRef(options);
        using GenerationResult asyncResult = await client.GenerateQwen21IdentityRefAsync(options);
        Assert.Equal(sync.PngSpan.ToArray(), asyncResult.PngSpan.ToArray());
    }

    [Fact]
    public void Managed_Validator_Rejects_Inconsistent_Identity_Requests()
    {
        Qwen21IdentityRefOptions baseline = Options();

        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { References = [] }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with
            {
                References = Enumerable.Range(0, Qwen21IdentityRef.MaxReferences + 1)
                                       .Select(_ => SolidRgb(32, 0x33)).ToArray()
            }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { IdentityRefWeight = 19.0f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { IdentityRefWeight = 101.0f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { ReferenceEncodeMaxSide = 17u }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { ReferenceEncodeMaxSide = 2064u }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { Flags = 1u }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            baseline with { References = [QziRasterImage.SolidMask(32, 32, 255)] }));

        // 0 is the ABI sentinel for the engine default and is accepted.
        RequestValidator.Validate(baseline with
        {
            IdentityRefWeight = 0.0f,
            ReferenceEncodeMaxSide = 0u
        });
    }

    private sealed class CollectingProgress<T>(List<T> items) : IProgress<T>
    {
        public void Report(T value) => items.Add(value);
    }
}
