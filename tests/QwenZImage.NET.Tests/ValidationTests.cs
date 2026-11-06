using QwenZImage.NET.Validation;
using Xunit;

namespace QwenZImage.NET.Tests;

public class ValidationTests
{
    private static QwenImageOptions Qwen() => new() { Prompt = "a red fox" };
    private static ZImageOptions ZImage() => new() { Prompt = "a red fox" };

    [Fact]
    public void Defaults_AreValid()
    {
        RequestValidator.Validate(Qwen());
        RequestValidator.Validate(ZImage());
        Assert.Equal(8, Qwen().EffectiveSteps);
        Assert.Equal(1.0f, Qwen().EffectiveGuidance);
        Assert.Equal(40, (Qwen() with { Tier = QwenImageTier.Base }).EffectiveSteps);
        Assert.Equal(4.0f, (Qwen() with { Tier = QwenImageTier.Base }).EffectiveGuidance);
    }

    [Fact]
    public void Turbo_Fixed_Sampler_Is_Enforced()
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Steps = 20 }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Guidance = 4.0f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { NegativePrompt = "blurry" }));
        RequestValidator.Validate(Qwen() with { Steps = 8, Guidance = 1.0f });
    }

    [Theory]
    [InlineData(0)]
    [InlineData(51)]
    [InlineData(-1)]
    public void Base_Steps_Range(int steps)
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Tier = QwenImageTier.Base, Steps = steps }));
    }

    [Fact]
    public void Base_Guidance_Range()
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Tier = QwenImageTier.Base, Guidance = 0.5f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Tier = QwenImageTier.Base, Guidance = 21f }));
        RequestValidator.Validate(Qwen() with { Tier = QwenImageTier.Base, Guidance = 4.0f });
    }

    [Theory]
    [InlineData(100, 1024)]
    [InlineData(1024, 100)]
    [InlineData(0, 1024)]
    [InlineData(2080, 1024)]
    public void Qwen21_Canvas_Must_Be_Multiple_Of_32(int width, int height)
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Width = width, Height = height }));
    }

    [Fact]
    public void Qwen21_Canvas_Area_Limit()
    {
        // 2048x2048 is exactly at the limit and therefore valid; the side
        // limit already bounds the area, so both rules are asserted here.
        RequestValidator.Validate(Qwen() with { Width = 2048, Height = 2048 });
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Width = 2048, Height = 2080 }));
    }

    [Fact]
    public void ZImage_Canvas_Must_Be_Multiple_Of_16()
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            ZImage() with { Width = 100, Height = 100 }));
        RequestValidator.Validate(ZImage() with { Width = 1008, Height = 1024 });
    }

    [Fact]
    public void Lora_Rules()
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { LoraScale = 2.0f }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Lora = "../escape.safetensors" }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Lora = "a.safetensors", LoraScale = 5.0f }));
        RequestValidator.Validate(Qwen() with { Lora = "a.safetensors", LoraScale = 1.0f });
    }

    [Fact]
    public void ZImage_Checkpoint_Rules()
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            ZImage() with { Checkpoint = "svdq-fp4-r256" }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            ZImage() with { Checkpoint = "not-a-pack" }));
        RequestValidator.Validate(ZImage() with { Checkpoint = "nunchaku-r256" });
        RequestValidator.Validate(ZImage() with { Checkpoint = null });
    }

    [Fact]
    public void Prompt_Limit_Is_In_Utf8_Bytes()
    {
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Prompt = new string('x', RequestValidator.MaxPromptBytes + 1) }));
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Prompt = "" }));
        // 3 bytes per character: the byte limit, not the char limit, is what matters.
        Assert.Throws<ArgumentException>(() => RequestValidator.Validate(
            Qwen() with { Prompt = new string('汉', RequestValidator.MaxPromptBytes / 3 + 1) }));
    }

    [Fact]
    public void All_Presets_Are_Valid_Once_A_Prompt_Is_Supplied()
    {
        foreach (GenerationPreset preset in QziPresets.All)
        {
            GenerationOptions options = QziPresets.WithPrompt(preset, "a red fox in snow");
            RequestValidator.Validate(options);
        }
    }
}
