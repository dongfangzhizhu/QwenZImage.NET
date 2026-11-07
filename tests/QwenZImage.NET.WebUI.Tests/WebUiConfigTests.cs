using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// The WebUI is now a thin host over the in-process native client: its only
/// job is to resolve QwenZImage.config.json and apply explicit overrides.
/// </summary>
public class WebUiConfigTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(),
        $"qzi-webui-{Guid.NewGuid():N}");
    private readonly string _configPath;

    public WebUiConfigTests()
    {
        Directory.CreateDirectory(_dir);
        _configPath = Path.Combine(_dir, QziConfig.FileName);
        File.WriteAllText(_configPath, """
            {
              "qwen21_dir": "/from-config/qwen21",
              "zimage_dir": "/from-config/zimage",
              "text_cache_dir": "/from-config/cache",
              "device_id": 3
            }
            """);
    }

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void Config_File_Is_The_Source_Of_Model_Paths()
    {
        WebUiConfig config = WebUiConfig.FromEnvironment(_dir, ["--config", _configPath]);

        Assert.Equal("/from-config/qwen21", config.ClientOptions.Qwen21Directory);
        Assert.Equal("/from-config/zimage", config.ClientOptions.ZImageDirectory);
        Assert.Equal("/from-config/cache", config.ClientOptions.CacheDirectory);
        Assert.Equal(3, config.ClientOptions.DeviceId);
        Assert.Equal(_configPath, config.ConfigPath);
        Assert.Equal("/from-config/qwen21", config.ModelDirectory);
    }

    [Fact]
    public void Explicit_Models_Switch_Overrides_Both_Packs()
    {
        WebUiConfig config = WebUiConfig.FromEnvironment(
            _dir, ["--config", _configPath, "--models", "/overridden"]);

        Assert.Equal("/overridden", config.ClientOptions.Qwen21Directory);
        Assert.Equal("/overridden", config.ClientOptions.ZImageDirectory);
        Assert.Equal(Path.Combine(_dir, "outputs"), config.OutputDirectory);
    }

    [Theory]
    [InlineData("turbo")]
    [InlineData("base")]
    public void Lora_Is_Mapped_On_Both_Tiers(string tier)
    {
        GenerateRequest request = new()
        {
            Model = "qwen21", Prompt = "a red fox", Tier = tier,
            Lora = "dmd2.safetensors", LoraScale = 1.5f,
            Steps = tier == "base" ? 20 : null,
            Guidance = tier == "base" ? 4.0f : null
        };

        QwenImageOptions options = Assert.IsType<QwenImageOptions>(RequestMapper.ToOptions(request));
        Assert.Equal("dmd2.safetensors", options.Lora);
        Assert.Equal(1.5f, options.LoraScale);
    }

    [Fact]
    public void Config_Propagates_Lora_And_Nunchaku_Directories()
    {
        string path = Path.Combine(_dir, "with-loras.json");
        File.WriteAllText(path, """
            { "qwen21_lora_dir": "/webui/loras",
              "zimage_nunchaku_dir": "/webui/svdq" }
            """);

        WebUiConfig config = WebUiConfig.FromEnvironment(_dir, ["--config", path]);
        Assert.Equal("/webui/loras", config.ClientOptions.Qwen21LoraDirectory);
        Assert.Equal("/webui/svdq", config.ClientOptions.ZImageNunchakuDirectory);
        Assert.Equal(path, config.ConfigPath);
    }

    [Fact]
    public void Output_File_Names_Are_Validated()
    {
        Assert.True(OutputStore.IsSafeName("job-42.png"));
        Assert.False(OutputStore.IsSafeName("../secret.png"));
        Assert.False(OutputStore.IsSafeName("sub/file.png"));
        Assert.False(OutputStore.IsSafeName(".."));
        Assert.False(OutputStore.IsSafeName(".hidden"));
    }

    [Fact]
    public void Health_Record_Defaults_Are_Sane()
    {
        WebUiEngineHealth health = new();
        Assert.Equal("none", health.ResidentModel);
        Assert.Equal("unknown", health.Backend);
        Assert.False(health.IsReferenceBackend);
    }
}
