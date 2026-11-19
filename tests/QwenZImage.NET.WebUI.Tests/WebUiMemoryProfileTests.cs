using System.Text.Json;
using QwenZImage.NET;
using QwenZImage.NET.WebUI;
using Xunit;

namespace QwenZImage.NET.WebUI.Tests;

/// <summary>
/// Wire-level coverage for the sixth additive ABI v1 increment (low-VRAM /
/// memory profile): the config key reaches client options, the health record
/// serialises the capability flag and resolved budget under stable names, and
/// the profile request binds case-insensitively.
/// </summary>
public class WebUiMemoryProfileTests
{
    [Fact]
    public void Config_Key_Reaches_Client_Options()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"qzi-mem-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, QziConfig.FileName);
            File.WriteAllText(path, """
                { "memory_profile": "low", "device_id": 0 }
                """);
            QziConfig config = QziConfig.Load(path);
            Assert.Equal("low", config.MemoryProfile);
            Assert.Equal("low", config.ToClientOptions().MemoryProfile);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Health_Record_Serialises_Stable_Snake_Case_Names()
    {
        WebUiEngineHealth health = new()
        {
            SupportsLowVram = true,
            MemoryProfile = "lowest",
            PoolReleaseMib = 256,
            ReferenceEncodeCap = 384,
            FreeVramMib = 6144,
            TotalVramMib = 8192
        };

        string json = JsonSerializer.Serialize(health, WebUiJson.Options);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;

        Assert.True(root.GetProperty("supports_low_vram").GetBoolean());
        Assert.Equal("lowest", root.GetProperty("memory_profile").GetString());
        Assert.Equal(256, root.GetProperty("pool_release_mib").GetInt32());
        Assert.Equal(384, root.GetProperty("reference_encode_cap").GetInt32());
        Assert.Equal(6144, root.GetProperty("free_vram_mib").GetInt64());
    }

    [Theory]
    [InlineData("auto", QziMemoryProfile.Auto)]
    [InlineData("HIGH", QziMemoryProfile.High)]
    [InlineData("balanced", QziMemoryProfile.Balanced)]
    [InlineData("low", QziMemoryProfile.Low)]
    [InlineData("lowest", QziMemoryProfile.Lowest)]
    public void Profile_Request_Parses_Case_Insensitively(string value, QziMemoryProfile expected)
    {
        Assert.True(Enum.TryParse<QziMemoryProfile>(value, true, out QziMemoryProfile parsed));
        Assert.Equal(expected, parsed);
        Assert.True(Enum.IsDefined(parsed));
    }

    [Fact]
    public void Invalid_Profile_Is_Rejected()
    {
        Assert.False(Enum.TryParse<QziMemoryProfile>("turbo", true, out QziMemoryProfile parsed));
        Assert.False(Enum.IsDefined((QziMemoryProfile)99));
    }
}
