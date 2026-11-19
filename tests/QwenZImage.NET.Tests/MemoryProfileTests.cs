using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// Sixth additive ABI v1 increment: low-VRAM / memory profile. Covers capability
/// probing, the deterministic native round trip (resolve / set / plan / VRAM
/// health) and managed validation. No GPU or model weights needed: the reference
/// backend reports a fixed 16 GiB device, so AUTO resolves to "balanced".
/// </summary>
public class MemoryProfileTests
{
    [Fact]
    public void LowVram_Capability_Is_Detected_On_The_Matching_Library()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Assert.True(client.SupportsLowVram);
    }

    [Fact]
    public void Auto_Resolves_From_Reference_Device_Vram_To_Balanced()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        QziMemoryPlan plan = client.GetMemoryPlan();

        Assert.Equal(QziMemoryProfile.Balanced, plan.Profile);
        Assert.Equal(16384, plan.TotalVramMib);
        Assert.Equal(12288, plan.FreeVramMib);
        Assert.Equal(4096, plan.PoolReleaseMib);
        Assert.Equal(768, plan.ReferenceEncodeCap);
    }

    [Fact]
    public void Health_Exposes_Real_Or_Synthetic_Vram_Not_Minus_One()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        EngineHealth health = client.GetHealth();
        Assert.True(health.TotalVramMib > 0);
        Assert.True(health.FreeVramMib >= 0);
        Assert.True(health.IsReferenceBackend);
    }

    [Theory]
    [InlineData(QziMemoryProfile.High, 24576, 768)]
    [InlineData(QziMemoryProfile.Balanced, 4096, 768)]
    [InlineData(QziMemoryProfile.Low, 1024, 512)]
    [InlineData(QziMemoryProfile.Lowest, 256, 384)]
    public void Concrete_Profiles_Tighten_Pool_And_Encode_Budgets(
        QziMemoryProfile profile, int expectedPool, int expectedCap)
    {
        using QziClient client = new(NativeTestLibrary.Options());
        client.SetMemoryProfile(profile);
        QziMemoryPlan plan = client.GetMemoryPlan();

        Assert.Equal(profile, plan.Profile);
        Assert.Equal(expectedPool, plan.PoolReleaseMib);
        Assert.Equal(expectedCap, plan.ReferenceEncodeCap);
    }

    [Fact]
    public void Auto_After_Concrete_Re_Resolves_From_Device_Vram()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        client.SetMemoryProfile(QziMemoryProfile.Lowest);
        Assert.Equal(QziMemoryProfile.Lowest, client.GetMemoryPlan().Profile);

        client.SetMemoryProfile(QziMemoryProfile.Auto);
        Assert.Equal(QziMemoryProfile.Balanced, client.GetMemoryPlan().Profile);
    }

    [Fact]
    public void Profile_Change_Does_Not_Consume_A_Generation_Slot()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        ulong servedBefore = client.GetHealth().Served;
        ulong rejectedBefore = client.GetHealth().Rejected;

        client.SetMemoryProfile(QziMemoryProfile.Low);
        _ = client.GetMemoryPlan();

        EngineHealth after = client.GetHealth();
        Assert.Equal(servedBefore, after.Served);
        Assert.Equal(rejectedBefore, after.Rejected);
    }

    [Fact]
    public void Out_Of_Range_Profile_Is_Rejected_By_Native()
    {
        using QziClient client = new(NativeTestLibrary.Options());
        Assert.Throws<QziException>(() => client.SetMemoryProfile((QziMemoryProfile)99));
    }
}
