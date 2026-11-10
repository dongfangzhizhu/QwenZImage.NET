using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// QZI_* variables are process-wide, but qzen_init() snapshots them once per
/// engine. These tests pin down the scoped publication + restore contract and
/// use the reference backend's deterministic PNG hash to prove that two engines
/// really captured their own configuration.
/// </summary>
public sealed class NativeConfigScopeTests : IDisposable
{
    private readonly Dictionary<string, string?> _original = [];

    // Managed-only variables are not scoped; track them too so a test that
    // sets them cannot leak into the next one (or into CI's DLL resolver).
    private static readonly string[] ManagedOnlyNames =
    [
        QziConfig.NativeLibraryVariable, QziConfig.DeviceIdVariable
    ];

    public NativeConfigScopeTests()
    {
        foreach (string name in NativeConfigScope.VariableNames.Concat(ManagedOnlyNames))
            _original[name] = Environment.GetEnvironmentVariable(name);
    }

    public void Dispose()
    {
        foreach ((string name, string? value) in _original)
            Environment.SetEnvironmentVariable(name, value);
    }

    private static QziClientOptions Full(string suffix) => new()
    {
        NativeLibraryPath = "/native/lib.so",
        ConfigPath = $"/config/{suffix}.json",
        Qwen21Directory = $"/packs/qwen21-{suffix}",
        ZImageDirectory = $"/packs/zimage-{suffix}",
        CacheDirectory = $"/cache/{suffix}",
        Qwen21LoraDirectory = $"/packs/loras-{suffix}",
        ZImageNunchakuDirectory = $"/packs/nunchaku-{suffix}",
        ZImageControlDirectory = $"/packs/zimage-control-{suffix}",
    };

    private static void SetAmbient(string prefix)
    {
        foreach (string name in NativeConfigScope.VariableNames)
            Environment.SetEnvironmentVariable(name, $"{prefix}:{name}");
    }

    private static void AssertAmbient(string prefix)
    {
        foreach (string name in NativeConfigScope.VariableNames)
            Assert.Equal($"{prefix}:{name}", Environment.GetEnvironmentVariable(name));
    }

    private void AssertOriginal()
    {
        foreach ((string name, string? value) in _original)
            Assert.Equal(value, Environment.GetEnvironmentVariable(name));
    }

    private static QziClientOptions WithDirectories(QziClientOptions baseline,
                                                     string lora, string nunchaku) => new()
    {
        NativeLibraryPath = baseline.NativeLibraryPath,
        Qwen21Directory = baseline.Qwen21Directory,
        ZImageDirectory = baseline.ZImageDirectory,
        CacheDirectory = baseline.CacheDirectory,
        Qwen21LoraDirectory = lora,
        ZImageNunchakuDirectory = nunchaku,
        // ConfigPath deliberately stays null: the ambient QZI_CONFIG must be
        // left alone by the scope so a container-only setup keeps working.
    };

    [Fact]
    public void Scope_Publishes_Every_Native_Config_Variable()
    {
        SetAmbient("ambient");
        QziClientOptions options = Full("scoped");

        using (new NativeConfigScope(options))
        {
            Assert.Equal("/config/scoped.json",
                Environment.GetEnvironmentVariable(QziConfig.EnvironmentVariable));
            Assert.Equal("/packs/qwen21-scoped",
                Environment.GetEnvironmentVariable(QziConfig.Qwen21DirectoryVariable));
            Assert.Equal("/packs/zimage-scoped",
                Environment.GetEnvironmentVariable(QziConfig.ZImageDirectoryVariable));
            Assert.Equal("/packs/loras-scoped",
                Environment.GetEnvironmentVariable(QziConfig.LoraDirectoryVariable));
            Assert.Equal("/packs/nunchaku-scoped",
                Environment.GetEnvironmentVariable(QziConfig.NunchakuDirectoryVariable));
            Assert.Equal("/packs/zimage-control-scoped",
                Environment.GetEnvironmentVariable(QziConfig.ZImageControlDirectoryVariable));
            Assert.Equal("/cache/scoped",
                Environment.GetEnvironmentVariable(QziConfig.ModelCacheVariable));
        }

        AssertAmbient("ambient");
    }

    [Fact]
    public void Scope_Restores_Exact_Process_State()
    {
        SetAmbient("ambient");
        // One present, one absent: both must come back exactly as they were.
        Environment.SetEnvironmentVariable(QziConfig.ModelCacheVariable, null);

        using (new NativeConfigScope(Full("scoped")))
            Assert.Equal("/cache/scoped",
                Environment.GetEnvironmentVariable(QziConfig.ModelCacheVariable));

        Assert.Equal("ambient:QZI_CONFIG",
            Environment.GetEnvironmentVariable(QziConfig.EnvironmentVariable));
        Assert.Null(Environment.GetEnvironmentVariable(QziConfig.ModelCacheVariable));
        Assert.Equal("ambient:QZI_QWEN21_LORA_DIR",
            Environment.GetEnvironmentVariable(QziConfig.LoraDirectoryVariable));
    }

    [Fact]
    public void Scope_Leaves_Unset_Options_At_Their_Ambient_Value()
    {
        SetAmbient("ambient");
        var options = new QziClientOptions { Qwen21LoraDirectory = "/only/loras" };

        using (new NativeConfigScope(options))
        {
            Assert.Equal("/only/loras",
                Environment.GetEnvironmentVariable(QziConfig.LoraDirectoryVariable));
            Assert.Equal("ambient:QZI_CONFIG",
                Environment.GetEnvironmentVariable(QziConfig.EnvironmentVariable));
            Assert.Equal("ambient:QZI_ZIMAGE_NUNCHAKU_DIR",
                Environment.GetEnvironmentVariable(QziConfig.NunchakuDirectoryVariable));
        }

        AssertAmbient("ambient");
    }

    [Fact]
    public void Scope_Does_Not_Touch_Managed_Only_Variables()
    {
        // QZI_NATIVE_LIBRARY / QZI_DEVICE_ID are consumed by the managed
        // loader, never by qzen_config_apply_environment(). Scoping them would
        // be pointless and could break the DLL resolver for later clients.
        Environment.SetEnvironmentVariable(QziConfig.NativeLibraryVariable, "/ambient/lib.so");
        Environment.SetEnvironmentVariable(QziConfig.DeviceIdVariable, "3");
        try
        {
            using (new NativeConfigScope(Full("scoped")))
            {
                Assert.Equal("/ambient/lib.so",
                    Environment.GetEnvironmentVariable(QziConfig.NativeLibraryVariable));
                Assert.Equal("3",
                    Environment.GetEnvironmentVariable(QziConfig.DeviceIdVariable));
            }
            Assert.Equal("/ambient/lib.so",
                Environment.GetEnvironmentVariable(QziConfig.NativeLibraryVariable));
            Assert.Equal("3",
                Environment.GetEnvironmentVariable(QziConfig.DeviceIdVariable));
        }
        finally
        {
            Environment.SetEnvironmentVariable(QziConfig.NativeLibraryVariable,
                _original.GetValueOrDefault(QziConfig.NativeLibraryVariable));
            Environment.SetEnvironmentVariable(QziConfig.DeviceIdVariable,
                _original.GetValueOrDefault(QziConfig.DeviceIdVariable));
        }
        AssertOriginal();
    }

    [Fact]
    public void Failed_Client_Creation_Still_Restores_The_Environment()
    {
        SetAmbient("ambient");
        QziClientOptions baseline = NativeTestLibrary.Options();
        // device_id < -1 is rejected by qzi_engine_create(), so construction
        // throws after the scope was applied. The failed scope must unwind.
        QziClientOptions bad = new()
        {
            NativeLibraryPath = baseline.NativeLibraryPath,
            Qwen21Directory = baseline.Qwen21Directory,
            ZImageDirectory = baseline.ZImageDirectory,
            CacheDirectory = baseline.CacheDirectory,
            ConfigPath = "/config/failing.json",
            Qwen21LoraDirectory = "/loras-failing",
            ZImageNunchakuDirectory = "/nunchaku-failing",
            DeviceId = -2
        };

        Assert.Throws<QziException>(() => new QziClient(bad));
        AssertAmbient("ambient");
    }

    [Fact]
    public void Dispose_Is_Idempotent()
    {
        SetAmbient("ambient");
        NativeConfigScope scope = new(Full("scoped"));
        scope.Dispose();
        scope.Dispose();
        AssertAmbient("ambient");
    }

    [Fact]
    public void Two_Engines_Snapshot_Their_Own_Config()
    {
        QziClientOptions baseline = NativeTestLibrary.Options();
        var request = new QwenImageOptions
        {
            Prompt = "a red fox in snow", Width = 32, Height = 32, Seed = 7
        };

        using QziClient a = new(WithDirectories(baseline, "/loras-a", "/nunchaku-a"));
        using QziClient b = new(WithDirectories(baseline, "/loras-b", "/nunchaku-b"));
        using QziClient sameAsA = new(WithDirectories(baseline, "/loras-a", "/nunchaku-a"));

        using GenerationResult ia = a.GenerateQwen21(request);
        using GenerationResult ib = b.GenerateQwen21(request);
        using GenerationResult ic = sameAsA.GenerateQwen21(request);

        byte[] pa = ia.PngSpan.ToArray();
        byte[] pb = ib.PngSpan.ToArray();
        byte[] pc = ic.PngSpan.ToArray();
        Assert.NotEqual(pa, pb);   // each engine kept its own config
        Assert.Equal(pa, pc);      // identical config reproduces identical bytes

        AssertOriginal();
    }

    [Fact]
    public void Concurrent_Client_Creation_Does_Not_Interleave_Scopes()
    {
        QziClientOptions baseline = NativeTestLibrary.Options();
        const int count = 8;
        var clients = new QziClient[count];

        try
        {
            Parallel.For(0, count, i =>
                clients[i] = new QziClient(
                    WithDirectories(baseline, $"/loras-{i}", $"/nunchaku-{i}")));

            var request = new QwenImageOptions
            {
                Prompt = "a red fox in snow", Width = 32, Height = 32, Seed = 11
            };
            var hashes = new HashSet<string>(StringComparer.Ordinal);
            foreach (QziClient client in clients)
            {
                using GenerationResult result = client.GenerateQwen21(request);
                hashes.Add(Convert.ToHexString(
                    System.Security.Cryptography.SHA256.HashData(result.PngSpan)));
            }

            // An interleaving scope would make some engines snapshot another
            // client's directories and produce duplicate images.
            Assert.Equal(count, hashes.Count);
        }
        finally
        {
            foreach (QziClient? client in clients) client?.Dispose();
        }

        AssertOriginal();
    }
}
