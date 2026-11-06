using Xunit;

namespace QwenZImage.NET.Tests;

/// <summary>
/// The config file is now the primary way model packs reach the native
/// backend, so its precedence rules and failure modes are part of the public
/// contract. All tests use an explicit path; no ambient config file is read.
/// </summary>
public class QziConfigTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(),
        $"qzi-config-{Guid.NewGuid():N}.json");

    public void Dispose() => File.Delete(_path);

    private QziConfig Write(string json)
    {
        File.WriteAllText(_path, json);
        return QziConfig.Load(_path);
    }

    [Fact]
    public void TopLevel_Values_Are_Loaded()
    {
        QziConfig config = Write("""
            {
              "qwen21_dir": "/packs/qwen21",
              "zimage_dir": "/packs/zimage",
              "text_cache_dir": "/cache",
              "device_id": 2,
              "thread_count": 6
            }
            """);

        Assert.Equal("/packs/qwen21", config.Qwen21Directory);
        Assert.Equal("/packs/zimage", config.ZImageDirectory);
        Assert.Equal("/cache", config.TextCacheDirectory);
        Assert.Equal(2, config.DeviceId);
        Assert.Equal(6, config.ThreadCount);
        Assert.Equal(_path, config.SourcePath);
        Assert.False(config.IsEmpty);
    }

    [Fact]
    public void Platform_Section_Overrides_TopLevel()
    {
        // Cover all three platform names so the assertion is OS-independent.
        QziConfig config = Write("""
            {
              "qwen21_dir": "top",
              "windows": { "qwen21_dir": "plat" },
              "linux":   { "qwen21_dir": "plat" },
              "osx":     { "qwen21_dir": "plat" }
            }
            """);

        Assert.Equal("plat", config.Qwen21Directory);
    }

    [Fact]
    public void Underscore_Keys_Are_Comments()
    {
        QziConfig config = Write("""
            { "_comment": "documentation", "_note": "more documentation",
              "zimage_nunchaku_dir": "/packs/svdq" }
            """);

        Assert.Equal("/packs/svdq", config.ZImageNunchakuDirectory);
    }

    [Fact]
    public void Unknown_Key_Is_Rejected()
    {
        File.WriteAllText(_path, """{ "qwen21dir": "/packs/qwen21" }""");
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => QziConfig.Load(_path));
        Assert.Contains("unknown config key 'qwen21dir'", error.Message);
    }

    [Fact]
    public void Wrong_Type_Is_Rejected()
    {
        File.WriteAllText(_path, """{ "device_id": "auto" }""");
        Assert.Throws<InvalidDataException>(() => QziConfig.Load(_path));
    }

    [Fact]
    public void Missing_Explicit_File_Throws()
    {
        string missing = _path + ".missing";
        Assert.Throws<FileNotFoundException>(() => QziConfig.Load(missing));
    }

    [Fact]
    public void Tilde_Is_Expanded_To_The_Home_Directory()
    {
        Assert.Equal(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                     QziConfig.Expand("~"));
        string? expanded = QziConfig.Expand("~/packs");
        Assert.NotNull(expanded);
        Assert.StartsWith(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), expanded);
        Assert.Equal("relative/path", QziConfig.Expand("relative/path"));
    }

    [Fact]
    public void ToClientOptions_Carries_The_Config_Path()
    {
        QziConfig config = Write("""
            { "qwen21_dir": "/packs/qwen21", "zimage_dir": "/packs/zimage",
              "text_cache_dir": "/cache", "native_library": "/native/lib.so" }
            """);

        QziClientOptions options = config.ToClientOptions();
        Assert.Equal("/packs/qwen21", options.Qwen21Directory);
        Assert.Equal("/packs/zimage", options.ZImageDirectory);
        Assert.Equal("/cache", options.CacheDirectory);
        Assert.Equal("/native/lib.so", options.NativeLibraryPath);
        Assert.Equal(_path, options.ConfigPath);
    }

    [Fact]
    public void FromConfig_Uses_An_Explicit_Path()
    {
        File.WriteAllText(_path, """{ "qwen21_dir": "/packs/qwen21" }""");
        QziClientOptions options = QziClientOptions.FromConfig(_path);
        Assert.Equal(_path, options.ConfigPath);
    }

    [Fact]
    public void With_Rejects_Unknown_Keys()
        => Assert.Throws<ArgumentException>(() => new QziConfig().With("typo", "x"));

    [Fact]
    public void SearchPaths_Walk_Up_From_The_Project_Directory()
    {
        // `dotnet run` starts the app in the project directory, several levels
        // below the solution root where QwenZImage.config.json normally lives.
        DirectoryInfo? parent = Directory.GetParent(Directory.GetCurrentDirectory());
        Assert.NotNull(parent);

        IReadOnlyList<string> paths = QziConfig.SearchPaths();
        Assert.Contains(Path.Combine(parent.FullName, QziConfig.FileName), paths);
        Assert.Equal(paths.Count, paths.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Contains(paths, p => p.EndsWith(Path.Combine(".config", "QwenZImage", "config.json"),
                                              StringComparison.OrdinalIgnoreCase));
    }
}
