namespace QwenZImage.NET;

/// <summary>
/// Temporarily publishes the <c>QZI_*</c> variables that the native config
/// loader consults inside <c>qzi_engine_create()</c>, then restores the exact
/// previous process environment.
/// </summary>
/// <remarks>
/// <para>
/// The C ABI carries only three directories. The remaining fields reach the
/// engine through <c>QZI_CONFIG</c> (which config file to read) and the
/// <c>QZI_*</c> overrides in <c>qzen_config_apply_environment()</c>. Those are
/// process-wide, so publishing them permanently means the last constructed
/// client silently wins for every client in the process. Scoping them to the
/// create call lets each engine snapshot its own values; <see cref="QziClient"/>
/// serialises construction so two scopes can never interleave.
/// </para>
/// <para>
/// An option that is null / blank is deliberately left untouched: the ambient
/// value may be a legitimate container-level setting (or the file path found by
/// <c>QziConfig.Load()</c>), and clearing it would break the "plain environment
/// variable, no config file" setup.
/// </para>
/// <para>
/// <c>QZI_NATIVE_LIBRARY</c> and <c>QZI_DEVICE_ID</c> are intentionally absent:
/// the native config loader never reads them. They are consumed on the managed
/// side by <c>NativeLibraryLoader</c> and by <c>qzi_engine_config</c>.
/// </para>
/// </remarks>
internal sealed class NativeConfigScope : IDisposable
{
    // The single source of truth for "what the native side reads at create
    // time". Keep in sync with qzen_config.c:env_overrides[] + QZI_CONFIG.
    private static readonly (string Name, Func<QziClientOptions, string?> Select)[] Variables =
    [
        (QziConfig.EnvironmentVariable, o => o.ConfigPath),
        (QziConfig.Qwen21DirectoryVariable, o => o.Qwen21Directory),
        (QziConfig.ZImageDirectoryVariable, o => o.ZImageDirectory),
        (QziConfig.LoraDirectoryVariable, o => o.Qwen21LoraDirectory),
        (QziConfig.NunchakuDirectoryVariable, o => o.ZImageNunchakuDirectory),
        (QziConfig.ModelCacheVariable, o => o.CacheDirectory),
    ];

    /// <summary>Names covered by the scope (also used by the config tests).</summary>
    internal static IReadOnlyList<string> VariableNames =>
        Array.ConvertAll(Variables, v => v.Name);

    private readonly List<(string Name, string? Previous)> _saved = [];
    private bool _disposed;

    internal NativeConfigScope(QziClientOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        try
        {
            foreach ((string name, Func<QziClientOptions, string?> select) in Variables)
            {
                string? value = select(options);
                if (string.IsNullOrWhiteSpace(value)) continue;
                _saved.Add((name, Environment.GetEnvironmentVariable(name)));
                Environment.SetEnvironmentVariable(name, value);
            }
        }
        catch
        {
            // A partially applied scope still has to be unwound, otherwise a
            // failed QziClient construction would leak its configuration into
            // the rest of the process.
            Restore(_saved);
            _disposed = true;
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Restore(_saved);
    }

    private static void Restore(IReadOnlyList<(string Name, string? Previous)> saved)
    {
        // Reverse order: a variable could appear twice in a future override
        // list, and the first snapshot is the one that must win on unwind.
        for (int i = saved.Count - 1; i >= 0; i--)
            Environment.SetEnvironmentVariable(saved[i].Name, saved[i].Previous);
    }
}
