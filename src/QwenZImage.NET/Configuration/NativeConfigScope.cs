using System.Runtime.InteropServices;

namespace QwenZImage.NET;

/// <summary>
/// Temporarily publishes the <c>QZI_*</c> variables that the native config
/// loader consults inside <c>qzi_engine_create()</c>, then restores the exact
/// previous process environment.
/// </summary>
/// <remarks>
/// On Unix, <c>Environment.SetEnvironmentVariable</c> only updates .NET's own
/// environment dictionary; it never calls libc <c>setenv()</c>. The native
/// config loader reads through <c>qzi_getenv()</c> -> <c>getenv()</c>, so a
/// managed-only publication is invisible to it and every engine would snapshot
/// the process-start environment instead of its own options. The scope
/// therefore mirrors every publish/restore into libc as well (see
/// <see cref="PosixEnvironment"/>).
/// </remarks>
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
    // Serialises the whole publish/restore pair. QziClient already serialises
    // engine creation, but scopes can also be constructed directly (tests),
    // and libc setenv()/unsetenv() mutate process-wide, non-thread-safe state.
    private static readonly object ScopeGate = new();

    // The single source of truth for "what the native side reads at create
    // time". Keep in sync with qzen_config.c:env_overrides[] + QZI_CONFIG.
    private static readonly (string Name, Func<QziClientOptions, string?> Select)[] Variables =
    [
        (QziConfig.EnvironmentVariable, o => o.ConfigPath),
        (QziConfig.Qwen21DirectoryVariable, o => o.Qwen21Directory),
        (QziConfig.ZImageDirectoryVariable, o => o.ZImageDirectory),
        (QziConfig.LoraDirectoryVariable, o => o.Qwen21LoraDirectory),
        (QziConfig.NunchakuDirectoryVariable, o => o.ZImageNunchakuDirectory),
        (QziConfig.ZImageControlDirectoryVariable, o => o.ZImageControlDirectory),
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

        lock (ScopeGate)
        {
            try
            {
                foreach ((string name, Func<QziClientOptions, string?> select) in Variables)
                {
                    string? value = select(options);
                    if (string.IsNullOrWhiteSpace(value)) continue;
                    _saved.Add((name, Environment.GetEnvironmentVariable(name)));
                    Publish(name, value);
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
    }

    public void Dispose()
    {
        lock (ScopeGate)
        {
            if (_disposed) return;
            _disposed = true;
            Restore(_saved);
        }
    }

    /// <summary>Publishes to both worlds .NET and native code read from.</summary>
    private static void Publish(string name, string? value)
    {
        Environment.SetEnvironmentVariable(name, value);
        PosixEnvironment.Publish(name, value);   // no-op on Windows
    }

    private static void Restore(IReadOnlyList<(string Name, string? Previous)> saved)
    {
        // Reverse order: a variable could appear twice in a future override
        // list, and the first snapshot is the one that must win on unwind.
        for (int i = saved.Count - 1; i >= 0; i--)
            Publish(saved[i].Name, saved[i].Previous);
    }
}

/// <summary>
/// Best-effort bridge from the managed environment dictionary to libc's
/// <c>setenv()</c> / <c>unsetenv()</c> on POSIX. .NET deliberately does not
/// call libc here (setenv is not thread-safe), which is why the native
/// <c>qzi_getenv()</c> would otherwise never see scoped <c>QZI_*</c> values.
/// Callers hold <c>NativeConfigScope.ScopeGate</c> while publishing, so the
/// libc mutation and the native <c>getenv()</c> inside engine creation cannot
/// interleave. A platform without these symbols keeps the legacy behaviour
/// instead of failing client construction.
/// </summary>
internal static class PosixEnvironment
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SetEnvFn([MarshalAs(UnmanagedType.LPUTF8Str)] string name,
                                  [MarshalAs(UnmanagedType.LPUTF8Str)] string value,
                                  int overwrite);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int UnsetEnvFn([MarshalAs(UnmanagedType.LPUTF8Str)] string name);

    private static readonly object InitGate = new();
    private static SetEnvFn? _setEnv;
    private static UnsetEnvFn? _unsetEnv;
    private static bool _initialized;

    internal static void Publish(string name, string? value)
    {
        if (OperatingSystem.IsWindows()) return;
        EnsureLoaded();
        try
        {
            if (value is null) _unsetEnv?.Invoke(name);
            else _setEnv?.Invoke(name, value, 1);
        }
        catch (Exception ex) when (ex is DllNotFoundException
                                   or EntryPointNotFoundException
                                   or BadImageFormatException)
        {
            // Keep the managed-only behaviour if libc cannot be invoked.
        }
    }

    private static void EnsureLoaded()
    {
        lock (InitGate)
        {
            if (_initialized) return;
            _initialized = true;
            // glibc first; musl resolves "libc"; macOS uses libSystem.
            foreach (string library in new[]
                     { "libc.so.6", "libc", "libSystem.B.dylib", "libSystem.dylib" })
            {
                if (!NativeLibrary.TryLoad(library, out IntPtr handle)) continue;
                if (!NativeLibrary.TryGetExport(handle, "setenv", out IntPtr setPtr) ||
                    !NativeLibrary.TryGetExport(handle, "unsetenv", out IntPtr unsetPtr))
                    continue;
                _setEnv = Marshal.GetDelegateForFunctionPointer<SetEnvFn>(setPtr);
                _unsetEnv = Marshal.GetDelegateForFunctionPointer<UnsetEnvFn>(unsetPtr);
                return;
            }
        }
    }
}
