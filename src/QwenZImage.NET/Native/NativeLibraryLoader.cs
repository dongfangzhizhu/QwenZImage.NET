using System.Reflection;
using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Resolves the <c>qzi</c> shared library. The NuGet package deliberately does
/// not bundle a CUDA engine or multi-gigabyte weights: callers point this at
/// whichever <c>qzi</c> build implements the ABI (the bundled reference build,
/// or an engine's own build). Resolution order:
/// <list type="number">
///   <item><c>QziClientOptions.NativeLibraryPath</c> / <c>QZI_NATIVE_LIBRARY</c></item>
///   <item>next to the managed assembly, including <c>runtimes/&lt;rid&gt;/native/</c></item>
///   <item><c>artifacts/native/&lt;platform&gt;/</c> walking up from the assembly (dev/CI)</item>
///   <item>the runtime's default probing (<c>PATH</c>, <c>LD_LIBRARY_PATH</c>, …)</item>
/// </list>
/// </summary>
internal static class NativeLibraryLoader
{
    private static readonly object Gate = new();
    private static bool _registered;
    private static string? _explicitPath;
    private static string? _resolvedPath;

    internal static string? ResolvedPath => _resolvedPath;

    internal static void Configure(string? explicitPath)
    {
        _explicitPath = !string.IsNullOrWhiteSpace(explicitPath)
            ? explicitPath
            : Environment.GetEnvironmentVariable("QZI_NATIVE_LIBRARY");

        if (Volatile.Read(ref _registered)) return;
        lock (Gate)
        {
            if (_registered) return;
            NativeLibrary.SetDllImportResolver(typeof(NativeLibraryLoader).Assembly, Resolve);
            _registered = true;
        }
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly,
                                  DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, NativeMethods.LibraryName, StringComparison.Ordinal))
            return IntPtr.Zero;

        foreach (string candidate in Candidates())
        {
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
            {
                _resolvedPath = candidate;
                return handle;
            }
        }
        return IntPtr.Zero; // let the runtime probe normally
    }

    private static IEnumerable<string> Candidates()
    {
        if (!string.IsNullOrWhiteSpace(_explicitPath))
            yield return _explicitPath!;

        string name = OperatingSystem.IsWindows() ? "qzi.dll"
                    : OperatingSystem.IsMacOS() ? "libqzi.dylib"
                    : "libqzi.so";
        string rid = RuntimeInformation.RuntimeIdentifier;
        string baseDir = AppContext.BaseDirectory;

        yield return Path.Combine(baseDir, name);
        yield return Path.Combine(baseDir, "runtimes", rid, "native", name);
        yield return Path.Combine(baseDir, "native", name);

        // Dev / CI layout: <repo>/artifacts/native/<Platform>-<Arch>/qzi.dll
        DirectoryInfo? dir = new(baseDir);
        for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            string artifacts = Path.Combine(dir.FullName, "artifacts", "native");
            if (!Directory.Exists(artifacts)) continue;
            foreach (string platform in Directory.EnumerateDirectories(artifacts))
            {
                yield return Path.Combine(platform, name);
            }
        }
    }
}
