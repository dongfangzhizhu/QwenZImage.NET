using System.Reflection;
using System.Runtime.InteropServices;

namespace QwenZImage.NET.Native;

/// <summary>
/// Resolves <c>QwenZImageNative.dll</c> / <c>libQwenZImageNative.so</c>, the
/// native backend implementing the qzi C ABI. The NuGet package bundles the
/// reference build for win-x64 and linux-x64; a CUDA build of the same ABI is
/// selected by <c>QZI_NATIVE_LIBRARY</c> or by replacing the bundled file.
/// When a caller explicitly selects a library via
/// <c>QziClientOptions.NativeLibraryPath</c> / <c>QZI_NATIVE_LIBRARY</c>, that
/// path is authoritative: a missing or unloadable library raises
/// <see cref="DllNotFoundException"/> instead of silently falling back to the
/// bundled reference backend (which would emit noise while appearing healthy).
/// Otherwise the resolution order is:
/// <list type="number">
///   <item><c>QziClientOptions.NativeLibraryPath</c> / <c>QZI_NATIVE_LIBRARY</c></item>
///   <item>next to the managed assembly, including <c>runtimes/&lt;rid&gt;/native/</c></item>
///   <item><c>NATIVE_DLL_SEARCH_DIRECTORIES</c> from deps.json (the NuGet <c>runtimes/</c> layout)</item>
///   <item><c>artifacts/runtimes/&lt;rid&gt;/native/</c> and <c>artifacts/native/&lt;platform&gt;/</c> (dev/CI)</item>
///   <item>the runtime's default probing (<c>PATH</c>, <c>LD_LIBRARY_PATH</c>, …)</item>
/// </list>
/// </summary>
internal static class NativeLibraryLoader
{
    private static readonly object Gate = new();
    private static bool _registered;
    private static string? _explicitPath;
    private static string? _resolvedPath;
    private static IntPtr _handle;

    internal static string? ResolvedPath => _resolvedPath;

    /// <summary>
    /// Handle of the library the resolver actually loaded. AbiGuard probes this
    /// handle with <c>NativeLibrary.TryGetExport</c> to discover additive ABI
    /// v1 exports (qzi_abi_features / qzi_qwen21_edit_layout) without ever
    /// triggering a missing-entry-point exception. Zero when the runtime loaded
    /// the library through its own probing and we could not capture a handle.
    /// </summary>
    internal static IntPtr Handle => _handle;

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

        // An explicitly requested backend must NEVER fall back to the bundled
        // reference build. Falling back is silent from the caller's point of
        // view: the process looks healthy, but every generated image is noise
        // because no model was ever loaded. Fail loudly instead.
        string? explicitPath = _explicitPath;
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            string full = Path.GetFullPath(explicitPath);
            if (!File.Exists(full))
            {
                throw new DllNotFoundException(
                    $"The explicitly selected native library does not exist: {full}. " +
                    "Check QZI_NATIVE_LIBRARY / QziClientOptions.NativeLibraryPath. " +
                    "The reference backend was NOT substituted, because that would " +
                    "silently produce noise instead of real images.");
            }

            try
            {
                IntPtr handle = NativeLibrary.Load(full);
                _resolvedPath = full;
                _handle = handle;
                return handle;
            }
            catch (Exception ex) when (ex is DllNotFoundException or BadImageFormatException)
            {
                throw new DllNotFoundException(
                    $"Could not load the explicitly selected native library '{full}': " +
                    $"{ex.Message} Verify the CUDA runtime (cublas / cudart / OpenMP) is on " +
                    "PATH and that the library matches this OS and architecture. " +
                    "The reference backend was NOT substituted, because that would " +
                    "silently produce noise instead of real images.", ex);
            }
        }

        foreach (string candidate in Candidates())
        {
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
            {
                _resolvedPath = candidate;
                _handle = handle;
                return handle;
            }
        }

        // No explicit path and no candidate hit: let the runtime probe, but try
        // the plain library name ourselves first so AbiGuard still gets a
        // handle for its additive-export probe. TryLoad(string, out IntPtr)
        // does NOT re-enter the DllImportResolver, so this cannot recurse.
        if (NativeLibrary.TryLoad(libraryName, out IntPtr fallback))
        {
            _resolvedPath = libraryName;
            _handle = fallback;
            return fallback;
        }
        return IntPtr.Zero; // let the runtime probe normally
    }

    private static IEnumerable<string> Candidates()
    {
        string name = OperatingSystem.IsWindows() ? "QwenZImageNative.dll"
                    : OperatingSystem.IsMacOS() ? "libQwenZImageNative.dylib"
                    : "libQwenZImageNative.so";
        string rid = RuntimeInformation.RuntimeIdentifier;
        string baseDir = AppContext.BaseDirectory;

        yield return Path.Combine(baseDir, name);
        yield return Path.Combine(baseDir, "runtimes", rid, "native", name);
        yield return Path.Combine(baseDir, "native", name);

        // When the package was installed from NuGet, deps.json lists the RID
        // native directories (usually inside ~/.nuget/packages). Probing them
        // explicitly also covers hosts that do not populate the default native
        // search path; falling through to IntPtr.Zero still uses runtime probing.
        if (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") is string searchDirs)
        {
            foreach (string directory in searchDirs.Split(
                         Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                yield return Path.Combine(directory, name);
        }

        // Dev / CI layout. The staged NuGet RID layout takes precedence over
        // the raw CMake output so tests exercise exactly what gets packed.
        DirectoryInfo? dir = new(baseDir);
        for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            string ridArtifacts = Path.Combine(dir.FullName, "artifacts", "runtimes");
            if (Directory.Exists(ridArtifacts))
            {
                foreach (string platform in Directory.EnumerateDirectories(ridArtifacts))
                    yield return Path.Combine(platform, "native", name);
            }

            string artifacts = Path.Combine(dir.FullName, "artifacts", "native");
            if (!Directory.Exists(artifacts)) continue;
            foreach (string platform in Directory.EnumerateDirectories(artifacts))
            {
                yield return Path.Combine(platform, name);
            }
        }
    }
}
