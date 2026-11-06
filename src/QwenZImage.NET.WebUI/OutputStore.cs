using System.Globalization;
using QwenZImage.NET;

namespace QwenZImage.NET.WebUI;

/// <summary>
/// Owns the WebUI's output gallery. Mirrors the Go WebUI's safety rules: names
/// are bare file names (no path traversal, no hidden files), deletion moves to
/// a <c>.trash</c> directory instead of unlinking, and restore never overwrites
/// a newer file.
/// </summary>
public sealed class OutputStore
{
    private const string TrashDirectory = ".trash";

    private readonly string _root;

    public OutputStore(string root)
    {
        _root = Path.GetFullPath(root);
        Directory.CreateDirectory(_root);
    }

    public string Root => _root;

    public static bool IsSafeName(string? name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.StartsWith('.')) return false;
        if (name is "." or "..") return false;
        if (name.Contains('/') || name.Contains('\\') || name.Contains('\0')) return false;
        return string.Equals(Path.GetFileName(name), name, StringComparison.Ordinal);
    }

    public static string ContentTypeFor(string name)
        => Path.GetExtension(name).ToLowerInvariant() switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            ".webp" => "image/webp",
            ".gif" => "image/gif",
            ".mp4" => "video/mp4",
            ".wav" => "audio/wav",
            _ => "application/octet-stream"
        };

    public string Resolve(string name) => Path.Combine(_root, name);

    public IReadOnlyList<OutputEntryDto> List()
    {
        var entries = new List<OutputEntryDto>();
        foreach (string path in Directory.EnumerateFiles(_root, "*", SearchOption.TopDirectoryOnly))
        {
            string name = Path.GetFileName(path);
            if (!IsSafeName(name)) continue;
            var info = new FileInfo(path);
            entries.Add(new OutputEntryDto
            {
                Name = name,
                Size = info.Length,
                ModTime = new DateTimeOffset(info.LastWriteTimeUtc, TimeSpan.Zero),
                Url = "/api/outputs/" + Uri.EscapeDataString(name),
                ContentType = ContentTypeFor(name)
            });
        }
        return entries.OrderByDescending(e => e.ModTime).ToList();
    }

    /// <summary>Persists a completed PNG and returns its bare file name.</summary>
    public async Task<string> SaveAsync(long jobId, GenerationResult result,
                                        CancellationToken cancellationToken)
    {
        string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
        string name = $"{stamp}-job{jobId}.png";
        for (int i = 1; File.Exists(Resolve(name)) && i < 1000; i++)
            name = $"{stamp}-job{jobId}-{i}.png";

        string path = Resolve(name);
        await using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write,
                                                FileShare.None, 64 * 1024, useAsync: true);
        await result.CopyToAsync(stream, cancellationToken);
        return name;
    }

    public bool TryTrash(string? name, out string error)
    {
        error = string.Empty;
        if (!IsSafeName(name))
        {
            error = "invalid output name";
            return false;
        }

        string path = Resolve(name!);
        var info = new FileInfo(path);
        if (!info.Exists || info.LinkTarget is not null)
        {
            error = "no such regular output file";
            return false;
        }

        string trash = Path.Combine(_root, TrashDirectory);
        Directory.CreateDirectory(trash);
        File.Move(path, Path.Combine(trash, name!), overwrite: true);
        return true;
    }

    public bool TryRestore(string? name, out string restoredName, out string error)
    {
        restoredName = string.Empty;
        error = string.Empty;
        if (!IsSafeName(name))
        {
            error = "invalid output name";
            return false;
        }

        string source = Path.Combine(_root, TrashDirectory, name!);
        var info = new FileInfo(source);
        if (!info.Exists || info.LinkTarget is not null)
        {
            error = "not in the trash";
            return false;
        }

        string target = Resolve(name!);
        if (File.Exists(target))
        {
            string stem = Path.GetFileNameWithoutExtension(name!);
            string ext = Path.GetExtension(name!);
            string candidate = name!;
            for (int i = 1; i < 1000 && File.Exists(Resolve(candidate)); i++)
                candidate = $"{stem}-restored-{i}{ext}";
            target = Resolve(candidate);
        }

        File.Move(source, target);
        restoredName = Path.GetFileName(target);
        return true;
    }
}
