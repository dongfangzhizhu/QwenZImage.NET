using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using QwenZImage.NET;

// Real-image regression harness.
//
// This is deliberately NOT a unit test: every case needs a CUDA build of
// QwenZImageNative plus multi-GB model packs, and a single case can take
// minutes. It refuses to run against the deterministic reference backend so
// that a noise image can never be recorded as a "real image" pass.
//
//   dotnet run --project tools/RealImageRegression -c Release -- \
//       --native artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll \
//       --out artifacts/real-image-regression/run-01 --only qwen21-turbo-512
//
// Cases live in cases.json (copied next to the executable). Results are
// written as one PNG per case plus report.md / report.json.

Cli cli = Cli.Parse(args);
if (cli.ShowHelp)
{
    Console.WriteLine(Cli.HelpText);
    return 0;
}

List<RegressionCase> cases;
try
{
    string casesPath = cli.CasesPath ?? Path.Combine(AppContext.BaseDirectory, "cases.json");
    cases = JsonSerializer.Deserialize<List<RegressionCase>>(
        File.ReadAllText(casesPath), JsonOptions.Read)
        ?? throw new InvalidDataException("cases.json deserialised to null");
}
catch (Exception ex)
{
    Console.Error.WriteLine($"cannot load cases: {ex.Message}");
    return 2;
}

if (cli.Only.Count > 0)
{
    HashSet<string> wanted = new(cli.Only, StringComparer.Ordinal);
    cases = cases.Where(c => wanted.Contains(c.Id)).ToList();
    string[] unknown = cli.Only.Where(id => !cases.Any(c => c.Id == id)).ToArray();
    if (unknown.Length > 0)
    {
        Console.Error.WriteLine($"unknown case id(s): {string.Join(", ", unknown)}");
        return 2;
    }
}

if (cli.List)
{
    foreach (RegressionCase c in cases) Console.WriteLine($"{c.Id,-34} {c.Describe()}");
    return 0;
}

if (cases.Count == 0)
{
    Console.Error.WriteLine("no cases selected");
    return 2;
}

if (cli.NativeLibrary is not null)
{
    if (!File.Exists(cli.NativeLibrary))
    {
        Console.Error.WriteLine($"native library not found: {cli.NativeLibrary}");
        return 2;
    }
    Environment.SetEnvironmentVariable("QZI_NATIVE_LIBRARY", Path.GetFullPath(cli.NativeLibrary));
}

Directory.CreateDirectory(cli.OutDirectory);
Console.WriteLine($"output: {Path.GetFullPath(cli.OutDirectory)}");
Console.WriteLine($"cases:  {cases.Count}");
Console.WriteLine($"config: {cli.ConfigPath ?? "auto-discovered"}");
Console.WriteLine($"native: {cli.NativeLibrary ?? "runtime default probing"}");

List<CaseResult> results = [];
int failures = 0;
try
{
    using QziClient client = new(QziClientOptions.FromConfig(cli.ConfigPath));
    EngineHealth health = client.GetHealth();
    Console.WriteLine($"backend: {(health.IsReferenceBackend ? "REFERENCE (noise)" : "real CUDA engine")} " +
                      $"| VRAM {health.FreeVramMib}/{health.TotalVramMib} MiB free");
    if (health.IsReferenceBackend)
    {
        Console.Error.WriteLine(
            "REFUSING TO RUN: the deterministic reference backend is loaded. " +
            "Point QZI_NATIVE_LIBRARY / --native at a -cuda artefact.");
        return 2;
    }

    foreach (RegressionCase c in cases)
    {
        Console.WriteLine();
        Console.WriteLine($"=== {c.Id}: {c.Describe()}");
        Stopwatch sw = Stopwatch.StartNew();
        CaseResult result = new() { Id = c.Id, Description = c.Describe() };
        List<string> phases = [];
        result.Phases = phases;
        try
        {
            GenerationOptions options = c.ToOptions();
            using CancellationTokenSource cts = new(TimeSpan.FromMinutes(cli.TimeoutMinutes));
            GenerationPhase last = (GenerationPhase)(-1);
            IProgress<GenerationProgress> progress = new SyncProgress(p =>
            {
                if (p.Phase == last) return;
                last = p.Phase;
                phases.Add(p.Phase.ToString());
                Console.WriteLine($"  [{sw.Elapsed.TotalSeconds,7:F1}s] {p.Phase}");
            });

            using GenerationResult image = client.Generate(options, progress, cts.Token);
            ReadOnlySpan<byte> png = image.PngSpan;
            (int pngWidth, int pngHeight) = ReadPngSize(png);
            string sha = Convert.ToHexStringLower(SHA256.HashData(png));
            string file = Path.Combine(cli.OutDirectory, c.Id + ".png");
            image.SaveAsync(file).GetAwaiter().GetResult();

            result.Ok = pngWidth == c.Width && pngHeight == c.Height;
            result.PngBytes = image.Length;
            result.PngWidth = pngWidth;
            result.PngHeight = pngHeight;
            result.Sha256 = sha;
            result.Seconds = Math.Round(image.Elapsed.TotalSeconds, 2);
            result.PngFile = Path.GetFileName(file);
            if (!result.Ok)
                result.Error = $"PNG IHDR {pngWidth}x{pngHeight} != requested {c.Width}x{c.Height}";
            Console.WriteLine($"  -> {image.Length} bytes, {pngWidth}x{pngHeight}, " +
                              $"{image.Elapsed.TotalSeconds:F1}s, sha256 {sha[..16]}...");
        }
        catch (Exception ex)
        {
            result.Ok = false;
            result.Error = $"{ex.GetType().Name}: {ex.Message}";
            Console.WriteLine($"  -> FAILED: {result.Error}");
        }
        result.WallSeconds = Math.Round(sw.Elapsed.TotalSeconds, 2);
        if (!result.Ok) failures++;
        results.Add(result);
        // A fatal CUDA fault later in the run must not discard the cases that
        // already completed, so persist the report after every single case.
        WriteReports(cli, results, failures);
    }

    // Real-engine determinism: a case may declare compare_with and must then
    // produce byte-identical PNGs to that earlier case (same seed + backend).
    foreach ((RegressionCase c, CaseResult r) in cases.Zip(results))
    {
        if (string.IsNullOrEmpty(c.CompareWith) || !r.Ok) continue;
        CaseResult? baseline = results.FirstOrDefault(x => x.Id == c.CompareWith);
        if (baseline is null || !baseline.Ok)
        {
            r.Determinism = "baseline unavailable";
            r.Ok = false;
            failures++;
        }
        else
        {
            r.Determinism = baseline.Sha256 == r.Sha256 ? "identical" : "DIFFERENT";
            if (r.Determinism != "identical")
            {
                r.Ok = false;
                r.Error = $"PNG differs from {c.CompareWith}";
                failures++;
            }
        }
    }

    health = client.GetHealth();
    Console.WriteLine();
    Console.WriteLine($"engine counters: served={health.Served} rejected={health.Rejected} " +
                      $"resident={health.ResidentModel}");
}
catch (Exception ex)
{
    Console.Error.WriteLine($"harness setup failed: {ex.GetType().Name}: {ex.Message}");
    return 2;
}

WriteReports(cli, results, failures);

Console.WriteLine();
Console.WriteLine($"=== {results.Count - failures}/{results.Count} passed; report: " +
                  $"{Path.Combine(cli.OutDirectory, "report.md")}");
return failures == 0 ? 0 : 1;

static void WriteReports(Cli cli, List<CaseResult> results, int failures)
{
    string reportJson = JsonSerializer.Serialize(new
    {
        generated_utc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
        backend = "real-cuda",
        cases = results
    }, JsonOptions.Write);
    File.WriteAllText(Path.Combine(cli.OutDirectory, "report.json"), reportJson);

    StringBuilder md = new();
    md.AppendLine("# Real-image regression");
    md.AppendLine();
    md.AppendLine($"Generated (UTC): {DateTime.UtcNow:O}  ");
    md.AppendLine($"CUDA native: `{cli.NativeLibrary ?? "runtime default"}`  ");
    md.AppendLine($"Config: `{cli.ConfigPath ?? "auto-discovered"}`  ");
    md.AppendLine($"Cases: {results.Count} | pass: {results.Count(x => x.Ok)} | fail: {failures}");
    md.AppendLine();
    md.AppendLine("| case | request | status | seconds | PNG bytes | SHA-256 (16) | determinism |");
    md.AppendLine("| --- | --- | --- | ---: | ---: | --- | --- |");
    foreach (CaseResult r in results)
    {
        md.AppendLine($"| `{r.Id}` | {Escape(r.Description)} | {(r.Ok ? "PASS" : "FAIL")} | " +
                      $"{r.Seconds:F1} | {r.PngBytes} | `{r.Sha256?[..Math.Min(16, r.Sha256.Length)]}` | " +
                      $"{r.Determinism ?? "-"} |");
        if (r.Error is not null) md.AppendLine($"|  | ↳ {Escape(r.Error)} |  |  |  |  |  |");
    }
    md.AppendLine();
    md.AppendLine("PNG files sit next to this report. `compare_with` cases must be byte-identical to their baseline.");
    md.AppendLine();
    md.AppendLine("## Phase traces");
    md.AppendLine();
    md.AppendLine("A `ModelLoad` phase on a case means the real engine reloaded weights for that request;");
    md.AppendLine("this is how LoRA / tier / checkpoint switching is verified end to end.");
    md.AppendLine();
    md.AppendLine("| case | phases |");
    md.AppendLine("| --- | --- |");
    foreach (CaseResult r in results)
        md.AppendLine($"| `{r.Id}` | {(r.Phases.Count == 0 ? "-" : string.Join(" → ", r.Phases))} |");
    File.WriteAllText(Path.Combine(cli.OutDirectory, "report.md"), md.ToString());
}

static (int, int) ReadPngSize(ReadOnlySpan<byte> png)
{
    ReadOnlySpan<byte> signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    if (png.Length < 24 || !png[..8].SequenceEqual(signature))
        throw new InvalidDataException("native output is not a PNG file");
    return (BinaryPrimitives.ReadInt32BigEndian(png[16..20]),
            BinaryPrimitives.ReadInt32BigEndian(png[20..24]));
}

static string Escape(string value) => value.Replace("|", "\\|", StringComparison.Ordinal);

internal static class JsonOptions
{
    public static readonly JsonSerializerOptions Read = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public static readonly JsonSerializerOptions Write = new() { WriteIndented = true };
}

internal sealed class RegressionCase
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("model")] public string Model { get; set; } = "qwen21";
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = "a red fox in snow";
    [JsonPropertyName("width")] public int Width { get; set; } = 512;
    [JsonPropertyName("height")] public int Height { get; set; } = 512;
    [JsonPropertyName("seed")] public ulong Seed { get; set; } = 42;
    [JsonPropertyName("tier")] public string? Tier { get; set; }
    [JsonPropertyName("steps")] public int? Steps { get; set; }
    [JsonPropertyName("guidance")] public float? Guidance { get; set; }
    [JsonPropertyName("negative_prompt")] public string? NegativePrompt { get; set; }
    [JsonPropertyName("lora")] public string? Lora { get; set; }
    [JsonPropertyName("lora_scale")] public float? LoraScale { get; set; }
    [JsonPropertyName("checkpoint")] public string? Checkpoint { get; set; }
    [JsonPropertyName("compare_with")] public string? CompareWith { get; set; }

    public GenerationOptions ToOptions() => Model switch
    {
        "qwen21" => new QwenImageOptions
        {
            Prompt = Prompt, Width = Width, Height = Height, Seed = Seed,
            Tier = string.Equals(Tier, "base", StringComparison.OrdinalIgnoreCase)
                ? QwenImageTier.Base : QwenImageTier.Turbo,
            Steps = Steps, Guidance = Guidance, NegativePrompt = NegativePrompt,
            Lora = Lora, LoraScale = LoraScale
        },
        "zimage" => new ZImageOptions
        {
            Prompt = Prompt, Width = Width, Height = Height, Seed = Seed,
            Checkpoint = Checkpoint
        },
        _ => throw new InvalidDataException($"unknown model \"{Model}\" in case {Id}")
    };

    public string Describe()
    {
        StringBuilder s = new($"{Model} {Width}x{Height} seed={Seed}");
        if (Model == "qwen21")
        {
            s.Append($" tier={Tier ?? "turbo"}");
            if (Steps is not null) s.Append($" steps={Steps}");
            if (Guidance is not null) s.Append($" guidance={Guidance}");
            if (Lora is not null) s.Append($" lora={Lora} scale={LoraScale ?? 1.0f}");
        }
        else s.Append($" checkpoint={Checkpoint ?? "bf16"}");
        return s.ToString();
    }
}

internal sealed class CaseResult
{
    public string Id { get; set; } = "";
    public string Description { get; set; } = "";
    public bool Ok { get; set; }
    public double Seconds { get; set; }
    public double WallSeconds { get; set; }
    public int PngBytes { get; set; }
    public int PngWidth { get; set; }
    public int PngHeight { get; set; }
    public string? Sha256 { get; set; }
    public string? PngFile { get; set; }
    public string? Determinism { get; set; }
    public string? Error { get; set; }
    public List<string> Phases { get; set; } = [];
}

internal sealed class SyncProgress(Action<GenerationProgress> action) : IProgress<GenerationProgress>
{
    public void Report(GenerationProgress value) => action(value);
}

internal sealed class Cli
{
    public string? CasesPath { get; private set; }
    public string OutDirectory { get; private set; } =
        Path.Combine("artifacts", "real-image-regression",
                     DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture));
    public List<string> Only { get; } = [];
    public string? NativeLibrary { get; private set; }
    public string? ConfigPath { get; private set; }
    public int TimeoutMinutes { get; private set; } = 30;
    public bool List { get; private set; }
    public bool ShowHelp { get; private set; }

    public const string HelpText = """
        Real-image regression harness (requires a CUDA native build + weights).

        Usage:
          --cases <file>          case list (default: cases.json next to the binary)
          --out <dir>             output directory for PNGs + report
          --only <id>             run one case (repeatable)
          --native <path>         CUDA QwenZImageNative.dll / .so (sets QZI_NATIVE_LIBRARY)
          --config <path>         QwenZImage.config.json (sets QZI_CONFIG via FromConfig)
          --timeout-minutes <n>   per-case cancellation timeout (default 30)
          --list                  list selected cases and exit
          --help                  show this help
        """;

    public static Cli Parse(string[] args)
    {
        Cli cli = new();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string? Value() => i + 1 < args.Length ? args[++i] : null;
            switch (a)
            {
                case "--cases": cli.CasesPath = Value(); break;
                case "--out": cli.OutDirectory = Value() ?? cli.OutDirectory; break;
                case "--only": if (Value() is { } id) cli.Only.Add(id); break;
                case "--native": cli.NativeLibrary = Value(); break;
                case "--config": cli.ConfigPath = Value(); break;
                case "--timeout-minutes":
                    if (int.TryParse(Value(), out int minutes) && minutes > 0)
                        cli.TimeoutMinutes = minutes;
                    break;
                case "--list": cli.List = true; break;
                case "--help" or "-h": cli.ShowHelp = true; break;
                default:
                    Console.Error.WriteLine($"unknown option: {a}");
                    cli.ShowHelp = true;
                    break;
            }
        }
        return cli;
    }
}
