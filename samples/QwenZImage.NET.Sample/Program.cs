using QwenZImage.NET;

// Minimal end-to-end sample. Works out of the box against the bundled
// deterministic reference backend (build it with scripts/build-native.* first),
// and against a real CUDA engine by pointing QZI_NATIVE_LIBRARY at its qzi
// build and supplying the model directories.

string modelDir = args.Length > 0
    ? args[0]
    : Environment.GetEnvironmentVariable("QZI_MODEL_DIR") ?? ".";
string outDir = Path.Combine(Directory.GetCurrentDirectory(), "outputs");
Directory.CreateDirectory(outDir);

var options = new QziClientOptions
{
    Qwen21Directory = modelDir,
    ZImageDirectory = modelDir,
    CacheDirectory = Path.Combine(Path.GetTempPath(), "qzi-text-cache")
};

try
{
    using QziClient client = new(options);
    Console.WriteLine($"qzi ABI v{client.AbiVersion}");
    Console.WriteLine($"native: {client.NativeLibraryPath ?? "runtime default probing"}");
    EngineHealth health = client.GetHealth();
    Console.WriteLine(health.IsReferenceBackend
        ? "backend: deterministic reference (no GPU / weights required)"
        : "backend: native engine");

    IProgress<GenerationProgress> progress = new Progress<GenerationProgress>(p =>
        Console.WriteLine($"  {p}"));

    // The reference backend is instant; a real engine takes minutes here.
    using GenerationResult qwen = await client.GenerateQwen21Async(
        new QwenImageOptions
        {
            Prompt = "A red fox walks through fresh snow, cinematic light",
            Width = 64,
            Height = 64,
            Seed = 42
        }, progress);
    string qwenPath = Path.Combine(outDir, "qwen21.png");
    await qwen.SaveAsync(qwenPath);
    Console.WriteLine($"Qwen-Image 2.1 -> {qwenPath} ({qwen.Length} bytes, {qwen.Elapsed.TotalSeconds:F2}s)");

    using GenerationResult zimage = await client.GenerateZImageAsync(
        new ZImageOptions
        {
            Prompt = "A red fox walks through fresh snow, cinematic light",
            Width = 64,
            Height = 64,
            Seed = 42,
            Checkpoint = "bf16"
        }, progress);
    string zimagePath = Path.Combine(outDir, "zimage.png");
    await zimage.SaveAsync(zimagePath);
    Console.WriteLine($"Z-Image       -> {zimagePath} ({zimage.Length} bytes, {zimage.Elapsed.TotalSeconds:F2}s)");

    // The .NET job queue replaces the old Go WebUI queue: strict serial
    // dispatch, cancellation, retry and state-change events.
    using QziGenerationQueue queue = new(client);
    QziJob job = queue.Submit(new ZImageOptions
    {
        Prompt = "a mountain lake at dawn",
        Width = 32,
        Height = 32
    });
    await queue.WaitForIdleAsync();
    Console.WriteLine($"queued job {job.Id}: {job.State}, {job.OutputLength} bytes");

    health = client.GetHealth();
    Console.WriteLine($"served={health.Served} rejected={health.Rejected} " +
                      $"resident={health.ResidentModel} uptime={health.Uptime.TotalSeconds:F1}s");
}
catch (DllNotFoundException ex)
{
    Console.Error.WriteLine(
        "No qzi native library found. Build the reference library first:\n" +
        "  cmake -S native -B native/build -G Ninja && cmake --build native/build\n" +
        "or set QZI_NATIVE_LIBRARY to your engine's qzi.dll / libqzi.so.\n" + ex.Message);
    return 1;
}
catch (QziException ex)
{
    Console.Error.WriteLine($"Engine error [{ex.Status}]: {ex.NativeMessage}");
    return 1;
}

return 0;
