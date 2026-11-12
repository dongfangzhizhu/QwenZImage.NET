namespace QwenZImage.NET.WebUI.ModelDownloads;

/// <summary>How a config key's value is interpreted when resolving a download target.</summary>
public enum ModelPathKind
{
    /// <summary>The key always names a directory; files are placed inside it.</summary>
    Directory,

    /// <summary>
    /// The key may name a single <c>.safetensors</c> file (use it verbatim)
    /// or a directory (append the upstream file name).
    /// </summary>
    FileOrDirectory
}

/// <summary>One concrete downloadable file (a quantisation / format choice).</summary>
/// <param name="Id">Stable variant id, e.g. <c>int8</c> or <c>bf16</c>.</param>
/// <param name="Label">Human-readable label for the UI.</param>
/// <param name="Url">WanGP Hugging Face <c>resolve/main</c> direct link.</param>
/// <param name="Size">Exact size in bytes reported by the HF repository.</param>
/// <param name="RelativePath">Optional override of the target relative path.</param>
public sealed record ModelVariant(
    string Id, string Label, string Url, long Size, string? RelativePath = null);

/// <summary>A logical model file. Variants are alternative contents for the same slot.</summary>
public sealed record ModelFile(
    string Id,
    string Name,
    string RelativePath,
    string Description,
    bool Required,
    IReadOnlyList<ModelVariant> Variants,
    IReadOnlyList<string>? Aliases = null,
    string? Note = null);

/// <summary>A set of files sharing one <c>QwenZImage.config.json</c> key.</summary>
public sealed record ModelGroup(
    string Id,
    string Name,
    string ConfigKey,
    string EnvironmentVariable,
    string DefaultSubdirectory,
    ModelPathKind PathKind,
    string Description,
    bool Optional,
    IReadOnlyList<ModelFile> Files,
    string? Note = null);

/// <summary>
/// Static catalogue of every weight this engine / WebUI can consume. URLs are
/// transcribed from the local WanGP checkout (<c>D:/AI/Wan2GP</c>):
/// <c>defaults/*.json</c>, <c>models/qwen21/qwen21_handler.py</c>,
/// <c>models/z_image/z_image_handler.py</c> and <c>preprocessing/*/assets.py</c>.
/// WanGP builds them with <c>shared/utils/hf.py::build_hf_url()</c>, i.e.
/// <c>https://huggingface.co/&lt;repo&gt;/resolve/main/&lt;path&gt;</c>.
/// </summary>
/// <remarks>
/// The core NuGet package and the C ABI are untouched: this is a host-layer
/// convenience that places weights into the directories the native engine
/// already reads. Weights are NOT bundled in this repository or the package;
/// their licences belong to the upstream model publishers.
/// </remarks>
public static class ModelManifest
{
    public const string HuggingFacePrefix = "https://huggingface.co/";

    /// <summary>
    /// Optional mirror base URL. Only the HF prefix is rewritten, so this can
    /// point at an HF mirror (<c>https://hf-mirror.com</c>) or a local test
    /// server; no caller-supplied URL is ever fetched.
    /// </summary>
    public const string MirrorEnvironmentVariable = "QZI_HF_MIRROR";

    /// <summary>Recommended variant: smaller INT8 ConvRot / quanto repacks.</summary>
    public const string DefaultVariant = "int8";

    private const string Qwen21Repo = "DeepBeepMeep/Qwen_image_2";
    private const string IdeogramRepo = "DeepBeepMeep/Ideogram4";
    private const string ZImageRepo = "DeepBeepMeep/Z-Image";
    private const string Wan21Repo = "DeepBeepMeep/Wan2.1";

    private static readonly string Qwen21EncoderFolder = "Qwen3-VL-8B-Instruct";
    private static readonly string ZImageEncoderFolder = "Qwen3";

    /// <summary>Every group, in UI display order.</summary>
    public static IReadOnlyList<ModelGroup> Groups { get; } = Build();

    /// <summary>Flat view of every logical file, keyed by its stable id.</summary>
    public static IReadOnlyDictionary<string, ModelFile> FilesById { get; } =
        Groups.SelectMany(g => g.Files).ToDictionary(f => f.Id, StringComparer.Ordinal);

    private static string Hf(string repo, string path)
        => HuggingFacePrefix + repo + "/resolve/main/" + path;

    private static ModelVariant V(string id, string label, string url, long size,
                                  string? relativePath = null)
        => new(id, label, url, size, relativePath);

    /// <summary>
    /// Rewrites the fixed Hugging Face prefix when <c>QZI_HF_MIRROR</c> is set.
    /// The rest of the URL always comes from the static catalogue.
    /// </summary>
    public static string ResolveUrl(string url, string? mirror)
    {
        if (string.IsNullOrWhiteSpace(mirror)) return url;
        string trimmed = mirror.Trim().TrimEnd('/');
        if (!Uri.TryCreate(trimmed, UriKind.Absolute, out Uri? parsed) ||
            (parsed.Scheme != Uri.UriSchemeHttp && parsed.Scheme != Uri.UriSchemeHttps))
            throw new ArgumentException(
                $"{MirrorEnvironmentVariable} must be an absolute http(s) URL.", nameof(mirror));
        if (!url.StartsWith(HuggingFacePrefix, StringComparison.Ordinal))
            throw new ArgumentException("Only WanGP Hugging Face URLs can be mirrored.", nameof(url));
        return trimmed + "/" + url[HuggingFacePrefix.Length..];
    }

    /// <summary>Normalises a catalogue relative path to the current platform.</summary>
    public static string CombineRelative(string root, string relative)
    {
        if (Path.IsPathRooted(relative) || relative.Split('/', '\\').Contains(".."))
            throw new ArgumentException($"Unsafe manifest path '{relative}'.", nameof(relative));
        string path = root;
        foreach (string part in relative.Split('/', '\\', StringSplitOptions.RemoveEmptyEntries))
            path = Path.Combine(path, part);
        return Path.GetFullPath(path);
    }

    private static ModelGroup[] Build()
    {
        return
        [
            // ------------------------------------------------ Qwen-Image 2.1
            new ModelGroup(
                "qwen21", "Qwen-Image 2.1", "qwen21_dir", QziConfig.Qwen21DirectoryVariable,
                "models/qwen21", ModelPathKind.Directory,
                "Base + Turbo DiT, VAE, Qwen3-VL-8B text tower and tokenizer. " +
                "These are the engine's canonical flat pack names (native/src/qzi_engine_backend.c::resolve_qwen21_pack).",
                Optional: false,
                Files:
                [
                    new ModelFile(
                        "qwen21.dit.base", "Base DiT (7B)", "dit_base.safetensors",
                        "Qwen-Image 2.1 base checkpoint; used by the base tier and image editing.",
                        Required: true,
                        Variants:
                        [
                            V("int8", "INT8 ConvRot · 7.26 GB (recommended)",
                              Hf(Qwen21Repo, "qwen_image_21_7B_int8_convrot.safetensors"), 7256804890),
                            V("bf16", "BF16 · 14.23 GB",
                              Hf(Qwen21Repo, "qwen_image_21_7B_bf16.safetensors"), 14230284656)
                        ],
                        Aliases: ["qwen_image_21_7B_int8_convrot.safetensors", "qwen_image_21_7B_bf16.safetensors"]),
                    new ModelFile(
                        "qwen21.dit.turbo", "Turbo DiT (8 steps)", "dit_turbo_int8.safetensors",
                        "Qwen's full eight-step distilled checkpoint.",
                        Required: true,
                        Variants:
                        [
                            V("int8", "INT8 ConvRot · 7.26 GB (recommended)",
                              Hf(Qwen21Repo, "qwen_image_21_turbo_7B_int8_convrot.safetensors"), 7256804896),
                            V("bf16", "BF16 · 14.23 GB",
                              Hf(Qwen21Repo, "qwen_image_21_turbo_7B_bf16.safetensors"), 14230284664)
                        ],
                        Aliases: ["qwen_image_21_turbo_7B_int8_convrot.safetensors", "qwen_image_21_turbo_7B_bf16.safetensors", "dit_turbo.safetensors"],
                        Note: "The file name is a fixed engine slot, not a quantisation claim: the engine detects INT8 vs BF16 from the safetensors header."),
                    new ModelFile(
                        "qwen21.vae", "VAE", "vae.safetensors",
                        "Shared BF16 VAE (encoder + decoder).",
                        Required: true,
                        Variants: [V("default", "BF16 · 1.35 GB", Hf(Qwen21Repo, "qwen_image_21/qwen_image_21_vae.safetensors"), 1350989616)],
                        Aliases: ["qwen_image_21_vae.safetensors"]),
                    new ModelFile(
                        "qwen21.vae_config", "VAE config", "vae_config.json",
                        "Per-channel latent statistics read next to the VAE checkpoint.",
                        Required: true,
                        Variants: [V("default", "JSON · 2 KB", Hf(Qwen21Repo, "qwen_image_21/vae_config.json"), 2079)],
                        Aliases: ["qwen_image_21/vae_config.json"]),
                    new ModelFile(
                        "qwen21.scheduler_config", "Scheduler config", "scheduler_config.json",
                        "WanGP scheduler metadata; the engine's eight Turbo sigmas are fixed in C.",
                        Required: false,
                        Variants: [V("default", "JSON · 485 B", Hf(Qwen21Repo, "qwen_image_21/scheduler_config.json"), 485)],
                        Aliases: ["qwen_image_21/scheduler_config.json"]),
                    new ModelFile(
                        "qwen21.text_encoder", "Qwen3-VL-8B text tower", "text_encoder_int8.safetensors",
                        "Qwen3-VL-8B-Instruct language tower used for prompt conditioning.",
                        Required: true,
                        Variants:
                        [
                            V("int8", "INT8 ConvRot · 8.94 GB (recommended)",
                              Hf(IdeogramRepo, Qwen21EncoderFolder + "/Qwen3-VL-8B-Instruct_int8_convrot.safetensors"), 8939652364),
                            V("bf16", "BF16 · 16.29 GB",
                              Hf(IdeogramRepo, Qwen21EncoderFolder + "/Qwen3-VL-8B-Instruct_bf16.safetensors"), 16289676320)
                        ],
                        Aliases: ["Qwen3-VL-8B-Instruct_int8_convrot.safetensors", "Qwen3-VL-8B-Instruct_bf16.safetensors", "text_encoder.safetensors"]),
                    new ModelFile(
                        "qwen21.tokenizer", "Tokenizer", "tokenizer.json",
                        "BPE tokenizer. WanGP ships it as tokenizer_legacy.json; the engine expects tokenizer.json.",
                        Required: true,
                        Variants: [V("default", "JSON · 11.42 MB", Hf(IdeogramRepo, Qwen21EncoderFolder + "/tokenizer_legacy.json"), 11422654)],
                        Aliases: ["tokenizer_legacy.json"]),
                ],
                Note: "WanGP sources: defaults/qwen_image_21_7B.json, defaults/qwen_image_21_turbo_7B.json and qwen21_handler.py::query_model_files()."),

            // ---------------------------------------------------- Qwen21 LoRAs
            new ModelGroup(
                "qwen21-loras", "Qwen-Image 2.1 LoRA adapters", "qwen21_lora_dir",
                QziConfig.LoraDirectoryVariable, "models/qwen21/loras", ModelPathKind.Directory,
                "Optional acceleration / editing adapters. The engine resolves a bare file name inside this directory.",
                Optional: true,
                Files:
                [
                    new ModelFile(
                        "lora.5step", "Pruna 5-step", "p_qwen_image_2.1_5step_v0.1.safetensors",
                        "WanGP 5-step acceleration LoRA.", Required: false,
                        Variants: [V("default", "BF16 · 335.6 MB", Hf(Qwen21Repo, "loras/p_qwen_image_2.1_5step_v0.1.safetensors"), 335606144)],
                        Note: "Also listed in WanGP loras_url_cache_v2.json."),
                    new ModelFile(
                        "lora.8step", "Pruna 8-step", "p_qwen_image_2.1_8step_v0.1.safetensors",
                        "WanGP 8-step acceleration LoRA.", Required: false,
                        Variants: [V("default", "BF16 · 335.6 MB", Hf(Qwen21Repo, "loras/p_qwen_image_2.1_8step_v0.1.safetensors"), 335606104)]),
                    new ModelFile(
                        "lora.viggle-turbo", "Viggle Turbo v0.2.1", "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
                        "WanGP built-in 6-step Viggle Turbo profile adapter.", Required: false,
                        Variants: [V("default", "BF16 · 1.36 GB", Hf(Qwen21Repo, "loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"), 1359147904)]),
                    new ModelFile(
                        "lora.viggle-swap", "Viggle Image Swap", "Qwen-Image-2.1-viggle-image-swap-v0.1-lora-r256.safetensors",
                        "WanGP character-swap profile adapter.", Required: false,
                        Variants: [V("default", "BF16 · 1.36 GB", Hf(Qwen21Repo, "loras/Qwen-Image-2.1-viggle-image-swap-v0.1-lora-r256.safetensors"), 1359157592)]),
                ],
                Note: "Licence: the Viggle adapters carry the Qwen Research License on the HF repo. Do not redistribute them."),

            // -------------------------------------------------------- Z-Image
            new ModelGroup(
                "zimage", "Z-Image Turbo", "zimage_dir", QziConfig.ZImageDirectoryVariable,
                "models/zimage", ModelPathKind.Directory,
                "DiT, Qwen3 text encoder, tokenizer and VAE. The engine merges every .safetensors in this flat directory by tensor name.",
                Optional: false,
                Files:
                [
                    new ModelFile(
                        "zimage.dit", "DiT (6B)", "dit.safetensors",
                        "Z-Image Turbo diffusion transformer.", Required: true,
                        Variants:
                        [
                            V("int8", "quanto BF16/INT8 · 6.89 GB (recommended)",
                              Hf(ZImageRepo, "ZImageTurbo_quanto_bf16_int8.safetensors"), 6886911511),
                            V("bf16", "BF16 · 12.31 GB",
                              Hf(ZImageRepo, "ZImageTurbo_bf16.safetensors"), 12309879106)
                        ],
                        Aliases: ["ZImageTurbo_quanto_bf16_int8.safetensors", "ZImageTurbo_bf16.safetensors"]),
                    new ModelFile(
                        "zimage.text_encoder", "Qwen3-4B text encoder", "text_encoder.safetensors",
                        "Qwen3 BPE text tower for prompt conditioning.", Required: true,
                        Variants:
                        [
                            V("int8", "quanto BF16/INT8 · 4.41 GB (recommended)",
                              Hf(ZImageRepo, ZImageEncoderFolder + "/qwen3_quanto_bf16_int8.safetensors"), 4414004409),
                            V("bf16", "BF16 · 8.04 GB",
                              Hf(ZImageRepo, ZImageEncoderFolder + "/qwen3_bf16.safetensors"), 8044986509)
                        ],
                        Aliases: ["qwen3_quanto_bf16_int8.safetensors", "qwen3_bf16.safetensors"]),
                    new ModelFile(
                        "zimage.tokenizer", "Tokenizer", "tokenizer.json",
                        "Qwen3 BPE tokenizer (required by the native pipeline).", Required: true,
                        Variants: [V("default", "JSON · 11.42 MB", Hf(ZImageRepo, ZImageEncoderFolder + "/tokenizer.json"), 11422654)]),
                    new ModelFile(
                        "zimage.vae", "VAE", "vae.safetensors",
                        "Z-Image Turbo BF16 VAE (encoder + decoder).", Required: true,
                        Variants: [V("default", "BF16 · 167.7 MB", Hf(ZImageRepo, "ZImageTurbo_VAE_bf16.safetensors"), 167666902)],
                        Aliases: ["ZImageTurbo_VAE_bf16.safetensors"]),
                    new ModelFile(
                        "zimage.vae_config", "VAE config", "ZImageTurbo_VAE_bf16_config.json",
                        "VAE architecture metadata kept next to the checkpoint.", Required: false,
                        Variants: [V("default", "JSON · 805 B", Hf(ZImageRepo, "ZImageTurbo_VAE_bf16_config.json"), 805)]),
                    new ModelFile(
                        "zimage.scheduler_config", "Scheduler config", "ZImageTurbo_scheduler_config.json",
                        "WanGP scheduler metadata.", Required: false,
                        Variants: [V("default", "JSON · 173 B", Hf(ZImageRepo, "ZImageTurbo_scheduler_config.json"), 173)]),
                    new ModelFile(
                        "zimage.tokenizer_config", "Tokenizer config", "tokenizer_config.json",
                        "Optional Qwen3 tokenizer metadata (the engine reads tokenizer.json only).", Required: false,
                        Variants: [V("default", "JSON · 9.7 KB", Hf(ZImageRepo, ZImageEncoderFolder + "/tokenizer_config.json"), 9732)]),
                    new ModelFile(
                        "zimage.vocab", "Vocabulary", "vocab.json",
                        "Optional Qwen3 tokenizer metadata.", Required: false,
                        Variants: [V("default", "JSON · 2.78 MB", Hf(ZImageRepo, ZImageEncoderFolder + "/vocab.json"), 2776833)]),
                    new ModelFile(
                        "zimage.merges", "BPE merges", "merges.txt",
                        "Optional Qwen3 tokenizer metadata.", Required: false,
                        Variants: [V("default", "TXT · 1.67 MB", Hf(ZImageRepo, ZImageEncoderFolder + "/merges.txt"), 1671853)]),
                ],
                Note: "WanGP sources: defaults/z_image.json and z_image_handler.py::query_model_files()."),

            // ------------------------------------------------- Nunchaku int4
            new ModelGroup(
                "zimage-nunchaku", "Z-Image Nunchaku INT4", "zimage_nunchaku_dir",
                QziConfig.NunchakuDirectoryVariable, "models/zimage_nunchaku", ModelPathKind.Directory,
                "Optional SVDQuant INT4 repacks selected by checkpoint=nunchaku / nunchaku-r<rank>.",
                Optional: true,
                Files:
                [
                    new ModelFile(
                        "nunchaku.r256", "SVDQ INT4 r256", "svdq-int4_r256-z-image-turbo.safetensors",
                        "The only INT4 pack published by WanGP; fp4 repacks are rejected by the engine.", Required: false,
                        Variants: [V("default", "INT4 · 4.55 GB", Hf(ZImageRepo, "svdq-int4_r256-z-image-turbo.safetensors"), 4546148632)]),
                ],
                Note: "WanGP source: defaults/z_image_nunchaku_r256_int4.json."),

            // ------------------------------------------- ControlNet Union 2.1
            new ModelGroup(
                "zimage-control", "Z-Image ControlNet Union 2.1", "zimage_control_dir",
                QziConfig.ZImageControlDirectoryVariable, "models/zimage_control",
                ModelPathKind.FileOrDirectory,
                "Weights for zimage-control and zimage-inpaint. The config key may point at one .safetensors file or at a directory.",
                Optional: true,
                Files:
                [
                    new ModelFile(
                        "control.union21", "Fun ControlNet Union 2.1",
                        "Z-Image-Turbo-Fun-Controlnet-Union2.1_quanto_bf16_int8.safetensors",
                        "Union 2.1 weights: pose / canny / depth / scribble guides and Union 2.x inpaint.", Required: false,
                        Variants:
                        [
                            V("int8", "quanto BF16/INT8 · 3.77 GB (recommended)",
                              Hf(ZImageRepo, "Z-Image-Turbo-Fun-Controlnet-Union2.1_quanto_bf16_int8.safetensors"), 3765209401),
                            V("bf16", "BF16 · 6.71 GB",
                              Hf(ZImageRepo, "Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors"), 6712485600,
                              "Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors")
                        ],
                        Note: "The engine matches the Z-Image-Turbo-Fun-Controlnet-Union2.1_*.safetensors prefix when the config key is a directory."),
                ],
                Note: "WanGP source: defaults/z_image_control2_1.json. Inpaint has no separate checkpoint; it reuses these weights."),

            // ---------------------------------------- guide preprocessing
            new ModelGroup(
                "preprocess", "ControlNet guide preprocessor assets", "preprocess_dir",
                QziConfig.PreprocessDirectoryVariable, "models/preprocess", ModelPathKind.Directory,
                "WanGP ONNX / PyTorch assets used by POST /api/preprocess (host layer only; the native engine never loads them).",
                Optional: true,
                Files:
                [
                    new ModelFile(
                        "preprocess.yolox", "YOLOX-L person detector", "pose/yolox_l.onnx",
                        "DWPose person detector used for pose guides.", Required: false,
                        Variants: [V("default", "ONNX · 216.7 MB", Hf(Wan21Repo, "pose/yolox_l.onnx"), 216746733)]),
                    new ModelFile(
                        "preprocess.dwpose", "DWPose SimCC keypoints", "pose/dw-ll_ucoco_384.onnx",
                        "DWPose keypoint model; both ONNX files are required for learned pose.", Required: false,
                        Variants: [V("default", "ONNX · 134.4 MB", Hf(Wan21Repo, "pose/dw-ll_ucoco_384.onnx"), 134399116)]),
                    new ModelFile(
                        "preprocess.depth", "Depth Anything V2 ViT-L", "depth/depth_anything_v2_vitl.pth",
                        "WanGP's Depth Anything V2 checkpoints. The WebUI's learned estimator expects an ONNX export (depth*.onnx); these .pth files must be converted before use.", Required: false,
                        Variants:
                        [
                            V("vitl", "ViT-L PyTorch · 1.34 GB", Hf(Wan21Repo, "depth/depth_anything_v2_vitl.pth"), 1341395338),
                            V("vitb", "ViT-B PyTorch · 390 MB", Hf(Wan21Repo, "depth/depth_anything_v2_vitb.pth"), 389961218,
                              "depth/depth_anything_v2_vitb.pth")
                        ],
                        Note: "Without a matching ONNX file the WebUI keeps using the dependency-free managed depth proxy. WanGP ships no ONNX export of Depth Anything V2."),
                    new ModelFile(
                        "preprocess.scribble", "Contour / scribble model", "scribble/netG_A_latest.pth",
                        "WanGP contour checkpoint. The WebUI's learned contour estimator expects netG_A_latest.onnx / contour*.onnx.", Required: false,
                        Variants: [V("default", "PyTorch · 17.2 MB", Hf(Wan21Repo, "scribble/netG_A_latest.pth"), 17173511)],
                        Note: "canny / scribble fall back to managed Canny + Zhang-Suen thinning when no ONNX export is present."),
                ],
                Note: "WanGP sources: preprocessing/dwpose/assets.py, preprocessing/depth_anything_v2/assets.py, preprocessing/scribble.py."),
        ];
    }
}
