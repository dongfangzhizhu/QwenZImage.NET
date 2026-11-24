# QwenZImage.NET

.NET 10 库，封装 **Qwen-Image 2.1**（turbo / base / LoRA / 图像编辑 / 身份参考）与 **Z-Image**（bf16 / Nunchaku int4 / ControlNet / 遮罩重绘）。托管代码通过源生成的 P/Invoke 直接调用进程内的 `QwenZImageNative`：生成路径里没有 Go、没有 sidecar 服务、没有 HTTP 或 JSON 跳转。

- NuGet：<https://www.nuget.org/packages/QwenZImage.NET>
- 仓库：<https://github.com/dongfangzhizhu/QwenZImage.NET>
- 当前版本：**0.3.0**

## 目录

- [安装与前置条件](#安装与前置条件)
- [快速上手](#快速上手)
- [0.3.0 新增能力](#030-新增能力)
- [能力](#能力)
- [配置](#配置)
- [模型权重](#模型权重)
- [WebUI](#webui)
- [参数规则](#参数规则)
- [NuGet 包内容](#nuget-包内容)
- [从源码构建](#从源码构建)
- [文档索引](#文档索引)
- [发布](#发布)
- [许可与声明](#许可与声明)

## 安装与前置条件

```bash
dotnet add package QwenZImage.NET
```

- 要求 **.NET 10**；一个包同时提供 `win-x64` 与 `linux-x64` native 库，无需另装运行时。macOS 不在支持范围内。
- **包内 native 是参考后端**（`qzi_stub.c`）：只输出由 prompt / seed / 尺寸哈希出的确定性**噪点图**，用于无 GPU 环境验证 ABI、进度、队列与内存契约，**不是图像模型**。
- 出真图需要自行编译 CUDA 版 native（见[从源码构建](#从源码构建)），再用 `QZI_NATIVE_LIBRARY` 指向它，或替换包内 `runtimes/<rid>/native/` 中的文件。
- 用 `client.GetHealth().IsReferenceBackend` 判断当前加载的是哪种后端。「参数都对却全是噪点」几乎都是这个原因。

## 快速上手

```csharp
using QwenZImage.NET;

using QziClient client = new();   // 自动读取 QwenZImage.config.json

IProgress<GenerationProgress> progress = new Progress<GenerationProgress>(
    p => Console.WriteLine($"{p.Phase} {p.Completed}/{p.Total}"));

using GenerationResult image = await client.GenerateZImageAsync(
    new ZImageOptions
    {
        Prompt = "A red fox walks through fresh snow, cinematic light",
        Width = 1024, Height = 1024, Seed = 42, Checkpoint = "bf16"
    }, progress);
await image.SaveAsync("zimage.png");

Console.WriteLine(client.GetHealth().IsReferenceBackend
    ? "reference backend (noise)" : "real C/CUDA backend");
```

客户端有三种构造方式：

| 方式 | 用途 |
| --- | --- |
| `new QziClient()` | 读取默认配置文件 |
| `new QziClient(QziClientOptions.FromConfig(path))` | 指定配置文件 |
| `new QziClient(new QziClientOptions { ... })` | 完全用代码指定，优先级最高 |

## 0.3.0 新增能力

| 新增 | 说明 | 入口 |
| --- | --- | --- |
| 完整 Qwen3-VL 视觉塔 | 身份参考不再只靠 VAE latent：`<\|image_pad\|>` + deepstack + 图像 grid mRoPE 与 VAE 参考 latent 双条件化，DiT 按 Wan2GP 4-slot joint 展开 | `GenerateQwen21IdentityRef(Async)` |
| 真正的低显存换入换出 | `low` / `lowest` 让 GPU 只保留 1 层 DiT 权重（约 0.2 GiB），其余每层每步从 checkpoint 换入；同种子输出与 `high` 字节一致。LoRA 请求自动回退常驻 | `SetMemoryProfile` |
| Z-Image 逐组件替换 | DiT / VAE / text encoder / tokenizer 可分别指向 BF16、Quanto INT8、Nunchaku int4、FP8 或 GGUF 文件；不设置时保持原单包行为 | `zimage_*_dir` 配置项或 `QZI_ZIMAGE_*` 环境变量 |

这三项改动已通过 **C ABI 冒烟测试 + 250 个托管单元测试**，并在本机 RTX 4090 + 本地模型上用真实 CUDA 后端完成出图验证；验证方式见[发布包真机验证](#发布包真机验证)。

## 能力

| 能力 | 入口 |
| --- | --- |
| Qwen-Image 2.1 文生图（turbo / base / LoRA） | `GenerateQwen21(Async)` · `QwenImageOptions` |
| 图像编辑（参考图 / mask / strength） | `GenerateQwen21Edit(Async)` · `Qwen21EditOptions` |
| 高级编辑（外扩 / LanPaint / RGBA） | `GenerateQwen21EditEx(Async)` · `Qwen21EditExOptions` |
| 身份参考（Wan2GP `image_refs`） | `GenerateQwen21IdentityRef(Async)` · `Qwen21IdentityRefOptions` |
| Z-Image Turbo（bf16 / Nunchaku int4） | `GenerateZImage(Async)` · `ZImageOptions` |
| Z-Image ControlNet Union 2.1 | `GenerateZImageControl(Async)` · `ZImageControlOptions` |
| Z-Image 遮罩重绘 | `GenerateZImageInpaint(Async)` · `ZImageInpaintOptions` |
| 低显存 / 内存 profile | `SetMemoryProfile` / `GetMemoryPlan` · `QziMemoryProfile` |
| 串行队列 / 取消 / 重试 / 进度事件 | `QziGenerationQueue` |
| 零拷贝直写流 | `WriteTo(Async)` |

### 可选能力探测

上表前八项是 ABI v1 的可选导出，用下列属性探测后再调用；`SupportsInpaint` 依赖 `SupportsControlNet`。

```text
SupportsImageEditing   SupportsControlNet       SupportsAdvancedEditing
SupportsInpaint        SupportsIdentityReference SupportsLowVram
```

### 关键语义

- **身份参考**：对齐 Wan2GP `image_refs`，支持 1–10 张带权重的参考图与编码上限。真实 CUDA 后端从纯噪声起步，用 VAE 参考 latent 与完整 Qwen3-VL 视觉塔（`<|image_pad|>` + deepstack + 图像 grid mRoPE）双条件化；DiT 采用 Wan2GP 的 4-slot joint 展开，可同时锁脸与大改服装。
- **低显存 profile**：`auto` 按设备总显存选择预算；`high` / `balanced` / `low` / `lowest` 调整 CUDA 分配器释放阈值（默认 24 GiB，`lowest` 为 256 MiB）与参考图编码上限（768 → 512 / 384 px）。`low` / `lowest` 还启用真正的 **DiT 逐块换入换出**（mmgp block swapping）：GPU 只保留 1 层 INT8 权重，其余 31 层每步从 checkpoint 换入，DiT 显存从 6.8 GiB 降到约 0.2 GiB，数值无损（同种子输出 SHA-256 与 `high` 一致）；带 LoRA 时自动回退常驻。WebUI 顶栏提供 `VRAM: ...` 下拉。
- **控制图**：核心库按 **raw raster** 语义消费，不做姿态 / 边缘 / 深度估计；WebUI 负责把上传的普通照片预处理成 guide。

### 并发

一个 engine 同时只服务一个请求，并发会返回 `QZI_ERR_BUSY`。多调用方请用 `QziGenerationQueue` 串行分发；它提供 `Queued / Running / Done / Error / Cancelled` 状态、`WatchAsync()` 事件流、取消与重试。

## 配置

`QwenZImage.config.json` 指定模型权重目录：top-level 对所有平台生效，`windows` / `linux` / `osx` 按当前平台覆盖；`~` 开头会展开，`_` 开头的键视为注释。

```json
{
  "device_id": -1,
  "thread_count": 0,
  "memory_profile": "auto",
  "qwen21_dir": "/models/qwen21",
  "qwen21_lora_dir": "/models/qwen21/loras",
  "zimage_dir": "/models/zimage",
  "zimage_nunchaku_dir": "/models/zimage_nunchaku",
  "zimage_control_dir": "/models/zimage/control_union_2.1.safetensors",
  "zimage_dit_dir": "/models/zimage/dit_fp8.safetensors",
  "zimage_vae_dir": "/models/zimage/vae_gguf.gguf",
  "zimage_text_dir": "/models/zimage/text_int8",
  "zimage_tokenizer_path": "/models/zimage/tokenizer.json",
  "text_cache_dir": "/models/cache",
  "windows": { "qwen21_dir": "D:/models/qwen21" }
}
```

**查找顺序**：`QZI_CONFIG` / `--config` → 程序目录 → 工作目录 → 二者向上最多 8 级 → `~/.config/QwenZImage/config.json`。

**优先级**：代码里显式传入的 `QziClientOptions` > `QZI_*` 环境变量 > 配置文件。每个 engine 在创建时快照自己的配置，多实例互不串扰。

| 环境变量 | 作用 |
| --- | --- |
| `QZI_CONFIG` | 配置文件路径 |
| `QZI_NATIVE_LIBRARY` | 指定 native 库路径（切换到 CUDA 构建） |
| `QZI_REQUIRE_REAL_BACKEND` | 设为 `1` 时禁止加载参考后端 |
| `QZI_QWEN21_DIR` / `QZI_QWEN21_LORA_DIR` | Qwen-Image 2.1 权重目录 |
| `QZI_ZIMAGE_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR` / `QZI_ZIMAGE_CONTROL_DIR` | Z-Image 默认 / Nunchaku / ControlNet 权重路径 |
| `QZI_ZIMAGE_DIT_DIR` / `QZI_ZIMAGE_VAE_DIR` / `QZI_ZIMAGE_TEXT_DIR` / `QZI_ZIMAGE_TOKENIZER_PATH` | 逐组件覆盖 DiT / VAE / text encoder / tokenizer，支持 BF16 / FP8 / GGUF / 现有 INT4、INT8 |
| `QZI_MODEL_CACHE` | 文本编码缓存目录 |
| `QZI_DEVICE_ID` | GPU 设备号 |
| `QZI_MEMORY_PROFILE` | `auto` / `high` / `balanced` / `low` / `lowest` |

## 模型权重

**仓库与 NuGet 包都不分发权重。** 仓库自带的 WebUI 提供 Models 面板，覆盖 WanGP 用到的全部权重（6 组 / 32 个文件，地址取自 WanGP 源码的 `https://huggingface.co/<repo>/resolve/main/<path>`）：

| 端点 | 作用 |
| --- | --- |
| `POST /api/models/download` | 下载；`.qzi-part` + `.qzi-etag` 配合 HTTP `Range` / `If-Range` 断点续传，校验长度后原子替换 |
| `POST /api/models/control` | pause / resume / reset |
| `GET /api/models/manual` | 返回每个文件的精确地址、`curl -L -C -` 命令与配置片段 |
| `GET` / `POST /api/models/config` | 只读预览 / 带 `{"apply":true}` 才备份并原子写回 6 个目录键，返回 `restart_required` |

`QZI_HF_MIRROR` 只重写固定的 Hugging Face 前缀。下载功能仅存在于 `QwenZImage.NET.WebUI`（不打包）；浏览器只能提交文件 id / variant，不能指定 URL 或目标路径。详见 [docs/WEBUI.md](docs/WEBUI.md)。

## WebUI

```bash
dotnet run --project src/QwenZImage.NET.WebUI -c Release   # http://127.0.0.1:7860/
```

ASP.NET Core Minimal API，Windows / Linux 同一套代码，提供 health、presets、串行队列、SSE 进度、画廊、取消重试、图像编辑 / ControlNet 上传、身份参考与模型下载。

出真图请用 `scripts/start-webui-cuda.ps1` / `.sh`：它会设置 `QZI_NATIVE_LIBRARY` 并开启 `QZI_REQUIRE_REAL_BACKEND=1`，加载到参考后端就直接返回 503，不会拿噪点图冒充结果。完整端点见 [docs/WEBUI.md](docs/WEBUI.md)。

## 参数规则

规则在 .NET 与 native 两层重复校验，避免把明显非法的请求送进分钟级模型加载。完整表格见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。

| 范围 | 规则 |
| --- | --- |
| Qwen21 | 宽高为 32 的倍数、`32..2048`、面积 ≤ `2048×2048`；turbo 固定 8 步 / guidance 1.0，base 为 steps 1..50 / guidance 1..20 |
| Qwen21 LoRA | `lora` 必须是裸文件名（≤ 255 UTF-8 字节），`lora_scale ∈ [0.05, 4]` |
| ZImage | 宽高为 16 的倍数、`16..2048`、面积 ≤ `2048×2048`；`checkpoint` 支持 `bf16` / `nunchaku` / `nunchaku-<pack>`，不支持 fp4；Turbo 固定 8 步、无 CFG 分支 |
| 编辑 / ControlNet | 参考图与控制图为平面 RGB8（每边 ≤ 4096、总像素 ≤ `2048²`），mask 为单通道；`ControlType` 是描述性元数据，核心库不跑预处理器；`ControlScale ∈ [0.05, 2.0]`（默认 0.65） |
| 通用 | prompt 非空且 ≤ 16384 UTF-8 字节，`flags` 必须为 0；`steps=0` / `guidance=0` / `lora_scale=0` 表示「用默认值 / 无 adapter」 |

## NuGet 包内容

以 `QwenZImage.NET.0.3.0.nupkg` 为例：

```text
├─ lib/net10.0/QwenZImage.NET.dll
├─ lib/net10.0/QwenZImage.NET.xml
├─ runtimes/win-x64/native/QwenZImageNative.dll
├─ runtimes/linux-x64/native/libQwenZImageNative.so
├─ README.md / LICENSE / THIRD_PARTY_NOTICES.md / engine/LICENSE
└─ docs/*.md
```

核心包**零运行期三方依赖**；WebUI、预处理 ONNX 资产与模型下载器都不在包内。

native 库解析顺序：`QziClientOptions.NativeLibraryPath` / `QZI_NATIVE_LIBRARY` → 运行时的 `NATIVE_DLL_SEARCH_DIRECTORIES`（即 NuGet 的 `runtimes/<rid>/native/`）→ 程序集同目录及其上级 `artifacts/` → 系统默认探测。

## 从源码构建

```bash
# 参考 native（无 GPU）
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build native/build
ctest --test-dir native/build --output-on-failure

# 全量构建 + 测试 + 打包
dotnet build QwenZImage.NET.sln -c Release
dotnet test  QwenZImage.NET.sln -c Release
dotnet pack  src/QwenZImage.NET/QwenZImage.NET.csproj -c Release -o artifacts/packages
```

真实 CUDA 构建用 `scripts/build-native-cuda.bat` / `.sh`：需 CUDA Toolkit 13+、MSVC 或 GCC/Clang + OpenMP，默认产出覆盖 Ampere → Blackwell 的 fat binary。打包前置条件是两个平台的 native 库都已 stage 到 `artifacts/runtimes/<rid>/native/`；缺任一平台，`QziEnsureNativeRuntimes` 会直接报错。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | 参数、默认值与校验规则 |
| [docs/NATIVE_ABI.md](docs/NATIVE_ABI.md) | C ABI v1、结构体布局与能力位 |
| [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) | 真实引擎接入与自定义 backend |
| [docs/WEBUI.md](docs/WEBUI.md) | WebUI 端点、预处理与模型下载 |

## 发布

发布走 **NuGet Trusted Publishing**：`.github/workflows/release.yml` 在 `v*` tag 上触发（或手动 dispatch），用 GitHub OIDC token 换短期 API key，仓库不保存长期 `NUGET_API_KEY`。版本以 `v<version>` tag 为准（tag 会覆盖 csproj 里的 `<Version>` 作为打包版本）。已发布版本：`0.1.0`、`0.2.0`、`0.2.1`、`0.3.0`。

从源码发版：

```bash
# 1. 本地先验证（参考后端，无需 GPU）
dotnet build QwenZImage.NET.sln -c Release
dotnet test  QwenZImage.NET.sln -c Release --no-build

# 2. 提升 csproj <Version>，提交后打 tag 推送，CI 自动打包发布
git tag v0.3.0 && git push origin v0.3.0
```

### 发布包真机验证

NuGet 包内是**参考后端**（确定性噪点），真图验证需另建 CUDA native 并让包 API 指向它。`tools/PublishedPackageSmokeTest` 是一个**独立控制台项目**：只有 `PackageReference`，没有对仓库源码的 `ProjectReference`，因此加载的就是 nuget.org 上刚发布的 DLL。

```bash
# 先编译真实后端：scripts/build-native-cuda.bat 89（按 GPU 架构调整）
dotnet run --project tools/PublishedPackageSmokeTest -c Release -- \
  --native artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll \
  --out artifacts/published-smoke
```

程序强制 `QZI_REQUIRE_REAL_BACKEND=1`，拒绝回退到噪点后端；它读取 `QwenZImage.config.json` 中的本地权重，依次验证 Z-Image 文生图、Qwen-Image 2.1 文生图、身份参考以及 `low` profile 的逐块换入换出（同 prompt / seed 与 `high` 输出 SHA-256 对比），全部通过才返回 0。

## 许可与声明

MIT，见 [LICENSE](LICENSE)。`engine/` 下的真实推理引擎来自 MIT 许可的上游项目，许可与来源见 [engine/UPSTREAM.md](engine/UPSTREAM.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目**不包含任何模型权重、不包含 Go 代码、不包含上游 HTTP 服务/CLI**；模型名称与商标归各自权利人所有。
