# QwenZImage.NET

把 **Qwen-Image 2.1**（turbo / base / LoRA / 图像编辑）与 **Z-Image**（bf16 / Nunchaku int4 / ControlNet / 遮罩重绘）封装成 .NET 10 库的独立项目。托管代码通过源生成的 P/Invoke 直接调用进程内的 `QwenZImageNative`：生成路径里没有 Go、没有 sidecar 服务、没有 HTTP 或 JSON 跳转。

- NuGet：<https://www.nuget.org/packages/QwenZImage.NET>
- 仓库：<https://github.com/dongfangzhizhu/QwenZImage.NET>

## 安装

```bash
dotnet add package QwenZImage.NET
```

要求 **.NET 10**。一个包同时包含 `win-x64` 与 `linux-x64` 的 native 库，不需要另外安装运行时；macOS 不在支持范围内。

**注意**：默认包内的 native 是**参考后端**（`qzi_stub.c`），只输出由 prompt / seed / 尺寸哈希出的确定性**噪点图**，用于无 GPU 环境验证 ABI、进度、队列与内存契约，**不是图像模型**。要生成真实图像，需用 `-DQZI_WITH_CUDA_ENGINE=ON` 自行构建 CUDA 版 `QwenZImageNative`，再用 `QZI_NATIVE_LIBRARY` 指向它，或替换 `runtimes/<rid>/native/` 里的文件。用 `client.GetHealth().IsReferenceBackend` 判断当前加载的是哪一种——「参数都对却全是噪点」基本都是这个原因。

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

`new QziClient()` 会读配置文件；也可 `new QziClient(QziClientOptions.FromConfig("/path/to/QwenZImage.config.json"))`，或完全用代码指定：`new QziClient(new QziClientOptions { Qwen21Directory = "/models/qwen21" })`。

## 能力

| 能力 | 入口 |
| --- | --- |
| Qwen-Image 2.1 文生图（turbo / base / LoRA） | `GenerateQwen21(Async)` · `QwenImageOptions` |
| 图像编辑（参考图 / mask / strength） | `GenerateQwen21Edit(Async)` · `Qwen21EditOptions` |
| 高级编辑（外扩 / LanPaint / RGBA） | `GenerateQwen21EditEx(Async)` · `Qwen21EditExOptions` |
| Z-Image Turbo（bf16 / Nunchaku int4） | `GenerateZImage(Async)` · `ZImageOptions` |
| Z-Image ControlNet Union 2.1 | `GenerateZImageControl(Async)` · `ZImageControlOptions` |
| Z-Image 遮罩重绘 | `GenerateZImageInpaint(Async)` · `ZImageInpaintOptions` |
| 串行队列 / 取消 / 重试 / 进度事件 | `QziGenerationQueue` |
| 零拷贝直写流 | `WriteTo(Async)` |

四个增量能力都是 ABI v1 的可选导出，用 `client.SupportsImageEditing` / `SupportsControlNet` / `SupportsAdvancedEditing` / `SupportsInpaint` 探测（inpaint 前置依赖 ControlNet）。核心库按 **raw raster** 语义消费控制图，不做姿态 / 边缘 / 深度估计；WebUI 会把上传的普通照片预处理成 guide。

一个 engine 同时只服务一个请求，并发会返回 `QZI_ERR_BUSY`；多调用方请用 `QziGenerationQueue` 串行分发，它提供 `Queued / Running / Done / Error / Cancelled` 状态、`WatchAsync()` 事件流、取消与重试。

## 配置

`QwenZImage.config.json` 指定模型权重目录。top-level 对所有平台生效，`windows` / `linux` / `osx` 按当前平台覆盖，`~` 开头会展开，`_` 开头的键视为注释。

```json
{
  "device_id": -1,
  "thread_count": 0,
  "qwen21_dir": "/models/qwen21",
  "qwen21_lora_dir": "/models/qwen21/loras",
  "zimage_dir": "/models/zimage",
  "zimage_nunchaku_dir": "/models/zimage_nunchaku",
  "zimage_control_dir": "/models/zimage/control_union_2.1.safetensors",
  "text_cache_dir": "/models/cache",
  "windows": { "qwen21_dir": "D:/models/qwen21" }
}
```

查找顺序：`QZI_CONFIG` / `--config` → 程序目录 → 工作目录 → 二者向上最多 8 级 → `~/.config/QwenZImage/config.json`。`QZI_*` 环境变量（`QZI_QWEN21_DIR`、`QZI_ZIMAGE_DIR`、`QZI_QWEN21_LORA_DIR`、`QZI_ZIMAGE_NUNCHAKU_DIR`、`QZI_ZIMAGE_CONTROL_DIR`、`QZI_MODEL_CACHE`、`QZI_DEVICE_ID`）优先于文件，代码里显式传入的 `QziClientOptions` 优先于环境变量。每个 engine 在创建时快照自己的配置，多实例互不串扰。

## 模型权重

**仓库与 NuGet 包都不分发权重。** 仓库自带的 WebUI 提供 Models 面板，覆盖 WanGP 用到的全部权重（6 组 / 32 个文件，地址取自 WanGP 源码的 `https://huggingface.co/<repo>/resolve/main/<path>`）：

- `POST /api/models/download` 下载，`.qzi-part` + `.qzi-etag` 配合 HTTP `Range` / `If-Range` 实现**断点续传**，完成校验长度后原子替换；`POST /api/models/control` 负责 pause / resume / reset。
- `GET /api/models/manual` 返回 Markdown，逐个文件给出精确地址、`curl -L -C -` 命令与需要合并进配置文件的 JSON 片段。
- `GET /api/models/config` 只预览；`POST /api/models/config` 带 `{"apply":true}` 才备份并原子写回 6 个目录键，返回 `restart_required`。

`QZI_HF_MIRROR` 只重写固定的 Hugging Face 前缀。下载功能仅存在于 `QwenZImage.NET.WebUI`（不打包），浏览器只能提交文件 id / variant，不能指定 URL 或目标路径。详见 [docs/WEBUI.md](docs/WEBUI.md)。

## WebUI（本地手动测试）

```bash
dotnet run --project src/QwenZImage.NET.WebUI -c Release   # http://127.0.0.1:7860/
```

ASP.NET Core Minimal API，Windows / Linux 同一套代码，提供 health、presets、串行队列、SSE 进度、画廊、取消重试、图像编辑 / ControlNet 上传与模型下载。要用 CUDA 后端出真图，用 `scripts/start-webui-cuda.ps1` / `.sh`：它会设置 `QZI_NATIVE_LIBRARY` 并开启 `QZI_REQUIRE_REAL_BACKEND=1`，加载到参考后端就直接返回 503，不会拿噪点图冒充结果。完整端点见 [docs/WEBUI.md](docs/WEBUI.md)。

## 参数要点

规则在 .NET 与 native 两层重复校验，避免把明显非法的请求送进分钟级模型加载。完整表格见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。

- **Qwen21**：宽高为 32 的倍数、`32..2048`、面积 ≤ `2048×2048`；turbo 固定 8 步 / guidance 1.0，base 为 steps 1..50 / guidance 1..20；`lora` 必须是裸文件名（≤ 255 UTF-8 字节），`lora_scale ∈ [0.05, 4]`。
- **ZImage**：宽高为 16 的倍数、`16..2048`、面积 ≤ `2048×2048`；`checkpoint` 支持 `bf16` / `nunchaku` / `nunchaku-<pack>`，不支持 fp4；Turbo 固定 8 步、无 CFG 分支。
- **编辑 / ControlNet**：参考图与控制图为平面 RGB8（每边 ≤ 4096、总像素 ≤ `2048²`），mask 为单通道；`ControlType` 是描述性元数据，核心库不跑预处理器；`ControlScale ∈ [0.05, 2.0]`（默认 0.65）。
- **通用**：prompt 非空且 ≤ 16384 UTF-8 字节，`flags` 必须为 0；`steps=0` / `guidance=0` / `lora_scale=0` 表示「用默认值 / 无 adapter」。

## NuGet 包内容

```
QwenZImage.NET.0.1.0.nupkg
├─ lib/net10.0/QwenZImage.NET.dll
├─ lib/net10.0/QwenZImage.NET.xml
├─ runtimes/win-x64/native/QwenZImageNative.dll
├─ runtimes/linux-x64/native/libQwenZImageNative.so
├─ README.md / LICENSE / THIRD_PARTY_NOTICES.md / engine/LICENSE
└─ docs/*.md
```

核心包**零运行期三方依赖**；WebUI、预处理 ONNX 资产与模型下载器都不在包内。native 库解析顺序：`QziClientOptions.NativeLibraryPath` / `QZI_NATIVE_LIBRARY` → 运行时的 `NATIVE_DLL_SEARCH_DIRECTORIES`（即 NuGet 的 `runtimes/<rid>/native/`）→ 程序集同目录及其上级 `artifacts/` → 系统默认探测。

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

真实 CUDA 构建用 `scripts/build-native-cuda.bat` / `.sh`（需 CUDA Toolkit 13+、MSVC 或 GCC/Clang + OpenMP；默认产出覆盖 Ampere → Blackwell 的 fat binary）。打包前置条件是两个平台的 native 库都已 stage 到 `artifacts/runtimes/<rid>/native/`，缺任一平台 `QziEnsureNativeRuntimes` 会直接报错。

## 文档

- [docs/PARAMETERS.md](docs/PARAMETERS.md) — 参数、默认值与校验规则
- [docs/NATIVE_ABI.md](docs/NATIVE_ABI.md) — C ABI v1、结构体布局与能力位
- [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) — 真实引擎接入与自定义 backend
- [docs/WEBUI.md](docs/WEBUI.md) — WebUI 端点、预处理与模型下载

## 发布

发布走 **NuGet Trusted Publishing**：`.github/workflows/release.yml` 在 `v*` tag 上触发，用 GitHub OIDC token 换短期 API key，仓库不保存长期 `NUGET_API_KEY`。当前版本 `0.1.0` 已发布。

## 许可与声明

MIT，见 [LICENSE](LICENSE)。`engine/` 下的真实推理引擎来自 MIT 许可的上游项目，许可与来源见 [engine/UPSTREAM.md](engine/UPSTREAM.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目**不包含任何模型权重、不包含 Go 代码、不包含上游 HTTP 服务/CLI**；模型名称与商标归各自权利人所有。
