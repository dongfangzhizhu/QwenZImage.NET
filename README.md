# QwenZImage.NET

> **High-performance .NET 10 wrapper over a stable C ABI for Qwen-Image 2.1 and Z-Image text-to-image generation.**
> Pure C at the bottom, source-generated P/Invoke at the top: no Go, no HTTP hop to a sidecar, no JSON marshalling in the hot path.

> ⚠️ **本项目自带的 native 库不生成真实图像。** 包内 `qzi.dll` / `libqzi.so` 是 deterministic **reference backend**：输出由 prompt / seed / 尺寸哈希出来的**噪点图**，`steps`、`guidance`、`seed` 只驱动进度条，不做任何扩散推理。原 Go WebUI 能出正常图像，是因为它连的是 `h3 serve`（真正的 C/CUDA 引擎 + 模型权重）；**那个引擎没有被搬进本仓库**。所以「同样的 steps 却是噪点」不是参数问题，而是后端根本不是图像模型。
>
> **最快的真图路径**：启动 `h3 serve --port 8411`，再用 `QZI_ENGINE_URL=http://127.0.0.1:8411` 启动本仓库的 WebUI（详见 [出真实图像](#出真实图像三条路径)）。库本身要出真图，仍需按 [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) 接入真实 `qzi_backend`。WebUI 检测到参考后端时会显示黄色横幅。

一个把 **Qwen-Image 2.1（turbo / base / LoRA）** 与 **Z-Image（bf16 / Nunchaku int4）** 两个文生图能力封装成 `.NET 10` 可调用包的独立项目。底层是 C 语言定义的稳定 ABI（C/CUDA 引擎可实现它），上层是 `.NET 10` 的高性能源生成 P/Invoke 封装。原 Go WebUI 的队列、进度、取消、重试、预设等功能全部用纯 `.NET` 重写，**仓库中不含任何 Go 代码、库调用链上不依赖 HTTP 服务**。

仓库另附一个**可选的跨平台 WebUI 手动测试控制台**（`src/QwenZImage.NET.WebUI`，ASP.NET Core Minimal API，Windows / Linux 均可运行），把原本 Go WebUI 的交互面用纯 `.NET` 复刻出来。它默认走进程内 native，也可以用 `QZI_ENGINE_URL` **可选地**连本机 `h3 serve` 以产出真实图像——该 HTTP 适配器只存在于 WebUI 测试工具中，不进 NuGet 包、不改变库的 P/Invoke 契约，也没有任何 Go 代码。详见 [docs/WEBUI.md](docs/WEBUI.md)。

```
┌──────────────────────────────────────────────────────────────┐
│  Your app (.NET 10)                                          │
│    QziClient / QziGenerationQueue / QziPresets               │
│    ArrayPool · pinned UTF-8 · UnmanagedCallersOnly callbacks │
├──────────────────────────────────────────────────────────────┤
│  QwenZImage.NET.dll  (LibraryImport, blittable structs)      │
├──────────────────────────────────────────────────────────────┤
│  qzi.dll / libqzi.so   ← 稳定 C ABI，native/include/qzi.h      │
│    qzi.c 控制层 + qzi_backend vtable（可插拔）                 │
│    ├─ reference backend（本仓库自带，确定性、无需 GPU）        │
│    └─ 真实 C/CUDA 引擎（外部实现，见 ENGINE_INTEGRATION.md）   │
└──────────────────────────────────────────────────────────────┘
```

## ⚠️ 当前状态（务必先读）

| 项 | 状态 |
| --- | --- |
| C ABI v1、`qzi` 共享库、`.NET 10` 封装、队列、预设、47 个 xUnit 测试、native smoke test | ✅ 已完成并通过 |
| **跨平台 WebUI 手动测试控制台**（health / presets / jobs / SSE / gallery / trash / unload） | ✅ 已完成，Windows + Linux 均可运行 |
| **WebUI 可选接入真实 `h3 serve`**（`QZI_ENGINE_URL` / `--engine`），产出真实图像 | ✅ 已完成，16 个协议单测 + 端到端 HTTP 冒烟已验证 |
| **单个 NuGet 包同时包含 .NET 程序集 + win-x64 + linux-x64 native** | ✅ 已完成，`QziEnsureNativeRuntimes` 会在缺平台时拒绝打包 |
| 本地 `git` 仓库与首次提交 | ✅ 已完成 |
| 推送 GitHub / 发布 NuGet | ⏸ **按约定暂缓**，命令与 CI 工作流已备好，见 [PLAN.md](PLAN.md) |
| 真实 CUDA 推理引擎 | ❌ **本仓库不含**。库要靠外部 `qzi_backend`；WebUI 可直接连本机已有的 `h3 serve` |
| 模型权重 | ❌ 不含。NuGet 包只有程序集与参考 native 库，不捆绑多 GB 权重 |

**NuGet 包里捆绑的 `qzi.dll` / `libqzi.so` 是确定性参考实现（reference backend）**：它按相同的 ABI、相同的 progress 阶段、相同的缓冲区所有权返回一张由 prompt / seed / 尺寸决定的 PNG，**不做真实扩散推理，不需要 GPU 或权重**。它的用途是让包开箱即用、让 CI 与单测在无 GPU 环境下验证 ABI 与内存契约。要生成真实图像，请把 `QZI_NATIVE_LIBRARY` 指向实现了同一 ABI 的引擎构建，并按 [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) 注册后端——托管程序集不需要改动。

## 出真实图像：三条路径

噪点图的根因是 `native/src/qzi_stub.c`：`make_rgb()` 直接写伪随机 RGB，`run_phases()` 的 `dit` 循环只上报进度，**没有任何去噪、DiT、VAE 计算**。`steps` 只决定进度条跑几格，因此调参与原 Go 版保持一致也不会改变结果。三条路径按代价从低到高：

| 路径 | 做法 | 适用范围 | 代价 |
|---|---|---|---|
| **A. WebUI 连 `h3 serve`（最快）** | 启动 `h3 serve --port 8411`，再 `QZI_ENGINE_URL=http://127.0.0.1:8411 dotnet run --project src/QwenZImage.NET.WebUI -c Release` | 仅 WebUI / 手动测试 | 零编译，直接用你已有的 `h3.exe` 与 `h3-models/` 权重；WebUI 的队列、SSE、画廊全部保留 |
| **B. 替换 native 库** | 自己构建同名 `qzi.dll` / `libqzi.so`，实现 `qzi_backend` vtable 并注册真实 C/CUDA 引擎，用 `QZI_NATIVE_LIBRARY` 覆盖 | 所有 .NET 调用方 | 需把 `h3_serve.c` 的编排搬进 native 库，要 CUDA Toolkit 与权重 |
| **C. 直接导出 `qzi_*`** | 另写一个实现 `native/include/qzi.h` 全部导出的库，不用本仓库的 `qzi.c` 控制层 | 所有 .NET 调用方 | 工作量最大，需自行实现单飞锁、校验、进度、buffer 所有权 |

路径 A 的适配器是 `H3ServeExecutor`：把强类型请求转成 `h3 serve` 的 `POST /generate` JSON（qwen21 turbo 强制 `steps=8` / `guidance=1.0` 且不发 `negative_prompt`；zimage 不发 `steps`/`guidance`/`frames`/`pdd8`），把返回的 PNG 字节用 `GenerationResult.FromPng(...)` 包回队列，轮询 `GET /progress` 转成 SSE 进度，并把 `/health`、`/unload` 映射到 WebUI 端点。B/C 的细节见 [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md)。

> 路径 A **只在 WebUI 里**，NuGet 包依旧是纯 Native P/Invoke；`QZI_ENGINE_URL` 不设置时行为与之前完全一致（仍是参考后端 + 黄色告警横幅）。

## 环境要求

- **.NET 10 SDK**（`global.json` 锁定 `10.0.100`，`rollForward: latestFeature`）
- 构建参考 native 库：**CMake ≥ 3.20 + C11 编译器**（Windows 可用 MinGW / MSVC，Linux 用 GCC/Clang）
- 打包前需要 **Windows 与 Linux 两个平台的 native 产物**（Windows 主机可用 WSL 一次构建两者）
- 真实推理（可选）：实现 `qzi_backend` 的 C/CUDA 引擎 + 对应模型权重

## 快速开始

```bash
# 1) 构建参考 native 库（输出到 artifacts/native/<Platform>-<Arch>/）
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure

# 2) 全量构建 + 测试（库、测试、示例、WebUI）
dotnet build QwenZImage.NET.sln -c Release
dotnet test  QwenZImage.NET.sln -c Release

# 3) 跑示例（默认走参考后端，输出 outputs/qwen21.png、outputs/zimage.png）
dotnet run --project samples/QwenZImage.NET.Sample -c Release

# 4) 启动手动测试 WebUI，浏览器打开 http://127.0.0.1:7860/
dotnet run --project src/QwenZImage.NET.WebUI -c Release     # 参考后端，输出噪点

# 4b) 想看真实图像：先起真实引擎，再把 WebUI 指过去
# h3 serve --port 8411 --qwen21-dir /path/to/qwen21 --zimage-dir /path/to/zimage
QZI_ENGINE_URL=http://127.0.0.1:8411 dotnet run --project src/QwenZImage.NET.WebUI -c Release
# 等价：dotnet run --project src/QwenZImage.NET.WebUI -c Release -- --engine http://127.0.0.1:8411
```

一键脚本（都会自动 stage 成 NuGet 的 RID 布局）：

```bash
# Windows 主机：一次构建 qzi.dll + (WSL) libqzi.so
powershell -File scripts/build-native-all.ps1

# 仅当前平台
powershell -File scripts/build-native.ps1   # Windows
bash scripts/build-native-linux.sh          # Linux 或 WSL
bash scripts/build-native.sh                # 通用 Unix 构建 + stage
```

若 native 库不在自动探测路径里（例如要接入真实 CUDA 引擎）：

```bash
export QZI_NATIVE_LIBRARY=/path/to/qzi.dll      # 或 libqzi.so / libqzi.dylib
```

## 在 .NET 项目里使用

发布后：`dotnet add package QwenZImage.NET`。本地开发可直接 `dotnet add reference src/QwenZImage.NET/QwenZImage.NET.csproj`。

```csharp
using QwenZImage.NET;

var options = new QziClientOptions
{
    Qwen21Directory = "/models/qwen21",
    ZImageDirectory = "/models/zimage",
    CacheDirectory  = "/tmp/qzi-text-cache",
    // NativeLibraryPath = @"C:\engine\qzi.dll"  // 或设置 QZI_NATIVE_LIBRARY
};

using QziClient client = new(options);

IProgress<GenerationProgress> progress = new Progress<GenerationProgress>(
    p => Console.WriteLine($"{p.Phase} {p.Completed}/{p.Total} ({p.Elapsed})"));

// Qwen-Image 2.1（turbo 固定 8 步 / guidance 1.0）
using GenerationResult qwen = await client.GenerateQwen21Async(
    new QwenImageOptions
    {
        Prompt = "A red fox walks through fresh snow, cinematic light",
        Width = 1024, Height = 1024, Seed = 42
    }, progress);
await qwen.SaveAsync("qwen21.png");

// Z-Image（固定 8 步 Turbo 调度，无 CFG 分支）
using GenerationResult zimage = await client.GenerateZImageAsync(
    new ZImageOptions
    {
        Prompt = "A red fox walks through fresh snow, cinematic light",
        Width = 1024, Height = 1024, Seed = 42, Checkpoint = "bf16"
    }, progress);
await zimage.SaveAsync("zimage.png");

// 零拷贝直写流：PNG 不落地到托管缓冲区
using FileStream fs = File.Create("zimage-streamed.png");
await client.WriteToAsync(fs, new ZImageOptions { Prompt = "same", Width = 512, Height = 512 });
```

### 串行队列（替代原 Go WebUI 的队列）

一个 native engine 只持有 **一张 GPU + 一个常驻模型**，并发第二次生成会返回 `QZI_ERR_BUSY`。`QziGenerationQueue` 用 `System.Threading.Channels` 做严格串行分发：

```csharp
using QziGenerationQueue queue = new(client, async (job, result, ct) =>
    await result.SaveAsync($"outputs/{job.Id}.png", ct));

QziJob job = queue.Submit(QziPresets.WithPrompt(QziPresets.ZImage[1], "a mountain lake"));
await queue.WaitForIdleAsync();          // 或 await foreach (var j in queue.WatchAsync())
Console.WriteLine($"{job.Id}: {job.State}, {job.OutputLength} bytes");

queue.Cancel(anotherId);                 // 排队中的任务不会调用引擎；运行中的走 native 取消回调
QziJob? retry = queue.Retry(job.Id);     // 复用完全相同的 request 重跑
```

支持状态 `Queued / Running / Done / Error / Cancelled`、`JobChanged` 事件、`WatchAsync()` 异步流（替代 SSE）。

## 手动测试 WebUI（跨平台）

`src/QwenZImage.NET.WebUI` 是纯 `.NET 10` / ASP.NET Core Minimal API 的本地控制台，Windows 与 Linux 同一套代码、无需 Node/npm、无 Go 进程。它复刻了原 Go WebUI 的交互面：health、presets、串行任务队列、取消、重试、SSE 实时进度、输出画廊、删除到 `.trash` / 恢复、卸载模型。

```bash
dotnet run --project src/QwenZImage.NET.WebUI -c Release
# 浏览器打开 http://127.0.0.1:7860/
QZI_WEBUI_URLS=http://127.0.0.1:51234 dotnet run --project src/QwenZImage.NET.WebUI -c Release  # 换端口

# 真实图像：连本机 h3 serve（原生 Windows / WSL / Linux 均可）
QZI_ENGINE_URL=http://127.0.0.1:8411 dotnet run --project src/QwenZImage.NET.WebUI -c Release
```

主要端点：`GET /api/health`、`GET /api/presets`、`POST|GET /api/jobs`、`GET /api/jobs/{id}`、`POST /api/jobs/{id}/cancel|/retry`、`GET /api/events`（SSE）、`GET|DELETE /api/outputs`、`POST /api/outputs/restore`、`GET|POST /api/unload`。可用环境变量：`QZI_ENGINE_URL`、`--engine`、`QZI_ENGINE_TIMEOUT_MIN`、`QZI_ENGINE_PROGRESS_MS`、`QZI_WEBUI_URLS`、`QZI_WEBUI_OUTPUT`、`QZI_MODEL_DIR`、`QZI_MODEL_CACHE`、`QZI_NATIVE_LIBRARY`、`QZI_DEVICE_ID`。完整说明见 [docs/WEBUI.md](docs/WEBUI.md)。

> WebUI 是**手动测试工具**，不打进 NuGet 包。设置 `QZI_ENGINE_URL` 时完全不加载 bundled native 库，`/api/health` 会返回 `transport: "h3-serve-http"`、`reference_backend: false`；未设置时显示黄色噪点告警横幅。即使 native 库缺失，服务仍会启动并在 `/api/health` 明确报错，方便排查环境问题。

## API 速查

| 类型 | 说明 |
| --- | --- |
| `QziClient` | 拥有的一个 native engine。同步 `Generate*` / `WriteTo`，异步 `Generate*Async` / `WriteToAsync`，`GetHealth()`、`Unload()`、`Dispose()` |
| `QwenImageOptions` / `ZImageOptions` | 强类型请求（继承 `GenerationOptions`），`required Prompt`、宽高、Seed |
| `GenerationResult` | PNG 结果：`Png` / `PngSpan`（零拷贝）、`CopyTo(Async)`、`SaveAsync`、`Dispose()` 归还池化缓冲区；`FromPng(...)` 把外部来源的 PNG 包成同一类型，供自定义 `IQziExecutor` 接入队列 |
| `GenerationProgress` | `Phase`（`ModelLoad / Text / Dit / Vae / Encode`）、`Completed`、`Total`、`Elapsed` |
| `EngineHealth` | 常驻模型、Busy、队列深度、显存、Served/Rejected、运行时长、`IsReferenceBackend` |
| `QziGenerationQueue` | 串行队列、取消、重试、事件流、输出 sink |
| `QziPresets` | 5 个 Qwen21 + 4 个 ZImage 预设模板，用 `WithPrompt()` 变成可执行请求 |
| `RequestValidator` | 托管侧参数校验（与 native 侧规则一致，先于分钟级模型加载失败） |
| `QziException` | native 错误映射，含 `Status` 与 `NativeMessage` |

## 参数与校验

规则在 `.NET` 与 native 两层重复执行，避免把明显非法的请求送进分钟级模型加载。完整表格见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。要点：

- **Qwen21**：宽高为 **32 的倍数**、`32..2048`、面积 ≤ `2048×2048`；`turbo` 强制 `steps=8`、`guidance=1.0`、不支持 `negative_prompt`；`base` 为 `steps 1..50`、`guidance 1..20`；`lora` 必须是**裸文件名**（禁止路径分隔符与 `..`）、`lora_scale 0.05..4`。
- **ZImage**：宽高为 **16 的倍数**、`16..2048`、面积 ≤ `2048×2048`；`checkpoint` 支持 `bf16`、`nunchaku`、`nunchaku-<pack>`；**`fp4` / `svdq-fp4` 不被支持**（引擎只解码 int4）；无 steps/guidance/negative prompt（Turbo 固定 8 步、无 CFG 分支）。
- 两者：prompt 非空且 ≤ **16384 UTF-8 字节**，保留字段 `flags` 必须为 0。

## 性能设计

- `LibraryImport` **源生成 P/Invoke** + 全 blittable POD 结构体，无反射式 marshalling；启动时 `qzi_abi_layout()` 校验 6 个结构体尺寸与 ABI 版本，packing 不匹配立即失败而不是踩坏内存。
- `ArrayPool<byte>` 池化、pinned NUL-terminated UTF-8 字符串（`GCHandle`）、`Span` / `ReadOnlySpan`，热路径**零 string 分配**。
- `static` + `UnmanagedCallersOnly` 反向回调（progress / cancel），无 delegate 分配、无 GC 生命周期问题。
- 成功路径 native PNG buffer → 一次 `memcpy` → 立刻 `qzi_image_release`；`WriteTo` 走 `UnmanagedMemoryStream` **真正零拷贝**。
- 纯同步 native 调用 + `Task.Run` 异步外壳，避免把长时 GPU 任务压在线程池之外的隐式调度上；队列保证单飞。

## NuGet 包内容（.NET + Windows + Linux native）

同一个 `QwenZImage.NET.<version>.nupkg` 按 NuGet 的 RID 约定同时携带三个运行体，安装后 `.NET` 会按当前平台自动选取 native 库：

```
QwenZImage.NET.0.1.0.nupkg
├─ lib/net10.0/QwenZImage.NET.dll          ← .NET 10 托管程序集
├─ lib/net10.0/QwenZImage.NET.xml          ← XML 文档
├─ runtimes/win-x64/native/qzi.dll         ← Windows x64 reference backend
├─ runtimes/linux-x64/native/libqzi.so     ← Linux x64 reference backend
├─ README.md / LICENSE / THIRD_PARTY_NOTICES.md
└─ docs/*.md                                ← ABI、参数、引擎接入、WebUI 文档
```

打包前的两个前置条件：

1. 两个平台的 native 库必须先构建并 stage 到 `artifacts/runtimes/<rid>/native/`（用上面的 `scripts/build-native-all.ps1`，或在 CI 中分别由 Windows / Ubuntu runner 构建后合并）。
2. `QwenZImage.NET.csproj` 的 `QziEnsureNativeRuntimes` target 会在 `GenerateNuspec` 前检查两者存在；**缺任何一个平台都会直接报错并拒绝产出缺平台的包**，不会出现"在 Linux 上打出来的包只有 .so"这种事故。

```bash
dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj -c Release -o artifacts/packages
```

CI 与 release 工作流都会用 PowerShell 打开 `.nupkg` 逐项校验 `lib/net10.0/QwenZImage.NET.dll`、`runtimes/win-x64/native/qzi.dll`、`runtimes/linux-x64/native/libqzi.so`，且 native 文件大小非零。

> 包内 native 库是**参考后端**。要接真实 CUDA 引擎，用 `QZI_NATIVE_LIBRARY` 指向你自己的同名实现，或直接替换应用输出目录里的 `runtimes/<rid>/native/` 文件；托管程序集与 API 不变。模型权重始终由使用方自带。

## native 库解析顺序

1. `QziClientOptions.NativeLibraryPath`（或环境变量 `QZI_NATIVE_LIBRARY`）— **最高优先级**，用于切换到真实引擎
2. 运行时提供的 `NATIVE_DLL_SEARCH_DIRECTORIES`（NuGet `runtimes/<rid>/native/` 即通过它生效）
3. 托管程序集同目录，含 `runtimes/<rid>/native/` 与 `native/`
4. 从程序集目录向上查找 `artifacts/runtimes/<rid>/native/` 与 `artifacts/native/<Platform>-<Arch>/`（本地开发 / CI）
5. 运行时默认探测（`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`）

文件名：Windows `qzi.dll`、Linux `libqzi.so`、macOS `libqzi.dylib`。当前包提供 **win-x64 与 linux-x64**；其他平台可自行编译同名库并用 `QZI_NATIVE_LIBRARY` 指定。

## 目录结构

```
native/include/qzi.h           稳定 C ABI（唯一必须遵守的契约）
native/include/qzi_backend.h   可插拔后端 vtable
native/src/qzi.c               ABI 控制层：生命周期、校验、单飞锁、错误、进度
native/src/qzi_stub.c          确定性参考后端
native/src/qzi_png.c           无依赖的最小 PNG 编码器（stored deflate）
native/tests/                  C ABI smoke test（ctest）
src/QwenZImage.NET/            .NET 10 库（NuGet 包体）
src/QwenZImage.NET.WebUI/      跨平台手动测试 WebUI（ASP.NET Core，不打包）
  wwwroot/                     纯静态前端：index.html / app.js / style.css
  H3ServeExecutor.cs          可选的真实 h3 serve HTTP 适配器（仅 WebUI）
tests/QwenZImage.NET.Tests/          31 个库 xUnit 测试
tests/QwenZImage.NET.WebUI.Tests/    16 个 h3 serve 协议单测（stub handler，无网络/GPU）
samples/                       端到端示例
integration/                   真实引擎后端模板
scripts/                       Windows / Linux / WSL native 构建与 RID staging
                               + mock-h3-serve.py（协议冒烟用假引擎）
artifacts/runtimes/<rid>/     打包前 stage 的 win-x64 / linux-x64 native
docs/                          ABI / 参数 / 引擎接入 / WebUI 文档
.github/workflows/             CI 与 NuGet 发布工作流（双平台 native 校验）
PLAN.md                        计划、决策记录与发布清单
```

## 测试

```bash
ctest --test-dir native/build --output-on-failure   # C ABI 契约
dotnet test QwenZImage.NET.sln -c Release           # 47 个托管测试（31 库 + 16 WebUI 协议）
```

托管测试包含：参数校验、ABI 布局、PNG 生成与阶段回调、确定性输出、零拷贝写流、health 计数、异步包装、预取消、native 层重复校验、队列串行/取消/重试/事件流；WebUI 协议测试覆盖 qwen21 turbo/base 与 zimage 的 `/generate` 请求体、错误码映射、`/health`、`/unload`、非 PNG 拒绝。全部不需要 GPU、权重或网络监听。

WebUI 在 CI 中另有 Linux 运行时冒烟测试：启动 Kestrel → 验证 `/api/health` 的 `ok=true` 与 `reference_backend` → 检查 presets → 提交一个 zimage job → 轮询到 `done` → 确认输出文件出现在 `/api/outputs`。此外用 `scripts/mock-h3-serve.py` 起一个假 `h3 serve`，验证 `QZI_ENGINE_URL` 通道：health 显示 `h3-serve` 且 `reference_backend=false`，提交任务后引擎返回的 PNG **字节原样**落盘到画廊。Windows 侧由解决方案构建覆盖。

## 路线与发布

见 [PLAN.md](PLAN.md)：阶段 0–11（调研 → C ABI → .NET 封装 → 队列/预设 → 文档/CI → 跨平台 WebUI + 双平台 native 包 → 噪点根因排查与 WebUI 真实引擎通道）已完成；推送 GitHub 并发布 `QwenZImage.NET` 到 NuGet 按用户决定**暂缓**，所需工作流与命令已就绪。

## 许可与声明

MIT，见 [LICENSE](LICENSE)。第三方与模型商标声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目**不包含任何模型权重、不包含 h3-spark.c 源码、不包含 Go 代码**；模型名称与商标归各自权利人所有。
