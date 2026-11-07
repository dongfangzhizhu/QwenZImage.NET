# QwenZImage.NET

> **High-performance .NET 10 wrapper over a stable C ABI for Qwen-Image 2.1 and Z-Image text-to-image generation.**
> Pure C/CUDA at the bottom, source-generated P/Invoke at the top: no Go, no sidecar server, no HTTP or JSON hop in the generation path.

一个把 **Qwen-Image 2.1（turbo / base / LoRA）** 与 **Z-Image（bf16 / Nunchaku int4）** 封装成 `.NET 10` 可调用包的独立项目。托管库通过 `LibraryImport` **直接 P/Invoke 到进程内的 `QwenZImageNative` 共享库**，后者内含从 MIT 许可的上游 C/CUDA 引擎复制并改名后的真实推理代码（`engine/`）。`QwenZImage.config.json` 指定模型权重目录。原 Go WebUI 的队列、进度、取消、重试、预设全部用纯 `.NET` 重写，仓库中不含任何 Go 代码。

> ⚠️ **有两种 native 构建，务必先看这一段。**
>
> - **默认 / CI 构建**（`-DQZI_WITH_CUDA_ENGINE=OFF`，即 `scripts/build-native.ps1` / `.sh`）链接 `qzi_stub.c`：这是一个确定性 **reference backend**，输出由 prompt / seed / 尺寸哈希出来的**噪点图**，只用来在无 GPU 环境验证 ABI、进度、单飞与内存契约。它**不是图像模型**。
> - **真实构建**（`-DQZI_WITH_CUDA_ENGINE=ON`，即 `scripts/build-native-cuda.bat` / `.sh`）把 `engine/` 里的真实 C/CUDA 引擎与 `native/src/qzi_engine_backend.c` 适配器链进同一个 `QwenZImageNative` 库，直接做 DiT 去噪 + VAE 解码，**产出真实图像**。
>
> 「和原版用同样的 steps 却全是噪点」的根因是链接了 reference backend，不是参数问题。用 `GET /api/health` 的 `engine.reference_backend` 或 `EngineHealth.IsReferenceBackend` 判断当前加载的是哪一种。

仓库另附一个**跨平台 WebUI 手动测试控制台**（`src/QwenZImage.NET.WebUI`，ASP.NET Core Minimal API，Windows / Linux 同一套代码），复刻原 Go WebUI 的交互面。它同样直接 P/Invoke 进程内 native 库，**没有 HTTP transport、没有外部服务要启动**。详见 [docs/WEBUI.md](docs/WEBUI.md)。

```
┌──────────────────────────────────────────────────────────────┐
│  Your app (.NET 10)                                          │
│    QziClient / QziGenerationQueue / QziPresets               │
│    ArrayPool · pinned UTF-8 · UnmanagedCallersOnly callbacks │
├──────────────────────────────────────────────────────────────┤
│  QwenZImage.NET.dll  (LibraryImport, blittable structs)      │
├──────────────────────────────────────────────────────────────┤
│  QwenZImageNative.dll / libQwenZImageNative.so                │
│    native/include/qzi.h  ← 稳定 C ABI v1（唯一托管/原生契约） │
│    qzi.c 控制层：校验 / 单飞 / 进度 / 取消 / buffer 所有权    │
│    qzi_backend vtable                                        │
│      ├─ qzi_stub.c          参考后端（噪点，CI 用）          │
│      └─ qzi_engine_backend.c → engine/ 真实 C/CUDA 推理引擎   │
│            Qwen21 / ZImage DiT + text tower + VAE + CUDA kernels│
└──────────────────────────────────────────────────────────────┘
                          ▲
              QwenZImage.config.json（模型权重路径）
```

## 当前状态

| 项 | 状态 |
| --- | --- |
| 稳定 C ABI v1、`.NET 10` 源生成 P/Invoke、队列、预设、托管 + native 测试 | ✅ 已完成并通过 |
| 跨平台 WebUI（health / presets / jobs / SSE / gallery / trash / unload） | ✅ 已完成，Windows + Linux |
| **真实 C/CUDA 引擎直接绑定**（`engine/` + `qzi_engine_backend.c`，无 HTTP） | ✅ Windows（MSVC 19.51 + CUDA 13.3）与 Linux（WSL GCC 12 + CUDA 13.0）双平台构建 + `ctest` 通过；默认 fat binary 覆盖 11 个 SM；Linux 产物无版本化 ICU 依赖 |
| `QwenZImage.config.json` 配置模型路径，`.NET` / native 双侧加载 | ✅ 已完成；LoRA / Nunchaku 目录支持配置文件、`QziConfig.With`、代码级 options、纯环境变量四种等价入口 |
| **LoRA 全链路一致性**（ABI / 托管校验 / native / 真实引擎 / 参考后端 / WebUI / 文档） | ✅ 两个 tier 均可用，裸文件名与 scale 统一校验，切换 adapter 或 scale 正确重载 DiT |
| **真图回归矩阵**（`tools/RealImageRegression`，需 GPU + 权重） | ✅ Windows CUDA 10/10 通过：turbo / base、LoRA scale 1.0 与 0.6、512 与 1024、Z-Image bf16 与 Nunchaku r32/r256；重复请求字节级确定性一致 |
| native 产物统一命名 `QwenZImageNative.dll` / `libQwenZImageNative.so` | ✅ 已完成，无 `h3` 命名产物 |
| 单个 NuGet 包含 .NET 程序集 + win-x64 + linux-x64 native | ✅ 已完成，`QziEnsureNativeRuntimes` 缺平台即拒绝打包 |
| 本地 git 仓库与提交 | ✅ 已完成 |
| 推送 GitHub / 发布 NuGet | ⏸ **按约定暂缓**，命令与 CI 工作流已备好，见 [PLAN.md](PLAN.md) |
| 模型权重 | ❌ 不含，需使用方自备并在配置文件里指向它们 |

## 配置文件：`QwenZImage.config.json`

格式与上游 `h3.config.json` 类似：top-level 字段对所有平台生效，`windows` / `linux` / `osx` 对象按当前平台覆盖；值可用 `~` 开头；`_` 开头的键视为注释；未知键在 .NET 侧报错。

```json
{
  "device_id": -1,
  "thread_count": 0,
  "qwen21_dir": "/models/qwen21",
  "qwen21_lora_dir": "/models/qwen21/loras",
  "zimage_dir": "/models/zimage",
  "zimage_nunchaku_dir": "/models/zimage_nunchaku",
  "text_cache_dir": "/models/cache",
  "windows": {
    "qwen21_dir": "D:/models/qwen21",
    "zimage_dir": "D:/models/zimage"
  }
}
```

查找顺序：显式 `QZI_CONFIG` / `--config <path>` → 程序目录 → 当前工作目录 → 从这两个目录**向上最多 8 级**（`dotnet run` 会把工作目录设为项目目录，解决方案根目录的配置因此也能找到）→ `~/.config/QwenZImage/config.json`。`QZI_*` 环境变量（`QZI_QWEN21_DIR`、`QZI_ZIMAGE_DIR`、`QZI_QWEN21_LORA_DIR`、`QZI_ZIMAGE_NUNCHAKU_DIR`、`QZI_MODEL_CACHE`、`QZI_DEVICE_ID` 等）优先于文件；代码中显式传入的 `QziClientOptions` 优先于二者。`QziClient` 会把解析出的路径发布为 `QZI_CONFIG`，并把 `Qwen21LoraDirectory` / `ZImageNunchakuDirectory` 发布成 `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR`；native 侧的 `qzen_config_apply_environment()` 在 `qzen_config_load()` 之后应用同一组覆盖。因此**配置文件、`QziConfig.With(...)`、代码级 options、纯环境变量**这四种方式都能作用到引擎——ABI 里装不下的字段不再静默丢失。

仓库根目录的 `QwenZImage.config.json` 是开发机示例，路径指向本机已有的权重目录；部署时改成自己的路径即可。权重文件本身不在仓库里。

## 环境要求

- **.NET 10 SDK**（`global.json` 锁定 `10.0.100`）
- 参考 native 构建：**CMake ≥ 3.20 + C11 编译器 + Ninja**（Windows MinGW/MSVC，Linux GCC/Clang）
- 真实 native 构建：上述工具 **+ NVIDIA CUDA Toolkit 13+（nvcc / cudart / cuBLAS / cuBLASLt）+ MSVC（Windows）或 GCC/Clang + OpenMP（Linux）**。Linux **不需要 ICU**：`engine/platform/posix/qzen_posix_unicode.c` 在运行时 `dlopen` 宿主系统的 ICU（找不到时用内置 ASCII/CJK 回退），产物因此不带版本化的 `libicuuc.so.NN` 依赖，可直接跨发行版分发。
- 默认 `QZEN_CUDA_ARCH` 是覆盖 Ampere → Blackwell 的 **fat binary**（`80;86;87;88;89;90;100;103;110;120;121`；不含 sm_75，因为 attention kernel 用 `mma.sync.m16n8k16`）：一个同名库适配常见显卡，无需按 GPU 重编译。CMake 会查询 `nvcc --list-gpu-arch` 自动剔除当前工具链不支持的架构；本地开发传单个 SM（如 `89`）可显著加快编译。
- 打包前需要 Windows 与 Linux 两个平台的 native 产物（Windows 主机可用 WSL 一次构建两者）
- 真实推理还需要**自备模型权重**并在配置文件里指向它们

## 快速开始

```bash
# 1) 参考 native（无 GPU，输出确定性噪点，用于 CI/测试）
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure

# 1b) 真实 C/CUDA native（需要 nvcc；产物单独放 artifacts/native/<Platform>-<Arch>-cuda/）
scripts/build-native-cuda.bat                   # Windows：默认 fat binary，自动加载 MSVC 环境
bash scripts/build-native-cuda.sh               # Linux：默认 fat binary
# 本地只编一个 SM 更快：scripts\build-native-cuda.bat 89 / bash scripts/build-native-cuda.sh 89
# 或手工：cmake -S native -B native/build-cuda -G Ninja \
#           -DQZI_WITH_CUDA_ENGINE=ON -DQZEN_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release

# 2) 全量构建 + 测试
dotnet build QwenZImage.NET.sln -c Release
dotnet test  QwenZImage.NET.sln -c Release

# 3) 跑示例
dotnet run --project samples/QwenZImage.NET.Sample -c Release

# 4) 手动测试 WebUI：浏览器打开 http://127.0.0.1:7860/
dotnet run --project src/QwenZImage.NET.WebUI -c Release
```

想让库 / WebUI 用真实后端，把 native 库指向真实构建即可（**托管程序集与 API 完全不用改**）：

```bash
export QZI_NATIVE_LIBRARY=/path/to/artifacts/native/Linux-x86_64-cuda/libQwenZImageNative.so
export QZI_CONFIG=/path/to/QwenZImage.config.json
# Windows: $env:QZI_NATIVE_LIBRARY = 'D:\...\Windows-AMD64-cuda\QwenZImageNative.dll'
```

一键脚本（会自动 stage 成 NuGet RID 布局）：

```bash
powershell -File scripts/build-native-all.ps1   # Windows + WSL Linux，参考后端
powershell -File scripts/build-native.ps1       # 仅 Windows，参考后端
bash scripts/build-native.sh                    # 仅当前 Unix 平台，参考后端
scripts/build-native-cuda.bat                # Windows 真实后端（默认 fat binary；传 89 只编单 SM）
bash scripts/build-native-cuda.sh            # Linux 真实后端（同上）
```

> 参考构建与 CUDA 构建输出到不同的 `artifacts/native` 子目录，避免互相覆盖；`QZI_STAGE_CUDA=1` 才把 CUDA 产物 stage 进 `artifacts/runtimes/<rid>/native/`。

## 在 .NET 项目里使用

发布后：`dotnet add package QwenZImage.NET`。本地开发可直接 `dotnet add reference src/QwenZImage.NET/QwenZImage.NET.csproj`。

```csharp
using QwenZImage.NET;

// 不传 options 时自动读取 QwenZImage.config.json
using QziClient client = new();
// 或显式指定：new QziClient(QziClientOptions.FromConfig("/path/to/QwenZImage.config.json"));
// 或完全手工：new QziClient(new QziClientOptions { Qwen21Directory = "/models/qwen21", ... });

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

Console.WriteLine(client.GetHealth().IsReferenceBackend
    ? "reference backend (noise)" : "real C/CUDA backend");
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

`src/QwenZImage.NET.WebUI` 是纯 `.NET 10` / ASP.NET Core Minimal API 的本地控制台，Windows 与 Linux 同一套代码、无 Node/npm、无 Go 进程。功能面：health、presets、串行任务队列、取消、重试、SSE 实时进度、输出画廊、删除到 `.trash` / 恢复、卸载模型。

```bash
dotnet run --project src/QwenZImage.NET.WebUI -c Release
QZI_WEBUI_URLS=http://127.0.0.1:51234 dotnet run --project src/QwenZImage.NET.WebUI -c Release
dotnet run --project src/QwenZImage.NET.WebUI -c Release -- --config /path/to/QwenZImage.config.json
```

主要端点：`GET /api/health`、`GET /api/presets`、`POST|GET /api/jobs`、`GET /api/jobs/{id}`、`POST /api/jobs/{id}/cancel|/retry`、`GET /api/events`（SSE）、`GET|DELETE /api/outputs`、`POST /api/outputs/restore`、`GET|POST /api/unload`。可用环境变量：`QZI_CONFIG`、`QZI_WEBUI_URLS`、`QZI_WEBUI_OUTPUT`、`QZI_MODEL_DIR`、`QZI_MODEL_CACHE`、`QZI_NATIVE_LIBRARY`、`QZI_DEVICE_ID`。完整说明见 [docs/WEBUI.md](docs/WEBUI.md)。

> WebUI 是**手动测试工具**，不打进 NuGet 包。`/api/health` 的 `transport` 恒为 `native-pinvoke`，并返回 `config_path` 与 `engine.reference_backend`。当加载到参考后端时页面显示黄色噪点告警横幅；换成 CUDA 构建后横幅消失。

## API 速查

| 类型 | 说明 |
| --- | --- |
| `QziClient` | 拥有的一个 native engine。同步 `Generate*` / `WriteTo`，异步 `Generate*Async` / `WriteToAsync`，`GetHealth()`、`Unload()`、`Dispose()` |
| `QziClientOptions` | 模型目录、缓存、设备、native 路径、`ConfigPath`、`Qwen21LoraDirectory` / `ZImageNunchakuDirectory`；`FromConfig(path?)` 从配置文件构造 |
| `QziConfig` | 配置文件加载器（平台覆盖、`~` 展开、未知键报错、搜索路径） |
| `QwenImageOptions` / `ZImageOptions` | 强类型请求（继承 `GenerationOptions`），`required Prompt`、宽高、Seed |
| `GenerationResult` | PNG 结果：`Png` / `PngSpan`（零拷贝）、`CopyTo(Async)`、`SaveAsync`、`Dispose()` 归还池化缓冲区 |
| `GenerationProgress` | `Phase`（`ModelLoad / Text / Dit / Vae / Encode`）、`Completed`、`Total`、`Elapsed` |
| `EngineHealth` | 常驻模型、Busy、队列深度、显存、Served/Rejected、运行时长、`IsReferenceBackend` |
| `QziGenerationQueue` | 串行队列、取消、重试、事件流、输出 sink |
| `QziPresets` | 5 个 Qwen21 + 4 个 ZImage 预设模板，用 `WithPrompt()` 变成可执行请求 |
| `RequestValidator` | 托管侧参数校验（与 native 侧规则一致，先于分钟级模型加载失败） |
| `QziException` | native 错误映射，含 `Status` 与 `NativeMessage` |

## 参数与校验

规则在 `.NET` 与 native 两层重复执行，避免把明显非法的请求送进分钟级模型加载。完整表格见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。要点：

- **Qwen21**：宽高为 **32 的倍数**、`32..2048`、面积 ≤ `2048×2048`；`turbo` 强制 `steps=8`、`guidance=1.0`、不支持 `negative_prompt`；`base` 为 `steps 1..50`、`guidance 1..20`；`lora` 必须是**裸文件名**（≤ 255 UTF-8 字节，禁止 `/`、`\`、`:`、`..` 与控制字符），`lora_scale` 必须有限且在 `0.05..4`。**LoRA 在 turbo / base 两个 tier 都可用**，切换 adapter 或 scale 会重载 DiT。
- **ZImage**：宽高为 **16 的倍数**、`16..2048`、面积 ≤ `2048×2048`；`checkpoint` 支持 `bf16`、`nunchaku`、`nunchaku-<pack>`；**`fp4` / `svdq-fp4`（大小写不敏感）不被支持**（引擎只解码 int4）；无 steps/guidance/negative prompt（Turbo 固定 8 步、无 CFG 分支）。
- **默认值**：native ABI 用 `steps=0` / `guidance=0` / `lora_scale=0` 表示「用 tier 默认值 / 无 adapter」；`qzi_generate_qwen21()` 在 backend 之前统一归一化，两个后端看到同一组有效值，非法值一律拒绝而不静默改写。详见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。
- 两者：prompt 非空且 ≤ **16384 UTF-8 字节**，保留字段 `flags` 必须为 0。

## 性能设计

- `LibraryImport` **源生成 P/Invoke** + 全 blittable POD 结构体，无反射式 marshalling；启动时 `qzi_abi_layout()` 校验 6 个结构体尺寸与 ABI 版本，packing 不匹配立即失败而不是踩坏内存。
- `ArrayPool<byte>` 池化、pinned NUL-terminated UTF-8 字符串（`GCHandle`）、`Span` / `ReadOnlySpan`，热路径**零 string 分配**。
- `static` + `UnmanagedCallersOnly` 反向回调（progress / cancel），无 delegate 分配、无 GC 生命周期问题。
- 成功路径 native PNG buffer → 一次 `memcpy` → 立刻 `qzi_image_release`；`WriteTo` 走 `UnmanagedMemoryStream` **真正零拷贝**。
- 纯同步 native 调用 + `Task.Run` 异步外壳；队列保证单飞，DiT 常驻避免分钟级重复加载。

## NuGet 包内容（.NET + Windows + Linux native）

```
QwenZImage.NET.0.1.0.nupkg
├─ lib/net10.0/QwenZImage.NET.dll
├─ lib/net10.0/QwenZImage.NET.xml
├─ runtimes/win-x64/native/QwenZImageNative.dll
├─ runtimes/linux-x64/native/libQwenZImageNative.so
├─ README.md / LICENSE / THIRD_PARTY_NOTICES.md / engine/LICENSE
└─ docs/*.md
```

打包前置条件：两个平台的 native 库必须 stage 到 `artifacts/runtimes/<rid>/native/`。`QwenZImage.NET.csproj` 的 `QziEnsureNativeRuntimes` target 会在 `GenerateNuspec` 前检查两者存在，缺任一平台直接报错。

```bash
dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj -c Release -o artifacts/packages
```

> 默认打进包里的仍是**参考后端**（无 GPU 依赖、开箱即用、输出噪点），因为 CI 没有 GPU。真实 CUDA 构建现在默认产出覆盖 Ampere → Blackwell 的 **fat binary**，一个同名库即可直接替换 `runtimes/<rid>/native/`，或用 `QZI_NATIVE_LIBRARY` 指向 `artifacts/native/<Platform>-<Arch>-cuda/` 里的产物，不必按显卡重编译；单 SM 构建仍可用脚本参数指定。托管 API 不变，权重始终由使用方自带。

## native 库解析顺序

1. `QziClientOptions.NativeLibraryPath`（或 `QZI_NATIVE_LIBRARY`）— 最高优先级
2. 运行时提供的 `NATIVE_DLL_SEARCH_DIRECTORIES`（NuGet `runtimes/<rid>/native/` 即通过它生效）
3. 托管程序集同目录，含 `runtimes/<rid>/native/` 与 `native/`
4. 从程序集目录向上查找 `artifacts/runtimes/<rid>/native/` 与 `artifacts/native/<Platform>-<Arch>/`
5. 运行时默认探测（`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`）

文件名：Windows `QwenZImageNative.dll`、Linux `libQwenZImageNative.so`、macOS `libQwenZImageNative.dylib`。当前提供 **win-x64 与 linux-x64**。

## 目录结构

```
QwenZImage.config.json          模型权重路径配置（平台覆盖 / 环境变量覆盖）
native/include/qzi.h            稳定 C ABI（唯一必须遵守的契约）
native/include/qzi_backend.h    可插拔后端 vtable
native/src/qzi.c                ABI 控制层：生命周期、校验、单飞锁、错误、进度
native/src/qzi_engine_backend.c 真实引擎适配器（in-process qzi_backend 实现）
native/src/qzi_stub.c           确定性参考后端（噪点，CI 用）
native/src/qzi_png.c            无依赖最小 PNG 编码器
native/tests/                   C ABI smoke test（按后端自动分支）
engine/                         vendored + 改名的真实 C/CUDA 推理引擎
  src/, platform/win32/, platform/posix/, third_party/, cmake/, CMakeLists.txt
  LICENSE / UPSTREAM.md         上游 MIT 许可与来源说明
src/QwenZImage.NET/             .NET 10 库（NuGet 包体）
  Configuration/QziConfig.cs    配置文件加载器
src/QwenZImage.NET.WebUI/       跨平台手动测试 WebUI（不打包）
tests/                          xUnit 托管测试
samples/                        端到端示例
integration/                    自定义 qzi_backend 模板
tools/RealImageRegression/      真图回归 harness（需 GPU + 权重，刻意不进 sln / CI）
scripts/                        native 构建、RID staging 与真图回归运行入口
docs/                           ABI / 参数 / 引擎接入 / WebUI 文档
.github/workflows/              CI 与 NuGet 发布工作流
PLAN.md                         计划、决策记录与发布清单
```

## 测试

```bash
ctest --test-dir native/build --output-on-failure       # 参考后端完整契约（Windows 1 项 / Linux 3 项）
ctest --test-dir native/build-cuda --output-on-failure  # 真实后端 ABI + 校验（无需权重）
dotnet test QwenZImage.NET.sln -c Release               # 托管测试
```

Linux 上 `ctest` 另有 `posix_unicode_with_icu` / `posix_unicode_without_icu` 两项，验证 POSIX ICU 兼容层的两条运行时路径：前者 `dlopen` 命中宿主 ICU 并把 `e + U+0301` 组合成 `é`，后者用 `-Wl,--wrap=dlopen` 强制「无 ICU」走内置回退（恒等 NFC）。二者共同覆盖字符分类、空白判断、含 4 字节代理对的 UTF-8 ↔ UTF-16 round-trip 与非法 UTF-8 报错。因为 `qzen_posix_unicode.c` 用 `pthread_once` 只解析一次 ICU，两条路径由两个独立可执行分担；链接器不支持 `--wrap` 时 configure 会优雅跳过该变体，宿主无 ICU 时对应测试记为 SKIP。这两项**不需要 CUDA**，GPU-less 的 Linux CI 在 reference 构建里就会跑。

托管测试共 **61 项**（54 库 + 7 WebUI）：配置文件加载与平台覆盖、`QziConfig` / 环境变量 / 代码级 options 覆盖链、参数校验（含 LoRA 裸文件名、长度、finite scale、双 tier）、ABI 布局、PNG 生成与阶段回调、确定性输出、零拷贝写流、health 计数、异步包装、预取消、native 层重复校验、队列串行/取消/重试/事件流、WebUI 运行时配置与双 tier LoRA 映射。全部不需要 GPU、权重或网络监听。

CI 另做 WebUI Linux 运行时冒烟：启动 Kestrel → 验证 `/api/health` 的 `ok=true` / `transport=native-pinvoke` / `config_path` → 检查 presets → 提交 zimage job → 轮询到完成 → 确认输出出现在 `/api/outputs`。

**真图回归**（不属于 CI，需要 CUDA 构建 + 多 GB 权重）用 `tools/RealImageRegression`：

```bash
.\scripts\run-real-image-regression.ps1 --list
.\scripts\run-real-image-regression.ps1 --out artifacts\regress\full
bash scripts/run-real-image-regression.sh --out artifacts/regress/full
```

`cases.json` 覆盖 qwen21 turbo/base、LoRA 及 scale 变化、1024×1024，以及 Z-Image bf16 与 Nunchaku r32/r256 checkpoint；每个用例输出 PNG、SHA-256、phase 序列与 `report.md` / `report.json`，并用 `compare_with` 做字节级确定性对照。工具检测到参考后端会直接拒绝运行。详见 [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) 的「真图回归矩阵」。

## 路线与发布

见 [PLAN.md](PLAN.md)：阶段 0–12 已完成（调研 → C ABI → .NET 封装 → 队列/预设 → 文档/CI → 跨平台 WebUI + 双平台 native 包 → 噪点根因排查 → **D10 直接绑定真实 native 后端**），阶段 12b（双平台 fat binary / 无 ICU 的 Linux 运行时）、阶段 12c（**LoRA / 参数全链路一致性**）与阶段 12d（**真图回归矩阵 10/10**，覆盖 LoRA、base、1024×1024、Z-Image bf16 与 Nunchaku r32/r256）与阶段 12e（**POSIX ICU 双路径检查接入 CTest**）亦已完成；按用户要求不做 macOS 适配；推送 GitHub 并发布 NuGet 按用户决定**暂缓**。

## 许可与声明

MIT，见 [LICENSE](LICENSE)。`engine/` 下的真实推理引擎来自 MIT 许可的上游项目，许可与来源见 [engine/UPSTREAM.md](engine/UPSTREAM.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目**不包含任何模型权重、不包含 Go 代码、不包含上游 HTTP 服务/CLI**；模型名称与商标归各自权利人所有。
