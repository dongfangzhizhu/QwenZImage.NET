# QwenZImage.NET

> **High-performance .NET 10 wrapper over a stable C ABI for Qwen-Image 2.1 and Z-Image text-to-image generation.**
> Pure C at the bottom, source-generated P/Invoke at the top: no Go, no HTTP hop, no JSON marshalling.

一个把 **Qwen-Image 2.1（turbo / base / LoRA）** 与 **Z-Image（bf16 / Nunchaku int4）** 两个文生图能力封装成 `.NET 10` 可调用包的独立项目。底层是 C 语言定义的稳定 ABI（C/CUDA 引擎可实现它），上层是 `.NET 10` 的高性能源生成 P/Invoke 封装。原 Go WebUI 的队列、进度、取消、重试、预设等功能全部用纯 `.NET` 重写，**仓库中不含任何 Go 代码、不依赖 HTTP 服务**。

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
| C ABI v1、`qzi` 共享库、`.NET 10` 封装、队列、预设、31 个 xUnit 测试、native smoke test | ✅ 已完成并通过 |
| 本地 `git` 仓库与首次提交 | ✅ 已完成 |
| 推送 GitHub / 发布 NuGet | ⏸ **按约定暂缓**，命令与 CI 工作流已备好，见 [PLAN.md](PLAN.md) |
| 真实 CUDA 推理引擎 | ❌ **本仓库不含**。需由外部引擎实现 `qzi_backend` 并注册 |
| 模型权重 | ❌ 不含。NuGet 包只有托管程序集，不捆绑多 GB 权重 |

**本仓库自带的 `reference backend` 是确定性参考实现**：它按相同的 ABI、相同的 progress 阶段、相同的缓冲区所有权返回一张由 prompt / seed / 尺寸决定的 PNG，**不做真实扩散推理，不需要 GPU 或权重**。它的用途是让 CI、单测和集成方在无 GPU 环境下验证 ABI 与内存契约。要生成真实图像，请把 `QZI_NATIVE_LIBRARY` 指向实现了同一 ABI 的引擎构建，并按 [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) 注册后端。

## 环境要求

- **.NET 10 SDK**（`global.json` 锁定 `10.0.100`，`rollForward: latestFeature`）
- 构建参考 native 库：**CMake ≥ 3.20 + C11 编译器**（Windows 可用 MinGW / MSVC，Linux 用 GCC/Clang）
- 真实推理（可选）：实现 `qzi_backend` 的 C/CUDA 引擎 + 对应模型权重

## 快速开始

```bash
# 1) 构建参考 native 库（输出到 artifacts/native/<Platform>-<Arch>/）
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure

# 2) 全量构建 + 测试
dotnet build QwenZImage.NET.sln -c Release
dotnet test  QwenZImage.NET.sln -c Release

# 3) 跑示例（默认走参考后端，输出 outputs/qwen21.png、outputs/zimage.png）
dotnet run --project samples/QwenZImage.NET.Sample -c Release
```

Windows 也可以直接双击/执行 `scripts/build-native.ps1`，Linux/macOS 用 `scripts/build-native.sh`。若 native 库不在自动探测路径里：

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

## API 速查

| 类型 | 说明 |
| --- | --- |
| `QziClient` | 拥有的一个 native engine。同步 `Generate*` / `WriteTo`，异步 `Generate*Async` / `WriteToAsync`，`GetHealth()`、`Unload()`、`Dispose()` |
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

- **Qwen21**：宽高为 **32 的倍数**、`32..2048`、面积 ≤ `2048×2048`；`turbo` 强制 `steps=8`、`guidance=1.0`、不支持 `negative_prompt`；`base` 为 `steps 1..50`、`guidance 1..20`；`lora` 必须是**裸文件名**（禁止路径分隔符与 `..`）、`lora_scale 0.05..4`。
- **ZImage**：宽高为 **16 的倍数**、`16..2048`、面积 ≤ `2048×2048`；`checkpoint` 支持 `bf16`、`nunchaku`、`nunchaku-<pack>`；**`fp4` / `svdq-fp4` 不被支持**（引擎只解码 int4）；无 steps/guidance/negative prompt（Turbo 固定 8 步、无 CFG 分支）。
- 两者：prompt 非空且 ≤ **16384 UTF-8 字节**，保留字段 `flags` 必须为 0。

## 性能设计

- `LibraryImport` **源生成 P/Invoke** + 全 blittable POD 结构体，无反射式 marshalling；启动时 `qzi_abi_layout()` 校验 6 个结构体尺寸与 ABI 版本，packing 不匹配立即失败而不是踩坏内存。
- `ArrayPool<byte>` 池化、pinned NUL-terminated UTF-8 字符串（`GCHandle`）、`Span` / `ReadOnlySpan`，热路径**零 string 分配**。
- `static` + `UnmanagedCallersOnly` 反向回调（progress / cancel），无 delegate 分配、无 GC 生命周期问题。
- 成功路径 native PNG buffer → 一次 `memcpy` → 立刻 `qzi_image_release`；`WriteTo` 走 `UnmanagedMemoryStream` **真正零拷贝**。
- 纯同步 native 调用 + `Task.Run` 异步外壳，避免把长时 GPU 任务压在线程池之外的隐式调度上；队列保证单飞。

## native 库解析顺序

1. `QziClientOptions.NativeLibraryPath`（或环境变量 `QZI_NATIVE_LIBRARY`）
2. 托管程序集同目录，含 `runtimes/<rid>/native/` 与 `native/`
3. 从程序集目录向上查找 `artifacts/native/<Platform>-<Arch>/`（本地开发 / CI）
4. 运行时默认探测（`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`）

文件名：Windows `qzi.dll`、Linux `libqzi.so`、macOS `libqzi.dylib`。NuGet 包**不捆绑 native 库**——CUDA 引擎与权重由使用方提供。

## 目录结构

```
native/include/qzi.h           稳定 C ABI（唯一必须遵守的契约）
native/include/qzi_backend.h   可插拔后端 vtable
native/src/qzi.c               ABI 控制层：生命周期、校验、单飞锁、错误、进度
native/src/qzi_stub.c          确定性参考后端
native/src/qzi_png.c           无依赖的最小 PNG 编码器（stored deflate）
native/tests/                  C ABI smoke test（ctest）
src/QwenZImage.NET/            .NET 10 库（NuGet 包体）
tests/QwenZImage.NET.Tests/    31 个 xUnit 测试
samples/                       端到端示例
integration/                   真实引擎后端模板
docs/                          ABI / 参数 / 引擎接入文档
.github/workflows/             CI 与 NuGet 发布工作流
PLAN.md                        计划、决策记录与发布清单
```

## 测试

```bash
ctest --test-dir native/build --output-on-failure   # C ABI 契约
dotnet test QwenZImage.NET.sln -c Release           # 31 个托管测试
```

托管测试包含：参数校验、ABI 布局、PNG 生成与阶段回调、确定性输出、零拷贝写流、health 计数、异步包装、预取消、native 层重复校验、队列串行/取消/重试/事件流。全部针对参考后端，因此**无需 GPU、无需权重、无需网络**。

## 路线与发布

见 [PLAN.md](PLAN.md)：阶段 0–4（调研 → C ABI → .NET 封装 → 队列/预设 → 文档/CI）已完成；阶段 5（推送 GitHub 并发布 `QwenZImage.NET` 到 NuGet）按用户决定**暂缓**，所需工作流与命令已就绪。

## 许可与声明

MIT，见 [LICENSE](LICENSE)。第三方与模型商标声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目**不包含任何模型权重、不包含 h3-spark.c 源码、不包含 Go 代码**；模型名称与商标归各自权利人所有。
