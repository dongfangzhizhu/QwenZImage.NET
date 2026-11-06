# 接入真实 C / CUDA 引擎

本仓库提供的是**契约 + 托管封装 + 参考实现**，不含真实的扩散推理引擎。要让 `QwenZImage.NET` **库本身**产出真实图像，需要一个实现 `qzi` ABI 的 native 库。下面路径 A / B 讲的就是这件事。

> NuGet 包默认已捆绑 `runtimes/win-x64/native/qzi.dll` 与 `runtimes/linux-x64/native/libqzi.so`（确定性参考后端），因此安装即可跑通。路径 A / B 的产物通过 `QZI_NATIVE_LIBRARY` / `QziClientOptions.NativeLibraryPath` 覆盖默认探测，或直接替换应用输出目录里的同名文件；托管程序集与调用代码无需改动。

## 为什么所有输出都是噪点图（根因）

`native/src/qzi_stub.c` 的 `make_rgb()` 直接往 RGB 缓冲写伪随机字节，`run_phases()` 的 `dit` 循环只是：

```c
for (i = 0; i < steps; i++) {
    qzi_emit_progress(callbacks, "dit", i, steps, start);
    if (qzi_cancelled(callbacks)) { ... }
}
```

**没有 DiT 前向、没有去噪、没有 VAE、没有文本编码**。`steps`、`guidance`、`seed` 只影响进度循环长度和哈希 tag，不影响像素语义。原 Go WebUI 出图正常，是因为它把请求转发给了 `h3 serve`——真正的 C/CUDA 引擎加上 `h3-models/` 里数十 GB 权重；那部分代码与权重从未进入本仓库。因此「和原版用同样的 steps 却全是噪点」不是参数问题，而是后端根本不是图像模型。

**只想马上看到真图**：不必改库。给 WebUI 设置 `QZI_ENGINE_URL=http://127.0.0.1:8411`（或 `--engine`）即可连本机 `h3 serve`，用现成引擎和权重出图。该适配器 `H3ServeExecutor` 只存在于手动测试控制台、不进 NuGet 包、不含 Go 代码；详见 [WEBUI.md](WEBUI.md)。下面 A / B 才是让 `QziClient` 直接出真图的长期方案。

## 路径 A：引擎自己导出 `qzi_*`（完全替换）

引擎（例如从 `h3-spark.c` 的 `h3 serve` 编排逻辑中抽出）直接实现并导出 `native/include/qzi.h` 中的全部函数：

```
qzi_abi_version / qzi_status_text / qzi_abi_layout
qzi_engine_create / qzi_engine_destroy / qzi_engine_health / qzi_engine_unload
qzi_generate_qwen21 / qzi_generate_zimage / qzi_image_release
```

这种情况下 `.NET` 侧**一行都不用改**：把 `QziClientOptions.NativeLibraryPath` 或 `QZI_NATIVE_LIBRARY` 指向该库即可。必须自己实现 [NATIVE_ABI.md](NATIVE_ABI.md) 里的单飞锁、校验、进度/取消、错误缓冲与缓冲区所有权。

好处是完全掌控；代价是要重写 `qzi.c` 的控制平面。

## 路径 B（推荐）：复用 `qzi.c` 控制层，只实现 `qzi_backend` vtable

`native/src/qzi.c` 已经实现了：ABI 版本/布局、参数校验、单飞 `busy` 锁、`served/rejected` 计数、progress/cancel 透传、错误格式化、image 释放。真实引擎只需填写 `native/include/qzi_backend.h` 的 9 个回调，然后：

```c
qzi_register_backend(&my_backend);   /* 必须在 qzi_engine_create 之前 */
```

vtable 会被**复制**，所以栈上的临时对象也可以。注册时机可以是引擎加载时的构造函数（`__attribute__((constructor))` / `DllMain` 之外的显式初始化），或者宿主在 `QziClient` 构造前调用一个导出的 `qzi_register_*` 函数。

`integration/h3_backend_template.c` 就是这个路径的可编译模板：它保留 `h3_state`、留出 5 个 `TODO` 区域，并在真实调用接好之前把 `generate_*` 透明委派给内置参考后端，因此可以先验证 ABI 再逐步替换。

### 9 个回调的职责

| 回调 | 必须做的事 |
| --- | --- |
| `init(void **out_context, config, error, cap)` | 分配 per-engine state，拷贝/解析 `qwen21_dir`、`zimage_dir`、`cache_dir`，检查设备；**不要在这里 mmap 多 GB 权重**，留到首次 generate 懒加载，这样路径错误会在第一次真实请求时暴露 |
| `shutdown(context)` | 释放 state、权重、CUDA 流、文本缓存 |
| `generate_qwen21` | 解析 tier（turbo/base）、LoRA、steps/guidance/seed；文本编码 → 去噪 → VAE → PNG；写满 `qzi_image` 的 5 个字段 |
| `generate_zimage` | 解析 checkpoint（bf16 / nunchaku / pack id）；固定 8 步 Turbo 调度、无 CFG 分支；同样输出 PNG |
| `health` | 填 `resident_model`、`device_id`、`free/total_vram_mib`；把 `reference_backend` 置 0。`busy` / `queue_depth` / `served` / `rejected` / `uptime_seconds` 由 `qzi.c` 填 |
| `unload` | 只释放**可重建**状态（文本塔、缓存）；**不要卸载 DiT**（重新加载要几分钟）。busy 时 `qzi.c` 会拒绝调用 |
| `release_image` | 释放 `generate_*` 分配的 `image->data`；`qzi.c` 随后把结构体清零 |
| `generate_*` 的进度 | `callbacks->progress` 非 NULL 时上报 `"model load"` → `"text"` → `"dit"`（每步一次）→ `"vae"` → `"encode"` |
| `generate_*` 的取消 | 在阶段之间与每个去噪步之间轮询 `callbacks->should_cancel`，非零则尽快返回 `QZI_ERR_CANCELLED` |

校验和单飞由 `qzi.c` 负责，backend 不需要重复实现（可以防御性地再做一次）。

## h3-spark.c 里可以复用的编排参考

调研得到的现有 C 引擎内部接口（**本仓库没有拷贝这些源码**，仅作为接入时的定位线索）：

| 位置 | 内容 |
| --- | --- |
| `h3_serve.c: serve_execute_qwen21()` / `: serve_execute_zimage()` | `/generate` 的完整编排：模型选择、参数解析、缓存、PNG 输出 |
| `h3_qwen21_pipeline.h: h3_qwen21_pipeline_run_ex(...)` | Qwen-Image 2.1 去噪 + VAE，带 `h3_qwen21_progress` 回调 |
| `h3_qwen21_pipeline.h: h3_qwen21_sigmas_turbo()` / `_base()` | Turbo 固定 8 步 / Base sigma 调度 |
| `h3_zimage_pipeline.h: h3_zpipe_load()` / `h3_zpipe_encode_text()` / `h3_zpipe_dit_forward()` / `h3_zpipe_denoise()` | Z-Image Turbo 8 步流水线 |
| `h3_zimage_pipeline.h: h3_png_write_rgb8()` | RGB8 → PNG |
| `docs/PROTOCOL.md` / `docs/SERVE.md` | 原 `h3 serve` 的字段与错误码语义（本项目的 [PARAMETERS.md](PARAMETERS.md) 已对齐） |

原 Go WebUI（`webui/`）只是 `h3 serve` 的 HTTP 客户端，本项目**不复用也不依赖**：队列、取消、重试、进度、预设已在 `.NET` 内用 `System.Threading.Channels` 等重写。

## 端到端接入步骤

1. 按路径 B 复制 `integration/h3_backend_template.c`，把 TODO 区域换成真实 pipeline 调用。
2. 让引擎库导出 `qzi_*`（路径 A）或确保链接了本仓库的 `qzi.c` 并调用 `qzi_register_backend()`（路径 B）。
3. 用 `native/tests/qzi_abi_smoke.c` 的思路做一次 C 层自检：ABI 版本、布局、PNG 签名、确定性、`qzi_image_release` 清零、health 计数。
4. 在 `.NET` 侧设置：

   ```csharp
   var options = new QziClientOptions
   {
       NativeLibraryPath = @"C:\engine\qzi.dll",
       Qwen21Directory   = @"D:\models\qwen21",
       ZImageDirectory   = @"D:\models\zimage",
       CacheDirectory    = @"D:\cache\qzi-text",
       DeviceId          = 0
   };
   using QziClient client = new(options);
   Console.WriteLine(client.GetHealth().IsReferenceBackend);  // 应为 False
   ```

5. 跑 `dotnet run --project samples/QwenZImage.NET.Sample`，确认 5 个进度阶段、PNG 输出与 `IsReferenceBackend == false`。

## 常见坑

- **并发**：一个 engine = 一张 GPU + 一个常驻模型。第二次并发生成会拿到 `QZI_ERR_BUSY`；用 `QziGenerationQueue` 串行化，不要自己起多个 `QziClient` 指向同一张卡。
- **缓冲区所有权**：`image->data` 必须在 `release_image` 里释放；`.NET` 侧成功后会立刻 `memcpy` 并调用 `qzi_image_release`，所以引擎不能复用那块内存。
- **字符串生命周期**：所有 `const char *` 只在调用期间有效，引擎不得保存指针（要保存就自己 `strdup`）。
- **结构体布局**：`qzi_abi_layout()` 是硬门槛。改了任何字段都必须升 `QZI_ABI_VERSION`，同时更新 `.NET` 的 `NativeStructs.cs` 与 `AbiGuard`。
- **符号导出**：Windows MinGW 下用 `-Wl,--exclude-all-symbols`，只让 `QZI_API` 的函数进入导出表；库名必须是 `qzi.dll`（不是 `libqzi.dll`），Linux 为 `libqzi.so`。
- **不要静默忽略非法参数**：Turbo 非 8 步、fp4 checkpoint、带路径的 LoRA 都应该明确失败，而不是被丢掉。
