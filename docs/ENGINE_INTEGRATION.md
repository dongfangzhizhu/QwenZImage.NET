# 真实 C/CUDA 后端：`QwenZImageNative`

本仓库**已经包含真实推理引擎**。托管库 `QwenZImage.NET` 通过 `LibraryImport` 直接 P/Invoke 进程内的 `QwenZImageNative.dll` / `libQwenZImageNative.so`，由该库完成文本编码 → DiT 去噪 → VAE 解码 → PNG 编码。**没有 HTTP 服务、没有 JSON hop、没有 Go 进程、没有 sidecar**。

引擎源码位于 [`engine/`](../engine/)：从 MIT 许可的上游 C/CUDA 项目复制了 Qwen-Image 2.1 与 Z-Image 两条链路所需的最小源码集（64 个文件），把 `h3_*` / `H3_*` 符号与文件名统一改成 `qzen_*` / `QZEN_*`，并重新实现了编排层。来源、改名规则与许可证见 [`engine/UPSTREAM.md`](../engine/UPSTREAM.md) 和 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。HTTP service、CLI、视频/WAN 管线与 Go WebUI **均未复制**。

## 两种 native 构建

同一个 CMake 工程按 `QZI_WITH_CUDA_ENGINE` 产出两个**同名但不同内容**的库，输出到不同目录，互不覆盖：

| 构建 | 开关 | 链接的 backend | 产物目录 | 输出 |
| --- | --- | --- | --- | --- |
| reference（默认 / CI） | `OFF` | `native/src/qzi_stub.c` | `artifacts/native/<Platform>-<Arch>/` | 确定性**噪点图**，无需 GPU / 权重 |
| real（真图） | `ON` | `native/src/qzi_engine_backend.c` + `engine/` 的 `qzi_engine_core` | `artifacts/native/<Platform>-<Arch>-cuda/` | **真实图像**，需要 CUDA + 权重 |

```
App (.NET 10) ── LibraryImport ──▶ QwenZImageNative
                                     │
  native/include/qzi.h  ← 唯一托管/原生契约 (C ABI v1)
  native/src/qzi.c      ← 校验 / 单飞 / 进度 / 取消 / 错误 / buffer 所有权
                                     │  qzi_backend vtable
              ┌──────────────────────┴──────────────────────┐
              │ QZI_HAS_REAL_BACKEND 未定义 → qzi_stub.c      │
              │ QZI_HAS_REAL_BACKEND 已定义 →                 │
              │    qzi_engine_backend.c → engine/ C/CUDA      │
              └───────────────────────────────────────────────┘
                                     ▲
                QwenZImage.config.json（模型权重路径）
```

`qzi.c` 的 `ensure_backend()` 在**编译期**选择默认 vtable：定义 `QZI_HAS_REAL_BACKEND` 时调用 `qzi_real_backend()`（由 `qzi_engine_backend.c` 导出），否则使用 `qzi_reference_vtable`。这样做是为了避开 Windows 上 DLL 构造函数的初始化顺序问题，也让 reference 构建完全不含引擎符号、可在无 GPU 机器上运行。运行时仍可用 `qzi_register_backend(&vtable)` 覆盖，传 `NULL` 恢复编译期默认值。

## 为什么之前所有输出都是噪点图（历史根因）

`qzi_stub.c` 的 `make_rgb()` 往 RGB 缓冲写伪随机字节，`run_phases()` 的 `dit` 循环只上报进度：

```c
for (i = 0; i < steps; i++) {
    qzi_emit_progress(callbacks, "dit", i, steps, start);
    if (qzi_cancelled(callbacks)) { ... }
}
```

**没有 DiT 前向、没有去噪、没有 VAE、没有文本编码**；`steps`、`guidance`、`seed` 只影响进度循环长度和哈希 tag。所以「和原版用同样的 steps 却全是噪点」不是参数问题，而是当时链接的 native 库根本不是图像模型。原 Go WebUI 出图正常，是因为它把请求转发给了上游 C/CUDA 服务，而那条链路当时尚未进入本仓库。

D10 之后已不再用「WebUI 连外部 HTTP 服务」这种方式绕过：真实引擎直接进 `engine/`，由库本身 P/Invoke 调用。判断当前加载的是哪一种，用 `GET /api/health` 的 `engine.reference_backend`，或 `EngineHealth.IsReferenceBackend`。

## 配置：`QwenZImage.config.json`

格式仿照上游 `h3.config.json`：top-level 字段对所有平台生效，`windows` / `linux` / `osx` 对象按当前平台覆盖；值可用 `~` 开头；`_` 开头的键作为注释；未知键在 `.NET` 侧报错。

```json
{
  "device_id": -1,
  "thread_count": 0,
  "qwen21_dir": "/models/qwen21",
  "qwen21_lora_dir": "/models/qwen21/loras",
  "zimage_dir": "/models/zimage",
  "zimage_nunchaku_dir": "/models/zimage_nunchaku",
  "text_cache_dir": "/models/cache"
}
```

| 键 | 用途 |
| --- | --- |
| `qwen21_dir` | Qwen-Image 2.1 权重包（turbo / base DiT、VAE、text tower、tokenizer） |
| `qwen21_lora_dir` | base tier 的 LoRA 目录；`lora` 参数必须是裸文件名，会在此目录下解析 |
| `zimage_dir` | Z-Image bf16 权重包 |
| `zimage_nunchaku_dir` | Nunchaku / SVDQ int4 checkpoint 目录，`nunchaku-r<rank>` 在此枚举 |
| `text_cache_dir` | 文本编码缓存 |
| `device_id` / `thread_count` | `-1` / `0` = 自动 |
| `native_library` | 可选，显式 native 库路径 |

两侧都要读这个文件，原因是 **C ABI 装不下全部字段**：

- `.NET` 侧 `QziConfig.Load()` → `QziClientOptions.FromConfig()`：完整解析、平台覆盖、`~` 展开、严格校验未知键；`qwen21_dir` / `zimage_dir` / `text_cache_dir` / `device_id` / `thread_count` 通过 `qzi_engine_config` 传给 native。
- native 侧 `engine/src/qzen_config.c` 的 `qzen_config_load(NULL, ...)`：在 `qzen_init()` 里读取，**补齐** `qwen21_lora_dir` / `zimage_nunchaku_dir` 等 ABI 未携带的字段。`QziClient` 会把最终采用的配置文件路径发布为 `QZI_CONFIG`，保证两侧读的是同一份文件。

查找顺序：显式 `QZI_CONFIG` / `--config <path>` → 程序目录 → 当前工作目录 → 从这两个目录向上最多 8 级（覆盖 `dotnet run` 从项目目录启动的情况）→ `~/.config/QwenZImage/config.json`。`QZI_*` 环境变量覆盖文件；代码里显式传入的 `QziClientOptions` 优先于二者。**权重文件本身不含在仓库 / NuGet 包里**。

## `qzi_engine_backend.c` 做了什么

它实现 `native/include/qzi_backend.h` 的 9 个回调，把上游 resident service 的 `serve_execute_qwen21()` / `serve_execute_zimage()` 编排逻辑移植成**进程内调用**（去掉 HTTP 状态）：

- `qzen_init()`：`qzen_config_load()` 读配置，合并 ABI 传入的目录；`device_id >= 0` 时设置 `CUDA_VISIBLE_DEVICES`；`enable_gpu_switches()` 打开 `QWEN21_GPU` / `QWEN21_VAE_GPU` / `QWEN21_TEXT_GPU` / `ZIMAGE_GPU` / `ZIMAGE_VAE_GPU`（已由用户设置则不覆盖）。**不在这里 mmap 多 GB 权重**，DiT 首次 generate 时懒加载并常驻。
- Qwen-Image 2.1：`qzen_qwen21_open` → `qzen_qwen21_vae_open` → `qzen_tokenizer_load` → `qzen_qwen21_text_encode` / `qzen_qwen21_text_forward` → `qzen_qwen21_sigmas_turbo` 或 `qzen_qwen21_sigmas_base` → `qzen_qwen21_pipeline_run_ex` → `qzi_png_encode_rgb8`。
- Z-Image：`qzen_zpipe_load` → `qzen_zpipe_attach_text` → `qzen_zpipe_encode_text` → `qzen_zpipe_init_latents` → `qzen_zpipe_denoise` → `qzen_zvae_load` / `qzen_zvae_decode` → `qzi_png_encode_rgb8`。
- 噪点采样用与上游一致的 splitmix64 + Box-Muller，保证同一 seed 复现同样的图像。
- PNG 直接编码进 `qzi_image.data` 内存缓冲（`qzi_png_encode_rgb8`），**不落临时文件**；`release_image` 用 `free()` 释放。
- progress bridge 把引擎回调转成 `qzi_callbacks` 的 `"model load"` / `"text"` / `"dit"` / `"vae"` / `"encode"`；Z-Image 的 `qzen_zpipe_denoise()` 回调没有 `opaque`，因此用单飞前提下的静态 `g_zimage_bridge`（ABI 层已串行化，不会并发）。
- `unload` 只释放文本塔与缓存，**不卸载 DiT**（重载要几分钟）；`health` 把 `reference_backend` 置 0，其余计数由 `qzi.c` 填。

参数校验、单飞、取消、错误缓冲与 image 所有权仍全部由 `qzi.c` 负责，backend 不重复实现。

## 构建与验证真实后端

```bash
# Windows（脚本自动加载 vcvars64，参数为 CUDA SM 版本）
scripts\build-native-cuda.bat 89

# Linux
bash scripts/build-native-cuda.sh 89

# 或手工
cmake -S native -B native/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DQZI_WITH_CUDA_ENGINE=ON -DQZEN_CUDA_ARCH=89 -DQZI_BUILD_TESTS=ON
cmake --build native/build-cuda
ctest --test-dir native/build-cuda --output-on-failure
```

`QZEN_CUDA_ARCH` 必须与显卡匹配（RTX 4090 = `89`，GB10 = `121`）；它会成为 `QZEN_SM_ARCH` 编译定义，架构不匹配会静默走低效路径或加载失败。Windows 需要 MSVC + CUDA Toolkit（nvcc / cudart / cuBLAS / cuBLASLt），Linux 还需要 OpenMP 与 ICU。

`build-native-cuda.bat` 会：如果当前 shell 已经有完整的 `cl` / `rc` / `nvcc` 环境就直接复用（不调用 `vcvars64.bat`）；否则自动 `call vcvars64.bat`；若注册表查询不可用导致 Windows SDK 的 `rc.exe` / `mt.exe` 没进 PATH，则**从文件系统**枚举 `%ProgramFiles(x86)%\Windows Kits\10\Include` 补齐 SDK 的 `PATH` / `INCLUDE` / `LIB`，避免 CMake 的 MSVC 编译检查失败。脚本最后用 `QZI_STAGE_CUDA=1` 把产物 stage 进 `artifacts/runtimes/win-x64/native/`。

`ctest` 的 `qzi_abi_smoke` 会按 `health.reference_backend` 分支：reference 构建跑完整生成契约（PNG 签名、确定性、release），真实构建只跑 ABI 布局与参数校验，**不加载多 GB 权重**。

把应用指向真实产物即可，托管程序集与 API 一行都不用改：

```bash
export QZI_NATIVE_LIBRARY=/path/to/artifacts/native/Linux-x86_64-cuda/libQwenZImageNative.so
export QZI_CONFIG=/path/to/QwenZImage.config.json
# Windows PowerShell:
# $env:QZI_NATIVE_LIBRARY = 'D:\...\Windows-AMD64-cuda\QwenZImageNative.dll'
```

`QZI_STAGE_CUDA=1 scripts/stage-native.sh`（或 `.ps1`）才会把 CUDA 产物 stage 进 `artifacts/runtimes/<rid>/native/`，避免覆盖 CI / 测试用的 reference DLL。

端到端确认（本机已验证：256×256、seed 42 的 qwen21 turbo 产出 196,947 字节 PNG）：

```bash
dotnet run --project src/QwenZImage.NET.WebUI -c Release
curl -s http://127.0.0.1:7860/api/health | python3 -m json.tool   # reference_backend: false
# 在 WebUI 提交一个 qwen21 turbo 任务，检查 outputs/ 里的 PNG
```

## 接入其它引擎（可选）

ABI 是可替换的，`engine/` 只是本仓库提供的默认真实实现。

- **路径 A：复用 `qzi.c` 控制层。** 实现 `qzi_backend` 的 9 个回调，在 `qzi_engine_create` 之前调用 `qzi_register_backend(&my_real_backend)`；vtable 会被复制，栈上对象也可以。[`integration/custom_backend_template.c`](../integration/custom_backend_template.c) 是可编译模板，留出了真实调用位置，并把尚未实现的 `generate_*` 委派给参考后端以便先验证 ABI。
- **路径 B：完全替换。** 自己实现并导出 `native/include/qzi.h` 的全部 `qzi_*` 函数，用 `QZI_NATIVE_LIBRARY` / `QziClientOptions.NativeLibraryPath` 指向它。代价是要自己实现单飞锁、校验、进度/取消、错误缓冲与缓冲区所有权。

9 个回调的职责、参数校验规则与状态码分别见 [`NATIVE_ABI.md`](NATIVE_ABI.md) 与 [`PARAMETERS.md`](PARAMETERS.md)。

## 常见坑

- **并发**：一个 engine = 一张 GPU + 一个常驻模型。第二次并发生成返回 `QZI_ERR_BUSY`；用 `QziGenerationQueue` 串行化，不要起多个 `QziClient` 指向同一张卡。
- **缓冲区所有权**：`image->data` 必须在 `release_image` 里释放；`.NET` 侧成功后立刻 `memcpy` 并调用 `qzi_image_release`，引擎不得复用该内存。
- **字符串生命周期**：请求里的 `const char *` 只在调用期间有效，需要保存就自己复制。
- **结构体布局**：`qzi_abi_layout()` 是硬门槛；改字段必须升 `QZI_ABI_VERSION` 并同步 `.NET` 的 `NativeStructs.cs` / `AbiGuard`。
- **符号导出**：产物名固定 `QwenZImageNative.dll` / `libQwenZImageNative.so`；Windows 只导出 `QZI_API` 函数，不要带 `lib` 前缀。
- **SM 架构**：`QZEN_CUDA_ARCH` 不匹配会导致 kernel 走错路径；构建脚本第一个参数就是它。
- **reference / CUDA 产物混用**：两者同名但内容不同，务必用 `-cuda` 目录或 `QZI_STAGE_CUDA` 区分，否则「明明跑了 build 却还是噪点」。
- **不要静默忽略非法参数**：Turbo 非 8 步、`svdq-fp4` checkpoint、带路径的 LoRA 都必须明确失败。
- **许可**：`engine/` 是上游 MIT 代码的改名子集，分发时必须保留 [`engine/LICENSE`](../engine/LICENSE) 的版权与许可声明；仓库根 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) 已列出全部第三方依赖。
