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
  "zimage_control_dir": "/models/zimage/control_union_2.1.safetensors",
  "text_cache_dir": "/models/cache"
}
```

| 键 | 用途 |
| --- | --- |
| `qwen21_dir` | Qwen-Image 2.1 权重包（turbo / base DiT、VAE、text tower、tokenizer） |
| `qwen21_lora_dir` | LoRA 目录（**turbo / base 两个 tier 共用**）；`lora` 参数必须是裸文件名，会在此目录下解析 |
| `zimage_dir` | Z-Image bf16 权重包 |
| `zimage_nunchaku_dir` | Nunchaku / SVDQ int4 checkpoint 目录，`nunchaku-r<rank>` 在此枚举 |
| `zimage_control_dir` | Z-Image Turbo Fun **ControlNet Union 2.1** 权重（`.safetensors` 文件或目录）。只有 `zimage-control` 请求会懒加载它；未配置时返回 `QZI_ERR_NOT_CONFIGURED` |
| `text_cache_dir` | 文本编码缓存 |
| `device_id` / `thread_count` | `-1` / `0` = 自动 |
| `native_library` | 可选，显式 native 库路径 |
| `preprocess_dir` | **仅 host 层**：WebUI 控制图预处理器（WanGP ONNX 权重）根目录，`QZI_PREPROCESS_DIR` 可覆盖。native 引擎与 C ABI **完全忽略**该键，核心 NuGet 包不加载任何预处理模型 |

两侧都要读这个文件，原因是 **C ABI 装不下全部字段**：

- `.NET` 侧 `QziConfig.Load()` → `QziConfig.ToClientOptions()`：完整解析、平台覆盖、`~` 展开、严格校验未知键；`qwen21_dir` / `zimage_dir` / `text_cache_dir` / `device_id` / `thread_count` 通过 `qzi_engine_config` 传给 native。`Qwen21LoraDirectory` / `ZImageNunchakuDirectory` 则走下面的环境变量通道。
- native 侧 `engine/src/qzen_config.c`：在 `qzen_init()` 里先 `qzen_config_load()` 读文件，**补齐** `qwen21_lora_dir` / `zimage_nunchaku_dir` 等 ABI 未携带的字段，再调用 `qzen_config_apply_environment()` 应用同一组 `QZI_*` 覆盖（含 `~` 展开）。`QziClient` 会在 `qzi_engine_create()` 期间把最终采用的配置文件路径临时发布为 `QZI_CONFIG`，保证两侧读的是同一份文件。

查找顺序：显式 `QZI_CONFIG` / `--config <path>` → 程序目录 → 当前工作目录 → 从这两个目录向上最多 8 级（覆盖 `dotnet run` 从项目目录启动的情况）→ `~/.config/QwenZImage/config.json`。`QZI_*` 环境变量覆盖文件；代码里显式传入的 `QziClientOptions` 优先于二者。**权重文件本身不含在仓库 / NuGet 包里**。

`qzen_config_apply_environment()` 的变量表与托管侧 `QziConfig.ApplyEnvironmentOverrides()` 一一对应：`QZI_QWEN21_DIR`、`QZI_ZIMAGE_DIR`、`QZI_QWEN21_LORA_DIR`、`QZI_ZIMAGE_NUNCHAKU_DIR`、`QZI_ZIMAGE_CONTROL_DIR`、`QZI_MODEL_CACHE`。`QziClient` 构造函数把这些变量连同 `QZI_CONFIG` 一起**作用域发布**（`NativeConfigScope.Variables` 与 `qzen_config.c:env_overrides[]` 必须保持同一张表），因此下面四种写法对引擎**等价生效**：

1. `QwenZImage.config.json` 里的 `qwen21_lora_dir` / `zimage_nunchaku_dir` / `zimage_control_dir`；
2. `QziConfig.With("qwen21_lora_dir", ...)` 或 `ToClientOptions()` 带出的代码级值；
3. 直接构造 `QziClientOptions` 并赋值两个目录；
4. 进程里直接设置 `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR` / `QZI_ZIMAGE_CONTROL_DIR`（没有配置文件也可以）。

优先级为 **显式代码值 / `QZI_*` 环境变量 > 配置文件**；native 侧不会把已覆盖的值再改回去。

> `QZI_PREPROCESS_DIR` **不在**上面这张作用域变量表里。它是 WebUI 专用的 host 层配置（控制图预处理器 ONNX 资产目录），由 `QziConfig` 解析成 `PreprocessDirectory` 后直接交给 `PreprocessAssets.Discover()`，不经过 `NativeConfigScope`、不发布给 native、也不进入 `qzen_config.c:env_overrides[]`。这样 `QZI_*` 作用域契约仍是 6 个目录变量，同时 WebUI 可以引入 `Microsoft.ML.OnnxRuntime` 等三方依赖而不污染核心包。

### 作用域发布与多实例隔离

`QZI_*` 是进程级变量，而 `qzen_init()` 只在创建 engine 时读取一次。若把代码级值永久写进进程环境，同进程里后构造的 client 会覆盖先构造 client 的配置，容器里手工设置的环境变量也会被悄悄改掉。现在的行为是：

1. `QziClient` 构造进入 `EngineCreateGate`（进程内静态锁，覆盖 `FromConfig()` 到 `qzi_engine_create()` 返回）。
2. `NativeConfigScope` 只对**非空白**的选项值临时 `SetEnvironmentVariable`，记录原值；未设置的选项保持环境原样，使「只有环境变量、没有配置文件」的部署仍然生效。`QZI_NATIVE_LIBRARY` / `QZI_DEVICE_ID` 只由托管侧消费，不在作用域内。
3. `qzi_engine_create()` 期间 native 读取并快照这些值到自己的 `qzen_state`。
4. 构造结束（成功或抛异常）立即按逆序还原进程环境；构造失败也走 `using` 回滚，不会把半套配置泄漏给后续代码。

因此同一进程可以并存多个配置不同的 `QziClient`，每个 engine 看到的就是构造它时的那套值。参考后端把这份配置指纹混入确定性 PNG，`NativeConfigScopeTests` 用它验证「不同配置 → 不同图片、相同配置 → 相同图片」，并覆盖并发构造与失败回滚。

> **Windows 陷阱（务必注意）**：C 运行时的 `getenv()` 读的是 CRT 启动时复制的环境快照，`.NET` 的 `Environment.SetEnvironmentVariable()` 走的是 Win32 API，两者不互通。早期版本在 Windows 上因此读不到托管侧发布的值。native 侧统一改用 `qzi_getenv()`（`native/include/qzi_env.h`，内部隐藏符号）：Windows 用 `GetEnvironmentVariableA` 读实时环境，POSIX 仍用 `getenv`。`qzen_config.c`、`qzi_stub.c` 与 C smoke 都走它；**自定义 backend 若要读 `QZI_*`，请从仓库源码编译 `native/src/qzi_env.c`（该函数是隐藏符号，不在 ABI 导出表里）或照此使用 `GetEnvironmentVariableA`，不要裸用 `getenv()`**。Windows 上的路径字节沿用进程 ANSI 代码页（与 `fopen` 一致），含非 ASCII 的路径建议写进配置文件。

## `qzi_engine_backend.c` 做了什么

它实现 `native/include/qzi_backend.h` 的 11 个回调（7 个原有 + 末尾四个可选的 `generate_qwen21_edit` / `generate_zimage_control` / `generate_qwen21_edit_ex` / `generate_zimage_inpaint`；四者必须留在 vtable 末尾且顺序固定，旧 backend 的初始化器偏移才不会被挤动），把上游 resident service 的 `serve_execute_qwen21()` / `serve_execute_zimage()` 编排逻辑移植成**进程内调用**（去掉 HTTP 状态）：

- `qzen_init()`：`qzen_config_load()` 读配置、`qzen_config_apply_environment()` 应用 `QZI_*` 覆盖（含 `qwen21_lora_dir` / `zimage_nunchaku_dir`、`~` 展开），再合并 ABI 传入的目录；`device_id >= 0` 时设置 `CUDA_VISIBLE_DEVICES`；`enable_gpu_switches()` 打开 `QWEN21_GPU` / `QWEN21_VAE_GPU` / `QWEN21_TEXT_GPU` / `ZIMAGE_GPU` / `ZIMAGE_VAE_GPU`（已由用户设置则不覆盖）。**不在这里 mmap 多 GB 权重**，DiT 首次 generate 时懒加载并常驻。
- Qwen-Image 2.1：`qzen_qwen21_open` → `qzen_qwen21_vae_open` → `qzen_tokenizer_load` → `qzen_qwen21_text_encode` / `qzen_qwen21_text_forward` → `qzen_qwen21_sigmas_turbo` 或 `qzen_qwen21_sigmas_base` → `qzen_qwen21_pipeline_run_ex` → `qzi_png_encode_rgb8`。编辑请求走同一个 `qzen_generate_qwen21()` 编排，只多一段参考图 VAE 编码与 mask 预处理（见下节）。
- Z-Image：`qzen_zpipe_load` → `qzen_zpipe_attach_text` → `qzen_zpipe_encode_text` → `qzen_zpipe_init_latents` → `qzen_zpipe_denoise` → `qzen_zvae_load` / `qzen_zvae_decode` → `qzi_png_encode_rgb8`。ControlNet 请求复用同一个 `qzen_generate_zimage_impl()`（`control == NULL` 即普通 t2i），只把去噪调用换成 `qzen_zpipe_denoise_control()` 并多一段控制图编码（见下节）。
- 噪点采样用与上游一致的 splitmix64 + Box-Muller，保证同一 seed 复现同样的图像。
- PNG 直接编码进 `qzi_image.data` 内存缓冲（`qzi_png_encode_rgb8`），**不落临时文件**；`release_image` 用 `free()` 释放。
- progress bridge 把引擎回调转成 `qzi_callbacks` 的 `"model load"` / `"text"` / `"image encode"`（图像编辑 / ControlNet 时）/ `"dit"` / `"vae"` / `"encode"`；Z-Image 的 `qzen_zpipe_denoise()` 回调没有 `opaque`，因此用单飞前提下的静态 `g_zimage_bridge`（ABI 层已串行化，不会并发）。ControlNet 权重懒加载会额外上报一次 `"model load"`。
- `unload` 只释放文本塔与缓存，**不卸载 DiT**（重载要几分钟）；`health` 把 `reference_backend` 置 0，其余计数由 `qzi.c` 填。

参数校验、单飞、取消、错误缓冲与 image 所有权仍全部由 `qzi.c` 负责，backend 不重复实现。

## LoRA 与 DiT 常驻状态机

Qwen-Image 2.1 的 LoRA 在 **turbo / base 两个 tier 都可用**，不是 base 专属。`qzen_qwen21_apply_lora()` 把 adapter **就地合并进 INT8 权重**，所以 DiT 一旦合并就带着该 adapter：换 adapter、改 scale、或去掉 LoRA，都必须重新加载 DiT。`ensure_qwen21_dit()` 的规则：

1. **先校验再加载**：打开多 GB 权重之前先检查 `lora` 是裸文件名（≤ `QZI_LORA_NAME_MAX`，无 `/`、`\`、`:`、`..` 与控制字符）、`qwen21_lora_dir` 已配置、拼接后的路径未超长。非法请求立即返回，不做无谓加载——这是 `qzi.c` 之后的第二道防线。
2. **0 与 1.0 等价**：`lora_scale == 0.0f` 是 ABI 默认值 sentinel，`qzi.c` 已归一化成 `1.0f`，backend 仍按 `> 0 ? scale : 1.0f` 兜底，避免把同一 adapter 误判成状态变化而触发重载。
3. **切换即清场**：`need_reload` 时先 `forget_qwen21_dit()` 关闭 DiT 并清空 tier / adapter / scale 记录，再打开 checkpoint 并重新合并；这个顺序保证「打开失败」也不会残留「adapter 已常驻」的假状态。
4. **失败不留脏权重**：`qzen_qwen21_apply_lora()` 失败时同样调用 `forget_qwen21_dit()`。否则半合并的 DiT 会被下一个无 LoRA 请求复用，静默产出错误结果。
5. **文本塔 / VAE 仍常驻**：`forget_qwen21_dit()` 只关 DiT，不重置 `s->resident`；`qzi_engine_unload()` 的语义保持不变。

`lora_resident` 容量是 `QZI_LORA_NAME_MAX + 1`，与 ABI 的 255 字节上限一致，不再出现长名字被 128 字节缓冲截断的问题。参考后端 `qzi_stub.c` 用同样的 tier / adapter / scale 变化条件上报 `"model load"`，并把这些字段连同 steps / guidance / negative_prompt 纳入确定性 PNG 哈希，使无 GPU 的 C smoke 也能验证参数贯通。

## 图像编辑：参考图 VAE 编码与 mask 恢复

`qzen_generate_qwen21_edit()` 与 t2i 共用 `qzen_generate_qwen21()` 这一条编排，**没有第二套 pipeline**。图像编辑唯一额外需要的是把参考图编码成 latents，并把它作为去噪起点：

1. **Cover fit**：ABI 交来的是参考图自身尺寸的平面 RGB8。若与输出画布尺寸不同，`resample_planar_cover()` 先居中裁剪缩放参考图，mask 用**同一套裁剪**同步缩放；尺寸本来就一致时直接透传调用方缓冲区，避免给 VAE 引入无谓的重采样模糊。
2. **VAE 编码**：`qzen_qwen21_vae_encode()` 把画布尺寸的 RGB 编成 `init_latent`，期间 progress bridge 上报 `"image encode"` `0/1 → 1/1`（并在前后轮询取消）。
3. **mask 收缩**：WanGP 把绘制的 mask 用 LANCZOS 缩到 `(width/16, height/16)` 再 `> 64` 二值化；对硬笔刷 mask，等价做法是对每个 16×16 块取面积均值再阈值化（`downsample_mask()`），得到 token-major 的 `latent_mask`。**白色 = 自由重绘，黑色 = 保持参考轨迹**。
4. **起点与去噪**：`qzen_qwen21_edit_start()` 按 WanGP 的 `denoising_strength` 规则算 `first_step = min(int(steps * (1 - strength)), steps - 1)`，在 `sigmas[first_step]` 处把 `init_latent` 与噪声按 sigma 混合，只重跑剩余步数。`strength = 1.0` 从第 0 步开始，混合结果恰好是纯噪声，与 t2i **字节级一致**（走同一条代码路径，不加额外算术）。注意这不能用「首个 `sigma <= strength`」替代：turbo 调度高度非线性，那样 `0.75` 会落在第 6 步而不是第 2 步。
5. **遮罩恢复**：`masked_steps = ceil((steps - first_step) * masking_strength)`。在前 `masked_steps` 步里，把 `latent_mask == 0` 的 latent 单元按 WanGP 公式 `known = original*(1 - sigma_next) + noise*sigma_next` 拉回参考轨迹；之后整张图自由演化，让重绘区域自然融合。`masking_strength` 已由 `qzi.c` 归一化（有 mask 时 `0` → `1.0`），引擎里仍做一次兜底，防止直接调用 engine 时被静默丢弃。

turbo 8 步、`strength = 0.75` 时 `first_step = 2`，因此 `"dit"` 只上报 6 步；参考后端 `qzi_stub.c` 用同一个公式缩短步数并把参考图 / mask / 两个 strength 纳入确定性哈希，无 GPU 的 `ctest` 与托管测试也能验证这条语义。参数校验、sentinel 归一化、单飞与取消仍全部由 `qzi.c` 负责，backend 不重复实现。

vtable 末尾的四个可选槽 `generate_qwen21_edit` / `generate_zimage_control` / `generate_qwen21_edit_ex` / `generate_zimage_inpaint` 必须**保持在最后且顺序固定**：自定义 backend 留 `NULL` 时，对应的 `qzi_generate_*()` 返回 `QZI_ERR_UNSUPPORTED`，而 t2i 完全不受影响；`integration/custom_backend_template.c` 就显式把四个槽都留空作为示例。能力探测（`qzi_abi_features()` 返回 `0xF` / `qzi_qwen21_edit_layout()` / `qzi_zimage_control_layout()` / `qzi_qwen21_edit_ex_layout()` / `qzi_zimage_inpaint_layout()`）见 [NATIVE_ABI.md](NATIVE_ABI.md)。高级编辑与遮罩重绘的实现见下两节。

## 高级图像编辑：Red Canvas / LanPaint / RGBA

第三个增量能力仍走 `qzen_generate_qwen21_impl()` 这一条编排，没有第二套 pipeline、也没有新权重：`qzen_generate_qwen21_edit_ex()` 只是把 `qzi_qwen21_edit_ex_request` 作为 `adv` 传进同一个实现（内嵌的普通编辑请求由 `edit_ref` 统一读取）。`qzi_engine_backend.c` 把它拆成 `qzen_qwen21_sample` 的新字段（`outpainting` / `lanpaint_steps` / `lanpaint_seed` / `alpha`），三项能力各自只改采样或解码的一段：

1. **Red Canvas 外扩**：参考图 cover-fit 后居中放进扣掉四边距的内部源矩形，四周填不透明红 `FF0000`，再交给 `qzen_qwen21_vae_encode()`；同时构造一张同尺寸的画布 mask——边距置 `255`（重绘），内部置 `0`（保留）或用户笔刷 mask cover-fit 后的值——再降采样到 1/16 latent 网格。prompt 追加 WanGP 固定后缀 `Remove the red paddings on the sides and show what's behind them.`。`qzen_qwen21_pipeline.c` 令 `step_offset = 0`（红色边距跑完整调度），把边距与绘制区对应的 latent token 从 `sigmas[0]` 的全噪声起步，其余 token 保持 `sigmas[first_step]` 的参考轨迹，并用 `latent_mask == 0` 把保留区 pin 到已知轨迹。边距是 32 对齐的，因此降采样后正好落在 latent 网格边界上。**保真度说明**：WanGP 的编辑模型是 condition-prefix（参考 latent 拼在带噪目标前，DiT 用注意力读它），所以它能从纯噪声起步仍保住源图；本引擎是 img2img sampler（参考图就是起点 latent），只能靠上述 latent masking 近似，外扩结果不是 WanGP 的逐像素复刻。
2. **LanPaint**：`qzen_qwen21_pipeline.c` 里的 C 移植对应 WanGP `shared/lanpaint.py` 的 Algorithms 1–4（BiG-score 双向引导 + Langevin 迭代）。`model_mode` 的 2/3/4/5 映射成 `lanpaint_steps = 2/5/10/15`，每个 sampler 步之前用 `q21_lp_target()` 求一次引导目标、`q21_lp_oscillate()` / `q21_lp_refine()` 做带 momentum 的二阶 Strang 分裂迭代；随机流由 `lanpaint_seed`（请求 seed 派生）独立产生，保证同 seed 可复现。它要求 `init_latent` 与 `latent_mask` 都存在，因此 ABI 层强制 LanPaint 带 mask、禁外扩、并把两个 strength 归一化成 `1.0`。`tokens_major` 复用为 momentum scratch，避免每个去噪步重复分配多 MB 缓冲。
3. **RGBA 输出**：`qzen_qwen21_vae_decode_ex()` 在 CPU 与 CUDA 两条解码路径上都把 `conv_out` 的第 4 通道写进可选的 `alpha` 平面（本来的 `144→4` 输出此前只取前 3 通道），量化统一走 `q21_quantize()` 的 `clamp + (x+1)*127.5`，不需要额外前向。原 `qzen_qwen21_vae_decode()` 保留为 `alpha = NULL` 的薄封装，所有旧调用方 ABI 不变。`qzi_png_encode_rgba8()` 把 RGB + alpha 交错成 4 通道并写 PNG colour type 6；参考后端用同一个编码器产出确定性的 RGBA 字节，便于无 GPU 的 `ctest` / xUnit 验证 colour type 与尺寸。

能力位 `QZI_FEATURE_QWEN21_EDIT_ADV = 0x4` 是第三个增量（第四个增量 `QZI_FEATURE_ZIMAGE_INPAINT = 0x8` 让 `qzi_abi_features()` 现返回 `0xF`）；`qzi_backend` 在该槽之后又追加了 `generate_zimage_inpaint`（现共 11 个回调），`integration/custom_backend_template.c` 把四个可选槽都显式留成 `NULL` 作为示例。留空时 `qzi_generate_qwen21_edit_ex()` 返回 `QZI_ERR_UNSUPPORTED`，t2i、普通编辑与 ControlNet 均不受影响。参数与校验见 [PARAMETERS.md](PARAMETERS.md)，C 契约见 [NATIVE_ABI.md](NATIVE_ABI.md)。

## Z-Image ControlNet：权重加载与 hint 注入

`engine/src/qzen_zimage_control.c/.h` 负责读取 Z-Image Turbo Fun **ControlNet Union 2.x** 的 `.safetensors`：15 个 control layer、2 个独立的 control noise refiner，以及 `before_proj` / `after_proj` / embedder 等线性层。加载器按 `in_dim` 判定版本，**只接受 33 通道的 v2 / v2.1**（`[control_latent(16) | keep_mask(1) | inpaint_latent(16)]`），v1 的 16 通道会明确拒绝——三者的 layer 数与 refiner 数不同，共用一套注入逻辑会静默产出错误结果。`qzen_zimage_control_upload()` 把权重上传到当前 DiT 的 GPU 上下文；上传失败（例如 24 GiB 卡放不下约 3.7 GiB 的 Union 权重加上常驻 DiT / 文本塔）不算致命错误，`qzen_zpipe_control_prepare()` 会退回 **CPU hint 路径**，结果正确但明显更慢。

真实后端的 `qzen_generate_zimage_impl()` 同时服务 t2i、ControlNet 与 inpaint 三条路径：`control_eff = control ? control : (inpaint ? &inpaint->base : NULL)` 让后两者共用同一份编排，`control_eff == NULL` 走原路径，`!= NULL` 时按下列顺序多执行一段，其余（文本编码、噪声初始化、VAE 解码、PNG 编码）完全复用：

1. **懒加载**：`ensure_zimage_control()` 只在首个 ControlNet 请求时读 `zimage_control_dir`，期间上报一次 `"model load"`。未配置时返回 `QZI_ERR_NOT_CONFIGURED` 并提示写进 `QwenZImage.config.json` 或设 `QZI_ZIMAGE_CONTROL_DIR`——这是配置错误而非引擎故障。权重常驻 `qzen_state.z_control`；切换 Z-Image checkpoint 会释放 DiT，因此用 `z_control_pipe != zpipe` 判断是否要重新上传。
2. **控制图编码**：上报 `"image encode" 0/1`，把 ABI 交来的平面 RGB8 用 `resample_planar_cover()` 居中裁剪缩放到输出画布（尺寸已一致时直接透传），再由 `qzen_zvae_encode()` 编成 16 通道 1/8 latent；`qzen_zpipe_control_context()` 补上全 0 的 `keep_mask` 与 `inpaint_latent` 得到 33 通道 context。本次加载的 VAE 会**复用于后续解码**，避免重复读取约 330 MB。
3. **逐步 hint**：`qzen_zpipe_denoise_control()` 每个 Euler 步都用当前 `x_t` 与 `tau = 1 - sigma` 重算 hints（control layer 的 AdaLN 依赖时间步，不能只算一次）。`zpipe_control_hints_compute()` 产出 2 条 noise refiner skip + 15 条 main skip；`zpipe_hint_for()` 按 WanGP 的 `control_layers_places = [0,2,...,28]` 只把第 `j` 条 main skip 注入**偶数** main layer `2*j`，两条 refiner skip 注入 `noise_refiner.0/1`。
4. **注入公式**：`hidden_states += control_scale * skip`（`alpha = control_context_scale`）。CPU 路径由 `zpipe_apply_hint_cpu()` 直接做浮点加法；GPU 路径由 `zpipe_apply_hint_gpu()` 把 hint 转 bf16 写入一块惰性分配的 scratch，再调 `qzen_gpu_scale_add_bf16()` 原地累加。`hints == NULL` 时 `zpipe_dit_forward_cpu/gpu()` 与 t2i 逐位一致，普通 `qzen_zpipe_denoise()` 永远传 `NULL`。
5. **收尾**：`done:` 统一释放 `guide` / `control_latent` / `control_context`；错误与取消路径同样经过这里，不泄漏多 GB 缓冲区。

`control_type` 只做校验、归一化与确定性哈希，**不触发姿态 / 边缘 / 深度预处理**——对齐 WanGP 的 `guide_preprocessing = "V"`（Use Z-Image Raw Format），调用方必须自己提供已预处理的控制图。`control_scale = 0` 由 `qzi.c` 归一化为 `0.65`（WanGP Union 2.1 的 `control_net_weight_alt`）；backend 仍做一次 `> 0 ? scale : 0.65f` 兜底，防止自定义调用方绕过 ABI 层。取值依据与 `control_type` 白名单的核对过程见 [PARAMETERS.md](PARAMETERS.md)。

## Z-Image 遮罩重绘：33 通道 context 的后 17 个平面

第四个增量补齐 WanGP `models/z_image/pipeline_z_image.py` 中 `control_in_dim == 33 && inpaint_mask is not None` 的分支。上面 ControlNet 章节里的 `qzen_zpipe_control_context()` 把 `keep_mask` / `inpaint_latent` 恒置 0（对应 WanGP 的 `inpaint_mask is None`）；`engine/src/qzen_zimage_pipeline.c` 新增的 `qzen_zpipe_control_context_ex()` 是它的超集，两者都传 `NULL` 时逐位复现原行为，因此普通 t2i / ControlNet 的数值不受影响。

`qzen_generate_zimage_inpaint()` 把 `qzi_zimage_inpaint_request` 交给同一个 `qzen_generate_zimage_impl()`，在「控制图编码」之后、`qzen_zpipe_denoise_control()` 之前多执行：

1. **mask cover fit**：mask 与控制图**各自独立**居中裁剪缩放到输出画布（尺寸已一致时直接透传），因为 ABI 允许两者来源分辨率不同。
2. **遮罩源图**：`inpaint_rgb = guide_rgb * (1 - mask)`，白（255）区抹黑、黑（0）区保留；中间值按线性比例参与。这与 WanGP `inpaint_image = control_image_tensor * (1-inpaint_mask)` 一致。
3. **第二次 VAE 编码**：复用已加载的 VAE 对 `inpaint_rgb` 调 `qzen_zvae_encode()` 得到 `inpaint_latent`，与 `control_latent` 同为 `[16][lh][lw]` 平面布局、同样含 `shift_factor` / `scaling_factor` 归一化。
4. **keep 平面**：`downsample_mask_keep()` 对每 8×8 块取面积均值得到 `[0,1]` 软 `keep_mask = 1 - painted`（WanGP 用 `F.interpolate(..., mode='nearest')`，硬笔刷边界上最多相差 8 像素；面积均值对软边 mask 更平滑）。
5. **context 拼接**：`qzen_zpipe_control_context_ex()` 产出 `[control_latent(16) | keep_mask(1) | inpaint_latent(16)]`；只给其中一个 mask 参数、或对 16 通道 v1 权重做 inpaint，都会在引擎层明确报错，不会拼出越界的 context。
6. **收尾**：`done:` 统一释放 `mask_canvas` / `inpaint_rgb` / `inpaint_latent` / `keep_mask`，错误与取消路径同样经过这里。`adv == NULL` 的 t2i、`control == NULL` 的普通编辑路径零改动。

progress bridge 把两次 VAE 编码都折算进同一个 `"image encode"` 阶段（真实后端仍是单次阶段上报）；参考后端为便于无 GPU 验证，会按两个独立阶段上报。能力位 `QZI_FEATURE_ZIMAGE_INPAINT = 0x8`、vtable 第 11 个槽与 `qzi_zimage_inpaint_layout()` 见 [NATIVE_ABI.md](NATIVE_ABI.md)，参数契约见 [PARAMETERS.md](PARAMETERS.md)。

## 构建与验证真实后端

```bash
# Windows / Linux：默认生成覆盖 Ampere -> Blackwell 的 fat binary
scripts\build-native-cuda.bat
bash scripts/build-native-cuda.sh

# 本地快速构建：只编一个 SM（脚本第一个参数是分号分隔的 SM 列表）
scripts\build-native-cuda.bat 89
bash scripts/build-native-cuda.sh 89

# 或手工
cmake -S native -B native/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DQZI_WITH_CUDA_ENGINE=ON -DQZEN_CUDA_ARCH=89 -DQZI_BUILD_TESTS=ON
cmake --build native/build-cuda
ctest --test-dir native/build-cuda --output-on-failure
```

`QZEN_CUDA_ARCH` 默认是分号分隔的多架构列表 `80;86;87;88;89;90;100;103;110;120;121`：一个二进制覆盖 Ampere / Ada / Hopper / Blackwell，NuGet 使用者不必按自己的显卡重编译。刻意不含 sm_75 —— attention kernel 使用 `mma.sync.m16n8k16`，要求 sm_80+。CMake 会查询 `nvcc --list-gpu-arch`，自动剔除当前 CUDA 工具链不认识的架构（例如 CUDA 12.x 没有 sm_121），全部不支持时明确报错。

单架构构建时该值成为 `QZEN_SM_ARCH`；fat binary 则由 configure 生成的 `qzen_arch_config.h` 提供 `QZEN_SM_ARCH_LIST`，`qzen_gpu.cu` 在初始化时检查设备是否落在列表内，不匹配会直接打印「rebuild with -DQZEN_CUDA_ARCH=<sm>」，而不是让每个 kernel 静默返回 `cudaErrorNoKernelImageForDevice`。

Windows 需要 MSVC + CUDA Toolkit 13+（nvcc / cudart / cuBLAS / cuBLASLt），Linux 需要 GCC/Clang + OpenMP。**Linux 不再需要 ICU 开发包**：`engine/platform/posix/qzen_posix_unicode.c` 在运行时 `dlopen` 宿主 ICU 并解析带版本后缀的符号（如 `u_charType_72`），完全没有 ICU 时退化为内置 UTF-8/UTF-16 转换与 ASCII/CJK 字符分类。这样 `libQwenZImageNative.so` 不带 `NEEDED libicuuc.so.NN`，可跨发行版加载；ICU 存在时 NFC 归一化仍由真实 ICU 完成。OpenMP 通过 `OpenMP::OpenMP_C` 显式链接，避免共享库留下未解析的 `omp_*` 符号。

`build-native-cuda.bat` 会：如果当前 shell 已经有完整的 `cl` / `rc` / `nvcc` 环境就直接复用（不调用 `vcvars64.bat`）；否则自动 `call vcvars64.bat`；若注册表查询不可用导致 Windows SDK 的 `rc.exe` / `mt.exe` 没进 PATH，则**从文件系统**枚举 `%ProgramFiles(x86)%\Windows Kits\10\Include` 补齐 SDK 的 `PATH` / `INCLUDE` / `LIB`，避免 CMake 的 MSVC 编译检查失败。脚本最后用 `QZI_STAGE_CUDA=1` 把产物 stage 进 `artifacts/runtimes/win-x64/native/`。

`ctest` 的 `qzi_abi_smoke` 会按 `health.reference_backend` 分支：reference 构建跑完整生成契约（PNG 签名、确定性、release、LoRA / 编辑 / 控制 / 高级编辑 / inpaint 参数引起的字节差异、RGB 与 RGBA 的 PNG colour type、`"model load"` 与 `"image encode"` phase、inpaint 的两个编码阶段），两种构建都跑 ABI 布局、四个能力位（`0xF`）与四个扩展 layout、LoRA 裸文件名 / 长度 / scale 边界、编辑 / 控制 / 高级编辑 / inpaint 参数边界、fp4 大小写校验以及 legacy backend 四个可选槽为 `NULL` 时的 `QZI_ERR_UNSUPPORTED`；真实构建**不加载多 GB 权重**。

POSIX 构建额外注册两项 ICU 双路径检查（Windows 只有 `qzi_abi_smoke` 一项，因此 Linux `ctest` 是 3 项）：`posix_unicode_with_icu` 正常 `dlopen` 宿主 ICU，`posix_unicode_without_icu` 用 `-Wl,--wrap=dlopen` 让所有 `dlopen` 失败以走内置回退。测试源是 `native/tests/qzi_posix_unicode_check.c`，直接编译 `engine/platform/posix/qzen_posix_unicode.c`，不依赖 CUDA，所以 GPU-less CI 也会跑。两条路径必须同时满足字符分类、空白判断、含 U+1F98A 代理对的 UTF-8 ↔ UTF-16 字节级 round-trip、非法 UTF-8 报错与已组合文本 NFC 不变；唯一允许的差异是 `e + U+0301`：真实 ICU 组合为 `U+00E9`，回退按设计保持恒等。因为兼容层用 `pthread_once` 只解析一次 ICU，两项测试用两个独立可执行；链接器不支持 `--wrap` 时 CMake 打印 STATUS 并跳过回退变体，宿主无 ICU 时对应测试以退出码 77 记为 SKIP。

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

## 真图回归矩阵

`ctest` 里的 `qzi_abi_smoke` 在真实构建下**不加载权重**，只能验证 ABI 与参数校验。要验证真实引擎在多个 checkpoint、LoRA 和更大分辨率下都能出图，用独立工具 `tools/RealImageRegression`。它刻意**不加入 `QwenZImage.NET.sln`**：每个用例都需要 GPU 和多 GB 权重，单次运行以分钟计，不适合无 GPU 的 CI。

前置条件：已构建 `-DQZI_WITH_CUDA_ENGINE=ON` 的产物、`QwenZImage.config.json` 指向真实权重目录（含 `qwen21_lora_dir`、`zimage_nunchaku_dir`，跑 ControlNet 时还要 `zimage_control_dir`）。工具会读取 `health.reference_backend`，**如果加载的是参考后端就直接拒绝运行**，避免把确定性噪点图记成「真图通过」。

```bash
# Windows
.\scripts\run-real-image-regression.ps1 --list
.\scripts\run-real-image-regression.ps1 --only qwen21-turbo-512 --out artifacts\regress\smoke
.\scripts\run-real-image-regression.ps1 --out artifacts\regress\full --timeout-minutes 40

# Linux / WSL
bash scripts/run-real-image-regression.sh --list
bash scripts/run-real-image-regression.sh --out artifacts/regress/full
```

用例矩阵在 `tools/RealImageRegression/cases.json`：qwen21 turbo 512 / 重复请求（确定性对照）/ LoRA scale 1.0 与 0.6 / 1024×1024 / base tier，以及 Z-Image bf16 512 与 1024、Nunchaku `r32` 与 `r256`。新增用例只需加一条 JSON；`--only <id>` 可重复用于挑选子集。

每个用例产出一个 PNG，外加 `report.md` / `report.json`，记录分辨率（从 PNG IHDR 读取并与请求比对）、字节数、SHA-256、耗时、phase 序列和引擎计数器。报告在**每个用例结束后增量落盘**，因此中途某个 CUDA 用例致命失败也不会丢掉已完成的结果。`compare_with` 用字节级 SHA-256 对照验证真实引擎的确定性；phase trace 里的 `ModelLoad` 用来实证 LoRA / tier / checkpoint 切换确实触发了权重重载。单用例超时由 `--timeout-minutes` 控制，默认 30 分钟。

## 接入其它引擎（可选）

ABI 是可替换的，`engine/` 只是本仓库提供的默认真实实现。

- **路径 A：复用 `qzi.c` 控制层。** 实现 `qzi_backend` 的 11 个回调（末尾 `generate_qwen21_edit` / `generate_zimage_control` / `generate_qwen21_edit_ex` / `generate_zimage_inpaint` 四个槽可选，但必须在最后且顺序固定），在 `qzi_engine_create` 之前调用 `qzi_register_backend(&my_real_backend)`；vtable 会被复制，栈上对象也可以。[`integration/custom_backend_template.c`](../integration/custom_backend_template.c) 是可编译模板，留出了真实调用位置，并把尚未实现的 `generate_*` 委派给参考后端以便先验证 ABI；模板把四个可选槽位都显式留成 `NULL`，因此它只服务 t2i。
- **路径 B：完全替换。** 自己实现并导出 `native/include/qzi.h` 的全部 `qzi_*` 函数，用 `QZI_NATIVE_LIBRARY` / `QziClientOptions.NativeLibraryPath` 指向它。代价是要自己实现单飞锁、校验、进度/取消、错误缓冲与缓冲区所有权。

11 个回调的职责、参数校验规则与状态码分别见 [`NATIVE_ABI.md`](NATIVE_ABI.md) 与 [`PARAMETERS.md`](PARAMETERS.md)。

## 常见坑

- **并发**：一个 engine = 一张 GPU + 一个常驻模型。第二次并发生成返回 `QZI_ERR_BUSY`；用 `QziGenerationQueue` 串行化，不要起多个 `QziClient` 指向同一张卡。
- **缓冲区所有权**：`image->data` 必须在 `release_image` 里释放；`.NET` 侧成功后立刻 `memcpy` 并调用 `qzi_image_release`，引擎不得复用该内存。
- **字符串生命周期**：请求里的 `const char *` 只在调用期间有效，需要保存就自己复制。
- **结构体布局**：`qzi_abi_layout()` 是硬门槛；改字段必须升 `QZI_ABI_VERSION` 并同步 `.NET` 的 `NativeStructs.cs` / `AbiGuard`。
- **符号导出**：产物名固定 `QwenZImageNative.dll` / `libQwenZImageNative.so`；Windows 只导出 `QZI_API` 函数，不要带 `lib` 前缀。
- **SM 架构**：默认列表覆盖 sm_80 → sm_121，设备不在其中时 native 初始化会打印明确的 rebuild 提示，而不是让所有 kernel 静默失败；本地可用脚本第一个参数（分号分隔的 SM 列表，如 `89`）只编一个架构加速。CUDA 12.x 不认识 Blackwell 架构时 CMake 会自动剔除，全部不支持才报错。
- **reference / CUDA 产物混用**：两者同名但内容不同，务必用 `-cuda` 目录或 `QZI_STAGE_CUDA` 区分，否则「明明跑了 build 却还是噪点」。
- **不要静默忽略非法参数**：Turbo 非 8 步、含 `fp4`（大小写不敏感）的 checkpoint、带路径 / `:` / 控制字符 / 超 255 字节的 LoRA、没有 LoRA 却带非 0 `lora_scale`，都必须明确失败；`steps` / `guidance` / `lora_scale` 的 `0` 是**默认值 sentinel**，不是「非法但被改成默认值」。编辑侧同理：`strength` / `masking_strength` 的 `0` 是 sentinel（分别 `0.75` / `1.0`），但**没有 mask 却带 `masking_strength`** 必须报错——否则它会被静默丢弃；参考图与 mask 尺寸不一致也必须在进队列之前拒绝。
- **四个增量能力都不是默认存在**：旧版 v1 库不导出 `qzi_abi_features()` / `qzi_qwen21_edit_layout()` / `qzi_zimage_control_layout()` / `qzi_qwen21_edit_ex_layout()` / `qzi_zimage_inpaint_layout()` / 四个 `qzi_generate_*()`；只有前三个增量的库也缺第四个的 layout 与 generate 导出。`.NET` 侧 `SupportsImageEditing` / `SupportsControlNet` / `SupportsAdvancedEditing` / `SupportsInpaint` 为 `false` 时不要调用对应 API（第三项前置依赖普通编辑、第四项前置依赖 ControlNet）。自定义 backend 把四个可选槽留 `NULL` 会得到 `QZI_ERR_UNSUPPORTED`，这是预期行为，不是崩溃。
- **高级编辑是第三个增量能力，不是普通编辑的别名**：旧版 v1 库或只有第二个增量的库不导出 `qzi_qwen21_edit_ex_layout()` / `qzi_generate_qwen21_edit_ex()`，且 `qzi_abi_features()` 不带 `0x4`。`.NET` 侧 `SupportsAdvancedEditing` 为 `false` 时不要调用 `GenerateQwen21EditEx`；WebUI 会在 `/api/jobs` 提交前返回 `400`。能力探测还**依赖普通编辑能力**（高级请求内嵌 `qzi_qwen21_edit_request`），因此旧库上普通编辑可用、高级编辑不可用是预期状态。
- **外扩的保真度是近似，不是逐像素复刻 WanGP**：WanGP 的 Qwen-Image 编辑是 condition-prefix 模型，参考 latent 作为注意力前缀，所以红色边距可以从纯噪声起步仍保住源图；本引擎是 img2img sampler（参考图即起点 latent），只能靠「边距与绘制区从全噪声起步、保留区 pin 到已知轨迹」近似。评估外扩质量时不要拿与 WanGP 的像素一致性当验收标准。
- **LanPaint 的约束不要在 backend 里放宽**：`model_mode 2/3/4/5` 必须有 mask、不能与任何非零边距并用、两个 strength 必须为 `1.0`（`qzi.c` 已归一化）；`qzen_qwen21_pipeline.c` 在缺 `init_latent` / `latent_mask` 时会直接报 `denoise: LanPaint requires ...`，因为它的 Langevin 迭代需要一个可精修的遮罩区域。擅自去掉这些约束会静默产出无意义结果。
- **RGBA 别在托管侧再合成一次 alpha**：VAE `conv_out` 本来就是 `144→4`，`qzen_qwen21_vae_decode_ex()` 已经用与 RGB 相同的量化写出第 4 通道，`qzi_png_encode_rgba8()` 直接输出 colour type 6；参考后端用同一个编码器做确定性对照。`rgba == 0` 走原 RGB 路径且字节不变，不要给 RGB 图硬塞一个全 255 的 alpha 平面。
- **ControlNet 权重是独立配置**：`zimage_control_dir` 不在 C ABI 里，也不在 `qzi_engine_config` 中；只在 `QwenZImage.config.json` / `QziConfig.With` / `QZI_ZIMAGE_CONTROL_DIR` 这条配置链上传。没配就提交 `zimage-control` 会返回 `QZI_ERR_NOT_CONFIGURED`（明确提示文件名），不会退化成无控制的 t2i。只支持 **Union v2 / v2.1**（33 通道）；指到 v1 权重会在加载阶段被拒绝。
- **不要把未预处理的照片直接送进 C ABI / 核心库**：`control_type` 不触发任何姿态 / 边缘 / 深度估计，`control_rgb` 会按原样 VAE 编码。对齐 WanGP 的 `guide_preprocessing = "V"`，**直接调用 NuGet 包或 C ABI 的调用方**须自备已成形的 guide，否则生成结果只是「被一张照片轻微扰动」，不会得到结构控制。**这不等于要求 WebUI 用户手工预处理**：`QwenZImage.NET.WebUI` 在 host 层用 `POST /api/preprocess` 把普通照片转成 guide（对齐 WanGP `wgp.py::process_map_video_guide` 的 `P/D/E/S/C/V`），再把得到的栅格作为普通 `control_rgb` 提交；`preprocess_dir` / `QZI_PREPROCESS_DIR` 只被 WebUI 读取，native 引擎完全忽略。
- **ControlNet 与常驻模型叠加显存**：Union 2.1 权重约 3.7 GiB，加上常驻 DiT 与文本塔，24 GiB 卡可能上传失败并退回 CPU hint 路径；这不是错误（结果正确但慢），但从 phase 耗时会明显看出差别。切换 Z-Image checkpoint 会释放 DiT，第一次再跑 ControlNet 会重新上传。
- **主机内存与 block staging**：`qzen_zimage_block_forward_gpu_host()` 每个去噪步会被调用约 50 次（15 control layers 两遍 + 2+2 refiner + 主 DiT）。早期实现每次都 `malloc(seq*dim*2)` 再 `free`，在双份主机 / 设备权重（本机约 45 GiB 私有提交量）压力下，冷启动首个 ControlNet 请求偶发返回 `denoise: cannot stage Z-Image GPU input`。现在 `qzen_zimage_gpu` 持有 grow-only 的 `host_stage`，在 `qzen_zimage_gpu_reserve()` 里一次性按 `plane = seq*dim` 分配，`zimg_read_bf16_f32()` 与 `qzen_zimage_t_embedder_forward_gpu()` 复用它；`qzen_gpu_tensor_from_bf16()` / `qzen_gpu_tensor_read_bf16()` 都是同步 `cudaMemcpy`，H2D 与 D2H 共用同一主机缓冲是安全的。若在更小的机器上仍看到 `cannot allocate Z-Image host staging buffer`，错误会带出行数与 MiB，说明是宿主内存/提交上限不足，而不是 kernel 失败。
- **LoRA 目录覆盖**：`qwen21_lora_dir` 不在 C ABI 里。用 `QwenZImage.config.json`、`QziClientOptions.Qwen21LoraDirectory` 或 `QZI_QWEN21_LORA_DIR` 指定；只改 `qzi_engine_config.qwen21_dir` 不会改变 LoRA 目录。目录未配置且有 LoRA 时返回 `QZI_ERR_NOT_CONFIGURED`。
- **换 LoRA 很贵**：adapter / scale 变化会重新打开 DiT（分钟级），并额外上报一次 `"model load"`；批量任务尽量把同一 adapter 的请求排在一起。
- **`QZI_*` 仍是进程级变量，但不再互相串扰**：`QziClient` 只在 `qzi_engine_create()` 期间作用域发布、随后还原，native 在 `qzen_init()` 读取一次并快照；创建 engine 之后再改变量不会改变已有 engine。注意 GPU 侧的 `CUDA_VISIBLE_DEVICES` 与常驻 CUDA 上下文仍是进程级资源，多 `QziClient` 指向同一张卡仍须用 `QziGenerationQueue` 串行化（见上文「并发」）。
- **许可**：`engine/` 是上游 MIT 代码的改名子集，分发时必须保留 [`engine/LICENSE`](../engine/LICENSE) 的版权与许可声明；仓库根 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) 已列出全部第三方依赖。
