# QwenZImage.NET Native C ABI (v1)

`native/include/qzi.h` 是**唯一**的托管/原生契约。任何实现并导出其中 `qzi_*` 函数的共享库都可以被 `QwenZImage.NET` 加载，无论内部是确定性参考实现还是真实 CUDA 引擎。本仓库统一产出 `QwenZImageNative.dll` / `libQwenZImageNative.so` 两个 flavour：默认链接 `qzi_stub.c`（噪点，CI 用），`-DQZI_WITH_CUDA_ENGINE=ON` 链接 `qzi_engine_backend.c` + `engine/`（真实推理）。两者 ABI 完全相同，托管侧一行不用改。本文件说明调用规则、内存所有权与版本策略；参数语义见 [PARAMETERS.md](PARAMETERS.md)，后端实现与构建见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。

## 设计原则

1. **纯 C 链接**：`extern "C"`，无 C++ 异常跨边界、无 STL、无平台特定运行时类型。
2. **固定 POD 布局**：所有结构体都是 POD；`qzi_abi_layout()` 报告 `sizeof()`，托管侧启动时逐一断言（见 `AbiGuard`），packing 不匹配立即抛异常，而不是静默踩坏内存。
3. **输入内存归调用方**：字符串为 NUL 结尾 UTF-8，仅在调用期间被读取；调用返回后引擎不得保留指针。
4. **输出内存归引擎**：`qzi_image.data` 指向完整 PNG 文件，直到 `qzi_image_release()` 被调用；每个成功的 image **必须且只能释放一次**。
5. **错误通过 caller buffer 返回**：所有可失败函数返回 `qzi_status`，失败时把可读的 NUL 结尾消息写入调用方的 `error` 缓冲区（建议 `QZI_ERROR_CAPACITY = 512`）。不使用 `errno`、不抛异常、不返回动态分配的错误字符串。
6. **同步单飞**：`qzi_generate_*` 是同步长调用（可能数分钟）。同一 engine 上并发第二次生成返回 `QZI_ERR_BUSY`，不会排队等待。上层用 `QziGenerationQueue` 串行化。
7. **进度 phase 是静态字面量**：`"model load"` / `"text"` / `"image encode"` / `"dit"` / `"vae"` / `"encode"`，由引擎拥有，不得释放。其中 `"image encode"` **只在图像编辑 / ControlNet 时出现**（把参考图或控制图 VAE 编码成 latents）。回调在生成线程上执行，必须快速返回且不得重入引擎。托管侧把未知字面量降级为 `GenerationPhase.Unknown`，不会抛异常。

## ABI 版本与布局

| 常量 | 值 |
| --- | --- |
| `QZI_ABI_VERSION` | `1` |
| `QZI_ERROR_CAPACITY` | `512` |
| `QZI_LAYOUT_SLOTS` | `6` |
| `QZI_LORA_NAME_MAX` | `255`（LoRA 裸文件名的最大 UTF-8 字节数，不含 NUL） |

> 新增 `QZI_LORA_NAME_MAX` 只影响输入校验，不改变任何结构体布局或函数签名，因此 **ABI 版本仍是 v1**。

图像编辑与 Z-Image ControlNet 同样是 **v1 的增量扩展**：没有改动任何既有结构体、既有导出函数或 `qzi_abi_layout()` 的 6 个槽位，只各新增一个请求结构体和若干独立导出函数。因此旧版 v1 库仍能正常服务 t2i 调用方，只是不导出这些新符号；新调用方必须先用 `qzi_abi_features()` 探测。

| 能力位（`qzi_abi_features()`） | 值 | 含义 |
| --- | --- | --- |
| `QZI_FEATURE_QWEN21_EDIT` | `0x1` | 该库实现了 `qzi_generate_qwen21_edit()` |
| `QZI_FEATURE_ZIMAGE_CONTROL` | `0x2` | 该库实现了 `qzi_generate_zimage_control()`（Z-Image Turbo Fun ControlNet Union） |
| `QZI_FEATURE_QWEN21_EDIT_ADV` | `0x4` | 该库实现了 `qzi_generate_qwen21_edit_ex()`（Red Canvas 外扩 / LanPaint / RGBA 输出） |
| `QZI_FEATURE_ZIMAGE_INPAINT` | `0x8` | 该库实现了 `qzi_generate_zimage_inpaint()`（Z-Image Union 2.x 遮罩重绘） |

当前两个内置 backend 的 `qzi_abi_features()` 均返回 `0xF`。

> 能力位描述的是**库**是否实现了入口，不是当前注册的 backend 是否支持：自定义 backend 把对应 vtable 槽留成 `NULL` 时，四个 `qzi_generate_qwen21_edit()` / `qzi_generate_zimage_control()` / `qzi_generate_qwen21_edit_ex()` / `qzi_generate_zimage_inpaint()` 入口都返回 `QZI_ERR_UNSUPPORTED`，而 `qzi_abi_features()` 依旧带该位。老版本库根本不导出 `qzi_abi_features()`，托管侧用 `NativeLibrary.TryGetExport` 探测，缺失即视为「所有可选能力都不支持」。

`qzi_abi_layout(size_t *sizes, size_t count)` 按以下槽位写 `sizeof()`（x86-64 / 64 位平台上实测值）：

| 槽位 | 宏 | 结构体 | x86-64 `sizeof` |
| --- | --- | --- | --- |
| 0 | `QZI_LAYOUT_ENGINE_CONFIG` | `qzi_engine_config` | 48 |
| 1 | `QZI_LAYOUT_QWEN21_REQUEST` | `qzi_qwen21_request` | 72 |
| 2 | `QZI_LAYOUT_ZIMAGE_REQUEST` | `qzi_zimage_request` | 40 |
| 3 | `QZI_LAYOUT_CALLBACKS` | `qzi_callbacks` | 24 |
| 4 | `QZI_LAYOUT_IMAGE` | `qzi_image` | 32 |
| 5 | `QZI_LAYOUT_HEALTH` | `qzi_health` | 64 |

> 32 位平台上的尺寸会不同，这是允许的：托管与 native 必须来自**同一次构建/同一个 ABI**，由 `qzi_abi_layout()` 在运行时校验，而不是硬编码字节数。托管侧结构体见 `src/QwenZImage.NET/Native/NativeStructs.cs`，函数指针用 `IntPtr` 表示以保证 blittable。

`qzi_qwen21_edit_layout()` / `qzi_zimage_control_layout()` / `qzi_qwen21_edit_ex_layout()` / `qzi_zimage_inpaint_layout()` 分别返回四个增量请求结构体的 `sizeof()`。它们**不占用**上面 6 个槽位：这样旧库不必导出新符号，`AbiGuard` 也能在把编辑 / ControlNet / 高级编辑 / inpaint 请求指针跨边界传出去之前，先比对结构体尺寸，packing 不匹配立即抛异常。缺席对应导出同样意味着「不支持该能力」。具体字节数不要硬编码，32 位平台会不同，一律以函数返回值与托管侧 `sizeof` 比对。

托管侧 `LibraryImport` 的调用约定由源生成器按平台处理；`qzi_progress_fn` / `qzi_cancel_fn` 反向回调在 `.NET` 侧用 `static` + `[UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]` 实现，不分配 delegate。

## 导出函数

| 函数 | 作用 |
| --- | --- |
| `uint32_t qzi_abi_version(void)` | 返回 `QZI_ABI_VERSION` |
| `const char *qzi_status_text(qzi_status)` | 静态状态描述，不释放 |
| `uint32_t qzi_abi_features(void)` | 增量能力位；当前返回 `QZI_FEATURE_QWEN21_EDIT \| QZI_FEATURE_ZIMAGE_CONTROL \| QZI_FEATURE_QWEN21_EDIT_ADV \| QZI_FEATURE_ZIMAGE_INPAINT`（`0xF`）。旧版 v1 库不导出此符号 |
| `size_t qzi_qwen21_edit_layout(void)` | 返回 `sizeof(qzi_qwen21_edit_request)`；旧版 v1 库不导出此符号 |
| `size_t qzi_zimage_control_layout(void)` | 返回 `sizeof(qzi_zimage_control_request)`；旧版 v1 库不导出此符号 |
| `size_t qzi_qwen21_edit_ex_layout(void)` | 返回 `sizeof(qzi_qwen21_edit_ex_request)`；第三个增量扩展，旧版 v1 库不导出此符号 |
| `size_t qzi_zimage_inpaint_layout(void)` | 返回 `sizeof(qzi_zimage_inpaint_request)`；第四个增量扩展，旧版 v1 库不导出此符号 |
| `qzi_status qzi_abi_layout(size_t *sizes, size_t count)` | 写 6 个 `sizeof()`；`count < 6` 返回 `QZI_ERR_INVALID_ARGUMENT` |
| `qzi_status qzi_register_backend(const qzi_backend *)` | 复制 vtable，运行时覆盖编译期默认 backend；`NULL` 恢复编译期默认值（reference 构建 → 参考后端，CUDA 构建 → 真实后端）；必须在 `qzi_engine_create` 之前调用 |
| `qzi_status qzi_engine_create(const qzi_engine_config *, qzi_engine **, char *error, size_t cap)` | 创建 engine；`config->abi_version` 必须等于 `QZI_ABI_VERSION` |
| `void qzi_engine_destroy(qzi_engine *)` | 释放 engine（调用前必须没有正在进行的生成） |
| `qzi_status qzi_engine_health(qzi_engine *, qzi_health *, char *, size_t)` | 状态/计数快照 |
| `qzi_status qzi_engine_unload(qzi_engine *, char *, size_t)` | 释放可重建的常驻状态（文本塔/缓存），**不卸载 DiT**；busy 时拒绝 |
| `qzi_status qzi_generate_qwen21(...)` | 同步生成 Qwen-Image 2.1 PNG |
| `qzi_status qzi_generate_zimage(...)` | 同步生成 Z-Image PNG |
| `qzi_status qzi_generate_qwen21_edit(...)` | 同步编辑参考图（img2img / masked inpainting），见下节；backend 无编辑槽时返回 `QZI_ERR_UNSUPPORTED` |
| `qzi_status qzi_generate_zimage_control(...)` | 同步用 ControlNet Union 生成 Z-Image，见下节；backend 无 control 槽时返回 `QZI_ERR_UNSUPPORTED`，真实后端未配置权重时返回 `QZI_ERR_NOT_CONFIGURED` |
| `qzi_status qzi_generate_qwen21_edit_ex(...)` | 同步高级编辑（Red Canvas / LanPaint / RGBA），见下节；backend 无 advanced-edit 槽时返回 `QZI_ERR_UNSUPPORTED` |
| `qzi_status qzi_generate_zimage_inpaint(...)` | 同步 Z-Image Union 2.x 遮罩重绘，见下节；backend 无 inpaint 槽时返回 `QZI_ERR_UNSUPPORTED`，真实后端未配置 Union 权重或加载了 16 通道 v1 权重时分别返回 `QZI_ERR_NOT_CONFIGURED` / `QZI_ERR_ENGINE` |
| `void qzi_image_release(qzi_engine *, qzi_image *)` | 释放 `data` 并把结构体清零；对已清零的结构体是幂等的 |

`qzi_register_backend` 引用的 `qzi_backend` 定义在 `native/include/qzi_backend.h`，现含 **11 个回调**（7 个原有 + 末尾四个可选的 `generate_qwen21_edit` / `generate_zimage_control` / `generate_qwen21_edit_ex` / `generate_zimage_inpaint`；槽位留 `NULL` 时对应能力返回 `QZI_ERR_UNSUPPORTED`，t2i 不受影响），由 `qzi_reference_backend()` 导出内置参考 vtable 供包装/对照使用。默认 vtable 由 `qzi.c` 的 `ensure_backend()` **编译期**选择：定义 `QZI_HAS_REAL_BACKEND` 时取 `qzi_real_backend()`（真实 C/CUDA 引擎），否则取 `qzi_reference_vtable`。这样 real 构建不依赖 DLL 构造函数顺序，reference 构建也完全不含引擎符号。

## 状态码

| 值 | 名称 | 含义 / `.NET` 映射 |
| --- | --- | --- |
| 0 | `QZI_OK` | 成功 |
| 1 | `QZI_ERR_INVALID_ARGUMENT` | 请求非法 / 校验失败 → `QziException(InvalidArgument)`（`.NET` 侧通常先抛 `ArgumentException`） |
| 2 | `QZI_ERR_NOT_CONFIGURED` | 模型包缺失 / engine 未创建 |
| 3 | `QZI_ERR_UNSUPPORTED` | tier、checkpoint 或特性不可用 |
| 4 | `QZI_ERR_BUSY` | 另一路生成正占用设备 |
| 5 | `QZI_ERR_CANCELLED` | 取消回调要求中止 → `OperationCanceledException` |
| 6 | `QZI_ERR_OUT_OF_MEMORY` | 分配失败 |
| 7 | `QZI_ERR_ENGINE` | 模型加载 / denoise / VAE / 设备错误 |
| 8 | `QZI_ERR_INTERNAL` | 内部错误 |

## 请求归一化与 LoRA 契约（v1）

`qzi_generate_qwen21()` 先用 `qzi_validate_qwen21()` 拒绝非法请求，再用 `qzi_normalize_qwen21()` 复制并归一化，最后把**归一化副本**交给 backend；调用方的结构体不变。参考后端与真实 CUDA 后端看到的有效值因此完全一致。

| 字段 | 合法输入 | 归一化 / 语义 |
| --- | --- | --- |
| `tier` | `NULL` / `"turbo"` / `"base"` | `NULL` → `"turbo"`；两个 tier 都允许 LoRA |
| `steps` | `0` 或 tier 对应合法值 | `0` → turbo `8` / base `40` |
| `guidance` | `0.0f` 或有限合法值 | `0.0f` → turbo `1.0f` / base `4.0f`；NaN / Inf 一律拒绝 |
| `lora` | `NULL` 或裸文件名 | ≤ `QZI_LORA_NAME_MAX` UTF-8 字节；禁止 `/`、`\`、`:`、`..`、`< 0x20`、`0x7f`；在 `qwen21_lora_dir` 中解析 |
| `lora_scale` | `0.0f` 或 `[0.05, 4]` 内的有限值 | `0.0f` 是「无 adapter / 默认强度」sentinel：有 `lora` 时归一化为 `1.0f`，没有 `lora` 时**是唯一允许的值** |

关键约束：

- **非法值一律拒绝，不静默 coerce**：`scale <= 0`（除 `0` sentinel 外）、`< 0.05`、`> 4`、NaN / Inf、无 `lora` 却带非 0 scale，全部返回 `QZI_ERR_INVALID_ARGUMENT`。
- `0.0f` 与 `1.0f` scale 在真实后端视为同一 adapter 状态，避免多 GB 的 DiT 重载。
- `lora` 切换、scale 改变或移除 adapter 都会触发 DiT 重载并上报一次 `"model load"` phase；真实后端在加载权重**之前**校验裸文件名与目录，合并失败会清空 DiT，不留半合并权重。
- `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR` 不在 ABI 里：它们通过 `QZI_CONFIG` 指定的配置文件或环境变量传入，native 侧由 `qzen_config_apply_environment()` 在 `qzen_config_load()` 之后应用（支持 `~` 展开）。`QziClient` 在 `qzi_engine_create()` 期间把代码级 `QziClientOptions` 值**作用域发布**并在返回后还原，所以每个 engine 快照自己的配置；创建完成后再改环境变量不影响已有 engine。
- `qzi_getenv()`（`native/include/qzi_env.h`）是**内部辅助函数，不是 ABI 导出**：Windows 上 CRT 的 `getenv()` 看不到托管侧通过 Win32 API 设置的值，所以 native 配置读取一律走它。ABI v1 的导出符号表与结构体布局没有因此改变。自行实现 backend 时若需要读 `QZI_*`，请从仓库源码编译 `native/src/qzi_env.c`（或照此使用 `GetEnvironmentVariableA`），不要裸用 `getenv()`。

## 图像编辑（ABI v1 增量扩展）

`qzi_generate_qwen21_edit()` 用 Qwen-Image 2.1 的**同一份 7B 权重**编辑一张参考图（img2img / 局部重绘），对齐 WanGP 的编辑语义。它是纯增量扩展：`qzi_qwen21_request` 与所有既有导出保持原布局，`qzi_abi_layout()` 仍是 6 个槽位，因此旧版 v1 库的 t2i 调用方完全不受影响。

```c
typedef struct qzi_qwen21_edit_request {
    qzi_qwen21_request base;             /* 普通 t2i 请求，规则不变 */
    const uint8_t     *reference_rgb;    /* 必填：[3][ref_w*ref_h] 平面 RGB8 */
    uint32_t           reference_width;
    uint32_t           reference_height;
    const uint8_t     *mask;             /* NULL 或 [ref_w*ref_h] 8-bit */
    float              strength;         /* 0 sentinel，或 0.05..1.0 */
    float              masking_strength; /* 0 sentinel，或 0.05..1.0 */
    uint32_t           flags;            /* 保留，必须为 0 */
} qzi_qwen21_edit_request;
```

| 字段 | 规则 / 语义 |
| --- | --- |
| `base` | 先走完整的 `qzi_validate_qwen21()`，t2i 的全部约束（尺寸、tier、steps、guidance、LoRA）都适用 |
| `reference_rgb` | **必填**，平面（planar）RGB8：先 `ref_w*ref_h` 字节的 R 平面，再 G、再 B；不是交错的 RGBRGB…。只在本次调用期间被读取，所有权归调用方（`.NET` 侧用 `GCHandle` 固定） |
| `reference_width` / `reference_height` | 每边 `1..4096`，总像素 `≤ 2048×2048`。会被**居中裁剪的 cover fit** 缩放到输出画布，因此参考图不必与 `base.width/height` 同尺寸 |
| `mask` | 可选，每个参考像素 1 字节。**白色（255）= 按 prompt 重绘，黑色（0）= 保持参考轨迹**；与参考图用同一套 cover fit 缩放，再下采样到 1/16 latent 网格并以 `> 64` 二值化（WanGP `latent_mask` 语义） |
| `strength` | WanGP 的 `denoising_strength`。`1.0` ≈ 从纯噪声开始（等同 t2i），越小越贴近参考图。`0.0f` 是「用引擎默认 `0.75`」的 sentinel，其余合法值 `0.05..1.0` 且必须有限 |
| `masking_strength` | 剩余调度中把未遮罩 latent 单元恢复到参考轨迹的比例，`0.05..1.0`；`0.0f` 是「默认 `1.0`」的 sentinel。**必须有 `mask`**，否则会被静默丢弃，native 直接拒绝 |
| `flags` | 必须为 `0` |

`qzi_generate_qwen21_edit()` 与 t2i 共用同一条路径：先 `qzi_validate_qwen21_edit()`（复用 `qzi_validate_qwen21()` 再追加编辑约束），再 `qzi_normalize_qwen21_edit()` 复制并解析两个 sentinel，最后把**归一化副本**交给 backend；调用方结构体不变。归一化规则：

| 输入 sentinel | 归一化结果 |
| --- | --- |
| `strength == 0.0f` | `0.75f` |
| `masking_strength == 0.0f` 且有 `mask` | `1.0f` |
| `masking_strength` 且**没有** `mask` | 必须为 `0.0f`，归一化后仍是 `0.0f`（不做遮罩恢复） |

参考后端与真实 CUDA 后端因此看到完全相同的有效值：参考后端把 `strength` 换算成 WanGP 的 `first_step = int(steps * (1 - strength))`，只跑剩余步数，并把参考图 / mask / 两个 strength 纳入确定性 PNG 哈希；真实后端先做 cover fit，再用 `qzen_qwen21_vae_encode()` 把参考图编码成 latents（上报 `"image encode"` phase），把 mask 收缩到 latent 网格，最后由 `qzen_qwen21_pipeline_run_ex()` 以 `qzen_qwen21_edit_start()` 换算出的起点步从该轨迹起跑。**图像编辑唯一额外需要的是把参考图 VAE 编码成 latents**，文本编码、DiT 去噪、VAE 解码与 PNG 编码都复用 t2i 的常驻管线。

调用前的能力探测（`.NET` 侧由 `AbiGuard` 完成，`QziClient.SupportsImageEditing` 暴露结果）：

```c
/* 这三个符号都是增量的：老版本 v1 库不导出它们（has_export 为占位伪代码）。 */
if (!has_export("qzi_abi_features") ||
    !(qzi_abi_features() & QZI_FEATURE_QWEN21_EDIT) ||
    !has_export("qzi_qwen21_edit_layout") ||
    qzi_qwen21_edit_layout() != sizeof(qzi_qwen21_edit_request))
    return QZI_ERR_UNSUPPORTED;
```

backend 侧对应 `qzi_backend.generate_qwen21_edit`，是 vtable 倒数第二个槽位（保持前 7 个槽位偏移不变）。留空为 `NULL` 时，t2i 照常工作，`qzi_generate_qwen21_edit()` 返回 `QZI_ERR_UNSUPPORTED`。单飞、进度、取消、`QZI_ERR_BUSY`、错误缓冲与 `qzi_image_release()` 的所有权规则与 `qzi_generate_qwen21()` 完全一致。

## Z-Image ControlNet Union（ABI v1 增量扩展）

`qzi_generate_zimage_control()` 用 **Z-Image Turbo Fun ControlNet Union 2.1** 的一张控制图条件化生成，对齐 WanGP 的 `z_image_control2_1` 架构。同样是纯增量扩展：`qzi_zimage_request` 与所有既有导出保持原布局，`qzi_abi_layout()` 仍是 6 个槽位。

```c
typedef struct qzi_zimage_control_request {
    qzi_zimage_request base;              /* 普通 t2i 请求，规则不变 */
    const uint8_t     *control_rgb;       /* 必填：[3][cw*ch] 平面 RGB8 */
    uint32_t           control_width;
    uint32_t           control_height;
    const char        *control_type;      /* NULL/auto/pose/canny/depth/scribble/raw */
    float              control_scale;     /* 0 sentinel，或 0.05..2.0 */
    uint32_t           flags;             /* 保留，必须为 0 */
} qzi_zimage_control_request;
```

| 字段 | 规则 / 语义 |
| --- | --- |
| `base` | 先走完整的 `qzi_validate_zimage()`，t2i 的全部约束（宽高为 16 的倍数、≤ 2048²、checkpoint）都适用 |
| `control_rgb` | **必填**，平面 RGB8（R/G/B 三个连续平面），只在本次调用期间被读取，所有权归调用方。会做**居中裁剪的 cover fit** 缩放到输出画布 |
| `control_width` / `control_height` | 每边 `1..4096`，总像素 `≤ 2048×2048` |
| `control_type` | 大小写不敏感，归一化成 `auto`/`pose`/`canny`/`depth`/`scribble`/`raw` 之一；`NULL` → `auto`。**仅为描述性元数据**，本库不运行姿态 / 边缘 / 深度估计器，对齐 WanGP 的 `guide_preprocessing = "V"`（Use Z-Image Raw Format）：调用方需自行提供已预处理的控制图。ABI 层面这是**永久的 raw 契约**，不会新增预处理参数；`QwenZImage.NET.WebUI` 在应用层用 `POST /api/preprocess` 生成该栅格后作为普通 `control_rgb` 传入 |
| `control_scale` | WanGP 的 `control_context_scale`。`0.0f` 是「用 Union 2.1 默认 `0.65`」的 sentinel，其余合法值 `0.05..2.0` 且必须有限；v1/v2 权重用 `0.75` 时可显式传入 |
| `flags` | 必须为 `0` |

`qzi_generate_zimage_control()` 先 `qzi_validate_zimage_control()`（复用 `qzi_validate_zimage()` 再追加控制图约束），再 `qzi_normalize_zimage_control()` 复制并解析 sentinel / 规范 `control_type`，最后把**归一化副本**交给 backend；调用方结构体不变。参考后端和真实 CUDA 后端因此看到相同有效值。

**真实后端的推理链**（`qzi_engine_backend.c` → `engine/src/qzen_zimage_pipeline.c`）：

1. 懒加载 Union 2.1 权重（一个独立 `.safetensors`，路径由 `zimage_control_dir` / `QZI_ZIMAGE_CONTROL_DIR` 提供），上报 `"model load"`；未配置路径返回 `QZI_ERR_NOT_CONFIGURED`。
2. 控制图 cover fit → `qzen_zvae_encode()` 编码成 16 通道 1/8 归一化 latent，上报 `"image encode"`。
3. 拼成 **33 通道 control context** = `control_latent(16) | keep_mask(1, 全 0) | inpaint_latent(16, 全 0)`，与 WanGP `pipeline_z_image.py` 无 inpaint mask 时的布局一致，再按 DiT 的 2×2 patch 规则 patchify 后过 `control_all_x_embedder.2-1`。
4. 每个去噪步用当前 `x_t` / `tau` 重算 hint：2 个独立 control noise refiner + 15 个 control layer（对应主 DiT 层 `0,2,…,28`），每层 `after_proj` 产生一条 skip。
5. 主 DiT forward 在对应 noise refiner / 偶数 main layer **之后**执行 `hidden_states += control_scale * skip`（`qzen_gpu_scale_add_bf16`，CPU 参考路径同样注入）。普通 t2i 走 `qzen_zpipe_denoise()`，hints 为 `NULL`，行为与字节级输出完全不变。

调用前的能力探测（`.NET` 侧由 `AbiGuard` 完成，`QziClient.SupportsControlNet` 暴露结果）：

```c
if (!has_export("qzi_abi_features") ||
    !(qzi_abi_features() & QZI_FEATURE_ZIMAGE_CONTROL) ||
    !has_export("qzi_zimage_control_layout") ||
    qzi_zimage_control_layout() != sizeof(qzi_zimage_control_request))
    return QZI_ERR_UNSUPPORTED;
```

backend 侧对应 `qzi_backend.generate_zimage_control`，是 vtable **最后一个**槽位；留 `NULL` 时 `qzi_generate_zimage_control()` 返回 `QZI_ERR_UNSUPPORTED`。参考后端把控制图像素、规范后的 `control_type`、`control_scale` 与尺寸纳入确定性 PNG 哈希，并像真实后端一样多上报一个 `"image encode"` phase，**不产出可用的控制效果**。

已知边界：仅支持 Union **v2 / v2.1**（embedder `in_dim = 33`、15 control layers + 2 独立 refiner），v1（`in_dim = 16`）会在权重加载时明确拒绝；不提供 canny / pose / depth / scribble 预处理器；控制图与编辑参考图一样只接受 .NET 侧解码出的 8 位非隔行 PNG。Union 2.1 权重约 3.7 GiB + 需上传到 GPU，与常驻 DiT / 文本塔叠加后接近 24 GiB 卡的容量上限，上传失败会回退到 CPU hint 路径（正确但明显更慢）。

## 高级图像编辑（ABI v1 第三增量扩展）

`qzi_generate_qwen21_edit_ex()` 在普通编辑请求之上暴露 WanGP `qwen21_handler` 的三项无需新权重的能力：Red Canvas 外扩、LanPaint、RGBA 输出。它同样是**纯增量**：把 `qzi_qwen21_edit_request` 作为首成员嵌进新结构体，既有结构体 / 导出 / 6 个 layout 槽位一律不动，旧调用方继续用 `qzi_generate_qwen21_edit()`。

```c
typedef struct qzi_qwen21_edit_ex_request {
    qzi_qwen21_edit_request base;  /* 普通一参考图编辑请求，规则不变 */
    uint32_t margin_top;           /* 输出画布上边新增像素，32 的倍数 */
    uint32_t margin_bottom;
    uint32_t margin_left;
    uint32_t margin_right;
    uint32_t model_mode;           /* QZI_EDIT_MODE_MASKED(0) / LANPAINT_2|5|10|15(2|3|4|5) */
    uint32_t rgba;                 /* 0 = RGB PNG, 1 = RGBA PNG */
    uint32_t flags;                /* 保留，必须为 0 */
} qzi_qwen21_edit_ex_request;
```

| 字段 | 规则 / 语义 |
| --- | --- |
| `base` | 先走完整的 `qzi_validate_qwen21_edit()`（含 t2i 全部约束）；`base.base.width/height` 在这里是**含边距的最终画布** |
| `margin_*` | 四边输出画布像素，必须是 `QZI_OUTPAINT_ALIGNMENT` = **32** 的倍数；扣掉左右 / 上下边距后内部源矩形必须仍 ≥ `32×32`；全 0 = 不外扩 |
| `model_mode` | 白名单 `QZI_EDIT_MODE_MASKED (0)` / `QZI_EDIT_MODE_LANPAINT_2 (2)` / `_5 (3)` / `_10 (4)` / `_15 (5)`。数值是 WanGP 下拉框索引而非迭代次数。非 0 时**必须带 `base.mask`**、**不得有任何非零边距**（WanGP 对 instruction outpainting 禁用 LanPaint） |
| `rgba` | `0` / `1`；`1` 返回 PNG colour type 6，保留 VAE decoder `conv_out` 的第 4 通道 |
| `flags` | 必须为 `0` |

`qzi_validate_qwen21_edit_ex()` 先复用 `qzi_validate_qwen21_edit()` 再追加以上约束；`qzi_normalize_qwen21_edit_ex()` 复制请求并解析内嵌 sentinel，LanPaint 模式下把 `base.strength` 与 `base.masking_strength` 都置为 `1.0`，因此参考后端与真实 CUDA 后端看到一致的有效值。调用方结构体不变。非法值（边距未对齐 / 源矩形过小、未知 `model_mode`、LanPaint 无 mask 或与外扩并用、`rgba > 1`、`flags != 0`）一律返回 `QZI_ERR_INVALID_ARGUMENT`，不静默 coerce。

能力探测链比第二个增量多一层依赖（`.NET` 侧由 `AbiGuard.Qwen21EditAdvSupported` 完成、`QziClient.SupportsAdvancedEditing` 暴露）：

```c
if (!has_export("qzi_abi_features") ||
    !(qzi_abi_features() & QZI_FEATURE_QWEN21_EDIT_ADV) ||
    !(qzi_abi_features() & QZI_FEATURE_QWEN21_EDIT) ||
    !has_export("qzi_qwen21_edit_ex_layout") ||
    qzi_qwen21_edit_ex_layout() != sizeof(qzi_qwen21_edit_ex_request) ||
    !has_export("qzi_generate_qwen21_edit_ex"))
    return QZI_ERR_UNSUPPORTED;
```

普通编辑能力是前置条件，因为高级请求内嵌了它。`qzi_qwen21_edit_ex_layout()` **不占用** `qzi_abi_layout()` 的 6 个槽位，与前面两个 `*_layout()` 同理。backend 侧对应 `qzi_backend.generate_qwen21_edit_ex`（第四个增量 `generate_zimage_inpaint` 追加在它之后，现共 11 个回调、末尾四个可选槽顺序固定）；留 `NULL` 时 `qzi_generate_qwen21_edit_ex()` 返回 `QZI_ERR_UNSUPPORTED`，四条既有生成路径不受影响。单飞、进度、取消、错误缓冲与 `qzi_image_release()` 的所有权规则与 `qzi_generate_qwen21_edit()` 完全一致。

参考后端把边距 / `model_mode` / `rgba` 纳入确定性哈希，并在 `rgba != 0` 时用 `qzi_png_encode_rgba8()` 产出 colour type 6 的 PNG，使无 GPU 的 `ctest` / xUnit 能验证参数贯通与输出格式；真实引擎的三段实现见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。

## Z-Image 遮罩重绘（ABI v1 第四增量扩展）

`qzi_generate_zimage_inpaint()` 补齐 WanGP `models/z_image/pipeline_z_image.py` 中 `control_in_dim == 33` 分支的**带 mask 路径**：普通 ControlNet 只填 33 通道 context 的前 16 个平面，另 17 个恒为 0（等价于 WanGP `inpaint_mask is None`）；本增量把它们真正填上。仍然是**纯增量**：把 `qzi_zimage_control_request` 作为首成员嵌进新结构体，既有结构体 / 导出 / 6 个 layout 槽位一律不动。

```c
typedef struct qzi_zimage_inpaint_request {
    qzi_zimage_control_request base;  /* 普通 ControlNet 请求，规则不变 */
    const uint8_t *mask;              /* 必填：[mw*mh] 单通道 8 位，白 = 重绘 */
    uint32_t       mask_width;
    uint32_t       mask_height;
    uint32_t       flags;             /* 保留，必须为 0 */
} qzi_zimage_inpaint_request;
```

| 字段 | 规则 / 语义 |
| --- | --- |
| `base` | 先走完整的 `qzi_validate_zimage_control()`（含 t2i 全部约束、`control_rgb`、`control_type`、`control_scale`、`flags`） |
| `mask` | **必填**，单通道 8 位平面。WHITE(255) = 按 prompt 重绘，BLACK(0) = 保留控制图。中间值按线性比例参与 |
| `mask_width` / `mask_height` | 每边 `1..4096`，总像素 `≤ 2048×2048`。与 `base.control_width/height` **可以不同**：两者各自独立做居中裁剪 cover fit 到输出画布 |
| `flags` | 必须为 `0` |

`qzi_validate_zimage_inpaint()` 先复用控制请求校验，再检查 mask 指针、尺寸与 `flags`；`qzi_normalize_zimage_inpaint()` 只归一化内嵌的 `control_type` / `control_scale` sentinel，mask 平面与尺寸原样透传。非法值一律 `QZI_ERR_INVALID_ARGUMENT`，不静默 coerce。

真实后端按 WanGP 逐值对齐的顺序构造 context：

```
inpaint_image  = control_image * (1 - mask)      # 白区抹成黑
inpaint_latent = vae.encode(inpaint_image)       # 第二次 VAE 编码
control_latent = vae.encode(control_image)       # 与原 ControlNet 路径相同
keep_mask      = 1 - mask                        # 收缩到 1/8 latent 网格
context        = [control_latent(16) | keep_mask(1) | inpaint_latent(16)]
```

`qzen_zpipe_control_context_ex()` 是 `qzen_zpipe_control_context()` 的超集：两个 mask 参数都传 `NULL` 时逐位复现原行为（普通 ControlNet / t2i 不受影响），同时提供 `keep_mask` 与 `inpaint_latent` 时填最后 17 个平面；只给其中一个、或对 16 通道 v1 权重做 inpaint，都会在引擎层明确报错。mask 收缩用 8×8 面积均值得到 `[0,1]` 软 keep 平面（WanGP 用 `F.interpolate(..., mode='nearest')`，硬笔刷边界上最多相差 8 像素，软边 mask 反而更平滑）。

能力探测链（`.NET` 侧由 `AbiGuard.ZImageInpaintSupported` 完成、`QziClient.SupportsInpaint` 暴露）**前置依赖 ControlNet 能力**，因为请求内嵌了它：

```c
if (!has_export("qzi_abi_features") ||
    !(qzi_abi_features() & QZI_FEATURE_ZIMAGE_INPAINT) ||
    !(qzi_abi_features() & QZI_FEATURE_ZIMAGE_CONTROL) ||
    !has_export("qzi_zimage_inpaint_layout") ||
    qzi_zimage_inpaint_layout() != sizeof(qzi_zimage_inpaint_request) ||
    !has_export("qzi_generate_zimage_inpaint"))
    return QZI_ERR_UNSUPPORTED;
```

`qzi_zimage_inpaint_layout()` 同样**不占用** `qzi_abi_layout()` 的 6 个槽位。backend 侧对应 `qzi_backend.generate_zimage_inpaint`，是 vtable **最后一个**槽位（现共 11 个回调，末尾四个可选槽顺序固定）；留 `NULL` 时 `qzi_generate_zimage_inpaint()` 返回 `QZI_ERR_UNSUPPORTED`，五条既有生成路径不受影响。单飞、进度、取消、错误缓冲与 `qzi_image_release()` 的所有权规则与 `qzi_generate_zimage_control()` 完全一致；`control_rgb` 与 `mask` 都只在调用期间读取，所有权始终归调用方。

参考后端把 mask 内容、mask 尺寸与「这是 inpaint 请求」纳入确定性哈希，并上报**两个** `"image encode"` 阶段（对应真实后端的两次 VAE 编码），使无 GPU 的 `ctest` / xUnit 能验证参数贯通；真实引擎实现见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。

已知边界：仅 Union **v2 / v2.1**（33 通道）支持，v1（16 通道）在权重加载时就会被拒绝；`control_type` 仍是描述性元数据，不运行预处理器；WanGP 的 `input_ref_images` 替换 `control_latent` 分支（其 UI 选项上游已禁用）不在本库范围内。

## 典型调用序列

```c
char error[QZI_ERROR_CAPACITY] = {0};
size_t sizes[QZI_LAYOUT_SLOTS];
qzi_engine *engine = NULL;

/* 0. 可选：运行时覆盖编译期默认 backend（必须在 create 之前）。
 *    本仓库的 CUDA 构建已默认选中真实后端，这一步通常不需要。 */
qzi_register_backend(&my_real_backend);

/* 1. 版本 + 布局自检 */
if (qzi_abi_version() != QZI_ABI_VERSION) abort();
if (qzi_abi_layout(sizes, QZI_LAYOUT_SLOTS) != QZI_OK) abort();

/* 2. 创建 engine；目录可为 NULL（参考后端不读权重；真实后端需要
 *    QwenZImage.config.json 或这里传入的目录） */
qzi_engine_config config = {0};
config.abi_version = QZI_ABI_VERSION;
config.qwen21_dir = "/models/qwen21";
config.zimage_dir = "/models/zimage";
config.cache_dir  = "/tmp/qzi-cache";
config.device_id  = -1;
config.thread_count = 0;
if (qzi_engine_create(&config, &engine, error, sizeof(error)) != QZI_OK) goto fail;

/* 3. 生成；callbacks 可为 NULL */
qzi_qwen21_request req = {0};
req.prompt = "a red fox in snow";
req.width = 1024; req.height = 1024;
req.steps = 8; req.guidance = 1.0f; req.lora_scale = 0.0f; req.seed = 42;
/* lora_scale == 0 表示「引擎默认 1.0」；没有 req.lora 时这是唯一允许的值。 */

qzi_callbacks cb = { on_progress, should_cancel, user_data };
qzi_image image = {0};
qzi_status s = qzi_generate_qwen21(engine, &req, &cb, &image, error, sizeof(error));
if (s == QZI_OK) {
    fwrite(image.data, 1, image.size, out_file);  /* data 是完整 PNG */
    qzi_image_release(engine, &image);            /* 恰好一次 */
} else {
    fprintf(stderr, "qzi: %s\n", error);
}

/* 4. 收尾 */
qzi_engine_destroy(engine);
```

图像编辑只多一组字段，调用序列完全相同（省略了能力探测与错误检查）：

```c
uint8_t reference[3 * ref_w * ref_h];   /* 平面 RGB8，调用方拥有 */
uint8_t mask[ref_w * ref_h];            /* 可省：不传 mask 就是全图编辑 */

qzi_qwen21_edit_request edit = {0};
edit.base.prompt = "replace the sky with an aurora";
edit.base.width = 1024; edit.base.height = 1024;
edit.base.steps = 8; edit.base.guidance = 1.0f; edit.base.seed = 42;
edit.reference_rgb = reference;
edit.reference_width = ref_w; edit.reference_height = ref_h;
edit.mask = mask;              /* NULL 表示整图按 strength 编辑 */
edit.strength = 0.6f;          /* 0 = 默认 0.75 */
edit.masking_strength = 1.0f;  /* 仅当有 mask；0 = 默认 1.0 */

qzi_image result = {0};
qzi_status s = qzi_generate_qwen21_edit(engine, &edit, &cb, &result,
                                        error, sizeof(error));
if (s == QZI_OK) qzi_image_release(engine, &result);
```

## 内存所有权速查

| 对象 | 分配方 | 释放方 | 时机 |
| --- | --- | --- | --- |
| 请求里的 UTF-8 字符串 / 结构体 | 调用方 | 调用方 | 函数调用返回后即可 |
| `qzi_qwen21_edit_request.reference_rgb` / `.mask` | 调用方 | 调用方 | 只在同步调用期间被读取（`.NET` 侧 `GCHandle` 固定在 `finally` 中释放），返回后即可释放 |
| `qzi_callbacks.user` | 调用方 | 调用方 | 生成结束后 |
| `qzi_image.data` | 引擎 / backend | `qzi_image_release()` | 读取完 PNG 后立刻 |
| `error` 缓冲区 | 调用方 | 调用方 | 栈上分配即可 |
| `qzi_engine *` | `qzi_engine_create` | `qzi_engine_destroy` | 生命周期结束 |
| backend 的 `void *context` | backend `init` | backend `shutdown` | engine create/destroy |
| `qzi_status_text` / progress `phase` 返回值 | 引擎静态存储 | **不释放** | — |

`.NET` 侧的做法：成功时把 native buffer 一次 `memcpy` 进 `ArrayPool<byte>` 数组，立刻 `qzi_image_release`；此后没有任何跨边界生命周期。`WriteTo` 路径则用 `UnmanagedMemoryStream` 直接读 native buffer 写目标流，同样在 `finally` 中释放。

## 单飞与取消

- ABI 层用互斥锁保护 `busy` 标记；第二个并发生成立即返回 `QZI_ERR_BUSY`（带错误消息），不会阻塞。
- `should_cancel` 在阶段之间与每个去噪步之间被轮询；返回非零 ⇒ backend 尽快停止并返回 `QZI_ERR_CANCELLED`。
- 校验失败与 busy 都会增加 `qzi_health.rejected`；一次成功生成增加 `served`。
- `qzi_engine_unload()` 在 busy 时被拒绝；它只清理可重建状态，避免分钟级的 DiT 重载。

## 版本演进规则

只要以下任一项变化，就**必须**提升 `QZI_ABI_VERSION`：既有结构体字段增删/顺序变化、既有函数签名变化、状态码语义变化、缓冲区所有权变化、**改变既有** phase 字面量的含义。新增可选能力应优先用能力位或新增独立导出函数表达，并保持旧结构体布局不变；`qzi_qwen21_edit_request` / `qzi_zimage_control_request` / `qzi_qwen21_edit_ex_request` / `qzi_zimage_inpaint_request` / `qzi_abi_features()` / 四个 `*_layout()` / 四个 `qzi_generate_*_edit|control|edit_ex|inpaint()` 都是这种增量形式，因此图像编辑、ControlNet、高级编辑与遮罩重绘仍留在 v1。

对 phase 字面量的例外说明：**新增**一个仅伴随新增能力出现的可选字面量（本次的 `"image encode"`）属于 additive，不必升版本——旧库根本不会上报它，托管侧对未知 phase 一律降级为 `GenerationPhase.Unknown`，不会抛异常也不会破坏既有枚举值。但**改动或删除**任何既有字面量、或改变其 `completed/total` 语义，仍必须升版本。inpaint 只是把同一个 `"image encode"` 阶段上报两次（`(0,2)/(1,2)` 与 `(1,2)/(2,2)`），普通编辑 / ControlNet 仍是 `(0,1)/(1,1)`，没有改变既有语义。`QwenZImage.NET` 的 `AbiGuard` 会同时校验版本号、6 个结构体尺寸，以及（若导出存在）四个扩展结构体尺寸分别匹配 `qzi_qwen21_edit_layout()` / `qzi_zimage_control_layout()` / `qzi_qwen21_edit_ex_layout()` / `qzi_zimage_inpaint_layout()`；第三个能力要求普通编辑能力已通过探测，第四个要求 ControlNet 能力已通过探测。因此不兼容的 native 库会在构造 `QziClient` 时立刻报错。

## 两个内置 backend

- **参考后端**：`native/src/qzi_stub.c` 实现的 `qzi-reference`，无需 GPU/权重/网络，按**所有会改变引擎输出的字段**（prompt、seed、尺寸、模型标签、tier、lora、lora_scale、steps、guidance、negative_prompt，编辑时另加参考图 / mask / strength / masking_strength / 参考尺寸，ControlNet 时另加控制图 / control_type / control_scale / 控制图尺寸，高级编辑时另加四边距 / model_mode / rgba，inpaint 时另加 mask 内容 / mask 尺寸）生成确定性噪点图，再由 `native/src/qzi_png.c` 编码为 PNG（stored DEFLATE 块，无 zlib 依赖）；`rgba != 0` 时走 `qzi_png_encode_rgba8()` 产出 colour type 6 的 RGBA PNG，供无 GPU 测试验证输出格式。它跟踪模型 / tier / adapter / scale / control 变化并像真实后端一样上报 `"model load"`，还跑完整的阶段（编辑 / ControlNet / 高级编辑时多一个 `"image encode"`，inpaint 时多两个）与 `dit` 步；编辑请求按 WanGP 的 `first_step` 规则只跑剩余步数，高级编辑与 inpaint 会给调用打上独立标记以便 C smoke 区分各条路径。`qzi_health.reference_backend == 1`。用途：CI、托管单测、C smoke test、ABI 对照与参数贯通验证。**它不是图像模型，输出不是真实生成结果。**
- **真实后端**：`native/src/qzi_engine_backend.c` + `engine/` 的 C/CUDA 引擎，仅在 `-DQZI_WITH_CUDA_ENGINE=ON` 时链入。做文本编码、DiT 去噪与 VAE 解码；编辑时额外把参考图 VAE 编码为 latents 并按 mask 恢复轨迹，ControlNet 时编码控制图、懒加载 Union 2.1 权重并逐步注入 hint，inpaint 时再编码一次遮罩后的控制图并填 `keep_mask` / `inpaint_latent` 两个 context 段；PNG 直接在内存里编码，`qzi_health.reference_backend == 0`。需要 CUDA Toolkit、匹配的 SM 架构以及自备的模型权重。

`native/tests/qzi_abi_smoke.c` 会根据 `health.reference_backend` 分支：reference 构建跑完整生成契约（PNG 签名、确定性、release、LoRA / 编辑 / 控制 / 高级编辑 / inpaint 参数的字节差异、RGB 与 RGBA 的 PNG colour type、`"model load"` phase、inpaint 的两个 `"image encode"` 阶段与常驻权重复用），两种构建都跑 ABI 布局、四个能力位（含 `qzi_abi_features() == 0xF`）/ 四个扩展结构体尺寸 / 编辑 / 控制 / 高级编辑 / inpaint 参数边界（边距对齐、LanPaint 与外扩互斥、RGBA 取值、mask 必填 / 尺寸 / flags / 继承的 ControlNet 规则）与 legacy backend（四个可选槽为 `NULL` 时返回 `QZI_ERR_UNSUPPORTED`）、LoRA 裸文件名 / 长度 / scale 边界与 fp4 大小写校验，真实构建**不加载多 GB 权重**。详见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。
