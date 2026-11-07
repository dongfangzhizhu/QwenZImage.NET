# QwenZImage.NET Native C ABI (v1)

`native/include/qzi.h` 是**唯一**的托管/原生契约。任何实现并导出其中 `qzi_*` 函数的共享库都可以被 `QwenZImage.NET` 加载，无论内部是确定性参考实现还是真实 CUDA 引擎。本仓库统一产出 `QwenZImageNative.dll` / `libQwenZImageNative.so` 两个 flavour：默认链接 `qzi_stub.c`（噪点，CI 用），`-DQZI_WITH_CUDA_ENGINE=ON` 链接 `qzi_engine_backend.c` + `engine/`（真实推理）。两者 ABI 完全相同，托管侧一行不用改。本文件说明调用规则、内存所有权与版本策略；参数语义见 [PARAMETERS.md](PARAMETERS.md)，后端实现与构建见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。

## 设计原则

1. **纯 C 链接**：`extern "C"`，无 C++ 异常跨边界、无 STL、无平台特定运行时类型。
2. **固定 POD 布局**：所有结构体都是 POD；`qzi_abi_layout()` 报告 `sizeof()`，托管侧启动时逐一断言（见 `AbiGuard`），packing 不匹配立即抛异常，而不是静默踩坏内存。
3. **输入内存归调用方**：字符串为 NUL 结尾 UTF-8，仅在调用期间被读取；调用返回后引擎不得保留指针。
4. **输出内存归引擎**：`qzi_image.data` 指向完整 PNG 文件，直到 `qzi_image_release()` 被调用；每个成功的 image **必须且只能释放一次**。
5. **错误通过 caller buffer 返回**：所有可失败函数返回 `qzi_status`，失败时把可读的 NUL 结尾消息写入调用方的 `error` 缓冲区（建议 `QZI_ERROR_CAPACITY = 512`）。不使用 `errno`、不抛异常、不返回动态分配的错误字符串。
6. **同步单飞**：`qzi_generate_*` 是同步长调用（可能数分钟）。同一 engine 上并发第二次生成返回 `QZI_ERR_BUSY`，不会排队等待。上层用 `QziGenerationQueue` 串行化。
7. **进度 phase 是静态字面量**：`"model load"` / `"text"` / `"dit"` / `"vae"` / `"encode"`，由引擎拥有，不得释放。回调在生成线程上执行，必须快速返回且不得重入引擎。

## ABI 版本与布局

| 常量 | 值 |
| --- | --- |
| `QZI_ABI_VERSION` | `1` |
| `QZI_ERROR_CAPACITY` | `512` |
| `QZI_LAYOUT_SLOTS` | `6` |
| `QZI_LORA_NAME_MAX` | `255`（LoRA 裸文件名的最大 UTF-8 字节数，不含 NUL） |

> 新增 `QZI_LORA_NAME_MAX` 只影响输入校验，不改变任何结构体布局或函数签名，因此 **ABI 版本仍是 v1**。

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

托管侧 `LibraryImport` 的调用约定由源生成器按平台处理；`qzi_progress_fn` / `qzi_cancel_fn` 反向回调在 `.NET` 侧用 `static` + `[UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]` 实现，不分配 delegate。

## 导出函数

| 函数 | 作用 |
| --- | --- |
| `uint32_t qzi_abi_version(void)` | 返回 `QZI_ABI_VERSION` |
| `const char *qzi_status_text(qzi_status)` | 静态状态描述，不释放 |
| `qzi_status qzi_abi_layout(size_t *sizes, size_t count)` | 写 6 个 `sizeof()`；`count < 6` 返回 `QZI_ERR_INVALID_ARGUMENT` |
| `qzi_status qzi_register_backend(const qzi_backend *)` | 复制 vtable，运行时覆盖编译期默认 backend；`NULL` 恢复编译期默认值（reference 构建 → 参考后端，CUDA 构建 → 真实后端）；必须在 `qzi_engine_create` 之前调用 |
| `qzi_status qzi_engine_create(const qzi_engine_config *, qzi_engine **, char *error, size_t cap)` | 创建 engine；`config->abi_version` 必须等于 `QZI_ABI_VERSION` |
| `void qzi_engine_destroy(qzi_engine *)` | 释放 engine（调用前必须没有正在进行的生成） |
| `qzi_status qzi_engine_health(qzi_engine *, qzi_health *, char *, size_t)` | 状态/计数快照 |
| `qzi_status qzi_engine_unload(qzi_engine *, char *, size_t)` | 释放可重建的常驻状态（文本塔/缓存），**不卸载 DiT**；busy 时拒绝 |
| `qzi_status qzi_generate_qwen21(...)` | 同步生成 Qwen-Image 2.1 PNG |
| `qzi_status qzi_generate_zimage(...)` | 同步生成 Z-Image PNG |
| `void qzi_image_release(qzi_engine *, qzi_image *)` | 释放 `data` 并把结构体清零；对已清零的结构体是幂等的 |

`qzi_register_backend` 引用的 `qzi_backend` 定义在 `native/include/qzi_backend.h`，由 `qzi_reference_backend()` 导出内置参考 vtable 供包装/对照使用。默认 vtable 由 `qzi.c` 的 `ensure_backend()` **编译期**选择：定义 `QZI_HAS_REAL_BACKEND` 时取 `qzi_real_backend()`（真实 C/CUDA 引擎），否则取 `qzi_reference_vtable`。这样 real 构建不依赖 DLL 构造函数顺序，reference 构建也完全不含引擎符号。

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
- `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR` 不在 ABI 里：它们通过 `QZI_CONFIG` 指定的配置文件或环境变量传入，native 侧由 `qzen_config_apply_environment()` 在 `qzen_config_load()` 之后应用（支持 `~` 展开）。`QziClient` 会把代码级 `QziClientOptions.Qwen21LoraDirectory` / `ZImageNunchakuDirectory` 发布成这两个变量。

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

## 内存所有权速查

| 对象 | 分配方 | 释放方 | 时机 |
| --- | --- | --- | --- |
| 请求里的 UTF-8 字符串 / 结构体 | 调用方 | 调用方 | 函数调用返回后即可 |
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

只要以下任一项变化，就**必须**提升 `QZI_ABI_VERSION`：结构体字段增删/顺序变化、函数签名变化、状态码语义变化、缓冲区所有权变化、phase 字面量集合变化。新增可选能力应优先用 `flags` 位或新增独立导出函数表达，并保持旧结构体布局不变。`QwenZImage.NET` 的 `AbiGuard` 会同时校验版本号和 6 个结构体尺寸，因此不兼容的 native 库会在构造 `QziClient` 时立刻报错。

## 两个内置 backend

- **参考后端**：`native/src/qzi_stub.c` 实现的 `qzi-reference`，无需 GPU/权重/网络，按**所有会改变引擎输出的字段**（prompt、seed、尺寸、模型标签、tier、lora、lora_scale、steps、guidance、negative_prompt）生成确定性 RGB 噪点图，再由 `native/src/qzi_png.c` 编码为 PNG（stored DEFLATE 块，无 zlib 依赖）。它跟踪模型 / tier / adapter / scale 变化并像真实后端一样上报 `"model load"`，还跑完整的 5 个阶段与 `dit` 步，`qzi_health.reference_backend == 1`。用途：CI、托管单测、C smoke test、ABI 对照与参数贯通验证。**它不是图像模型，输出不是真实生成结果。**
- **真实后端**：`native/src/qzi_engine_backend.c` + `engine/` 的 C/CUDA 引擎，仅在 `-DQZI_WITH_CUDA_ENGINE=ON` 时链入。做文本编码、DiT 去噪与 VAE 解码，PNG 直接在内存里编码，`qzi_health.reference_backend == 0`。需要 CUDA Toolkit、匹配的 SM 架构以及自备的模型权重。

`native/tests/qzi_abi_smoke.c` 会根据 `health.reference_backend` 分支：reference 构建跑完整生成契约（PNG 签名、确定性、release、LoRA 参数字节差异、`"model load"` phase），两种构建都跑 ABI 布局、LoRA 裸文件名 / 长度 / scale 边界与 fp4 大小写校验，真实构建**不加载多 GB 权重**。详见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。
