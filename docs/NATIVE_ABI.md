# QwenZImage.NET Native C ABI (v1)

`native/include/qzi.h` 是**唯一**的托管/原生契约。任何实现并导出其中 `qzi_*` 函数的共享库都可以被 `QwenZImage.NET` 加载，无论内部是 CPU 参考实现还是 CUDA 引擎。本文件说明调用规则、内存所有权与版本策略；参数语义见 [PARAMETERS.md](PARAMETERS.md)，后端接入见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。

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
| `qzi_status qzi_register_backend(const qzi_backend *)` | 复制 vtable；`NULL` 恢复内置参考后端；必须在 `qzi_engine_create` 之前调用 |
| `qzi_status qzi_engine_create(const qzi_engine_config *, qzi_engine **, char *error, size_t cap)` | 创建 engine；`config->abi_version` 必须等于 `QZI_ABI_VERSION` |
| `void qzi_engine_destroy(qzi_engine *)` | 释放 engine（调用前必须没有正在进行的生成） |
| `qzi_status qzi_engine_health(qzi_engine *, qzi_health *, char *, size_t)` | 状态/计数快照 |
| `qzi_status qzi_engine_unload(qzi_engine *, char *, size_t)` | 释放可重建的常驻状态（文本塔/缓存），**不卸载 DiT**；busy 时拒绝 |
| `qzi_status qzi_generate_qwen21(...)` | 同步生成 Qwen-Image 2.1 PNG |
| `qzi_status qzi_generate_zimage(...)` | 同步生成 Z-Image PNG |
| `void qzi_image_release(qzi_engine *, qzi_image *)` | 释放 `data` 并把结构体清零；对已清零的结构体是幂等的 |

`qzi_register_backend` 引用的 `qzi_backend` 定义在 `native/include/qzi_backend.h`，由 `qzi_reference_backend()` 导出内置参考 vtable 供包装/对照使用。

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

## 典型调用序列

```c
char error[QZI_ERROR_CAPACITY] = {0};
size_t sizes[QZI_LAYOUT_SLOTS];
qzi_engine *engine = NULL;

/* 0. 可选：换成真实引擎后端（必须在 create 之前） */
qzi_register_backend(&my_h3_backend);

/* 1. 版本 + 布局自检 */
if (qzi_abi_version() != QZI_ABI_VERSION) abort();
if (qzi_abi_layout(sizes, QZI_LAYOUT_SLOTS) != QZI_OK) abort();

/* 2. 创建 engine；目录可为 NULL（参考后端不读权重） */
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
req.steps = 8; req.guidance = 1.0f; req.lora_scale = 1.0f; req.seed = 42;

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

## 内置参考后端

`native/src/qzi_stub.c` 实现的 `qzi-reference` 后端：无需 GPU/权重/网络，按 prompt、seed、尺寸、模型标签生成确定性 RGB 图，再由 `native/src/qzi_png.c` 编码为 PNG（stored DEFLATE 块，无 zlib 依赖）。它跑完整的 5 个阶段并上报 8 个 `dit` 步，`qzi_health.reference_backend == 1`。用途：CI、托管单测、C smoke test、以及给真实引擎做 ABI 对照。**它不是图像模型，输出不是真实生成结果。**
