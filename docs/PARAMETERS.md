# 请求参数与校验规则

本文件是 `QwenZImage.NET` 参数规则的唯一说明。规则在两处实现并保持完全一致：

- 托管侧：`src/QwenZImage.NET/Validation/RequestValidator.cs`（先于 native 调用失败，避免分钟级模型加载）
- native 侧：`native/src/qzi.c` 的 `qzi_validate_qwen21()` / `qzi_validate_zimage()`（保护直接从 C/C++ 调用 ABI 的集成方）

两者不一致时以 `native/include/qzi.h` 的注释和本文件为准，并应视为缺陷修复。

## 通用规则

| 字段 | 规则 |
| --- | --- |
| `Prompt` | 必填、非空；UTF-8 字节数 ≤ **16384**（不是字符数） |
| `Seed` | `uint64`，同一请求 + 同一 seed + 同一后端 ⇒ 参考实现下字节级一致 |
| `Width` / `Height` | 见下表；边长最大 **2048**；`width * height ≤ 2048*2048` |
| `flags`（native 保留字段） | 必须为 **0**；`.NET` 侧不暴露 |
| 成功返回 | 一段完整的 `image/png` 字节；`.NET` 侧封装为 `GenerationResult` |

## Qwen-Image 2.1（`QwenImageOptions`）

| 字段 | 默认值 | 规则 |
| --- | --- | --- |
| `Width` / `Height` | 1024 / 1024 | **32 的倍数**，正数，`≤ 2048`，面积 `≤ 2048×2048` |
| `Tier` | `Turbo` | `Turbo` 或 `Base`；native 字符串为 `"turbo"` / `"base"`，`NULL` 等同 `"turbo"` |
| `Steps` | `null` → turbo `8`，base `40` | **Turbo 必须恰好 8**（传入其它值报错，不静默忽略）；Base `1..50` |
| `Guidance` | `null` → turbo `1.0`，base `4.0` | **Turbo 必须恰好 `1.0`**；Base `1..20` |
| `NegativePrompt` | `null` | 仅 Base 可用；Turbo 无 CFG 分支，非空即报错 |
| `Lora` | `null` | 必须是**裸文件名**：禁止 `/`、`\`、`..`；由引擎在自己的 LoRA 目录中解析 |
| `LoraScale` | `null` → `1.0` | `0.05..4`；设置了 `LoraScale` 却没有 `Lora` 会报错（否则会被引擎静默丢弃） |

对应 native 结构体 `qzi_qwen21_request`。校验失败返回 `QZI_ERR_INVALID_ARGUMENT`；未知 tier 同样返回该状态；不支持的 LoRA / 特性返回 `QZI_ERR_UNSUPPORTED`。

## Z-Image（`ZImageOptions`）

| 字段 | 默认值 | 规则 |
| --- | --- | --- |
| `Width` / `Height` | 1024 / 1024 | **16 的倍数**，正数，`≤ 2048`，面积 `≤ 2048×2048` |
| `Checkpoint` | `"bf16"` | `null` / `""` / `"bf16"`（默认）、`"nunchaku"`，或以 `"nunchaku-"` 开头的 pack id（如 `"nunchaku-r256"`） |
| 不支持的 checkpoint | — | 任何含 `fp4` 的 id（如 `svdq-fp4-r256`）以及未知值 → 托管侧 `ArgumentException`，native 侧 `QZI_ERR_UNSUPPORTED`（引擎只解码 int4） |

Z-Image Turbo 使用**固定 8 步调度、无 CFG 分支**，因此请求中**没有** `steps`、`guidance`、`negative_prompt` 字段——这不是遗漏，而是模型本身的约束。对应 native 结构体 `qzi_zimage_request`。

## 画布示例

| 请求 | 结果 |
| --- | --- |
| Qwen21 `2048×2048` | ✅ 合法（恰好等于面积上限） |
| Qwen21 `2048×2080` | ❌ 超出面积上限 |
| Qwen21 `100×64` | ❌ 100 不是 32 的倍数 |
| ZImage `1008×1008` | ❌ 1008 不是 16 的倍数 |
| ZImage `1536×1536` | ✅ 合法，见 `QziPresets.ZImage` |

## 预设

`QziPresets.Qwen21`（5 项）与 `QziPresets.ZImage`（4 项）中的 `Options` 故意带空 `Prompt`，它们是模板而非可执行请求。使用前必须：

```csharp
GenerationOptions request = QziPresets.WithPrompt(QziPresets.ZImage[1], "a mountain lake");
```

`ValidationTests.All_Presets_Are_Valid_With_A_Prompt` 会保证所有预设加上 prompt 后都能通过校验。

## 进度阶段

native 侧通过 `qzi_progress_fn` 上报的 `phase` 是**静态字面量**，不得释放：

| phase | `GenerationPhase` | `completed/total` 含义 |
| --- | --- | --- |
| `"model load"` | `ModelLoad` | 权重加载阶段 |
| `"text"` | `Text` | 文本编码 |
| `"dit"` | `Dit` | 去噪步数（Turbo 为 8 步） |
| `"vae"` | `Vae` | VAE 解码 |
| `"encode"` | `Encode` | PNG 编码 |

取消通过 `qzi_cancel_fn` 在阶段/去噪步之间轮询；返回非零 ⇒ `QZI_ERR_CANCELLED`，`.NET` 侧转换成 `OperationCanceledException`。
