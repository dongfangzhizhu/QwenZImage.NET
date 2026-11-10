# 请求参数与校验规则

本文件是 `QwenZImage.NET` 参数规则的唯一说明。规则在两处实现并保持完全一致：

- 托管侧：`src/QwenZImage.NET/Validation/RequestValidator.cs`（先于 native 调用失败，避免分钟级模型加载）
- native 侧：`native/src/qzi.c` 的 `qzi_validate_qwen21()` / `qzi_validate_qwen21_edit()` / `qzi_validate_qwen21_edit_ex()` / `qzi_validate_zimage()` / `qzi_validate_zimage_control()`（保护直接从 C/C++ 调用 ABI 的集成方）

两者不一致时以 `native/include/qzi.h` 的注释和本文件为准，并应视为缺陷修复。

`qzi_generate_qwen21()` 在校验通过后、调用 backend 之前，还会把 ABI 的「默认值 sentinel」统一归一化（见下节）。因此参考后端与真实 CUDA 后端看到的是**同一组有效参数**，无 GPU 的 C smoke 测试可以用确定性输出反推参数是否真正贯通。

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
| `Guidance` | `null` → turbo `1.0`，base `4.0` | **Turbo 必须恰好 `1.0`**；Base `1..20`，且必须有限（非 NaN / Inf） |
| `NegativePrompt` | `null` | 仅 Base 可用；Turbo 无 CFG 分支，非空即报错 |
| `Lora` | `null` | **裸文件名**，UTF-8 字节数 ≤ **255**（`QZI_LORA_NAME_MAX`）；禁止 `/`、`\`、`:`、`..` 与控制字符（`< 0x20`、`0x7f`）；在 `qwen21_lora_dir` 中解析。**turbo / base 两个 tier 都可用** |
| `LoraScale` | `null` → `1.0` | 必须有限（非 NaN / Inf）且在 `0.05..4`；**没有 `Lora` 却设置 `LoraScale` 会报错**（否则会被引擎静默丢弃）。托管侧在没有 LoRA 时向 native 发送 `0.0f` sentinel，而不是 `1.0f` |

对应 native 结构体 `qzi_qwen21_request`。校验失败返回 `QZI_ERR_INVALID_ARGUMENT`；未知 tier 同样返回该状态；不支持的 LoRA / 特性返回 `QZI_ERR_UNSUPPORTED`。

### 默认值归一化与 ABI sentinel（C 调用方必读）

`qzi_generate_qwen21()` 校验之后会复制一份请求并归一化，再把**归一化后的副本**交给 backend；调用方传入的结构体不会被修改。这样两个后端都不必各自猜默认值。

| 字段 | 输入 sentinel | 归一化结果 |
| --- | --- | --- |
| `tier` | `NULL` / `""` | `"turbo"` |
| `steps` | `0` | turbo `8`，base `40` |
| `guidance` | `0.0f` | turbo `1.0f`，base `4.0f` |
| `lora_scale`（有 `lora`） | `0.0f` | `1.0f` |
| `lora_scale`（无 `lora`） | 必须恰好 `0.0f` | 保持 `0.0f`（表示「无 adapter / 默认强度」） |

也就是说：**没有 LoRA 时唯一合法的 `lora_scale` 是 `0.0f`**；带 LoRA 时 `0.0f` 与 `1.0f` 等价，真实后端也把它们视作同一种 adapter 状态，不会触发多 GB 的 DiT 重载。非法值（负数、NaN、Inf、`> 4`、`< 0.05`、无 LoRA 却非零）一律**拒绝**，绝不静默 coerce。

### LoRA 与 DiT 重载

`qzen_qwen21_apply_lora()` 会把 adapter 就地合并进 INT8 权重。因此**切换 adapter、改变 scale、或去掉 LoRA 都需要重新加载 DiT**（分钟级），并上报一次 `"model load"` phase。真实后端在加载多 GB 权重**之前**先校验裸文件名与 `qwen21_lora_dir`；合并失败会关闭并清空 DiT，避免半合并权重被下一个无 LoRA 请求复用。参考后端同样跟踪 tier / adapter / scale 变化，让无 GPU 的测试也能验证这条状态机。

## Qwen-Image 2.1 图像编辑（`Qwen21EditOptions`）

`Qwen21EditOptions` 继承 `QwenImageOptions`，因此上表所有 t2i 规则（尺寸、tier、steps、guidance、LoRA、prompt）都继续适用；它用同一份 7B 权重编辑一张参考图，对齐 WanGP 的 img2img / 局部重绘语义。对应 native 结构体 `qzi_qwen21_edit_request`（`base` + 编辑字段），入口 `qzi_generate_qwen21_edit()`。

| 字段 | 默认值 | 规则 |
| --- | --- | --- |
| `Reference` | **必填** | `QziRasterImage`，3 通道**平面** RGB8（`[R][G][B]` 三个 `W*H` 平面）。每边 `1..4096`，总像素 `≤ 2048×2048`。可用 `FromPlanar` / `FromInterleavedRgb` / `FromPng` / `FromPngBase64` 构造；PNG 支持 8 位非隔行灰阶 / RGB / RGBA，透明像素按白底合成 |
| `Mask` | `null` | 可选，`QziRasterImage`，1 通道、尺寸必须与 `Reference` 完全一致（`FromPngMask` / `FromPngBase64(..., mask: true)` 把亮度×alpha 转成灰度）。**白色区域按 prompt 重绘，黑色区域保持参考轨迹**；native 侧缩放到 1/16 latent 网格并以 `> 64` 二值化 |
| `Strength` | `null` → `0.75` | WanGP 的 `denoising_strength`：`0.05..1.0` 且必须有限。`1.0` 从纯噪声开始（≈ t2i），越小越贴近参考图。`EffectiveStrength` 是实际发送值 |
| `MaskingStrength` | `null` → 有 mask 时 `1.0`，无 mask 时 `0` | `0.05..1.0` 且必须有限；表示剩余调度中把未遮罩 latent 恢复到参考轨迹的比例。**必须有 `Mask`**，否则会被静默丢弃，托管与 native 双侧都直接报错。`EffectiveMaskingStrength` 是实际发送值 |

参考图与 mask 会先经过**居中裁剪的 cover fit** 缩放到 `Width × Height` 输出画布，所以不要求与输出同尺寸；两者用同一套裁剪，保证笔刷与像素对齐。若尺寸本来就一致，则不做重采样。

### 编辑参数归一化（C 调用方必读）

`qzi_generate_qwen21_edit()` 校验后复制一份请求并归一化，再把**副本**交给 backend；调用方结构体不变。两个后端看到的有效参数因此一致：

| 字段 | 输入 sentinel | 归一化结果 |
| --- | --- | --- |
| `strength` | `0.0f` | `0.75f`（`QZI_EDIT_DEFAULT_STRENGTH`） |
| `masking_strength`（有 `mask`） | `0.0f` | `1.0f` |
| `masking_strength`（无 `mask`） | 必须恰好 `0.0f` | 保持 `0.0f`（不做遮罩恢复） |

**非法值一律拒绝，不静默 coerce**：负数、NaN / Inf、`> 1.0`、`< 0.05`、无 mask 却带 `masking_strength`、`flags != 0`、缺 `reference_rgb`、参考图超限，均返回 `QZI_ERR_INVALID_ARGUMENT`。`.NET` 侧 `RequestValidator.Validate(Qwen21EditOptions)` 先于 native 调用执行同一套规则，避免把非法请求送进分钟级模型加载；能力探测失败（`SupportsImageEditing == false`）抛 `NotSupportedException`。

### 编辑语义与对齐 WanGP

- **强度 → 起点步**：`first_step = int(steps * (1 - strength))`，只重跑剩余步数；turbo 8 步下 `strength=0.75` 从第 2 步起跑。参考后端也按此规则缩短 `dit` 步，让无 GPU 测试的 phase 序列与真实引擎一致。
- **图像编辑唯一多出的一步是把参考图 VAE 编码成 latents**：真实后端用 `qzen_qwen21_vae_encode()` 编码（`qzen_qwen21_edit_start()` 以此为去噪起点），期间上报 `"image encode"` phase；文本编码、DiT 去噪、VAE 解码、PNG 编码全部复用 t2i 常驻管线。
- **mask 恢复**：mask 白色区域自由重绘，黑色区域在 `masking_strength` 比例的剩余调度内被拉回参考轨迹，等价于 WanGP 的 `masking_strength` / `latent_mask` 语义。无 mask 时整张图按 `Strength` 编辑。
- **确定性**：参考后端把参考图、mask、两个 strength 与参考尺寸一起纳入哈希，因此改任一参数都会改变输出字节；C smoke 用这一点验证参数是否真正贯通。

WebUI 的等价字段见 [WEBUI.md](WEBUI.md)（`reference_image` / `mask_image` / `strength` / `masking_strength`）。

## Qwen-Image 2.1 高级编辑（`Qwen21EditExOptions`）

`Qwen21EditExOptions` 继承 `Qwen21EditOptions`，因此上节所有编辑规则（`Reference` / `Mask` / `Strength` / `MaskingStrength`，以及 t2i 的尺寸、tier、steps、guidance、LoRA、prompt）继续适用。它是**第三个 ABI v1 增量扩展**：native 结构体 `qzi_qwen21_edit_ex_request`（把普通编辑请求作为首成员，再追加高级字段），入口 `qzi_generate_qwen21_edit_ex()`，`.NET` 侧 `GenerateQwen21EditEx(Async)`。三项能力都只用同一份 7B 权重，**不需要新的 checkpoint 或额外模型目录**。

| 字段 | 默认值 | 规则 |
| --- | --- | --- |
| `Width` / `Height` | 1024 / 1024 | 语义变为**最终输出画布**（含外扩边距），其余约束与 `QwenImageOptions` 相同（32 的倍数、`≤ 2048`、面积 `≤ 2048×2048`） |
| `MarginTop` / `MarginBottom` / `MarginLeft` / `MarginRight` | `0` | 输出画布四边各增加多少像素；必须是 **32 的倍数**（WanGP `OUTPAINTING_ALIGNMENT`），四者全 `0` 表示不外扩。扣掉左右 / 上下边距后的内部源矩形必须仍 ≥ `32×32` |
| `ModelMode` | `MaskedDenoising` (`0`) | `Qwen21EditMode`：`MaskedDenoising` 或 `LanPaint2` / `LanPaint5` / `LanPaint10` / `LanPaint15`（数值 **2/3/4/5** 是 WanGP 下拉框索引，不是迭代次数，分别对应 2/5/10/15 次内嵌 Langevin 迭代）。`EffectiveModelMode` 是实际发送的 `uint` |
| `Rgba` | `false` | `true` 输出 **PNG colour type 6**（RGBA），保留 VAE decoder `conv_out` 的第 4 通道作为模型生成的 alpha；`false` 仍是普通 RGB（colour type 2） |
| `Flags` | `0` | 保留字段，必须为 `0` |

`HasOutpainting` / `IsLanPaint` 是两个只读派生量。参数语义与 WanGP 对齐：

- **Red Canvas 外扩**：参考图居中裁剪 cover-fit 到扣掉边距后的内部矩形，四周填不透明红色 `FF0000`；prompt 自动追加固定后缀 `Remove the red paddings on the sides and show what's behind them.`；红色边距**跑完整去噪调度**（`step_offset = 0`），`Strength` 只作用于内部绘制区。全零边距时与普通编辑完全等价。
- **LanPaint**：训练-free 的 BiG-score Langevin 精修，插在每个 sampler 步之前，无需额外权重。它**必须有 `Mask`**、**与外扩互斥**（WanGP 对 instruction outpainting 会禁用 LanPaint），并把 `denoising_strength` 与 `masking_strength` 都强制为 `1.0`。为避免静默覆盖调用方意图，托管侧若显式传了非 `1.0` 的 `Strength` / `MaskingStrength` 就直接报 `ArgumentException`；native 侧 `qzi_normalize_qwen21_edit_ex()` 归一化时统一置 `1.0`，两个后端看到同一组有效值。
- **RGBA**：VAE decoder 的 `conv_out` 本来就是 `144→4`，第 4 通道此前被丢弃；`rgba != 0` 时用与 RGB 相同的 `clamp + (x+1)*127.5` 量化保留它，写成 RGBA PNG。CPU 与 CUDA 两条解码路径共用同一个量化函数。

native 侧 `qzi_validate_qwen21_edit_ex()` 先复用 `qzi_validate_qwen21_edit()` 校验内嵌的普通请求，再检查 `flags`、四边距的 32 像素对齐与残余源矩形、`model_mode` 白名单、LanPaint 的 mask 要求与外扩互斥、`rgba ∈ {0,1}`；非法值一律返回 `QZI_ERR_INVALID_ARGUMENT`，不静默 coerce。`.NET` 侧 `RequestValidator.Validate(Qwen21EditExOptions)` 在 native 之前执行同一套规则（含 `RequestValidator.OutpaintAlignment = 32`）；老 native 库缺第三个能力时 `SupportsAdvancedEditing == false`，调用抛 `NotSupportedException`（`QZI_ERR_UNSUPPORTED`）。

能力探测比普通编辑多一层依赖：`Qwen21EditExOptions` 内嵌了普通编辑请求，因此 `AbiGuard` 要求 `SupportsImageEditing` **与** `qzi_abi_features() & 0x4`、`qzi_qwen21_edit_ex_layout()` 尺寸比对、`qzi_generate_qwen21_edit_ex` 导出四者全部通过才报告支持。WebUI 里只要出现任一高级字段（`margin_*` / `model_mode` / `rgba`），`RequestMapper` 就把普通编辑作业升级成 `Qwen21EditExOptions`；全部字段都省略时仍走第二个增量扩展，旧库照样可用。

**保真度边界**：WanGP 的 Qwen-Image 编辑是 condition-prefix 模型（参考 latent 拼在带噪目标前面，DiT 通过注意力读取它），本引擎是 img2img sampler（参考图就是起点 latent）。因此 Red Canvas 外扩通过「边距 + 绘制区从全噪声起步、保留区 pin 到已知轨迹」来近似，**不是逐像素复刻 WanGP 的外扩效果**；详见 [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md)。多参考图、20B edit / edit_plus / layered、Viggle / Pruna、prompt enhancer `EDITING` 仍不在本库范围内。

WebUI 的等价字段见 [WEBUI.md](WEBUI.md)（`margin_*` / `model_mode` / `rgba`），C 契约见 [NATIVE_ABI.md](NATIVE_ABI.md)。

## Z-Image（`ZImageOptions`）

| 字段 | 默认值 | 规则 |
| --- | --- | --- |
| `Width` / `Height` | 1024 / 1024 | **16 的倍数**，正数，`≤ 2048`，面积 `≤ 2048×2048` |
| `Checkpoint` | `"bf16"` | `null` / `""` / `"bf16"`（默认）、`"nunchaku"`，或以 `"nunchaku-"` 开头的 pack id（如 `"nunchaku-r256"`） |
| 不支持的 checkpoint | — | 任何含 `fp4` 的 id（**大小写不敏感**，如 `svdq-fp4-r256`、`SVDQ-FP4-r256`）以及未知值 → 托管侧 `ArgumentException`，native 侧 `QZI_ERR_UNSUPPORTED`（引擎只解码 int4）。两侧用同一套大小写无关判定 |

Z-Image Turbo 使用**固定 8 步调度、无 CFG 分支**，因此请求中**没有** `steps`、`guidance`、`negative_prompt` 字段——这不是遗漏，而是模型本身的约束。对应 native 结构体 `qzi_zimage_request`。

## Z-Image ControlNet Union（`ZImageControlOptions`）

`ZImageControlOptions` 继承 `ZImageOptions`，因此上表所有 t2i 规则（16 的倍数、checkpoint）继续适用；它用 Z-Image Turbo Fun **ControlNet Union 2.1** 的一张控制图条件化生成，对齐 WanGP 的 `z_image_control2_1`。对应 native 结构体 `qzi_zimage_control_request`（`base` + 控制字段），入口 `qzi_generate_zimage_control()`，`.NET` 侧 `GenerateZImageControl(Async)`；老 native 库缺该能力时 `SupportsControlNet == false`，调用抛 `NotSupportedException`。

| 字段 | 默认值 | 规则 |
| --- | --- | --- |
| `ControlImage` | **必填** | `QziRasterImage`，3 通道**平面** RGB8。每边 `1..4096`，总像素 `≤ 2048×2048`。与 `Reference` 共用 `FromPng` / `FromPngBase64` 等解码器；输出到画布时做**居中裁剪 cover fit** |
| `ControlType` | `null` → `auto` | 大小写不敏感，合法值 `auto` / `pose` / `canny` / `depth` / `scribble` / `raw`；`EffectiveControlType` 返回规范后的小写值。**描述性元数据**：本库不运行姿态 / 边缘 / 深度估计器，对齐 WanGP `guide_preprocessing = "V"`，调用方需提供已预处理的控制图 |
| `ControlScale` | `0`（sentinel）→ `0.65` | WanGP 的 `control_context_scale`（Union 2.1 默认 `0.65`；v1/v2 权重习惯 `0.75`）。`0.05..2.0` 且必须有限；`0` 与不设等价。`EffectiveControlScale` 是实际发送值 |

归一化与校验：native 侧 `qzi_validate_zimage_control()` 先复用 `qzi_validate_zimage()` 校验 `base`，再校验控制图与两个控制字段；非法值（NaN / Inf、越界、未知 `control_type`、`flags != 0`、缺 `control_rgb`、尺寸超限）一律返回 `QZI_ERR_INVALID_ARGUMENT`，不静默 coerce。`.NET` 侧 `RequestValidator.Validate(ZImageControlOptions)` 先于 native 执行同一套规则。

真实后端流程：懒加载 Union 2.1 权重（`zimage_control_dir`）→ VAE 编码控制图为 16 通道 1/8 latent → 拼成 **33 通道 control context**（`control_latent | keep_mask(全 0) | inpaint_latent(全 0)`）→ 每个去噪步重算 15 条 control layer skip + 2 条 noise refiner skip，并在主 DiT 对应层之后以 `control_scale` 注入。普通 t2i 完全不走这条路径。参考后端把控制图像素、规范后的 `control_type`、`control_scale` 与尺寸纳入确定性哈希，并多上报一个 `"image encode"` phase。

边界：仅支持 Union **v2 / v2.1**（33 通道、15 control layers + 2 个独立 control noise refiner），v1（16 通道）在权重加载时明确拒绝；无 inpaint mask / outpainting 入口；约 3.7 GiB 权重叠加常驻 DiT + 文本塔，24 GiB 卡上可能触发 GPU 上传失败并回退 CPU hint 路径（结果正确但更慢）。未配置 `zimage_control_dir` 时真实后端返回 `QZI_ERR_NOT_CONFIGURED`。

**与 WanGP 的两处取值核对**（避免把 UI 标签误当推理参数）：

- `ControlScale = 0.65` 对应 WanGP `defaults/z_image_control2_1.json` 与 `settings/z_image_control2_1_settings.json` 的 `control_net_weight_alt`。实际传参链是 `wgp.py` 把该 UI 值装进 `context_scale[0]` → `z_image_main.py` 赋给 `control_context_scale` → `pipeline_z_image.py` 用作 hint 的缩放系数；后者函数签名里的 `0.75` 只是未被覆盖时的兜底（v1 / v2 的习惯值，`defaults/z_image_control*.json` 也确为 `0.75`）。本库只支持 33 通道权重且配置指向 **Union 2.1**，故默认取 `0.65`；若要复现 v2 的 `0.75`，显式传 `ControlScale = 0.75f` 即可，范围 `0.05..2.0` 已覆盖。
- `control_type` 保留 `pose` / `canny` / `depth` / `scribble` / `raw`（`auto` 为默认别名）。WanGP 三份 `defaults/z_image_control*.json` 的 description 都写「Pose, Canny, Depth, and Scribble」；`z_image_handler.py` 的 `guide_preprocessing.selection = ["", "PV", "DV", "EV", "V"]` 描述的是**预处理方式**（姿态 / 深度 / 边缘 / 原始格式），不是控制类型枚举。本库固定走 `"V"`（Use Z-Image Raw Format），不运行任何预处理器，因此 `control_type` 只作为描述性元数据参与参数校验与确定性哈希，保留 `scribble` 不会造成推理行为偏差。

WebUI 的等价字段见 [WEBUI.md](WEBUI.md)（`control_image` / `control_type` / `control_scale`，`model = "zimage-control"`）。

## 画布示例

| 请求 | 结果 |
| --- | --- |
| Qwen21 `2048×2048` | ✅ 合法（恰好等于面积上限） |
| Qwen21 `2048×2080` | ❌ 超出面积上限 |
| Qwen21 `100×64` | ❌ 100 不是 32 的倍数 |
| ZImage `1008×1008` | ❌ 1008 不是 16 的倍数 |
| ZImage `1536×1536` | ✅ 合法，见 `QziPresets.ZImage` |
| 编辑：参考图 `768×512` + 输出 `1024×1024`、无 mask | ✅ 合法，cover fit 后整图按 `Strength` 编辑 |
| 编辑：`strength = 1.5` | ❌ 超出 `0.05..1.0`，托管 / native 双侧都拒绝 |
| 编辑：`masking_strength = 0.5` 但没有 `Mask` | ❌ 会被静默丢弃，直接报错 |
| 编辑：`Mask` 尺寸与 `Reference` 不一致 | ❌ 笔刷无法对齐，直接报错 |
| 高级编辑：`1024×1024` + 四边各 `128` 外扩 | ✅ 合法，内部源矩形 `768×768`，边距都是 32 的倍数 |
| 高级编辑：上边距 `100` | ❌ `100` 不是 32 的倍数 |
| 高级编辑：`1024×1024` + 左右各 `512` | ❌ 内部源宽度为 0，不足 `32×32` |
| 高级编辑：LanPaint + 任意非零边距 | ❌ 两者互斥（WanGP 对 Red Canvas 禁用 LanPaint） |
| 高级编辑：LanPaint 但无 `Mask` | ❌ LanPaint 只在遮罩编辑区做 Langevin 精修 |
| 高级编辑：LanPaint 且 `Strength = 0.6` | ❌ 该模式强制两个 strength 为 `1.0`，显式传其它值直接报错 |

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
| `"image encode"` | `ImageEncode` | **图像编辑 / Z-Image ControlNet**：把参考图或控制图 VAE 编码成 latents（0/1 → 1/1） |
| `"dit"` | `Dit` | 去噪步数（Turbo 为 8 步；编辑时只跑 `steps - first_step` 步） |
| `"vae"` | `Vae` | VAE 解码 |
| `"encode"` | `Encode` | PNG 编码 |

未识别的字面量会被托管侧降级为 `GenerationPhase.Unknown`，不会抛异常。切换模型、tier、LoRA adapter 或 LoRA scale 都会触发一次 `"model load"`；参考后端也模拟这一行为，因此无 GPU 的测试能验证参数与重载的贯通。

取消通过 `qzi_cancel_fn` 在阶段/去噪步之间轮询；返回非零 ⇒ `QZI_ERR_CANCELLED`，`.NET` 侧转换成 `OperationCanceledException`。
