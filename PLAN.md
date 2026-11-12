# QwenZImage.NET 项目计划与决策记录

> 目标：把上游 MIT 许可的 C/CUDA 工程中的 **Qwen-Image 2.1** 与 **Z-Image** 两个文生图能力，封装成一个独立、可开源、可由 `.NET` 调用的 NuGet 包 `QwenZImage.NET`。

## 1. 已确认的需求 / 决策记录

| # | 决策 | 来源 |
| --- | --- | --- |
| D1 | **新建独立目录** `D:/AI/convert/QwenZImage.NET`，不改动原工程 | 用户 |
| D2 | **底层 C，.NET 10 封装，追求高性能**；**不要 Go 部分**，原 Go WebUI 的队列 / 进度 / 取消 / 重试 / 预设功能用纯 `.NET` 实现 | 用户明确指示 |
| D3 | **只做 Native P/Invoke**：不做 HTTP 客户端、不做双 transport、不引入 Go WebUI | 用户对封装方式的二选一 |
| D4 | 项目名 / NuGet 包 ID = **`QwenZImage.NET`**，仓库 = `dongfangzhizhu/QwenZImage.NET` | 用户 |
| D5 | 启用 `git` 版本控制、写入本计划文件、准备 CI 与 NuGet 发布工作流 | 用户 |
| D6 | **先本地完成**：本次不 push GitHub、不 publish NuGet（**阶段 13 已被后续 D11 覆盖**） | 用户在确认发布节奏时选择「先本地完成（推荐）」 |
| D7 | 增加一个**跟原版 Go WebUI 类似的 WebUI 项目用于手动测试**，跨平台支持 Windows 与 Linux；仍遵守 D2，用纯 `.NET` / ASP.NET Core 实现 | 用户 |
| D8 | **发布的 NuGet 包要同时包含 .NET 包与 Windows、Linux 两个平台的 native 包** | 用户 |
| D9 | 用户反馈「所有模型输出都是噪点图，steps 与原版相同却不正常」。排查确认是 bundled reference backend 不做扩散推理，而非参数问题。曾实现 WebUI 专用 `QZI_ENGINE_URL` HTTP 适配器连上游服务，**后被用户判定方向错误并完全删除**（见 D10） | 用户反馈 + 排查结论 |
| D10 | **库必须直接绑定 native 后端运行**：用与 `h3.config.json` 相似的配置文件指定模型路径；编译出的 native 后端改用新名称、抛弃 `h3` 命名。用户确认：真实引擎源码**复制进仓库并改名**、命名为 **`QwenZImageNative` + `QwenZImage.config.json`**、上一轮 WebUI HTTP 通道**完全删除** | 用户明确纠正 + 三项选择 |
| D11 | **开源发布**：整理仓库（只保留源码/文档/配置/脚本/CI，不提交任何构建产物），远程改到 `dongfangzhizhu/QwenZImage.NET` 并推送；NuGet 改用 **Trusted Publishing**（GitHub Actions OIDC → 1 小时短期 API key），不再保存长期 `NUGET_API_KEY` | 用户在阶段 13 前明确指示 |

技术调研结论（决定 D2/D3/D10 的关键事实）：

- 上游服务是 **C 写的** HTTP/JSON 服务；Go WebUI 只是它的 HTTP 客户端，**不是底层模型实现**。真正的推理能力在 `h3core` / `h3serve` 的 C/CUDA 对象库里，因此「不用 Go」不会损失推理能力。
- 原引擎一次只驻留一个模型、单槽队列；新的 C ABI 用 `QZI_ERR_BUSY` 表达同样的单飞语义，由 `.NET` 队列串行化。
- `h3.c` 编排器与视频/WAN 管线不被 `qwen21` / `zimage` 链引用，可排除；两条 pipeline 的最小源码集为 64 个文件，已复制进 `engine/` 并机械改名（残留 `h3_*` / `H3_*` **标识符**为 0，产物名与符号均为 `QwenZImageNative` / `qzen_*`）。注释中保留的 `MiniMax-H3` 是上游**模型架构名**，属于事实描述，不是本项目产物命名；配置文件里的 `h3-models` 是开发机上已存在的**权重数据目录名**，可自由改名。
- 上游 LICENSE 为 MIT（Copyright (c) 2026 Salvatore Sanfilippo），允许复制/修改/改名，须保留版权与许可声明 → 已落到 `engine/LICENSE`、`engine/UPSTREAM.md`、`THIRD_PARTY_NOTICES.md`。

## 2. 架构（D10）

```
Your app (.NET 10)                     Browser (manual testing only)
  └─ QziClient / QziGenerationQueue   └─ QwenZImage.NET.WebUI (ASP.NET Core, :7860)
       / QziPresets                        │  Minimal API + SSE + wwwroot 静态前端
       / QziConfig.Load()                 │  health / presets / jobs / cancel / retry
            │                             │  outputs / trash / restore / unload
            └──────────────┬──────────────┘
                           ▼
              QwenZImage.NET.dll   (LibraryImport 源生成 P/Invoke)
                           ▼  进程内，无 HTTP / 无 socket / 无 JSON hop
              QwenZImageNative.dll / libQwenZImageNative.so
                ← native/include/qzi.h (稳定 C ABI v1，唯一托管/原生契约)
                 ├─ qzi.c 控制层：校验 / 单飞 / 计数 / 进度 / 取消 / 错误 / 缓冲区所有权
                 └─ qzi_backend vtable（编译期由 QZI_HAS_REAL_BACKEND 选择）
                      ├─ OFF：qzi_stub.c 参考后端（确定性噪点，CI / 测试）
                      └─ ON ：qzi_engine_backend.c → engine/ 真实 C/CUDA 推理引擎
                                Qwen21 DiT / text tower / VAE / edit
                                ZImage DiT / text tower / VAE / ControlNet / CUDA kernels
                          ▲
              QwenZImage.config.json（模型权重路径，QZI_CONFIG / --config）
```

**D10 边界**：`QziClient` 直接 P/Invoke 同一个进程内的 `QwenZImageNative`；WebUI 与用户应用处于同一层，都只走 `QziClient`。**没有任何外部服务、HTTP transport、sidecar 或 Go 进程**。`QZI_CONFIG` 让 native backend 读取 ABI 里没有的 `qwen21_lora_dir` / `zimage_nunchaku_dir` / `zimage_control_dir` 等字段；`QziClient` 通过 `NativeConfigScope` 在 `qzi_engine_create()` 期间**作用域发布** `QZI_CONFIG` 与六个 `QZI_*` 目录变量（含 `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR` / `QZI_ZIMAGE_CONTROL_DIR`），构造结束即还原，构造期间由 `EngineCreateGate` 串行化。native 侧 `qzen_config_apply_environment()` 在文件加载后应用，显式传值优先；所有 native 配置读取统一走 `qzi_getenv()`（Windows 用 Win32 API 读实时环境，绕过 CRT 的启动快照）。每个 engine 因此快照自己的配置，多实例互不串扰，且 ABI v1 的导出符号与结构体布局不变。

**为什么保留两种构建**：CI 与托管测试必须在无 GPU 环境运行，且真实 CUDA 库依赖 cudart / cuBLAS 与 GPU 驱动，不适合塞进默认包。因此 `QZI_WITH_CUDA_ENGINE=OFF` 的参考后端仍是默认打包形态；`ON` 的构建默认生成覆盖 sm_80 → sm_121 的 **fat binary**（CMake 查询 `nvcc --list-gpu-arch` 并按工具链能力自动过滤），一个同名 DLL/SO 即可适配常见显卡，不必按 GPU 重编译。用 `QZI_NATIVE_LIBRARY` 覆盖或替换 `runtimes/<rid>/native/` 即可，托管 API 不变。CI 不含 GPU，所以发布 workflow 里打包的仍是参考后端。

**打包形态（D8）**：单个 `.nupkg` 按 NuGet RID 约定同时携带 `lib/net10.0/QwenZImage.NET.dll`、`runtimes/win-x64/native/QwenZImageNative.dll`、`runtimes/linux-x64/native/libQwenZImageNative.so`。`QziEnsureNativeRuntimes` MSBuild target 在 `GenerateNuspec` 前强制校验两者存在。

## 3. 阶段与状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 调研上游：`h3_serve.c` 编排、两条 pipeline 头文件、配置加载语义、LICENSE | ✅ |
| 1 | 解决方案骨架：`global.json`（.NET 10）、`Directory.Build.props`、`.gitignore` / `.gitattributes` / `.editorconfig`、项目 + `.sln` | ✅ |
| 2 | C ABI v1：`qzi.h` / `qzi_backend.h`、`qzi.c` 控制层、参考后端、无依赖 PNG 编码器、CMake、C smoke test | ✅ |
| 3 | `.NET 10` 封装：Native 层 + 公开模型 / 校验 / 结果 / 客户端；同步 + 异步 + 零拷贝 `WriteTo` | ✅ |
| 4 | 纯 `.NET` 队列与预设：`Channels` 串行分发、取消、重试、事件流、9 个预设 | ✅ |
| 5 | 测试：46 个 xUnit（42 库 + 4 WebUI，阶段 5 当时；阶段 12f 后 69 项，阶段 14 后 73 项，阶段 15 后 94 项 = 72 库 + 22 WebUI，阶段 16 后 119 项 = 86 库 + 33 WebUI，阶段 17 后 139 项 = 97 库 + 42 WebUI，**阶段 19 后 162 项 = 110 库 + 52 WebUI**，**阶段 20 后 202 项 = 118 库 + 84 WebUI**）+ `ctest` native smoke（阶段 5 时各 1/1；阶段 12e 后 Windows 1 项 / Linux 3 项） | ✅ 全绿 |
| 6 | 开源文档：README / LICENSE / THIRD_PARTY_NOTICES / NATIVE_ABI / PARAMETERS / ENGINE_INTEGRATION / WEBUI / 本计划 / 后端模板 | ✅ |
| 7 | CI 与发布工作流：`ci.yml`、`release.yml`、`scripts/build-native.*` | ✅ |
| 8 | `git init` + 本地提交 | ✅（按 D6 当时只做本地，后续由 D11 推送） |
| 9 | **跨平台 WebUI**（D7）：Minimal API、SSE、`wwwroot` 静态前端、CI 运行时冒烟 | ✅ |
| 10 | **单包双平台 native**（D8）：RID staging、`csproj` 打包 + 缺失校验、`NativeLibraryLoader` 支持 `NATIVE_DLL_SEARCH_DIRECTORIES` | ✅ |
| 11 | **噪点根因排查**（D9）：定位到 `qzi_stub.c` 伪随机 RGB + 空转 `dit`；README / WebUI 黄色告警 | ✅；其中 HTTP 适配器方案在 D10 被删除 |
| 12 | **D10 直接绑定真实 native 后端**：删除 `H3ServeExecutor` / `IWebUiEngine` / `NativeEngineAdapter` 与 `QZI_ENGINE_URL`；统一 native 名 `QwenZImageNative`；新增 `QwenZImage.config.json` + `QziConfig` + `qzen_config.c`；复制并改名 64 个引擎文件；写 `qzi_engine_backend.c` 真实适配器；`qzi.c` 编译期选择 backend；`build-native-cuda.bat/.sh`；C smoke 按后端分支；合规文件；收尾补齐文档一致性、`engine/LICENSE` 入包、`QziConfig` 向上查找配置、CUDA 脚本的 MSVC/SDK 环境兜底 | ✅ |
| 12b | **真实后端跨 GPU / 跨发行版可分发**（阶段 12 之后的收尾）：默认 fat binary 覆盖 11 个 SM + CMake 按 nvcc 能力过滤；Linux 运行时 `dlopen` ICU、产物无版本化 `libicuuc.so.NN` 依赖；OpenMP 显式链接；修复 `stage-native.sh` 的 `-cuda` 路径匹配 | ✅ |
| 12c | **LoRA / 参数全链路一致性**（用户要求「确保功能完整性和一致性，包括 LoRA 支持」；不做 macOS 适配）：统一 ABI 契约与两侧校验（裸文件名 + 255 字节 + `QZI_LORA_NAME_MAX`、scale finite / `0.05..4`、fp4 大小写不敏感、默认值归一化）；打通 `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR` 配置覆盖链并让 native 读环境变量；修复真实后端 DiT/LoRA 状态机（提前校验、0 与 1.0 等价、合并失败清场）；参考后端把 tier / LoRA / scale / steps / guidance / negative prompt 纳入确定性哈希并跟踪重载 phase；WebUI 在 turbo / base 都开放 LoRA；补测试与文档 | ✅ |
| 12d | **真图回归矩阵**（用户指定开发节奏第 1 项）：新增独立 `tools/RealImageRegression` harness（刻意不进 sln / CI，检测并拒绝参考后端），`cases.json` 覆盖 qwen21 turbo/base、LoRA scale 1.0 / 0.6、512 / 1024，以及 Z-Image bf16 与 Nunchaku r32 / r256；逐用例输出 PNG + SHA-256 + phase trace，报告增量落盘；Windows / Linux 运行脚本；10/10 真图通过并人工抽查确认非噪点 | ✅ |
| 12e | **POSIX ICU 双路径检查接进 CTest**（用户指定节奏第 2 项）：把一次性 `.scratch` 脚本整理成 `native/tests/qzi_posix_unicode_check.c`，CMake 仅在 POSIX 且链接器支持 `--wrap=dlopen` 时构建 `posix_unicode_with_icu` / `posix_unicode_without_icu` 两个可执行，不支持时优雅跳过；宿主无 ICU 时 CTest 记 SKIP；顺带清理真图回归审计发现的 `qzi_engine_backend.c` fp4 不可达死代码与 `checkpoint + 13` 偏移错误 | ✅ |
| 12f | **消除 `QZI_*` 进程级环境变量多实例串扰**（用户指定节奏第 3 项）：`NativeConfigScope` 只在 `qzi_engine_create()` 期间发布 6 个变量并逆序还原（含构造失败回滚），`QziClient` 用静态锁串行化构造；`QziConfig` 补齐变量名常量；参考后端把 create 时配置指纹混入确定性 PNG；新增 8 项 xUnit（作用域发布/精确还原/未设选项/仅托管变量/失败回滚/双 engine 快照/并发构造隔离/Dispose 幂等）。**审计发现 Windows CRT `getenv()` 看不到 .NET 设置的环境变量**，新增内部 `qzi_getenv()`（Win32 `GetEnvironmentVariableA` / POSIX `getenv`），`qzen_config.c`、`qzi_stub.c` 与 C smoke 统一改用它，修复真实后端与参考后端的配置读取；文档与 PLAN 同步 | ✅ |
| 13 | **推送 GitHub 开源 + NuGet Trusted Publishing**（D11） | ✅ 仓库整理、地址同步、OIDC 发布工作流已就绪；命令与一次性前置条件见第 5 节 |
| 14 | **WebUI 不再默认产出噪点图**（用户反馈「生成图像依然是没有加载模型的无意义图像」）：`NativeLibraryLoader` 显式路径改为 fail-fast（不再静默回退到参考后端）；`WebUiConfig` 自动探测本机 `-cuda` 产物并支持 `QZI_WEBUI_NATIVE=reference` 与 `QZI_REQUIRE_REAL_BACKEND=1`（后者让 WebUI 拒绝生成而不是交付噪点图）；`/api/health` 暴露 `native_library` / `native_library_source`；新增 `start-webui-cuda.ps1` / `.sh` 一键启动；修正页面横幅与 README / WEBUI.md 的误导性指引；新增 4 项选择逻辑测试 | ✅ |
| 15 | **Qwen-Image 2.1 图像编辑 / 对齐 WanGP**（用户要求「Qwen-Image 2.1功能不完整，应该还可以编辑图像，对齐wangp版本的功能」）：参考图 img2img、局部 mask 重绘、`denoising_strength` / `masking_strength`、白色重绘 / 黑色保持参考轨迹。ABI **v1 增量扩展**（`qzi_qwen21_edit_request` + `qzi_abi_features()` 能力位 + `qzi_qwen21_edit_layout()` + `qzi_generate_qwen21_edit()`，既有结构体 / 导出 / 6 个 layout 槽位不变）；`qzi.c` 新增 `qzi_validate_qwen21_edit()` / 归一化；`qzi_backend` 末尾追加可选 `generate_qwen21_edit` 槽位；真实 CUDA 后端复用 `qzen_generate_qwen21()`，只多参考图 cover fit + `qzen_qwen21_vae_encode()` + 1/16 latent mask + `qzen_qwen21_edit_start()` 与 WanGP 式遮罩恢复；引擎层 `qzen_qwen21_pipeline` 增加 `init_latent` / `strength` / `latent_mask` / `masking_strength`；参考后端按同一 `first_step` 规则缩短 `dit` 步并把编辑参数纳入确定性哈希；.NET 新增 `QziRasterImage`（无依赖 PNG 解码）/ `Qwen21EditOptions` / `GenerateQwen21Edit(Async)` / `SupportsImageEditing` / `GenerationPhase.ImageEncode` 与 `AbiGuard` 增量探测；WebUI 新增 `qwen21-edit` 模式、参考图上传与 canvas 笔刷 mask、4 个 snake_case 字段、`supports_image_editing` 与 job `edit` / `has_mask`；补 21 项托管测试、C smoke 编辑校验 / 能力位 / legacy backend、5 份文档 + 本计划 + backend 模板同步；真实 CUDA 双轮真图验证通过（见下方「阶段 15 本机验证结果」） | ✅ |
| 16 | **Z-Image Turbo Fun ControlNet Union 2.1 / 对齐 WanGP**（用户要求「Z-image 应该也支持 controlnet，对齐wangp版本的功能」）：第二个 **ABI v1 增量扩展**——`qzi_zimage_control_request`（`base` + 控制图 / type / scale）、`QZI_FEATURE_ZIMAGE_CONTROL = 0x2`、`qzi_zimage_control_layout()`、`qzi_generate_zimage_control()`，`qzi_abi_features()` 当时返回 `0x3`（阶段 17 后为 `0x7`）；`qzi_backend` 末尾再追加可选 `generate_zimage_control` 槽位，旧槽位偏移与 6 个 layout 槽不变。`engine/src/qzen_zimage_control.c/.h` 加载 33 通道 Union 2.x 权重（15 control layers + 2 独立 noise refiner、明确拒绝 v1）；`engine/src/qzen_zimage_pipeline.c` 补齐此前**只有声明没有实现**的 `qzen_zpipe_denoise_control()` 与 CPU/GPU hint 注入，按 WanGP `control_layers_places = [0,2,...,28]` 在偶数 main layer 与两个 noise refiner 后以 `hidden_states += control_scale * skip` 注入，普通 t2i 传 `NULL` hints 保持逐位一致。真实 CUDA 后端 `qzi_engine_backend.c` 把 t2i / ControlNet 合成一个 `qzen_generate_zimage_impl()`：懒加载 Union 权重、控制图 cover fit + `qzen_zvae_encode()`、拼 33 通道 context、逐步重算 hints、复用 VAE 解码；`release_zimage()` 与 `done:` 统一释放。配置链补 `zimage_control_dir` / `QZI_ZIMAGE_CONTROL_DIR` / `QziConfig` / `NativeConfigScope`。.NET 新增 `ZImageControlOptions` / `GenerateZImageControl(Async)` / `SupportsControlNet` / `AbiGuard` 第二能力探测；WebUI 新增 `zimage-control` 模式、控制图上传、`control_image` / `control_type` / `control_scale`、`supports_controlnet` 与 job `control` / `has_control`，并与编辑模式互斥；参考后端把控制字段纳入确定性哈希并模拟 `"image encode"`。补 25 项托管测试、C smoke 双能力位 / 双 layout / 参数边界 / legacy backend、backend 模板第二个 `NULL` 槽，以及 5 份文档 + 本计划同步 | ✅ |
| 17 | **Qwen-Image 2.1 高级编辑 / 继续对齐 WanGP**（用户要求「继续推进开发，对齐 WanGP 版本的功能」；阶段 15/16 已完成并提交，工作区里留着上一轮未收尾的第三个增量）：**Red Canvas 外扩**（四边 32 对齐边距、内部 cover fit、固定 prompt 后缀、边距跑完整去噪调度）、**LanPaint**（`model_mode` 2/3/4/5 → 2/5/10/15 次内嵌 Langevin 迭代，C 移植 WanGP `shared/lanpaint.py` Algorithms 1–4，需 mask、与外扩互斥、强制两个 strength = 1.0）、**RGBA 输出**（VAE `conv_out` 第 4 通道 → PNG colour type 6）。第三个 **ABI v1 增量扩展**：`qzi_qwen21_edit_ex_request`（内嵌普通编辑请求）、`QZI_FEATURE_QWEN21_EDIT_ADV = 0x4`、`qzi_qwen21_edit_ex_layout()`、`qzi_generate_qwen21_edit_ex()`，`qzi_abi_features()` 返回 `0x7`；`qzi_backend` 末尾追加第 10 个可选槽（模板三个 `NULL`）。引擎层 `qzen_qwen21_sample` 增加 `outpainting` / `lanpaint_steps` / `lanpaint_seed` / `alpha`，`qzen_qwen21_vae_decode_ex()` 与 `qzi_png_encode_rgba8()` 打通 CPU/CUDA 双解码路径，原 RGB 入口保留为薄封装；参考后端把高级字段纳入确定性哈希并产出 RGBA PNG。.NET 新增 `Qwen21EditExOptions` / `Qwen21EditMode` / `GenerateQwen21EditEx(Async)` / `SupportsAdvancedEditing` / `AbiGuard` 第三能力探测（前置依赖普通编辑能力）/ `RequestValidator.Validate(Qwen21EditExOptions)` / `NativeStructs` 布局 / `Invoke` 五路分派；WebUI 新增高级编辑面板（四边距 / model mode / RGBA）、`supports_advanced_editing` 与 job 的 `advanced_edit` / `outpaint` / `rgba` badge，且**仅当出现高级字段时**才把普通编辑升级成 `Qwen21EditExOptions`，普通编辑仍兼容旧库；补 xUnit 与 C smoke 用例、6 份文档 + 后端模板同步 | ✅ |
| 18 | **阶段 17 收尾审计：C smoke 接线 + WanGP 边界复核**（用户要求「请继续执行任务」）：审计发现 `native/tests/qzi_abi_smoke.c` 中阶段 17 新增的 `check_edit_ex_validation()` / `check_edit_ex_generation()` **已定义却从未在 `main()` 里调用**（死代码），而 README / `docs/NATIVE_ABI.md` / `docs/ENGINE_INTEGRATION.md` / 本计划均已声称覆盖「三个能力位 / 三个扩展 layout / 高级编辑参数边界 / 完整生成契约」。本轮补上 `QZI_FEATURE_QWEN21_EDIT_ADV` 能力位与 `qzi_qwen21_edit_ex_layout()` 断言，把 7 条高级编辑参数边界接入 `extra_rejected`、4 次高级编辑生成接入 `health.served`（`14 → 18`），并同步 resident model 注释；legacy backend 第三个 `NULL` 槽断言本就存在，复核保留。另按 WanGP 本机 `5b63ae9d` 源码改写 #10 边界：20B 变体已不存在、编辑 LoRA 已由 ABI 覆盖、`EDITING` 属应用层 prompt 重写、多参考图需 Qwen3-VL-8B vision tower、Turbo 8 条 sigma 与本引擎逐值一致、KV cache 属性能选项。未新增 ABI 导出 / 结构体 / .NET API | ✅ |
| 19 | **Z-Image Turbo Fun ControlNet Union 2.x 遮罩重绘 / 继续对齐 WanGP**（用户要求「Qwen-Image 2.1、Z-image 功能要完整实现，对齐 wangp 版本的功能，包括 controlnet、编辑图像等功能」）：接手工作区里**上一轮未提交的第四个 ABI v1 增量**（17 个修改文件 + 2 个新测试共约 1,640 行），审计后补齐收尾。WanGP `models/z_image/pipeline_z_image.py` 中 `control_in_dim == 33 && inpaint_mask is not None` 分支此前在本库只实现了一半：普通 ControlNet 把 `keep_mask` / `inpaint_latent` 恒置 0。本轮补齐 `qzi_zimage_inpaint_request`（内嵌普通 ControlNet 请求 + 必填单通道 mask）、`QZI_FEATURE_ZIMAGE_INPAINT = 0x8`、`qzi_zimage_inpaint_layout()`、`qzi_generate_zimage_inpaint()`，`qzi_abi_features()` 返回 `0xF`；`qzi.c` 新增 `qzi_validate_zimage_inpaint()` / 归一化（继承控制校验 + mask 指针 / 尺寸 / flags）；`qzi_backend` 末尾追加**第 11 个**可选槽，两个内置 vtable 挂满、模板四个槽留 `NULL`。引擎层 `qzen_zpipe_control_context_ex()` 作为原 builder 的超集（两参数都 `NULL` 时逐位复现、只给一个或 v1 权重明确报错），真实后端 `qzen_generate_zimage_impl()` 增加 `inpaint` 形参并统一 `control_eff` 指针：控制图与 mask 各自 cover fit、`inpaint_rgb = control*(1-mask)` 第二次 `qzen_zvae_encode()`、`downsample_mask_keep()` 用 8×8 面积均值得到软 keep 平面，拼 `[control_latent(16) \| keep_mask(1) \| inpaint_latent(16)]`，`done:` 统一释放四块新缓冲；`adv == NULL` 的 t2i / 普通编辑路径零改动。.NET 新增 `ZImageInpaintOptions` / `GenerateZImageInpaint(Async)` / `SupportsInpaint` / `AbiGuard` 第四能力探测（**前置依赖 ControlNet**）/ `RequestValidator.Validate` / `QziZImageInpaintRequest` / `Invoke` 六路分派；WebUI 新增 `zimage-inpaint` 模式、复用控制图上传与 qwen21 笔刷 mask 画布、`control_image + mask_image` 自动升级、`supports_inpaint` 与 job `inpaint` / `has_mask` badge，编辑与 inpaint 仍互斥。参考后端把 mask 内容 / 尺寸纳入确定性哈希，并让 `run_phases()` 的 `editing` 变成「VAE 编码阶段数」以真实上报**两个** `"image encode"`。**收尾补齐**：`native/tests/qzi_abi_smoke.c` 接入第四能力位 / layout / 8 条参数边界 / 5 次参考后端生成 / legacy 第 11 槽断言与 `served` 计数（阶段 18 修过的同类死代码问题本轮未再复发）；`integration/custom_backend_template.c` 补第四个 `NULL`；6 份文档 + 本计划 + 发布检查清单同步 | ✅ |
| 20 | **反向回调崩溃根治 + WebUI 控制图预处理 / 继续对齐 WanGP**（用户三个现场问题：① `QziClient.cs:795` 的 `System.ExecutionEngineException` / `HResult=0x80131506`；② WebUI 提示「Upload an already preprocessed control image ... consumes the control raster exactly as supplied ("raw" semantics)」把手工预处理变成默认负担；③ 前端提示「Control image loaded (1024×683). Used as-is (no preprocessing).」）。**分层决策（按用户要求）**：`QwenZImage.NET` 仍只含核心模型与推理、零三方依赖，C ABI 的 `control_rgb` **raw 契约永久不变**（不新增预处理参数、不改 `qzi_abi_features()` / vtable / 引擎）；所有 estimator 只在允许引入三方库的 `QwenZImage.NET.WebUI` host 层。**① 崩溃**：旧 `CallbackScope` 把 `GCHandle.ToIntPtr()` 直接当 `QziCallbacks.User`，late/stale 回调可能解引用已释放句柄；`IProgress<T>.Report`（尤其 SSE subscriber）在 `[UnmanagedCallersOnly] OnProgress` 里抛异常时会试图穿过 CUDA/ggml/C++ 栈展开，触发 CLR fatal `0x80131506`。改为进程级 `ConcurrentDictionary<long, CallbackState> Live` + 单调递增 token，`Resolve(void*)` 对 stale/garbage token 返回 `null`；`OnProgress` 捕获**所有** `Exception` 并 `Record` 首个 fault；`OnShouldCancel` 见 fault 返回 `1` 让 native loop 停止；`QziClient.Invoke(...)` 签名改收 `CallbackScope scope`，P/Invoke 返回后 `scope.TakeFault()` → 释放可能的 image → 托管侧抛 `InvalidOperationException`（inner 指向真实 UI/SSE 异常）。`QziGenerationQueue.Publish` 包裹 `JobChanged?.Invoke` 与 `channel.Writer.TryWrite`，`SynchronousProgress<T>.Report` 同样吞掉 observer 异常；WebUI `.csproj` 设 `<CETCompat>false</CETCompat>` 规避 CET + reverse P/Invoke 在本机调试器下的 fatal 路径。**② 预处理**：WebUI 新增 `Preprocessing/`（`ControlImagePreprocessor` / `PreprocessAssets` / `PoseEstimator` / `DenseMapEstimator` / `RasterOps` / `PngWriter`）与 `GET\|POST /api/preprocess`（`SemaphoreSlim` 串行 + `Task.Run`），算子 `auto` / `raw` / `gray` / `canny` / `scribble` / `depth` / `pose`，兼容 WanGP `wgp.py::process_map_video_guide` 的 `P→pose` / `D→depth` / `E→canny` / `S→scribble` / `C→gray` / `V→raw` 及 `PV/DV/EV` 组合；`Microsoft.ML.OnnxRuntime` **只在 WebUI 引用**，用于 WanGP YOLOX-L + DWPose SimCC 骨架、Depth Anything V2 / contour dense map；缺 ONNX 资产时 depth 回落 `managed-depth-proxy` 并回传 `warning`、canny/scribble 回落纯托管 Sobel/Canny/NMS/Zhang-Suen，pose 明确报错并提示 `QZI_PREPROCESS_DIR`。`preprocess_dir` / `QZI_PREPROCESS_DIR` 只被 host 层 `PreprocessAssets.Discover()` 读取，已加入 `QziConfig` 严格解析器但**不发布给 native**（`NativeConfigScope.Variables` / `qzen_config.c:env_overrides[]` 保持不变）。**③ 前端**：`app.js` 用 `controlSourcePng`（原始上传）与 `controlPng`（`/api/preprocess` 产出的 guide）双状态，只提交后者；删除 `Used as-is (no preprocessing)` toast，新增 guide 算子 / 最大边 / 重跑按钮与 provider / warning 预览，standalone ControlNet 默认 `canny`、inpaint 默认 `raw`，预处理未完成时阻止提交；`index.html` 的「must ALREADY be preprocessed」注释与空态文案同步。**测试**：新增 `CallbackFaultTests.cs`（7 项：throwing progress 封边并托管重抛 / 引擎故障后可继续使用 / 确定性输出不变 / fault 被记录并停止引擎 / stale token 被忽略 / 取消仍可用 / queue 的坏 observer 被吞）与 `ControlImagePreprocessingTests.cs`（7 个算子、WanGP 字母、PNG roundtrip、Canny 边缘定位、asset discovery、wire format、错误参数）。文档 5 份 + README + 本计划同步 | ✅ |

本机验证结果（D10 完成度）：

- `dotnet build QwenZImage.NET.sln -c Release`：0 警告 0 错误（5 个项目）。
- `dotnet test QwenZImage.NET.sln -c Release`：**42 + 4 = 46** 全部通过（新增 `SearchPaths_Walk_Up_From_The_Project_Directory`）。
- WebUI 运行时冒烟（Windows、参考后端）：`/api/health` 返回 `transport: native-pinvoke`、自动解析出仓库根 `QwenZImage.config.json`、job 完成并出现在画廊。修复了 `dotnet run` 从项目目录启动时找不到根配置的问题（`QziConfig.SearchPaths()` 增加有界向上的父目录遍历）。
- 参考后端 `ctest --test-dir native/build`：1/1；真实后端 `ctest --test-dir native/build-cuda`：1/1。
- 真实 CUDA 后端用 **MSVC 19.51.36260 + CUDA 13.3 + SM 89** 编译链接出 `QwenZImageNative.dll`（2.2 MB），与参考后端（105 KB）分目录存放，互不覆盖。
- `.NET` 端到端：WebUI 设 `QZI_NATIVE_LIBRARY` 指向 CUDA 构建后，`/api/health` 返回 `backend: "QwenZImageNative"`、`transport: "native-pinvoke"`、`reference_backend: false`、`config_path` 已解析。
- **真图验证**：提交 qwen21 turbo / 256×256 / seed 42 / 8 steps，约 2 分钟后产出 196,947 字节 PNG，内容为清晰的雪地红狐，**不再是噪点**。
- `dotnet pack` 产出 `QwenZImage.NET.0.1.0.nupkg` + `.snupkg`（含两个 RID native）。

阶段 12b 本机验证结果（本轮新增）：

- Windows 真实后端以**默认 fat binary**（`80;86;87;88;89;90;100;103;110;120;121`）重编通过：MSVC 19.51.36260 + CUDA 13.3，`QwenZImageNative.dll` 22,176,768 字节，生成的 `qzen_arch_config.h` 为 `QZEN_SM_ARCH 0` / `QZEN_SM_ARCH_LIST "80,...,121"`，且该字符串确实嵌入 DLL（架构提示宏生效）；`ctest` 1/1。
- Linux 真实后端（WSL Debian 12 / GCC 12.2 + CUDA 13.0）先以单 SM 89、再以**默认 fat binary** 各重编一次，`ctest` 均 1/1；最终 `libQwenZImageNative.so` 22,434,336 字节，`readelf -d` 的 `NEEDED` **不再含 `libicuuc.so.72`**，改为正常的 `libgomp.so.1`，未解析 ICU 符号为 0，架构列表宏同样嵌入产物。此前该 `.so` 带版本化 ICU 依赖，跨发行版会加载失败。
- 修掉两个既有构建缺陷：① OpenMP 只以 `COMPILE_LANGUAGE` 保护的编译选项传入，链接阶段求值为空，导致 `.so` 留下未解析 `omp_*` / `GOMP_parallel` 并让 smoke 链接失败 → 改用 `OpenMP::OpenMP_C` 显式链接；② `scripts/stage-native.sh` 的 `find -path '*/-cuda/*'` 永远匹配不到 `Linux-x86_64-cuda`，CUDA 产物 stage 不到且 reference 排除规则失效 → 改为 `'*-cuda/*'`，两种 flavour 均已实测。
- `dotnet build` 0 警告 0 错误（5 项目）；`dotnet test` **42 + 4 = 46** 全绿；`dotnet pack` 产出 16 项包（含 `lib/net10.0/*.dll`、两个 RID native、4 份 docs、`engine/LICENSE`）。打包前已把 RID staging 恢复为 reference 后端。
- POSIX ICU 兼容层另做了**独立运行时双路径验证**：正常链接（`dlopen` 命中系统 ICU 72）与用 `-Wl,--wrap=dlopen` 强制「无 ICU」两条路径，字符分类、UTF-8/UTF-16（含 4 字节代理对）round-trip、非法 UTF-8 报错、NFC 调用全部通过；前者把 `e + U+0301` 正确组合为 `é`，后者按设计退化为恒等归一化。该一次性检查脚本当时放在 `.scratch/`（git 忽略）；**阶段 12e 已把它整理成仓库内正式 CTest**，两条路径由两个独立进程分别验证。

阶段 12c 本机验证结果（本轮新增）：

- Windows / Linux **参考后端**均以最终源码重建 + `ctest` 各 1/1；C smoke 覆盖 LoRA 裸文件名 / 长度 / scale 边界、无 LoRA 带 scale、uppercase `SVDQ-FP4-r256`、adapter 与 scale 切换触发 `"model load"` phase、以及 LoRA 参数确实改变确定性 PNG 字节。顺手修掉 smoke 里 `check_lora_validation()` 把 `error` 指针的 `sizeof` 当容量传的缺陷（改为显式 `error_cap`）。
- Windows（MSVC 19.51 + CUDA 13.3）与 Linux（WSL GCC 12.2 + CUDA 13.0）**真实 CUDA fat binary** 各重建一次、`ctest` 各 1/1：`QwenZImageNative.dll` 22,178,304 B、`.so` 22,434,432 B；两者 `qzen_arch_config.h` 均为 11 个 SM 列表且字符串确实嵌入产物；Linux `readelf -d` 的 `NEEDED` 仍无 `libicuuc.so.NN`，未解析 ICU 符号为 0。
- `dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误；`dotnet test` **54 + 7 = 61** 全绿（原 46 → 新增 15 项：LoRA 名称 / 长度 / finite scale / 双 tier、base guidance finite、配置覆盖链、WebUI 双 tier 映射等）。`QziConfigTests` 关闭并行以免环境变量测试互相干扰。
- `dotnet pack` 产出 `.nupkg` + `.snupkg`，包内含 `lib/net10.0/*.dll`、两个 RID native、4 份 docs、`engine/LICENSE`；打包前已把 RID staging 从 CUDA flavour 恢复为参考后端（win 106,067 B / linux 34,152 B）。
- 文档一致性：`PARAMETERS.md`（LoRA 契约、ABI sentinel 归一化表、DiT 重载）、`NATIVE_ABI.md`（`QZI_LORA_NAME_MAX`、归一化与 LoRA 契约）、`ENGINE_INTEGRATION.md`（配置覆盖优先级、真实后端 LoRA 状态机、进程级 `QZI_*` 边界）、`WEBUI.md`（双 tier LoRA、环境变量语义）、README、本计划全部同步。
- 用户明确「不需要适配 macOS」，阶段 12c 及后续节奏均不含 macOS 平台工作。阶段 13（push / publish）当时仍按 D6 暂缓，后续由 D11 改为执行。

阶段 12d 本机验证结果（本轮新增）：

- RTX 4090 24GB + Windows CUDA fat binary `QwenZImageNative.dll`（SM 80..121）跑完整 10-case 矩阵，**10/10 PASS**，`engine counters: served=10 rejected=0`。报告在 `artifacts/real-image-regression/full/`（git 忽略），含 `report.md` / `report.json` 与 10 张 PNG。
- **Qwen21**：`turbo 512` 787,072 B / 126.9 s；同参数重复请求 SHA-256 完全相同（`compare_with` = `identical`），真实引擎确定性成立。LoRA `qwen_image_2.1_DMD2-2steps.safetensors` 的 scale 1.0 / 0.6 各产出不同 SHA（无 LoRA `622f020b…`、1.0 `34c82cad…`、0.6 `cd04e575…`），引擎日志显示 adapter 合并进 `128/128 projections`；`1024×1024` 3,147,060 B / 122.8 s；`base` + `negative_prompt` 192.5 s（text encoder 跑条件 / 非条件两遍）。
- **Z-Image**：`bf16 512` 118.2 s、`bf16 1024` 10.2 s（权重常驻后）、`nunchaku-r32` 85.7 s、`nunchaku-r256` 61.9 s，四者 SHA 互不相同；checkpoint 切换都触发 `ModelLoad`。
- phase trace 与阶段 12c 的状态机一致：首个用例及每次换 tier / LoRA / scale / checkpoint 均为 `ModelLoad → Text → Dit → Vae → Encode`；只有「参数完全相同的重复请求」和「bf16 512 → 1024」复用常驻权重、不出现 `ModelLoad`。
- 人工抽查 6 张（qwen21 基线 / LoRA / 1024、Z-Image Nunchaku r32 / r256、Z-Image bf16 1024）均为清晰的雪地红狐照片，无噪点、无平铺或截断。512 PNG 字节数一律 787,072 是 native 编码器行 filter 全 0、固定开销所致，与图像内容无关。
- 审计顺带发现：`qzi_engine_backend.c` 里 `nunchaku-fp4-r<rank>` 分支的 `checkpoint + 13` 偏移错误（应为 `+14`），且该分支因 ABI 已拒绝所有含 `fp4` 的 checkpoint 而**不可达**。已在阶段 12e 直接删除该 fp4 分支、只保留 `nunchaku` / `nunchaku-r<rank>` 解析，使引擎与 ABI 契约一致；改动是不可达路径清理，不影响本次矩阵结果。
- 工具自身改动：报告改为**逐用例增量落盘**（避免中途 CUDA 致命失败丢结果），并把每个用例的 phase 序列写进 `report.json` / `report.md`。新工具与脚本编译 0 警告 0 错误，仍不加入 `QwenZImage.NET.sln`。

阶段 12e 本机验证结果（本轮新增）：

- Windows 参考后端 `ctest` 1/1（新测试按 `NOT WIN32` 条件不注册）；WSL Linux 参考后端 `ctest` **3/3**：`qzi_abi_smoke`、`posix_unicode_with_icu`（`dlopen` 命中宿主 ICU，`e + U+0301` 组合成 `é`）、`posix_unicode_without_icu`（`-Wl,--wrap=dlopen` 强制无 ICU，恒等 NFC）。两条路径共同验证字符分类、空白判断、UTF-8 ↔ UTF-16（含 U+1F98A 代理对）字节级 round-trip、非法 UTF-8 报错、已组合文本 NFC 不变。
- 优雅降级：宿主没有可用 ICU 时 `posix_unicode_with_icu` 以退出码 77 被 CTest 记为 **SKIP**（共享断言仍全部执行）；链接器不支持 `--wrap` 时 configure 打印 STATUS、只注册 ICU 那一个变体，不会构建失败。测试不依赖 CUDA，GPU-less 的 Linux CI 在 reference 构建里就会跑。
- 真实 CUDA 后端因 fp4 清理增量重建：Windows（Ninja + MSVC 19.51 + CUDA 13.3）`ctest` 1/1，DLL 22,177,280 B；WSL（GCC 12.2 + CUDA 13.0）`ctest` **3/3**，`.so` 22,434,432 B；`readelf -d` 的 `NEEDED` 仍无 `libicuuc.so.NN`、未解析 ICU 符号 0，`qzen_arch_config.h` 仍是 11 个 SM。
- `dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误；`dotnet test` **54 + 7 = 61** 全绿；`dotnet pack` 产出 `.nupkg` / `.snupkg`，包内两个 RID native 仍是 reference 大小（win 106,067 B / linux 34,152 B），未被 CUDA 构建覆盖。

阶段 12f 本机验证结果（本轮新增）：

- Windows 参考后端以最终源码重建（新增 `src/qzi_env.c`），`ctest --test-dir native/build -C Release` **1/1**；smoke 新增的 4 条环境断言通过：未设置读为 0、Win32 API 在 CRT 启动后设置的值 `qzi_getenv()` 能读到、超长值返回 -1 且不静默截断、删除后读为 0。用最小 DLL 单独实测确认 **MinGW CRT 的 `getenv()` 在同一情形下始终返回 NULL**（这是审计定位到的根因）。
- `dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误（5 项目）；`dotnet test` **62 + 7 = 69** 全绿（新增 8 项作用域 / 快照 / 并发 / 回滚测试）。测试程序集本来就通过 `CollectionBehavior(DisableTestParallelization = true)` 串行运行，避免环境变量测试互相干扰。
- 修复前基线：`Two_Engines_Snapshot_Their_Own_Config` 因 Windows 上 native 读不到作用域变量而三张 PNG 完全相同；`Concurrent_Client_Creation_Does_Not_Interleave_Scopes` 还用了非法的 16×16 画布（已改为 32×32）。修复后两项均通过，证明不同 LoRA / Nunchaku 目录的两个 engine 产出不同 SHA、相同配置字节级一致、8 个并发构造的 8 张图片哈希互异。
- Windows 真实 CUDA fat binary 增量重建（Ninja + MSVC 14.51.36231 + CUDA 13.3）：`ctest` **1/1**，`QwenZImageNative.dll` 22,184,960 B；Linux 真实 CUDA fat binary 增量重建（WSL GCC 12.2 + CUDA 13.0）：`ctest` **3/3**，`.so` 22,434,496 B，`readelf -d` 的 `NEEDED` 仍无 `libicuuc.so.NN`，`qzen_arch_config.h` 仍为 11 个 SM。两者都因 `qzen_config.c` / `qzi_env.c` 改动重编，配置链（`QZI_CONFIG` / `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR`）在 Windows 真实后端上现在也能收到托管侧发布的值。
- Linux **参考后端**重建后 `ctest` **3/3**（`qzi_abi_smoke` + 两个 POSIX ICU 路径）；Windows 参考后端 1/1。两者 smoke 都新增了作用域环境变量断言。
- `bash scripts/stage-native.sh` 把两个 RID staging 恢复为**参考后端**（win 106,637 B / linux 38,440 B），随后 `dotnet pack` 产出 `.nupkg` + `.snupkg`，包内 16 项：`lib/net10.0/*.dll`+`.xml`、两个 RID native、4 份 docs、`README.md` / `LICENSE` / `THIRD_PARTY_NOTICES.md` / `engine/LICENSE`。
- WebUI 运行时冒烟（Windows、参考后端，端口 7860）：`/api/health` 返回 `ok=true`、`transport: native-pinvoke`、`reference_backend: true`、`config_path` 自动解析到仓库根 `QwenZImage.config.json`；提交 qwen21 turbo / 32×32 / seed 7 后 job 变为 `done`，产出 3,172 B PNG 并出现在画廊，未因配置作用域化出现回归。

阶段 13 执行结果（本轮新增）：

- **仓库整洁度审计**：当前 tree 的 163 个文件全部是源码 / 文档 / 配置 / 脚本 / CI 定义，0 个未跟踪文件；11 个历史提交中从未加入过二进制、图片或构建产物（最大的两个 blob 都是 `engine/src/qzen_gpu.cu` 源码版本）。唯一命中的 `artifacts/.gitkeep` 是 0 字节目录占位，与 `.gitignore` 的 `artifacts/*` + `!artifacts/.gitkeep` 配套，保证 clone 后产物根目录存在，按结构文件保留而非删除。
- **远程与推送**：`origin` 已用 `git remote set-url` 切到 `https://github.com/dongfangzhizhu/QwenZImage.NET.git`（原 `origin` 已存在，直接 `git remote add` 会失败）；远端原为空仓库，`git push -u origin main` 一次推送 11 个提交成功，无历史冲突。
- **提交 `5562ffa`**：同步 `csproj` 的 `Authors` / `PackageProjectUrl` / `RepositoryUrl`、WebUI 页脚链接、PLAN 与 README；`release.yml` 改为 OIDC Trusted Publishing（`publish` job `id-token: write` + 单次 `NuGet/login@v1` + 短期 key 复用于 `.nupkg` / `.snupkg`，workflow 级权限降为 `contents: read`）；删除全部 `secrets.NUGET_API_KEY` 依赖与静态 key 发布命令。
- **本机验证**：`git diff --check` 干净；`dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误（5 项目）；`dotnet test` **62 + 7 = 69** 全绿；Windows 参考后端 `ctest` **1/1**；`dotnet pack` 产出 16 项 `.nupkg` 与 5 项 `.snupkg`，nuspec 的 `<authors>` / `<projectUrl>` / `<repository url>` 均已指向 `dongfangzhizhu/QwenZImage.NET`；`release.yml` 经 YAML 解析确认权限块与 login 步骤正确。
- **尚未执行 tag 发布**：首次实际发布仍需用户在 nuget.org 建 Trusted Publishing policy、在 GitHub 设 `NUGET_USER` secret（见第 5 节），之后推 `v*` tag 触发；刻意不在策略缺失时打 tag，避免触发必然失败的发布工作流。

阶段 14 本机验证结果（本轮新增）：

- **根因定位**：本机 RTX 4090（SM 8.9 / 24 GB）与 22,184,960 B 的 CUDA fat binary 均可用，模型权重目录齐全。`artifacts/runtimes/win-x64/native/QwenZImageNative.dll` 却是 106,637 B 的参考后端——阶段 12f 与阶段 13 为跑无 GPU 测试 / 保持 NuGet 默认包而执行 `stage-native` 时把 CUDA 产物换回去了；WebUI 的 csproj 正是从该目录复制 native。三个叠加原因：构建 CUDA 只是产出到 `artifacts/native/*-cuda/` 而**未选中**、staging 被后续流程**覆盖回参考库**、显式路径加载失败时 loader **静默回退**到 stub。页面横幅与 README 当时只说「构建 CUDA 后端」，没说必须选中，导致反复复发。
- **修复后默认启动实测**：不带 `QZI_NATIVE_LIBRARY` / `QZI_WEBUI_NATIVE` / `QZI_CONFIG` 直接 `dotnet run`，`/api/health` 返回 `ok=true`、`native_library` = `artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll`、`native_library_source` = `local CUDA build (auto-detected)`、`engine.reference_backend` = **false**。提交 qwen21 turbo / 512² / seed 42，phase 为 `ModelLoad → Text → Dit → Vae → Encode`，116.4 s 后产出 **787,072 B** PNG（与显式设置 `QZI_NATIVE_LIBRARY` 时字节级一致），目视确认为清晰雪地红狐，无噪点。
- **fail-fast 实测**：`QZI_NATIVE_LIBRARY` 指向不存在的路径时，`/api/health` 为 `ok=false` 且错误为 `DllNotFoundException`（含完整路径与「未替换参考后端」说明），`POST /api/jobs` 返回 **503**，不再生成噪点图。
- **拒绝噪点实测**：`QZI_WEBUI_NATIVE=reference` + `QZI_REQUIRE_REAL_BACKEND=1` 时 health 为 `ok=false` 并说明原因，`POST /api/jobs` 同样 **503**。
- `git diff --check` 干净；`dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误（5 项目）；`dotnet test` **62 + 11 = 73** 全绿（新增 4 项：自动探测 / 显式优先 / 强制 reference / 拒绝噪点开关）。RID staging 仍保持参考后端，NuGet 包内容与 CI 的 GPU-free 假设未被改动。

阶段 15 本机验证结果（本轮新增）：

- `dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误；`dotnet test` **72 + 22 = 94** 全绿（新增 21 项编辑相关托管测试）。Windows 参考后端 `ctest` **1/1**；全新目录 `native/build-verify` 从零 configure + build + `ctest` **1/1**；真实 CUDA fat binary `ctest` **1/1**。
- `objdump` 确认 CUDA `QwenZImageNative.dll` 导出新增的 `qzi_abi_features()`、`qzi_generate_qwen21_edit()`、`qzi_qwen21_edit_layout()` 三个符号；`gcc -std=c11 -Wall -Wextra -c integration/custom_backend_template.c` 通过（模板末尾编辑槽显式留 `NULL`，旧 backend 仍可编译）。`git diff --check` 干净（仅 CRLF 提示）。
- WebUI 以 CUDA 后端启动（`QZI_NATIVE_LIBRARY` 指向 `artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll`、`QZI_REQUIRE_REAL_BACKEND=1`），`/api/health` 返回 `ok=true`、`transport=native-pinvoke`、`native_library_source=QZI_NATIVE_LIBRARY`、`supports_image_editing=true`、`engine.backend=QwenZImageNative`、`reference_backend=false`。
- **真实 CUDA 编辑验证（512×512 / turbo / seed 42 / prompt「a red fox walking through fresh snow, cinematic light」）**：参考图为「左半蓝色底 + 白圆，右半纯橙色」，mask 为「左半纯黑、右半纯白」，`masking_strength=1.0`。两轮均走 `POST /api/jobs` 的 `model=qwen21-edit` + `reference_image` / `mask_image` 上传，输出 787,072 B 的 512×512 RGB PNG，非噪点。
  - `strength=0.6`：phase 序列 `ModelLoad(0s) → Text(78.2s) → ImageEncode(159.5s) → Encode(192.7s done)`，出现 `"image encode"` 证明 VAE 编码参考图的路径真实执行；输出 `state=done`、`error=null`。参考图 vs 输出的半幅平均绝对差（MAD）为 **左半 0.541 / 右半 17.115**——黑 mask 区几乎逐像素锁定，白 mask 区开始变化但重绘幅度受 `strength` 限制。
  - `strength=1.0`：MAD **左半 0.981 / 右半 88.468**。左半 RGB 均值从参考的 `(62.2, 87.6, 155.3)` 到输出的 `(61.4, 87.5, 155.7)`，蓝色底与白圆完整保留；右半从 `(220.0, 120.0, 40.0)` 变为 `(171.0, 180.0, 193.2)`，橙色被完全替换成雪地场景与红狐，`comparison_strength1.png` 目视确认。
  - 两组数据由同一脚本按 2 像素步长采样计算，构成 mask 语义（白区重绘 / 黑区保持参考轨迹）与 WanGP `denoising_strength` 步数比例语义的直接证据：右半 MAD 随 `strength` 从 0.6 → 1.0 放大约 5.2 倍，左半始终 < 1。
- 验证产物落在 `artifacts/edit-verify/`（被 `.gitignore` 忽略），验证后清理；`native/build-verify/` 同样为一次性构建目录，不进版本库。真实编辑效果仍需要 CUDA 构建 + 权重，参考后端只做确定性字节对照。

阶段 16 本机验证结果（本轮新增）：

- `dotnet build QwenZImage.NET.sln -c Release` 0 警告 0 错误（5 项目）；`dotnet test` **86 + 33 = 119** 全绿。Windows 参考后端 `ctest` **1/1**；`scripts\build-native-cuda.bat 89`（默认 `native/build-cuda`，单 SM 8.9）23/23 链接 + `ctest` **1/1**；`gcc -std=c11 -Wall -Wextra -Inative/include -c integration/custom_backend_template.c` 通过；`git diff --check` 干净（仅 CRLF 提示）。
- `objdump -p artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll` 确认导出 `qzi_abi_features` / `qzi_generate_qwen21_edit` / `qzi_generate_zimage_control` / `qzi_qwen21_edit_layout` / `qzi_zimage_control_layout` 五个增量符号；`dotnet pack` 产出 16 项 `.nupkg` 与 `.snupkg`，包内两个 RID native 仍为参考后端（115,875 B DLL / 38,440 B SO）。
- WebUI 以 CUDA 后端启动（`QZI_NATIVE_LIBRARY` 指向 `artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll`、`QZI_REQUIRE_REAL_BACKEND=1`、`--urls http://127.0.0.1:17860`），`/api/health` 返回 `ok=true`、`supports_image_editing=true`、`supports_controlnet=true`、`reference_backend=false`、`native_library_source=QZI_NATIVE_LIBRARY`。
- **真实 CUDA ControlNet 真图矩阵（512×512 / bf16 / seed 42 / 8 steps / prompt「a red fox walking through fresh snow, cinematic light」）**：控制图为脚本生成的 8 位非隔行 RGB PNG（A = pose 风格火柴人 2,352 B；B = canny 风格网格 6,545 B），经 `POST /api/jobs` 的 `model=zimage-control` + `control_image` / `control_type` / `control_scale` 提交。四例均产出 787,072 B 的 512×512 RGB PNG、`state=done`、`error=null`，phase 含 `ImageEncode`（控制图 VAE 编码）与 `Dit`。
  - `control_a`（pose / 0.65）SHA-256 `bbe49c89…`；`control_b`（canny / 0.65）`fad022be…`，两者 MAD **61.108** —— 不同控制图得到不同结果，hint 确实进入主 DiT。
  - `control_a_lowscale`（同图 / 0.05）`b42698e3…`，对 `control_a` MAD **99.296** —— `control_scale` 生效。
  - `t2i_baseline`（`model=zimage`、无控制图）`57bc5d16…`，对 `control_a` MAD **99.140**；其 SHA-256 与修复前单独跑的 t2i 探针完全一致，`control_a` 的 SHA-256 也与修复前成功复跑的结果完全一致 —— 普通 t2i 路径与 hint 注入的数值均未被改动。
  - 两次**全新进程冷启动**（停掉 WebUI 后重启，`resident_model=none`）把 ControlNet 作为第一个请求提交，均一次成功；修复前该场景在首个 `Dit` 步报 `denoise: cannot stage Z-Image GPU input`。
- **修复的根因与改动**：`engine/src/qzen_zimage_dit.c` 的 `qzen_zimage_block_forward_gpu_host()` 每次调用都 `malloc(seq*dim*2)` 再 `free`，而 ControlNet 每个去噪步要跑约 50 次 block forward（15 control layers 两遍 + 2+2 refiner + 主 DiT）；叠加常驻 DiT / Union 2.1 双份主机与设备权重后进程私有提交量约 45 GiB，CRT 堆上的大块分配在首次冷启动时偶发返回 `NULL`。改为让 `qzen_zimage_gpu` 持有一个 grow-only 的 `host_stage` 缓冲：在 `qzen_zimage_gpu_reserve()` 中按 `plane = seq*dim` 一次性分配并给出带行数 / MiB 的明确错误，`zimg_read_bf16_f32()` 与 `qzen_zimage_t_embedder_forward_gpu()` 复用同一缓冲，`qzen_zimage_gpu_free()` 统一释放。`qzen_gpu_tensor_from_bf16()` / `qzen_gpu_tensor_read_bf16()` 都是同步 `cudaMemcpy`，因此 H2D 上传与 D2H 下载复用同一主机缓冲是安全的。修复后首个请求 phase 为 `ModelLoad → Text → ModelLoad → ImageEncode → Dit → Encode`，无失败。
- 验证产物（控制图 / 输出 PNG、`report.json`、一次性 `verify.py` / `finish.py`）落在 `artifacts/control-verify/`，被 `.gitignore` 忽略；`native/build-cuda/` 为本地构建目录，同样不进版本库。验证完成后 RID staging 已用 `scripts/stage-native.sh` 恢复为参考后端。

阶段 17 本机验证结果（本轮新增）：

- 静态 / 构建：`dotnet build QwenZImage.NET.sln -c Release` **0 警告 0 错误**（5 项目）；`dotnet test --no-build` **97 + 42 = 139** 全绿。Windows 参考后端 `bash scripts/build-native.sh` + `ctest` **1/1**；`scripts\build-native-cuda.bat 89` **8/8** 链接 + `ctest` **1/1**；`gcc -std=c11 -Wall -Wextra -Werror -Inative/include -c integration/custom_backend_template.c` 通过；`git diff --check` 干净（仅 CRLF 提示）。
- `objdump -p artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll` 确认 **7 个**增量导出：`qzi_abi_features`、`qzi_qwen21_edit_layout`、`qzi_zimage_control_layout`、`qzi_qwen21_edit_ex_layout`、`qzi_generate_qwen21_edit`、`qzi_generate_zimage_control`、`qzi_generate_qwen21_edit_ex`（阶段 16 为 5 个）。`qzi_abi_features()` 返回 `0x7`，`qzi_backend` 共 10 个回调；参考 / 真实两个 vtable 都挂满三个可选槽，模板三个槽留 `NULL`，C smoke 新增「第三个槽为 `NULL` 时返回 `QZI_ERR_UNSUPPORTED`」的断言并在两种构建下都跑过。
- `dotnet pack QwenZImage.NET.sln -c Release --no-build` 产出 `.nupkg` + `.snupkg`，包内 **16 项**（`lib/net10.0` 程序集与 XML 文档、4 份 `docs/*.md`、README / LICENSE / `engine/LICENSE` / THIRD_PARTY_NOTICES、两个 RID native）；包内两个 RID native 仍是**参考后端**（120,716 B DLL / 38,440 B SO），与恢复后的 `artifacts/runtimes` 逐字节一致。WebUI 项目 `IsPackable=false` 的警告是预期行为。
- **真实 CUDA 高级编辑真图矩阵**（Qwen-Image 2.1 turbo / bf16 / seed 42 / 8 steps / prompt「a quiet mountain lake at dawn, cinematic light」；参考图与 mask 都是脚本生成的 256×256 8 位非隔行 PNG）：WebUI 以 `QZI_NATIVE_LIBRARY` 指向 CUDA 产物 + `QZI_REQUIRE_REAL_BACKEND=1` 启动，`/api/health` 返回 `ok=true`、`engine.reference_backend=false`、`supports_image_editing / supports_controlnet / supports_advanced_editing = true`、`native_library_source=QZI_NATIVE_LIBRARY`。四例 `state=done`、`error=null`：
  - `edit_plain`（普通编辑基线，256×256 RGB）196,947 B、colour type 2、SHA-256 `dbb5786d…`。
  - `outpaint`（四边各 32 px、最终画布 **320×320**）307,608 B、colour type 2、SHA-256 `6b89c495…`；与基线重叠 256×256 区 MAD **48.868** —— 红画布边距与放大后的画布确实进入采样。
  - `rgba`（`rgba=1`，256×256）262,488 B、PNG **colour type 6 / 4 通道**、SHA-256 `0614c7df…`；剥离 alpha 后 RGB 三平面与普通编辑基线 **MAD 0.000（逐字节一致）**，证明新增 alpha 不改变 RGB 数值；alpha 平面 min 238 / max 255 / 12 个不同取值，是 VAE `conv_out` 第 4 通道生成的非恒定值（**不是抠图 mask**）。
  - `lanpaint`（`model_mode=2`，每步 2 次内嵌 Langevin 迭代，省略 `strength` 由后端强制 1.0）196,947 B、colour type 2、SHA-256 `2c4a4bb4…`；对基线 MAD **25.273**。显式传 `strength=0.75` 时 `/api/jobs` 返回 **400**（`RequestValidator` 按设计拒绝而非静默覆盖），验证约束链在托管 / native / WebUI 三处生效。
- **本轮修掉的真实编译 / 接线问题**：`qzen_generate_qwen21_impl()` 新增 `adv` 形参后，t2i 与普通编辑两处旧调用少传一个实参（MSVC `C2198`），第三增量 adapter 与 `qzen_vtable` 第 10 个槽尚未接上；且编排体只认旧 `edit` 指针，高级请求走不到参考图编码 / mask 恢复。本轮统一改用 `edit_ref`，补齐 Red Canvas 合成（FF0000 边距 + 内部 cover fit + 边距 255 / 内部 0 的 mask）、`qzen_qwen21_sample` 四个新字段、RGBA 打包与 `emit_png_ex()`、`done:` 三处新增释放，并挂上 `qzen_generate_qwen21_edit_ex`。`adv == NULL` 的 t2i / 普通编辑 / ControlNet 路径保持零改动。
- 验证产物（`artifacts/adv-verify/` 下 4 张输出 PNG、`report.json` 与一次性 `verify.py`）落在 `.gitignore` 忽略的 `artifacts/` 内；验证完成后 WebUI 进程已停止、RID staging 已用 `scripts/stage-native.sh` 恢复为参考后端。

阶段 18 本机验证结果（本轮新增）：

- 只改 `native/tests/qzi_abi_smoke.c` 与 `PLAN.md`，无 ABI / 引擎 / .NET / WebUI 改动，因此不重跑真图矩阵。
- Windows 参考后端：`bash scripts/build-native.sh` 重建 + `ctest --test-dir native/build -C Release` **1/1**；新接线确实生效——`health.served == 18`、`health.rejected == 2 + extra_rejected`（含 7 条高级编辑边界）三条断言全过，说明 4 次高级编辑生成与 7 次拒绝都真实发生，而非只编译了函数。
- 真实 CUDA 后端：在已有 `native/build-cuda` 构建树上仅重编 `qzi_abi_smoke` 目标后 `ctest` **1/1**，验证高级编辑参数边界在真实后端（不加载权重）路径也跑通。MSVC / Windows SDK 的 `INCLUDE` / `LIB` 按文件系统路径直接给出（与脚本自身的 SDK 兜底同源），未调用 `reg.exe`，也未绕过任何安全策略。
- `dotnet build QwenZImage.NET.sln -c Release` **0 警告 0 错误**（5 项目）；`dotnet test --no-build` **97 + 42 = 139** 全绿；`git diff --check` 干净。
- **WanGP 复核结论**（`D:/AI/Wan2GP` @ `5b63ae9d`，2026-11-10）：`models/qwen21/qwen21_handler.py::query_supported_types()` 仅 `qwen_image_21_7B`；`prompt.py::_get_qwen_prompt_embeds()` 用 `processor(images=...)` + `pixel_values` / `image_grid_thw` / `image_pad_mask` 把每张参考图编成 attention prefix，确认多参考图必须引入 vision tower；`schedulers.py::sample_sigmas = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)` 与 `engine/src/qzen_qwen21_pipeline.c::q21_turbo_sigmas` **逐值一致**；`enhancer.py::EDITING` 是一段 prompt 重写指令文本，属应用层 LLM 能力。据此 #10 已去掉过时的「20B 变体」表述并写明各项归属。

阶段 19 本机验证结果（本轮新增）：

- 静态 / 构建：`dotnet build QwenZImage.NET.sln -c Release -m:1` **0 警告 0 错误**（5 项目）；`dotnet test --no-build` **110 + 52 = 162** 全绿。Windows 参考后端 `bash scripts/build-native.sh` + `ctest --test-dir native/build -C Release` **1/1**；真实后端 `ctest --test-dir native/build-cuda -C Release` **1/1**；`gcc -std=c11 -Wall -Wextra -Werror -Inative/include -c integration/custom_backend_template.c` 通过；`git diff --check` 干净（仅 CRLF 提示）。
- `objdump -p artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll` 确认 **9 个**增量导出：`qzi_abi_features`、四个 `qzi_*_layout()`（编辑 / 控制 / 高级编辑 / inpaint）、四个 `qzi_generate_*()` 增量入口（阶段 17 为 7 个）。`qzi_abi_features()` 返回 `0xF`，`qzi_backend` 共 11 个回调，末尾四个可选槽顺序固定。
- **本轮发现并修掉的发布隐患**：上一轮 CUDA 构建把 `artifacts/runtimes/win-x64/native` 覆盖成 2.2 MB 的 CUDA flavour，`dotnet test` 因而加载到真后端，`IsReferenceBackend` 等 26 项断言批量失败（110 项里 84 通过 / 26 失败）。用 `bash scripts/stage-native.sh` 恢复参考后端后 162 项全绿；该检查已写入发布前清单。
- `dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj -c Release --no-build` 产出 `src/QwenZImage.NET/bin/Release/QwenZImage.NET.0.1.0.nupkg`（204,123 B，**16 项**）：`lib/net10.0` 程序集与 XML 文档、4 份 `docs/*.md`、README / LICENSE / `engine/LICENSE` / THIRD_PARTY_NOTICES、两个 RID native；包内两个 native 与恢复后的 `artifacts/runtimes` **逐字节一致**（123,459 B DLL SHA-256 `882bf55a…` / 38,440 B SO `78828f1f…`），确认默认包仍是 GPU-free 参考后端。WebUI 项目 `IsPackable=false` 的警告是预期行为。
- **真实 CUDA 遮罩重绘真图矩阵**（`artifacts/inpaint-verify/verify.py`，WebUI 用 `QZI_NATIVE_LIBRARY` 指向 CUDA 产物 + `QZI_REQUIRE_REAL_BACKEND=1`，端口 17860；`/api/health` 返回 `ok=true`、`engine.reference_backend=false`、`supports_inpaint=true`、`native_library_source=QZI_NATIVE_LIBRARY`）。控制图是 256×256「左蓝底白圆 / 右橙色」，mask 是 8 位灰阶；统一 `prompt="a red fox walking through fresh snow, cinematic light"` / `seed=42` / `checkpoint=bf16` / `control_type=raw` / `control_scale=0.65`。**7 例全部 `state=done`、`error=null`**，均为 256×256 RGB、colour type 2、196,947 B，SHA-256 各不相同：`t2i_baseline cfcaf3fc…`、`control_plain 145902778…`、`inpaint_half 89ab4181…`、`inpaint_inverted cd5df5a2…`、`inpaint_all_keep 4ed95f43…`、`inpaint_low_scale 780bf912…`、`inpaint_diff_size 7524e4ab…`（最后一例的 mask 是 **128×256**，验证控制图与 mask 各自独立 cover fit）。作业 badge 也正确：inpaint 四例 `inpaint / has_mask / control / has_control = true`，普通 ControlNet 只有 `control / has_control`，t2i 全为 `false`。
  - **mask 极性成立**：相对全黑（全 keep）的 mask，`inpaint_half` 的左 / 右半幅 MAD 为 **36.765 / 118.612**（白区重绘明显更多），反转 mask 后变成 **104.578 / 8.243**，关系精确翻转；`inpaint_half` 对普通 ControlNet 的左 / 右 MAD 为 129.610 / 130.589，普通 ControlNet 对 t2i 的 MAD 为 **55.701**，证明控制图 hint、mask 后 17 平面都真实进入采样，而不是被忽略或退化成 plain ControlNet。
  - **约束链**：缺 `mask_image` / 缺 `control_image` / 非法 `control_type=bogus` / `control_scale=9.0` 四种请求都在 `/api/jobs` 返回 **400**，错误文本分别来自 `MapZImageInpaint` 与 `RequestValidator`，未排队占用 GPU。
  - **阶段上报实测**：SSE 捕获 89 个 progress 事件、12 个 `ImageEncode`，`client/total` 只有 `('stage19-verify', 1)` —— 真实 CUDA 后端把两次 VAE 编码折算进**单个** `0/1 → 1/1` 阶段，只有 GPU-free 参考后端按 `0/2 → 1/2 → 2/2` 拆两段。本轮据此修正了 `docs/PARAMETERS.md`、`docs/WEBUI.md` 与本计划 #5 / #14 中「inpaint 上报两次」的笼统表述。
  - **确定性**：矩阵完整跑了两轮（第二轮用修正后的阶段断言），7 例 SHA-256 前缀逐项一致，说明真后端在同一权重 / seed / 参数下可复现；`inpaint_low_scale`（0.2）与 `inpaint_diff_size` 都改变了输出字节，验证 `control_scale` 与 mask 尺寸确实贯通。
  - 产物（7 张 PNG、`report.json`、`verify.py`、WebUI outputs）落在 `.gitignore` 忽略的 `artifacts/inpaint-verify/`；验证完成后 WebUI 进程已停止，`artifacts/runtimes` 仍是参考后端（123,459 B DLL / 38,440 B SO）。

阶段 20 本机验证结果（本轮新增）：

- 静态 / 构建：`dotnet build QwenZImage.NET.sln -c Release` **0 警告 0 错误**（5 项目）；`dotnet test --no-build` **118 库 + 84 WebUI = 202** 全绿（阶段 19 为 162）。新增的 `CallbackFaultTests` 过滤运行先 2 项失败（`health.Served >= 1` 与 queue job 状态竞态），修正断言后 **7/7** 通过；`ControlImagePreprocessingTests` 并入 WebUI 84 项。
- **GPU-free 参考后端预处理全模式**（`.scratch/preprocess-verify/verify.py`，WebUI 端口 17861，1024×683 合成照片）：`raw → raw`、`gray → managed-gray`、`canny → managed-canny`、`scribble → managed-scribble`、`depth → managed-depth-proxy`、`pose → onnx-wan2gp-dwpose`（`learned=True`，8.17 s）、`auto → raw`，全部返回 1024×683 合法 PNG；再加 4 个错误用例（空图 / 非法 mode / 越界 `max_side` / 非法 base64），结果 `ALL OK: 7 modes + 4 error cases`。`/api/health` 的 `preprocessing` 块为 `learned_pose=true` / `learned_depth=false` / `learned_contour=false`（本机只找到 WanGP pose ONNX，depth 只有 `.pth`、无 contour ONNX，故按设计回退托管算子并回传 `warning`）。pose 在合成场景上输出全黑是合理结果——DWPose 未检测到 person 时不画骨架，真实 ONNX session 已成功加载并推理；尚未用真人照片复核骨架非全黑。
- **真实 CUDA 端到端（用户三个问题的合并场景）**（`.scratch/cuda-verify/verify.py`，WebUI 端口 17862，`QZI_NATIVE_LIBRARY` 指向 CUDA 产物 + `QZI_REQUIRE_REAL_BACKEND=1`）：`/api/health` 为 `reference=False` / `source=QZI_NATIVE_LIBRARY` / `controlnet=True` / `preprocessing_pose=True`；普通照片经 `/api/preprocess` 生成 `canny 512×512` guide（`managed-canny`，2,610 B）；提交 `zimage-control` job 后**主动断开 SSE 订阅者**（模拟 progress observer 消失，正是旧 `0x80131506` 的触发路径），job 仍 `state=done`、`error=None`，输出 **787,072 B** 的 512×512 RGB PNG（colour type 3 索引色解码正常、采样 251 个不同值，确认为真图而非噪点占位）；`/api/health` 复查 `served=1` / `busy=False`，引擎未损坏，输出 `ALL OK: WebUI survived the SSE disconnect and produced a real image`。耗时约 5 分钟（含 native 文本塔加载）。
- `QZI_PREPROCESS_DIR` 未写入 `NativeConfigScope.Variables` 与 `qzen_config.c:env_overrides[]`，`qzi_abi_features()` 仍为 `0xF`、`qzi_backend` 仍 11 个回调；本轮**没有**改动任何 C ABI 导出、结构体、layout、能力位或 `engine/` 源码，符合「核心库只含模型与推理」的分层约束。
- native 回归：`ctest --test-dir native/build -C Release` **1/1**（参考后端 0.13 s）、`ctest --test-dir native/build-cuda -C Release` **1/1**（真实后端 1.84 s）——本轮未改任何 native 源码，用于确认 ABI / vtable 未被连带破坏。`git diff --check` 干净。
- 打包边界：`dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj -c Release --no-build` 成功产出 `.nupkg` / `.snupkg`，包内 **16 项**、含 `lib/net10.0` 程序集与两个 RID native，且**不含**任何 `Onnx` / `Preprocessing` 条目；`Microsoft.ML.OnnxRuntime` 只出现在 `QwenZImage.NET.WebUI.csproj`（`IsPackable=false`）。核心 csproj 唯一的 `PackageReference` 是 `PrivateAssets=All` 的 SourceLink（构建期工具，非运行期依赖）。`artifacts/runtimes` 仍是参考后端（123,459 B DLL / 38,440 B SO），未被 CUDA 产物覆盖（沿用阶段 19 记下的发布前检查项）。
- 产物落在 `.gitignore` 忽略的 `.scratch/preprocess-verify/` 与 `.scratch/cuda-verify/`；验证完成后两个 WebUI 进程均已停止。

## 4. 交付物清单

```
QwenZImage.NET/
├─ README.md                     NuGet 包 README / 项目首页
├─ LICENSE                       MIT
├─ THIRD_PARTY_NOTICES.md        第三方源码（含上游引擎）与模型商标声明
├─ PLAN.md                       本文件
├─ QwenZImage.config.json        模型权重路径配置示例
├─ global.json / Directory.Build.props / QwenZImage.NET.sln
├─ native/                        C ABI、参考后端、真实后端适配器、CMake、CTest
│   ├─ include/qzi.h              稳定 C ABI v1
│   ├─ include/qzi_backend.h      vtable 契约
│   ├─ include/qzi_env.h          内部环境变量读取辅助（非 ABI 导出）
│   ├─ src/qzi_env.c              Win32 实时环境 / POSIX getenv 双实现
│   ├─ src/qzi_engine_backend.c   真实引擎 in-process 适配器
│   └─ tests/                     C ABI smoke（含作用域环境断言）+ POSIX ICU 双路径检查
├─ engine/                       vendored + 改名的真实 C/CUDA 推理引擎
│   ├─ src/, platform/win32/, platform/posix/, third_party/, cmake/, CMakeLists.txt
│   ├─ src/qzen_zimage_control.c/.h  ControlNet Union 2.x 权重加载 / GPU 上传
│   └─ LICENSE / UPSTREAM.md     上游 MIT 许可与来源说明
├─ src/QwenZImage.NET/            NuGet 包体（零三方依赖，只含模型与推理）
│   ├─ Configuration/QziConfig.cs         配置文件加载器（含 zimage_control_dir / preprocess_dir）
│   ├─ Configuration/NativeConfigScope.cs  QZI_* 作用域发布 / 还原（6 个目录变量，不含 QZI_PREPROCESS_DIR）
│   ├─ Native/CallbackScope.cs             进程级 token 表 + 反向回调异常封边（阶段 20）
│   ├─ Models/QziRasterImage.cs            无依赖 PNG 解码 / 平面 RGB8 栅格（编辑 / 控制图）
│   └─ Models/GenerationOptions.cs        Qwen21EditOptions / Qwen21EditExOptions / ZImageControlOptions / ZImageInpaintOptions
├─ src/QwenZImage.NET.WebUI/      跨平台手动测试 WebUI（不打包，含编辑 / 高级编辑 / ControlNet / Inpaint 模式）
│   └─ Preprocessing/             host 层 guide 预处理（ControlImagePreprocessor / PreprocessAssets /
│                                 PoseEstimator / DenseMapEstimator / RasterOps / PngWriter；仅此处引用 OnnxRuntime）
├─ tests/                         202 个 xUnit 测试（118 库 + 84 WebUI）
├─ samples/                       端到端示例
├─ integration/                   自定义 qzi_backend 模板
├─ tools/RealImageRegression/     真图回归 harness（需 GPU + 权重，不进 sln / CI）
├─ docs/                          ABI / 参数 / 引擎接入 / WebUI 文档
├─ scripts/                       参考与 CUDA 构建、RID staging、真图回归、一键真后端 WebUI 启动
├─ .github/workflows/             CI（双平台 native + WebUI 冒烟）+ NuGet Trusted Publishing
└─ artifacts/                     本地构建产物（git 忽略，仅 .gitkeep 空目录占位被跟踪）
    └─ runtimes/<rid>/native/     打包前 stage 的两个 RID native
```

包内布局：

```
lib/net10.0/QwenZImage.NET.dll
lib/net10.0/QwenZImage.NET.xml
runtimes/win-x64/native/QwenZImageNative.dll
runtimes/linux-x64/native/libQwenZImageNative.so
README.md / LICENSE / THIRD_PARTY_NOTICES.md / engine/LICENSE / docs/*.md
```

**NuGet 包默认捆绑参考后端的 Windows / Linux native 库**，不捆绑模型权重。真实 CUDA 构建需按显卡 SM 架构由使用方生成同名库并通过 `QZI_NATIVE_LIBRARY` 覆盖；刻意不把多 GB 权重与特定 GPU 架构塞进包里。

## 5. 阶段 13 发布清单（D11：Trusted Publishing）

**仓库**：`https://github.com/dongfangzhizhu/QwenZImage.NET`。`origin` 已从 `alexhegit/...` 切换为此地址；`csproj` 的 `Authors` / `PackageProjectUrl` / `RepositoryUrl`、WebUI 页脚链接、README 与本计划均已同步，SourceLink 因此能生成正确的源码映射。

**发布方式**：NuGet.org **Trusted Publishing**（OIDC 联邦凭据）。`release.yml` 的 `publish` job 以 `permissions.id-token: write` 向 GitHub 申请 OIDC token，`NuGet/login@v1` 用它换取**有效期 1 小时**的短期 API key，再由 `dotnet nuget push` 发布 `.nupkg` / `.snupkg`。一个 OIDC token 只能换一个 key，所以工作流**只登录一次**、紧贴 push 之前，并让两个包共用该 key。仓库**不保存任何长期 `NUGET_API_KEY`**。

一次性前置条件（需在本仓库之外完成，代码无法代替）：

1. **nuget.org** → 登录包所有者账号 → *Trusted Publishing* → 新建 policy（大小写不敏感）：
   - Repository Owner：`dongfangzhizhu`
   - Repository：`QwenZImage.NET`
   - Workflow File：`release.yml`（只填文件名，**不要**带 `.github/workflows/` 前缀）
   - Environment：留空（本工作流未使用 GitHub Environments）
   - 私有仓库的 policy 初始有 7 天临时激活期，首次成功发布后转永久。
2. **GitHub 仓库** → Settings → Secrets and variables → Actions → repository secret `NUGET_USER` = **nuget.org 的 profile 用户名（不是邮箱）**，必须与 policy owner 一致。
3. 本机已配置 git 推送凭据（或已登录 `gh`），且 GitHub 上目标仓库已创建、有推送权限。
4. 确认 NuGet.org 上 `QwenZImage.NET` 这个 package ID 可用，policy owner 有权发布它。

> **本地无需 API key**：Trusted Publishing 只能在 GitHub Actions 运行环境里凭 OIDC token 换 key。若确需从本机手动发布，只能另用个人 API key；**推荐一律走 tag 自动发布**，避免密钥散落。

```bash
# 13.1 切换远程并推送（origin 已存在，必须用 set-url，不能用 remote add）
cd D:/AI/convert/QwenZImage.NET
git remote set-url origin https://github.com/dongfangzhizhu/QwenZImage.NET.git
git remote -v
git branch -M main
git push -u origin main

# 13.2 确认 CI 全绿：native job 分别在 Windows / Ubuntu 构建并上传 RID 产物，
#      verify job 下载两者后 build/test/pack 并校验包内三个运行体，
#      Linux 还会跑 WebUI 运行时冒烟。CI 不需要 GPU（跑的是参考后端）。

# 13.3 打 tag 触发自动发布（release.yml 只在 v* tag / 手动 dispatch 上运行）
git tag -a v0.1.0 -m "QwenZImage.NET 0.1.0"
git push origin v0.1.0
#      publish job：pack → 校验包内容 → 校验 NUGET_USER → NuGet/login@v1(OIDC)
#      → push .nupkg / .snupkg → 创建 GitHub Release 并上传包
```

发布前检查：

- [ ] `src/QwenZImage.NET/QwenZImage.NET.csproj` 的 `Version`、`Authors`、`RepositoryUrl`、`PackageProjectUrl` 均指向 `dongfangzhizhu/QwenZImage.NET`
- [ ] README「当前状态」表与「路线与发布」段落已按 Trusted Publishing 更新，无 `alexhegit` 残留
- [ ] NuGet.org Trusted Publishing policy 的 owner / repo / workflow file / environment 与 `dongfangzhizhu/QwenZImage.NET` + `release.yml` 精确一致
- [ ] GitHub repository secret `NUGET_USER` 已配置为 nuget.org profile 名；仓库内无 `secrets.NUGET_API_KEY` 依赖
- [ ] `release.yml` 的 `publish` job 具备 `id-token: write`，且只调用一次 `NuGet/login`（一个 OIDC token 只换一个 key）
- [ ] `artifacts/` 下所有产物均被 `.gitignore` 忽略；待推送的 tree 只含源码、文档、配置、脚本与 CI 定义
- [ ] `dotnet pack` / `dotnet test` 前确认 `artifacts/runtimes/win-x64/native` 与 `linux-x64/native` 是**参考后端**（本机约 123 KB DLL / 38 KB SO，而非 2.2 MB / 22 MB 的 CUDA flavour）。`scripts/build-native-cuda.bat` 会把 `win-x64` staging 覆盖成 CUDA 产物，此时托管测试会加载真后端、`IsReferenceBackend` 断言批量失败；用 `bash scripts/stage-native.sh`（不带 `QZI_STAGE_CUDA=1`）恢复即可
- [ ] `docs/NATIVE_ABI.md` 与 `AbiGuard` 的版本号一致（v1；`QZI_LORA_NAME_MAX` 只是输入校验常量，不升版本）
- [ ] LoRA 契约在 `qzi.h` / `qzi.c` / `RequestValidator.cs` / `docs/PARAMETERS.md` / WebUI 五处一致：两个 tier 均可用、≤ 255 字节裸文件名、`lora_scale == 0.0f` sentinel、scale finite 且 `0.05..4`
- [ ] 四个增量能力在 `qzi.h`（feature 位 / request 布局）/ `qzi.c`（归一化）/ `qzi_backend.h`（末尾槽位顺序）/ `AbiGuard.cs` / `RequestValidator.cs` / WebUI `RequestMapper.cs` 六处一致；`qzi_abi_features()` 返回 `0xF`，`qzi_backend` 共 11 个回调，且 `integration/custom_backend_template.c` 末尾四个槽都是 `NULL`；`qzi_qwen21_edit_ex_layout()` / `qzi_zimage_inpaint_layout()` 分别与 `sizeof(QziQwen21EditExRequest)` / `sizeof(QziZImageInpaintRequest)` 一致；C smoke 的 `main()` 确实调用四个增量的校验与生成检查函数（不留下「已定义未调用」的死代码）
- [ ] 高级编辑的 32 像素对齐、残余 `32×32` 源矩形、`model_mode` 白名单（0 / 2 / 3 / 4 / 5）、LanPaint 需 mask 且与外扩互斥、两个 strength 强制 `1.0`、`rgba ∈ {0,1}` 在 `qzi.c` / `RequestValidator.cs` / `RequestMapper.cs` / `docs/PARAMETERS.md` / WebUI 五处一致；`Qwen21EditExOptions` 的 `Width`/`Height` 是最终画布（含边距）
- [ ] inpaint 的 mask 必填 / 单通道 / 每边 `≤ 4096` / 总像素 `≤ 2048²` / 两个 `flags == 0`、控制图与 mask 尺寸**可以不同**（各自独立 cover fit）、第四能力**前置依赖 ControlNet**，在 `qzi.c` / `RequestValidator.cs` / `RequestMapper.cs` / `docs/PARAMETERS.md` / WebUI 五处一致；`ZImageInpaintOptions` 是 `ZImageControlOptions` 的派生 record，所有模式匹配（`RequestValidator.Validate` / `QziClient.Generate` / `WebUiDtos.ControlState` / `WebUiCapabilities.ErrorFor` / `WebUiEngine.Generate`）都必须**先匹配派生类**
- [ ] `control_type` 白名单、`control_scale` sentinel（0 → 0.65，`0.05..2.0`）与 `zimage_control_dir` 配置链在 `qzi.c` / `QziConfig.cs` / `NativeConfigScope.Variables` / `qzen_config.c:env_overrides[]` / `docs/PARAMETERS.md` / WebUI 六处一致
- [ ] `QZI_*` 作用域契约在 `NativeConfigScope.Variables` / `qzen_config.c:env_overrides[]` / `qzi_stub.c:config_fingerprint()` 三处变量表一致（现为 6 个目录变量）；native 配置读取统一用 `qzi_getenv()`，不出现裸 `getenv("QZI_...")`；`QZI_PREPROCESS_DIR` / `preprocess_dir` **不在**这三张表里（阶段 20 只由 WebUI 读取）
- [ ] 核心 `QwenZImage.NET.csproj` 不引用任何三方图像处理 / ONNX 包；`Microsoft.ML.OnnxRuntime` 只出现在 `QwenZImage.NET.WebUI.csproj`，且 WebUI 项目仍为 `IsPackable=false`；`dotnet pack` 的核心包内不含 `Preprocessing/` 程序集或 ONNX 权重
- [ ] `CallbackScope.cs` 的 `User` 仍是 token（`Live` 表查找），没有回退成 `GCHandle`；`OnProgress` / `OnShouldCancel` 外层 `try/catch` 完整；`QziClient.Invoke` 所有六路分派都传 `scope` 并在返回后调用 `TakeFault()`；WebUI `.csproj` 保留 `<CETCompat>false</CETCompat>`
- [ ] `dotnet pack` 包内包含 `README.md` / `LICENSE` / `THIRD_PARTY_NOTICES.md` / `engine/LICENSE` / `docs/*.md`
- [ ] 包内同时包含 `lib/net10.0/QwenZImage.NET.dll`、`runtimes/win-x64/native/QwenZImageNative.dll`、`runtimes/linux-x64/native/libQwenZImageNative.so`
- [ ] `.snupkg` 符号包可上传、SourceLink 指向公开仓库
- [ ] README 中「默认捆绑 reference native、真实 CUDA 为 fat binary 且由使用方自行构建 / 替换」的说明保持准确
- [ ] `engine/LICENSE` 与 `THIRD_PARTY_NOTICES.md` 中上游 MIT 版权声明完整

## 6. 已知边界 / 后续工作

1. **参考后端不是图像模型**：默认包里的 native 输出确定性合成 PNG，只用于验证 ABI、内存所有权、进度与队列；真实生成请用 `-DQZI_WITH_CUDA_ENGINE=ON` 构建。
2. **真实引擎已实现**：`native/src/qzi_engine_backend.c` 完整实现 `qzi_backend`，编排参考上游 `serve_execute_qwen21()` / `serve_execute_zimage()`，PNG 直接在内存里编码而非临时文件。多 checkpoint（Z-Image bf16 / Nunchaku r32 / r256）、LoRA scale 变化、base tier 与 1024×1024 已在阶段 12d 用 `tools/RealImageRegression` 10-case 矩阵回归通过；该工具需要 GPU + 权重，**不进 CI**，CI 仍只跑参考后端。
3. **平台**：参考库 Windows / Linux 均已构建、`ctest` 通过；真实 CUDA 后端也已在 Windows（MSVC 19.51 + CUDA 13.3）与 Linux（WSL Debian 12 / GCC 12.2 + CUDA 13.0）双平台实测，默认 fat binary 各 11 个 SM，`ctest` Windows 1/1、Linux 3/3（含 POSIX ICU 双路径测试）。WSL 自带 CUDA 13.0，但 `nvcc` 不在默认 PATH，构建前需 `export PATH=/usr/local/cuda/bin:$PATH`。Linux 产物不再依赖版本化 ICU。**按用户明确要求「不需要适配 macOS」，macOS 不在开发范围内**（POSIX 兼容层虽可探测 `libicuuc.N.dylib`，但 CMake 仍链接 `stdc++`，若将来启用需换成 `c++`）。
4. **CI 不含 GPU**：CI 只构建参考后端；真实后端由使用方或带 GPU 的自托管 runner 构建。
5. **ABI 演进**：任何**既有**结构体 / 签名 / 所有权 / phase 语义变化都必须升 `QZI_ABI_VERSION`，并同步 `NativeStructs.cs` 与 `AbiGuard`。新增可选能力优先用独立导出 + 能力位表达（阶段 15 的图像编辑、阶段 16 的 ControlNet、阶段 17 的高级编辑、阶段 19 的 Z-Image 遮罩重绘共四个增量均如此，仍为 v1，feature mask 依次为 `0x1`/`0x3`/`0x7`/`0xF`）；新增一个仅伴随新能力出现的可选 phase 字面量（`"image encode"`）也属于 additive，因为托管侧把未知字面量降级为 `GenerationPhase.Unknown`，旧库根本不会上报它。阶段 19 的 inpaint 在真实后端多做一次 VAE 编码，但 progress bridge 仍把两次编码折算进同一个 `"image encode"` 阶段（`0/1 → 1/1`）；只有 GPU-free 参考后端为便于测试按 `0/2 → 1/2 → 2/2` 拆成两个独立阶段。两者都不改变字面量含义，也不影响旧库上报的 `(0,1)/(1,1)` 序列，故按 additive 处理。
6. **异步语义**：native 调用本质同步长任务，`Generate*Async` 是 `Task.Run` 外壳；高并发行走 `QziGenerationQueue`，不要用多个 `QziClient` 抢同一张 GPU。
7. **配置通道的平台差异**：`QZI_*` 仍靠进程级环境变量传递（ABI v1 没有这些字段），但已通过作用域发布 + create 时快照消除串扰。Windows 上 **CRT 的 `getenv()` 读不到 .NET 之后设置的值**，所有 native 配置读取必须走 `qzi_getenv()`；自定义 backend 若直接 `getenv("QZI_*")`，在 Windows 上会静默失效。Windows 路径字节沿用进程 ANSI 代码页（与 `fopen` 一致），含非 ASCII 的路径建议写进配置文件。
8. **进程级 GPU 资源仍未隔离**：`CUDA_VISIBLE_DEVICES`、CUDA 上下文与常驻权重是进程/设备级资源；作用域配置不能让两个 `QziClient` 安全共享同一张卡，仍须按 #6 串行化。彻底的 per-client GPU 隔离需要独立进程，当前不在范围内。
9. **Trusted Publishing 依赖外部配置**：OIDC 换 key 依赖 nuget.org 上的 policy 与 GitHub 仓库 secret `NUGET_USER`；policy 的 owner / repo / workflow file / environment 必须与 `dongfangzhizhu/QwenZImage.NET` 和 `release.yml` 精确一致，改仓库归属或工作流文件名都要同步更新 policy，否则发布会在 login 步骤失败。仓库内不保存长期密钥，因此本机手动 push 需要另备个人 API key。
10. **图像编辑的已知边界**：只支持 8 位非隔行灰阶 / RGB / RGBA PNG（含 base64 data URL）作为参考图 / mask，透明参考像素按白底合成、透明 mask 像素按 0 处理；参考图每边 ≤ 4096、总像素 ≤ `2048×2048`。mask 是**硬笔刷**语义（16×16 面积均值收缩到 1/16 latent 网格后 `> 64` 二值化），不做羽化 / 抗锯齿；`strength` 采用 WanGP 的**步数比例**规则而非 sigma 阈值。编辑只作用于 Qwen-Image 2.1，且**一张参考图 + 可选 mask**：阶段 17 已把 WanGP 的 outpainting、LanPaint `model_modes` 与 RGBA 输出纳入本库（见 #13）；WanGP 剩余差异已按本机源码复核（`D:/AI/Wan2GP`，commit `5b63ae9d`，`models/qwen21/qwen21_handler.py`）：`query_supported_types()` 只返回 `qwen_image_21_7B`，**20B edit / edit_plus / layered 变体在当前 WanGP 已不存在**；`turbo`（8 条固定 sigma，与本引擎 `q21_turbo_sigmas` 逐值一致）与 `viggle_image_swap`（内置 LoRA）只是同一 7B 家族的 `qwen21_variant`；编辑 LoRA 由本库 ABI 的 `lora` / `lora_scale` 覆盖（WanGP `get_lora_dir="qwen21"` 只是权重目录），`prompt_enhancer` 的 `EDITING` 是调用方用 LLM 重写 prompt 的**应用层**步骤而非推理引擎能力。真正未覆盖的是**最多 10 张有序参考图 + `<image1>…<imageN>` 模板**：它要求 Qwen3-VL-8B 的 **vision tower**（`vision_encoder` / `processor` / `pixel_values` / `image_grid_thw` / `image_pad_mask`）与额外 vision 权重，本引擎目前只有语言塔（`engine/src/qzen_qwen21_text.c`），故暂不纳入；WanGP 的 `qwen21_kv_cache`（INT8 / INT8_RAM）是条件 K/V 复用的性能选项，也不属功能缺口。Z-Image 侧的对应能力是 **ControlNet（见 #11）** 与阶段 19 的**遮罩重绘 / inpaint（见 #14）**；仍不提供 outpainting 入口（Red Canvas 外扩只属于 Qwen-Image 2.1）。参考后端只做确定性字节对照，**不产出可看的编辑 / 重绘效果**，真实效果验证仍需 CUDA 构建 + 权重。
11. **Z-Image ControlNet 的已知边界**：仅支持 Union **v2 / v2.1**（33 通道 `[control_latent(16) | keep_mask(1) | inpaint_latent(16)]`、15 control layers + 2 独立 noise refiner），v1（16 通道）在权重加载时明确拒绝；**普通 ControlNet 请求**里 `keep_mask` 与 `inpaint_latent` 恒为 0（等价于 WanGP `inpaint_mask is None`），带 mask 的重绘由第四增量 `qzi_generate_zimage_inpaint()` 提供（见 #14），但仍不提供 outpainting。WanGP 本机源码（`models/z_image/pipeline_z_image.py`，`control_in_dim == 33 && inpaint_mask is not None`）中该分支是完整可用的，早先「disabled for the moment」的描述已不适用于当前上游版本。`control_type` 只是描述性元数据，**核心库与 C ABI 不运行** pose / canny / depth / scribble 预处理器，对齐 WanGP `guide_preprocessing = "V"`；阶段 20 起 WebUI 在 host 层补上 guide 预处理（见 #15），但 ABI 契约不变；`control_scale` 默认 `0.65`（WanGP Union 2.1 的 `control_net_weight_alt`，v1/v2 习惯 `0.75`，可显式覆盖）。Union 2.1 权重约 3.7 GiB，叠加常驻 DiT / 文本塔后 24 GiB 卡可能 GPU 上传失败并回退 CPU hint 路径（正确但更慢）；双份主机 / 设备权重使 WebUI 进程私有提交量在本机（32 GiB RAM + RTX 4090）约 45 GiB，内存较小的机器仍需留意。未配置 `zimage_control_dir` 时真实后端返回 `QZI_ERR_NOT_CONFIGURED`，参考后端只做确定性字节对照。阶段 16 已把逐 block 的主机 staging 改为 `qzen_zimage_gpu` 内 grow-only 的共享缓冲，消除了冷启动首个 ControlNet 请求偶发的 `cannot stage Z-Image GPU input`（详见「阶段 16 本机验证结果」）。
12. **能力探测与降级**：`qzi_abi_features()` 当前内置两个后端都返回 `0xF`（`QZI_FEATURE_QWEN21_EDIT | QZI_FEATURE_ZIMAGE_CONTROL | QZI_FEATURE_QWEN21_EDIT_ADV | QZI_FEATURE_ZIMAGE_INPAINT`），`qzi_backend` 共 11 个回调、末尾四个可选槽顺序固定。旧 native 库缺导出时 `AbiGuard` 让 `SupportsImageEditing` / `SupportsControlNet` / `SupportsAdvancedEditing` / `SupportsInpaint` 为 `false`，WebUI 会禁用对应模式或在提交前返回 `400`；第三个能力**前置依赖普通编辑能力**（高级请求内嵌 `qzi_qwen21_edit_request`），第四个能力**前置依赖 ControlNet**（inpaint 请求内嵌 `qzi_zimage_control_request`），所以只有前两个增量的库上普通编辑 / ControlNet 可用、高级编辑 / 遮罩重绘不可用是预期状态。WebUI 在 `supports_inpaint == false` 时把选中的 `zimage-inpaint` 回退成 plain `zimage`，而不是静默丢掉 mask。自定义 backend 把 vtable 末尾四个槽留 `NULL` 时对应调用得到 `QZI_ERR_UNSUPPORTED`。要注意 `supports_controlnet == true` 只说明**符号与布局存在**，不等于权重已配置——真实后端仍可能因缺 `zimage_control_dir` 返回 `QZI_ERR_NOT_CONFIGURED`；`supports_inpaint == true` 同理，且 inpaint 还要求加载到 33 通道 Union v2/v2.1 权重（16 通道 v1 明确拒绝）。`supports_advanced_editing == true` 只说明第三个增量的导出与布局存在；WebUI 仍会在缺少 `reference_image` 或高级字段与模型不匹配时返回 `400`。
13. **高级编辑（阶段 17）的已知边界**：三项能力只用同一份 7B 权重，但**保真度不等同于 WanGP**——WanGP 的 Qwen-Image 编辑是 condition-prefix 模型（参考 latent 作为注意力前缀），本引擎是 img2img sampler（参考图即起点 latent），Red Canvas 外扩靠「边距与绘制区从全噪声起步、保留区 pin 到已知轨迹」近似，不是逐像素复刻；LanPaint 是 WanGP `shared/lanpaint.py` Algorithms 1–4 的 C 移植，迭代次数固定为 2/5/10/15，不暴露更多超参。RGBA 的 alpha 是 VAE `conv_out` 第 4 通道按 `(x+1)*127.5` 量化的结果，**不是分割 / 抠图 mask**，不要当作干净的透明通道使用。外扩与 LanPaint 互斥、LanPaint 必须有 mask 且强制两个 strength 为 `1.0`、四边距必须 32 对齐并留下 ≥ `32×32` 源矩形，这些约束在 `qzi.c` / `RequestValidator.cs` / `RequestMapper.cs` 三处一致，故意不放宽以免静默产出无意义结果。本轮已用真实 CUDA 后端 + 权重跑通四例真图（普通编辑基线 / 外扩 / RGBA / LanPaint，见「阶段 17 本机验证结果」），确认三条路径都能产出合法 PNG 且 RGBA 的 RGB 平面与普通编辑逐字节一致；但这只是**功能与数值透明性验证**，外扩 / LanPaint 的主观画质、以及它们与 WanGP condition-prefix 模型的语义差距仍需使用方在目标权重上目视确认。阶段 15 / 16 的真图矩阵不受本轮影响。
14. **Z-Image 遮罩重绘（阶段 19）的已知边界**：只支持 8 位非隔行灰阶 / RGB / RGBA PNG（含 base64 data URL）作为 mask / 控制图，mask 解码后折叠为单通道，透明 mask 像素按 0 处理；每边 ≤ 4096、总像素 ≤ `2048×2048`，`flags` 必须为 0。mask 是**硬笔刷**语义（255 = 重绘、0 = 保持），控制图与 mask **尺寸可以不同**，各自独立做 centre-crop cover fit 到输出画布，不做等比留边。`downsample_mask_keep()` 用 8×8 **面积均值**把 mask 收缩到 1/8 latent 网格得到软 `keep_mask ∈ [0,1]`，而 WanGP 用 `F.interpolate(..., mode='nearest')`；因此硬笔刷边界最多相差 8 像素，软边 / 抗锯齿 mask 在本实现里更平滑，不属于逐像素复刻。仅 33 通道 Union v2 / v2.1 可用（16 通道 v1 在引擎层明确拒绝）；`control_type` 仍只是描述性元数据，不运行 pose / canny / depth / scribble 预处理器。**与 WanGP 的剩余差异**：WanGP `pipeline_z_image.py` 的 inpaint 分支在 `input_ref_images` 非空时会把 `control_latent` 换成另一张参考图（`convert_image_to_tensor(input_ref_images[0])`），但 `z_image_handler.py` 中对应的 `image_ref_choices`（"Reference Image for Inpainting"）整段已被注释，当前上游 UI 实际未暴露该工作流；本库同样始终用同一张 `control_image` 编码 `control_latent` 与 `inpaint_latent`，即不支持「控制图与重绘底图分离」。WanGP `z_image_handler.py` 对 `z_image_control2` / `z_image_control2_1` 明确设了 `inpaint_support = True`，`infos.py` 也写明 "Control v2: source + mask for inpainting"，确认本轮补齐的正是上游已开放的能力。本库的 `control_scale` 与 WanGP `control_context_scale` 同义（skip 连接的 `alpha`，Union 2.1 默认 0.65）；Z-Image 仍没有 outpainting 入口。inpaint 需要**两次 VAE 编码**（控制图 + `control*(1-mask)`），比普通 ControlNet 多一次编码；真实后端的 progress bridge 仍把它们折算进同一个 `"image encode"` 阶段（`0/1 → 1/1`），只有参考后端按 `0/2 → 1/2 → 2/2` 拆成两个独立阶段供无 GPU 测试断言。普通 t2i / ControlNet 的 33 通道 context 后 17 个平面仍恒为 0，逐位不变。参考后端只把 mask 内容 / 尺寸纳入确定性哈希并模拟两个编码阶段，**不产出可看的重绘效果**；真实效果需 CUDA 构建 + Union 2.1 权重验证（见「阶段 19 本机验证结果」）。
15. **回调边界与预处理的分层（阶段 20）**：`QziCallbacks.User` 从 `GCHandle.ToIntPtr()` 改为**进程级 token**（单调递增 `long`），这是内部实现细节，**不进 C ABI**：native 仍把它当不透传的 `void*` 回传，旧 native 库无需重建。`[UnmanagedCallersOnly]` 的两个回调现在是「不抛异常」契约——任何 `IProgress<T>.Report` / SSE subscriber / `JobChanged` 观察者的异常都被记录而非跨越 native 栈展开，代价是故障以 P/Invoke 返回后的 `InvalidOperationException` 呈现（inner exception 才是真实原因），且该次生成会被 `ShouldCancel` 提前停止、reference backend 不计入 `health.served`；这是刻意的容错取舍，不是丢错。`<CETCompat>false</CETCompat>` 只设在 **WebUI host 项目**，核心库 csproj 不动；它规避本机调试器 / func-eval 场景下 CET + reverse P/Invoke 的 fatal 路径，不影响 NuGet 包的使用方进程。**预处理严格属于 host 层**：`QwenZImage.NET` 与 C ABI 永久保持 raw raster 语义，不加载 ONNX、不引入 `Microsoft.ML.OnnxRuntime` / System.Drawing / ImageSharp；`preprocess_dir` / `QZI_PREPROCESS_DIR` 只是 WebUI 的资产发现键，`QziConfig` 认识它但 `NativeConfigScope` 不发布给 native。直接调用核心库 / C ABI 的使用方**仍需自备已成形的 guide**（见 #11），WebUI 用户则上传普通照片即可。learned estimator 是可选增强：无 pose ONNX 时 `pose` 明确报错并提示配置资产目录；无 depth / contour ONNX 时 `depth` 回退相对深度代理、canny/scribble 回退托管算子并在响应里带 `warning`，`provider` 字段始终标明实际来源，便于判断结果是否由 learned 模型产出。WanGP 的 DWPose / Depth Anything V2 权重本体不随仓库分发，`auto` 只按 `control_type` 推导算子，**不会**推断照片内容；inpaint 模式默认 `raw`，避免对已是控制栅格的图二次边缘检测。
