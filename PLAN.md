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
                                Qwen21 DiT / text tower / VAE
                                ZImage DiT / text tower / VAE / CUDA kernels
                          ▲
              QwenZImage.config.json（模型权重路径，QZI_CONFIG / --config）
```

**D10 边界**：`QziClient` 直接 P/Invoke 同一个进程内的 `QwenZImageNative`；WebUI 与用户应用处于同一层，都只走 `QziClient`。**没有任何外部服务、HTTP transport、sidecar 或 Go 进程**。`QZI_CONFIG` 让 native backend 读取 ABI 里没有的 `qwen21_lora_dir` / `zimage_nunchaku_dir` 等字段；`QziClient` 通过 `NativeConfigScope` 在 `qzi_engine_create()` 期间**作用域发布** `QZI_CONFIG` 与五个 `QZI_*` 目录变量（含 `QZI_QWEN21_LORA_DIR` / `QZI_ZIMAGE_NUNCHAKU_DIR`），构造结束即还原，构造期间由 `EngineCreateGate` 串行化。native 侧 `qzen_config_apply_environment()` 在文件加载后应用，显式传值优先；所有 native 配置读取统一走 `qzi_getenv()`（Windows 用 Win32 API 读实时环境，绕过 CRT 的启动快照）。每个 engine 因此快照自己的配置，多实例互不串扰，且 ABI v1 的导出符号与结构体布局不变。

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
| 5 | 测试：46 个 xUnit（42 库 + 4 WebUI，阶段 5 当时；阶段 12f 后 69 项，阶段 14 后 73 项）+ `ctest` native smoke（阶段 5 时各 1/1；阶段 12e 后 Windows 1 项 / Linux 3 项） | ✅ 全绿 |
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
├─ engine/                       vendored + 改名的真实 C/CUDA 推理引擎（64 文件）
│   ├─ src/, platform/win32/, platform/posix/, third_party/, cmake/, CMakeLists.txt
│   └─ LICENSE / UPSTREAM.md     上游 MIT 许可与来源说明
├─ src/QwenZImage.NET/            NuGet 包体
│   ├─ Configuration/QziConfig.cs         配置文件加载器
│   └─ Configuration/NativeConfigScope.cs  QZI_* 作用域发布 / 还原
├─ src/QwenZImage.NET.WebUI/      跨平台手动测试 WebUI（不打包）
├─ tests/                         73 个 xUnit 测试（62 库 + 11 WebUI）
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
- [ ] `docs/NATIVE_ABI.md` 与 `AbiGuard` 的版本号一致（v1；`QZI_LORA_NAME_MAX` 只是输入校验常量，不升版本）
- [ ] LoRA 契约在 `qzi.h` / `qzi.c` / `RequestValidator.cs` / `docs/PARAMETERS.md` / WebUI 五处一致：两个 tier 均可用、≤ 255 字节裸文件名、`lora_scale == 0.0f` sentinel、scale finite 且 `0.05..4`
- [ ] `QZI_*` 作用域契约在 `NativeConfigScope.Variables` / `qzen_config.c:env_overrides[]` / `qzi_stub.c:config_fingerprint()` 三处变量表一致；native 配置读取统一用 `qzi_getenv()`，不出现裸 `getenv("QZI_...")`
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
5. **ABI 演进**：任何结构体/签名/所有权变化都必须升 `QZI_ABI_VERSION`，并同步 `NativeStructs.cs` 与 `AbiGuard`。
6. **异步语义**：native 调用本质同步长任务，`Generate*Async` 是 `Task.Run` 外壳；高并发行走 `QziGenerationQueue`，不要用多个 `QziClient` 抢同一张 GPU。
7. **配置通道的平台差异**：`QZI_*` 仍靠进程级环境变量传递（ABI v1 没有这些字段），但已通过作用域发布 + create 时快照消除串扰。Windows 上 **CRT 的 `getenv()` 读不到 .NET 之后设置的值**，所有 native 配置读取必须走 `qzi_getenv()`；自定义 backend 若直接 `getenv("QZI_*")`，在 Windows 上会静默失效。Windows 路径字节沿用进程 ANSI 代码页（与 `fopen` 一致），含非 ASCII 的路径建议写进配置文件。
8. **进程级 GPU 资源仍未隔离**：`CUDA_VISIBLE_DEVICES`、CUDA 上下文与常驻权重是进程/设备级资源；作用域配置不能让两个 `QziClient` 安全共享同一张卡，仍须按 #6 串行化。彻底的 per-client GPU 隔离需要独立进程，当前不在范围内。
9. **Trusted Publishing 依赖外部配置**：OIDC 换 key 依赖 nuget.org 上的 policy 与 GitHub 仓库 secret `NUGET_USER`；policy 的 owner / repo / workflow file / environment 必须与 `dongfangzhizhu/QwenZImage.NET` 和 `release.yml` 精确一致，改仓库归属或工作流文件名都要同步更新 policy，否则发布会在 login 步骤失败。仓库内不保存长期密钥，因此本机手动 push 需要另备个人 API key。
