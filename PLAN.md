# QwenZImage.NET 项目计划与决策记录

> 目标：把现有 `h3-spark.c` 中的 **Qwen-Image 2.1** 与 **Z-Image** 两个文生图能力，封装成一个独立、可开源、可由 `.NET` 调用的 NuGet 包 `QwenZImage.NET`。

## 1. 已确认的需求 / 决策记录

| # | 决策 | 来源 |
| --- | --- | --- |
| D1 | **新建独立目录** `D:/AI/convert/QwenZImage.NET`，不改动原 `h3-spark.c` 工程 | 用户 |
| D2 | **底层 C，.NET 10 封装，追求高性能**；**不要 Go 部分**，原 Go WebUI 的队列 / 进度 / 取消 / 重试 / 预设功能用纯 `.NET` 实现 | 用户明确指示 |
| D3 | **只做 Native P/Invoke**：不做 HTTP 客户端、不做双 transport、不引入 Go WebUI | 用户对封装方式的二选一 |
| D4 | 项目名 / NuGet 包 ID = **`QwenZImage.NET`**，仓库 = `alexhegit/QwenZImage.NET` | 用户 |
| D5 | 启用 `git` 版本控制、写入本计划文件、准备 CI 与 NuGet 发布工作流 | 用户 |
| D6 | **先本地完成**：本次不 push GitHub、不 publish NuGet | 用户在确认发布节奏时选择「先本地完成（推荐）」 |
| D7 | 增加一个**跟原版 Go WebUI 类似的 WebUI 项目用于手动测试**，要求跨平台支持 Windows 与 Linux；仍遵守 D2「不要 golang」，用纯 `.NET` / ASP.NET Core 实现 | 用户 |
| D8 | **发布的 NuGet 包要同时包含 .NET 包与 Windows、Linux 两个平台的 native 包** | 用户 |

技术现状调研结论（决定 D2/D3 的关键事实）：

- `h3 serve` 本身是 **C 写的** HTTP/JSON 服务，暴露 `/generate`、`/health`、`/progress`、`/unload`；`qwen21` 与 `zimage` 是其中两条文生图 pipeline，成功响应为 `image/png`。
- `webui/` 下的 Go 代码只是该 C 引擎的 WebUI / HTTP 客户端，**不是底层模型实现**；因此「不用 Go」不会损失推理能力。
- 原引擎一次只驻留一个模型、单槽队列，第二个并发 `/generate` 直接 503；新的 C ABI 用 `QZI_ERR_BUSY` 表达同样的单飞语义，由 `.NET` 队列串行化。

## 2. 架构

```
Your app (.NET 10)                     Browser (manual testing only)
  └─ QziClient / QziGenerationQueue   └─ QwenZImage.NET.WebUI (ASP.NET Core, :7860)
       / QziPresets                        │  Minimal API + SSE + wwwroot 静态前端
            │                              │  health / presets / jobs / cancel / retry
            │                              │  outputs / trash / restore / unload
            └──────────────┬───────────────┘
                           ▼
              QwenZImage.NET.dll   (LibraryImport 源生成 P/Invoke)
                           ▼
              qzi.dll / libqzi.so   ← native/include/qzi.h (稳定 C ABI v1)
                 ├─ qzi.c 控制层：校验 / 单飞 / 计数 / 进度 / 取消 / 错误 / 缓冲区所有权
                 └─ qzi_backend vtable
                      ├─ reference backend（本仓库自带，确定性、无 GPU）
                      └─ 真实 C/CUDA 引擎（外部实现，见 docs/ENGINE_INTEGRATION.md）
```

**边界选择理由**：把稳定 ABI 放在 C 层，可以让真实引擎只实现 vtable 而不必理解 `.NET`；把队列/预设放 `.NET` 层，可以彻底移除 Go 与「WebUI → 引擎 HTTP 服务」的跳转。WebUI 与调用方应用处于同一层，都直接调 `QziClient`，中间没有任何独立进程。托管侧用 `LibraryImport` + blittable POD + pinned UTF-8 + `ArrayPool` + `UnmanagedCallersOnly`，热路径没有 JSON、没有反射式 marshalling、没有 delegate 分配。

**打包形态（D8）**：单个 `.nupkg` 按 NuGet RID 约定同时携带 `lib/net10.0/QwenZImage.NET.dll`、`runtimes/win-x64/native/qzi.dll`、`runtimes/linux-x64/native/libqzi.so`。构建脚本把 CMake 的 per-platform 产物 stage 到 `artifacts/runtimes/<rid>/native/`；`QziEnsureNativeRuntimes` MSBuild target 在 `GenerateNuspec` 前强制校验两者存在，缺任一平台即拒绝打包。CI / release 分别在 Windows 与 Ubuntu runner 构建各自 native，再合并到 pack job。

## 3. 阶段与状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 调研原工程：`docs/PROTOCOL.md`、`docs/SERVE.md`、`h3_serve.c`、`h3_qwen21_pipeline.h`、`h3_zimage_pipeline.h`、Go client | ✅ |
| 1 | 解决方案骨架：`global.json`（.NET 10）、`Directory.Build.props`、`.gitignore` / `.gitattributes` / `.editorconfig`、3 个项目 + `.sln` | ✅ |
| 2 | C ABI v1：`qzi.h` / `qzi_backend.h`、`qzi.c` 控制层、参考后端 `qzi_stub.c`、无依赖 PNG 编码器 `qzi_png.c`、CMake、C smoke test | ✅ |
| 3 | `.NET 10` 封装：Native 层 7 个文件 + 公开模型 / 校验 / 结果 / 客户端；同步 + 异步 + 零拷贝 `WriteTo` | ✅ |
| 4 | 纯 `.NET` 队列与预设：`Channels` 串行分发、取消、重试、事件流、9 个预设 | ✅ |
| 5 | 测试：31 个 xUnit（校验 / interop / 队列）+ `ctest` native smoke | ✅ 全绿 |
| 6 | 开源文档：README / LICENSE / THIRD_PARTY_NOTICES / NATIVE_ABI / PARAMETERS / ENGINE_INTEGRATION / 本计划 / 后端模板 | ✅ |
| 7 | CI 与发布工作流：`ci.yml`、`release.yml`、`scripts/build-native.*` | ✅ |
| 8 | `git init` + 首次提交 | ✅（本地，不 push） |
| 9 | **跨平台 WebUI 手动测试控制台**（D7）：`QwenZImage.NET.WebUI`、Minimal API、SSE、`wwwroot` 静态前端、`docs/WEBUI.md`、加入 `.sln`、CI 运行时冒烟 | ✅ |
| 10 | **单包双平台 native**（D8）：WSL 构建 `libqzi.so`、RID staging 脚本、`csproj` 打包 + 缺失校验、`NativeLibraryLoader` 支持 `NATIVE_DLL_SEARCH_DIRECTORIES`、CI/release 双平台校验 | ✅ |
| 11 | **推送 GitHub 开源 + 发布 NuGet** | ⏸ **按 D6 暂缓**，命令见第 5 节 |
| 12 | 接入真实 CUDA 引擎（实现 `qzi_backend`） | ⏳ 依赖外部引擎；模板与文档已就绪 |

当前本机验证结果：`dotnet build` 0 警告 0 错误（4 个项目）、`dotnet test` 31/31、Windows + WSL Linux `ctest` 各 1/1、`dotnet pack` 产出 `QwenZImage.NET.0.1.0.nupkg` + `.snupkg`（含两个 RID native）、WebUI 全端点 + SSE + 输出画廊端到端实测通过。

## 4. 交付物清单

```
QwenZImage.NET/
├─ README.md                     NuGet 包 README / 项目首页
├─ LICENSE                       MIT
├─ THIRD_PARTY_NOTICES.md        第三方与模型商标声明
├─ PLAN.md                       本文件
├─ global.json / Directory.Build.props / QwenZImage.NET.sln
├─ .gitignore / .gitattributes / .editorconfig
├─ src/QwenZImage.NET/            NuGet 包体（net10.0 托管程序集 + 包元数据）
├─ src/QwenZImage.NET.WebUI/      跨平台手动测试 WebUI（不打包）
│   └─ wwwroot/                   index.html / app.js / style.css
├─ native/                        C ABI、参考后端、CMake、C smoke test
├─ tests/QwenZImage.NET.Tests/    31 个 xUnit 测试
├─ samples/QwenZImage.NET.Sample/ 端到端示例
├─ integration/                   真实引擎后端模板
├─ docs/                          ABI / 参数 / 引擎接入 / WebUI 文档
├─ scripts/                       Windows / Linux / WSL native 构建 + RID staging
├─ .github/workflows/             CI（双平台 native + WebUI 冒烟）+ NuGet 发布
└─ artifacts/                     本地构建产物（git 忽略，仅保留 .gitkeep）
    └─ runtimes/<rid>/native/     打包前 stage 的 qzi.dll / libqzi.so
```

**NuGet 包捆绑的是参考后端的 Windows / Linux native 库**（D8，各约 100 KB / 34 KB），**不捆绑模型权重、CUDA 引擎或 Go 代码**。接入真实引擎时用 `QZI_NATIVE_LIBRARY` / `QziClientOptions.NativeLibraryPath` 指向同一 ABI 的自有 `qzi` 实现即可覆盖，托管程序集不需要改动。刻意不把多 GB 权重与特定 GPU 架构塞进包里。

包内布局：

```
lib/net10.0/QwenZImage.NET.dll
lib/net10.0/QwenZImage.NET.xml
runtimes/win-x64/native/qzi.dll
runtimes/linux-x64/native/libqzi.so
README.md / LICENSE / THIRD_PARTY_NOTICES.md / docs/*.md
```

## 5. 阶段 11 发布清单（暂缓执行，供后续一次性完成）

前置条件（当前本机**均不满足**）：已安装并登录 `gh` **或** 配好 git 凭据；`NUGET_API_KEY` 可用；GitHub 上 `alexhegit/QwenZImage.NET` 仓库已创建且有推送权限。

> 本地仓库**已配置** `origin = https://github.com/alexhegit/QwenZImage.NET.git`（仅配置远程，未推送），这样 SourceLink 能生成正确的源码映射。若目标仓库不同，用 `git remote set-url origin <url>` 修改。

```bash
# 11.1 建仓并推送（示例，需自行确认仓库归属）
cd D:/AI/convert/QwenZImage.NET
git remote -v                 # 确认 origin；必要时 git remote set-url origin <url>
git branch -M main
git push -u origin main

# 11.2 确认 CI 全绿：native job 分别在 Windows / Ubuntu 构建并上传 RID 产物，
#      verify job 下载两者后在两个 OS 上 build/test/pack 并校验包内三个运行体，
#      Linux 还会跑 WebUI 运行时冒烟。release.yml 在 v* tag 上触发，
#      需要仓库 secret: NUGET_API_KEY。

# 11.3 打 tag 触发自动发布
git tag -a v0.1.0 -m "QwenZImage.NET 0.1.0"
git push origin v0.1.0

# 11.4 或者手动发布（先确保两个 RID native 已 stage）
powershell -File scripts/build-native-all.ps1    # Windows + WSL Linux
dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj -c Release -o artifacts/packages
dotnet nuget push artifacts/packages/QwenZImage.NET.0.1.0.nupkg \
  --api-key "$NUGET_API_KEY" --source https://api.nuget.org/v3/index.json --skip-duplicate
```

发布前检查：

- [ ] `src/QwenZImage.NET/QwenZImage.NET.csproj` 的 `Version`、`Authors`、`RepositoryUrl` 正确
- [ ] README 中「当前状态」表格更新为已发布、已推送
- [ ] `docs/NATIVE_ABI.md` 与 `AbiGuard` 的版本号一致（v1）
- [ ] `dotnet pack` 包内包含 `README.md` / `LICENSE` / `THIRD_PARTY_NOTICES.md` / `docs/*.md`
- [ ] 包内同时包含 `lib/net10.0/QwenZImage.NET.dll`、`runtimes/win-x64/native/qzi.dll`、`runtimes/linux-x64/native/libqzi.so`（CI 与 release 已自动校验）
- [ ] `.snupkg` 符号包可上传、SourceLink 指向公开仓库
- [ ] 仓库 secret `NUGET_API_KEY` 已配置；CI 不需要 GPU（跑的是参考后端）
- [ ] README 中「NuGet 包捆绑 reference native 库、不含权重/CUDA 引擎」的说明保持准确

## 6. 已知边界 / 后续工作

1. **参考后端不是图像模型**：输出是确定性合成 PNG，只用于验证 ABI、内存所有权、进度与队列；真实生成必须接入外部引擎。
2. **真实引擎未实现**：`h3_backend_template.c` 留了 5 个 TODO；接入时参考 `docs/ENGINE_INTEGRATION.md` 与 `h3_serve.c` 的 `serve_execute_qwen21` / `serve_execute_zimage`。
3. **平台**：Windows（MinGW/MSVC）与 Linux（WSL Debian gcc 12）的 native 库均已在本机构建、`ctest` 通过并 stage 进包；`ci.yml` 另在 GitHub 的 ubuntu/windows runner 上复验。macOS 尚未构建（可自行编译 `libqzi.dylib` 后用 `QZI_NATIVE_LIBRARY` 指定）。
4. **ABI 演进**：任何结构体/签名/所有权变化都必须升 `QZI_ABI_VERSION`，并同步 `NativeStructs.cs` 与 `AbiGuard`。
5. **异步语义**：native 调用本质同步长任务，`Generate*Async` 是 `Task.Run` 外壳；高并发行走 `QziGenerationQueue`，不要用多个 `QziClient` 抢同一张 GPU。
