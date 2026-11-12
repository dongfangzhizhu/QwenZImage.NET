# QwenZImage.NET

> **High-performance .NET 10 wrapper over a stable C ABI for Qwen-Image 2.1 and Z-Image text-to-image generation, plus Qwen-Image 2.1 image editing (img2img / masked inpainting).**
> Pure C/CUDA at the bottom, source-generated P/Invoke at the top: no Go, no sidecar server, no HTTP or JSON hop in the generation path.

一个把 **Qwen-Image 2.1（turbo / base / LoRA / 图像编辑）** 与 **Z-Image（bf16 / Nunchaku int4）** 封装成 `.NET 10` 可调用包的独立项目。Qwen-Image 2.1 支持参考图编辑、局部 mask 重绘、`strength`（WanGP `denoising_strength`）与 `masking_strength`，用同一份 7B 权重。托管库通过 `LibraryImport` **直接 P/Invoke 到进程内的 `QwenZImageNative` 共享库**，后者内含从 MIT 许可的上游 C/CUDA 引擎复制并改名后的真实推理代码（`engine/`）。`QwenZImage.config.json` 指定模型权重目录。原 Go WebUI 的队列、进度、取消、重试、预设全部用纯 `.NET` 重写，仓库中不含任何 Go 代码。

> ⚠️ **有两种 native 构建，务必先看这一段。**
>
> - **默认 / CI 构建**（`-DQZI_WITH_CUDA_ENGINE=OFF`，即 `scripts/build-native.ps1` / `.sh`）链接 `qzi_stub.c`：这是一个确定性 **reference backend**，输出由 prompt / seed / 尺寸哈希出来的**噪点图**，只用来在无 GPU 环境验证 ABI、进度、单飞与内存契约。它**不是图像模型**。
> - **真实构建**（`-DQZI_WITH_CUDA_ENGINE=ON`，即 `scripts/build-native-cuda.bat` / `.sh`）把 `engine/` 里的真实 C/CUDA 引擎与 `native/src/qzi_engine_backend.c` 适配器链进同一个 `QwenZImageNative` 库，直接做 DiT 去噪 + VAE 解码，**产出真实图像**。
>
> 「和原版用同样的 steps 却全是噪点」的根因是链接了 reference backend，不是参数问题。用 `GET /api/health` 的 `engine.reference_backend` 或 `EngineHealth.IsReferenceBackend` 判断当前加载的是哪一种。

仓库另附一个**跨平台 WebUI 手动测试控制台**（`src/QwenZImage.NET.WebUI`，ASP.NET Core Minimal API，Windows / Linux 同一套代码），复刻原 Go WebUI 的交互面。它同样直接 P/Invoke 进程内 native 库，**没有 HTTP transport、没有外部服务要启动**。详见 [docs/WEBUI.md](docs/WEBUI.md)。

```
┌──────────────────────────────────────────────────────────────┐
│  Your app (.NET 10)                                          │
│    QziClient / QziGenerationQueue / QziPresets               │
│    ArrayPool · pinned UTF-8 · UnmanagedCallersOnly callbacks │
├──────────────────────────────────────────────────────────────┤
│  QwenZImage.NET.dll  (LibraryImport, blittable structs)      │
├──────────────────────────────────────────────────────────────┤
│  QwenZImageNative.dll / libQwenZImageNative.so                │
│    native/include/qzi.h  ← 稳定 C ABI v1（唯一托管/原生契约） │
│    qzi.c 控制层：校验 / 单飞 / 进度 / 取消 / buffer 所有权    │
│    qzi_backend vtable                                        │
│      ├─ qzi_stub.c          参考后端（噪点，CI 用）          │
│      └─ qzi_engine_backend.c → engine/ 真实 C/CUDA 推理引擎   │
│            DiT / text tower / VAE / edit / ControlNet / CUDA   │
└──────────────────────────────────────────────────────────────┘
                          ▲
              QwenZImage.config.json（模型权重路径）
```

## 当前状态

| 项 | 状态 |
| --- | --- |
| 稳定 C ABI v1、`.NET 10` 源生成 P/Invoke、队列、预设、托管 + native 测试 | ✅ 已完成并通过 |
| 跨平台 WebUI（health / presets / jobs / SSE / gallery / trash / unload） | ✅ 已完成，Windows + Linux |
| **真实 C/CUDA 引擎直接绑定**（`engine/` + `qzi_engine_backend.c`，无 HTTP） | ✅ Windows（MSVC 19.51 + CUDA 13.3）与 Linux（WSL GCC 12 + CUDA 13.0）双平台构建 + `ctest` 通过；默认 fat binary 覆盖 11 个 SM；Linux 产物无版本化 ICU 依赖 |
| `QwenZImage.config.json` 配置模型路径，`.NET` / native 双侧加载 | ✅ 已完成；LoRA / Nunchaku 目录支持配置文件、`QziConfig.With`、代码级 options、纯环境变量四种等价入口 |
| **LoRA 全链路一致性**（ABI / 托管校验 / native / 真实引擎 / 参考后端 / WebUI / 文档） | ✅ 两个 tier 均可用，裸文件名与 scale 统一校验，切换 adapter 或 scale 正确重载 DiT |
| **Qwen-Image 2.1 图像编辑**（参考图 / 局部 mask / strength / masking_strength，对齐 WanGP） | ✅ ABI v1 增量扩展（`qzi_generate_qwen21_edit` + 能力位），native 校验与归一化、真实 CUDA 后端 VAE 编码 + 遮罩恢复、参考后端确定性对照、.NET `Qwen21EditOptions` / `QziRasterImage`、WebUI 上传与绘制 mask |
| **Z-Image Turbo Fun ControlNet Union 2.1**（控制图 / control_type / control_scale，对齐 WanGP `z_image_control2_1`） | ✅ 第二个 ABI v1 增量扩展（`qzi_generate_zimage_control()` + `QZI_FEATURE_ZIMAGE_CONTROL`）；`engine/src/qzen_zimage_control.*` 加载 33 通道权重，真实 CUDA 后端懒加载 → 控制图 VAE 编码 → 每步重算 15 条 control layer + 2 条 refiner skip 并注入主 DiT；普通 t2i 路径完全不变；.NET `ZImageControlOptions`、`SupportsControlNet`、WebUI `zimage-control` 模式 |
| **Qwen-Image 2.1 高级编辑**（Red Canvas 外扩 / LanPaint / RGBA 输出，继续对齐 WanGP `qwen21_handler`） | ✅ 第三个 ABI v1 增量扩展（`qzi_qwen21_edit_ex_request` + `QZI_FEATURE_QWEN21_EDIT_ADV = 0x4` + `qzi_generate_qwen21_edit_ex()`）；三项均复用同一份 7B 权重、无需新 checkpoint；.NET `Qwen21EditExOptions` / `Qwen21EditMode` / `SupportsAdvancedEditing`、WebUI 高级编辑面板与 `supports_advanced_editing`、参考后端确定性 RGBA PNG 对照；真实 CUDA 后端四例真图验证（普通编辑 / 外扩 / RGBA / LanPaint）全部通过，RGBA 的 RGB 平面与普通编辑逐字节一致 |
| **Z-Image 遮罩重绘 / Inpaint**（Union 2.x 33 通道后 17 平面，对齐 WanGP `pipeline_z_image.py` 带 mask 分支） | ✅ 第四个 ABI v1 增量扩展（`qzi_zimage_inpaint_request` + `QZI_FEATURE_ZIMAGE_INPAINT = 0x8` + `qzi_generate_zimage_inpaint()`）；真实后端两次 VAE 编码（控制图 + `control*(1-mask)`）、`keep_mask = 1-mask` 收缩到 1/8 latent、复用 `qzen_zpipe_control_context_ex()` 拼 33 通道 context；普通 t2i / ControlNet 逐位不变；.NET `ZImageInpaintOptions` / `SupportsInpaint`、WebUI `zimage-inpaint` 模式与 `supports_inpaint`、能力前置依赖 ControlNet |
| **反向回调崩溃修复**（`QziClient.cs` 的 `System.ExecutionEngineException` / `0x80131506`） | ✅ `CallbackScope` 改用进程级 token 表替代裸 `GCHandle`；`[UnmanagedCallersOnly]` 回调捕获全部托管异常并记录首个 fault，`ShouldCancel` 据此停止 native loop，P/Invoke 返回后在托管侧抛出带 inner exception 的 `InvalidOperationException`；`QziGenerationQueue` 的 observer / channel 异常同样封在边界内；WebUI `<CETCompat>false</CETCompat>`；7 项专门回归测试 + 真实 CUDA 下 SSE 断连仍出图 |
| **WebUI 控制图预处理 / 对齐 WanGP `guide_preprocessing`**（核心库保持 raw raster 契约） | ✅ `QwenZImage.NET.WebUI` 新增 `GET\|POST /api/preprocess`：`auto` / `raw` / `gray` / `canny` / `scribble` / `depth` / `pose`，兼容 WanGP 字母 `V/C/E/EV/S/D/DV/P/PV`；`Microsoft.ML.OnnxRuntime` 仅 host 层引用，加载 WanGP YOLOX-L + DWPose / Depth Anything V2 / contour 权重，缺资产时回退纯托管算子；前端上传普通照片即自动生成 guide，删除「Used as-is (no preprocessing)」提示；核心 NuGet 零三方依赖、C ABI 不变 |
| **WebUI 模型权重下载**（对齐 WanGP 地址，全模型覆盖 + 断点续传 + 手动下载地址 + 配置写回） | ✅ `src/QwenZImage.NET.WebUI/ModelDownloads/`：静态 WanGP manifest（6 组 / 32 个文件 / 4 个 HF repo）、`GET /api/models` 快照、`POST /api/models/download`（`.qzi-part` + `.qzi-etag` + HTTP `Range` / `If-Range`）、`POST /api/models/control`（pause / resume / reset）、`GET /api/models/manual`（精确 `resolve/main` 地址 + `curl -L -C -` 命令 + 配置片段）、`GET|POST /api/models/config`（默认只读预览，`{"apply":true}` 才备份并原子写回）；前端全宽 Models 面板；核心 NuGet 与 C ABI **完全不变** |
| **真图回归矩阵**（`tools/RealImageRegression`，需 GPU + 权重） | ✅ Windows CUDA 10/10 通过：turbo / base、LoRA scale 1.0 与 0.6、512 与 1024、Z-Image bf16 与 Nunchaku r32/r256；重复请求字节级确定性一致 |
| **多实例配置隔离**（`QZI_*` 作用域发布 + 实时环境读取） | ✅ 每个 engine 在 `qzi_engine_create()` 时快照自己的配置，构造结束即还原；Windows CRT 环境快照陷阱已用 `qzi_getenv()` 消除 |
| native 产物统一命名 `QwenZImageNative.dll` / `libQwenZImageNative.so` | ✅ 已完成，无 `h3` 命名产物 |
| 单个 NuGet 包含 .NET 程序集 + win-x64 + linux-x64 native | ✅ 已完成，`QziEnsureNativeRuntimes` 缺平台即拒绝打包 |
| 本地 git 仓库与提交 | ✅ 已完成 |
| 推送 GitHub / 发布 NuGet | ✅ 仓库 `dongfangzhizhu/QwenZImage.NET`；发布走 **NuGet Trusted Publishing**（GitHub Actions OIDC 换短期密钥，仓库不存长期 secret），由 `v*` tag 触发，见 [PLAN.md](PLAN.md) |
| 模型权重 | ❌ 包内不含、仓库不分发；WebUI 的 Models 面板可一键下载全部权重（支持断点续传），也可按 `/api/models/manual` 给出的精确地址手动下载后修改 `QwenZImage.config.json` |

## 配置文件：`QwenZImage.config.json`

格式与上游 `h3.config.json` 类似：top-level 字段对所有平台生效，`windows` / `linux` / `osx` 对象按当前平台覆盖；值可用 `~` 开头；`_` 开头的键视为注释；未知键在 .NET 侧报错。

```json
{
  "device_id": -1,
  "thread_count": 0,
  "qwen21_dir": "/models/qwen21",
  "qwen21_lora_dir": "/models/qwen21/loras",
  "zimage_dir": "/models/zimage",
  "zimage_nunchaku_dir": "/models/zimage_nunchaku",
  "zimage_control_dir": "/models/zimage/control_union_2.1.safetensors",
  "text_cache_dir": "/models/cache",
  "windows": {
    "qwen21_dir": "D:/models/qwen21",
    "zimage_dir": "D:/models/zimage"
  }
}
```

查找顺序：显式 `QZI_CONFIG` / `--config <path>` → 程序目录 → 当前工作目录 → 从这两个目录**向上最多 8 级**（`dotnet run` 会把工作目录设为项目目录，解决方案根目录的配置因此也能找到）→ `~/.config/QwenZImage/config.json`。`QZI_*` 环境变量（`QZI_QWEN21_DIR`、`QZI_ZIMAGE_DIR`、`QZI_QWEN21_LORA_DIR`、`QZI_ZIMAGE_NUNCHAKU_DIR`、`QZI_ZIMAGE_CONTROL_DIR`、`QZI_MODEL_CACHE`、`QZI_DEVICE_ID` 等）优先于文件；代码中显式传入的 `QziClientOptions` 优先于二者。`QziClient` 在**构造 engine 的那一刻**把解析出的 `QZI_CONFIG` 与代码级目录值（`QZI_QWEN21_DIR`、`QZI_ZIMAGE_DIR`、`QZI_QWEN21_LORA_DIR`、`QZI_ZIMAGE_NUNCHAKU_DIR`、`QZI_ZIMAGE_CONTROL_DIR`、`QZI_MODEL_CACHE`）临时发布出去，创建完成后立即还原进程环境；构造过程还有进程内互斥，所以同一进程里多个 `QziClient` 各自快照自己的配置、不会互相串扰。native 侧的 `qzen_config_apply_environment()` 在 `qzen_config_load()` 之后应用同一组覆盖。因此**配置文件、`QziConfig.With(...)`、代码级 options、纯环境变量**这四种方式都能作用到引擎——ABI 里装不下的字段不再静默丢失。

> Windows 上 C 运行时的 `getenv()` 持有启动时的环境快照，看不到 `.NET` 之后设置的值；native 统一通过 `qzi_getenv()`（`native/include/qzi_env.h`，库内部符号、不是 ABI 导出）读实时进程环境，所以这条通道在两个平台都有效。自行实现 backend 时若要读 `QZI_*`，请从仓库源码编译 `native/src/qzi_env.c`（Windows 用 `GetEnvironmentVariableA`），不要裸用 `getenv()`。

仓库根目录的 `QwenZImage.config.json` 是开发机示例，路径指向本机已有的权重目录；部署时改成自己的路径即可。权重文件本身不在仓库里。

## 模型权重下载（WebUI Models 面板）

本仓库**不分发任何权重**。WebUI 的 Models 面板把 WanGP 用到的全部权重列成一张可操作的清单，覆盖 Qwen-Image 2.1（DiT / VAE / 文本塔 / tokenizer / config）、Qwen-Image 2.1 LoRA、Z-Image Turbo（bf16 / Nunchaku INT4）、Z-Image ControlNet Union 2.1、以及控制图预处理 ONNX / `.pth` 资产，共 **6 组 32 个文件**。下载地址全部来自本机 WanGP 源码（`D:/AI/Wan2GP` 的 `defaults/*.json`、各 model handler 与 `preprocessing/*/assets.py`），即 `https://huggingface.co/<repo>/resolve/main/<path>`（repo 为 `DeepBeepMeep/Qwen_image_2`、`DeepBeepMeep/Ideogram4`、`DeepBeepMeep/Z-Image`、`DeepBeepMeep/Wan2.1`）。WanGP 未 pin commit，revision 跟随 `main`。

**三种使用方式：**

1. **WebUI 直接下载**：打开页面下方 Models 面板 → 选择全局 variant（默认 `int8`，即更小的 quanto / ConvRot 重打包）→ `Download required` 或缺什么点什么。下载器串行工作，每个文件写入 `<target>.qzi-part` 并保存 ETag 到 `<target>.qzi-etag`；中断 / 暂停 / 重启进程后以 `Range: bytes=N-` + `If-Range` 续传，完成后校验长度并原子替换目标文件。`Pause` / `Resume` / `Reset partials` 只操作临时文件，**不删除已完成的权重**。
2. **手动下载**：`GET /api/models/manual` 返回一份 Markdown，逐个文件列出**精确的 WanGP 原始地址**、目标绝对路径、可复制的断点续传命令（`curl -L -C - --retry 3 --fail -o "<target>" "<url>"`），以及第 2 步要合并进配置文件的 JSON 片段；面板里也能加载 / 复制 / 下载该文档。设置了 `QZI_HF_MIRROR` 时文档会**同时**给出 WanGP 原始地址和镜像后的有效地址。
3. **改配置文件**：`GET /api/models/config` 默认只返回**预览**（不写盘）；确认后 `POST /api/models/config` 带 `{"apply":true}` 才把 6 个模型目录键合并进当前平台 section，保留其他平台 / 注释 / 未知合法键，先生成 `QwenZImage.config.json.bak.<时间戳>` 备份，再用 `.qzi-tmp` + 原子替换写入。返回 `restart_required=true`，因为 native 引擎只在 `qzi_engine_create()` 时读一次配置。

**API 一览**（全部 snake_case，浏览器只能提交文件 id / variant，不能提交 URL 或目标路径）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/models` | 完整快照：分组 / 每个文件的状态、目标路径、variant、大小、进度、已安装字节 |
| `GET` | `/api/models/manual` | Markdown 手动下载指引（精确地址 + `curl -L -C -` + 配置片段） |
| `GET` | `/api/models/config` | 配置写回预览（只读） |
| `POST` | `/api/models/config` | `{"apply":true}` 时备份并写回配置文件 |
| `POST` | `/api/models/download` | `{"files":[...],"variant":"int8","force":false,"all":false}` → `202` |
| `POST` | `/api/models/control` | `{"action":"pause|resume|reset|reset-all","file":"<id>"}` |

可用 `QZI_HF_MIRROR`（绝对 http(s) 地址，如 `https://hf-mirror.com`）只重写固定的 Hugging Face 前缀；清单里的 repo / 路径不可由调用方指定，避免 SSRF。下载功能**只存在于 host 层** `QwenZImage.NET.WebUI`（`IsPackable=false`），核心 NuGet 包与 C ABI 仍只含模型与推理、零运行期三方依赖。详见 [docs/WEBUI.md](docs/WEBUI.md)。

## 环境要求

- **.NET 10 SDK**（`global.json` 锁定 `10.0.100`）
- 参考 native 构建：**CMake ≥ 3.20 + C11 编译器 + Ninja**（Windows MinGW/MSVC，Linux GCC/Clang）
- 真实 native 构建：上述工具 **+ NVIDIA CUDA Toolkit 13+（nvcc / cudart / cuBLAS / cuBLASLt）+ MSVC（Windows）或 GCC/Clang + OpenMP（Linux）**。Linux **不需要 ICU**：`engine/platform/posix/qzen_posix_unicode.c` 在运行时 `dlopen` 宿主系统的 ICU（找不到时用内置 ASCII/CJK 回退），产物因此不带版本化的 `libicuuc.so.NN` 依赖，可直接跨发行版分发。
- 默认 `QZEN_CUDA_ARCH` 是覆盖 Ampere → Blackwell 的 **fat binary**（`80;86;87;88;89;90;100;103;110;120;121`；不含 sm_75，因为 attention kernel 用 `mma.sync.m16n8k16`）：一个同名库适配常见显卡，无需按 GPU 重编译。CMake 会查询 `nvcc --list-gpu-arch` 自动剔除当前工具链不支持的架构；本地开发传单个 SM（如 `89`）可显著加快编译。
- 打包前需要 Windows 与 Linux 两个平台的 native 产物（Windows 主机可用 WSL 一次构建两者）
- 真实推理还需要**自备模型权重**并在配置文件里指向它们

## 快速开始

```bash
# 1) 参考 native（无 GPU，输出确定性噪点，用于 CI/测试）
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure

# 1b) 真实 C/CUDA native（需要 nvcc；产物单独放 artifacts/native/<Platform>-<Arch>-cuda/）
scripts/build-native-cuda.bat                   # Windows：默认 fat binary，自动加载 MSVC 环境
bash scripts/build-native-cuda.sh               # Linux：默认 fat binary
# 本地只编一个 SM 更快：scripts\build-native-cuda.bat 89 / bash scripts/build-native-cuda.sh 89
# 或手工：cmake -S native -B native/build-cuda -G Ninja \
#           -DQZI_WITH_CUDA_ENGINE=ON -DQZEN_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release

# 2) 全量构建 + 测试
dotnet build QwenZImage.NET.sln -c Release
dotnet test  QwenZImage.NET.sln -c Release

# 3) 跑示例
dotnet run --project samples/QwenZImage.NET.Sample -c Release

# 4) 手动测试 WebUI：浏览器打开 http://127.0.0.1:7860/
dotnet run --project src/QwenZImage.NET.WebUI -c Release
```

> ⚠️ **上面第 4 步默认是噪点后端**（GPU-free 的参考实现，不加载模型）。要生成真实图像，必须构建并**选中** CUDA 后端——只构建是不够的：两者同名但落在不同的 `artifacts/native` 子目录，且为跑无 GPU 测试 / 打包而执行的 `stage-native.ps1` 会把 `artifacts/runtimes/<rid>/native/` 换回参考库。

```powershell
scripts\build-native-cuda.bat 89       # 只需构建一次（89 = RTX 4090）
.\scripts\start-webui-cuda.ps1          # 以后每次启动用这个
```
```bash
bash scripts/build-native-cuda.sh 89
bash scripts/start-webui-cuda.sh
```

启动脚本会设 `QZI_NATIVE_LIBRARY` 指向 `-cuda` 产物，并开启 `QZI_REQUIRE_REAL_BACKEND=1`：若加载到的仍是参考后端，WebUI 直接拒绝生成任务（HTTP 503），不会再用噪点图冒充结果。在开发检出目录里只要存在 `artifacts/native/*-cuda/` 产物，直接 `dotnet run` 也会自动选中它；`QZI_WEBUI_NATIVE=reference` 可强制回参考后端做无 GPU 冒烟。`/api/health` 的 `native_library` / `native_library_source` 会显示实际加载的文件与选择依据。详见 [docs/WEBUI.md](docs/WEBUI.md)。

手动指定也可以（托管程序集与 API 完全不用改）；显式路径现在是权威的，加载失败会直接报错而**不会**静默退回噪点后端：

```bash
export QZI_NATIVE_LIBRARY=/path/to/artifacts/native/Linux-x86_64-cuda/libQwenZImageNative.so
export QZI_CONFIG=/path/to/QwenZImage.config.json
# Windows: $env:QZI_NATIVE_LIBRARY = 'D:\...\Windows-AMD64-cuda\QwenZImageNative.dll'
```

一键脚本（会自动 stage 成 NuGet RID 布局）：

```bash
powershell -File scripts/build-native-all.ps1   # Windows + WSL Linux，参考后端
powershell -File scripts/build-native.ps1       # 仅 Windows，参考后端
bash scripts/build-native.sh                    # 仅当前 Unix 平台，参考后端
scripts/build-native-cuda.bat                # Windows 真实后端（默认 fat binary；传 89 只编单 SM）
bash scripts/build-native-cuda.sh            # Linux 真实后端（同上）
```

> 参考构建与 CUDA 构建输出到不同的 `artifacts/native` 子目录，避免互相覆盖；`QZI_STAGE_CUDA=1` 才把 CUDA 产物 stage 进 `artifacts/runtimes/<rid>/native/`。**stage 只影响 NuGet 包内容，不影响运行时选择**：默认打包形态必须保持 GPU-free，所以运行真图请用 `start-webui-cuda.*` 或 `QZI_NATIVE_LIBRARY`。

## 在 .NET 项目里使用

发布后：`dotnet add package QwenZImage.NET`。本地开发可直接 `dotnet add reference src/QwenZImage.NET/QwenZImage.NET.csproj`。

```csharp
using QwenZImage.NET;

// 不传 options 时自动读取 QwenZImage.config.json
using QziClient client = new();
// 或显式指定：new QziClient(QziClientOptions.FromConfig("/path/to/QwenZImage.config.json"));
// 或完全手工：new QziClient(new QziClientOptions { Qwen21Directory = "/models/qwen21", ... });

IProgress<GenerationProgress> progress = new Progress<GenerationProgress>(
    p => Console.WriteLine($"{p.Phase} {p.Completed}/{p.Total} ({p.Elapsed})"));

// Qwen-Image 2.1（turbo 固定 8 步 / guidance 1.0）
using GenerationResult qwen = await client.GenerateQwen21Async(
    new QwenImageOptions
    {
        Prompt = "A red fox walks through fresh snow, cinematic light",
        Width = 1024, Height = 1024, Seed = 42
    }, progress);
await qwen.SaveAsync("qwen21.png");

// Qwen-Image 2.1 图像编辑：参考图 + 可选的白色笔刷 mask（同一份 7B 权重）
if (client.SupportsImageEditing)
{
    QziRasterImage reference = QziRasterImage.FromPng(
        await File.ReadAllBytesAsync("input.png"));
    QziRasterImage mask = QziRasterImage.FromPngMask(
        await File.ReadAllBytesAsync("mask.png"));   // 也可省略 → 整图编辑

    using GenerationResult edited = await client.GenerateQwen21EditAsync(
        new Qwen21EditOptions
        {
            Prompt = "replace the sky with an aurora, keep the mountains",
            Reference = reference,
            Mask = mask,
            Strength = 0.6f,          // null → 0.75；1.0 ≈ 文生图
            MaskingStrength = 1.0f,  // 仅当有 mask；null → 1.0
            Width = 1024, Height = 1024, Seed = 42
        }, progress);
    await edited.SaveAsync("edited.png");
}

// Qwen-Image 2.1 高级编辑：外扩 / LanPaint / RGBA（仍是同一份 7B 权重）
if (client.SupportsAdvancedEditing)
{
    using GenerationResult outpainted = await client.GenerateQwen21EditExAsync(
        new Qwen21EditExOptions
        {
            Prompt = "a quiet mountain lake at dawn",
            Reference = QziRasterImage.FromPng(
                await File.ReadAllBytesAsync("input.png")),
            // Width / Height 是最终画布；四边各外扩 128 px，中间剩 768×768
            Width = 1024, Height = 1024, Seed = 42,
            MarginTop = 128, MarginBottom = 128,
            MarginLeft = 128, MarginRight = 128,
            ModelMode = Qwen21EditMode.MaskedDenoising,
            Rgba = true                 // VAE 生成的第 4 通道作为 alpha
        }, progress);
    await outpainted.SaveAsync("outpainted-rgba.png");
}

// Z-Image（固定 8 步 Turbo 调度，无 CFG 分支）
using GenerationResult zimage = await client.GenerateZImageAsync(
    new ZImageOptions
    {
        Prompt = "A red fox walks through fresh snow, cinematic light",
        Width = 1024, Height = 1024, Seed = 42, Checkpoint = "bf16"
    }, progress);
await zimage.SaveAsync("zimage.png");

// Z-Image ControlNet Union 2.1：用控制栅格条件化生成
// 核心库按「原样使用」(raw) 语义消费 control_rgb，不做姿态/边缘/深度估计：
// 直接调用 NuGet 包时需自备 guide PNG；WebUI 则会把上传的普通照片
// 经 POST /api/preprocess 转成 guide 后再提交（见下文手动测试 WebUI）。
if (client.SupportsControlNet)
{
    QziRasterImage control = QziRasterImage.FromPng(
        await File.ReadAllBytesAsync("pose-guide.png"));  // 控制栅格（已成形）
    using GenerationResult controlled = await client.GenerateZImageControlAsync(
        new ZImageControlOptions
        {
            Prompt = "a red fox walking through fresh snow, cinematic light",
            ControlImage = control,
            ControlType = "pose",     // 描述性元数据，核心库不运行预处理器
            ControlScale = 0.65f,     // 0 / 不设 → 0.65（WanGP Union 2.1）
            Width = 1024, Height = 1024, Seed = 42
        }, progress);
    await controlled.SaveAsync("zimage-control.png");
}

// 零拷贝直写流：PNG 不落地到托管缓冲区
using FileStream fs = File.Create("zimage-streamed.png");
await client.WriteToAsync(fs, new ZImageOptions { Prompt = "same", Width = 512, Height = 512 });

Console.WriteLine(client.GetHealth().IsReferenceBackend
    ? "reference backend (noise)" : "real C/CUDA backend");
```

### 串行队列（替代原 Go WebUI 的队列）

一个 native engine 只持有 **一张 GPU + 一个常驻模型**，并发第二次生成会返回 `QZI_ERR_BUSY`。`QziGenerationQueue` 用 `System.Threading.Channels` 做严格串行分发：

```csharp
using QziGenerationQueue queue = new(client, async (job, result, ct) =>
    await result.SaveAsync($"outputs/{job.Id}.png", ct));

QziJob job = queue.Submit(QziPresets.WithPrompt(QziPresets.ZImage[1], "a mountain lake"));
await queue.WaitForIdleAsync();          // 或 await foreach (var j in queue.WatchAsync())
Console.WriteLine($"{job.Id}: {job.State}, {job.OutputLength} bytes");

queue.Cancel(anotherId);                 // 排队中的任务不会调用引擎；运行中的走 native 取消回调
QziJob? retry = queue.Retry(job.Id);     // 复用完全相同的 request 重跑
```

支持状态 `Queued / Running / Done / Error / Cancelled`、`JobChanged` 事件、`WatchAsync()` 异步流（替代 SSE）。

## 手动测试 WebUI（跨平台）

`src/QwenZImage.NET.WebUI` 是纯 `.NET 10` / ASP.NET Core Minimal API 的本地控制台，Windows 与 Linux 同一套代码、无 Node/npm、无 Go 进程。功能面：health、presets、串行任务队列、取消、重试、SSE 实时进度、输出画廊、删除到 `.trash` / 恢复、卸载模型，**Qwen-Image 2.1 图像编辑**（模型选 `qwen21-edit`，上传参考图、在 canvas 上绘制白色 mask、调节 strength / masking strength），以及 **Z-Image ControlNet**（模型选 `zimage-control`，上传**普通照片**即可，由 WebUI 的 `POST /api/preprocess` 自动生成 pose / canny / depth / scribble / gray 控制栅格，再选择 control type / scale；上传已成形的 guide 时选 `raw` 原样透传）。遮罩重绘（`zimage-inpaint`）默认 `raw`，避免对已是控制栅格的图再跑边缘检测。两者（编辑 / 控制）互斥，选错模型会在入队前返回 `400`。预处理只在 WebUI host 层运行，核心 NuGet 包与 C ABI 仍是零三方依赖的 raw raster 契约。

```bash
dotnet run --project src/QwenZImage.NET.WebUI -c Release
QZI_WEBUI_URLS=http://127.0.0.1:51234 dotnet run --project src/QwenZImage.NET.WebUI -c Release
dotnet run --project src/QwenZImage.NET.WebUI -c Release -- --config /path/to/QwenZImage.config.json
```

主要端点：`GET /api/health`、`GET /api/presets`、`GET|POST /api/preprocess`（控制图预处理）、`POST|GET /api/jobs`、`GET /api/jobs/{id}`、`POST /api/jobs/{id}/cancel|/retry`、`GET /api/events`（SSE）、`GET|DELETE /api/outputs`、`POST /api/outputs/restore`、`GET|POST /api/unload`。可用环境变量：`QZI_CONFIG`、`QZI_WEBUI_URLS`、`QZI_WEBUI_OUTPUT`、`QZI_MODEL_DIR`、`QZI_MODEL_CACHE`、`QZI_NATIVE_LIBRARY`、`QZI_WEBUI_NATIVE`、`QZI_REQUIRE_REAL_BACKEND`、`QZI_DEVICE_ID`、`QZI_PREPROCESS_DIR`（或配置键 `preprocess_dir`，指向 WanGP ONNX 预处理权重；仅 WebUI 读取，native 引擎忽略）、`QZI_HF_MIRROR`（模型下载镜像，只重写固定的 Hugging Face 前缀）。模型权重相关端点：`GET /api/models`、`GET /api/models/manual`、`GET|POST /api/models/config`、`POST /api/models/download`、`POST /api/models/control`，见上一节。完整说明见 [docs/WEBUI.md](docs/WEBUI.md)。

> WebUI 是**手动测试工具**，不打进 NuGet 包。`/api/health` 的 `transport` 恒为 `native-pinvoke`，并返回 `config_path`、`engine.reference_backend`、`supports_image_editing`、`supports_controlnet`、`supports_advanced_editing`、`supports_inpaint` 与 `preprocessing`（可用算子、已发现的 ONNX 资产）；编辑请求用 `POST /api/jobs` 的 `reference_image` / `mask_image` / `strength` / `masking_strength` 字段（或 `model: qwen21-edit`），ControlNet 请求用 `control_image` / `control_type` / `control_scale`（或 `model: zimage-control`），遮罩重绘在此基础上再加 `mask_image`（或 `model: zimage-inpaint`），高级编辑在 `qwen21-edit` 基础上再加 `margin_*` / `model_mode` / `rgba`。四项能力都是 ABI v1 增量导出，旧 native 库会让对应开关为 `false` 并禁用界面选项（只出现普通编辑字段时仍走旧的 `qwen21-edit` 请求；inpaint 还前置依赖 ControlNet 能力）。当加载到参考后端时页面显示黄色噪点告警横幅；换成 CUDA 构建后横幅消失。

## API 速查

| 类型 | 说明 |
| --- | --- |
| `QziClient` | 拥有的一个 native engine。同步 `Generate*` / `GenerateQwen21Edit` / `GenerateQwen21EditEx` / `GenerateZImageControl` / `WriteTo`，异步 `Generate*Async` / `GenerateQwen21EditAsync` / `GenerateQwen21EditExAsync` / `GenerateZImageControlAsync` / `WriteToAsync`，`GetHealth()`、`Unload()`、`Dispose()`；`SupportsImageEditing` / `SupportsControlNet` / `SupportsAdvancedEditing` 分别探测三个 ABI v1 增量能力 |
| `QziClientOptions` | 模型目录、缓存、设备、native 路径、`ConfigPath`、`Qwen21LoraDirectory` / `ZImageNunchakuDirectory`；`FromConfig(path?)` 从配置文件构造 |
| `QziConfig` | 配置文件加载器（平台覆盖、`~` 展开、未知键报错、搜索路径） |
| `QwenImageOptions` / `ZImageOptions` | 强类型请求（继承 `GenerationOptions`），`required Prompt`、宽高、Seed |
| `Qwen21EditOptions` | 继承 `QwenImageOptions`；`required Reference`、`Mask?`、`Strength?`（默认 0.75）、`MaskingStrength?`（默认 1.0，需 `Mask`） |
| `Qwen21EditExOptions` / `Qwen21EditMode` | 继承 `Qwen21EditOptions`；四边 `Margin*`（32 的倍数，`Width`/`Height` 是最终画布）、`ModelMode`（`MaskedDenoising` / `LanPaint2/5/10/15`）、`Rgba`（PNG colour type 6）。三个能力共用同一份 7B 权重 |
| `ZImageControlOptions` | 继承 `ZImageOptions`；`required ControlImage`、`ControlType?`（默认 `auto`）、`ControlScale?`（默认 0.65）；`EffectiveControlType` / `EffectiveControlScale` 为实际发送值 |
| `QziRasterImage` | 平面 8-bit 栅格：参考图 / 控制图 3 通道 RGB、mask 1 通道。`FromPlanar` / `FromInterleavedRgb` / `SolidMask` / `FromPng` / `FromPngMask` / `FromPngBase64`，无需 System.Drawing 等图像依赖 |
| `GenerationResult` | PNG 结果：`Png` / `PngSpan`（零拷贝）、`CopyTo(Async)`、`SaveAsync`、`Dispose()` 归还池化缓冲区 |
| `GenerationProgress` | `Phase`（`ModelLoad / Text / ImageEncode / Dit / Vae / Encode`，未知字面量降级为 `Unknown`）、`Completed`、`Total`、`Elapsed`；`ImageEncode` 在图像编辑与 Z-Image ControlNet 时出现 |
| `EngineHealth` | 常驻模型、Busy、队列深度、显存、Served/Rejected、运行时长、`IsReferenceBackend` |
| `QziGenerationQueue` | 串行队列、取消、重试、事件流、输出 sink |
| `QziPresets` | 5 个 Qwen21 + 4 个 ZImage 预设模板，用 `WithPrompt()` 变成可执行请求 |
| `RequestValidator` | 托管侧参数校验（与 native 侧规则一致，先于分钟级模型加载失败） |
| `QziException` | native 错误映射，含 `Status` 与 `NativeMessage` |

## 参数与校验

规则在 `.NET` 与 native 两层重复执行，避免把明显非法的请求送进分钟级模型加载。完整表格见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。要点：

- **Qwen21**：宽高为 **32 的倍数**、`32..2048`、面积 ≤ `2048×2048`；`turbo` 强制 `steps=8`、`guidance=1.0`、不支持 `negative_prompt`；`base` 为 `steps 1..50`、`guidance 1..20`；`lora` 必须是**裸文件名**（≤ 255 UTF-8 字节，禁止 `/`、`\`、`:`、`..` 与控制字符），`lora_scale` 必须有限且在 `0.05..4`。**LoRA 在 turbo / base 两个 tier 都可用**，切换 adapter 或 scale 会重载 DiT。
- **ZImage**：宽高为 **16 的倍数**、`16..2048`、面积 ≤ `2048×2048`；`checkpoint` 支持 `bf16`、`nunchaku`、`nunchaku-<pack>`；**`fp4` / `svdq-fp4`（大小写不敏感）不被支持**（引擎只解码 int4）；无 steps/guidance/negative prompt（Turbo 固定 8 步、无 CFG 分支）。
- **默认值**：native ABI 用 `steps=0` / `guidance=0` / `lora_scale=0` 表示「用 tier 默认值 / 无 adapter」；`qzi_generate_qwen21()` 在 backend 之前统一归一化，两个后端看到同一组有效值，非法值一律拒绝而不静默改写。详见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。
- **图像编辑**：`Reference` 必填（平面 RGB8，每边 ≤ 4096、总像素 ≤ `2048×2048`），`Mask` 尺寸须与参考图一致且为单通道；`Strength` / `MaskingStrength` 为有限值 `0.05..1.0`（`null` 分别为 `0.75` / `1.0`），`MaskingStrength` 必须有 mask；参考图与 mask 会 cover fit 到输出画布，白色 mask 区域重绘、黑色区域保持参考轨迹。ABI 侧 `strength=0` / `masking_strength=0` 是同样的默认值 sentinel。详见 [docs/PARAMETERS.md](docs/PARAMETERS.md) 与 [docs/NATIVE_ABI.md](docs/NATIVE_ABI.md)。
- **Z-Image ControlNet**：`ControlImage` 必填（平面 RGB8，尺寸限制同上），`ControlType` 为 `auto` / `pose` / `canny` / `depth` / `scribble` / `raw`（`null` → `auto`，仅描述性元数据，**核心库不运行预处理器**：底层固定按 raw raster 消费 `ControlImage`；需要姿态 / 边缘 / 深度 guide 时，由应用层生成——WebUI 用 `POST /api/preprocess` 把普通照片转成控制栅格，对齐 WanGP `wgp.py::process_map_video_guide` 的 `P/D/E/S/C/V` 映射），`ControlScale` 为有限值 `0.05..2.0`（`0` / 不设 → `0.65`）。需配置 `zimage_control_dir` 指向 Union 2.1 权重，仅支持 33 通道的 v2 / v2.1；未配置时真实后端返回 `QZI_ERR_NOT_CONFIGURED`。详见 [docs/PARAMETERS.md](docs/PARAMETERS.md) 与 [docs/NATIVE_ABI.md](docs/NATIVE_ABI.md)。
- **高级图像编辑**：`Qwen21EditExOptions` 继承上面所有编辑规则；`Width`/`Height` 是**含外扩边距的最终画布**，四个 `Margin*` 必须是 **32 的倍数**且留下至少 `32×32` 的内部源矩形；`ModelMode` 只能是 `MaskedDenoising(0)` / `LanPaint2|5|10|15(2/3/4/5)`，LanPaint 必须有 `Mask`、不能与外扩并用、且强制两个 strength 为 `1.0`；`Rgba` 输出 PNG colour type 6。三项能力无需新权重，但 native 库必须是第三个增量扩展（`SupportsAdvancedEditing`）。详见 [docs/PARAMETERS.md](docs/PARAMETERS.md) 与 [docs/NATIVE_ABI.md](docs/NATIVE_ABI.md)。
- 三者：prompt 非空且 ≤ **16384 UTF-8 字节**，保留字段 `flags` 必须为 0。

## 性能设计

- `LibraryImport` **源生成 P/Invoke** + 全 blittable POD 结构体，无反射式 marshalling；启动时 `qzi_abi_layout()` 校验 6 个结构体尺寸与 ABI 版本，packing 不匹配立即失败而不是踩坏内存。
- `ArrayPool<byte>` 池化、pinned NUL-terminated UTF-8 字符串（`GCHandle`）、`Span` / `ReadOnlySpan`，热路径**零 string 分配**。
- `static` + `UnmanagedCallersOnly` 反向回调（progress / cancel），无 delegate 分配。`CallbackScope` 用进程级 `ConcurrentDictionary<long, CallbackState>` token 表代替裸 `GCHandle`：stale / garbage token 直接忽略，不会解引用已释放句柄；两个回调捕获**所有**托管异常（`IProgress<T>.Report` / SSE subscriber 抛错也不会穿过 CUDA/ggml native 栈），记录首个 fault、让 `ShouldCancel` 返回 1 停止循环，并在 P/Invoke 返回后于托管侧以 `InvalidOperationException`（含 inner exception）重抛。这是 `System.ExecutionEngineException` / `0x80131506` 的根治点。
- 成功路径 native PNG buffer → 一次 `memcpy` → 立刻 `qzi_image_release`；`WriteTo` 走 `UnmanagedMemoryStream` **真正零拷贝**。
- 纯同步 native 调用 + `Task.Run` 异步外壳；队列保证单飞，DiT 常驻避免分钟级重复加载。

## NuGet 包内容（.NET + Windows + Linux native）

```
QwenZImage.NET.0.1.0.nupkg
├─ lib/net10.0/QwenZImage.NET.dll
├─ lib/net10.0/QwenZImage.NET.xml
├─ runtimes/win-x64/native/QwenZImageNative.dll
├─ runtimes/linux-x64/native/libQwenZImageNative.so
├─ README.md / LICENSE / THIRD_PARTY_NOTICES.md / engine/LICENSE
└─ docs/*.md
```

打包前置条件：两个平台的 native 库必须 stage 到 `artifacts/runtimes/<rid>/native/`。`QwenZImage.NET.csproj` 的 `QziEnsureNativeRuntimes` target 会在 `GenerateNuspec` 前检查两者存在，缺任一平台直接报错。

```bash
dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj -c Release -o artifacts/packages
```

> 默认打进包里的仍是**参考后端**（无 GPU 依赖、开箱即用、输出噪点），因为 CI 没有 GPU。真实 CUDA 构建现在默认产出覆盖 Ampere → Blackwell 的 **fat binary**，一个同名库即可直接替换 `runtimes/<rid>/native/`，或用 `QZI_NATIVE_LIBRARY` 指向 `artifacts/native/<Platform>-<Arch>-cuda/` 里的产物，不必按显卡重编译；单 SM 构建仍可用脚本参数指定。托管 API 不变，权重始终由使用方自带。

## native 库解析顺序

1. `QziClientOptions.NativeLibraryPath`（或 `QZI_NATIVE_LIBRARY`）— 最高优先级
2. 运行时提供的 `NATIVE_DLL_SEARCH_DIRECTORIES`（NuGet `runtimes/<rid>/native/` 即通过它生效）
3. 托管程序集同目录，含 `runtimes/<rid>/native/` 与 `native/`
4. 从程序集目录向上查找 `artifacts/runtimes/<rid>/native/` 与 `artifacts/native/<Platform>-<Arch>/`
5. 运行时默认探测（`PATH` / `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`）

文件名：Windows `QwenZImageNative.dll`、Linux `libQwenZImageNative.so`、macOS `libQwenZImageNative.dylib`。当前提供 **win-x64 与 linux-x64**。

## 目录结构

```
QwenZImage.config.json          模型权重路径配置（平台覆盖 / 环境变量覆盖）
native/include/qzi.h            稳定 C ABI（唯一必须遵守的契约）
native/include/qzi_backend.h    可插拔后端 vtable
native/src/qzi.c                ABI 控制层：生命周期、校验、单飞锁、错误、进度
native/src/qzi_engine_backend.c 真实引擎适配器（in-process qzi_backend 实现）
native/src/qzi_stub.c           确定性参考后端（噪点，CI 用）
native/src/qzi_png.c            无依赖最小 PNG 编码器
native/tests/                   C ABI smoke test（按后端自动分支）
engine/                         vendored + 改名的真实 C/CUDA 推理引擎
  src/, platform/win32/, platform/posix/, third_party/, cmake/, CMakeLists.txt
  LICENSE / UPSTREAM.md         上游 MIT 许可与来源说明
src/QwenZImage.NET/             .NET 10 库（NuGet 包体）
  Configuration/QziConfig.cs    配置文件加载器
src/QwenZImage.NET.WebUI/       跨平台手动测试 WebUI（不打包）
  ModelDownloads/               WanGP 权重清单 / 断点续传下载器 / 配置写回（仅 host 层）
  Preprocessing/                控制图 guide 预处理（仅此处引用 OnnxRuntime）
tests/                          xUnit 托管测试（含 ModelDownloadsTests.cs）
samples/                        端到端示例
integration/                    自定义 qzi_backend 模板
tools/RealImageRegression/      真图回归 harness（需 GPU + 权重，刻意不进 sln / CI）
scripts/                        native 构建、RID staging 与真图回归运行入口
docs/                           ABI / 参数 / 引擎接入 / WebUI 文档
.github/workflows/              CI 与 NuGet 发布工作流
PLAN.md                         计划、决策记录与发布清单
```

## 测试

```bash
ctest --test-dir native/build --output-on-failure       # 参考后端完整契约（Windows 1 项 / Linux 3 项）
ctest --test-dir native/build-cuda --output-on-failure  # 真实后端 ABI + 校验（无需权重）
dotnet test QwenZImage.NET.sln -c Release               # 托管测试
```

Linux 上 `ctest` 另有 `posix_unicode_with_icu` / `posix_unicode_without_icu` 两项，验证 POSIX ICU 兼容层的两条运行时路径：前者 `dlopen` 命中宿主 ICU 并把 `e + U+0301` 组合成 `é`，后者用 `-Wl,--wrap=dlopen` 强制「无 ICU」走内置回退（恒等 NFC）。二者共同覆盖字符分类、空白判断、含 4 字节代理对的 UTF-8 ↔ UTF-16 round-trip 与非法 UTF-8 报错。因为 `qzen_posix_unicode.c` 用 `pthread_once` 只解析一次 ICU，两条路径由两个独立可执行分担；链接器不支持 `--wrap` 时 configure 会优雅跳过该变体，宿主无 ICU 时对应测试记为 SKIP。这两项**不需要 CUDA**，GPU-less 的 Linux CI 在 reference 构建里就会跑。

托管测试共 **217 项**（118 库 + 99 WebUI；阶段 20 为 202 项，阶段 21 新增 `ModelDownloadsTests.cs` 15 项）：配置文件加载与平台覆盖、`QziConfig` / 环境变量 / 代码级 options 覆盖链、参数校验（含 LoRA 裸文件名、长度、finite scale、双 tier）、ABI 布局、PNG 生成与阶段回调、确定性输出、零拷贝写流、health 计数、异步包装、预取消、native 层重复校验、队列串行/取消/重试/事件流、WebUI 运行时配置与双 tier LoRA 映射、`QZI_*` 作用域发布 / 还原、构造失败回滚、多 engine 配置快照与并发构造隔离，以及 native 后端选择（自动探测 CUDA / 显式路径优先 / 强制 reference / 拒绝噪点开关）。

编辑相关测试覆盖：`QziRasterImage` 的平面通道顺序 / 尺寸校验与 PNG 解码（灰阶 / RGB / RGBA、alpha 合成、base64 data URL、非法与不支持的输入）、`Qwen21EditOptions` 的校验（mask 尺寸、strength / masking_strength 边界与依赖）、编辑能力探测、编辑请求生成与确定性（换参考图 / mask / strength 都会改变输出字节）、WanGP 步数换算与 `"image encode"` phase、同步 / 异步 / 零拷贝 / 队列四条调用路径，以及 WebUI 的 `reference_image` / `mask_image` / `strength` / `masking_strength` 字段映射、编辑别名与缺字段时的 `400` 语义。

ControlNet 相关测试覆盖：`ZImageControlOptions` 的控制图尺寸 / `control_type` 白名单 / `control_scale` 有限性与边界、能力位与 `qzi_zimage_control_request` 布局探测、控制请求生成与确定性（换控制图 / type / scale 都改变输出字节）、`"image encode"` phase、同步 / 异步 / 零拷贝路径，以及 WebUI 的 `zimage-control` 别名、`control_image` / `control_type` / `control_scale` 映射、编辑与 ControlNet 互斥、缺控制字段时的 `400` 语义。高级编辑相关测试覆盖：`Qwen21EditExOptions` 的边距 32 对齐与残余源矩形、`model_mode` 白名单、LanPaint 需 mask / 与外扩互斥 / strength 约束、`rgba` 归一化、第三能力位与 `qzi_qwen21_edit_ex_request` 布局探测、外扩边距改变输出字节、RGBA PNG colour type 6 与同步 / 异步路径，以及 WebUI 的 `margin_*` / `model_mode` / `rgba` 映射与「无高级字段仍走普通编辑」的兼容行为。遮罩重绘相关测试覆盖：`ZImageInpaintOptions` 的 mask 必填 / 单通道 / 尺寸上限、第四能力位与 `qzi_zimage_inpaint_request` 布局探测、inpaint 与 ControlNet 输出的字节差异、mask 内容与尺寸改变输出、相同请求确定性、同步 / 异步 / 零拷贝路径，以及 WebUI 的 `zimage-inpaint` 别名、`control_image + mask_image` 自动升级、编辑与 inpaint 互斥、`supports_inpaint: false` 时的 `400` 语义。C smoke 另覆盖四个能力位（`qzi_abi_features() == 0xF`）、四个扩展 layout、编辑 / 控制 / 高级编辑 / inpaint 参数的归一化与生成契约、inpaint 的两个 `"image encode"` 阶段，以及 legacy backend 四个可选槽留 `NULL` 时返回 `QZI_ERR_UNSUPPORTED`。全部不需要 GPU、权重或网络监听。

**真实 CUDA 编辑验证**（不属于 CI，需 CUDA 构建 + 权重）：以 512×512 的「左蓝底白圆 / 右橙色」参考图配「左黑 / 右白」mask、`masking_strength=1.0` 提交 `qwen21-edit` 任务。`strength=0.6` 时参考图与输出的半幅平均绝对差为左 **0.541** / 右 **17.115**，`strength=1.0` 时为左 **0.981** / 右 **88.468**：黑 mask 区始终逐像素锁定，白 mask 区被完全重绘成雪地红狐，且重绘幅度随 `strength` 按 WanGP 的步数比例规则放大。phase 序列出现 `ImageEncode`，确认参考图确实经过 VAE 编码而非走文本生成路径。详见 [PLAN.md](PLAN.md) 的「阶段 15 本机验证结果」。

**真实 CUDA 遮罩重绘验证**（不属于 CI，需 CUDA 构建 + Union 2.1 权重）：以 256×256「左蓝底白圆 / 右橙色」控制图配「左黑 / 右白」单通道 mask 提交 `zimage-inpaint` 任务。相对全黑（全 keep）的 mask，`inpaint_half` 左 / 右半幅平均绝对差为 **36.765 / 118.612**，反转 mask 后为 **104.578 / 8.243**：白区被明显重绘、黑区更贴近控制图，且极性随 mask 精确翻转；普通 ControlNet 对 t2i 的 MAD 为 **55.701**，inpaint 对普通 ControlNet 为 129.610 / 130.589，确认 33 通道 context 的后 17 个平面真实参与采样而非退化成 plain ControlNet。7 例真图（含 `control_scale=0.2`、128×256 异尺寸 mask 的独立 cover fit、两次独立运行的逐项 SHA-256 一致）全部产出合法 PNG；缺 mask / 缺控制图 / 非法 `control_type` / 越界 `control_scale` 四种请求均在 `/api/jobs` 返回 `400`。真实后端把两次 VAE 编码折算进**单个** `ImageEncode` 阶段（`0/1 → 1/1`），只有 GPU-free 参考后端拆成 `0/2 → 1/2 → 2/2` 两段。详见 [PLAN.md](PLAN.md) 的「阶段 19 本机验证结果」。

模型下载相关测试（15 项，全部用可注入 `HttpClient` + 本地 `RangeHandler` / `ThrottledStream`，**不需要真实网络或权重**）覆盖：manifest 的 URL / 大小 / 路径安全性、`QZI_HF_MIRROR` 重写、快照与手动指引、下载计划展开、配置写回（预览只读 / 备份 / 保留其他 section）、完整下载、HTTP `Range` + `If-Range` 续传、服务器忽略 Range 时重新全量下载、截断续传、pause / resume / reset / `Reset()` 竞态。另用本地 HTTP Range server 对真实 WebUI 端点做过端到端验证（暂停在 512 KiB → 带 `Range: bytes=524288-` 与 ETag 续传 → 2 MiB 载荷逐字节一致 → 临时文件清理 → 配置真实写回），21/21 检查通过。

CI 另做 WebUI Linux 运行时冒烟：启动 Kestrel → 验证 `/api/health` 的 `ok=true` / `transport=native-pinvoke` / `config_path` → 检查 presets → 提交 zimage job → 轮询到完成 → 确认输出出现在 `/api/outputs`。

**真图回归**（不属于 CI，需要 CUDA 构建 + 多 GB 权重）用 `tools/RealImageRegression`：

```bash
.\scripts\run-real-image-regression.ps1 --list
.\scripts\run-real-image-regression.ps1 --out artifacts\regress\full
bash scripts/run-real-image-regression.sh --out artifacts/regress/full
```

`cases.json` 覆盖 qwen21 turbo/base、LoRA 及 scale 变化、1024×1024，以及 Z-Image bf16 与 Nunchaku r32/r256 checkpoint；每个用例输出 PNG、SHA-256、phase 序列与 `report.md` / `report.json`，并用 `compare_with` 做字节级确定性对照。工具检测到参考后端会直接拒绝运行。详见 [docs/ENGINE_INTEGRATION.md](docs/ENGINE_INTEGRATION.md) 的「真图回归矩阵」。

## 路线与发布

仓库：`https://github.com/dongfangzhizhu/QwenZImage.NET`。

见 [PLAN.md](PLAN.md)：阶段 0–12 已完成（调研 → C ABI → .NET 封装 → 队列/预设 → 文档/CI → 跨平台 WebUI + 双平台 native 包 → 噪点根因排查 → **D10 直接绑定真实 native 后端**），阶段 12b（双平台 fat binary / 无 ICU 的 Linux 运行时）、阶段 12c（**LoRA / 参数全链路一致性**）、阶段 12d（**真图回归矩阵 10/10**，覆盖 LoRA、base、1024×1024、Z-Image bf16 与 Nunchaku r32/r256）、阶段 12e（**POSIX ICU 双路径检查接入 CTest**）、阶段 12f（**消除 `QZI_*` 进程级环境变量串扰**）、阶段 13（Trusted Publishing）与阶段 14（WebUI 默认不再产出噪点图）亦已完成；**阶段 15（Qwen-Image 2.1 图像编辑 / 对齐 WanGP）** 完成 ABI v1 增量扩展 + native 校验与归一化 + 真实 CUDA 后端 VAE 编码 / mask 恢复 + .NET API + WebUI 上传与绘制 mask；真实 CUDA 双轮编辑真图验证（白 mask 区重绘 / 黑 mask 区锁定、`strength` 生效）已通过。**阶段 16（Z-Image Turbo Fun ControlNet Union 2.1 / 对齐 WanGP `z_image_control2_1`）** 完成第二个 ABI v1 增量扩展、33 通道权重加载器、真实 CUDA 后端 hint 注入与逐步重算、.NET `ZImageControlOptions` / `SupportsControlNet` 与 WebUI `zimage-control` 模式。**阶段 17（Qwen-Image 2.1 高级编辑）** 是第三个增量（`0x4` 能力位、`qzi_generate_qwen21_edit_ex`），在同一份 7B 权重上补齐 Red Canvas 外扩（32 对齐四边距）、LanPaint（`model_mode` 2/3/4/5）与 VAE 生成 alpha 的 RGBA PNG 输出。**阶段 18** 收尾审计把高级编辑的 C smoke 检查接进 `main()` 并按本机 WanGP `5b63ae9d` 源码复核边界。**阶段 19（Z-Image Union 2.x 遮罩重绘）**：第四个 ABI v1 增量扩展（`qzi_zimage_inpaint_request` + `QZI_FEATURE_ZIMAGE_INPAINT = 0x8` + `qzi_generate_zimage_inpaint()`），补齐 WanGP `pipeline_z_image.py` 中 33 通道 context 的带 mask 分支——两次 VAE 编码、`keep_mask = 1 - mask` 收缩到 1/8 latent、`context = [control_latent(16) | keep_mask(1) | inpaint_latent(16)]`；`qzi_abi_features()` 现为 `0xF`、`qzi_backend` 共 11 个回调。.NET 新增 `ZImageInpaintOptions` / `SupportsInpaint`，WebUI 加 `zimage-inpaint` 模式并复用控制图上传与笔刷 mask；6 份文档与 **162 项测试**同步。**阶段 20（回调崩溃根治 + WebUI 控制图预处理 / 继续对齐 WanGP）**：不改 C ABI 与推理引擎，只在 `.NET` 与 host 层收口三个现场问题——① `QziClient.cs` 的 `System.ExecutionEngineException` / `0x80131506` 由 `CallbackScope` 的 token 表 + 异常封边根治；② 允许 `QwenZImage.NET.WebUI` 引入 `Microsoft.ML.OnnxRuntime` 与图像算子，新增 `GET|POST /api/preprocess`，把「必须上传已预处理控制图」从 WebUI 的默认路径中移除；③ 前端删除「Used as-is (no preprocessing)」，改为上传普通照片即生成 guide 并可重跑。核心 NuGet 包仍只含模型与推理、保持零三方依赖与 raw raster 契约。全量 **202 项测试**全绿（118 库 + 84 WebUI），真实 CUDA 后端下「普通照片 → canny guide → `zimage-control` → SSE 断连 → 合法 PNG」端到端通过。**阶段 21（WebUI 模型权重下载 / 断点续传 / 手动地址 / 配置写回）为最新一轮**：下载地址全部取自本机 WanGP 源码，新增 `ModelDownloads/` 静态清单与下载服务、6 个 `/api/models*` 端点、前端 Models 面板、手动下载 Markdown 指引与安全的配置写回；下载能力只在 host 层，核心 NuGet 与 C ABI 不变。全量 **217 项测试**全绿（118 库 + 99 WebUI），并用本地 HTTP Range server 对真实端点完成 21/21 端到端检查。**阶段 22（发布前 CI 回归清零）**：修掉 Linux 上 native `getenv()` 看不到 .NET 环境变量（`PosixEnvironment` 同步 libc `setenv`）、下载测试夹具写死 Windows 平台、以及暂停后 reset 残留 `.qzi-part` 三个问题，Windows 与 WSL Linux 双平台各 **217/217** 全绿；Trusted Publishing 的逐步发布清单见 [PLAN.md](PLAN.md) 第 5 节。**阶段 23（首次发布失败定位）**：`CI` 已全绿，红的是 `v0.1.0` 触发的 `Release to NuGet`——GitHub 侧把 nuget.org 用户名误填成 secret 的**名字**（只有 `WENLEE`，缺 `NUGET_USER`），补建 `NUGET_USER = wenlee` 后构建 / 217 测试 / pack / 包内容校验全过；随后 `NuGet/login@v1` 返回 `No matching trust policy owned by user ... was found`（tag 与 `workflow_dispatch` 均同一 401），说明剩余阻塞在 nuget.org 侧 Trusted Publishing policy 的 owner / 字段 / scopes，需按 [PLAN.md](PLAN.md) 第 5 节核对后直接 Re-run，包尚未发布、无需改版本号。按用户要求不做 macOS 适配。

**发布采用 NuGet Trusted Publishing**：`.github/workflows/release.yml` 在 `v*` tag 上触发，`publish` job 以 `permissions.id-token: write` 申请 GitHub OIDC token，`NuGet/login@v1` 用它换取有效期 1 小时的短期 API key，随后发布 `.nupkg` / `.snupkg`；仓库**不保存长期 `NUGET_API_KEY`**。首次发布前需完成两项外部配置：

1. 在 **nuget.org** 为包所有者账号创建 Trusted Publishing policy：Repository Owner `dongfangzhizhu`、Repository `QwenZImage.NET`、Workflow File `release.yml`（只填文件名）、Environment 留空。
2. 在 **GitHub 仓库**设置 repository secret `NUGET_USER` = nuget.org 的 **profile 用户名（不是邮箱）**，须与 policy owner 一致。

完整清单与 tag 发布命令见 [PLAN.md](PLAN.md) 第 5 节。

## 许可与声明

MIT，见 [LICENSE](LICENSE)。`engine/` 下的真实推理引擎来自 MIT 许可的上游项目，许可与来源见 [engine/UPSTREAM.md](engine/UPSTREAM.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目**不包含任何模型权重、不包含 Go 代码、不包含上游 HTTP 服务/CLI**；模型名称与商标归各自权利人所有。
