# Vendored inference engine — provenance and licensing

`engine/` contains the real Qwen-Image 2.1 / Z-Image C/CUDA inference engine.
It is a **renamed subset** of the MIT-licensed upstream project
`h3-spark.c` by Salvatore Sanfilippo (Copyright (c) 2026). The upstream MIT
license is reproduced verbatim in [`LICENSE`](LICENSE) and in the project-level
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), as the license requires.

## What was copied

Only the files needed by the two text-to-image pipelines:

| Area | Files |
| --- | --- |
| Weight loading | `qzen_safetensors.c`, `qzen_weights.c`, `qzen_quanto.c`, `qzen_nunchaku.c`, `qzen_tokenizer.c` |
| CUDA / device | `qzen_gpu.cu`, `qzen_cuda.c`, `qzen_device.c`, `qzen_gpu_stubs.c` |
| Qwen-Image 2.1 | `qzen_qwen21_dit/text/vae/pipeline.c` plus headers |
| Z-Image | `qzen_zimage_dit/text/vae/pipeline.c` plus headers |
| Windows POSIX shim | `platform/win32/*` |
| JSON parser | `third_party/cJSON.c` / `cJSON.h` (MIT, Dave Gamble and cJSON contributors) |

The upstream orchestrator (`h3.c`), the HTTP service (`h3_serve.c`), the CLI,
the video/WAN pipelines and the Go WebUI were **not** copied. The generation
orchestration that used to live in the HTTP service is re-implemented as an
in-process `qzi_backend` vtable in
[`native/src/qzi_engine_backend.c`](../native/src/qzi_engine_backend.c); there
is no server, no socket and no JSON hop between .NET and the engine.

`src/qzen_config.c`, `src/qzen_config.h`, `CMakeLists.txt` and
`cmake/QzenPlatform.cmake` are new code written for this repository.

## Mechanical rename applied

| Upstream | Here |
| --- | --- |
| `h3.h`, `h3_*.c/.h/.cu` | `qzen.h`, `qzen_*.c/.h/.cu` |
| `h3_*` / `H3_*` symbols and macros | `qzen_*` / `QZEN_*` |
| `H3_SM_ARCH` | `QZEN_SM_ARCH` |
| `h3.config.json`, `H3_CONFIG` | `QwenZImage.config.json`, `QZI_CONFIG` |
| `h3 serve`, the `h3` binary | `QwenZImageNative` shared library |

All 64 copied files were verified to contain zero residual `h3` identifiers
and zero unresolved quoted includes. No `h3`-named artifact is produced by the
build: the sole native output is `QwenZImageNative.dll` /
`libQwenZImageNative.so`.

## Behavioural differences from the upstream service

* `qzi.c` owns validation, single-flight serialisation, progress, cancellation,
  error strings and PNG buffer ownership; the engine never sees the network.
* PNG bytes are produced in memory by `qzi_png_encode_rgb8()` instead of being
  written to a temporary file and read back.
* Model paths come from `QwenZImage.config.json` (via `QZI_CONFIG`), with the
  `qzi_engine_config` fields from .NET taking precedence when explicitly set.
* The DiT stays resident across calls; `unload()` only releases re-creatable
  state, matching the upstream resident service.
* The bundled `qzi_stub.c` reference backend is **not** derived from upstream;
  it is original code that emits deterministic noise for ABI/CI tests.

## Model weights

No weights, tokenizers, LoRA adapters or VAE checkpoints are distributed here.
Point `QwenZImage.config.json` at packs obtained separately from their rights
holders and comply with their licenses.
