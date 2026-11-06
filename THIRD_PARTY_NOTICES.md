# Third-Party Notices

QwenZImage.NET itself is licensed under the MIT license (see `LICENSE`).
This file lists third-party source code that is **included in this repository**,
plus dependencies that are only used at build/test time.

## Bundled source code

### 1. Qwen-Image 2.1 / Z-Image C/CUDA engine (`engine/`)

`engine/` is a renamed subset of the upstream project `h3-spark.c`, used under
its MIT license. See [`engine/UPSTREAM.md`](engine/UPSTREAM.md) for the exact
file list and the mechanical rename. The upstream license is reproduced in
[`engine/LICENSE`](engine/LICENSE) and below:

> MIT License
>
> Copyright (c) 2026 Salvatore Sanfilippo
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.

The vendored engine is compiled into the native shared library
`QwenZImageNative.dll` / `libQwenZImageNative.so` only when building with
`-DQZI_WITH_CUDA_ENGINE=ON`. The default GPU-less build uses the original
reference backend instead and contains none of this code.

No Go code, no HTTP server and no copy of the upstream CLI/WebUI is included.
The optional `QwenZImage.NET.WebUI` is an original ASP.NET Core Minimal API app;
it reproduces the *functional surface* of the old Go WebUI, not its source.

### 2. cJSON (`engine/third_party/cJSON.c`, `cJSON.h`)

MIT-licensed, Copyright (c) 2009-2017 Dave Gamble and cJSON contributors. The
license text is reproduced at the top of `cJSON.h`.

### 3. Original code written for this repository

* `native/include/qzi.h`, `qzi_backend.h`, `native/src/qzi.c`,
  `qzi_engine_backend.c`, `qzi_stub.c`, `qzi_png.c` and the native smoke test.
  `qzi_png.c` implements PNG chunks, stored (uncompressed) DEFLATE blocks,
  CRC-32 and Adler-32 directly; it does **not** link zlib, libpng,
  stb_image_write or any other library.
* `src/QwenZImage.NET/**` (the .NET 10 wrapper) and
  `src/QwenZImage.NET.WebUI/**`.
* `qzi_stub.c` produces synthetic noise images from the request parameters. It
  performs no diffusion inference and embeds no model, dataset or weights.

## Model weights and trademarks

- **No model weights, checkpoints, tokenizers, LoRA adapters or VAE files are
  distributed with this project or its NuGet package.** Users must obtain them
  separately from their respective rights holders and comply with the
  applicable model licenses. `QwenZImage.config.json` only contains example
  paths to directories on the developer's machine.
- "Qwen", "Qwen-Image" and "Z-Image" and related marks are the property of
  their respective owners. Their appearance in identifiers, documentation and
  parameter names is descriptive only and does not imply endorsement,
  affiliation or sponsorship.
- "Nunchaku", "SVDQ" and pack names such as `nunchaku-r256` are referenced only
  as engine checkpoint identifiers; this project does not ship those packs.
- `.NET`, `NuGet`, `GitHub`, `Windows`, `Linux` and `CUDA` are trademarks of
  their respective owners.

## Build-time and test-time dependencies (not redistributed)

| Component | Where it is used |
| --- | --- |
| .NET 10 SDK / runtime (Microsoft) | building and running the managed library |
| Microsoft.SourceLink.GitHub 8.0.0 | deterministic builds, source debugging |
| Microsoft.NET.Test.Sdk 17.12.0 | test host |
| xunit 2.9.2 / xunit.runner.visualstudio 2.8.2 | unit tests |
| CMake, Ninja, GCC/Clang/MinGW/MSVC | native builds |
| NVIDIA CUDA Toolkit (nvcc, cudart, cuBLAS, cuBLASLt) | only for `-DQZI_WITH_CUDA_ENGINE=ON` |
| Windows SDK / ICU / OpenMP runtime | host build of the real backend |

Each of these carries its own license; consult the respective package or
toolchain for the exact terms. They are referenced through NuGet / the
toolchain and are not copied into the produced `QwenZImage.NET` package.
Consumers who redistribute a native build must also retain the MIT notices
above and comply with the licenses of the model weights they pair with it.
