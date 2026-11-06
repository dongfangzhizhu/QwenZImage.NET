# Third-Party Notices

QwenZImage.NET itself is licensed under the MIT license (see `LICENSE`).

## Bundled source code

This repository does **not** vendor third-party source code. In particular:

- No source from `h3-spark.c`, no Go code from any WebUI, and no copy of the
  original C/Go HTTP server is included here. The `qzi_*` C ABI, the .NET
  wrapper, and the optional `QwenZImage.NET.WebUI` (an original ASP.NET Core
  Minimal API app using only the .NET framework) were written for this project.
  The WebUI reproduces the *functional surface* of the old Go WebUI, not its
  source code.
- The minimal PNG encoder in `native/src/qzi_png.c` is original code written for
  this repository. It implements PNG chunks, stored (uncompressed) DEFLATE
  blocks, CRC-32 and Adler-32 directly, and does **not** link zlib, libpng,
  stb_image_write or any other library.
- The deterministic reference backend in `native/src/qzi_stub.c` produces
  synthetic images from the request parameters. It performs no diffusion
  inference and embeds no model, no dataset and no weights.

## Model weights and trademarks

- **No model weights, checkpoints, tokenizers, LoRA adapters or VAE files are
  distributed with this project or its NuGet package.** Users must obtain them
  separately from their respective rights holders and comply with the
  applicable model licenses.
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
| CMake, Ninja, GCC/Clang/MinGW/MSVC | building the bundled win-x64 / linux-x64 reference native libraries |

Each of these carries its own open-source license; consult the respective
package or toolchain for the exact terms. They are referenced through NuGet /
the toolchain and are not copied into the produced `QwenZImage.NET` package.
Consumers who redistribute the package should also consult the licenses of the
native engine and model weights they choose to pair with it.
