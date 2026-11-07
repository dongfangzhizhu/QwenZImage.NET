#!/usr/bin/env bash
# Start the WebUI against the REAL Qwen-Image 2.1 / Z-Image C/CUDA backend.
#
#   bash scripts/start-webui-cuda.sh
#   QZI_WEBUI_URLS=http://127.0.0.1:7865 bash scripts/start-webui-cuda.sh
#
# `dotnet run` loads the bundled reference library, which emits deterministic
# noise. Building the CUDA engine does NOT change that by itself - the -cuda
# artefact lives in a different directory and must be selected explicitly.
# This script selects it and sets QZI_REQUIRE_REAL_BACKEND=1 so the WebUI
# refuses to serve generation jobs (HTTP 503, ok=false in /api/health) if the
# noise backend is loaded anyway.
#
# It deliberately does NOT stage the CUDA library into
# artifacts/runtimes/<rid>/native: that directory is what `dotnet pack` puts
# into the NuGet package, and it stays on the GPU-free reference build for CI.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="$root/src/QwenZImage.NET.WebUI/QwenZImage.NET.WebUI.csproj"
native="$root/artifacts/native/Linux-x86_64-cuda/libQwenZImageNative.so"
config="$root/QwenZImage.config.json"

if [[ ! -f "$native" ]]; then
  echo "CUDA native library not found: $native" >&2
  echo "Build it first: bash scripts/build-native-cuda.sh" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "Model pack config not found: $config" >&2
  exit 1
fi

# A reference build is ~40 KB; the CUDA fat binary is tens of MB.
bytes="$(wc -c < "$native")"
if (( bytes < 1048576 )); then
  echo "$native is only $bytes bytes - that is the reference build, not the CUDA engine." >&2
  echo "Rebuild with: bash scripts/build-native-cuda.sh" >&2
  exit 1
fi

export QZI_NATIVE_LIBRARY="$native"
export QZI_CONFIG="$config"
export QZI_REQUIRE_REAL_BACKEND=1
export QZI_WEBUI_URLS="${QZI_WEBUI_URLS:-http://127.0.0.1:7860}"
export QZI_WEBUI_OUTPUT="${QZI_WEBUI_OUTPUT:-$root/artifacts/webui-cuda-output}"

echo "==> native : $native ($bytes bytes)"
echo "==> config : $config"
echo "==> output : $QZI_WEBUI_OUTPUT"
echo "==> url    : $QZI_WEBUI_URLS"
echo '==> QZI_REQUIRE_REAL_BACKEND=1 (the app will refuse to serve noise)'
echo '    Verify after startup: /api/health must report reference_backend=false'

exec dotnet run --project "$project" -c Release --no-launch-profile -- "$@"
