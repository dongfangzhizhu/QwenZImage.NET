#!/usr/bin/env bash
# Collect built qzi shared libraries into the NuGet RID staging layout:
#   artifacts/runtimes/win-x64/native/QwenZImageNative.dll
#   artifacts/runtimes/linux-x64/native/libQwenZImageNative.so
#
# Sources are the per-platform CMake outputs in artifacts/native/:
#   <Platform>-<Arch>/        deterministic reference backend (default, CI/tests)
#   <Platform>-<Arch>-cuda/   real Qwen-Image 2.1 / Z-Image C/CUDA backend
# By default only the reference tree is staged so the managed test suite never
# accidentally loads a GPU build. Set QZI_STAGE_CUDA=1 to stage the real one.
# Any platform that has not been built yet is skipped (never faked), so this can
# run on Windows, Linux, WSL or macOS. Run both platforms before `dotnet pack`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_root="$root/artifacts/native"
dst_root="$root/artifacts/runtimes"
want_cuda="${QZI_STAGE_CUDA:-0}"

stage() {
  local name="$1" rid="$2" src dst
  if [[ "$want_cuda" == "1" ]]; then
    src="$(find "$src_root" -type f -path '*/-cuda/*' -name "$name" -print -quit 2>/dev/null || true)"
  else
    src="$(find "$src_root" -type f -name "$name" ! -path '*/-cuda/*' -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "$src" ]]; then
    echo "  skip  $name (not built yet)"
    return 0
  fi
  dst="$dst_root/$rid/native/$name"
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
  echo "  stage $dst  <-  $src"
}

echo "==> staging native runtimes into $dst_root"
stage QwenZImageNative.dll      win-x64
stage libQwenZImageNative.so    linux-x64
