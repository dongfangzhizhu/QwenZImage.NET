#!/usr/bin/env bash
# Collect built qzi shared libraries into the NuGet RID staging layout:
#   artifacts/runtimes/win-x64/native/qzi.dll
#   artifacts/runtimes/linux-x64/native/libqzi.so
#
# Sources are the per-platform CMake outputs in artifacts/native/<Platform>-<Arch>/.
# Any platform that has not been built yet is skipped (never faked), so this can
# run on Windows, Linux, WSL or macOS. Run both platforms before `dotnet pack`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_root="$root/artifacts/native"
dst_root="$root/artifacts/runtimes"

stage() {
  local name="$1" rid="$2" src dst
  src="$(find "$src_root" -type f -name "$name" -print -quit 2>/dev/null || true)"
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
stage qzi.dll      win-x64
stage libqzi.so    linux-x64
