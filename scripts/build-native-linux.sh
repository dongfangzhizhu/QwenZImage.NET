#!/usr/bin/env bash
# Build libqzi.so for linux-x64. Run this on Linux OR inside WSL from Windows:
#   wsl.exe -e bash /mnt/d/AI/convert/QwenZImage.NET/scripts/build-native-linux.sh
#
# Output: artifacts/native/Linux-<arch>/libqzi.so, then staged into
#         artifacts/runtimes/linux-x64/native/libqzi.so for `dotnet pack`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${QZI_BUILD_DIR:-$root/native/build-linux}"
build_type="${QZI_BUILD_TYPE:-Release}"
generator="${CMAKE_GENERATOR:-Ninja}"

echo "==> configuring qzi for Linux ($build_type) in $build_dir"
cmake -S "$root/native" -B "$build_dir" -G "$generator" \
      -DCMAKE_BUILD_TYPE="$build_type" -DQZI_BUILD_TESTS=ON

echo "==> building libqzi.so"
cmake --build "$build_dir" --config "$build_type" --parallel

echo "==> running C ABI smoke test on Linux"
ctest --test-dir "$build_dir" -C "$build_type" --output-on-failure

echo "==> staging linux-x64 runtime"
bash "$root/scripts/stage-native.sh"
