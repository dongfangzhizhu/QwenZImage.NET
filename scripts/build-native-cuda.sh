#!/usr/bin/env bash
# Build libQwenZImageNative.so with the REAL Qwen-Image 2.1 / Z-Image C/CUDA
# engine (engine/), instead of the deterministic reference backend.
#
# Usage:  scripts/build-native-cuda.sh [cuda-sm-arch-list] [build-dir]
#   e.g.  scripts/build-native-cuda.sh 89
#         scripts/build-native-cuda.sh "80;86;89;90;120;121"
#
# The default is a fat binary covering Ampere -> Blackwell so the resulting
# NuGet runtime works on a consumer's GPU without a rebuild. A single SM is
# much faster to compile for local development.
#
# Requires: CUDA toolkit (nvcc), a host C/C++ compiler supported by that CUDA
# version, CMake >= 3.20, Ninja (optional) and OpenMP. Model weights are NOT
# needed to build.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
arch="${1:-${QZEN_CUDA_ARCH:-80;86;87;88;89;90;100;103;110;120;121}}"
build_dir="${2:-${QZI_BUILD_DIR:-$root/native/build-cuda}}"
build_type="${QZI_BUILD_TYPE:-Release}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[QwenZImageNative] nvcc not found; install the CUDA toolkit." >&2
  exit 1
fi

cmake_args=(-S "$root/native" -B "$build_dir"
            "-DCMAKE_BUILD_TYPE=$build_type"
            "-DQZI_WITH_CUDA_ENGINE=ON"
            "-DQZEN_CUDA_ARCH=$arch"
            "-DQZI_BUILD_TESTS=ON")
if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
  cmake_args+=(-G "$CMAKE_GENERATOR")
fi

# A previous configure with a different generator/toolchain leaves a cache
# that CMake refuses to reuse. Drop just the cache, keep the object files.
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
  echo "==> clearing a stale CMake cache in $build_dir"
  rm -f "$build_dir/CMakeCache.txt"
  rm -rf "$build_dir/CMakeFiles"
fi

echo "==> configuring real C/CUDA backend (SM $arch) in $build_dir"
cmake "${cmake_args[@]}"

echo "==> building libQwenZImageNative.so"
cmake --build "$build_dir" --config "$build_type" --parallel

echo "==> running the C ABI smoke test"
ctest --test-dir "$build_dir" -C "$build_type" --output-on-failure

echo "==> staging linux-x64 native for NuGet"
QZI_STAGE_CUDA=1 bash "$root/scripts/stage-native.sh"

echo "==> done: artifacts/runtimes/linux-x64/native/libQwenZImageNative.so"
