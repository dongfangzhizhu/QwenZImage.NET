#!/usr/bin/env bash
# Build the bundled reference `qzi` shared library and run the C ABI smoke test.
#
# The library lands in artifacts/native/<Platform>-<Arch>/, which is exactly
# where NativeTestLibrary (and NativeLibraryLoader) look for it. No GPU, model
# weights or network access are required.
#
# Optional environment variables:
#   QZI_BUILD_DIR      build tree            (default: native/build)
#   QZI_BUILD_TYPE     build configuration  (default: Release)
#   CMAKE_GENERATOR    e.g. "Ninja"         (default: CMake's own default)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${QZI_BUILD_DIR:-$root/native/build}"
build_type="${QZI_BUILD_TYPE:-Release}"

cmake_args=(-S "$root/native" -B "$build_dir"
            "-DCMAKE_BUILD_TYPE=$build_type"
            "-DQZI_BUILD_TESTS=ON")
if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
  cmake_args+=(-G "$CMAKE_GENERATOR")
fi

echo "==> configuring qzi ($build_type) in $build_dir"
cmake "${cmake_args[@]}"

echo "==> building qzi"
cmake --build "$build_dir" --config "$build_type" --parallel

echo "==> running C ABI smoke test"
ctest --test-dir "$build_dir" -C "$build_type" --output-on-failure

echo "==> native library output:"
find "$root/artifacts/native" -maxdepth 2 -type f \
     \( -name 'qzi.dll' -o -name 'libqzi.so' -o -name 'libqzi.dylib' \) -print
