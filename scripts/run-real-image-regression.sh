#!/usr/bin/env bash
# Run the real-image regression matrix against the Linux/WSL CUDA native build.
# Requires the model packs referenced by QwenZImage.config.json.
#
#   bash scripts/run-real-image-regression.sh --list
#   bash scripts/run-real-image-regression.sh --only zimage-nunchaku-r32-512
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="$root/tools/RealImageRegression/QwenZImage.RealImageRegression.csproj"
native="$root/artifacts/native/Linux-x86_64-cuda/libQwenZImageNative.so"
config="$root/QwenZImage.config.json"

if [[ ! -f "$native" ]]; then
  echo "CUDA native library not found: $native" >&2
  echo "Build it first with: bash scripts/build-native-cuda.sh" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then echo "config not found: $config" >&2; exit 1; fi

exec dotnet run --project "$project" -c Release -- --native "$native" --config "$config" "$@"
