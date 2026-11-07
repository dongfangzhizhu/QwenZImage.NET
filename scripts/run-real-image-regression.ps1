# Run the real-image regression matrix against the Windows CUDA native build.
# Requires the model packs referenced by QwenZImage.config.json.
#
#   .\scripts\run-real-image-regression.ps1
#   .\scripts\run-real-image-regression.ps1 --list
#   .\scripts\run-real-image-regression.ps1 --only qwen21-turbo-512 --out artifacts\regress\smoke
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root 'tools/RealImageRegression/QwenZImage.RealImageRegression.csproj'
$native = Join-Path $root 'artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll'
$config = Join-Path $root 'QwenZImage.config.json'

if (-not (Test-Path $native)) {
  throw "CUDA native library not found: $native`nBuild it first with .\scripts\build-native-cuda.bat"
}
if (-not (Test-Path $config)) { throw "config not found: $config" }

& dotnet run --project $project -c Release -- --native $native --config $config @args
exit $LASTEXITCODE
