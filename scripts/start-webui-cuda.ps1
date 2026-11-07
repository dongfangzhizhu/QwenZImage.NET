# Start the WebUI against the REAL Qwen-Image 2.1 / Z-Image C/CUDA backend.
#
#   .\scripts\start-webui-cuda.ps1
#   .\scripts\start-webui-cuda.ps1 -Urls http://127.0.0.1:7865
#   .\scripts\start-webui-cuda.ps1 -NoBuild
#
# Why this script exists: `dotnet run` loads the bundled reference library,
# which emits deterministic noise. Building the CUDA engine does NOT change
# that by itself - the -cuda artefact lives in a different directory and must
# be selected explicitly. This script selects it, and sets
# QZI_REQUIRE_REAL_BACKEND=1 so the WebUI refuses to serve generation jobs
# (HTTP 503, ok=false in /api/health) if the noise backend is loaded anyway.
#
# It deliberately does NOT stage the CUDA library into
# artifacts/runtimes/<rid>/native: that directory is what `dotnet pack` puts
# into the NuGet package, and it is kept on the GPU-free reference build for
# CI and for the default package.
param(
  [string]$Urls = 'http://127.0.0.1:7860',
  [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root 'src/QwenZImage.NET.WebUI/QwenZImage.NET.WebUI.csproj'
$native = Join-Path $root 'artifacts/native/Windows-AMD64-cuda/QwenZImageNative.dll'
$config = Join-Path $root 'QwenZImage.config.json'

if (-not (Test-Path $native)) {
  throw "CUDA native library not found: $native`nBuild it first: .\scripts\build-native-cuda.bat"
}
if (-not (Test-Path $config)) {
  throw "Model pack config not found: $config"
}

# A reference build is ~100 KB; the CUDA fat binary is tens of MB. Catching a
# stale/copied reference file here gives a much clearer error than noise images.
$bytes = (Get-Item $native).Length
if ($bytes -lt 1MB) {
  throw "$native is only $bytes bytes - that is the reference build, not the CUDA engine. Rebuild with .\scripts\build-native-cuda.bat"
}

$env:QZI_NATIVE_LIBRARY = $native
$env:QZI_CONFIG = $config
$env:QZI_REQUIRE_REAL_BACKEND = '1'
$env:QZI_WEBUI_URLS = $Urls
if (-not $env:QZI_WEBUI_OUTPUT) {
  $env:QZI_WEBUI_OUTPUT = Join-Path $root 'artifacts/webui-cuda-output'
}

Write-Host "==> native : $native ($bytes bytes)"
Write-Host "==> config : $config"
Write-Host "==> output : $env:QZI_WEBUI_OUTPUT"
Write-Host "==> url    : $Urls"
Write-Host '==> QZI_REQUIRE_REAL_BACKEND=1 (the app will refuse to serve noise)'
Write-Host '    Verify after startup: /api/health must report reference_backend=false'

$runArgs = @('run', '--project', $project, '-c', 'Release', '--no-launch-profile')
if ($NoBuild) { $runArgs += '--no-build' }
& dotnet @runArgs
exit $LASTEXITCODE
