# Build BOTH native platforms on a Windows host, then stage them for NuGet.
#   1. QwenZImageNative.dll     - native Windows build (CMake / Visual Studio or MinGW)
#   2. libQwenZImageNative.so   - native Linux build via WSL (Debian with gcc/cmake/ninja)
# After this, `dotnet pack src/QwenZImage.NET/QwenZImage.NET.csproj` succeeds.
#
# On Linux hosts just run scripts/build-native.sh (which stages linux-x64) and
# obtain QwenZImageNative.dll from a Windows CI runner; CI does exactly that.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Write-Host '=== 1/3 Windows native (win-x64) ==='
& (Join-Path $PSScriptRoot 'build-native.ps1')

Write-Host '=== 2/3 Linux native (linux-x64) via WSL ==='
$linuxScriptWin = Join-Path $PSScriptRoot 'build-native-linux.sh'
$wslScript = $null
try {
  $wslScript = (& wsl.exe wslpath -u $linuxScriptWin 2>$null | Out-String).Trim()
} catch {
  Write-Warning 'WSL is unavailable; skipping the Linux build.'
}
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslScript)) {
  Write-Warning 'WSL is unavailable; skipping the Linux build.'
} else {
  & wsl.exe -e bash $wslScript
  if ($LASTEXITCODE -ne 0) { throw "Linux native build failed (exit $LASTEXITCODE)" }
}

Write-Host '=== 3/3 stage both RIDs for NuGet ==='
& (Join-Path $PSScriptRoot 'stage-native.ps1')

$win = Join-Path $root 'artifacts/runtimes/win-x64/native/QwenZImageNative.dll'
$linux = Join-Path $root 'artifacts/runtimes/linux-x64/native/libQwenZImageNative.so'
Write-Host "win-x64  : $(if (Test-Path $win) { 'OK' } else { 'MISSING' })  $win"
Write-Host "linux-x64: $(if (Test-Path $linux) { 'OK' } else { 'MISSING' })  $linux"
