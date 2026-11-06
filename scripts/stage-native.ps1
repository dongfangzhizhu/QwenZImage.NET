# Windows counterpart of stage-native.sh. Copies every built qzi library found
# under artifacts/native/ into the NuGet RID staging layout:
#   artifacts/runtimes/win-x64/native/qzi.dll
#   artifacts/runtimes/linux-x64/native/libqzi.so
# Missing platforms are skipped; run scripts/build-native-all.ps1 to build both.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$srcRoot = Join-Path $root 'artifacts/native'
$dstRoot = Join-Path $root 'artifacts/runtimes'

$map = @(
  @{ Name = 'qzi.dll';   Rid = 'win-x64' },
  @{ Name = 'libqzi.so'; Rid = 'linux-x64' }
)

Write-Host "==> staging native runtimes into $dstRoot"
foreach ($item in $map) {
  $src = $null
  if (Test-Path $srcRoot) {
    $src = Get-ChildItem -Path $srcRoot -Recurse -File -Filter $item.Name |
           Select-Object -First 1
  }
  if (-not $src) {
    Write-Host "  skip  $($item.Name) (not built yet)"
    continue
  }
  $dstDir = Join-Path $dstRoot "$($item.Rid)/native"
  New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
  $dst = Join-Path $dstDir $item.Name
  Copy-Item -Path $src.FullName -Destination $dst -Force
  Write-Host "  stage $dst  <-  $($src.FullName)"
}
