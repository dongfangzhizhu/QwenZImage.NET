# Windows counterpart of stage-native.sh. Copies every built native library
# found under artifacts/native/ into the NuGet RID staging layout:
#   artifacts/runtimes/win-x64/native/QwenZImageNative.dll
#   artifacts/runtimes/linux-x64/native/libQwenZImageNative.so
# Missing platforms are skipped; run scripts/build-native-all.ps1 to build both.
#
# Two build flavours live in separate trees:
#   <Platform>-<Arch>/        deterministic reference backend (default, CI/tests)
#   <Platform>-<Arch>-cuda/   real Qwen-Image 2.1 / Z-Image C/CUDA backend
# By default only the reference tree is staged so the managed test suite never
# accidentally loads a GPU build. Set QZI_STAGE_CUDA=1 to stage the real one.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$srcRoot = Join-Path $root 'artifacts/native'
$dstRoot = Join-Path $root 'artifacts/runtimes'
$wantCuda = $env:QZI_STAGE_CUDA -eq '1'

$map = @(
  @{ Name = 'QwenZImageNative.dll';   Rid = 'win-x64' },
  @{ Name = 'libQwenZImageNative.so'; Rid = 'linux-x64' }
)

Write-Host "==> staging native runtimes into $dstRoot (flavour: $(if ($wantCuda) { 'cuda' } else { 'reference' }))"
foreach ($item in $map) {
  $src = $null
  if (Test-Path $srcRoot) {
    $src = Get-ChildItem -Path $srcRoot -Recurse -File -Filter $item.Name |
           Where-Object { (($_.FullName -match '-cuda[\\/]') -eq $wantCuda) } |
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
