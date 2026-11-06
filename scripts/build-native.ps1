# Build the bundled reference `qzi` shared library and run the C ABI smoke test.
# Windows counterpart of build-native.sh. Uses CMake's default generator
# (Visual Studio on GitHub runners), so no Ninja/MSYS setup is required.
#
# Optional environment variables: QZI_BUILD_DIR, QZI_BUILD_TYPE, CMAKE_GENERATOR.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$buildDir = if ($env:QZI_BUILD_DIR) { $env:QZI_BUILD_DIR } else { Join-Path $root 'native/build' }
$buildType = if ($env:QZI_BUILD_TYPE) { $env:QZI_BUILD_TYPE } else { 'Release' }

$cmakeArgs = @(
  '-S', (Join-Path $root 'native'),
  '-B', $buildDir,
  "-DCMAKE_BUILD_TYPE=$buildType",
  '-DQZI_BUILD_TESTS=ON'
)
if ($env:CMAKE_GENERATOR) { $cmakeArgs += @('-G', $env:CMAKE_GENERATOR) }

Write-Host "==> configuring qzi ($buildType) in $buildDir"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }

Write-Host '==> building qzi'
& cmake --build $buildDir --config $buildType --parallel
if ($LASTEXITCODE -ne 0) { throw "cmake build failed with exit code $LASTEXITCODE" }

Write-Host '==> running C ABI smoke test'
& ctest --test-dir $buildDir -C $buildType --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "ctest failed with exit code $LASTEXITCODE" }

Write-Host '==> native library output:'
Get-ChildItem -Path (Join-Path $root 'artifacts/native') -Recurse -File |
  Where-Object { $_.Name -in @('qzi.dll', 'libqzi.so', 'libqzi.dylib') } |
  ForEach-Object { $_.FullName }

Write-Host '==> staging for NuGet'
& (Join-Path $PSScriptRoot 'stage-native.ps1')
