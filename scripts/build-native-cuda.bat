@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ---------------------------------------------------------------------------
rem Build QwenZImageNative.dll with the REAL Qwen-Image 2.1 / Z-Image C/CUDA
rem engine (engine/), instead of the deterministic reference backend.
rem
rem Usage:  scripts\build-native-cuda.bat [cuda-sm-arch-list] [build-dir]
rem   e.g.  scripts\build-native-cuda.bat 89
rem         scripts\build-native-cuda.bat "80;86;89;90;120;121" D:\temp\qzi-cuda
rem
rem The default is a fat binary covering Ampere -^> Blackwell so the resulting
rem NuGet runtime works on a consumer's GPU without a rebuild. A single SM is
rem much faster to compile for local development.
rem
rem Requires: CUDA toolkit (nvcc), a MSVC toolchain supported by that CUDA
rem version, CMake >= 3.20 and Ninja. Model weights are NOT needed to build.
rem
rem Everything runs inside this one cmd process because vcvars64.bat sets
rem INCLUDE/LIB/PATH in the process environment; CMake and nvcc must inherit
rem exactly those values. If the caller already has cl.exe + rc.exe + nvcc.exe
rem on PATH (e.g. a VS developer prompt or a pinned CI toolchain), the script
rem reuses that environment and skips vcvars64.bat entirely.
rem ---------------------------------------------------------------------------

set "ROOT=%~dp0.."
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=%QZEN_CUDA_ARCH%"
if "%ARCH%"=="" set "ARCH=80;86;87;88;89;90;100;103;110;120;121"
set "BUILD_DIR=%~2"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%ROOT%\native\build-cuda"

rem If the caller already configured a complete MSVC + CUDA environment, use
rem it as-is. This avoids vcvars64.bat's registry lookups (blocked in some
rem sandboxed/CI shells) and lets CI pin an exact MSVC/SDK/nvcc combination.
where cl >nul 2>&1
if errorlevel 1 goto :need_vcvars
where rc >nul 2>&1
if errorlevel 1 goto :need_vcvars
where nvcc >nul 2>&1
if errorlevel 1 goto :need_vcvars
echo ==^> using the pre-configured MSVC + CUDA environment
goto :env_ready

:need_vcvars
rem Prefer the toolchain that matches the CUDA toolkit's supported MSVC range.
set "VCVARS="
for %%P in (
  "%ProgramFiles%\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
  if not defined VCVARS if exist %%~P set "VCVARS=%%~P"
)

if not defined VCVARS if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
  )
)

if not defined VCVARS (
  echo [QwenZImageNative] Cannot find vcvars64.bat. Install the Visual Studio C++ toolchain. 1>&2
  exit /b 1
)

echo ==^> MSVC environment: !VCVARS!
call "!VCVARS!"
if errorlevel 1 (
  echo [QwenZImageNative] vcvars64.bat failed. 1>&2
  exit /b 1
)
where cl >nul 2>&1
if errorlevel 1 (
  echo [QwenZImageNative] cl.exe is not on PATH after vcvars64.bat. 1>&2
  exit /b 1
)
where nvcc >nul 2>&1
if errorlevel 1 (
  echo [QwenZImageNative] nvcc.exe is not on PATH; install the CUDA toolkit. 1>&2
  exit /b 1
)

rem vcvars64.bat locates the Windows SDK through the registry. If that lookup
rem is unavailable, rc.exe/mt.exe and the SDK headers/libs stay out of the
rem environment and CMake's MSVC compiler test fails with "rc ... no such file".
rem Repair the environment from the filesystem instead (no registry access).
where rc >nul 2>&1
if not errorlevel 1 goto :env_ready
set "QZI_KITS=!ProgramFiles(x86)!\Windows Kits\10"
set "QZI_SDKVER="
for /f "tokens=*" %%V in ('dir /b /on "!QZI_KITS!\Include" 2^>nul') do (
  if exist "!QZI_KITS!\bin\%%V\x64\rc.exe" set "QZI_SDKVER=%%V"
)
if not defined QZI_SDKVER (
  echo [QwenZImageNative] Cannot find a Windows SDK with rc.exe under !QZI_KITS!. 1>&2
  exit /b 1
)
echo ==^> adding Windows SDK !QZI_SDKVER! from the filesystem
set "PATH=!QZI_KITS!\bin\!QZI_SDKVER!\x64;!PATH!"
set "INCLUDE=!QZI_KITS!\Include\!QZI_SDKVER!\ucrt;!QZI_KITS!\Include\!QZI_SDKVER!\shared;!QZI_KITS!\Include\!QZI_SDKVER!\um;!QZI_KITS!\Include\!QZI_SDKVER!\winrt;!INCLUDE!"
set "LIB=!QZI_KITS!\Lib\!QZI_SDKVER!\ucrt\x64;!QZI_KITS!\Lib\!QZI_SDKVER!\um\x64;!LIB!"
set "LIBPATH=!LIB!;!LIBPATH!"
where rc >nul 2>&1
if errorlevel 1 (
  echo [QwenZImageNative] rc.exe still not on PATH after the SDK fallback. 1>&2
  exit /b 1
)

:env_ready
rem A previous configure with a different generator/toolchain leaves a cache
rem that CMake refuses to reuse. Drop just the cache, keep the object files.
if exist "!BUILD_DIR!\CMakeCache.txt" (
  echo ==^> clearing a stale CMake cache in !BUILD_DIR!
  del /q "!BUILD_DIR!\CMakeCache.txt"
  if exist "!BUILD_DIR!\CMakeFiles" rmdir /s /q "!BUILD_DIR!\CMakeFiles"
)

echo ==^> configuring real C/CUDA backend ^(SM !ARCH!^) in !BUILD_DIR!
cmake -S "!ROOT!\native" -B "!BUILD_DIR!" -G Ninja ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DQZI_WITH_CUDA_ENGINE=ON ^
      -DQZEN_CUDA_ARCH=!ARCH! ^
      -DQZI_BUILD_TESTS=ON
if errorlevel 1 exit /b 1

echo ==^> building QwenZImageNative.dll
cmake --build "!BUILD_DIR!" --parallel
if errorlevel 1 exit /b 1

echo ==^> running the C ABI smoke test
ctest --test-dir "!BUILD_DIR!" --output-on-failure
if errorlevel 1 exit /b 1

echo ==^> staging win-x64 native for NuGet
set "QZI_STAGE_CUDA=1"
powershell -NoProfile -ExecutionPolicy Bypass -File "!ROOT!\scripts\stage-native.ps1"
if errorlevel 1 exit /b 1

echo ==^> done: artifacts/runtimes/win-x64/native/QwenZImageNative.dll
exit /b 0
