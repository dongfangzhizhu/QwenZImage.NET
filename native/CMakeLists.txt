cmake_minimum_required(VERSION 3.20)

project(QwenZImageNative VERSION 0.1.0 LANGUAGES C
        DESCRIPTION "QwenZImage.NET native backend (qzi C ABI)")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The exported library is always QwenZImageNative.dll / libQwenZImageNative.so,
# independent of which backend is linked in. The deterministic reference
# backend needs nothing but a C compiler; the real C/CUDA engine is added by
# -DQZI_WITH_CUDA_ENGINE=ON (see ../engine/CMakeLists.txt).
option(QZI_BUILD_TESTS "Build the native ABI smoke test" ON)
option(QZI_WITH_CUDA_ENGINE
       "Link the real Qwen-Image 2.1 / Z-Image C/CUDA engine (needs nvcc)" OFF)

# Languages must be enabled in the top-level directory scope so the targets
# in engine/ can compile .cu sources. Reference-only builds stay pure C.
if(QZI_WITH_CUDA_ENGINE)
  include(../engine/cmake/QzenPlatform.cmake)
  enable_language(CXX)
  enable_language(CUDA)
endif()

add_library(QwenZImageNative SHARED
  src/qzi.c
  src/qzi_png.c
)

if(QZI_WITH_CUDA_ENGINE)
  add_subdirectory(../engine ${CMAKE_CURRENT_BINARY_DIR}/engine)
  target_sources(QwenZImageNative PRIVATE src/qzi_engine_backend.c)
  target_link_libraries(QwenZImageNative PRIVATE qzi_engine_core)
  target_compile_definitions(QwenZImageNative PRIVATE QZI_HAS_REAL_BACKEND)
  # The adapter compiles against the engine's POSIX compatibility surface.
  qzen_apply_platform_flags(QwenZImageNative)
  if(WIN32)
    target_link_options(QwenZImageNative PRIVATE "/STACK:8388608")
  endif()
else()
  target_sources(QwenZImageNative PRIVATE src/qzi_stub.c)
endif()

target_include_directories(QwenZImageNative
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(QwenZImageNative PRIVATE QZI_BUILDING_DLL)

if(WIN32)
  # MinGW would otherwise emit libQwenZImageNative.dll; .NET probes
  # QwenZImageNative.dll. CRITICAL_SECTION needs no extra library on Windows.
  set_target_properties(QwenZImageNative PROPERTIES
    PREFIX "" OUTPUT_NAME "QwenZImageNative")
  if(NOT MSVC)
    # Only QZI_API (__declspec(dllexport)) symbols belong in the export table.
    target_link_options(QwenZImageNative PRIVATE -Wl,--exclude-all-symbols)
  endif()
else()
  set_target_properties(QwenZImageNative PROPERTIES
    OUTPUT_NAME "QwenZImageNative")  # -> libQwenZImageNative.so
  find_package(Threads REQUIRED)
  target_link_libraries(QwenZImageNative PRIVATE Threads::Threads)
endif()

# Keep warnings visible on every toolchain, but do not fail the build on MSVC:
# the CI Windows runner uses the Visual Studio generator and MSVC /W4 emits a
# few benign diagnostics that differ from GCC/Clang. GCC/Clang use the same
# "warn, don't fail" policy below, so all three compilers behave consistently.
# With the real engine, qzen_apply_platform_flags() already supplies /W3 (or
# -fopenmp on Linux); adding /W4 on top would trigger cl's D9025 override
# warning for every adapter translation unit.
if(NOT QZI_WITH_CUDA_ENGINE)
  if(MSVC)
    target_compile_options(QwenZImageNative PRIVATE /W4)
  else()
    target_compile_options(QwenZImageNative PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# Symbol hygiene: only qzi_* / QZI_API functions leave the shared object.
# (Visibility is already hidden by default; QZI_API marks the exports.)

if(QZI_BUILD_TESTS)
  add_executable(qzi_abi_smoke tests/qzi_abi_smoke.c)
  target_include_directories(qzi_abi_smoke PRIVATE include)
  target_link_libraries(qzi_abi_smoke PRIVATE QwenZImageNative)
  add_custom_command(TARGET qzi_abi_smoke POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:QwenZImageNative> $<TARGET_FILE_DIR:qzi_abi_smoke>
    COMMENT "Placing QwenZImageNative next to qzi_abi_smoke")
  enable_testing()
  add_test(NAME qzi_abi_smoke COMMAND qzi_abi_smoke)

  # POSIX ICU compatibility layer. It resolves ICU exactly once per process
  # (pthread_once), so the two runtime paths need separate executables: one
  # that dlopen()s the host ICU and one where GNU ld's --wrap=dlopen forces
  # every open to fail. This is independent of the CUDA engine, so GPU-less
  # Linux CI runs it on the reference build too.
  if(NOT WIN32)
    include(CheckLinkerFlag)
    check_linker_flag(C "-Wl,--wrap=dlopen" QZI_HAVE_WRAP_DLOPEN)

    set(QZI_POSIX_UNICODE_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/platform/posix/qzen_posix_unicode.c")
    set(QZI_POSIX_UNICODE_INC
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/platform/posix/include")
    find_package(Threads REQUIRED)

    add_executable(qzi_posix_unicode_icu
        tests/qzi_posix_unicode_check.c ${QZI_POSIX_UNICODE_SRC})
    target_include_directories(qzi_posix_unicode_icu PRIVATE
        ${QZI_POSIX_UNICODE_INC})
    target_link_libraries(qzi_posix_unicode_icu PRIVATE
        Threads::Threads ${CMAKE_DL_LIBS})
    add_test(NAME posix_unicode_with_icu COMMAND qzi_posix_unicode_icu)
    # Exit 77 means "no usable ICU on this host"; the shared assertions still
    # ran and passed, so report it as a skip instead of a failure.
    set_tests_properties(posix_unicode_with_icu PROPERTIES
        SKIP_RETURN_CODE 77)

    if(QZI_HAVE_WRAP_DLOPEN)
      add_executable(qzi_posix_unicode_fallback
          tests/qzi_posix_unicode_check.c ${QZI_POSIX_UNICODE_SRC})
      target_compile_definitions(qzi_posix_unicode_fallback PRIVATE
          QZEN_TEST_NO_ICU)
      target_include_directories(qzi_posix_unicode_fallback PRIVATE
          ${QZI_POSIX_UNICODE_INC})
      target_link_libraries(qzi_posix_unicode_fallback PRIVATE
          Threads::Threads ${CMAKE_DL_LIBS})
      target_link_options(qzi_posix_unicode_fallback PRIVATE
          "-Wl,--wrap=dlopen")
      add_test(NAME posix_unicode_without_icu
          COMMAND qzi_posix_unicode_fallback)
    else()
      message(STATUS
        "QwenZImageNative: linker has no --wrap=dlopen; skipping the no-ICU CTest variant")
    endif()
  endif()
endif()

# Convenience: put binaries in artifacts/native/<platform> so the .NET test
# host can find them through QZI_NATIVE_LIBRARY without installing anything.
# The real C/CUDA build goes into a sibling "-cuda" directory: it must never
# silently replace the deterministic reference DLL that the managed test suite
# (and GPU-less CI) loads. stage-native.* picks one tree explicitly.
set(QZI_FLAVOR "")
if(QZI_WITH_CUDA_ENGINE)
  set(QZI_FLAVOR "-cuda")
endif()
set(QZI_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/native/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}${QZI_FLAVOR}")
set_target_properties(QwenZImageNative PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${QZI_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${QZI_OUTPUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${QZI_OUTPUT_DIR}"
)
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(QwenZImageNative PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${cfg} "${QZI_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${cfg} "${QZI_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${cfg} "${QZI_OUTPUT_DIR}"
  )
endforeach()
