cmake_minimum_required(VERSION 3.20)

project(qzi VERSION 0.1.0 LANGUAGES C DESCRIPTION "QwenZImage.NET native C ABI")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The native library bundled in this repository implements the ABI with the
# deterministic reference backend. A real CUDA engine plugs in through
# qzi_register_backend() at load time -- no rebuild of this library is needed.
# See docs/ENGINE_INTEGRATION.md and integration/h3_backend_template.c.
option(QZI_BUILD_TESTS "Build the native ABI smoke test" ON)

add_library(qzi SHARED
  src/qzi.c
  src/qzi_stub.c
  src/qzi_png.c
)

target_include_directories(qzi
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(qzi PRIVATE QZI_BUILDING_DLL)

if(WIN32)
  # MinGW would otherwise emit libqzi.dll; .NET probes qzi.dll.
  # CRITICAL_SECTION needs no extra library on Windows.
  set_target_properties(qzi PROPERTIES PREFIX "" OUTPUT_NAME "qzi")
  if(NOT MSVC)
    # Only QZI_API (__declspec(dllexport)) symbols belong in the export table.
    target_link_options(qzi PRIVATE -Wl,--exclude-all-symbols)
  endif()
else()
  set_target_properties(qzi PROPERTIES OUTPUT_NAME "qzi")  # -> libqzi.so
  find_package(Threads REQUIRED)
  target_link_libraries(qzi PRIVATE Threads::Threads)
endif()

# Keep warnings visible on every toolchain, but do not fail the build on MSVC:
# the CI Windows runner uses the Visual Studio generator and MSVC /W4 emits a
# few benign diagnostics that differ from GCC/Clang. GCC/Clang use the same
# "warn, don't fail" policy below, so all three compilers behave consistently.
if(MSVC)
  target_compile_options(qzi PRIVATE /W4)
else()
  target_compile_options(qzi PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Symbol hygiene: only qzi_* / QZI_API functions leave the shared object.
# (Visibility is already hidden by default; QZI_API marks the exports.)

if(QZI_BUILD_TESTS)
  add_executable(qzi_abi_smoke tests/qzi_abi_smoke.c)
  target_include_directories(qzi_abi_smoke PRIVATE include)
  target_link_libraries(qzi_abi_smoke PRIVATE qzi)
  add_custom_command(TARGET qzi_abi_smoke POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:qzi> $<TARGET_FILE_DIR:qzi_abi_smoke>
    COMMENT "Placing qzi.dll next to qzi_abi_smoke.exe")
  enable_testing()
  add_test(NAME qzi_abi_smoke COMMAND qzi_abi_smoke)
endif()

# Convenience: put binaries in artifacts/native/<platform> so the .NET test
# host can find them through QZI_NATIVE_LIBRARY without installing anything.
set(QZI_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts/native/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set_target_properties(qzi PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${QZI_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${QZI_OUTPUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${QZI_OUTPUT_DIR}"
)
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(qzi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${cfg} "${QZI_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${cfg} "${QZI_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${cfg} "${QZI_OUTPUT_DIR}"
  )
endforeach()
