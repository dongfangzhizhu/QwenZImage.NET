# Real Qwen-Image 2.1 / Z-Image inference engine, vendored and renamed.
#
# Added via add_subdirectory() from ../native/CMakeLists.txt only when
# -DQZI_WITH_CUDA_ENGINE=ON. The resulting objects are linked into the single
# QwenZImageNative shared library together with the qzi C ABI control layer;
# there is no separate engine DLL and no HTTP hop.
#
# A CUDA toolkit is required (nvcc + cudart/cublas/cublasLt). Hosts without
# CUDA build the deterministic reference backend instead (the default), which
# keeps CI and the whole managed test suite runnable without a GPU.

set(QZEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# Ada / RTX 4090 by default. Override with -DQZEN_CUDA_ARCH=121 for GB10.
set(QZEN_CUDA_ARCH "89" CACHE STRING "CUDA compute capability to emit")
set(CMAKE_CUDA_ARCHITECTURES "${QZEN_CUDA_ARCH}")

set(QZEN_C_SOURCES
    src/qzen_safetensors.c
    src/qzen_weights.c
    src/qzen_quanto.c
    src/qzen_nunchaku.c
    src/qzen_tokenizer.c
    src/qzen_cuda.c
    src/qzen_device.c
    src/qzen_gpu_stubs.c
    src/qzen_qwen21_dit.c
    src/qzen_qwen21_text.c
    src/qzen_qwen21_vae.c
    src/qzen_qwen21_pipeline.c
    src/qzen_zimage_dit.c
    src/qzen_zimage_text.c
    src/qzen_zimage_vae.c
    src/qzen_zimage_pipeline.c
    src/qzen_config.c
    third_party/cJSON.c)

set(QZEN_CUDA_SOURCES src/qzen_gpu.cu)

if(WIN32)
    list(APPEND QZEN_C_SOURCES
        platform/win32/qzen_win_posix.c
        platform/win32/qzen_win_socket.c
        platform/win32/qzen_win_thread.c
        platform/win32/qzen_win_dir.c
        platform/win32/qzen_win_unicode.c)
endif()

# An OBJECT library rather than a static one: the device code in qzen_gpu.cu
# must take part in the final device-link step of the shared library.
add_library(qzi_engine_core OBJECT ${QZEN_C_SOURCES} ${QZEN_CUDA_SOURCES})

target_include_directories(qzi_engine_core PUBLIC
    "${QZEN_ROOT}/src"
    "${QZEN_ROOT}"
    "${QZEN_ROOT}/third_party"
    "${QZEN_ROOT}/platform/win32"
    "${QZEN_ROOT}/platform/win32/include")

target_compile_definitions(qzi_engine_core PUBLIC QZEN_SM_ARCH=${QZEN_CUDA_ARCH})
qzen_apply_platform_flags(qzi_engine_core)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(qzi_engine_core PUBLIC
    CUDA::cudart CUDA::cublas CUDA::cublasLt)
if(WIN32)
    target_link_libraries(qzi_engine_core PUBLIC ws2_32)
endif()

# nvcc needs the force-included compatibility header on its host pass too.
target_compile_options(qzi_engine_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")

set_target_properties(qzi_engine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
