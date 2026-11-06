# Compiler flags for the vendored QwenZImageNative C/CUDA engine.
#
# Included by native/CMakeLists.txt before it adds engine/ as a subdirectory,
# so the helper is visible both to the shared library target (which compiles
# the qzi_backend adapter) and to qzi_engine_core (which compiles the engine).
#
# The engine sources target POSIX. On Windows, qzen_platform.h is force-
# included into every translation unit and platform/win32/*.c supplies the
# missing calls. On Linux the engine uses the native POSIX surface directly.
# This mirrors the upstream build; see engine/UPSTREAM.md.

function(qzen_apply_platform_flags target)
    if(WIN32)
        set(QZEN_PLATFORM_HEADER
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../platform/win32/qzen_platform.h")
        get_filename_component(QZEN_PLATFORM_HEADER "${QZEN_PLATFORM_HEADER}" ABSOLUTE)

        target_compile_definitions(${target} PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _CRT_SECURE_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
            WIN32_LEAN_AND_MEAN
            NOMINMAX
            QZEN_WIN32=1)

        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:C>:/std:c11;/utf-8;/openmp:llvm;/W3;/FI${QZEN_PLATFORM_HEADER}>"
            "$<$<COMPILE_LANGUAGE:CXX>:/std:c++17;/utf-8;/openmp:llvm;/W3;/FI${QZEN_PLATFORM_HEADER}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O2;-Xcompiler=/utf-8,-Xcompiler=/openmp:llvm,-Xcompiler=/FI${QZEN_PLATFORM_HEADER}>")

        # qzen_gpu.cu exceeds the default COFF 64k section limit.
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>")
    else()
        # -fopenmp as a *compile option* alone is not enough: generator
        # expressions guarded by COMPILE_LANGUAGE are empty at link time, so
        # the shared object ends up with unresolved omp_get_thread_num /
        # GOMP_parallel symbols and ld refuses to link qzi_abi_smoke against
        # it. Carry the OpenMP runtime through as a real link dependency.
        find_package(OpenMP QUIET COMPONENTS C)
        if(OpenMP_C_FOUND)
            target_compile_options(${target} PRIVATE
                "$<$<COMPILE_LANGUAGE:C>:${OpenMP_C_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_C_FLAGS}>")
            target_link_libraries(${target} PUBLIC OpenMP::OpenMP_C)
        else()
            target_compile_options(${target} PRIVATE
                "$<$<COMPILE_LANGUAGE:C>:-fopenmp>"
                "$<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")
            target_link_options(${target} PRIVATE -fopenmp)
        endif()
        # No -licuuc: a link-time dependency on libicuuc.so.72 would make the
        # NuGet linux-x64 runtime unloadable on distros shipping .so.70/.so.74.
        # qzen_posix_unicode.c dlopen()s ICU instead, hence CMAKE_DL_LIBS.
        target_link_libraries(${target} PUBLIC
            m pthread stdc++ ${CMAKE_DL_LIBS})
    endif()
endfunction()
