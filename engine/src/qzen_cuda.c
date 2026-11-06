#include "qzen_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* The arch this cubin was built for, injected by CMake and by
 * Makefile.linux.  A hand-rolled build that passes neither leaves it at 0,
 * and then --info says "not recorded" instead of guessing. */
#ifndef QZEN_SM_ARCH
#define QZEN_SM_ARCH 0
#endif

int qzen_cuda_probe(qzen_device_info *info, char *error, size_t error_size) {
    if (!info) return 0;
    memset(info, 0, sizeof(*info));

    int device = 0;
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cudaSetDevice failed: %s",
                     cudaGetErrorString(status));
        }
        return 0;
    }

    struct cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device);
    if (status != cudaSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cudaGetDeviceProperties failed: %s",
                     cudaGetErrorString(status));
        }
        return 0;
    }

    snprintf(info->name, sizeof(info->name), "%s", props.name);
    snprintf(info->architecture, sizeof(info->architecture), "sm_%d%d",
             props.major, props.minor);
    info->physical_memory = (uint64_t)sysconf(_SC_PHYS_PAGES) *
                            (uint64_t)sysconf(_SC_PAGESIZE);
    info->recommended_working_set = props.totalGlobalMem;
    info->max_buffer_length = props.totalGlobalMem;
    info->apple_gpu_family = (int)(props.major * 10 + props.minor);
    info->metal4 = props.major >= 12 ? 1 : 0;
    info->unified_memory = props.unifiedAddressing ? 1 : 0;
    info->build_arch = QZEN_SM_ARCH;
    return 1;
}
