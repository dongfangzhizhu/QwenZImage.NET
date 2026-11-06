#include "qzen_device.h"

#ifdef __APPLE__
#include "qzen_metal.h"
#else
#include "qzen_cuda.h"
#endif

int qzen_device_probe(qzen_device_info *info, char *error, size_t error_size) {
#ifdef __APPLE__
    return qzen_metal_probe(info, error, error_size);
#else
    return qzen_cuda_probe(info, error, error_size);
#endif
}

int qzen_device_arch_state(const qzen_device_info *info) {
    if (!info || info->build_arch <= 0 || info->apple_gpu_family <= 0) return -1;
    return info->build_arch == info->apple_gpu_family ? 1 : 0;
}
