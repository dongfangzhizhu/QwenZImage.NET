#ifndef QZEN_SAFETENSORS_H
#define QZEN_SAFETENSORS_H

#include "qzen.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    QZEN_DTYPE_UNKNOWN = 0,
    QZEN_DTYPE_BOOL,
    QZEN_DTYPE_I8,
    QZEN_DTYPE_U8,
    QZEN_DTYPE_I16,
    QZEN_DTYPE_U16,
    QZEN_DTYPE_F16,
    QZEN_DTYPE_BF16,
    QZEN_DTYPE_I32,
    QZEN_DTYPE_U32,
    QZEN_DTYPE_F32,
    QZEN_DTYPE_I64,
    QZEN_DTYPE_U64,
    QZEN_DTYPE_F64,
    /* 8-bit floats.  Nothing in this engine computes with them: they are named
     * so a header that contains one still parses, which is what lets a
     * Nunchaku svdq-fp4 pack be identified and refused by name instead of
     * failing to open with "tensor shape does not match byte offsets". */
    QZEN_DTYPE_F8_E4M3,
    QZEN_DTYPE_F8_E5M2
} qzen_dtype;

typedef struct {
    char *name;
    qzen_dtype dtype;
    int ndim;
    uint64_t shape[8];
    uint64_t data_begin;
    uint64_t data_end;
    uint64_t file_offset;
} qzen_st_tensor;

typedef struct {
    char *path;
    uint64_t file_size;
    uint64_t header_size;
    qzen_st_tensor *tensors;
    size_t tensor_count;
} qzen_st_header;

int qzen_st_read_header(const char *path, qzen_st_header *header,
                      char *error, size_t error_size);
void qzen_st_free_header(qzen_st_header *header);
const qzen_st_tensor *qzen_st_find(const qzen_st_header *header, const char *name);
uint64_t qzen_st_tensor_elements(const qzen_st_tensor *tensor);
int qzen_st_read_data(const qzen_st_header *header, const qzen_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size);
/* Read a tensor as F32 on the host, normalising checkpoint dtypes:
 * F32 is copied verbatim, F16 is widened half -> float, and an INT8 ConvRot
 * weight is dequantised with its per-row scale and un-rotated with the
 * 256-point regular Hadamard along the input dimension (scale is the
 * '<base>.weight_scale' sidecar and is required for I8). */
int qzen_st_read_f32(const qzen_st_header *header, const qzen_st_tensor *tensor,
                   const qzen_st_header *scale_header,
                   const qzen_st_tensor *scale, float **values,
                   char *error, size_t error_size);
/* Host-side ConvRot dequantiser telemetry (arithmetic only, no disk I/O). */
void qzen_st_reset_dequant_stats(void);
void qzen_st_dequant_stats(double *seconds, uint64_t *elements);
size_t qzen_dtype_size(qzen_dtype dtype);
const char *qzen_dtype_name(qzen_dtype dtype);

int qzen_st_inventory_dir(const char *directory, qzen_component_info *info,
                        char *error, size_t error_size);

#endif
