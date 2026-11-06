#ifndef QZEN_QUANTO_H
#define QZEN_QUANTO_H

#include "qzen_safetensors.h"
#include "qzen_weights.h"

#include <stddef.h>
#include <stdint.h>

/* Quanto qint8 checkpoints (WanGP `quanto_bf16_int8`, used by Z-Image Turbo)
 * store every quantized Linear as a tensor pair
 *
 *   "<name>._data"   I8   [out, in]     quantized integer payload
 *   "<name>._scale"  BF16 [out, 1]     one BF16 scale per output row
 *
 * with no Hadamard rotation and no group scales, so dequantization is a plain
 * per-element multiply:
 *
 *   W_bf16 = bf16(float(_data[r, c]) * float(_scale[r]))
 *
 * This is much simpler than the INT8 ConvRot layout in qzen_weights.c: the
 * payload is read in natural [out, in] row-major order and can be consumed
 * directly as a BF16 GEMM weight matrix.  Unquantized tensors (norms, biases,
 * embeddings, the modulating Linear's bias) are plain BF16 and are read with
 * the ordinary qzen_st helpers.
 */

/* Locate and validate the quanto pair for a Linear with shape [rows, cols].
 * On success the tensor/header pointers (all optional) describe "<name>._data"
 * and "<name>._scale" and can be handed to qzen_st_read_data. */
int qzen_quanto_find(const qzen_weight_store *store, const char *name,
                   uint64_t rows, uint64_t cols,
                   const qzen_st_tensor **data, const qzen_st_header **data_header,
                   const qzen_st_tensor **scale, const qzen_st_header **scale_header,
                   char *error, size_t error_size);

/* Dequantize into a caller-owned BF16 buffer of rows * cols elements.
 * The I8 payload and BF16 scales are staged in temporary host buffers, so
 * peak memory is rows * (cols + 2) bytes beyond the output. */
int qzen_quanto_dequant_bf16(const qzen_weight_store *store, const char *name,
                           uint64_t rows, uint64_t cols, uint16_t *out,
                           char *error, size_t error_size);

/* Allocate-and-load convenience: dequantize into a fresh BF16 buffer of
 * rows * cols elements, owned by the caller (free()). */
uint16_t *qzen_quanto_load_bf16(const qzen_weight_store *store, const char *name,
                              uint64_t rows, uint64_t cols,
                              char *error, size_t error_size);

#endif
