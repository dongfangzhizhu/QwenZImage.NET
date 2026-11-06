#ifndef QZEN_WEIGHTS_H
#define QZEN_WEIGHTS_H

#include "qzen_gpu.h"
#include "qzen_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct qzen_weight_store qzen_weight_store;

/* Which Nunchaku SVDQuant release a store holds.  The int4 and fp4 packs have
 * the same tensor names, so nothing else tells them apart. */
typedef enum {
    QZEN_SVDQ_NONE = 0,
    QZEN_SVDQ_INT4,
    QZEN_SVDQ_FP4
} qzen_svdq_kind;

/* Open every safetensors header of a component without reading tensor
 * payloads.  `path` is either a directory of shards or a single
 * `.safetensors` file, which is what a pack selected out of a directory of
 * packs resolves to. */
qzen_weight_store *qzen_weight_store_open(const char *path,
                                      char *error, size_t error_size);
void qzen_weight_store_free(qzen_weight_store *store);
size_t qzen_weight_store_shards(const qzen_weight_store *store);

const qzen_st_tensor *qzen_weight_find(const qzen_weight_store *store,
                                   const char *name,
                                   const qzen_st_header **header);

/* Identify a Nunchaku SVDQuant pack: the low-rank width (the second dimension
 * of any `proj_up`) and the code format, which is the dtype of the matching
 * `wscales` -- BF16 for int4, float8 for fp4.  Both come from the tensors, not
 * from `__metadata__`, so a repack that dropped the metadata still identifies.
 * Returns QZEN_SVDQ_NONE and leaves *rank at 0 for a store that is not one. */
qzen_svdq_kind qzen_weight_svdq_identify(const qzen_weight_store *store,
                                     uint32_t *rank);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
qzen_gpu_tensor *qzen_weight_load_bf16(const qzen_weight_store *store, qzen_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
qzen_gpu_tensor *qzen_weight_load_f32(const qzen_weight_store *store, qzen_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

/* INT8 ConvRot checkpoints (Comfy/WanGP `int8_convrot`) store the DiT linears as
 * "<name>" I8 [rows, cols] plus a per-output-row F32 scale sidecar
 * "<name minus .weight>.weight_scale" of [rows, 1] or [rows].  There is no BF16
 * copy in the file, so these weights are read straight into device memory and the
 * scale is used as-is by the INT8 GEMM; the ConvRot Hadamard rotation lives on the
 * activation side and is not applied here. */
qzen_gpu_tensor *qzen_weight_load_i8_convrot(const qzen_weight_store *store, qzen_gpu *gpu,
                                         const char *name, uint64_t rows,
                                         uint64_t cols, qzen_gpu_tensor **scales,
                                         char *error, size_t error_size);

/* Non-zero when the named tensor exists and is INT8, i.e. the checkpoint stores
 * quantized DiT weights instead of BF16 ones. */
int qzen_weight_is_int8(const qzen_weight_store *store, const char *name);
/* ConvRot checkpoints store qkv rows as three contiguous thirds [q][k][v], but
 * QwenZImageNative consumes them as per-head triples [h0q, h0k, h0v, h1q, ...]. These entries
 * read the payload in destination-block order (one 128-row block at a time, no
 * second pass over the tensor) and return the per-head layout; the per-row
 * scales follow the same permutation. Rotation runs along K, so only the row
 * order changes. `rows` must be 3 * heads * head_dim. */
qzen_gpu_tensor *qzen_weight_load_qkv_perhead(const qzen_weight_store *store, qzen_gpu *gpu,
                                          const char *name, uint64_t rows,
                                          uint64_t cols, uint32_t head_dim,
                                          qzen_gpu_tensor **scales, char *error,
                                          size_t error_size);
qzen_gpu_tensor *qzen_weight_load_qkv_perhead_bf16(
                                          const qzen_weight_store *store, qzen_gpu *gpu,
                                          const char *name, uint64_t rows,
                                          uint64_t cols, uint32_t head_dim,
                                          char *error, size_t error_size);

#endif
