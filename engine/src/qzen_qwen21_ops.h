/* Shared primitives for the Qwen-Image 2.1 CPU references (DiT and the
 * Qwen3-VL language tower): thread fan-out, the row-dot f32 GEMM, the
 * normalisations and the safetensors readers. Header-only and static inline so
 * both translation units compile their own copy without exporting symbols.
 */

#ifndef QZEN_QWEN21_OPS_H
#define QZEN_QWEN21_OPS_H

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "qzen_safetensors.h"

#define Q21_HIDDEN 4096
#define Q21_HEAD_DIM 128
#define Q21_FFN 12288
#define Q21_EPS 1e-6f
#define Q21_MAX_THREADS 32

static inline double q21_now(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- threading */

typedef void (*q21_range_fn)(void *context, int slot, int start, int end);

typedef struct {
    q21_range_fn run;
    void *context;
    int slot;
    int start;
    int end;
} q21_slice;

static inline void *q21_slice_main(void *opaque) {
    q21_slice *slice = opaque;
    slice->run(slice->context, slice->slot, slice->start, slice->end);
    return NULL;
}

/* Split [0, total) over at most threads workers. Slot 0 is the caller so a
 * worker can claim private scratch by ordinal; a failed spawn runs inline. */
static inline void q21_parallel(int total, int threads, q21_range_fn run,
                                void *context) {
    if (total <= 0) return;
    if (threads <= 1 || total < threads * 2) {
        run(context, 0, 0, total);
        return;
    }
    if (threads > Q21_MAX_THREADS) threads = Q21_MAX_THREADS;
    pthread_t handles[Q21_MAX_THREADS];
    q21_slice slices[Q21_MAX_THREADS];
    int created = 0;
    for (int index = 1; index < threads; index++) {
        long long start = (long long)total * index / threads;
        long long end = (long long)total * (index + 1) / threads;
        if (end <= start) continue;
        slices[created].run = run;
        slices[created].context = context;
        slices[created].slot = index;
        slices[created].start = (int)start;
        slices[created].end = (int)end;
        if (pthread_create(&handles[created], NULL, q21_slice_main, &slices[created]) != 0) {
            run(context, index, (int)start, (int)end);
            continue;
        }
        created++;
    }
    long long first_end = (long long)total / threads;
    run(context, 0, 0, (int)first_end);
    for (int index = 0; index < created; index++) pthread_join(handles[index], NULL);
}

/* --------------------------------------------------------------- linear map */
/* y[t][o] = sum_i x[t][i] * w[o][i]. Weight rows are the slow axis when the
 * activation block fits in cache (one pass over the weights), tokens otherwise. */

typedef struct {
    const float *restrict x;
    const float *restrict w;
    float *restrict y;
    int tokens;
    int in;
    int out;
} q21_gemm_job;

static inline void q21_gemm_over_outputs(void *context, int slot, int start, int end) {
    (void)slot;
    q21_gemm_job *job = context;
    for (int o = start; o < end; o++) {
        const float *w = job->w + (size_t)o * job->in;
        for (int t = 0; t < job->tokens; t++) {
            const float *row = job->x + (size_t)t * job->in;
            float sum = 0.0f;
            for (int i = 0; i < job->in; i++) sum += row[i] * w[i];
            job->y[(size_t)t * job->out + o] = sum;
        }
    }
}

static inline void q21_gemm_over_tokens(void *context, int slot, int start, int end) {
    (void)slot;
    q21_gemm_job *job = context;
    for (int t = start; t < end; t++) {
        const float *row = job->x + (size_t)t * job->in;
        float *dst = job->y + (size_t)t * job->out;
        for (int o = 0; o < job->out; o++) {
            const float *w = job->w + (size_t)o * job->in;
            float sum = 0.0f;
            for (int i = 0; i < job->in; i++) sum += row[i] * w[i];
            dst[o] = sum;
        }
    }
}

static inline void q21_linear(const float *x, int tokens, const float *w, int out, int in,
                              float *y, int threads) {
    q21_gemm_job job = {x, w, y, tokens, in, out};
    if ((double)tokens * (double)in <= 2.0e6) {
        q21_parallel(out, threads, q21_gemm_over_outputs, &job);
    } else {
        q21_parallel(tokens, threads, q21_gemm_over_tokens, &job);
    }
}

/* ------------------------------------------------------------- activations */

static inline void q21_layer_norm(const float *restrict x, int tokens, int dim,
                                  float *restrict y) {
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * dim;
        float *dst = y + (size_t)t * dim;
        double mean = 0.0;
        for (int i = 0; i < dim; i++) mean += (double)row[i];
        mean /= (double)dim;
        double variance = 0.0;
        for (int i = 0; i < dim; i++) {
            double delta = (double)row[i] - mean;
            variance += delta * delta;
        }
        variance /= (double)dim;
        float inverse = 1.0f / sqrtf((float)variance + Q21_EPS);
        float centre = (float)mean;
        for (int i = 0; i < dim; i++) dst[i] = (row[i] - centre) * inverse;
    }
}

/* RMSNorm over the last dimension with a dense weight (an unweighted norm is
 * per-head RMSNorm in the callers below). */
static inline void q21_rms_norm(const float *x, int tokens, int dim, const float *weight,
                                float *y) {
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * dim;
        float *dst = y + (size_t)t * dim;
        double sum = 0.0;
        for (int i = 0; i < dim; i++) sum += (double)row[i] * (double)row[i];
        float inverse = 1.0f / sqrtf((float)(sum / (double)dim) + Q21_EPS);
        for (int i = 0; i < dim; i++) dst[i] = row[i] * inverse * weight[i];
    }
}

static inline void q21_silu(float *x, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float value = x[index];
        x[index] = value / (1.0f + expf(-value));
    }
}

static inline void q21_gelu_tanh(float *x, size_t count) {
    const float k = 0.7978845608028654f; /* sqrt(2/pi) */
    for (size_t index = 0; index < count; index++) {
        float value = x[index];
        x[index] = 0.5f * value * (1.0f + tanhf(k * (value + 0.044715f * value * value * value)));
    }
}

static inline void q21_swiglu(float *restrict gate, const float *restrict value, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float item = gate[index];
        gate[index] = (item / (1.0f + expf(-item))) * value[index];
    }
}

/* ---------------------------------------------------------------- weights */

static inline void q21_scale_name(const char *name, char *out, size_t size) {
    size_t length = strlen(name);
    size_t keep = length;
    const char suffix[] = ".weight";
    size_t suffix_length = sizeof(suffix) - 1;
    if (length > suffix_length && strcmp(name + length - suffix_length, suffix) == 0) {
        keep = length - suffix_length;
    }
    snprintf(out, size, "%.*s.weight_scale", (int)keep, name);
}

static inline int q21_expect(const qzen_st_tensor *tensor, const char *name, int ndim,
                             const uint64_t *shape, char *error, size_t error_size) {
    if (!tensor) {
        snprintf(error, error_size, "%s: tensor missing from the checkpoint", name);
        return 0;
    }
    if (tensor->ndim != ndim) {
        snprintf(error, error_size, "%s: rank %d, expected %d", name, tensor->ndim, ndim);
        return 0;
    }
    for (int index = 0; index < ndim; index++) {
        if (tensor->shape[index] != shape[index]) {
            snprintf(error, error_size, "%s: shape[%d] %llu, expected %llu", name, index,
                     (unsigned long long)tensor->shape[index],
                     (unsigned long long)shape[index]);
            return 0;
        }
    }
    return 1;
}

/* Read a tensor as F32. I8 rows go through the shared ConvRot dequantiser
 * (row scale + 1/16 fold + Hadamard un-rotation); BF16 and F32 are widened here. */
static inline float *q21_read_f32(const qzen_st_header *header, const char *name, int ndim,
                                  const uint64_t *shape, char *error, size_t error_size) {
    const qzen_st_tensor *tensor = qzen_st_find(header, name);
    if (!q21_expect(tensor, name, ndim, shape, error, error_size)) return NULL;
    size_t elements = (size_t)qzen_st_tensor_elements(tensor);
    if (tensor->dtype == QZEN_DTYPE_I8) {
        char scale_name[256];
        q21_scale_name(name, scale_name, sizeof(scale_name));
        const qzen_st_tensor *scale = qzen_st_find(header, scale_name);
        if (!scale) {
            snprintf(error, error_size, "%s: INT8 weight without %s", name, scale_name);
            return NULL;
        }
        float *values = NULL;
        if (!qzen_st_read_f32(header, tensor, header, scale, &values, error, error_size)) {
            return NULL;
        }
        return values;
    }
    float *values = malloc(elements * sizeof(float));
    if (!values) {
        snprintf(error, error_size, "%s: out of memory for %zu floats", name, elements);
        return NULL;
    }
    if (tensor->dtype == QZEN_DTYPE_F32) {
        if (!qzen_st_read_data(header, tensor, values, elements * sizeof(float), error,
                             error_size)) {
            free(values);
            return NULL;
        }
        return values;
    }
    if (tensor->dtype == QZEN_DTYPE_BF16) {
        uint16_t *raw = malloc(elements * sizeof(uint16_t));
        if (!raw) {
            snprintf(error, error_size, "%s: out of memory for %zu halves", name, elements);
            free(values);
            return NULL;
        }
        int ok = qzen_st_read_data(header, tensor, raw, elements * sizeof(uint16_t), error,
                                 error_size);
        for (size_t index = 0; ok && index < elements; index++) {
            uint32_t bits = (uint32_t)raw[index] << 16;
            memcpy(&values[index], &bits, sizeof(float));
        }
        free(raw);
        if (!ok) {
            free(values);
            return NULL;
        }
        return values;
    }
    snprintf(error, error_size, "%s: dtype %s cannot be read as F32", name,
             qzen_dtype_name(tensor->dtype));
    free(values);
    return NULL;
}

/* Read rows [first_row, first_row + rows) of a 2-D tensor into F32, so the
 * language tower can pull just the embedding rows its token ids touch instead
 * of the whole 1.2 GB table. */
static inline int q21_read_rows_f32(const qzen_st_header *header, const qzen_st_tensor *tensor,
                                    uint64_t first_row, uint64_t rows, uint64_t columns,
                                    float *out, char *error, size_t error_size) {
    if (!tensor || tensor->ndim != 2 || tensor->shape[0] < first_row + rows ||
        tensor->shape[1] != columns) {
        snprintf(error, error_size, "%s: cannot read %llu rows of %llu columns",
                 tensor && tensor->name ? tensor->name : "tensor",
                 (unsigned long long)rows, (unsigned long long)columns);
        return 0;
    }
    if (tensor->dtype != QZEN_DTYPE_F32 && tensor->dtype != QZEN_DTYPE_BF16) {
        snprintf(error, error_size, "%s: dtype %s cannot be read as row floats",
                 tensor->name ? tensor->name : "tensor", qzen_dtype_name(tensor->dtype));
        return 0;
    }
    size_t element = qzen_dtype_size(tensor->dtype);
    size_t bytes = (size_t)(rows * columns) * element;
    unsigned char *raw = malloc(bytes);
    if (!raw) {
        snprintf(error, error_size, "%s: out of memory for %zu row bytes",
                 tensor->name ? tensor->name : "tensor", bytes);
        return 0;
    }
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0) {
        snprintf(error, error_size, "%s: cannot open the checkpoint", header->path);
        free(raw);
        return 0;
    }
    off_t offset = (off_t)(tensor->file_offset + first_row * columns * element);
    size_t done = 0;
    while (done < bytes) {
        ssize_t count = pread(descriptor, raw + done, bytes - done, offset + (off_t)done);
        if (count <= 0) break;
        done += (size_t)count;
    }
    close(descriptor);
    if (done != bytes) {
        snprintf(error, error_size, "%s: short read of row payload",
                 tensor->name ? tensor->name : "tensor");
        free(raw);
        return 0;
    }
    size_t elements = (size_t)(rows * columns);
    for (size_t index = 0; index < elements; index++) {
        if (tensor->dtype == QZEN_DTYPE_F32) {
            memcpy(&out[index], raw + index * sizeof(float), sizeof(float));
        } else {
            uint16_t half;
            memcpy(&half, raw + index * sizeof(uint16_t), sizeof(half));
            uint32_t bits = (uint32_t)half << 16;
            memcpy(&out[index], &bits, sizeof(float));
        }
    }
    free(raw);
    return 1;
}

#endif
