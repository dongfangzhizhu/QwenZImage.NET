#include "qzen_zimage_dit.h"

#include "qzen_nunchaku.h"

#include <math.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- small helpers --------------------------------------------------------- */

static float zimg_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void zimg_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Read a plain BF16 tensor [dim0, dim1] into a BF16 buffer. */
static int read_bf16(const qzen_weight_store *store, const char *name,
                     uint64_t dim0, uint64_t dim1, uint16_t *out,
                     char *error, size_t error_size) {
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        zimg_fail(error, error_size, "required Z-Image weight is absent: %s", name);
        return 0;
    }
    uint64_t elements = dim0 * dim1;
    int shape_ok = tensor->dtype == QZEN_DTYPE_BF16;
    if (shape_ok && tensor->ndim == 1) {
        shape_ok = dim1 == 1 && tensor->shape[0] == dim0;
    } else if (shape_ok && tensor->ndim == 2) {
        shape_ok = tensor->shape[0] == dim0 && tensor->shape[1] == dim1;
    } else if (shape_ok) {
        shape_ok = 0;
    }
    if (!shape_ok) {
        zimg_fail(error, error_size,
                  "weight %s is %s[%d] with shape [%llu, %llu], expected BF16[2] [%llu, %llu]",
                  name, qzen_dtype_name(tensor->dtype), tensor->ndim,
                  (unsigned long long)tensor->shape[0],
                  (unsigned long long)tensor->shape[1],
                  (unsigned long long)dim0, (unsigned long long)dim1);
        return 0;
    }
    return qzen_st_read_data(header, tensor, out, (size_t)elements * 2,
                           error, error_size);
}

/* Read a plain BF16 vector into F32. */
static int read_bf16_f32(const qzen_weight_store *store, const char *name,
                         uint64_t count, float *out,
                         char *error, size_t error_size) {
    uint16_t *staged = malloc((size_t)count * 2);
    if (!staged) {
        zimg_fail(error, error_size, "cannot stage %s", name);
        return 0;
    }
    int ok = read_bf16(store, name, count, 1, staged, error, error_size);
    if (ok) {
        for (uint64_t index = 0; index < count; index++) {
            out[index] = zimg_bf16_to_f32(staged[index]);
        }
    }
    free(staged);
    return ok;
}

/* "<base>.weight" -> "<base>.bias": checkpoint bias tensors sit next to the
 * weight tensor, sharing the Linear's name. */
static void bias_tensor_name(const char *weight_name, char *out, size_t size) {
    static const char suffix[] = ".weight";
    size_t length = strlen(weight_name);
    size_t keep = length;
    if (length >= sizeof(suffix) - 1 &&
        strcmp(weight_name + length - (sizeof(suffix) - 1), suffix) == 0) {
        keep = length - (sizeof(suffix) - 1);
    }
    snprintf(out, size, "%.*s.bias", (int)keep, weight_name);
}

qzen_zimage_config qzen_zimage_default_config(void) {
    qzen_zimage_config config;
    config.dim = 3840;
    config.heads = 30;
    config.head_dim = 128;
    config.ffn_dim = 10240;
    config.adaln_dim = 256;
    config.norm_eps = 1e-5f;
    config.final_norm_eps = 1e-6f;
    config.rope_axes[0] = 32;
    config.rope_axes[1] = 48;
    config.rope_axes[2] = 48;
    config.rope_lens[0] = 1536;
    config.rope_lens[1] = 512;
    config.rope_lens[2] = 512;
    config.rope_theta = 256.0f;
    config.t_scale = 1000.0f;
    config.seq_multiple = 32;
    return config;
}

/* --- Linear ---------------------------------------------------------------- */

/* One place that knows every pointer a Linear can own, so the three loaders and
 * the free path cannot disagree about which of them start NULL. */
static void zimg_linear_reset(qzen_zimage_linear *linear) {
    linear->weight = NULL;
    linear->bias = NULL;
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
    linear->weight_f32 = NULL;
    memset(&linear->packed, 0, sizeof(linear->packed));
    linear->has_packed = 0;
    linear->gpu_codes = NULL;
    linear->gpu_scales = NULL;
    linear->gpu_smooth = NULL;
    linear->gpu_up = NULL;
    linear->gpu_down = NULL;
}

int qzen_zimage_linear_load(const qzen_weight_store *store, const char *name,
                          uint32_t out, uint32_t in, int has_bias,
                          qzen_zimage_linear *linear,
                          char *error, size_t error_size) {
    linear->out = out;
    linear->in = in;
    zimg_linear_reset(linear);
    linear->weight = malloc((size_t)out * in * 2);
    if (!linear->weight) {
        zimg_fail(error, error_size, "cannot allocate linear weight %s", name);
        return 0;
    }
    if (!read_bf16(store, name, out, in, linear->weight, error, error_size)) {
        qzen_zimage_linear_free(linear);
        return 0;
    }
    if (has_bias) {
        char bias_name[224];
        bias_tensor_name(name, bias_name, sizeof(bias_name));
        linear->bias = malloc((size_t)out * sizeof(float));
        if (!linear->bias ||
            !read_bf16_f32(store, bias_name, out, linear->bias, error, error_size)) {
            qzen_zimage_linear_free(linear);
            return 0;
        }
    }
    return 1;
}

int qzen_zimage_linear_load_quanto(const qzen_weight_store *store, const char *name,
                                 uint32_t out, uint32_t in, int has_bias,
                                 qzen_zimage_linear *linear,
                                 char *error, size_t error_size) {
    linear->out = out;
    linear->in = in;
    zimg_linear_reset(linear);
    linear->weight = qzen_quanto_load_bf16(store, name, out, in, error, error_size);
    if (!linear->weight) return 0;
    if (has_bias) {
        char bias_name[192];
        bias_tensor_name(name, bias_name, sizeof(bias_name));
        linear->bias = malloc((size_t)out * sizeof(float));
        if (!linear->bias ||
            !read_bf16_f32(store, bias_name, out, linear->bias, error, error_size)) {
            qzen_zimage_linear_free(linear);
            return 0;
        }
    }
    return 1;
}

void qzen_zimage_linear_free(qzen_zimage_linear *linear) {
    if (linear->gpu_weight) qzen_gpu_tensor_free(linear->gpu_weight);
    if (linear->gpu_bias) qzen_gpu_tensor_free(linear->gpu_bias);
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
    if (linear->gpu_codes) qzen_gpu_tensor_free(linear->gpu_codes);
    if (linear->gpu_scales) qzen_gpu_tensor_free(linear->gpu_scales);
    if (linear->gpu_smooth) qzen_gpu_tensor_free(linear->gpu_smooth);
    if (linear->gpu_up) qzen_gpu_tensor_free(linear->gpu_up);
    if (linear->gpu_down) qzen_gpu_tensor_free(linear->gpu_down);
    linear->gpu_codes = NULL;
    linear->gpu_scales = NULL;
    linear->gpu_smooth = NULL;
    linear->gpu_up = NULL;
    linear->gpu_down = NULL;
    qzen_nunchaku_packed_free(&linear->packed);
    linear->has_packed = 0;
    free(linear->weight_f32);
    linear->weight_f32 = NULL;
    free(linear->weight);
    free(linear->bias);
    linear->weight = NULL;
    linear->bias = NULL;
}

const uint16_t *qzen_zimage_linear_host_weight(qzen_zimage_linear *linear) {
    if (!linear) return NULL;
    if (linear->weight) return linear->weight;
    if (!linear->has_packed) return NULL;
    char error[256] = {0};
    linear->weight = qzen_nunchaku_packed_to_bf16(&linear->packed, error,
                                               sizeof(error));
    if (!linear->weight)
        fprintf(stderr,
                "QwenZImageNative zimage: cannot materialise a packed weight: %s\n", error);
    return linear->weight;
}

/* Which quantisation the block under `prefix` was stored with.  The three
 * flavours disagree on tensor names as much as on encoding, so this is read
 * off the file rather than guessed from the section name.
 *
 * Nunchaku exports fuse to_q/to_k/to_v into one `attention.to_qkv` and w1/w3
 * into `feed_forward.net.0.proj` (w3 in the first half, w1 in the second --
 * verified against the quanto checkpoint of the same weights), so a Nunchaku
 * directory cannot be loaded through the split-name path at all. */
typedef enum {
    ZIMG_STORE_BF16,
    ZIMG_STORE_QUANTO,
    ZIMG_STORE_NUNCHAKU
} zimg_store_kind;

static zimg_store_kind block_store_kind(const qzen_weight_store *store,
                                        const char *prefix) {
    char name[224];
    snprintf(name, sizeof(name), "%s.attention.to_qkv", prefix);
    if (qzen_nunchaku_present(store, name)) return ZIMG_STORE_NUNCHAKU;
    snprintf(name, sizeof(name), "%s.attention.to_q.weight._data", prefix);
    if (qzen_weight_find(store, name, NULL)) return ZIMG_STORE_QUANTO;
    return ZIMG_STORE_BF16;
}

/* One row range of a Nunchaku matrix.  `full_out` is the row count of the
 * whole tensor and `rows` the slice: a fused to_qkv is 3 * dim rows and each
 * of q/k/v is one third of it, and the underlying pack is cached, so the three
 * slices cost one pass.
 *
 * The slice is kept packed (PLAN §19).  `linear->weight` stays NULL until a
 * host path asks for it through qzen_zimage_linear_host_weight, which is what
 * makes loading a Nunchaku pack cost 4.55 GB of disk instead of 10.6 GB of
 * expanded weights.  That cache is right for the tests and for
 * zimg_linear_widen; the CPU fallback must not use it (see
 * qzen_zimage_linear_forward, PLAN §21). */
static int load_nunchaku_slice(const qzen_weight_store *store, const char *base,
                               uint32_t full_out, uint32_t in,
                               uint32_t row_offset, uint32_t rows,
                               qzen_zimage_linear *linear,
                               char *error, size_t error_size) {
    linear->out = rows;
    linear->in = in;
    zimg_linear_reset(linear);
    if (!qzen_nunchaku_load_packed_rows(store, base, full_out, in, row_offset,
                                      rows, &linear->packed, error,
                                      error_size))
        return 0;
    linear->has_packed = 1;
    return 1;
}

/* Checkpoint layout: the 30 main blocks store quanto qint8 pairs while the
 * noise/context refiners keep plain BF16 Linears. */
static int load_block_linear(const qzen_weight_store *store, const char *name,
                             uint32_t out, uint32_t in, int has_bias,
                             int quantized, qzen_zimage_linear *linear,
                             char *error, size_t error_size) {
    if (quantized) {
        return qzen_zimage_linear_load_quanto(store, name, out, in, has_bias,
                                            linear, error, error_size);
    }
    return qzen_zimage_linear_load(store, name, out, in, has_bias,
                                 linear, error, error_size);
}

/* The CPU fallback's expansion scratch, shared by every packed layer.
 *
 * A packed layer has no host matrix, so a CPU forward has to produce one.  Two
 * wrong ways to do it, both measured on a 256x256 one-step request with
 * ZIMAGE_GPU=0 (PLAN §21):
 *
 *   - keep it in the Linear (qzen_zimage_linear_host_weight): 136 layers of bf16
 *     is 10.6 GB on top of the 4.55 GB pack, 20.30 GiB peak / 13.24 GiB
 *     retained, and none of it is ever given back;
 *   - expand into a fresh block per call: memory is fine, but 21 GB of fresh
 *     pages per denoise step cost more in page faults than the arithmetic, and
 *     the per-element low-rank term costs rank multiply-adds for every element
 *     of every layer -- 5 min per step against 73 s.
 *
 * So: one buffer, grown to the largest layer (157 MB), reused by every layer,
 * holding only the quantised branch.  The low-rank branch is applied to the
 * activations as two GEMMs instead, which is what the device kernel does too.
 * The forward is single-threaded at this level (every OpenMP region is inside a
 * kernel), so one buffer is enough; qzen_zimage_cpu_scratch_release gives it back
 * when the model goes away. */
static uint16_t *zimg_cpu_scratch;
static size_t zimg_cpu_scratch_elems;

void qzen_zimage_cpu_scratch_release(void) {
    free(zimg_cpu_scratch);
    zimg_cpu_scratch = NULL;
    zimg_cpu_scratch_elems = 0;
}

static uint16_t *zimg_cpu_scratch_reserve(size_t elems) {
    if (elems <= zimg_cpu_scratch_elems) return zimg_cpu_scratch;
    uint16_t *grown = realloc(zimg_cpu_scratch, elems * sizeof(uint16_t));
    if (!grown) return NULL;
    zimg_cpu_scratch = grown;
    zimg_cpu_scratch_elems = elems;
    return grown;
}

/* output += proj_up @ (proj_down @ input), the SVDQuant low-rank branch.
 *
 * proj_down is [rank, in] and proj_up is [out, rank], so the branch is
 * up @ down applied to the activations: two GEMMs against `rows` instead of a
 * rank-length inner product for each of the out*in weights.  The numbers differ
 * from the per-element form by one bf16 rounding of the sum (the device path
 * has the same property, §19.4); the gate in test_nunchaku_block pins that. */
static int zimg_linear_add_lowrank(const qzen_nunchaku_packed *packed,
                                   const float *input, uint32_t rows,
                                   float *output) {
    const uint32_t in = packed->in;
    const uint32_t out = packed->out;
    const uint32_t rank = packed->rank;
    if (!rank || !packed->proj_up || !packed->proj_down) return 1;
    /* One [rows, rank] scratch for the whole call rather than a 1 KB malloc per
     * row: a failed allocation here would drop the branch silently, and a
     * quietly wrong velocity field is worse than a refused request. */
    float *mixed_all = malloc((size_t)rows * rank * sizeof(float));
    if (!mixed_all) {
        fprintf(stderr,
                "QwenZImageNative zimage: cannot allocate %zu bytes for the low-rank branch\n",
                (size_t)rows * rank * sizeof(float));
        return 0;
    }
    int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (row64 = 0; row64 < (int64_t)rows; row64++) {
        uint32_t row = (uint32_t)row64;
        const float *x = input + (size_t)row * in;
        float *y = output + (size_t)row * out;
        float *mixed = mixed_all + (size_t)row * rank;
        for (uint32_t r = 0; r < rank; r++) {
            const uint16_t *down = packed->proj_down + (size_t)r * in;
            float sum = 0.0f;
            for (uint32_t c = 0; c < in; c++) {
                sum += x[c] * zimg_bf16_to_f32(down[c]);
            }
            mixed[r] = sum;
        }
        for (uint32_t o = 0; o < out; o++) {
            const uint16_t *up = packed->proj_up + (size_t)o * rank;
            float sum = 0.0f;
            for (uint32_t r = 0; r < rank; r++) {
                sum += zimg_bf16_to_f32(up[r]) * mixed[r];
            }
            y[o] += sum;
        }
    }
    free(mixed_all);
    return 1;
}

/* out[rows, out_dim] = input[rows, in_dim] @ weight[in, out]^T + bias.
 * Weight rows are contiguous in the input dimension, so the inner loop walks
 * the input vector and the weight row together.
 *
 * A packed layer goes through the shared scratch (see above), so a CPU forward
 * never materialises more than one layer of weights. */
int qzen_zimage_linear_forward(const qzen_zimage_linear *linear,
                             const float *input, uint32_t rows, float *output) {
    uint32_t in = linear->in;
    uint32_t out = linear->out;
    const uint16_t *weight = linear->weight;
    int packed = 0;
    if (!weight && linear->has_packed) {
        char error[256] = {0};
        uint16_t *scratch = zimg_cpu_scratch_reserve((size_t)out * in);
        if (!scratch) {
            fprintf(stderr, "QwenZImageNative zimage: cannot reserve %zu bytes of scratch\n",
                    (size_t)out * in * sizeof(uint16_t));
            return 0;
        }
        if (!qzen_nunchaku_packed_quant_into(&linear->packed, scratch, error,
                                           sizeof(error))) {
            fprintf(stderr, "QwenZImageNative zimage: cannot expand a packed weight: %s\n",
                    error);
            return 0;
        }
        weight = scratch;
        packed = 1;
    }
    if (!weight) return 0;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            uint32_t row = (uint32_t)row64;
            const float *x = input + (size_t)row * in;
            float *y = output + (size_t)row * out;
            for (uint32_t o = 0; o < out; o++) {
                const uint16_t *w = weight + (size_t)o * in;
                float sum = linear->bias ? linear->bias[o] : 0.0f;
                for (uint32_t c = 0; c < in; c++) {
                    sum += x[c] * zimg_bf16_to_f32(w[c]);
                }
                y[o] = sum;
            }
        }
    }
    if (packed && !zimg_linear_add_lowrank(&linear->packed, input, rows, output))
        return 0;
    return 1;
}

/* --- RMSNorm --------------------------------------------------------------- */

int qzen_zimage_rms_load(const qzen_weight_store *store, const char *name,
                       uint32_t width, float eps, qzen_zimage_rms *rms,
                       char *error, size_t error_size) {
    rms->width = width;
    rms->eps = eps;
    rms->gpu_weight = NULL;
    rms->weight = malloc((size_t)width * 2);
    if (!rms->weight) {
        zimg_fail(error, error_size, "cannot allocate RMSNorm weight %s", name);
        return 0;
    }
    if (!read_bf16(store, name, width, 1, rms->weight, error, error_size)) {
        qzen_zimage_rms_free(rms);
        return 0;
    }
    return 1;
}

void qzen_zimage_rms_free(qzen_zimage_rms *rms) {
    if (rms->gpu_weight) qzen_gpu_tensor_free(rms->gpu_weight);
    rms->gpu_weight = NULL;
    free(rms->weight);
    rms->weight = NULL;
}

int qzen_zimage_rms_forward(const qzen_zimage_rms *rms, const float *input,
                          uint32_t rows, float *output) {
    float scale = 1.0f / (float)rms->width;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            uint32_t row = (uint32_t)row64;
            const float *x = input + (size_t)row * rms->width;
            float *y = output + (size_t)row * rms->width;
            float sum = 0.0f;
            for (uint32_t c = 0; c < rms->width; c++) sum += x[c] * x[c];
            float inverse = 1.0f / sqrtf(sum * scale + rms->eps);
            for (uint32_t c = 0; c < rms->width; c++) {
                y[c] = x[c] * inverse * zimg_bf16_to_f32(rms->weight[c]);
            }
        }
    }
    return 1;
}

/* --- Block ----------------------------------------------------------------- */

struct qzen_zimage_block {
    qzen_zimage_config config;
    int modulated;
    qzen_zimage_linear to_q;
    qzen_zimage_linear to_k;
    qzen_zimage_linear to_v;
    qzen_zimage_linear to_out;
    qzen_zimage_linear w1;
    qzen_zimage_linear w2;
    qzen_zimage_linear w3;
    qzen_zimage_linear adaln;   /* [4*dim, 256], bias included */
    qzen_zimage_rms norm_q;
    qzen_zimage_rms norm_k;
    qzen_zimage_rms attn_norm1;
    qzen_zimage_rms attn_norm2;
    qzen_zimage_rms ffn_norm1;
    qzen_zimage_rms ffn_norm2;
};

int qzen_zimage_block_modulated(const qzen_zimage_block *block) {
    return block->modulated;
}

const qzen_zimage_linear *qzen_zimage_block_linear(const qzen_zimage_block *block,
                                               const char *name) {
    if (!block || !name) return NULL;
    if (strcmp(name, "to_q") == 0) return &block->to_q;
    if (strcmp(name, "to_k") == 0) return &block->to_k;
    if (strcmp(name, "to_v") == 0) return &block->to_v;
    if (strcmp(name, "to_out") == 0) return &block->to_out;
    if (strcmp(name, "w1") == 0) return &block->w1;
    if (strcmp(name, "w2") == 0) return &block->w2;
    if (strcmp(name, "w3") == 0) return &block->w3;
    if (strcmp(name, "adaln") == 0) return &block->adaln;
    return NULL;
}

qzen_zimage_block *qzen_zimage_block_load(const qzen_weight_store *store,
                                      const char *section, uint32_t index,
                                      const qzen_zimage_config *config,
                                      char *error, size_t error_size) {
    qzen_zimage_block *block = calloc(1, sizeof(qzen_zimage_block));
    if (!block) {
        zimg_fail(error, error_size, "cannot allocate Z-Image block %s.%u", section, index);
        return NULL;
    }
    block->config = *config;
    block->modulated = strcmp(section, "context_refiner") != 0;
    char prefix[160];
    snprintf(prefix, sizeof(prefix), "%s.%u", section, index);
    const zimg_store_kind kind = block_store_kind(store, prefix);
    char name[224];
#define ZIMG_NAME(buffer, format, ...) \
    do { snprintf(buffer, sizeof(buffer), format, __VA_ARGS__); } while (0)

    if (kind == ZIMG_STORE_NUNCHAKU) {
        uint32_t dim = config->dim;
        uint32_t ffn = config->ffn_dim;
        ZIMG_NAME(name, "%s.attention.to_qkv", prefix);
        if (!load_nunchaku_slice(store, name, 3 * dim, dim, 0, dim,
                                 &block->to_q, error, error_size) ||
            !load_nunchaku_slice(store, name, 3 * dim, dim, dim, dim,
                                 &block->to_k, error, error_size) ||
            !load_nunchaku_slice(store, name, 3 * dim, dim, 2 * dim, dim,
                                 &block->to_v, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.attention.to_out.0", prefix);
        if (!load_nunchaku_slice(store, name, dim, dim, 0, dim, &block->to_out,
                                 error, error_size)) goto failure;
        /* net.0.proj is [w3; w1] stacked, net.2 is w2. */
        ZIMG_NAME(name, "%s.feed_forward.net.0.proj", prefix);
        if (!load_nunchaku_slice(store, name, 2 * ffn, dim, 0, ffn,
                                 &block->w3, error, error_size) ||
            !load_nunchaku_slice(store, name, 2 * ffn, dim, ffn, ffn,
                                 &block->w1, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.feed_forward.net.2", prefix);
        if (!load_nunchaku_slice(store, name, dim, ffn, 0, dim, &block->w2,
                                 error, error_size)) goto failure;
    } else {
        int quantized = kind == ZIMG_STORE_QUANTO;
        ZIMG_NAME(name, "%s.attention.to_q.weight", prefix);
        if (!load_block_linear(store, name, config->dim, config->dim, 0,
                               quantized, &block->to_q, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.attention.to_k.weight", prefix);
        if (!load_block_linear(store, name, config->dim, config->dim, 0,
                               quantized, &block->to_k, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.attention.to_v.weight", prefix);
        if (!load_block_linear(store, name, config->dim, config->dim, 0,
                               quantized, &block->to_v, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.attention.to_out.0.weight", prefix);
        if (!load_block_linear(store, name, config->dim, config->dim, 0,
                               quantized, &block->to_out, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.feed_forward.w1.weight", prefix);
        if (!load_block_linear(store, name, config->ffn_dim, config->dim, 0,
                               quantized, &block->w1, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.feed_forward.w2.weight", prefix);
        if (!load_block_linear(store, name, config->dim, config->ffn_dim, 0,
                               quantized, &block->w2, error, error_size)) goto failure;
        ZIMG_NAME(name, "%s.feed_forward.w3.weight", prefix);
        if (!load_block_linear(store, name, config->ffn_dim, config->dim, 0,
                               quantized, &block->w3, error, error_size)) goto failure;
    }
    ZIMG_NAME(name, "%s.attention.norm_q.weight", prefix);
    if (!qzen_zimage_rms_load(store, name, config->head_dim, config->norm_eps, &block->norm_q,
                            error, error_size)) goto failure;
    ZIMG_NAME(name, "%s.attention.norm_k.weight", prefix);
    if (!qzen_zimage_rms_load(store, name, config->head_dim, config->norm_eps, &block->norm_k,
                            error, error_size)) goto failure;
    ZIMG_NAME(name, "%s.attention_norm1.weight", prefix);
    if (!qzen_zimage_rms_load(store, name, config->dim, config->norm_eps, &block->attn_norm1,
                            error, error_size)) goto failure;
    ZIMG_NAME(name, "%s.attention_norm2.weight", prefix);
    if (!qzen_zimage_rms_load(store, name, config->dim, config->norm_eps, &block->attn_norm2,
                            error, error_size)) goto failure;
    ZIMG_NAME(name, "%s.ffn_norm1.weight", prefix);
    if (!qzen_zimage_rms_load(store, name, config->dim, config->norm_eps, &block->ffn_norm1,
                            error, error_size)) goto failure;
    ZIMG_NAME(name, "%s.ffn_norm2.weight", prefix);
    if (!qzen_zimage_rms_load(store, name, config->dim, config->norm_eps, &block->ffn_norm2,
                            error, error_size)) goto failure;
    if (block->modulated) {
        /* Nunchaku leaves adaLN in BF16 -- it is 15360 x 256, not worth four
         * bits -- so only the quanto checkpoint has it quantised. */
        ZIMG_NAME(name, "%s.adaLN_modulation.0.weight", prefix);
        if (!load_block_linear(store, name, 4 * config->dim,
                               config->adaln_dim, 1,
                               kind == ZIMG_STORE_QUANTO, &block->adaln,
                               error, error_size)) goto failure;
    }
#undef ZIMG_NAME
    return block;

failure:
    qzen_zimage_block_free(block);
    return NULL;
}

void qzen_zimage_block_free(qzen_zimage_block *block) {
    if (!block) return;
    qzen_zimage_linear_free(&block->to_q);
    qzen_zimage_linear_free(&block->to_k);
    qzen_zimage_linear_free(&block->to_v);
    qzen_zimage_linear_free(&block->to_out);
    qzen_zimage_linear_free(&block->w1);
    qzen_zimage_linear_free(&block->w2);
    qzen_zimage_linear_free(&block->w3);
    qzen_zimage_linear_free(&block->adaln);
    qzen_zimage_rms_free(&block->norm_q);
    qzen_zimage_rms_free(&block->norm_k);
    qzen_zimage_rms_free(&block->attn_norm1);
    qzen_zimage_rms_free(&block->attn_norm2);
    qzen_zimage_rms_free(&block->ffn_norm1);
    qzen_zimage_rms_free(&block->ffn_norm2);
    free(block);
}

/* RoPE over one [rows, heads, head_dim] tensor: interleaved pairs
 * (x[2i], x[2i+1]) rotated by (cos[i], sin[i]) from the [rows, head_dim / 2]
 * tables. */
static void apply_rope(float *tensor, uint32_t rows, uint32_t heads,
                       uint32_t head_dim, const float *cos_table,
                       const float *sin_table) {
    uint32_t half = head_dim / 2;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            uint32_t row = (uint32_t)row64;
            const float *cos_row = cos_table + (size_t)row * half;
            const float *sin_row = sin_table + (size_t)row * half;
            float *tokens = tensor + (size_t)row * heads * head_dim;
            for (uint32_t head = 0; head < heads; head++) {
                float *x = tokens + (size_t)head * head_dim;
                for (uint32_t pair = 0; pair < half; pair++) {
                    float x0 = x[2 * pair];
                    float x1 = x[2 * pair + 1];
                    float c = cos_row[pair];
                    float s = sin_row[pair];
                    x[2 * pair] = x0 * c - x1 * s;
                    x[2 * pair + 1] = x1 * c + x0 * s;
                }
            }
        }
    }
}

/* Full-sequence bidirectional SDPA over [heads, seq, head_dim]. */
static void sdpa(const float *q, const float *k, const float *v,
                 uint32_t seq, uint32_t heads, uint32_t head_dim,
                 float *output, float *scores) {
    float scale = 1.0f / sqrtf((float)head_dim);
    {
        int64_t head64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (head64 = 0; head64 < (int64_t)heads; head64++) {
            uint32_t head = (uint32_t)head64;
            /* per-head slice of the shared [seq, heads] score buffer */
            float *head_scores = scores + (size_t)head * seq;
            for (uint32_t i = 0; i < seq; i++) {
                const float *q_row = q + ((size_t)i * heads + head) * head_dim;
                float maximum = -INFINITY;
                for (uint32_t j = 0; j < seq; j++) {
                    const float *k_row = k + ((size_t)j * heads + head) * head_dim;
                    float dot = 0.0f;
                    for (uint32_t c = 0; c < head_dim; c++) dot += q_row[c] * k_row[c];
                    dot *= scale;
                    head_scores[j] = dot;
                    if (dot > maximum) maximum = dot;
                }
                float sum = 0.0f;
                for (uint32_t j = 0; j < seq; j++) {
                    head_scores[j] = expf(head_scores[j] - maximum);
                    sum += head_scores[j];
                }
                float inverse = 1.0f / sum;
                float *out_row = output + ((size_t)i * heads + head) * head_dim;
                memset(out_row, 0, head_dim * sizeof(float));
                for (uint32_t j = 0; j < seq; j++) {
                    const float *v_row = v + ((size_t)j * heads + head) * head_dim;
                    float weight = head_scores[j] * inverse;
                    for (uint32_t c = 0; c < head_dim; c++) out_row[c] += weight * v_row[c];
                }
            }
        }
    }
}

int qzen_zimage_block_forward(const qzen_zimage_block *block, float *x,
                            uint32_t seq, const float *adaln,
                            const float *rope_cos, const float *rope_sin,
                            char *error, size_t error_size) {
    const qzen_zimage_config *config = &block->config;
    uint32_t dim = config->dim;
    uint32_t ffn = config->ffn_dim;
    uint32_t heads = config->heads;
    uint32_t head_dim = config->head_dim;
    int modulated = block->modulated;

    if (modulated && !adaln) {
        zimg_fail(error, error_size, "modulated block requires an adaLN input");
        return 0;
    }

    float *mod = NULL;
    if (modulated) {
        mod = malloc((size_t)4 * dim * sizeof(float));
        if (!mod) {
            zimg_fail(error, error_size, "cannot allocate adaLN modulation buffer");
            return 0;
        }
        qzen_zimage_linear_forward(&block->adaln, adaln, 1, mod);
        /* scale_msa += 1 and scale_mlp += 1; gates stay raw (tanh applied later). */
        for (uint32_t c = 0; c < dim; c++) {
            mod[c] += 1.0f;
            mod[2 * dim + c] += 1.0f;
        }
    }

    size_t plane = (size_t)seq * dim;
    float *normed = malloc(plane * sizeof(float));
    float *q = malloc(plane * sizeof(float));
    float *k = malloc(plane * sizeof(float));
    float *v = malloc(plane * sizeof(float));
    float *attn = malloc(plane * sizeof(float));
    if (attn) memset(attn, 0, plane * sizeof(float));
    float *branch = malloc(plane * sizeof(float));
    float *scores = malloc((size_t)seq * heads * sizeof(float));
    float *ffn_a = malloc((size_t)seq * ffn * sizeof(float));
    float *ffn_b = malloc((size_t)seq * ffn * sizeof(float));
    if (!normed || !q || !k || !v || !attn || !branch || !scores || !ffn_a || !ffn_b) {
        zimg_fail(error, error_size, "cannot allocate Z-Image block buffers");
        free(mod); free(normed); free(q); free(k); free(v);
        free(attn); free(branch); free(scores); free(ffn_a); free(ffn_b);
        return 0;
    }

    /* Attention sub-block: h = norm1(x) [* scale_msa]; norm2 comes after the
     * output projection, matching attention_norm2(attention(...)) in the
     * reference. */
    qzen_zimage_rms_forward(&block->attn_norm1, x, seq, normed);
    if (modulated) {
        {
            int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (index = 0; index < (int64_t)plane; index++) {
                normed[index] *= mod[index % dim];
            }
        }
    }
    qzen_zimage_linear_forward(&block->to_q, normed, seq, q);
    qzen_zimage_linear_forward(&block->to_k, normed, seq, k);
    qzen_zimage_linear_forward(&block->to_v, normed, seq, v);
    /* QK RMSNorm over head_dim, then RoPE, per token (row-major [seq, heads, head_dim]). */
    for (uint32_t row = 0; row < seq; row++) {
        qzen_zimage_rms_forward(&block->norm_q, q + (size_t)row * dim, heads,
                              q + (size_t)row * dim);
        qzen_zimage_rms_forward(&block->norm_k, k + (size_t)row * dim, heads,
                              k + (size_t)row * dim);
    }
    apply_rope(q, seq, heads, head_dim, rope_cos, rope_sin);
    apply_rope(k, seq, heads, head_dim, rope_cos, rope_sin);
    sdpa(q, k, v, seq, heads, head_dim, attn, scores);
    qzen_zimage_linear_forward(&block->to_out, attn, seq, branch);
    qzen_zimage_rms_forward(&block->attn_norm2, branch, seq, branch);
    if (modulated) {
        {
            int64_t row;
            uint32_t c;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (row = 0; row < (int64_t)seq; row++) {
                for (c = 0; c < dim; c++) {
                    branch[(size_t)row * dim + c] *= tanhf(mod[dim + c]);
                }
            }
        }
    }
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)plane; index++) {
            x[index] += branch[index];
        }
    }

    /* FFN sub-block: n = norm1(x) [* scale_mlp]; norm2 after w2. */
    qzen_zimage_rms_forward(&block->ffn_norm1, x, seq, normed);
    if (modulated) {
        {
            int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (index = 0; index < (int64_t)plane; index++) {
                normed[index] *= mod[2 * dim + index % dim];
            }
        }
    }
    qzen_zimage_linear_forward(&block->w1, normed, seq, ffn_a);
    qzen_zimage_linear_forward(&block->w3, normed, seq, ffn_b);
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)((size_t)seq * ffn); index++) {
            float gated = ffn_a[index];
            gated = gated / (1.0f + expf(-gated)); /* SiLU */
            ffn_a[index] = gated * ffn_b[index];
        }
    }
    qzen_zimage_linear_forward(&block->w2, ffn_a, seq, branch);
    qzen_zimage_rms_forward(&block->ffn_norm2, branch, seq, branch);
    if (modulated) {
        {
            int64_t row;
            uint32_t c;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (row = 0; row < (int64_t)seq; row++) {
                for (c = 0; c < dim; c++) {
                    branch[(size_t)row * dim + c] *= tanhf(mod[3 * dim + c]);
                }
            }
        }
    }
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)plane; index++) {
            x[index] += branch[index];
        }
    }

    free(mod);
    free(normed);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(branch);
    free(scores);
    free(ffn_a);
    free(ffn_b);
    return 1;
}

/* --- RoPE tables ----------------------------------------------------------- */

int qzen_zimage_rope_tables(const qzen_zimage_config *config, const int32_t *ids,
                          uint32_t seq, float *cos, float *sin) {
    uint32_t half = config->head_dim / 2;
    uint32_t offset = 0;
    for (int axis = 0; axis < 3; axis++) {
        uint32_t width = config->rope_axes[axis] / 2; /* pairs per axis */
        for (uint32_t pair = 0; pair < width; pair++) {
            /* freq = theta^(-2*pair / axis_dim), computed in double like the
             * reference (arange(0, d, 2) / d on a float64 tensor). */
            double exponent = (double)(2 * pair) / (double)config->rope_axes[axis];
            double frequency = 1.0 / pow((double)config->rope_theta, exponent);
            for (uint32_t row = 0; row < seq; row++) {
                double angle = (double)ids[(size_t)row * 3 + axis] * frequency;
                float angle32 = (float)angle;
                cos[(size_t)row * half + offset + pair] = cosf(angle32);
                sin[(size_t)row * half + offset + pair] = sinf(angle32);
            }
        }
        offset += width;
    }
    return offset == half;
}

/* --- Timestep embedding ---------------------------------------------------- */

void qzen_zimage_timestep_embedding(float t, uint32_t dim, float *out) {
    uint32_t half = dim / 2;
    float log_base = logf(10000.0f);
    for (uint32_t index = 0; index < half; index++) {
        float frequency = expf(-log_base * (float)index / (float)half);
        float angle = t * frequency;
        out[index] = cosf(angle);          /* cos half first, like the reference */
        out[half + index] = sinf(angle);
    }
}

int qzen_zimage_t_embedder_load(const qzen_weight_store *store,
                              const qzen_zimage_config *config,
                              qzen_zimage_t_embedder *embedder,
                              char *error, size_t error_size) {
    if (!qzen_zimage_linear_load(store, "t_embedder.mlp.0.weight", 1024,
                               config->adaln_dim, 1, &embedder->fc1,
                               error, error_size)) {
        return 0;
    }
    if (!qzen_zimage_linear_load(store, "t_embedder.mlp.2.weight",
                               config->adaln_dim, 1024, 1, &embedder->fc2,
                               error, error_size)) {
        qzen_zimage_t_embedder_free(embedder);
        return 0;
    }
    return 1;
}

void qzen_zimage_t_embedder_free(qzen_zimage_t_embedder *embedder) {
    qzen_zimage_linear_free(&embedder->fc1);
    qzen_zimage_linear_free(&embedder->fc2);
}

int qzen_zimage_t_embedder_forward(const qzen_zimage_t_embedder *embedder,
                                 const float *input, uint32_t rows,
                                 float *output) {
    uint32_t mid = embedder->fc1.out;
    uint32_t out = embedder->fc2.out;
    float *hidden = malloc((size_t)rows * mid * sizeof(float));
    if (!hidden) return 0;
    qzen_zimage_linear_forward(&embedder->fc1, input, rows, hidden);
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)((size_t)rows * mid); index++) {
            float value = hidden[index];
            hidden[index] = value / (1.0f + expf(-value)); /* SiLU */
        }
    }
    qzen_zimage_linear_forward(&embedder->fc2, hidden, rows, output);
    free(hidden);
    (void)out;
    return 1;
}

/* --- Final layer ----------------------------------------------------------- */

int qzen_zimage_final_layer_load(const qzen_weight_store *store,
                               const qzen_zimage_config *config,
                               uint32_t out, qzen_zimage_final_layer *layer,
                               char *error, size_t error_size) {
    layer->dim = config->dim;
    layer->out = out;
    layer->norm_eps = config->final_norm_eps;
    if (!qzen_zimage_linear_load(store, "all_final_layer.2-1.adaLN_modulation.1.weight",
                               config->dim, config->adaln_dim, 1,
                               &layer->adaLN, error, error_size)) {
        return 0;
    }
    if (!qzen_zimage_linear_load(store, "all_final_layer.2-1.linear.weight", out,
                               config->dim, 1, &layer->out_proj,
                               error, error_size)) {
        qzen_zimage_final_layer_free(layer);
        return 0;
    }
    return 1;
}

void qzen_zimage_final_layer_free(qzen_zimage_final_layer *layer) {
    qzen_zimage_linear_free(&layer->adaLN);
    qzen_zimage_linear_free(&layer->out_proj);
}

int qzen_zimage_final_layer_forward(const qzen_zimage_final_layer *layer,
                                  const float *x, uint32_t rows,
                                  const float *adaln, float *output) {
    uint32_t dim = layer->dim;
    uint32_t embed = layer->adaLN.in;
    float *activated = malloc((size_t)embed * sizeof(float));
    float *scale = malloc((size_t)dim * sizeof(float));
    float *normed = malloc((size_t)rows * dim * sizeof(float));
    if (!activated || !scale || !normed) {
        free(activated);
        free(scale);
        free(normed);
        return 0;
    }
    /* FinalLayer.adaLN_modulation is Sequential(SiLU, Linear): the timestep
     * embedding is activated *before* the projection. Projecting the raw
     * embedding produced a scale that was far too large, so the sampler walked
     * away from the image every step and the latent never left the noise floor
     * (a plausible-looking velocity field, decoded into coloured static). */
    for (uint32_t c = 0; c < embed; c++) {
        float value = adaln[c];
        activated[c] = value / (1.0f + expf(-value));
    }
    qzen_zimage_linear_forward(&layer->adaLN, activated, 1, scale);
    free(activated);
    for (uint32_t c = 0; c < dim; c++) scale[c] += 1.0f;
    /* Affine-free LayerNorm, eps 1e-6, then elementwise scale. */
    float inverse_width = 1.0f / (float)dim;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            uint32_t row = (uint32_t)row64;
            const float *source = x + (size_t)row * dim;
            float mean = 0.0f;
            for (uint32_t c = 0; c < dim; c++) mean += source[c];
            mean *= inverse_width;
            float variance = 0.0f;
            for (uint32_t c = 0; c < dim; c++) {
                float centered = source[c] - mean;
                variance += centered * centered;
            }
            variance *= inverse_width;
            float inverse_std = 1.0f / sqrtf(variance + layer->norm_eps);
            for (uint32_t c = 0; c < dim; c++) {
                normed[(size_t)row * dim + c] =
                    (source[c] - mean) * inverse_std * scale[c];
            }
        }
    }
    qzen_zimage_linear_forward(&layer->out_proj, normed, rows, output);
    free(scale);
    free(normed);
    return 1;
}

/* --- GPU forward path -------------------------------------------------------- */

/* Widen a Linear's weight to F32 once so the per-step host-side adaLN GEMM is
 * pure F32 MACs (the CPU path keeps widening per element and is untouched).
 * Bit-identical accumulation order to qzen_zimage_linear_forward. */
static int zimg_linear_widen(qzen_zimage_linear *linear) {
    if (linear->weight_f32) return 1;
    const uint16_t *weight = qzen_zimage_linear_host_weight(linear);
    if (!weight) return 0;
    size_t count = (size_t)linear->out * linear->in;
    float *widened = malloc(count * sizeof(float));
    if (!widened) return 0;
    for (size_t index = 0; index < count; index++)
        widened[index] = zimg_bf16_to_f32(weight[index]);
    linear->weight_f32 = widened;
    return 1;
}

/* Single-row GEMM over a widened weight, OpenMP-parallel over the outputs. */
static void zimg_linear_forward_row1(const qzen_zimage_linear *linear,
                                     const float *x, float *out) {
    const float *weight = linear->weight_f32;
    uint32_t in = linear->in;
    uint32_t out_dim = linear->out;
    const float *bias = linear->bias;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)out_dim; row64++) {
            uint32_t row = (uint32_t)row64;
            const float *w = weight + (size_t)row * in;
            float sum = bias ? bias[row] : 0.0f;
            for (uint32_t c = 0; c < in; c++) sum += w[c] * x[c];
            out[row] = sum;
        }
    }
}

static uint16_t zimg_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* round to nearest even */
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Defined with the GPU context below; shared by the block and t_embedder
 * H2D/D2H round trips. */
static uint16_t *zgu_host_stage(qzen_zimage_gpu *zgu, size_t elements);

/* Download a BF16 tensor into an F32 host buffer, using the GPU context's
 * reusable staging buffer. */
static int zimg_read_bf16_f32(qzen_zimage_gpu *zgu,
                              const qzen_gpu_tensor *tensor, float *out,
                              size_t count) {
    uint16_t *staged = zgu_host_stage(zgu, count);
    if (!staged) return 0;
    int ok = qzen_gpu_tensor_read_bf16(tensor, staged, count);
    if (ok) {
        for (size_t index = 0; index < count; index++)
            out[index] = zimg_bf16_to_f32(staged[index]);
    }
    return ok;
}

struct qzen_zimage_gpu {
    qzen_gpu *gpu;
    qzen_zimage_config config;
    uint32_t seq; /* reserved rows for the [seq, ...] scratch tensors */
    qzen_gpu_tensor *normed;
    qzen_gpu_tensor *q;
    qzen_gpu_tensor *k;
    qzen_gpu_tensor *v;
    qzen_gpu_tensor *attn;
    qzen_gpu_tensor *branch;
    qzen_gpu_tensor *ffn_a;
    qzen_gpu_tensor *ffn_b;
    qzen_gpu_tensor *img;
    qzen_gpu_tensor *cap;
    qzen_gpu_tensor *full;
    qzen_gpu_tensor *rope_cos; /* F32 [seq, head_dim / 2] */
    qzen_gpu_tensor *rope_sin;
    qzen_gpu_tensor *rope_cos_cap; /* F32 [seq, head_dim / 2], caption rows */
    qzen_gpu_tensor *rope_sin_cap;
    qzen_gpu_tensor *out64; /* BF16 [seq, 64] */
    qzen_gpu_tensor *row_map; /* U32 [seq], all zero */
    qzen_gpu_tensor *modulation; /* BF16 [QZEN_ZIMAGE_MOD_SLOTS, dim] */
    qzen_gpu_tensor *zero; /* BF16 [seq, dim] */
    qzen_gpu_tensor *final_scale; /* BF16 [1, dim] */
    qzen_gpu_tensor *norm_ones; /* BF16 [dim] */
    qzen_gpu_tensor *norm_zeros; /* BF16 [dim] */
    /* One expanded Nunchaku matrix, grown to the largest layer and reused by
     * every block: 157 MB instead of the 10.6 GB of resident BF16 weights it
     * replaces (PLAN §19). */
    qzen_gpu_tensor *dequant;
    size_t dequant_elements;
    /* [rows, rank] scratch for the SVDQuant low-rank branch. */
    qzen_gpu_tensor *lora;
    size_t lora_elements;
    /* Reusable BF16 host staging for the per-block H2D/D2H round trips.
     * qzen_zimage_block_forward_gpu_host() is called ~50 times per ControlNet
     * denoise step; a fresh seq*dim malloc on every call fragments the CRT
     * heap next to the ~20 GiB of resident DiT + Union 2.1 weights and can
     * fail transiently with "cannot stage Z-Image GPU input". Grown only when
     * the reserved row count grows, never freed in the hot path. */
    uint16_t *host_stage;
    size_t host_stage_elements;
};

/* Grow the shared BF16 staging buffer to `elements`. All the GPU tensor
 * upload/read helpers below use synchronous cudaMemcpy, so the buffer can be
 * reused for an upload and then for a download within the same block. */
static uint16_t *zgu_host_stage(qzen_zimage_gpu *zgu, size_t elements) {
    if (!zgu || !elements) return NULL;
    if (elements <= zgu->host_stage_elements) return zgu->host_stage;
    uint16_t *grown = (uint16_t *)malloc(elements * sizeof(*grown));
    if (!grown) return NULL;
    free(zgu->host_stage);
    zgu->host_stage = grown;
    zgu->host_stage_elements = elements;
    return grown;
}

qzen_zimage_gpu *qzen_zimage_gpu_create(const qzen_zimage_config *config,
                                    char *error, size_t error_size) {
    qzen_zimage_gpu *zgu = calloc(1, sizeof(qzen_zimage_gpu));
    if (!zgu) {
        zimg_fail(error, error_size, "cannot allocate Z-Image GPU context");
        return NULL;
    }
    zgu->config = *config;
    zgu->gpu = qzen_gpu_create(NULL, error, error_size);
    if (!zgu->gpu) {
        free(zgu);
        return NULL;
    }
    uint32_t dim = config->dim;
    uint16_t *ones = malloc((size_t)dim * 2);
    uint16_t *zeros = malloc((size_t)dim * 2);
    if (!ones || !zeros) {
        free(ones); free(zeros);
        qzen_gpu_free(zgu->gpu);
        free(zgu);
        zimg_fail(error, error_size, "cannot stage Z-Image norm tensors");
        return NULL;
    }
    for (uint32_t c = 0; c < dim; c++) {
        ones[c] = zimg_f32_to_bf16(1.0f);
        zeros[c] = zimg_f32_to_bf16(0.0f);
    }
    zgu->norm_ones = qzen_gpu_tensor_from_bf16(zgu->gpu, ones, dim);
    zgu->norm_zeros = qzen_gpu_tensor_from_bf16(zgu->gpu, zeros, dim);
    zgu->modulation = qzen_gpu_tensor_new_bf16(
        zgu->gpu, (size_t)QZEN_ZIMAGE_MOD_SLOTS * dim);
    zgu->final_scale = qzen_gpu_tensor_new_bf16(zgu->gpu, dim);
    free(ones);
    free(zeros);
    if (!zgu->norm_ones || !zgu->norm_zeros || !zgu->modulation ||
        !zgu->final_scale) {
        zimg_fail(error, error_size,
                  "cannot allocate Z-Image GPU modulation tensors: %s",
                  qzen_gpu_error(zgu->gpu));
        qzen_zimage_gpu_free(zgu);
        return NULL;
    }
    return zgu;
}

void qzen_zimage_gpu_free(qzen_zimage_gpu *zgu) {
    if (!zgu) return;
    qzen_gpu_tensor_free(zgu->normed);
    qzen_gpu_tensor_free(zgu->q);
    qzen_gpu_tensor_free(zgu->k);
    qzen_gpu_tensor_free(zgu->v);
    qzen_gpu_tensor_free(zgu->attn);
    qzen_gpu_tensor_free(zgu->branch);
    qzen_gpu_tensor_free(zgu->ffn_a);
    qzen_gpu_tensor_free(zgu->ffn_b);
    qzen_gpu_tensor_free(zgu->img);
    qzen_gpu_tensor_free(zgu->cap);
    qzen_gpu_tensor_free(zgu->full);
    qzen_gpu_tensor_free(zgu->rope_cos);
    qzen_gpu_tensor_free(zgu->rope_sin);
    qzen_gpu_tensor_free(zgu->rope_cos_cap);
    qzen_gpu_tensor_free(zgu->rope_sin_cap);
    qzen_gpu_tensor_free(zgu->out64);
    qzen_gpu_tensor_free(zgu->row_map);
    qzen_gpu_tensor_free(zgu->modulation);
    qzen_gpu_tensor_free(zgu->zero);
    qzen_gpu_tensor_free(zgu->final_scale);
    qzen_gpu_tensor_free(zgu->norm_ones);
    qzen_gpu_tensor_free(zgu->norm_zeros);
    if (zgu->dequant) qzen_gpu_tensor_free(zgu->dequant);
    zgu->dequant = NULL;
    zgu->dequant_elements = 0;
    if (zgu->lora) qzen_gpu_tensor_free(zgu->lora);
    zgu->lora = NULL;
    zgu->lora_elements = 0;
    free(zgu->host_stage);
    zgu->host_stage = NULL;
    zgu->host_stage_elements = 0;
    if (zgu->gpu) qzen_gpu_free(zgu->gpu);
    free(zgu);
}

qzen_gpu *qzen_zimage_gpu_handle(qzen_zimage_gpu *zgu) { return zgu->gpu; }

static void zimg_gpu_release_scratch(qzen_zimage_gpu *zgu) {
    qzen_gpu_tensor_free(zgu->normed);
    qzen_gpu_tensor_free(zgu->q);
    qzen_gpu_tensor_free(zgu->k);
    qzen_gpu_tensor_free(zgu->v);
    qzen_gpu_tensor_free(zgu->attn);
    qzen_gpu_tensor_free(zgu->branch);
    qzen_gpu_tensor_free(zgu->ffn_a);
    qzen_gpu_tensor_free(zgu->ffn_b);
    qzen_gpu_tensor_free(zgu->img);
    qzen_gpu_tensor_free(zgu->cap);
    qzen_gpu_tensor_free(zgu->full);
    qzen_gpu_tensor_free(zgu->rope_cos);
    qzen_gpu_tensor_free(zgu->rope_sin);
    qzen_gpu_tensor_free(zgu->rope_cos_cap);
    qzen_gpu_tensor_free(zgu->rope_sin_cap);
    qzen_gpu_tensor_free(zgu->out64);
    qzen_gpu_tensor_free(zgu->row_map);
    qzen_gpu_tensor_free(zgu->zero);
    zgu->normed = zgu->q = zgu->k = zgu->v = zgu->attn = zgu->branch = NULL;
    zgu->ffn_a = zgu->ffn_b = NULL;
    zgu->img = zgu->cap = zgu->full = NULL;
    zgu->rope_cos = zgu->rope_sin = NULL;
    zgu->rope_cos_cap = zgu->rope_sin_cap = NULL;
    zgu->out64 = NULL;
    zgu->row_map = zgu->zero = NULL;
    zgu->seq = 0;
}

int qzen_zimage_gpu_reserve(qzen_zimage_gpu *zgu, uint32_t seq,
                          char *error, size_t error_size) {
    const qzen_zimage_config *config = &zgu->config;
    uint32_t dim = config->dim;
    uint32_t half = config->head_dim / 2;
    if (seq <= zgu->seq) return 1;
    zimg_gpu_release_scratch(zgu);
    size_t plane = (size_t)seq * dim;
    size_t wide = (size_t)seq * config->ffn_dim;
    /* Allocate the host staging buffer once, up front, while the failure is
     * still attributable to scratch reservation rather than to a random
     * block forward in the middle of a denoise step. */
    if (!zgu_host_stage(zgu, plane)) {
        zimg_fail(error, error_size,
                  "cannot allocate Z-Image host staging buffer for %u rows "
                  "(%.1f MiB)", seq, (double)(plane * 2) / 1048576.0);
        return 0;
    }
    zgu->normed = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->q = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->k = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->v = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->attn = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->branch = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->ffn_a = qzen_gpu_tensor_new_bf16(zgu->gpu, wide);
    zgu->ffn_b = qzen_gpu_tensor_new_bf16(zgu->gpu, wide);
    zgu->img = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->cap = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->full = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    zgu->rope_cos = qzen_gpu_tensor_new_f32(zgu->gpu, (size_t)seq * half);
    zgu->rope_sin = qzen_gpu_tensor_new_f32(zgu->gpu, (size_t)seq * half);
    zgu->rope_cos_cap = qzen_gpu_tensor_new_f32(zgu->gpu, (size_t)seq * half);
    zgu->rope_sin_cap = qzen_gpu_tensor_new_f32(zgu->gpu, (size_t)seq * half);
    zgu->out64 = qzen_gpu_tensor_new_bf16(zgu->gpu, (size_t)seq * 64);
    zgu->zero = qzen_gpu_tensor_new_bf16(zgu->gpu, plane);
    if (!zgu->normed || !zgu->q || !zgu->k || !zgu->v || !zgu->attn ||
        !zgu->branch || !zgu->ffn_a || !zgu->ffn_b || !zgu->img ||
        !zgu->cap || !zgu->full || !zgu->rope_cos || !zgu->rope_sin ||
        !zgu->rope_cos_cap || !zgu->rope_sin_cap || !zgu->out64 ||
        !zgu->zero) {
        zimg_fail(error, error_size,
                  "cannot allocate Z-Image GPU scratch: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    /* The row map selects modulation row 0 for every token; the zero residual
     * backs the final layer's broadcast scale (gate with a zero residual). */
    uint32_t *map = calloc(seq, sizeof(uint32_t));
    uint16_t *zero_host = calloc(plane, 2);
    qzen_gpu_tensor *row_map =
        map ? qzen_gpu_tensor_from_u32(zgu->gpu, map, seq) : NULL;
    int zero_ok = zero_host &&
                  qzen_gpu_tensor_write_bf16(zgu->zero, zero_host, plane);
    free(map);
    free(zero_host);
    if (!row_map || !zero_ok) {
        if (row_map) qzen_gpu_tensor_free(row_map);
        zimg_fail(error, error_size,
                  "cannot initialize Z-Image GPU scratch: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    if (zgu->row_map) qzen_gpu_tensor_free(zgu->row_map);
    zgu->row_map = row_map;
    zgu->seq = seq;
    return 1;
}

qzen_gpu_tensor *qzen_zimage_gpu_img(qzen_zimage_gpu *zgu) { return zgu->img; }
qzen_gpu_tensor *qzen_zimage_gpu_cap(qzen_zimage_gpu *zgu) { return zgu->cap; }
qzen_gpu_tensor *qzen_zimage_gpu_full(qzen_zimage_gpu *zgu) { return zgu->full; }
qzen_gpu_tensor *qzen_zimage_gpu_rope_cos_cap(qzen_zimage_gpu *zgu) {
    return zgu->rope_cos_cap;
}
qzen_gpu_tensor *qzen_zimage_gpu_rope_sin_cap(qzen_zimage_gpu *zgu) {
    return zgu->rope_sin_cap;
}
qzen_gpu_tensor *qzen_zimage_gpu_out64(qzen_zimage_gpu *zgu) { return zgu->out64; }
qzen_gpu_tensor *qzen_zimage_gpu_modulation(qzen_zimage_gpu *zgu) {
    return zgu->modulation;
}

qzen_gpu_tensor *qzen_zimage_gpu_rope_cos(qzen_zimage_gpu *zgu) {
    return zgu->rope_cos;
}

qzen_gpu_tensor *qzen_zimage_gpu_rope_sin(qzen_zimage_gpu *zgu) {
    return zgu->rope_sin;
}

int qzen_zimage_block_modulation_gpu(const qzen_zimage_block *block,
                                   qzen_zimage_gpu *zgu, const float *adaln,
                                   char *error, size_t error_size) {
    if (!block->modulated) return 1;
    uint32_t dim = zgu->config.dim;
    float *mod_raw = malloc((size_t)4 * dim * sizeof(float));
    if (!mod_raw) {
        zimg_fail(error, error_size, "cannot allocate adaLN modulation buffer");
        return 0;
    }
    zimg_linear_forward_row1(&block->adaln, adaln, mod_raw);
    int ok = qzen_zimage_gpu_set_modulation(zgu, mod_raw, dim, error, error_size);
    free(mod_raw);
    return ok;
}

int qzen_zimage_gpu_set_rope(qzen_zimage_gpu *zgu, const float *cos,
                           const float *sin, uint32_t seq, uint32_t half,
                           char *error, size_t error_size) {
    if (seq > zgu->seq) {
        zimg_fail(error, error_size, "Z-Image GPU scratch not reserved");
        return 0;
    }
    if (!qzen_gpu_tensor_write_f32(zgu->rope_cos, cos, (size_t)seq * half) ||
        !qzen_gpu_tensor_write_f32(zgu->rope_sin, sin, (size_t)seq * half)) {
        zimg_fail(error, error_size, "cannot upload Z-Image RoPE tables: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    return 1;
}

int qzen_zimage_gpu_set_modulation(qzen_zimage_gpu *zgu, const float *mod_raw,
                                 uint32_t dim, char *error, size_t error_size) {
    uint16_t *staged = malloc((size_t)QZEN_ZIMAGE_MOD_SLOTS * dim * 2);
    if (!staged) {
        zimg_fail(error, error_size, "cannot stage Z-Image modulation");
        return 0;
    }
    for (uint32_t c = 0; c < dim; c++) {
        /* scale slots keep the raw value (the adaLN kernel adds 1 itself);
         * gate slots receive tanh; the shift slot stays zero. */
        staged[c] = zimg_f32_to_bf16(mod_raw[c]);
        staged[dim + c] = zimg_f32_to_bf16(tanhf(mod_raw[dim + c]));
        staged[2 * dim + c] = zimg_f32_to_bf16(mod_raw[2 * dim + c]);
        staged[3 * dim + c] = zimg_f32_to_bf16(tanhf(mod_raw[3 * dim + c]));
        staged[4 * dim + c] = 0;
    }
    int ok = qzen_gpu_tensor_write_bf16(
        zgu->modulation, staged, (size_t)QZEN_ZIMAGE_MOD_SLOTS * dim);
    free(staged);
    if (!ok) {
        zimg_fail(error, error_size, "cannot upload Z-Image modulation: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    return 1;
}

/* --- weight upload ---------------------------------------------------------- */

/* Upload a packed Nunchaku slice: five tensors instead of an expanded matrix.
 * codes, scales and proj_up are the three row-indexed tensors, so the Q/K
 * permutation copies those three; proj_down is [rank][in] and is not touched.
 * The permutation goes to temporaries so the host pack stays canonical. */
static int zimg_upload_packed(qzen_zimage_linear *linear, qzen_zimage_gpu *zgu,
                              int rope_permute, char *error,
                              size_t error_size) {
    const qzen_nunchaku_packed *packed = &linear->packed;
    const uint32_t out = packed->out;
    const uint32_t in = packed->in;
    const uint32_t rank = packed->rank;
    const uint32_t groups = packed->groups;
    const size_t row_bytes = in / 2u;
    const uint8_t *codes = packed->codes;
    const uint16_t *scales = packed->scales;
    const uint16_t *up = packed->proj_up;
    uint8_t *codes_staged = NULL;
    uint16_t *scales_staged = NULL;
    uint16_t *up_staged = NULL;

    if (rope_permute) {
        const uint32_t head_dim = zgu->config.head_dim;
        const uint32_t half = head_dim / 2u;
        if (!head_dim || out % head_dim) {
            zimg_fail(error, error_size,
                      "cannot permute a %u-row packed weight by %u", out,
                      head_dim);
            return 0;
        }
        codes_staged = malloc((size_t)out * row_bytes);
        scales_staged = malloc((size_t)out * groups * 2);
        up_staged = malloc((size_t)out * rank * 2);
        if (!codes_staged || !scales_staged || !up_staged) {
            zimg_fail(error, error_size,
                      "cannot stage the permuted packed Q/K weight");
            free(codes_staged);
            free(scales_staged);
            free(up_staged);
            return 0;
        }
        for (uint32_t o = 0; o < out; o++) {
            const uint32_t head = o / head_dim;
            const uint32_t d = o % head_dim;
            const uint32_t source_d = d < half ? 2u * d : 2u * (d - half) + 1u;
            const uint32_t from = head * head_dim + source_d;
            memcpy(codes_staged + (size_t)o * row_bytes,
                   packed->codes + (size_t)from * row_bytes, row_bytes);
            memcpy(scales_staged + (size_t)o * groups,
                   packed->scales + (size_t)from * groups,
                   (size_t)groups * 2);
            memcpy(up_staged + (size_t)o * rank,
                   packed->proj_up + (size_t)from * rank, (size_t)rank * 2);
        }
        codes = codes_staged;
        scales = scales_staged;
        up = up_staged;
    }

    int ok = 0;
    qzen_gpu_tensor *device_codes = qzen_gpu_tensor_new_i8(zgu->gpu,
                                                       (size_t)out * row_bytes);
    qzen_gpu_tensor *device_scales = NULL;
    qzen_gpu_tensor *device_smooth = NULL;
    qzen_gpu_tensor *device_up = NULL;
    qzen_gpu_tensor *device_down = NULL;
    if (!device_codes ||
        !qzen_gpu_tensor_write_i8(device_codes, (const int8_t *)codes,
                                (size_t)out * row_bytes))
        goto done;
    device_scales = qzen_gpu_tensor_from_bf16(zgu->gpu, scales,
                                            (size_t)out * groups);
    device_smooth = qzen_gpu_tensor_from_f32(zgu->gpu, packed->smooth, in);
    device_up = qzen_gpu_tensor_from_bf16(zgu->gpu, up, (size_t)out * rank);
    device_down = qzen_gpu_tensor_from_bf16(zgu->gpu, packed->proj_down,
                                          (size_t)rank * in);
    if (!device_scales || !device_smooth || !device_up || !device_down)
        goto done;
    if (linear->gpu_codes) qzen_gpu_tensor_free(linear->gpu_codes);
    if (linear->gpu_scales) qzen_gpu_tensor_free(linear->gpu_scales);
    if (linear->gpu_smooth) qzen_gpu_tensor_free(linear->gpu_smooth);
    if (linear->gpu_up) qzen_gpu_tensor_free(linear->gpu_up);
    if (linear->gpu_down) qzen_gpu_tensor_free(linear->gpu_down);
    linear->gpu_codes = device_codes;
    linear->gpu_scales = device_scales;
    linear->gpu_smooth = device_smooth;
    linear->gpu_up = device_up;
    linear->gpu_down = device_down;
    device_codes = NULL;
    device_scales = NULL;
    device_smooth = NULL;
    device_up = NULL;
    device_down = NULL;
    ok = 1;
done:
    free(codes_staged);
    free(scales_staged);
    free(up_staged);
    if (device_codes) qzen_gpu_tensor_free(device_codes);
    if (device_scales) qzen_gpu_tensor_free(device_scales);
    if (device_smooth) qzen_gpu_tensor_free(device_smooth);
    if (device_up) qzen_gpu_tensor_free(device_up);
    if (device_down) qzen_gpu_tensor_free(device_down);
    if (!ok)
        zimg_fail(error, error_size,
                  "cannot upload the packed Nunchaku weight: %s",
                  qzen_gpu_error(zgu->gpu));
    return ok;
}

/* Upload a BF16 linear, optionally permuting the output rows of every
 * head_dim-sized group (even dimensions first) so the stock half-split RoPE
 * kernel reproduces Z-Image's interleaved (2i, 2i+1) pairing. */
static int zimg_upload_linear(qzen_zimage_linear *linear, qzen_zimage_gpu *zgu,
                              int rope_permute, char *error,
                              size_t error_size) {
    uint32_t out = linear->out;
    uint32_t in = linear->in;
    uint16_t *staged = NULL;
    const uint16_t *source = linear->weight;
    if (linear->has_packed) {
        /* The codes go up as they are; only the Q/K row permutation has to be
         * applied, and it is applied to the copy on its way to the device so
         * that the host pack stays the canonical matrix the CPU path uses. */
        if (!zimg_upload_packed(linear, zgu, rope_permute, error, error_size))
            return 0;
    } else {
        if (rope_permute) {
            uint32_t head_dim = zgu->config.head_dim;
            uint32_t heads = out / head_dim;
            uint32_t half = head_dim / 2;
            staged = malloc((size_t)out * in * 2);
            if (!staged) {
                zimg_fail(error, error_size,
                          "cannot stage permuted Z-Image Q/K weight");
                return 0;
            }
            for (uint32_t head = 0; head < heads; head++) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    uint32_t source_d = d < half ? 2 * d : 2 * (d - half) + 1;
                    memcpy(staged + ((size_t)head * head_dim + d) * in,
                           linear->weight +
                               ((size_t)head * head_dim + source_d) * in,
                           (size_t)in * 2);
                }
            }
            source = staged;
        }
        if (linear->gpu_weight) qzen_gpu_tensor_free(linear->gpu_weight);
        linear->gpu_weight =
            qzen_gpu_tensor_from_bf16(zgu->gpu, source, (size_t)out * in);
        free(staged);
        if (!linear->gpu_weight) {
            zimg_fail(error, error_size,
                      "cannot upload Z-Image linear weight: %s",
                      qzen_gpu_error(zgu->gpu));
            return 0;
        }
    }
    if (linear->gpu_bias) {
        qzen_gpu_tensor_free(linear->gpu_bias);
        linear->gpu_bias = NULL;
    }
    if (linear->bias) {
        uint16_t *bias_bf16 = malloc((size_t)out * 2);
        if (!bias_bf16) {
            zimg_fail(error, error_size, "cannot stage Z-Image bias");
            return 0;
        }
        for (uint32_t o = 0; o < out; o++)
            bias_bf16[o] = zimg_f32_to_bf16(linear->bias[o]);
        linear->gpu_bias =
            qzen_gpu_tensor_from_bf16(zgu->gpu, bias_bf16, out);
        free(bias_bf16);
        if (!linear->gpu_bias) {
            zimg_fail(error, error_size,
                      "cannot upload Z-Image bias: %s",
                      qzen_gpu_error(zgu->gpu));
            return 0;
        }
    }
    return 1;
}

/* The one GEMM every block calls, whichever way the weight is stored: expand a
 * packed layer into the shared scratch matrix, run the same BF16 GEMM, then add
 * the low-rank branch as two GEMMs through the [rows, rank] scratch.
 *
 * The rank-256 branch is why it is not fused into the dequant kernel: doing it
 * there cost 5.02 s of DiT against 0.28 s for expanded weights (PLAN §19.2),
 * while two cuBLAS calls of [rows, in] x [in, rank] x [rank, out] are 1/out of
 * the main GEMM's work. */
static int zimg_linear_gpu(qzen_zimage_gpu *zgu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input,
                           const qzen_zimage_linear *linear, uint32_t rows) {
    if (!linear->has_packed)
        return qzen_gpu_linear_bf16(zgu->gpu, output, input, linear->gpu_weight,
                                  linear->gpu_bias, rows, linear->in,
                                  linear->out);
    const size_t count = (size_t)linear->out * linear->in;
    if (zgu->dequant_elements < count) {
        if (zgu->dequant) qzen_gpu_tensor_free(zgu->dequant);
        zgu->dequant = qzen_gpu_tensor_new_bf16(zgu->gpu, count);
        zgu->dequant_elements = zgu->dequant ? count : 0;
        if (!zgu->dequant) return 0;
    }
    if (!qzen_gpu_nunchaku_dequant(zgu->gpu, zgu->dequant, linear->gpu_codes,
                                 linear->gpu_scales, linear->gpu_smooth,
                                 linear->out, linear->in))
        return 0;
    if (!qzen_gpu_linear_bf16(zgu->gpu, output, input, zgu->dequant,
                            linear->gpu_bias, rows, linear->in, linear->out))
        return 0;
    const uint32_t rank = linear->packed.rank;
    if (!rank) return 1;
    const size_t lora_count = (size_t)rows * rank;
    if (zgu->lora_elements < lora_count) {
        if (zgu->lora) qzen_gpu_tensor_free(zgu->lora);
        zgu->lora = qzen_gpu_tensor_new_bf16(zgu->gpu, lora_count);
        zgu->lora_elements = zgu->lora ? lora_count : 0;
        if (!zgu->lora) return 0;
    }
    /* (x @ down^T) is [rows, rank]; (@ up^T) accumulates into the output. */
    return qzen_gpu_linear_bf16(zgu->gpu, zgu->lora, input, linear->gpu_down,
                              NULL, rows, linear->in, rank) &&
           qzen_gpu_linear_bf16_accum(zgu->gpu, output, zgu->lora,
                                    linear->gpu_up, rows, rank, linear->out);
}

/* Upload a per-head-dim norm weight with the same row permutation. */
static int zimg_upload_head_norm(qzen_zimage_rms *rms, qzen_zimage_gpu *zgu,
                                 char *error, size_t error_size) {
    uint32_t head_dim = zgu->config.head_dim;
    uint32_t half = head_dim / 2;
    uint16_t *staged = malloc((size_t)head_dim * 2);
    if (!staged) {
        zimg_fail(error, error_size, "cannot stage Z-Image head norm weight");
        return 0;
    }
    for (uint32_t d = 0; d < head_dim; d++) {
        uint32_t source_d = d < half ? 2 * d : 2 * (d - half) + 1;
        staged[d] = rms->weight[source_d];
    }
    if (rms->gpu_weight) qzen_gpu_tensor_free(rms->gpu_weight);
    rms->gpu_weight =
        qzen_gpu_tensor_from_bf16(zgu->gpu, staged, head_dim);
    free(staged);
    if (!rms->gpu_weight) {
        zimg_fail(error, error_size,
                  "cannot upload Z-Image head norm weight: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    return 1;
}

int qzen_zimage_linear_upload(qzen_zimage_linear *linear, qzen_zimage_gpu *zgu,
                            char *error, size_t error_size) {
    return zimg_upload_linear(linear, zgu, 0, error, error_size);
}

int qzen_zimage_rms_upload(qzen_zimage_rms *rms, qzen_zimage_gpu *zgu,
                         char *error, size_t error_size) {
    if (rms->gpu_weight) qzen_gpu_tensor_free(rms->gpu_weight);
    rms->gpu_weight =
        qzen_gpu_tensor_from_bf16(zgu->gpu, rms->weight, rms->width);
    if (!rms->gpu_weight) {
        zimg_fail(error, error_size,
                  "cannot upload Z-Image norm weight: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    return 1;
}

int qzen_zimage_block_upload(qzen_zimage_block *block, qzen_zimage_gpu *zgu,
                           char *error, size_t error_size) {
    /* to_q / to_k / norm_q / norm_k carry the RoPE layout permutation. */
    if (!zimg_upload_linear(&block->to_q, zgu, 1, error, error_size)) return 0;
    if (!zimg_upload_linear(&block->to_k, zgu, 1, error, error_size)) return 0;
    if (!zimg_upload_linear(&block->to_v, zgu, 0, error, error_size)) return 0;
    if (!zimg_upload_linear(&block->to_out, zgu, 0, error, error_size))
        return 0;
    if (!zimg_upload_linear(&block->w1, zgu, 0, error, error_size)) return 0;
    if (!zimg_upload_linear(&block->w2, zgu, 0, error, error_size)) return 0;
    if (!zimg_upload_linear(&block->w3, zgu, 0, error, error_size)) return 0;
    if (!zimg_upload_head_norm(&block->norm_q, zgu, error, error_size))
        return 0;
    if (!zimg_upload_head_norm(&block->norm_k, zgu, error, error_size))
        return 0;
    if (!qzen_zimage_rms_upload(&block->attn_norm1, zgu, error, error_size))
        return 0;
    if (!qzen_zimage_rms_upload(&block->attn_norm2, zgu, error, error_size))
        return 0;
    if (!qzen_zimage_rms_upload(&block->ffn_norm1, zgu, error, error_size))
        return 0;
    if (!qzen_zimage_rms_upload(&block->ffn_norm2, zgu, error, error_size))
        return 0;
    /* The adaLN modulation linear stays on the host: it is a single-row GEMM
     * whose output is packed on the host anyway.  Widen it once here so the
     * per-step GEMM is pure F32 MACs. */
    if (block->modulated && !zimg_linear_widen(&block->adaln)) {
        zimg_fail(error, error_size,
                  "cannot widen Z-Image adaLN weight: out of memory");
        return 0;
    }
    return 1;
}

int qzen_zimage_t_embedder_upload(qzen_zimage_t_embedder *embedder,
                                qzen_zimage_gpu *zgu, char *error,
                                size_t error_size) {
    if (!zimg_upload_linear(&embedder->fc1, zgu, 0, error, error_size))
        return 0;
    return zimg_upload_linear(&embedder->fc2, zgu, 0, error, error_size);
}

int qzen_zimage_final_layer_upload(qzen_zimage_final_layer *layer,
                                 qzen_zimage_gpu *zgu, char *error,
                                 size_t error_size) {
    if (!zimg_upload_linear(&layer->out_proj, zgu, 0, error, error_size))
        return 0;
    /* The adaLN linear stays on the host (widened for the per-step GEMM). */
    if (!zimg_linear_widen(&layer->adaLN)) {
        zimg_fail(error, error_size,
                  "cannot widen final-layer adaLN weight: out of memory");
        return 0;
    }
    return 1;
}

/* --- block forward ----------------------------------------------------------- */

#define ZIMG_OP(call)                                                          \
    do {                                                                       \
        if (!(call)) {                                                         \
            zimg_fail(error, error_size, "Z-Image GPU forward failed: %s",     \
                      qzen_gpu_error(zgu->gpu));                                 \
            return 0;                                                          \
        }                                                                      \
    } while (0)

int qzen_zimage_block_forward_gpu(const qzen_zimage_block *block,
                                qzen_zimage_gpu *zgu, qzen_gpu_tensor *x,
                                uint32_t seq, qzen_gpu_tensor *modulation,
                                qzen_gpu_tensor *rope_cos,
                                qzen_gpu_tensor *rope_sin, char *error,
                                size_t error_size) {
    const qzen_zimage_config *config = &zgu->config;
    uint32_t dim = config->dim;
    uint32_t ffn = config->ffn_dim;
    uint32_t heads = config->heads;
    uint32_t head_dim = config->head_dim;
    float eps = config->norm_eps;
    int modulated = block->modulated;
    qzen_gpu_tensor *normed = zgu->normed;
    qzen_gpu_tensor *q = zgu->q;
    qzen_gpu_tensor *k = zgu->k;
    qzen_gpu_tensor *v = zgu->v;
    qzen_gpu_tensor *attn = zgu->attn;
    qzen_gpu_tensor *branch = zgu->branch;

    if (modulated && !modulation) {
        zimg_fail(error, error_size, "modulated block requires an adaLN input");
        return 0;
    }

    /* Attention sub-block. */
    if (modulated) {
        ZIMG_OP(qzen_gpu_adaln_bf16(zgu->gpu, normed, x, block->attn_norm1.gpu_weight,
                                  modulation, zgu->row_map, seq, dim,
                                  QZEN_ZIMAGE_MOD_SLOTS, QZEN_ZIMAGE_MOD_SHIFT_SLOT,
                                  0, eps));
    } else {
        ZIMG_OP(qzen_gpu_rms_norm_bf16(zgu->gpu, normed, x,
                                     block->attn_norm1.gpu_weight, seq, dim,
                                     eps));
    }
    ZIMG_OP(zimg_linear_gpu(zgu, q, normed, &block->to_q, seq));
    ZIMG_OP(zimg_linear_gpu(zgu, k, normed, &block->to_k, seq));
    ZIMG_OP(zimg_linear_gpu(zgu, v, normed, &block->to_v, seq));
    /* QK RMSNorm over head_dim, then RoPE, both in the permuted layout. */
    ZIMG_OP(qzen_gpu_head_rms_norm_bf16(zgu->gpu, q, block->norm_q.gpu_weight,
                                      seq, heads, head_dim, eps));
    ZIMG_OP(qzen_gpu_head_rms_norm_bf16(zgu->gpu, k, block->norm_k.gpu_weight,
                                      seq, heads, head_dim, eps));
    ZIMG_OP(qzen_gpu_rope_text_bf16(zgu->gpu, q, k, rope_cos,
                                  rope_sin, seq, heads, heads, head_dim));
    ZIMG_OP(qzen_gpu_sdpa_bf16(zgu->gpu, attn, q, k, v, seq, heads, head_dim,
                             1.0f / sqrtf((float)head_dim)));
    ZIMG_OP(zimg_linear_gpu(zgu, branch, attn, &block->to_out, seq));
    ZIMG_OP(qzen_gpu_rms_norm_bf16(zgu->gpu, branch, branch,
                                 block->attn_norm2.gpu_weight, seq, dim, eps));
    if (modulated) {
        ZIMG_OP(qzen_gpu_gate_bf16(zgu->gpu, x, x, branch, modulation,
                                 zgu->row_map, seq, dim, QZEN_ZIMAGE_MOD_SLOTS,
                                 1));
    } else {
        ZIMG_OP(qzen_gpu_add_bf16(zgu->gpu, x, x, branch, (size_t)seq * dim));
    }

    /* FFN sub-block. */
    if (modulated) {
        ZIMG_OP(qzen_gpu_adaln_bf16(zgu->gpu, normed, x, block->ffn_norm1.gpu_weight,
                                  modulation, zgu->row_map, seq, dim,
                                  QZEN_ZIMAGE_MOD_SLOTS, QZEN_ZIMAGE_MOD_SHIFT_SLOT,
                                  2, eps));
    } else {
        ZIMG_OP(qzen_gpu_rms_norm_bf16(zgu->gpu, normed, x,
                                     block->ffn_norm1.gpu_weight, seq, dim,
                                     eps));
    }
    ZIMG_OP(zimg_linear_gpu(zgu, zgu->ffn_a, normed, &block->w1, seq));
    ZIMG_OP(zimg_linear_gpu(zgu, zgu->ffn_b, normed, &block->w3, seq));
    ZIMG_OP(qzen_gpu_silu_mul_bf16(zgu->gpu, zgu->ffn_a, zgu->ffn_a, zgu->ffn_b,
                                 (size_t)seq * ffn));
    ZIMG_OP(zimg_linear_gpu(zgu, branch, zgu->ffn_a, &block->w2, seq));
    ZIMG_OP(qzen_gpu_rms_norm_bf16(zgu->gpu, branch, branch,
                                 block->ffn_norm2.gpu_weight, seq, dim, eps));
    if (modulated) {
        ZIMG_OP(qzen_gpu_gate_bf16(zgu->gpu, x, x, branch, modulation,
                                 zgu->row_map, seq, dim, QZEN_ZIMAGE_MOD_SLOTS,
                                 3));
    } else {
        ZIMG_OP(qzen_gpu_add_bf16(zgu->gpu, x, x, branch, (size_t)seq * dim));
    }
    return 1;
}

int qzen_zimage_block_forward_gpu_host(const qzen_zimage_block *block,
                                     qzen_zimage_gpu *zgu, float *x,
                                     uint32_t seq, const float *adaln,
                                     const float *rope_cos,
                                     const float *rope_sin, char *error,
                                     size_t error_size) {
    const qzen_zimage_config *config = &zgu->config;
    uint32_t dim = config->dim;
    uint32_t half = config->head_dim / 2;
    size_t count = (size_t)seq * dim;
    if (!qzen_zimage_gpu_reserve(zgu, seq, error, error_size)) return 0;
    if (!qzen_zimage_gpu_set_rope(zgu, rope_cos, rope_sin, seq, half, error,
                                error_size))
        return 0;
    float *mod_raw = NULL;
    if (block->modulated) {
        if (!adaln) {
            zimg_fail(error, error_size,
                      "modulated block requires an adaLN input");
            return 0;
        }
        mod_raw = malloc((size_t)4 * dim * sizeof(float));
        if (!mod_raw) {
            zimg_fail(error, error_size, "cannot allocate adaLN modulation buffer");
            return 0;
        }
        zimg_linear_forward_row1(&block->adaln, adaln, mod_raw);
    }
    if (mod_raw && !qzen_zimage_gpu_set_modulation(zgu, mod_raw, dim, error,
                                                 error_size)) {
        free(mod_raw);
        return 0;
    }
    free(mod_raw);
    uint16_t *staged = zgu_host_stage(zgu, count);
    if (!staged) {
        zimg_fail(error, error_size,
                  "cannot stage Z-Image GPU input (%u rows x %u dim, %.1f MiB)",
                  seq, dim, (double)(count * 2) / 1048576.0);
        return 0;
    }
    for (size_t index = 0; index < count; index++)
        staged[index] = zimg_f32_to_bf16(x[index]);
    qzen_gpu_tensor *x_gpu =
        qzen_gpu_tensor_from_bf16(zgu->gpu, staged, count);
    if (!x_gpu) {
        zimg_fail(error, error_size, "cannot upload Z-Image GPU input: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    int ok = qzen_gpu_begin(zgu->gpu) &&
             qzen_zimage_block_forward_gpu(block, zgu, x_gpu, seq,
                                         block->modulated ? zgu->modulation
                                                          : NULL,
                                         zgu->rope_cos, zgu->rope_sin,
                                         error, error_size) &&
             zimg_read_bf16_f32(zgu, x_gpu, x, count) &&
             qzen_gpu_submit(zgu->gpu);
    qzen_gpu_tensor_free(x_gpu);
    if (!ok && (!error || !*error))
        zimg_fail(error, error_size, "Z-Image GPU block forward failed: %s",
                  qzen_gpu_error(zgu->gpu));
    return ok;
}

/* --- timestep embedder / final layer on the GPU ------------------------------ */

int qzen_zimage_t_embedder_forward_gpu(const qzen_zimage_t_embedder *embedder,
                                     qzen_zimage_gpu *zgu, const float *input,
                                     uint32_t rows, float *output,
                                     char *error, size_t error_size) {
    uint32_t mid = embedder->fc1.out;
    uint32_t out = embedder->fc2.out;
    size_t in_count = (size_t)rows * embedder->fc1.in;
    size_t mid_count = (size_t)rows * mid;
    size_t out_count = (size_t)rows * out;
    uint16_t *in_bf16 = zgu_host_stage(zgu, in_count);
    if (!in_bf16) {
        zimg_fail(error, error_size, "cannot stage t_embedder input");
        return 0;
    }
    for (size_t index = 0; index < in_count; index++)
        in_bf16[index] = zimg_f32_to_bf16(input[index]);
    qzen_gpu_tensor *input_gpu = qzen_gpu_tensor_from_bf16(zgu->gpu, in_bf16, in_count);
    qzen_gpu_tensor *hidden = qzen_gpu_tensor_new_bf16(zgu->gpu, mid_count);
    qzen_gpu_tensor *out_gpu = qzen_gpu_tensor_new_bf16(zgu->gpu, out_count);
    if (!input_gpu || !hidden || !out_gpu) {
        qzen_gpu_tensor_free(input_gpu);
        qzen_gpu_tensor_free(hidden);
        qzen_gpu_tensor_free(out_gpu);
        zimg_fail(error, error_size, "cannot allocate t_embedder GPU buffers");
        return 0;
    }
    int ok = qzen_gpu_begin(zgu->gpu) &&
             zimg_linear_gpu(zgu, hidden, input_gpu, &embedder->fc1, rows) &&
             qzen_gpu_silu_bf16(zgu->gpu, hidden, hidden, mid_count) &&
             zimg_linear_gpu(zgu, out_gpu, hidden, &embedder->fc2, rows) &&
             zimg_read_bf16_f32(zgu, out_gpu, output, out_count) &&
             qzen_gpu_submit(zgu->gpu);
    qzen_gpu_tensor_free(input_gpu);
    qzen_gpu_tensor_free(hidden);
    qzen_gpu_tensor_free(out_gpu);
    if (!ok && (!error || !*error))
        zimg_fail(error, error_size, "t_embedder GPU forward failed: %s",
                  qzen_gpu_error(zgu->gpu));
    return ok;
}

int qzen_zimage_final_layer_forward_gpu(const qzen_zimage_final_layer *layer,
                                      qzen_zimage_gpu *zgu, qzen_gpu_tensor *x,
                                      uint32_t rows, const float *adaln,
                                      qzen_gpu_tensor *output, char *error,
                                      size_t error_size) {
    uint32_t dim = layer->dim;
    /* The adaLN modulation stays a host GEMM (single row), matching the CPU
     * reference exactly; only the scale is uploaded. SiLU first, like
     * FinalLayer.adaLN_modulation = Sequential(SiLU, Linear). */
    float *scale = malloc((size_t)dim * sizeof(float));
    float *activated = malloc((size_t)layer->adaLN.in * sizeof(float));
    if (!scale || !activated) {
        free(scale);
        free(activated);
        zimg_fail(error, error_size, "cannot allocate final-layer scale");
        return 0;
    }
    for (uint32_t c = 0; c < layer->adaLN.in; c++) {
        float value = adaln[c];
        activated[c] = value / (1.0f + expf(-value));
    }
    zimg_linear_forward_row1(&layer->adaLN, activated, scale);
    free(activated);
    for (uint32_t c = 0; c < dim; c++) scale[c] += 1.0f;
    uint16_t *scale_bf16 = malloc((size_t)dim * 2);
    if (!scale_bf16) {
        free(scale);
        zimg_fail(error, error_size, "cannot stage final-layer scale");
        return 0;
    }
    for (uint32_t c = 0; c < dim; c++) scale_bf16[c] = zimg_f32_to_bf16(scale[c]);
    int ok = qzen_gpu_tensor_write_bf16(zgu->final_scale, scale_bf16, dim);
    free(scale_bf16);
    free(scale);
    if (!ok) {
        zimg_fail(error, error_size, "cannot upload final-layer scale: %s",
                  qzen_gpu_error(zgu->gpu));
        return 0;
    }
    /* Affine-free LayerNorm (ones/zeros affine), then the broadcast scale as a
     * gate against the zero residual, then the output projection. */
    ok = qzen_gpu_begin(zgu->gpu) &&
         qzen_gpu_layer_norm_bf16(zgu->gpu, zgu->branch, x, zgu->norm_ones,
                                zgu->norm_zeros, rows, dim,
                                layer->norm_eps) &&
         qzen_gpu_gate_bf16(zgu->gpu, zgu->branch, zgu->zero, zgu->branch,
                          zgu->final_scale, zgu->row_map, rows, dim, 1, 0) &&
         zimg_linear_gpu(zgu, output, zgu->branch, &layer->out_proj, rows) &&
         qzen_gpu_submit(zgu->gpu);    if (!ok && (!error || !*error))
        zimg_fail(error, error_size, "final layer GPU forward failed: %s",
                  qzen_gpu_error(zgu->gpu));
    return ok;
}
