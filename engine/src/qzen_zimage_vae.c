#include "qzen_zimage_vae.h"
#include "qzen_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* --- helpers ---------------------------------------------------------------- */

static void zvae_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static const char *zvae_trace;
#define ZVAE_TRACE_LOG(...)                                              \
    do {                                                                 \
        if (zvae_trace) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } \
    } while (0)

static float *zvae_alloc(size_t elements) {
    return malloc(elements * sizeof(float));
}

/* Read a BF16 tensor with any of the supported shapes into F32 (row major). */
static int zvae_read_bf16_f32(const qzen_weight_store *store, const char *name,
                              float **out, uint64_t *elements,
                              char *error, size_t error_size) {
    const qzen_st_header *header = NULL;
    ZVAE_TRACE_LOG("  find %s\n", name);
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    ZVAE_TRACE_LOG("  found %p\n", (const void *)tensor);
    if (!tensor) {
        zvae_fail(error, error_size, "required VAE weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype != QZEN_DTYPE_BF16) {
        zvae_fail(error, error_size, "VAE weight %s is %s, expected BF16", name,
                  qzen_dtype_name(tensor->dtype));
        return 0;
    }
    uint64_t count = qzen_st_tensor_elements(tensor);
    ZVAE_TRACE_LOG("  read %s (%llu elements)\n", name, (unsigned long long)count);
    uint16_t *staged = malloc((size_t)count * 2);
    if (!staged) {
        zvae_fail(error, error_size, "cannot stage %s", name);
        return 0;
    }
    if (!qzen_st_read_data(header, tensor, staged, (size_t)count * 2, error, error_size)) {
        free(staged);
        return 0;
    }
    float *values = zvae_alloc((size_t)count);
    if (!values) {
        free(staged);
        zvae_fail(error, error_size, "cannot allocate %s", name);
        return 0;
    }
    for (uint64_t index = 0; index < count; index++) {
        uint32_t bits = (uint32_t)staged[index] << 16;
        memcpy(&values[index], &bits, sizeof(float));
    }
    free(staged);
    ZVAE_TRACE_LOG("  read done %s\n", name);
    *out = values;
    *elements = count;
    return 1;
}

/* --- weight structs --------------------------------------------------------- */

typedef struct {
    uint32_t out;
    uint32_t in;
    float *weight;   /* [out, in, 3, 3] */
    float *bias;     /* [out] */
} zvae_conv3x3;

typedef struct {
    uint32_t channels;
    float *weight;   /* [channels] */
    float *bias;     /* [channels] */
} zvae_norm;

typedef struct {
    zvae_norm norm1;
    zvae_conv3x3 conv1;
    zvae_norm norm2;
    zvae_conv3x3 conv2;
    int has_shortcut;
    zvae_conv3x3 shortcut;   /* 1x1: weight [out, in] */
} zvae_resnet;

typedef struct {
    uint32_t count;          /* 3 resnets per up block */
    zvae_resnet *resnets;
    int has_upsample;
    zvae_conv3x3 upsample;
} zvae_up_block;

typedef struct {
    zvae_resnet resnets[2];
    zvae_norm attn_norm;
    float *to_q, *to_q_bias, *to_k, *to_k_bias, *to_v, *to_v_bias;
    float *to_out, *to_out_bias;   /* [512, 512] + bias */
} zvae_mid;

/* Single-head spatial self-attention parameters.  The decoder and encoder mid
 * blocks share one numerical kernel (zvae_attention_forward); only these
 * tensors differ between them. */
typedef struct {
    zvae_norm norm;                 /* GroupNorm weight/bias [channels] */
    float *to_q, *to_q_bias, *to_k, *to_k_bias, *to_v, *to_v_bias;
    float *to_out, *to_out_bias;    /* [channels, channels] + bias */
} zvae_attn;

typedef struct {
    zvae_resnet resnets[2];         /* 2 resnets per down block */
    int has_downsample;
    zvae_conv3x3 downsample;        /* 3x3 stride-2 conv, present for b = 0..2 */
} zvae_down_block;

/* Lazily loaded encoder half of the AutoencoderKL (ControlNet guide images).
 * Stays zeroed until the first qzen_zvae_encode(). */
typedef struct {
    zvae_conv3x3 conv_in;           /* 3 -> ch[0] */
    zvae_down_block down[4];
    zvae_resnet mid_resnets[2];     /* ch[3] -> ch[3] */
    zvae_attn mid_attn;
    zvae_norm conv_norm_out;        /* GroupNorm(32, ch[3]) */
    zvae_conv3x3 conv_out;          /* ch[3] -> 2 * latent_channels */
    int loaded;
} zvae_encoder;

typedef struct zvae_gpu zvae_gpu;

struct qzen_zvae {
    qzen_zvae_config config;
    zvae_conv3x3 conv_in;
    zvae_mid mid;
    zvae_up_block up_blocks[4];
    zvae_norm norm_out;
    zvae_conv3x3 conv_out;
    zvae_gpu *gpu;   /* device weight mirror + CUDA context, built lazily */
    char *source_path;    /* path passed to qzen_zvae_load, for lazy encoder load */
    zvae_encoder encoder; /* lazily loaded ControlNet encoder */
};

qzen_zvae_config qzen_zvae_default_config(void) {
    qzen_zvae_config config;
    config.latent_channels = 16;
    config.out_channels = 3;
    config.up_channels[0] = 512;
    config.up_channels[1] = 512;
    config.up_channels[2] = 256;
    config.up_channels[3] = 128;
    config.layers_per_block = 2;
    config.groups = 32;
    config.eps = 1e-6f;
    config.scaling_factor = 0.3611f;
    config.shift_factor = 0.1159f;
    return config;
}

/* --- loading ---------------------------------------------------------------- */

/* Which decoder to use, from ZIMAGE_VAE_GPU.  See zvae_path_mode below; the
 * device implementation lives at the end of this file. */
typedef enum { ZVAE_PATH_AUTO, ZVAE_PATH_CPU, ZVAE_PATH_DEVICE } zvae_path;

static zvae_path zvae_path_mode(void);
zvae_gpu *zvae_gpu_create(const qzen_zvae *vae, char *error, size_t error_size);
void zvae_gpu_free(zvae_gpu *state);
static int zvae_decode_gpu(const qzen_zvae *vae, zvae_gpu *state,
                           const float *latents, uint32_t latent_h,
                           uint32_t latent_w, float *pixels, char *error,
                           size_t error_size);

static void zvae_conv_free(zvae_conv3x3 *conv) {
    free(conv->weight);
    free(conv->bias);
    conv->weight = conv->bias = NULL;
}

static int zvae_conv_load(const qzen_weight_store *store, const char *prefix,
                          uint32_t out, uint32_t in, int kernel,
                          zvae_conv3x3 *conv, char *error, size_t error_size) {
    char name[256];
    ZVAE_TRACE_LOG(" conv_load %s\n", prefix);
    conv->out = out;
    conv->in = in;
    conv->weight = conv->bias = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint64_t elements = 0;
    if (!zvae_read_bf16_f32(store, name, &conv->weight, &elements, error, error_size)) return 0;
    if (elements != (uint64_t)out * in * kernel * kernel) {
        zvae_fail(error, error_size, "%s has %llu elements, expected [%u, %u, %d, %d]",
                  name, (unsigned long long)elements, out, in, kernel, kernel);
        return 0;
    }
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (!zvae_read_bf16_f32(store, name, &conv->bias, &elements, error, error_size)) return 0;
    if (elements != out) {
        zvae_fail(error, error_size, "%s has %llu elements, expected [%u]",
                  name, (unsigned long long)elements, out);
        return 0;
    }
    return 1;
}

static int zvae_norm_load(const qzen_weight_store *store, const char *prefix,
                          uint32_t channels, zvae_norm *norm,
                          char *error, size_t error_size) {
    char name[256];
    norm->channels = channels;
    norm->weight = norm->bias = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint64_t elements = 0;
    if (!zvae_read_bf16_f32(store, name, &norm->weight, &elements, error, error_size)) return 0;
    if (elements != channels) {
        zvae_fail(error, error_size, "%s.weight has %llu elements, expected [%u]",
                  prefix, (unsigned long long)elements, channels);
        return 0;
    }
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (!zvae_read_bf16_f32(store, name, &norm->bias, &elements, error, error_size)) return 0;
    if (elements != channels) {
        zvae_fail(error, error_size, "%s.bias has %llu elements, expected [%u]",
                  prefix, (unsigned long long)elements, channels);
        return 0;
    }
    return 1;
}

static void zvae_resnet_free(zvae_resnet *resnet) {
    free(resnet->norm1.weight); free(resnet->norm1.bias);
    zvae_conv_free(&resnet->conv1);
    free(resnet->norm2.weight); free(resnet->norm2.bias);
    zvae_conv_free(&resnet->conv2);
    zvae_conv_free(&resnet->shortcut);
}

/* Resnets are "resnets.<j>"; the 1x1 shortcut exists only when the channel
 * count changes (checked by presence of conv_shortcut.weight). */
static int zvae_resnet_load(const qzen_weight_store *store, const char *prefix,
                            uint32_t in_channels, uint32_t out_channels,
                            const qzen_zvae_config *config, zvae_resnet *resnet,
                            char *error, size_t error_size) {
    char name[256];
    memset(resnet, 0, sizeof(*resnet));
    snprintf(name, sizeof(name), "%s.norm1", prefix);
    if (!zvae_norm_load(store, name, in_channels, &resnet->norm1, error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    if (!zvae_conv_load(store, name, out_channels, in_channels, 3, &resnet->conv1, error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s.norm2", prefix);
    if (!zvae_norm_load(store, name, out_channels, &resnet->norm2, error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    if (!zvae_conv_load(store, name, out_channels, out_channels, 3, &resnet->conv2, error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s.conv_shortcut.weight", prefix);
    resnet->has_shortcut = 0;
    if (qzen_weight_find(store, name, NULL)) {
        resnet->has_shortcut = 1;
        snprintf(name, sizeof(name), "%s.conv_shortcut", prefix);
        if (!zvae_conv_load(store, name, out_channels, in_channels, 1, &resnet->shortcut,
                            error, error_size)) return 0;
    }
    (void)config;
    return 1;
}

/* Free the lazily loaded encoder weights; safe on a zeroed struct. */
static void zvae_encoder_free(zvae_encoder *enc) {
    zvae_conv_free(&enc->conv_in);
    for (uint32_t b = 0; b < 4; b++) {
        for (uint32_t r = 0; r < 2; r++) {
            zvae_resnet_free(&enc->down[b].resnets[r]);
        }
        zvae_conv_free(&enc->down[b].downsample);
    }
    for (uint32_t r = 0; r < 2; r++) zvae_resnet_free(&enc->mid_resnets[r]);
    free(enc->mid_attn.norm.weight); free(enc->mid_attn.norm.bias);
    free(enc->mid_attn.to_q);       free(enc->mid_attn.to_q_bias);
    free(enc->mid_attn.to_k);       free(enc->mid_attn.to_k_bias);
    free(enc->mid_attn.to_v);       free(enc->mid_attn.to_v_bias);
    free(enc->mid_attn.to_out);     free(enc->mid_attn.to_out_bias);
    free(enc->conv_norm_out.weight); free(enc->conv_norm_out.bias);
    zvae_conv_free(&enc->conv_out);
    memset(enc, 0, sizeof(*enc));
}

void qzen_zvae_free(qzen_zvae *vae) {
    if (!vae) return;
    zvae_gpu_free(vae->gpu);
    zvae_encoder_free(&vae->encoder);
    free(vae->source_path);
    zvae_conv_free(&vae->conv_in);
    for (uint32_t r = 0; r < 2; r++) zvae_resnet_free(&vae->mid.resnets[r]);
    free(vae->mid.attn_norm.weight); free(vae->mid.attn_norm.bias);
    free(vae->mid.to_q); free(vae->mid.to_q_bias);
    free(vae->mid.to_k); free(vae->mid.to_k_bias);
    free(vae->mid.to_v); free(vae->mid.to_v_bias);
    free(vae->mid.to_out); free(vae->mid.to_out_bias);
    for (uint32_t b = 0; b < 4; b++) {
        for (uint32_t r = 0; r < vae->up_blocks[b].count; r++) {
            zvae_resnet_free(&vae->up_blocks[b].resnets[r]);
        }
        free(vae->up_blocks[b].resnets);
        zvae_conv_free(&vae->up_blocks[b].upsample);
    }
    free(vae->norm_out.weight); free(vae->norm_out.bias);
    zvae_conv_free(&vae->conv_out);
    free(vae);
}

qzen_zvae *qzen_zvae_load(const char *directory, const qzen_zvae_config *config,
                      char *error, size_t error_size) {
    zvae_trace = getenv("ZVAE_TRACE");
    ZVAE_TRACE_LOG("open store\n");
    qzen_weight_store *store = qzen_weight_store_open(directory, error, error_size);
    if (!store) return NULL;
    qzen_zvae *vae = calloc(1, sizeof(qzen_zvae));
    if (!vae) {
        qzen_weight_store_free(store);
        zvae_fail(error, error_size, "cannot allocate qzen_zvae");
        return NULL;
    }
    vae->config = config ? *config : qzen_zvae_default_config();
    /* Remember the source so the encoder can be loaded lazily from the same
     * store on the first encode.  qzen_weight_store_open() accepts a shard
     * directory or a single .safetensors file, so re-opening this exact path
     * finds the encoder tensors.  A malloc failure here must not break the
     * (much more common) decode path, so it is non-fatal. */
    if (directory && *directory) {
        size_t length = strlen(directory) + 1;
        vae->source_path = malloc(length);
        if (vae->source_path) memcpy(vae->source_path, directory, length);
    }
    ZVAE_TRACE_LOG("conv_in\n");
    uint32_t latent = vae->config.latent_channels;
    uint32_t out0 = vae->config.up_channels[0];
    int ok =
        zvae_conv_load(store, "decoder.conv_in", out0, latent, 3, &vae->conv_in, error, error_size);
    ZVAE_TRACE_LOG("mid resnets\n");
    /* mid block */
    for (uint32_t r = 0; ok && r < 2; r++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "decoder.mid_block.resnets.%u", r);
        ok = zvae_resnet_load(store, prefix, out0, out0, config, &vae->mid.resnets[r], error, error_size);
    }
    if (ok) {
        ok = zvae_norm_load(store, "decoder.mid_block.attentions.0.group_norm",
                            out0, &vae->mid.attn_norm, error, error_size);
    }
    ZVAE_TRACE_LOG("mid attention\n");
    static const char *qkv[4] = {"to_q", "to_k", "to_v", "to_out.0"};
    float **mats[4] = {&vae->mid.to_q, &vae->mid.to_k, &vae->mid.to_v, &vae->mid.to_out};
    float **bies[4] = {&vae->mid.to_q_bias, &vae->mid.to_k_bias, &vae->mid.to_v_bias, &vae->mid.to_out_bias};
    for (uint32_t m = 0; ok && m < 4; m++) {
        char prefix[96];
        snprintf(prefix, sizeof(prefix), "decoder.mid_block.attentions.0.%s", qkv[m]);
        zvae_conv3x3 loaded;
        memset(&loaded, 0, sizeof(loaded));
        ok = zvae_conv_load(store, prefix, out0, out0, 1, &loaded, error, error_size);
        if (ok) {
            *mats[m] = loaded.weight;
            *bies[m] = loaded.bias;
        }
    }
    ZVAE_TRACE_LOG("up blocks\n");
    /* up blocks */
    for (uint32_t b = 0; ok && b < 4; b++) {
        uint32_t out_ch = vae->config.up_channels[b];
        uint32_t in_ch = b == 0 ? out0 : vae->config.up_channels[b - 1];
        zvae_up_block *block = &vae->up_blocks[b];
        block->count = vae->config.layers_per_block + 1;
        block->resnets = calloc(block->count, sizeof(zvae_resnet));
        if (!block->resnets) {
            zvae_fail(error, error_size, "cannot allocate up block %u resnets", b);
            ok = 0;
            break;
        }
        for (uint32_t j = 0; ok && j < block->count; j++) {
            uint32_t resnet_in = j == 0 ? in_ch : out_ch;
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "decoder.up_blocks.%u.resnets.%u", b, j);
            ok = zvae_resnet_load(store, prefix, resnet_in, out_ch, config,
                                  &block->resnets[j], error, error_size);
        }
        if (ok && b < 3) {
            char prefix[64];
            block->has_upsample = 1;
            snprintf(prefix, sizeof(prefix), "decoder.up_blocks.%u.upsamplers.0.conv", b);
            ok = zvae_conv_load(store, prefix, out_ch, out_ch, 3, &block->upsample, error, error_size);
        }
    }
    ZVAE_TRACE_LOG("conv_norm_out\n");
    if (ok) ok = zvae_norm_load(store, "decoder.conv_norm_out",
                                vae->config.up_channels[3], &vae->norm_out, error, error_size);
    if (ok) ok = zvae_conv_load(store, "decoder.conv_out", vae->config.out_channels,
                                vae->config.up_channels[3], 3, &vae->conv_out, error, error_size);
    qzen_weight_store_free(store);
    if (!ok) {
        qzen_zvae_free(vae);
        return NULL;
    }
    return vae;
}

/* --- math kernels ----------------------------------------------------------- */

/* SiLU in place. */
static void zvae_silu(float *values, size_t count) {
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)count; index++) {
            float value = values[index];
            values[index] = value / (1.0f + expf(-value));
        }
    }
}

/* GroupNorm over [C, H, W]: nn.GroupNorm statistics per (sample, group),
 * i.e. over group channels x H x W, with per-channel weight/bias. */
static void zvae_group_norm(const zvae_norm *norm, uint32_t groups, float eps,
                            float *data, uint32_t channels, uint32_t height,
                            uint32_t width) {
    size_t plane = (size_t)height * width;
    uint32_t per_group = channels / groups;
    {
        int64_t group;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (group = 0; group < (int64_t)groups; group++) {
            uint32_t c_begin = (uint32_t)group * per_group;
            double sum = 0.0;
            double sum_sq = 0.0;
            for (uint32_t c = c_begin; c < c_begin + per_group; c++) {
                const float *channel = data + (size_t)c * plane;
                for (size_t index = 0; index < plane; index++) {
                    double value = channel[index];
                    sum += value;
                    sum_sq += value * value;
                }
            }
            double count = (double)per_group * plane;
            float mean = (float)(sum / count);
            float variance = (float)(sum_sq / count - (double)mean * mean);
            float inverse = 1.0f / sqrtf(variance + eps);
            for (uint32_t c = c_begin; c < c_begin + per_group; c++) {
                float *channel = data + (size_t)c * plane;
                float weight = norm->weight[c];
                float bias = norm->bias[c];
                for (size_t index = 0; index < plane; index++) {
                    channel[index] = (channel[index] - mean) * inverse * weight + bias;
                }
            }
        }
    }
}

/* 3x3 stride-1 pad-1 convolution, in/out [C, H, W].  Parallel over output
 * channels; the inner loops walk contiguous input/output rows so gcc can
 * vectorize the horizontal stencil.  input and output must not alias. */
static void zvae_conv3x3_forward(const zvae_conv3x3 *conv, const float *input,
                                 float *output, uint32_t height, uint32_t width) {
    size_t plane = (size_t)height * width;
    const uint32_t in_ch = conv->in;
    const uint32_t out_ch = conv->out;
    {
        int64_t o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (o = 0; o < (int64_t)out_ch; o++) {
            float *out_plane = output + (size_t)o * plane;
            const float bias = conv->bias[o];
            for (size_t index = 0; index < plane; index++) out_plane[index] = bias;
            const float *weight = conv->weight + (size_t)o * in_ch * 9;
            for (uint32_t c = 0; c < in_ch; c++) {
                const float *in_plane = input + (size_t)c * plane;
                const float *w = weight + (size_t)c * 9;
                for (uint32_t ky = 0; ky < 3; ky++) {
                    int32_t y_offset = (int32_t)ky - 1;
                    float w0 = w[ky * 3 + 0], w1 = w[ky * 3 + 1], w2 = w[ky * 3 + 2];
                    uint32_t y_begin = ky == 0 ? 1 : 0;
                    uint32_t y_end = ky == 2 ? height - 1 : height;
                    for (uint32_t y = y_begin; y < y_end; y++) {
                        const float *in_row =
                            in_plane + (size_t)((int32_t)y + y_offset) * width;
                        float *out_row = out_plane + (size_t)y * width;
                        out_row[0] += w1 * in_row[0] + w2 * in_row[1];
                        for (uint32_t x = 1; x < width - 1; x++) {
                            out_row[x] += w0 * in_row[x - 1] + w1 * in_row[x] + w2 * in_row[x + 1];
                        }
                        if (width > 1) {
                            out_row[width - 1] += w0 * in_row[width - 2] + w1 * in_row[width - 1];
                        }
                    }
                }
            }
        }
    }
}

/* 3x3 stride-2 pad-1 convolution: [in, H, W] -> [out, H/2, W/2].  Output
 * (o, y, x) reads input (c, 2y+ky-1, 2x+kx-1) for ky,kx in 0..2, treating
 * out-of-range input coordinates as zero (standard PyTorch Conv2d stride=2
 * padding=1 for even H/W).  input and output must not alias. */
static void zvae_conv3x3_stride2_forward(const zvae_conv3x3 *conv,
                                         const float *input, float *output,
                                         uint32_t height, uint32_t width) {
    const uint32_t in_ch = conv->in;
    const uint32_t out_ch = conv->out;
    const uint32_t out_h = height / 2;
    const uint32_t out_w = width / 2;
    const size_t in_plane = (size_t)height * width;
    const size_t out_plane = (size_t)out_h * out_w;
    {
        int64_t o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (o = 0; o < (int64_t)out_ch; o++) {
            float *out_p = output + (size_t)o * out_plane;
            const float bias = conv->bias[o];
            for (size_t index = 0; index < out_plane; index++) out_p[index] = bias;
            const float *weight = conv->weight + (size_t)o * in_ch * 9;
            for (uint32_t c = 0; c < in_ch; c++) {
                const float *in_p = input + (size_t)c * in_plane;
                const float *w = weight + (size_t)c * 9;
                for (uint32_t ky = 0; ky < 3; ky++) {
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        const float wk = w[ky * 3 + kx];
                        for (uint32_t y = 0; y < out_h; y++) {
                            int32_t iy = (int32_t)(2 * y) + (int32_t)ky - 1;
                            if (iy < 0 || iy >= (int32_t)height) continue;
                            const float *in_row = in_p + (size_t)iy * width;
                            float *out_row = out_p + (size_t)y * out_w;
                            for (uint32_t x = 0; x < out_w; x++) {
                                int32_t ix = (int32_t)(2 * x) + (int32_t)kx - 1;
                                if (ix < 0 || ix >= (int32_t)width) continue;
                                out_row[x] += wk * in_row[ix];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* 1x1 convolution (resnet shortcuts), in/out [C, H, W], no aliasing. */
static void zvae_conv1x1_forward(const zvae_conv3x3 *conv, const float *input,
                                 float *output, uint32_t height, uint32_t width) {
    size_t plane = (size_t)height * width;
    const uint32_t in_ch = conv->in;
    const uint32_t out_ch = conv->out;
    {
        int64_t o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (o = 0; o < (int64_t)out_ch; o++) {
            const float *w = conv->weight + (size_t)o * in_ch;
            float bias = conv->bias[o];
            float *out_plane = output + (size_t)o * plane;
            for (size_t index = 0; index < plane; index++) {
                float sum = bias;
                for (uint32_t c = 0; c < in_ch; c++) {
                    sum += w[c] * input[(size_t)c * plane + index];
                }
                out_plane[index] = sum;
            }
        }
    }
}

/* ResnetBlock2D without timestep embedding: gn1 -> silu -> conv1 -> gn2 ->
 * silu -> conv2 -> (+ shortcut), output_scale_factor 1.0.  hidden holds
 * max(in, out) planes, scratch/output hold out_ch planes; none alias. */
static void zvae_resnet_forward(const zvae_resnet *resnet, const float *input,
                                float *hidden, float *scratch, float *output,
                                uint32_t groups, float eps, uint32_t height,
                                uint32_t width) {
    uint32_t in_ch = resnet->norm1.channels;
    uint32_t out_ch = resnet->norm2.channels;
    size_t plane = (size_t)height * width;
    memcpy(hidden, input, (size_t)in_ch * plane * sizeof(float));
    zvae_group_norm(&resnet->norm1, groups, eps, hidden, in_ch, height, width);
    zvae_silu(hidden, (size_t)in_ch * plane);
    zvae_conv3x3_forward(&resnet->conv1, hidden, scratch, height, width);
    zvae_group_norm(&resnet->norm2, groups, eps, scratch, out_ch, height, width);
    zvae_silu(scratch, (size_t)out_ch * plane);
    zvae_conv3x3_forward(&resnet->conv2, scratch, output, height, width);
    if (resnet->has_shortcut) {
        zvae_conv1x1_forward(&resnet->shortcut, input, hidden, height, width);
    }
    size_t count = (size_t)out_ch * plane;
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)count; index++) {
            output[index] += resnet->has_shortcut ? hidden[index] : input[index];
        }
    }
}

/* Mid-block spatial self-attention (one head over HW tokens): GroupNorm(32)
 * on [C, HW], to_q/k/v (bias), scaled dot-product with fp32 softmax, to_out,
 * residual connection to the un-normalized input. */
static int zvae_attention_forward(const zvae_attn *attn, uint32_t groups,
                                  float eps, const float *input, float *output,
                                  uint32_t height, uint32_t width) {
    uint32_t channels = attn->norm.channels;
    size_t tokens_count = (size_t)height * width;
    size_t plane = (size_t)channels * tokens_count;
    float *normed = zvae_alloc(plane);
    float *q = zvae_alloc(plane);
    float *k = zvae_alloc(plane);
    float *v = zvae_alloc(plane);
    float *attn_buf = zvae_alloc(plane);
    if (!normed || !q || !k || !v || !attn_buf) {
        free(normed); free(q); free(k); free(v); free(attn_buf);
        return 0;
    }
    memcpy(normed, input, plane * sizeof(float));
    zvae_group_norm(&attn->norm, groups, eps, normed, channels, height, width);
    /* q/k/v = W x normed per token: weight rows are [C], the normed column is
     * strided by tokens_count in the [C, HW] layout. */
    {
        int64_t i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (i = 0; i < (int64_t)tokens_count; i++) {
            float *q_row = q + (size_t)i * channels;
            float *k_row = k + (size_t)i * channels;
            float *v_row = v + (size_t)i * channels;
            for (uint32_t o = 0; o < channels; o++) {
                const float *wq = attn->to_q + (size_t)o * channels;
                const float *wk = attn->to_k + (size_t)o * channels;
                const float *wv = attn->to_v + (size_t)o * channels;
                float sq = 0.0f, sk = 0.0f, sv = 0.0f;
                for (uint32_t c = 0; c < channels; c++) {
                    float value = normed[(size_t)c * tokens_count + i];
                    sq += wq[c] * value;
                    sk += wk[c] * value;
                    sv += wv[c] * value;
                }
                q_row[o] = sq + attn->to_q_bias[o];
                k_row[o] = sk + attn->to_k_bias[o];
                v_row[o] = sv + attn->to_v_bias[o];
            }
        }
    }
    /* SDPA: one head, fp32 softmax (upcast_softmax=True is a no-op here),
     * per-thread score rows. */
    const float scale = 1.0f / sqrtf((float)channels);
    int attention_ok = 1;
    {
        int64_t i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&&: attention_ok)
#endif
        for (i = 0; i < (int64_t)tokens_count; i++) {
            float *scores = malloc(tokens_count * sizeof(float));
            if (!scores) { attention_ok = 0; continue; }
            const float *q_row = q + (size_t)i * channels;
            float maximum = -INFINITY;
            for (size_t j = 0; j < tokens_count; j++) {
                const float *k_row = k + j * channels;
                float dot = 0.0f;
                for (uint32_t c = 0; c < channels; c++) dot += q_row[c] * k_row[c];
                dot *= scale;
                scores[j] = dot;
                if (dot > maximum) maximum = dot;
            }
            float sum = 0.0f;
            for (size_t j = 0; j < tokens_count; j++) {
                scores[j] = expf(scores[j] - maximum);
                sum += scores[j];
            }
            float inverse = 1.0f / sum;
            float *out_row = attn_buf + (size_t)i * channels;
            memset(out_row, 0, channels * sizeof(float));
            for (size_t j = 0; j < tokens_count; j++) {
                const float *v_row = v + j * channels;
                float weight = scores[j] * inverse;
                for (uint32_t c = 0; c < channels; c++) out_row[c] += weight * v_row[c];
            }
            free(scores);
        }
    }
    /* to_out + residual, back to the [C, HW] layout. */
    {
        int64_t i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (i = 0; i < (int64_t)tokens_count; i++) {
            const float *attn_row = attn_buf + (size_t)i * channels;
            for (uint32_t o = 0; o < channels; o++) {
                const float *w = attn->to_out + (size_t)o * channels;
                float sum = attn->to_out_bias[o];
                for (uint32_t c = 0; c < channels; c++) sum += w[c] * attn_row[c];
                output[(size_t)o * tokens_count + i] = sum + input[(size_t)o * tokens_count + i];
            }
        }
    }
    free(normed); free(q); free(k); free(v); free(attn_buf);
    return attention_ok;
}

/* Decoder mid-block attention: wire the shared kernel to the decoder tensors.
 * The numerics are unchanged; only the parameter source moved. */
static int zvae_mid_attention(const qzen_zvae *vae, const float *input,
                              float *output, uint32_t height, uint32_t width) {
    zvae_attn attn;
    attn.norm = vae->mid.attn_norm;
    attn.to_q    = vae->mid.to_q;      attn.to_q_bias    = vae->mid.to_q_bias;
    attn.to_k    = vae->mid.to_k;      attn.to_k_bias    = vae->mid.to_k_bias;
    attn.to_v    = vae->mid.to_v;      attn.to_v_bias    = vae->mid.to_v_bias;
    attn.to_out  = vae->mid.to_out;    attn.to_out_bias  = vae->mid.to_out_bias;
    return zvae_attention_forward(&attn, vae->config.groups, vae->config.eps,
                                  input, output, height, width);
}

/* Nearest-neighbour x2 upsampling, [C, H, W] -> [C, 2H, 2W]. */
static void zvae_upsample_nearest(const float *input, float *output,
                                  uint32_t channels, uint32_t height,
                                  uint32_t width) {
    size_t out_plane = (size_t)2 * height * 2 * width;
    {
        int64_t c;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (c = 0; c < (int64_t)channels; c++) {
            const float *in_plane = input + (size_t)c * height * width;
            float *out_plane_ptr = output + (size_t)c * out_plane;
            for (uint32_t y = 0; y < height; y++) {
                const float *in_row = in_plane + (size_t)y * width;
                float *out_row0 = out_plane_ptr + (size_t)(2 * y) * (2 * width);
                float *out_row1 = out_row0 + 2 * width;
                for (uint32_t x = 0; x < width; x++) {
                    float value = in_row[x];
                    out_row0[2 * x] = value;
                    out_row0[2 * x + 1] = value;
                    out_row1[2 * x] = value;
                    out_row1[2 * x + 1] = value;
                }
            }
        }
    }
}

/* --- decode ----------------------------------------------------------------- */

/* Which decoder the caller wants.
 *
 * The device decoder has been in this file since it landed, but behind
 * ZIMAGE_VAE_GPU=1, so serve never once ran it: at 512 px it takes 0.33 s
 * against 22.18 s on the CPU (§23).  Unset now means "device when there is
 * one", with a CPU fallback -- a decode that does not fit must not fail a
 * request the CPU path could have finished.  "0" pins the reference path (the
 * gates use it to compare against the torch fixture), "1" demands the device
 * and surfaces its error instead of quietly going slow.
 */
static zvae_path zvae_path_mode(void) {
    const char *value = getenv("ZIMAGE_VAE_GPU");
    if (!value || !*value) return ZVAE_PATH_AUTO;
    if (strcmp(value, "0") == 0) return ZVAE_PATH_CPU;
    return ZVAE_PATH_DEVICE;
}

/* One line, once per process: which decoder the requests are actually using.
 * Without it "the VAE got 30x faster" is indistinguishable from "the card is
 * not being used and the number came from somewhere else". */
static void zvae_announce_device(void) {
    static int announced;
    if (announced) return;
    announced = 1;
    fprintf(stderr, "QwenZImageNative zvae: decoding on the device\n");
    fflush(stderr);
}

int qzen_zvae_decode(const qzen_zvae *vae, const float *latents,
                   uint32_t latent_h, uint32_t latent_w, float *pixels,
                   char *error, size_t error_size) {
    const zvae_path path = zvae_path_mode();
    if (path != ZVAE_PATH_CPU) {
        qzen_zvae *mutable_vae = (qzen_zvae *)vae;
        char device_error[256] = "";
        if (!mutable_vae->gpu) {
            mutable_vae->gpu = zvae_gpu_create(vae, device_error,
                                               sizeof(device_error));
        }
        if (mutable_vae->gpu) {
            if (zvae_decode_gpu(vae, mutable_vae->gpu, latents, latent_h,
                                latent_w, pixels, device_error,
                                sizeof(device_error))) {
                zvae_announce_device();
                return 1;
            }
            /* A failed device decode leaves `pixels` partially written; the
             * CPU path below overwrites all of it. */
            zvae_gpu_free(mutable_vae->gpu);
            mutable_vae->gpu = NULL;
        }
        if (path == ZVAE_PATH_DEVICE) {
            zvae_fail(error, error_size, "%s",
                      device_error[0] ? device_error
                                      : "the device decoder is unavailable");
            return 0;
        }
        fprintf(stderr,
                "QwenZImageNative zvae: device decode unavailable (%s); using the CPU path\n",
                device_error[0] ? device_error : "no device");
        fflush(stderr);
    }
    const qzen_zvae_config *config = &vae->config;
    const uint32_t channels_mid = config->up_channels[0];   /* 512 */
    size_t p1 = (size_t)latent_h * latent_w;
    size_t p3 = p1 * 16, p4 = p1 * 64;
    /* Working buffers sized for the largest stage that uses each slot.
     * A/B: up to 512 x p3; C/D/E/F: up to 256 x p4; G/H: 128 x p4. */
    float *a = zvae_alloc(channels_mid * p3);
    float *b = zvae_alloc(channels_mid * p3);
    float *c = zvae_alloc(256 * p4);
    float *d = zvae_alloc(256 * p4);
    float *e = zvae_alloc(256 * p4);
    float *f = zvae_alloc(256 * p4);
    float *g = zvae_alloc(128 * p4);
    float *h = zvae_alloc(128 * p4);
    if (!a || !b || !c || !d || !e || !f || !g || !h) {
        zvae_fail(error, error_size, "cannot allocate VAE decode buffers (%.1f GB)",
                  (double)(channels_mid * p3 * 2 + (size_t)256 * p4 * 4 +
                           (size_t)128 * p4 * 2) * 4.0 / 1e9);
        free(a); free(b); free(c); free(d); free(e); free(f); free(g); free(h);
        return 0;
    }

    /* Undo the pipeline scaling: latents / scaling_factor + shift_factor. */
    for (uint32_t ch = 0; ch < config->latent_channels; ch++) {
        const float *source = latents + (size_t)ch * p1;
        float *target = a + (size_t)ch * p1;
        for (size_t index = 0; index < p1; index++) {
            target[index] = source[index] / config->scaling_factor + config->shift_factor;
        }
    }
    zvae_conv3x3_forward(&vae->conv_in, a, b, latent_h, latent_w);   /* b: 512@p1 */

    /* mid block at latent resolution. */
    zvae_resnet_forward(&vae->mid.resnets[0], b, a, c, d, config->groups,
                        config->eps, latent_h, latent_w);            /* d: 512@p1 */
    if (!zvae_mid_attention(vae, d, a, latent_h, latent_w)) {        /* a: 512@p1 */
        zvae_fail(error, error_size, "cannot allocate attention buffers");
        goto fail;
    }
    zvae_resnet_forward(&vae->mid.resnets[1], a, b, c, d, config->groups,
                        config->eps, latent_h, latent_w);            /* d: 512@p1 */

    /* up blocks.  Buffer roles per stage (capacities: a/b = 512*p3,
     * c/d/e/f = 256*p4, g/h = 128*p4):
     *   block 0 (512, res @p1, up -> p2):  cur c/d -> hidden a, scratch b
     *     upsample c -> a (512@p2), conv a -> b
     *   block 1 (512, res @p2, up -> p3):  cur b/d -> hidden a, scratch c
     *     upsample d -> a (512@p3), conv a -> b
     *   block 2 (512->256, res @p3, up -> p4): cur b/d -> hidden a, scratch c
     *     upsample d -> e (256@p4), conv e -> c
     *   block 3 (256->128, res @p4):       cur c -> hidden d, scratch e, out g/h */
    {
        /* block 0 */
        const zvae_up_block *up = &vae->up_blocks[0];
        zvae_resnet_forward(&up->resnets[0], d, a, b, c, config->groups, config->eps,
                            latent_h, latent_w);                       /* c: 512@p1 */
        zvae_resnet_forward(&up->resnets[1], c, a, b, d, config->groups, config->eps,
                            latent_h, latent_w);
        zvae_resnet_forward(&up->resnets[2], d, a, b, c, config->groups, config->eps,
                            latent_h, latent_w);
        zvae_upsample_nearest(c, a, channels_mid, latent_h, latent_w);  /* a: 512@p2 */
        zvae_conv3x3_forward(&up->upsample, a, b, latent_h * 2, latent_w * 2);

        /* block 1 */
        up = &vae->up_blocks[1];
        zvae_resnet_forward(&up->resnets[0], b, a, c, d, config->groups, config->eps,
                            latent_h * 2, latent_w * 2);                /* d: 512@p2 */
        zvae_resnet_forward(&up->resnets[1], d, a, c, b, config->groups, config->eps,
                            latent_h * 2, latent_w * 2);
        zvae_resnet_forward(&up->resnets[2], b, a, c, d, config->groups, config->eps,
                            latent_h * 2, latent_w * 2);
        zvae_upsample_nearest(d, a, channels_mid, latent_h * 2, latent_w * 2); /* 512@p3 */
        zvae_conv3x3_forward(&up->upsample, a, b, latent_h * 4, latent_w * 4); /* b: 512@p3 */

        /* block 2: 512 -> 256 */
        up = &vae->up_blocks[2];
        zvae_resnet_forward(&up->resnets[0], b, a, c, d, config->groups, config->eps,
                            latent_h * 4, latent_w * 4);                /* d: 256@p3 */
        zvae_resnet_forward(&up->resnets[1], d, a, c, b, config->groups, config->eps,
                            latent_h * 4, latent_w * 4);
        zvae_resnet_forward(&up->resnets[2], b, a, c, d, config->groups, config->eps,
                            latent_h * 4, latent_w * 4);
        zvae_upsample_nearest(d, e, 256, latent_h * 4, latent_w * 4);   /* e: 256@p4 */
        zvae_conv3x3_forward(&up->upsample, e, c, latent_h * 8, latent_w * 8); /* c: 256@p4 */

        /* block 3: 256 -> 128, no upsampler */
        up = &vae->up_blocks[3];
        zvae_resnet_forward(&up->resnets[0], c, d, e, g, config->groups, config->eps,
                            latent_h * 8, latent_w * 8);                /* g: 128@p4 */
        zvae_resnet_forward(&up->resnets[1], g, d, e, h, config->groups, config->eps,
                            latent_h * 8, latent_w * 8);
        zvae_resnet_forward(&up->resnets[2], h, d, e, g, config->groups, config->eps,
                            latent_h * 8, latent_w * 8);
    }

    /* output: gn -> silu -> conv 128 -> 3. */
    zvae_group_norm(&vae->norm_out, config->groups, config->eps, g,
                    config->up_channels[3], latent_h * 8, latent_w * 8);
    zvae_silu(g, (size_t)128 * p4);
    zvae_conv3x3_forward(&vae->conv_out, g, pixels, latent_h * 8, latent_w * 8);
    free(a); free(b); free(c); free(d); free(e); free(f); free(g); free(h);
    return 1;

fail:
    free(a); free(b); free(c); free(d); free(e); free(f); free(g); free(h);
    return 0;
}

/* --- lazy encoder load (ControlNet) ----------------------------------------- */

/* The encoder tensors live in the same safetensors store as the decoder.
 * qzen_zvae_load() remembered its path; this reads the encoder half on the
 * first qzen_zvae_encode() so an ordinary decode pays nothing for it. */
static int zvae_encoder_load(qzen_zvae *vae, char *error, size_t error_size) {
    zvae_encoder *enc = &vae->encoder;
    if (enc->loaded) return 1;
    if (!vae->source_path) {
        zvae_fail(error, error_size, "VAE encoder source path is unavailable");
        return 0;
    }
    qzen_weight_store *store = qzen_weight_store_open(vae->source_path, error, error_size);
    if (!store) return 0;

    const qzen_zvae_config *config = &vae->config;
    /* Encoder block_out_channels is the reverse of the decoder's up_channels:
     * {128, 256, 512, 512} for the default config. */
    uint32_t ch[4];
    for (uint32_t b = 0; b < 4; b++) ch[b] = config->up_channels[3 - b];
    const uint32_t doubled = 2 * config->latent_channels; /* mean + logvar */

    char prefix[128];
    int ok = zvae_conv_load(store, "encoder.conv_in", ch[0], 3, 3,
                            &enc->conv_in, error, error_size);

    for (uint32_t b = 0; ok && b < 4; b++) {
        for (uint32_t r = 0; ok && r < 2; r++) {
            uint32_t in_ch = r == 0 ? (b == 0 ? ch[0] : ch[b - 1]) : ch[b];
            snprintf(prefix, sizeof(prefix),
                     "encoder.down_blocks.%u.resnets.%u", b, r);
            ok = zvae_resnet_load(store, prefix, in_ch, ch[b], config,
                                  &enc->down[b].resnets[r], error, error_size);
        }
        if (ok && b < 3) {
            enc->down[b].has_downsample = 1;
            snprintf(prefix, sizeof(prefix),
                     "encoder.down_blocks.%u.downsamplers.0.conv", b);
            ok = zvae_conv_load(store, prefix, ch[b], ch[b], 3,
                                &enc->down[b].downsample, error, error_size);
        }
    }

    for (uint32_t r = 0; ok && r < 2; r++) {
        snprintf(prefix, sizeof(prefix), "encoder.mid_block.resnets.%u", r);
        ok = zvae_resnet_load(store, prefix, ch[3], ch[3], config,
                              &enc->mid_resnets[r], error, error_size);
    }
    if (ok) ok = zvae_norm_load(store,
                                "encoder.mid_block.attentions.0.group_norm",
                                ch[3], &enc->mid_attn.norm, error, error_size);
    static const char *qkv[4] = {"to_q", "to_k", "to_v", "to_out.0"};
    float **mats[4] = {&enc->mid_attn.to_q, &enc->mid_attn.to_k,
                       &enc->mid_attn.to_v, &enc->mid_attn.to_out};
    float **bies[4] = {&enc->mid_attn.to_q_bias, &enc->mid_attn.to_k_bias,
                       &enc->mid_attn.to_v_bias, &enc->mid_attn.to_out_bias};
    for (uint32_t m = 0; ok && m < 4; m++) {
        snprintf(prefix, sizeof(prefix),
                 "encoder.mid_block.attentions.0.%s", qkv[m]);
        zvae_conv3x3 loaded;
        memset(&loaded, 0, sizeof(loaded));
        int loaded_ok = zvae_conv_load(store, prefix, ch[3], ch[3], 1,
                                       &loaded, error, error_size);
        /* Keep even a partial allocation in the struct so cleanup frees it. */
        *mats[m] = loaded.weight;
        *bies[m] = loaded.bias;
        ok = loaded_ok;
    }

    if (ok) ok = zvae_norm_load(store, "encoder.conv_norm_out", ch[3],
                                &enc->conv_norm_out, error, error_size);
    if (ok) ok = zvae_conv_load(store, "encoder.conv_out", doubled, ch[3], 3,
                                &enc->conv_out, error, error_size);

    qzen_weight_store_free(store);
    if (!ok) {
        zvae_encoder_free(enc);
        return 0;
    }
    enc->loaded = 1;
    return 1;
}

/* Run one encoder ResnetBlock2D through the 4 working buffers: input is
 * buf[cur], hidden/scratch/output are the other three.  On return the output
 * buffer holds the new current tensor. */
static void zvae_encode_resnet_step(float *buf[4], int *cur_idx,
                                    const zvae_resnet *resnet,
                                    uint32_t groups, float eps,
                                    uint32_t h, uint32_t w) {
    int i = *cur_idx;
    zvae_resnet_forward(resnet, buf[i], buf[(i + 1) & 3],
                        buf[(i + 2) & 3], buf[(i + 3) & 3],
                        groups, eps, h, w);
    *cur_idx = (i + 3) & 3;
}

int qzen_zvae_encode(qzen_zvae *vae, const uint8_t *rgb,
                     uint32_t width, uint32_t height, float *latent,
                     char *error, size_t error_size) {
    if (!vae)    { zvae_fail(error, error_size, "qzen_zvae_encode: vae is NULL"); return 0; }
    if (!rgb)    { zvae_fail(error, error_size, "qzen_zvae_encode: rgb is NULL"); return 0; }
    if (!latent) { zvae_fail(error, error_size, "qzen_zvae_encode: latent is NULL"); return 0; }
    if (width == 0 || height == 0 || (width % 8) != 0 || (height % 8) != 0) {
        zvae_fail(error, error_size,
                  "qzen_zvae_encode: width/height must be non-zero multiples of 8 (got %ux%u)",
                  width, height);
        return 0;
    }
    if (!zvae_encoder_load(vae, error, error_size)) return 0;

    const qzen_zvae_config *config = &vae->config;
    const uint32_t groups = config->groups;
    const float eps = config->eps;
    const uint32_t latent_ch = config->latent_channels;
    const zvae_encoder *enc = &vae->encoder;

    uint32_t ch[4];
    for (uint32_t b = 0; b < 4; b++) ch[b] = config->up_channels[3 - b];

    const size_t plane0 = (size_t)height * width;
    /* Largest single working tensor: the first down block runs at full
     * resolution with ch[0] channels (128 * H * W).  Derive the max over all
     * stages defensively (hidden = max(in, out) channels). */
    size_t cap = (size_t)ch[0] * plane0;
    for (uint32_t b = 0; b < 4; b++) {
        size_t spatial = (size_t)(height >> b) * (width >> b);
        uint32_t prev = b == 0 ? ch[0] : ch[b - 1];
        uint32_t widest = ch[b] > prev ? ch[b] : prev;
        size_t need = (size_t)widest * spatial;
        if (need > cap) cap = need;
    }

    float *buf[4];
    for (int k = 0; k < 4; k++) buf[k] = zvae_alloc(cap);
    if (!buf[0] || !buf[1] || !buf[2] || !buf[3]) {
        zvae_fail(error, error_size, "cannot allocate VAE encode buffers");
        for (int k = 0; k < 4; k++) free(buf[k]);
        return 0;
    }

    /* 1. planar RGB8 [3,H,W] 0..255 -> F32 [-1,1] into buf[0]. */
    for (uint32_t c = 0; c < 3; c++) {
        const uint8_t *src = rgb + (size_t)c * plane0;
        float *dst = buf[0] + (size_t)c * plane0;
        for (size_t i = 0; i < plane0; i++) {
            dst[i] = (float)src[i] * (2.0f / 255.0f) - 1.0f;
        }
    }

    /* 2. conv_in: 3 -> ch[0] at HxW. */
    zvae_conv3x3_forward(&enc->conv_in, buf[0], buf[1], height, width);
    int cur_idx = 1;
    uint32_t h = height, w = width;

    /* 3. four down blocks: two resnets, then a stride-2 conv on b = 0..2. */
    for (uint32_t b = 0; b < 4; b++) {
        zvae_encode_resnet_step(buf, &cur_idx, &enc->down[b].resnets[0],
                                groups, eps, h, w);
        zvae_encode_resnet_step(buf, &cur_idx, &enc->down[b].resnets[1],
                                groups, eps, h, w);
        if (enc->down[b].has_downsample) {
            int i = cur_idx;
            zvae_conv3x3_stride2_forward(&enc->down[b].downsample,
                                         buf[i], buf[(i + 1) & 3], h, w);
            cur_idx = (i + 1) & 3;
            h /= 2;
            w /= 2;
        }
    }

    /* 4. mid block: resnet -> attention -> resnet at H/8 x W/8. */
    zvae_encode_resnet_step(buf, &cur_idx, &enc->mid_resnets[0],
                            groups, eps, h, w);
    {
        int i = cur_idx;
        if (!zvae_attention_forward(&enc->mid_attn, groups, eps,
                                    buf[i], buf[(i + 1) & 3], h, w)) {
            zvae_fail(error, error_size, "cannot allocate VAE attention buffers");
            for (int k = 0; k < 4; k++) free(buf[k]);
            return 0;
        }
        cur_idx = (i + 1) & 3;
    }
    zvae_encode_resnet_step(buf, &cur_idx, &enc->mid_resnets[1],
                            groups, eps, h, w);

    /* 5. conv_norm_out -> SiLU -> conv_out ch[3] -> 2*latent_ch. */
    zvae_group_norm(&enc->conv_norm_out, groups, eps, buf[cur_idx], ch[3], h, w);
    zvae_silu(buf[cur_idx], (size_t)ch[3] * h * w);
    float *moments = buf[(cur_idx + 1) & 3];
    zvae_conv3x3_forward(&enc->conv_out, buf[cur_idx], moments, h, w);

    /* 6. posterior mode = the first latent_ch channels (the mean);
     * discard the trailing logvar channels and normalise for the DiT. */
    const size_t p = (size_t)h * w;   /* (H/8)*(W/8) */
    for (uint32_t c = 0; c < latent_ch; c++) {
        const float *mean = moments + (size_t)c * p;
        float *dst = latent + (size_t)c * p;
        for (size_t i = 0; i < p; i++) {
            dst[i] = (mean[i] - config->shift_factor) * config->scaling_factor;
        }
    }

    for (int k = 0; k < 4; k++) free(buf[k]);
    return 1;
}

/* --- device decode (default since §23) -------------------------------------- */

#include "qzen_gpu.h"

/* Device mirror of the host weight structs. Tensor pairs stand in for
 * (weight, bias); channels live on the resnet so the forward helpers can
 * validate shapes without reaching back into the host structs. */
typedef struct {
    qzen_gpu_tensor *weight, *bias;
} zgpu_pair;

typedef struct {
    uint32_t in_channels, out_channels;
    zgpu_pair norm1, conv1, norm2, conv2, shortcut;
    int has_shortcut;
} zgpu_resnet;

typedef struct {
    uint32_t count;
    zgpu_resnet *resnets;
    int has_upsample;
    zgpu_pair upsample;
} zgpu_up_block;

typedef struct {
    zgpu_resnet resnets[2];
    zgpu_pair attn_norm;
    zgpu_pair to_q, to_k, to_v, to_out;
} zgpu_mid;

struct zvae_gpu {
    qzen_gpu *gpu;
    zgpu_pair conv_in;
    zgpu_mid mid;
    zgpu_up_block up_blocks[4];
    zgpu_pair norm_out;
    zgpu_pair conv_out;
};

static qzen_gpu_tensor *zgpu_pair_tensor(qzen_gpu *gpu, const float *host,
                                       uint64_t elements) {
    if (!host || !elements) return NULL;
    return qzen_gpu_tensor_from_f32(gpu, host, (size_t)elements);
}

static int zgpu_pair_load(qzen_gpu *gpu, zgpu_pair *pair, const float *weight,
                          uint64_t weight_elements, const float *bias,
                          uint64_t bias_elements) {
    pair->weight = zgpu_pair_tensor(gpu, weight, weight_elements);
    pair->bias = zgpu_pair_tensor(gpu, bias, bias_elements);
    return pair->weight && pair->bias;
}

static int zgpu_conv_load(qzen_gpu *gpu, zgpu_pair *pair,
                          const zvae_conv3x3 *conv, uint32_t kernel) {
    return zgpu_pair_load(gpu, pair, conv->weight,
                          (uint64_t)conv->out * conv->in * kernel * kernel,
                          conv->bias, conv->out);
}

static int zgpu_norm_load(qzen_gpu *gpu, zgpu_pair *pair,
                          const zvae_norm *norm) {
    return zgpu_pair_load(gpu, pair, norm->weight, norm->channels,
                          norm->bias, norm->channels);
}

static int zgpu_resnet_load(qzen_gpu *gpu, zgpu_resnet *device,
                            const zvae_resnet *host) {
    device->in_channels = host->norm1.channels;
    device->out_channels = host->norm2.channels;
    device->has_shortcut = host->has_shortcut;
    return zgpu_norm_load(gpu, &device->norm1, &host->norm1) &&
           zgpu_conv_load(gpu, &device->conv1, &host->conv1, 3) &&
           zgpu_norm_load(gpu, &device->norm2, &host->norm2) &&
           zgpu_conv_load(gpu, &device->conv2, &host->conv2, 3) &&
           (!host->has_shortcut ||
            zgpu_conv_load(gpu, &device->shortcut, &host->shortcut, 1));
}

static void zgpu_pair_free(zgpu_pair *pair) {
    qzen_gpu_tensor_free(pair->weight);
    qzen_gpu_tensor_free(pair->bias);
    pair->weight = pair->bias = NULL;
}

static void zgpu_resnet_free(zgpu_resnet *resnet) {
    zgpu_pair_free(&resnet->norm1);
    zgpu_pair_free(&resnet->conv1);
    zgpu_pair_free(&resnet->norm2);
    zgpu_pair_free(&resnet->conv2);
    zgpu_pair_free(&resnet->shortcut);
}

void zvae_gpu_free(zvae_gpu *state) {
    if (!state) return;
    zgpu_resnet_free(&state->mid.resnets[0]);
    zgpu_resnet_free(&state->mid.resnets[1]);
    zgpu_pair_free(&state->mid.attn_norm);
    zgpu_pair_free(&state->mid.to_q);
    zgpu_pair_free(&state->mid.to_k);
    zgpu_pair_free(&state->mid.to_v);
    zgpu_pair_free(&state->mid.to_out);
    for (uint32_t b = 0; b < 4; b++) {
        for (uint32_t r = 0; r < state->up_blocks[b].count; r++) {
            zgpu_resnet_free(&state->up_blocks[b].resnets[r]);
        }
        free(state->up_blocks[b].resnets);
        zgpu_pair_free(&state->up_blocks[b].upsample);
    }
    zgpu_pair_free(&state->conv_in);
    zgpu_pair_free(&state->norm_out);
    zgpu_pair_free(&state->conv_out);
    qzen_gpu_free(state->gpu);
    free(state);
}

zvae_gpu *zvae_gpu_create(const qzen_zvae *vae, char *error, size_t error_size) {
    zvae_gpu *state = calloc(1, sizeof(*state));
    if (!state) {
        zvae_fail(error, error_size, "cannot allocate Z-Image VAE GPU state");
        return NULL;
    }
    state->gpu = qzen_gpu_create(NULL, error, error_size);
    if (!state->gpu) {
        free(state);
        return NULL;
    }
    qzen_gpu *gpu = state->gpu;
    const qzen_zvae_config *config = &vae->config;
    uint32_t mid_channels = config->up_channels[0];
    int ok = zgpu_conv_load(gpu, &state->conv_in, &vae->conv_in, 3);
    for (uint32_t r = 0; ok && r < 2; r++) {
        ok = zgpu_resnet_load(gpu, &state->mid.resnets[r],
                              &vae->mid.resnets[r]);
    }
    if (ok) {
        ok = zgpu_norm_load(gpu, &state->mid.attn_norm, &vae->mid.attn_norm);
    }
    if (ok) {
        ok = zgpu_pair_load(gpu, &state->mid.to_q, vae->mid.to_q,
                            (uint64_t)mid_channels * mid_channels,
                            vae->mid.to_q_bias, mid_channels) &&
             zgpu_pair_load(gpu, &state->mid.to_k, vae->mid.to_k,
                            (uint64_t)mid_channels * mid_channels,
                            vae->mid.to_k_bias, mid_channels) &&
             zgpu_pair_load(gpu, &state->mid.to_v, vae->mid.to_v,
                            (uint64_t)mid_channels * mid_channels,
                            vae->mid.to_v_bias, mid_channels) &&
             zgpu_pair_load(gpu, &state->mid.to_out, vae->mid.to_out,
                            (uint64_t)mid_channels * mid_channels,
                            vae->mid.to_out_bias, mid_channels);
    }
    for (uint32_t b = 0; ok && b < 4; b++) {
        zgpu_up_block *block = &state->up_blocks[b];
        const zvae_up_block *host = &vae->up_blocks[b];
        block->count = host->count;
        block->has_upsample = host->has_upsample;
        block->resnets = calloc(block->count ? block->count : 1,
                                sizeof(zgpu_resnet));
        if (!block->resnets) {
            zvae_fail(error, error_size, "cannot allocate GPU up block %u", b);
            ok = 0;
            break;
        }
        for (uint32_t j = 0; ok && j < block->count; j++) {
            ok = zgpu_resnet_load(gpu, &block->resnets[j], &host->resnets[j]);
        }
        if (ok && block->has_upsample) {
            ok = zgpu_conv_load(gpu, &block->upsample, &host->upsample, 3);
        }
    }
    if (ok) ok = zgpu_norm_load(gpu, &state->norm_out, &vae->norm_out);
    if (ok) ok = zgpu_conv_load(gpu, &state->conv_out, &vae->conv_out, 3);
    if (!ok) {
        if (error && error_size && !error[0]) {
            zvae_fail(error, error_size, "VAE GPU weight upload failed: %s",
                      qzen_gpu_error(gpu));
        }
        zvae_gpu_free(state);
        return NULL;
    }
    return state;
}

/* GPU op wrapper: on failure records the op name and the CUDA error and
 * returns 0, which the decode body turns into a goto done. */
static int zgpu_check(int result, const char *what, qzen_gpu *gpu,
                      char *error, size_t error_size) {
    if (!result) {
        zvae_fail(error, error_size, "VAE GPU decode failed: %s: %s", what,
                  qzen_gpu_error(gpu));
    }
    return result;
}

/* ResnetBlock2D without timestep embedding on the device. The GroupNorm
 * kernel reads its input and writes a separate (or the same) tensor, so the
 * CPU path's input copy into 'hidden' disappears here. */
static int zgpu_resnet_forward(qzen_gpu *gpu, const zgpu_resnet *resnet,
                               uint32_t groups, float eps,
                               qzen_gpu_tensor *input, qzen_gpu_tensor *hidden,
                               qzen_gpu_tensor *scratch, qzen_gpu_tensor *output,
                               uint32_t height, uint32_t width) {
    uint32_t in_ch = resnet->in_channels;
    uint32_t out_ch = resnet->out_channels;
    if (!qzen_gpu_group_norm_chw_f32(gpu, hidden, input, resnet->norm1.weight,
                                   resnet->norm1.bias, in_ch, height, width,
                                   groups, eps, 1) ||
        !qzen_gpu_conv2d_f32(gpu, scratch, hidden, resnet->conv1.weight,
                           resnet->conv1.bias, out_ch, in_ch, height, width,
                           3, 1) ||
        !qzen_gpu_group_norm_chw_f32(gpu, scratch, scratch, resnet->norm2.weight,
                                   resnet->norm2.bias, out_ch, height, width,
                                   groups, eps, 1) ||
        !qzen_gpu_conv2d_f32(gpu, output, scratch, resnet->conv2.weight,
                           resnet->conv2.bias, out_ch, out_ch, height, width,
                           3, 1))
        return 0;
    if (resnet->has_shortcut &&
        !qzen_gpu_conv2d_f32(gpu, hidden, input, resnet->shortcut.weight,
                           resnet->shortcut.bias, out_ch, in_ch, height,
                           width, 1, 0))
        return 0;
    const qzen_gpu_tensor *residual = resnet->has_shortcut ? hidden : input;
    return qzen_gpu_add_f32(gpu, output, output, residual,
                          out_ch * height * width);
}

/* Mid-block spatial self-attention. Layout dance: the [C, HW] input is
 * GroupNormed, transposed to [HW, C] rows for the cuBLAS projections and the
 * SDPA, then transposed back onto the residual. Scratch lives only for this
 * call; the [seq, seq] score matrix dominates (1 GiB at 1024 px). */
static int zgpu_mid_attention(
    qzen_gpu *gpu, const zgpu_mid *mid, uint32_t channels, uint32_t groups,
    float eps, qzen_gpu_tensor *input, qzen_gpu_tensor *output,
    qzen_gpu_tensor *normed, uint32_t height, uint32_t width) {
    size_t tokens = (size_t)height * width;
    float scale = 1.0f / sqrtf((float)channels);
    qzen_gpu_tensor *transposed = qzen_gpu_tensor_new_f32(gpu, channels * tokens);
    qzen_gpu_tensor *q = qzen_gpu_tensor_new_f32(gpu, channels * tokens);
    qzen_gpu_tensor *k = qzen_gpu_tensor_new_f32(gpu, channels * tokens);
    qzen_gpu_tensor *v = qzen_gpu_tensor_new_f32(gpu, channels * tokens);
    qzen_gpu_tensor *attn = qzen_gpu_tensor_new_f32(gpu, channels * tokens);
    qzen_gpu_tensor *scores = qzen_gpu_tensor_new_f32(gpu, tokens * tokens);
    int ok = transposed && q && k && v && attn && scores;
    if (ok) {
        ok = qzen_gpu_group_norm_chw_f32(gpu, normed, input,
                                       mid->attn_norm.weight,
                                       mid->attn_norm.bias, channels, height,
                                       width, groups, eps, 0) &&
             qzen_gpu_transpose_f32(gpu, transposed, normed, channels,
                                  (uint32_t)tokens) &&
             qzen_gpu_linear_f32(gpu, q, transposed, mid->to_q.weight,
                               mid->to_q.bias, (uint32_t)tokens, channels,
                               channels) &&
             qzen_gpu_linear_f32(gpu, k, transposed, mid->to_k.weight,
                               mid->to_k.bias, (uint32_t)tokens, channels,
                               channels) &&
             qzen_gpu_linear_f32(gpu, v, transposed, mid->to_v.weight,
                               mid->to_v.bias, (uint32_t)tokens, channels,
                               channels) &&
             qzen_gpu_vae_sdpa_f32(gpu, attn, q, k, v, scores,
                                 (uint32_t)tokens, channels, scale) &&
             qzen_gpu_linear_f32(gpu, transposed, attn, mid->to_out.weight,
                               mid->to_out.bias, (uint32_t)tokens, channels,
                               channels) &&
             qzen_gpu_transpose_f32(gpu, normed, transposed, (uint32_t)tokens,
                                  channels);
    }
    qzen_gpu_tensor_free(transposed);
    qzen_gpu_tensor_free(q);
    qzen_gpu_tensor_free(k);
    qzen_gpu_tensor_free(v);
    qzen_gpu_tensor_free(attn);
    qzen_gpu_tensor_free(scores);
    if (!ok) return 0;
    return qzen_gpu_add_f32(gpu, output, normed, input,
                          channels * height * width);
}

static int zvae_decode_gpu(const qzen_zvae *vae, zvae_gpu *state,
                           const float *latents, uint32_t latent_h,
                           uint32_t latent_w, float *pixels, char *error,
                           size_t error_size) {
    const qzen_zvae_config *config = &vae->config;
    const uint32_t mid_channels = config->up_channels[0];
    const uint32_t block2_channels = config->up_channels[2];   /* 256 */
    const uint32_t block3_channels = config->up_channels[3];   /* 128 */
    const uint32_t groups = config->groups;
    const float eps = config->eps;
    const size_t p1 = (size_t)latent_h * latent_w;
    const size_t p3 = p1 * 16, p4 = p1 * 64;
    qzen_gpu *gpu = state->gpu;

    /* Buffers are claimed at first use and released after their last reader,
     * because the stream-ordered pool hands freed blocks to the next stage:
     * holding every role simultaneously would need ~17 GB at 1024 px, while
     * the staged schedule below peaks around 11 GB (block 2's upsampler and
     * block 3's first resnet). Capacities cover the largest stage that uses
     * each slot: d never holds 256@p4 because block 3 consumes it as a
     * 128-channel hidden state, and block 3 re-claims its scratch at 128
     * channels after block 2's 256-channel upsampler staging is spent. */
    qzen_gpu_tensor *a = qzen_gpu_tensor_new_f32(gpu, mid_channels * p3);
    qzen_gpu_tensor *b = qzen_gpu_tensor_new_f32(gpu, mid_channels * p3);
    qzen_gpu_tensor *c = NULL;
    qzen_gpu_tensor *d = qzen_gpu_tensor_new_f32(gpu, block2_channels * p3);
    qzen_gpu_tensor *e = NULL;
    qzen_gpu_tensor *g = NULL;
    qzen_gpu_tensor *h_buf = NULL;
    qzen_gpu_tensor *mid_scratch = qzen_gpu_tensor_new_f32(gpu, mid_channels * p1);
    qzen_gpu_tensor *pixels_dev =
        qzen_gpu_tensor_new_f32(gpu, (size_t)config->out_channels * p4);
    float *staged = zvae_alloc((size_t)config->latent_channels * p1);
    int ok = a && b && d && mid_scratch && pixels_dev && staged;
    if (!ok) {
        zvae_fail(error, error_size, "cannot allocate VAE GPU buffers");
        goto done;
    }

    /* Undo the pipeline scaling on the host (16 x p1 elements, negligible). */
    for (uint32_t ch = 0; ch < config->latent_channels; ch++) {
        const float *source = latents + (size_t)ch * p1;
        float *target = staged + (size_t)ch * p1;
        for (size_t index = 0; index < p1; index++) {
            target[index] = source[index] / config->scaling_factor +
                            config->shift_factor;
        }
    }
    if (!qzen_gpu_tensor_write_f32(a, staged, config->latent_channels * p1)) {
        zvae_fail(error, error_size, "cannot upload VAE latents: %s",
                  qzen_gpu_error(gpu));
        ok = 0;
        goto done;
    }

    if (!zgpu_check(qzen_gpu_begin(gpu), "qzen_gpu_begin", gpu, error,
                    error_size)) {
        ok = 0;
        goto done;
    }
    if (!zgpu_check(qzen_gpu_conv2d_f32(gpu, b, a, state->conv_in.weight,
                                      state->conv_in.bias, mid_channels,
                                      config->latent_channels, latent_h,
                                      latent_w, 3, 1),
                    "decoder.conv_in", gpu, error, error_size)) {
        ok = 0;
        goto done;
    }

    /* mid block at latent resolution. */
    if (!zgpu_check(zgpu_resnet_forward(gpu, &state->mid.resnets[0], groups,
                                        eps, b, a, mid_scratch, d, latent_h,
                                        latent_w),
                    "mid_block.resnets.0", gpu, error, error_size) ||
        !zgpu_check(zgpu_mid_attention(
                        gpu, &state->mid, vae->mid.attn_norm.channels, groups,
                        eps, d, a, b, latent_h, latent_w),
                    "mid_block.attentions.0", gpu, error, error_size) ||
        !zgpu_check(zgpu_resnet_forward(gpu, &state->mid.resnets[1], groups,
                                        eps, a, b, mid_scratch, d, latent_h,
                                        latent_w),
                    "mid_block.resnets.1", gpu, error, error_size)) {
        ok = 0;
        goto done;
    }
    qzen_gpu_tensor_free(mid_scratch);
    mid_scratch = NULL;

    /* up blocks; buffer roles follow the CPU decode. */
    {
        c = qzen_gpu_tensor_new_f32(gpu, block2_channels * p4);
        if (!c) {
            zvae_fail(error, error_size, "cannot allocate VAE GPU buffers");
            ok = 0;
            goto done;
        }
        const zgpu_up_block *up = &state->up_blocks[0];
        if (!zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[0], groups, eps,
                                            d, a, b, c, latent_h, latent_w),
                        "up_blocks.0.resnets.0", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[1], groups, eps,
                                            c, a, b, d, latent_h, latent_w),
                        "up_blocks.0.resnets.1", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[2], groups, eps,
                                            d, a, b, c, latent_h, latent_w),
                        "up_blocks.0.resnets.2", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_upsample_nearest2x_f32(
                            gpu, a, c, mid_channels, latent_h, latent_w),
                        "up_blocks.0.upsampler", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_conv2d_f32(gpu, b, a, up->upsample.weight,
                                          up->upsample.bias, mid_channels,
                                          mid_channels, latent_h * 2,
                                          latent_w * 2, 3, 1),
                        "up_blocks.0.upsampler.conv", gpu, error,
                        error_size)) {
            ok = 0;
            goto done;
        }

        up = &state->up_blocks[1];
        if (!zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[0], groups, eps,
                                            b, a, c, d, latent_h * 2,
                                            latent_w * 2),
                        "up_blocks.1.resnets.0", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[1], groups, eps,
                                            d, a, c, b, latent_h * 2,
                                            latent_w * 2),
                        "up_blocks.1.resnets.1", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[2], groups, eps,
                                            b, a, c, d, latent_h * 2,
                                            latent_w * 2),
                        "up_blocks.1.resnets.2", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_upsample_nearest2x_f32(
                            gpu, a, d, mid_channels, latent_h * 2,
                            latent_w * 2),
                        "up_blocks.1.upsampler", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_conv2d_f32(gpu, b, a, up->upsample.weight,
                                          up->upsample.bias, mid_channels,
                                          mid_channels, latent_h * 4,
                                          latent_w * 4, 3, 1),
                        "up_blocks.1.upsampler.conv", gpu, error,
                        error_size)) {
            ok = 0;
            goto done;
        }

        up = &state->up_blocks[2];
        if (!zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[0], groups, eps,
                                            b, a, c, d, latent_h * 4,
                                            latent_w * 4),
                        "up_blocks.2.resnets.0", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[1], groups, eps,
                                            d, a, c, b, latent_h * 4,
                                            latent_w * 4),
                        "up_blocks.2.resnets.1", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[2], groups, eps,
                                            b, a, c, d, latent_h * 4,
                                            latent_w * 4),
                        "up_blocks.2.resnets.2", gpu, error, error_size)) {
            ok = 0;
            goto done;
        }
        /* a/b are done; their pool blocks feed the upsampler staging. */
        qzen_gpu_tensor_free(a);
        a = NULL;
        qzen_gpu_tensor_free(b);
        b = NULL;
        e = qzen_gpu_tensor_new_f32(gpu, block2_channels * p4);
        if (!e) {
            zvae_fail(error, error_size, "cannot allocate VAE GPU buffers");
            ok = 0;
            goto done;
        }
        if (!zgpu_check(qzen_gpu_upsample_nearest2x_f32(
                            gpu, e, d, block2_channels, latent_h * 4,
                            latent_w * 4),
                        "up_blocks.2.upsampler", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_conv2d_f32(gpu, c, e, up->upsample.weight,
                                          up->upsample.bias, block2_channels,
                                          block2_channels, latent_h * 8,
                                          latent_w * 8, 3, 1),
                        "up_blocks.2.upsampler.conv", gpu, error,
                        error_size)) {
            ok = 0;
            goto done;
        }
        /* e returns to the pool and comes back at block 3's 128 channels. */
        qzen_gpu_tensor_free(e);
        e = qzen_gpu_tensor_new_f32(gpu, block3_channels * p4);
        g = qzen_gpu_tensor_new_f32(gpu, block3_channels * p4);
        if (!e || !g) {
            zvae_fail(error, error_size, "cannot allocate VAE GPU buffers");
            ok = 0;
            goto done;
        }
        /* d also returns: block 3 runs its 256 -> 128 resnet shortcut-first
         * with an in-place GroupNorm, so every block 3 slot is 128-channel.
         * The CPU order computes the 1x1 shortcut last, but both orders read
         * the same unmodified input, so the arithmetic is identical. */
        qzen_gpu_tensor_free(d);
        d = qzen_gpu_tensor_new_f32(gpu, block3_channels * p4);
        if (!d) {
            zvae_fail(error, error_size, "cannot allocate VAE GPU buffers");
            ok = 0;
            goto done;
        }
        up = &state->up_blocks[3];
        {
            const zgpu_resnet *r0 = &up->resnets[0];
            uint32_t in_ch = r0->in_channels;
            uint32_t out_ch = r0->out_channels;
            uint32_t out_h = latent_h * 8, out_w = latent_w * 8;
            if (!zgpu_check(qzen_gpu_conv2d_f32(gpu, d, c, r0->shortcut.weight,
                                              r0->shortcut.bias, out_ch,
                                              in_ch, out_h, out_w, 1, 0),
                            "up_blocks.3.resnets.0.shortcut", gpu, error,
                            error_size) ||
                !zgpu_check(qzen_gpu_group_norm_chw_f32(
                                gpu, c, c, r0->norm1.weight, r0->norm1.bias,
                                in_ch, out_h, out_w, groups, eps, 1),
                            "up_blocks.3.resnets.0.norm1", gpu, error,
                            error_size) ||
                !zgpu_check(qzen_gpu_conv2d_f32(gpu, e, c, r0->conv1.weight,
                                              r0->conv1.bias, out_ch, in_ch,
                                              out_h, out_w, 3, 1),
                            "up_blocks.3.resnets.0.conv1", gpu, error,
                            error_size) ||
                !zgpu_check(qzen_gpu_group_norm_chw_f32(
                                gpu, e, e, r0->norm2.weight, r0->norm2.bias,
                                out_ch, out_h, out_w, groups, eps, 1),
                            "up_blocks.3.resnets.0.norm2", gpu, error,
                            error_size) ||
                !zgpu_check(qzen_gpu_conv2d_f32(gpu, g, e, r0->conv2.weight,
                                              r0->conv2.bias, out_ch, out_ch,
                                              out_h, out_w, 3, 1),
                            "up_blocks.3.resnets.0.conv2", gpu, error,
                            error_size) ||
                !zgpu_check(qzen_gpu_add_f32(gpu, g, g, d,
                                           out_ch * out_h * out_w),
                            "up_blocks.3.resnets.0.add", gpu, error,
                            error_size)) {
                ok = 0;
                goto done;
            }
        }
        qzen_gpu_tensor_free(c);
        c = NULL;
        h_buf = qzen_gpu_tensor_new_f32(gpu, block3_channels * p4);
        if (!h_buf) {
            zvae_fail(error, error_size, "cannot allocate VAE GPU buffers");
            ok = 0;
            goto done;
        }
        if (!zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[1], groups, eps,
                                            g, d, e, h_buf, latent_h * 8,
                                            latent_w * 8),
                        "up_blocks.3.resnets.1", gpu, error, error_size) ||
            !zgpu_check(zgpu_resnet_forward(gpu, &up->resnets[2], groups, eps,
                                            h_buf, d, e, g, latent_h * 8,
                                            latent_w * 8),
                        "up_blocks.3.resnets.2", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_group_norm_chw_f32(
                            gpu, g, g, state->norm_out.weight,
                            state->norm_out.bias, block3_channels,
                            latent_h * 8, latent_w * 8, groups, eps, 1),
                        "conv_norm_out", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_conv2d_f32(gpu, pixels_dev, g,
                                          state->conv_out.weight,
                                          state->conv_out.bias,
                                          config->out_channels,
                                          block3_channels, latent_h * 8,
                                          latent_w * 8, 3, 1),
                        "conv_out", gpu, error, error_size) ||
            !zgpu_check(qzen_gpu_submit(gpu), "qzen_gpu_submit", gpu, error,
                        error_size)) {
            ok = 0;
            goto done;
        }
    }

    if (!qzen_gpu_tensor_read_f32(pixels_dev, pixels,
                                (size_t)config->out_channels * p4)) {
        zvae_fail(error, error_size, "cannot download VAE pixels: %s",
                  qzen_gpu_error(gpu));
        ok = 0;
    }

done:
    free(staged);
    qzen_gpu_tensor_free(a);
    qzen_gpu_tensor_free(b);
    qzen_gpu_tensor_free(c);
    qzen_gpu_tensor_free(d);
    qzen_gpu_tensor_free(e);
    qzen_gpu_tensor_free(g);
    qzen_gpu_tensor_free(h_buf);
    qzen_gpu_tensor_free(mid_scratch);
    qzen_gpu_tensor_free(pixels_dev);
    return ok;
}
