/* Qwen-Image 2.1 VAE decoder: f32 CPU reference.
 *
 * The Wan2GP decoder (models/qwen21/vae.py) is a 2-D image decoder wrapped in
 * 3-D modules whose single frame collapses: CausalConv3d squeezes the time axis,
 * the temporal upsample is only used through a feature cache, and DupUp3D with
 * first_chunk keeps the last of its duplicated frames. What remains, and what
 * this file implements, is:
 *
 *   post_quant_conv 64->64 1x1
 *   conv_in 64->1152 3x3
 *   mid: resnet(1152) -> attention(1152, one head, scale 1/sqrt(1152)) -> resnet
 *   up blocks 1152->1152->576->288->144->144 (3 resnets each, 4 upsamplers)
 *        x = upsample_nearest2x_conv(x) + dupup3d(branch_input)
 *   norm_out RMS(144) -> SiLU -> conv_out 144->4 3x3
 *
 * Normalisation is x * sqrt(C) / ||x||_2(C) * gamma (+ bias), channels first.
 */

#include "qzen_qwen21_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qzen_gpu.h"
#include "qzen_qwen21_ops.h"
#include "third_party/cJSON.h"

#define Q21_VAE_BLOCKS 5
#define Q21_VAE_RESNETS 3
#define Q21_VAE_WIDE 1152
#define Q21_VAE_CHANNELS QZEN_QWEN21_VAE_CHANNELS

/* Encoder geometry (the mirror of the decoder above). */
#define Q21_ENC_BLOCKS 5
#define Q21_ENC_WIDE 768
#define Q21_ENC_INPUT 4
#define Q21_ENC_OUT 128 /* mean[64] ++ logvar[64] */

/* Device mirror of the host weights, built lazily on the first decode when
 * QWEN21_VAE_GPU=1. Declared here so the host struct can hold the pointer. */
typedef struct q21_vae_gpu q21_vae_gpu;
static int q21_vae_gpu_requested(void);
static q21_vae_gpu *q21_vae_gpu_create(qzen_qwen21_vae *vae, char *error,
                                       size_t error_size);
static void q21_vae_gpu_free(q21_vae_gpu *state);
static int q21_vae_gpu_decode(qzen_qwen21_vae *vae, q21_vae_gpu *state,
                              const float *latent, int height, int width,
                              unsigned char *rgb, char *error,
                              size_t error_size);

typedef struct {
    float *weight; /* [out][in][k][k] */
    float *bias;   /* [out] */
    int out_c, in_c, k;
} q21_conv;

typedef struct {
    q21_conv conv1, conv2, shortcut; /* shortcut is 1x1, empty when in_c == out_c */
    float *gamma1, *gamma2;
    int in_c, out_c;
} q21_resnet;

/* The encoder is the mirror of the decoder and lives in the same checkpoint
 * (102 encoder.* tensors, verified in the file header), so image editing needs
 * no second download. It is loaded on the first encode rather than in open():
 * a t2i-only server never pays its ~0.4 GiB, and a server that edits loads it
 * once.
 *
 *   conv_in 4->96 3x3            (channel 3 is the edit mask)
 *   5 down blocks: 96->96, 96->192, 192->384, 384->768, 768->768
 *        each block is two resnets; the first four are followed by a stride-2
 *        3x3 conv (pad right/bottom, diffusers Downsample2D padding=0)
 *   mid: resnet -> attention -> resnet at 768
 *   norm_out RMS(768) -> SiLU -> conv_out 768->128 3x3 -> quant_conv 128->128
 *        the 128 channels are mean[64] ++ logvar[64]; encode returns the mean,
 *        because a deterministic edit must not resample the latent.
 */
typedef struct {
    int ready;
    q21_conv conv_in, conv_out, quant_conv;
    q21_resnet down[Q21_ENC_BLOCKS][2];
    q21_conv down_conv[Q21_ENC_BLOCKS]; /* stride 2; empty on the last block */
    int in_c[Q21_ENC_BLOCKS];
    int out_c[Q21_ENC_BLOCKS];
    int downsample[Q21_ENC_BLOCKS];
    int temporal[Q21_ENC_BLOCKS]; /* temperal_downsample: the shortcut's factor_t */
    q21_resnet mid[2];
    q21_conv attn_qkv, attn_proj;
    float *attn_gamma;
    float *norm_out_gamma;
} q21_encoder;

struct qzen_qwen21_vae {
    int threads;
    qzen_st_header header;
    q21_conv post_quant, conv_in, conv_out;
    q21_resnet mid[2];
    q21_conv attn_qkv, attn_proj;
    float *attn_gamma;
    q21_resnet up[Q21_VAE_BLOCKS][Q21_VAE_RESNETS];
    q21_conv up_conv[Q21_VAE_BLOCKS]; /* nearest x2 + 3x3, empty without an upsampler */
    float *norm_out_gamma;
    int width[Q21_VAE_BLOCKS];        /* channels entering each up block */
    int channels[Q21_VAE_BLOCKS];     /* channels leaving it */
    int upsamples[Q21_VAE_BLOCKS];    /* 1 when the block has an upsampler */
    int temporal[Q21_VAE_BLOCKS];     /* 1 for the 3-D (factor_t = 2) DupUp3D */
    float *latent_mean, *latent_std;
    q21_encoder enc; /* lazy: first qzen_qwen21_vae_encode() call */
    q21_vae_gpu *gpu; /* QWEN21_VAE_GPU=1, created on first decode */
};

/* The encoder is declared up here because its loader sits above the shared
 * weight loaders (q21_vae_conv_load and friends are used by both directions and
 * live below the decoder). Forward declarations keep the reading order
 * "encoder, then decoder" without moving the shared loaders. */
static int q21_vae_conv_load(qzen_qwen21_vae *vae, q21_conv *conv, const char *name, int out_c,
                             int in_c, int k, char *error, size_t error_size);
static float *q21_vae_gamma_load(qzen_qwen21_vae *vae, const char *name, int channels, char *error,
                                 size_t error_size);
static int q21_vae_resnet_load(qzen_qwen21_vae *vae, q21_resnet *res, const char *prefix, int in_c,
                               int out_c, char *error, size_t error_size);

/* ------------------------------------------------------------------- conv */

static void q21_conv_channels(void *context, int slot, int start, int end) {
    (void)slot;
    const struct {
        const q21_conv *conv;
        const float *x;
        float *y;
        int height, width;
    } *job = context;
    const q21_conv *conv = job->conv;
    const int height = job->height, width = job->width;
    const int plane = height * width;
    const int kernel = conv->k;
    for (int o = start; o < end; o++) {
        float *dst = job->y + (size_t)o * plane;
        const float bias = conv->bias ? conv->bias[o] : 0.0f;
        for (int index = 0; index < plane; index++) dst[index] = bias;
        for (int i = 0; i < conv->in_c; i++) {
            const float *src = job->x + (size_t)i * plane;
            const float *weights = conv->weight + ((size_t)o * conv->in_c + i) * kernel * kernel;
            for (int ky = 0; ky < kernel; ky++) {
                for (int kx = 0; kx < kernel; kx++) {
                    const float w = weights[ky * kernel + kx];
                    if (w == 0.0f) continue;
                    for (int y = 0; y < height; y++) {
                        const int sy = y + ky - kernel / 2;
                        if (sy < 0 || sy >= height) continue;
                        float *row = dst + (size_t)y * width;
                        const float *source = src + (size_t)sy * width;
                        for (int x = 0; x < width; x++) {
                            const int sx = x + kx - kernel / 2;
                            if (sx < 0 || sx >= width) continue;
                            row[x] += w * source[sx];
                        }
                    }
                }
            }
        }
    }
}

typedef struct {
    const q21_conv *conv;
    const float *x;
    float *y;
    int height, width;
} q21_conv_job;

static void q21_conv2d(const q21_conv *conv, const float *x, int height, int width, float *y,
                       int threads) {
    q21_conv_job job = {conv, x, y, height, width};
    q21_parallel(conv->out_c, threads, q21_conv_channels, &job);
}

/* Stride-2 3x3, the encoder's downsampler (diffusers Downsample2D with
 * padding=0: F.pad(x, (0,1,0,1)) then conv stride 2, so output = input / 2 and
 * the sample at (y,x) reads input rows 2y..2y+2 and columns 2x..2x+2, with the
 * bottom/right taps that fall past the edge dropped). */
typedef struct {
    const q21_conv *conv;
    const float *x;
    float *y;
    int height, width, out_h, out_w;
} q21_conv_stride2_job;

static void q21_conv_channels_stride2(void *context, int slot, int start, int end) {
    (void)slot;
    const q21_conv_stride2_job *job = context;
    const q21_conv *conv = job->conv;
    const int height = job->height, width = job->width;
    const int out_h = job->out_h, out_w = job->out_w;
    const int kernel = conv->k;
    for (int o = start; o < end; o++) {
        float *dst = job->y + (size_t)o * out_h * out_w;
        const float bias = conv->bias ? conv->bias[o] : 0.0f;
        for (int index = 0; index < out_h * out_w; index++) dst[index] = bias;
        for (int i = 0; i < conv->in_c; i++) {
            const float *src = job->x + (size_t)i * height * width;
            const float *weights =
                conv->weight + ((size_t)o * conv->in_c + i) * kernel * kernel;
            for (int ky = 0; ky < kernel; ky++) {
                for (int kx = 0; kx < kernel; kx++) {
                    const float w = weights[ky * kernel + kx];
                    if (w == 0.0f) continue;
                    for (int y = 0; y < out_h; y++) {
                        const int sy = 2 * y + ky;
                        if (sy >= height) continue;
                        float *row = dst + (size_t)y * out_w;
                        const float *source = src + (size_t)sy * width;
                        for (int x = 0; x < out_w; x++) {
                            const int sx = 2 * x + kx;
                            if (sx >= width) continue;
                            row[x] += w * source[sx];
                        }
                    }
                }
            }
        }
    }
}

static void q21_conv2d_stride2(const q21_conv *conv, const float *x, int height, int width,
                               float *y, int out_h, int out_w, int threads) {
    q21_conv_stride2_job job = {conv, x, y, height, width, out_h, out_w};
    q21_parallel(conv->out_c, threads, q21_conv_channels_stride2, &job);
}

/* x * sqrt(C) / ||x||_2(C) * gamma + bias, per spatial position. */
static void q21_vae_rms(const float *x, int channels, int height, int width,
                        const float *gamma, const float *bias, float *y) {
    const int plane = height * width;
    const float scale = sqrtf((float)channels);
    for (int index = 0; index < plane; index++) {
        double sum = 0.0;
        for (int c = 0; c < channels; c++) {
            const double value = x[(size_t)c * plane + index];
            sum += value * value;
        }
        float norm = (float)sqrt(sum);
        if (norm < 1e-12f) norm = 1e-12f;
        const float factor = scale / norm;
        for (int c = 0; c < channels; c++) {
            float value = x[(size_t)c * plane + index] * factor * gamma[c];
            if (bias) value += bias[c];
            y[(size_t)c * plane + index] = value;
        }
    }
}

static void q21_vae_silu(float *x, size_t count) { q21_silu(x, count); }

/* Stage checksum, enabled with QZEN_QWEN21_VAE_STAGES=1. The format matches
 * tools/qwen21_vae_stages.py so the two tables can be diffed directly. */
static void q21_vae_stage_log(const char *name, const float *x, size_t count) {
    if (!getenv("QZEN_QWEN21_VAE_STAGES")) return;
    double sum = 0.0;
    float absmax = 0.0f;
    for (size_t index = 0; index < count; index++) {
        sum += x[index];
        const float magnitude = fabsf(x[index]);
        if (magnitude > absmax) absmax = magnitude;
    }
    fprintf(stderr, "%-12s n=%zu sum=%.6e mean=%.6e absmax=%.6e head %.9g %.9g %.9g %.9g\n",
            name, count, sum, sum / (double)count, absmax,
            count > 0 ? x[0] : 0.0f, count > 1 ? x[1] : 0.0f,
            count > 2 ? x[2] : 0.0f, count > 3 ? x[3] : 0.0f);
    fflush(stderr);
}

/* DupUp3D with first_chunk. Verified against torch (tools/qwen21_vae_stages.py):
 * out[o, 2y+hy, 2x+wx] = in[(o*factor + frame*4 + hy*2 + wx) / repeats, y, x]
 * where factor = factor_t * 4, repeats = out_c * factor / in_c, and the
 * first_chunk crop keeps the last of the factor_t duplicated frames
 * (frame = factor_t - 1 on the single-frame image path). */
static void q21_vae_dupup(const float *x, int in_c, int out_c, int temporal, int height,
                          int width, float *y) {
    const int factor_t = temporal ? 2 : 1;
    const int factor = factor_t * 4;
    const int repeats = out_c * factor / in_c;
    const int plane = height * width;
    const int frame = temporal ? factor_t - 1 : 0; /* first_chunk keeps the last frame */
    for (int o = 0; o < out_c; o++) {
        float *dst = y + (size_t)o * plane * 4;
        for (int hy = 0; hy < 2; hy++) {
            for (int wx = 0; wx < 2; wx++) {
                const int flat = o * factor + frame * 4 + hy * 2 + wx;
                const float *src = x + (size_t)(flat / repeats) * plane;
                for (int index_y = 0; index_y < height; index_y++) {
                    float *row = dst + (size_t)(index_y * 2 + hy) * (width * 2) + wx;
                    const float *source = src + (size_t)index_y * width;
                    for (int index_x = 0; index_x < width; index_x++) {
                        row[index_x * 2] = source[index_x];
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------- resnet */

static void q21_vae_resnet(const q21_resnet *res, const float *x, int height, int width,
                           float *out, float *scratch1, float *scratch2, int threads) {
    const int plane = height * width;
    if (res->shortcut.weight) {
        q21_conv2d(&res->shortcut, x, height, width, out, threads);
    } else {
        memcpy(out, x, (size_t)res->out_c * plane * sizeof(float));
    }
    q21_vae_rms(x, res->in_c, height, width, res->gamma1, NULL, scratch1);
    q21_vae_silu(scratch1, (size_t)res->in_c * plane);
    q21_conv2d(&res->conv1, scratch1, height, width, scratch2, threads);
    q21_vae_rms(scratch2, res->out_c, height, width, res->gamma2, NULL, scratch1);
    q21_vae_silu(scratch1, (size_t)res->out_c * plane);
    q21_conv2d(&res->conv2, scratch1, height, width, scratch2, threads);
    for (size_t index = 0; index < (size_t)res->out_c * plane; index++) {
        out[index] += scratch2[index];
    }
}

/* QwenImage21AvgDown3D: the shortcut of a residual down block. It is a pure
 * reshape-and-mean -- no weights at all -- so it is easy to leave out and hard
 * to notice: the encoder still produces a picture, just not the right one.
 *
 *   out[o] = mean over group_size of the (channels x factor_t x 2 x 2) window
 *   flat index = c * (ft*fs*fs) + t * (fs*fs) + dy * fs + dx
 *   group_size = in_c * ft * fs * fs / out_c        (= 4 here, 1 on the last
 *                                                    block, where it is identity)
 *
 * factor_t is 2 on the middle blocks: the single frame is padded with one zero
 * frame in FRONT (F.pad(..., pad_t, 0)), so the even channel of each pair is
 * zero and the odd one is the spatial mean. Getting that ordering wrong shifts
 * every channel by one and the round trip collapses to a colour cast. */
typedef struct {
    const float *x;
    float *y;
    int factor, group, ft, fs;
    int height, width, out_h, out_w;
} q21_avg_down_job;

static void q21_avg_down_channels(void *context, int slot, int start, int end) {
    (void)slot;
    const q21_avg_down_job *job = context;
    const int fs = job->fs, ft = job->ft, group = job->group, factor = job->factor;
    const int in_plane = job->height * job->width;
    const int out_plane = job->out_h * job->out_w;
    const float inverse = 1.0f / (float)group;
    for (int o = start; o < end; o++) {
        float *dst = job->y + (size_t)o * out_plane;
        memset(dst, 0, (size_t)out_plane * sizeof(float));
        for (int g = 0; g < group; g++) {
            const int flat = o * group + g;
            const int channel = flat / factor;
            const int rest = flat % factor;
            const int t = rest / (fs * fs);
            const int inner = rest % (fs * fs);
            const int dy = inner / fs;
            const int dx = inner % fs;
            if (ft == 2 && t == 0) continue; /* the padded zero frame */
            const float *src = job->x + (size_t)channel * in_plane;
            for (int yy = 0; yy < job->out_h; yy++) {
                const int sy = yy * fs + dy;
                if (sy >= job->height) continue;
                const float *row = src + (size_t)sy * job->width;
                float *out_row = dst + (size_t)yy * job->out_w;
                for (int xx = 0; xx < job->out_w; xx++) {
                    const int sx = xx * fs + dx;
                    if (sx >= job->width) continue;
                    out_row[xx] += row[sx];
                }
            }
        }
        for (int index = 0; index < out_plane; index++) dst[index] *= inverse;
    }
}

static void q21_avg_down_shortcut(const int in_c, const int out_c, int ft, int fs,
                                  const float *x, int height, int width, float *y,
                                  int out_h, int out_w, int threads) {
    const int factor = ft * fs * fs;
    q21_avg_down_job job = {x,        y,
                            factor,   in_c * factor / out_c,
                            ft,       fs,
                            height,   width,
                            out_h,    out_w};
    q21_parallel(out_c, threads, q21_avg_down_channels, &job);
}

/* ------------------------------------------------------------- attention */

typedef struct {
    const float *q, *k, *v;
    float *out;
    float *scores; /* [threads][positions] */
    int channels, positions;
    float scale;
} q21_vae_attention_job;

/* Single head over the spatial positions, scale 1/sqrt(channels); the value
 * accumulation walks the positions so q/k/v stay in [channels][positions]. */
static void q21_vae_attention_rows(void *context, int slot, int start, int end) {
    q21_vae_attention_job *job = context;
    float *weights = job->scores + (size_t)slot * (size_t)job->positions;
    for (int index = start; index < end; index++) {
        const float *query = job->q + index;
        float best = -INFINITY;
        for (int other = 0; other < job->positions; other++) {
            const float *key = job->k + other;
            float sum = 0.0f;
            for (int c = 0; c < job->channels; c++) {
                sum += query[(size_t)c * job->positions] * key[(size_t)c * job->positions];
            }
            sum *= job->scale;
            weights[other] = sum;
            if (sum > best) best = sum;
        }
        float total = 0.0f;
        for (int other = 0; other < job->positions; other++) {
            float value = expf(weights[other] - best);
            weights[other] = value;
            total += value;
        }
        const float inverse = 1.0f / total;
        for (int c = 0; c < job->channels; c++) {
            float sum = 0.0f;
            for (int other = 0; other < job->positions; other++) {
                sum += weights[other] * job->v[(size_t)c * job->positions + other];
            }
            job->out[(size_t)c * job->positions + index] = sum * inverse;
        }
    }
}


/* ------------------------------------------------------------- decoding */

/* One ResidualUpBlock. The caller owns *holder ([width[block]][h][w] on entry,
 * [channels[block]][2h][2w] after an upsampling block). */
static int q21_vae_up_block(qzen_qwen21_vae *vae, int block, float **holder, int *height,
                            int *width, char *error, size_t error_size) {
    const int in_c = vae->width[block], out_c = vae->channels[block];
    const int h = *height, w = *width, plane = h * w;
    const int wide = vae->upsamples[block] ? w * 2 : w;
    const int high = vae->upsamples[block] ? h * 2 : h;
    const size_t grown = (size_t)out_c * wide * high;
    float *x = *holder;
    float *copy = malloc((size_t)in_c * plane * sizeof(float));
    float *a = malloc(grown * sizeof(float));
    float *b = malloc(grown * sizeof(float));
    float *scratch = malloc((size_t)(in_c > out_c ? in_c : out_c) * plane * sizeof(float));
    int ok = 0;
    if (!copy || !a || !b || !scratch) {
        snprintf(error, error_size, "out of memory for up block %d", block);
        goto done;
    }
    memcpy(copy, x, (size_t)in_c * plane * sizeof(float));
    for (int r = 0; r < Q21_VAE_RESNETS; r++) {
        q21_vae_resnet(&vae->up[block][r], x, h, w, a, scratch, b, vae->threads);
        memcpy(x, a, (size_t)out_c * plane * sizeof(float));
    }
    if (vae->upsamples[block]) {
        for (int c = 0; c < out_c; c++) {
            const float *src = x + (size_t)c * plane;
            float *dst = b + (size_t)c * wide * high;
            for (int y = 0; y < high; y++) {
                const float *row = src + (size_t)(y / 2) * w;
                float *target = dst + (size_t)y * wide;
                for (int xx = 0; xx < wide; xx++) target[xx] = row[xx / 2];
            }
        }
        q21_conv2d(&vae->up_conv[block], b, high, wide, a, vae->threads);
        q21_vae_dupup(copy, in_c, out_c, vae->temporal[block], h, w, b);
        for (size_t index = 0; index < grown; index++) a[index] += b[index];
    } else {
        memcpy(a, x, grown * sizeof(float));
    }
    free(*holder);
    *holder = a;
    a = NULL;
    *height = high;
    *width = wide;
    ok = 1;
done:
    free(copy);
    free(a);
    free(b);
    free(scratch);
    return ok;
}

int qzen_qwen21_vae_decode(qzen_qwen21_vae *vae, const float *latent, int height, int width,
                         unsigned char *rgb, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!vae || !latent || !rgb || height <= 0 || width <= 0) {
        snprintf(error, error_size, "invalid VAE decode request");
        return 0;
    }
    if (q21_vae_gpu_requested()) {
        if (!vae->gpu) vae->gpu = q21_vae_gpu_create(vae, error, error_size);
        if (!vae->gpu) return 0;
        return q21_vae_gpu_decode(vae, vae->gpu, latent, height, width, rgb,
                                  error, error_size);
    }
    const int threads = vae->threads;
    const int plane = height * width;
    const int channels = Q21_VAE_WIDE;
    const size_t wide = (size_t)channels * plane;
    int ok = 0;
    float *post = malloc((size_t)Q21_VAE_CHANNELS * plane * sizeof(float));
    float *x = malloc(wide * sizeof(float));
    float *a = malloc(wide * sizeof(float));
    float *b = malloc(wide * sizeof(float));
    float *c = malloc(wide * sizeof(float));
    float *attended = NULL, *scores = NULL, *qkv = NULL;
    if (!post || !x || !a || !b || !c) {
        snprintf(error, error_size, "out of memory for the %dx%d latent decode", height, width);
        goto done;
    }
    q21_conv2d(&vae->post_quant, latent, height, width, post, threads);
    q21_vae_stage_log("post_quant", post, (size_t)Q21_VAE_CHANNELS * plane);
    q21_conv2d(&vae->conv_in, post, height, width, x, threads);
    q21_vae_stage_log("conv_in", x, wide);
    free(post);
    post = NULL;

    /* mid: resnet -> single head attention -> resnet */
    q21_vae_resnet(&vae->mid[0], x, height, width, a, b, c, threads);
    memcpy(x, a, wide * sizeof(float));
    q21_vae_rms(x, channels, height, width, vae->attn_gamma, NULL, a);
    qkv = malloc(3 * wide * sizeof(float));
    attended = malloc(wide * sizeof(float));
    scores = malloc((size_t)threads * (size_t)plane * sizeof(float));
    if (!qkv || !attended || !scores) {
        snprintf(error, error_size, "out of memory for the mid attention");
        goto done;
    }
    q21_conv2d(&vae->attn_qkv, a, height, width, qkv, threads); /* [3C][h][w] */
    {
        const size_t stride = (size_t)plane;
        q21_vae_attention_job job = {qkv,          qkv + channels * stride,
                                     qkv + 2 * channels * stride, attended, scores,
                                     channels,     plane,
                                     1.0f / sqrtf((float)channels)};
        q21_parallel(plane, threads, q21_vae_attention_rows, &job);
    }
    free(qkv);
    qkv = NULL;
    q21_conv2d(&vae->attn_proj, attended, height, width, a, threads);
    for (size_t index = 0; index < wide; index++) x[index] += a[index];
    free(attended);
    free(scores);
    attended = NULL;
    scores = NULL;
    q21_vae_resnet(&vae->mid[1], x, height, width, a, b, c, threads);
    memcpy(x, a, wide * sizeof(float));

    int current_h = height, current_w = width;
    for (int block = 0; block < Q21_VAE_BLOCKS; block++) {
        if (!q21_vae_up_block(vae, block, &x, &current_h, &current_w, error, error_size)) {
            goto done;
        }
        char stage[32];
        snprintf(stage, sizeof(stage), "up_blocks.%d", block);
        q21_vae_stage_log(stage, x, (size_t)vae->channels[block] * current_h * current_w);
    }
    /* norm_out RMS(144) -> SiLU -> conv_out 4 channels -> RGB8 */
    {
        const int last = vae->channels[Q21_VAE_BLOCKS - 1];
        const size_t pixels = (size_t)current_h * current_w;
        const size_t count = (size_t)last * pixels;
        /* a and b are still sized for the mid block (1152 * latent plane); the
         * tail runs at 144 * output plane, which is up to 32x larger. */
        float *grown_a = realloc(a, count * sizeof(float));
        float *grown_b = realloc(b, count * sizeof(float));
        if (!grown_a || !grown_b) {
            snprintf(error, error_size, "out of memory for the decode tail");
            goto done;
        }
        a = grown_a;
        b = grown_b;
        q21_vae_rms(x, last, current_h, current_w, vae->norm_out_gamma, NULL, a);
        q21_vae_stage_log("norm_out", a, count);
        q21_vae_silu(a, count);
        q21_conv2d(&vae->conv_out, a, current_h, current_w, b, threads);
        q21_vae_stage_log("conv_out", b, (size_t)vae->conv_out.out_c * pixels);
        for (size_t index = 0; index < pixels; index++) {
            for (int channel = 0; channel < 3; channel++) {
                float value = b[(size_t)channel * pixels + index];
                value = value < -1.0f ? -1.0f : value > 1.0f ? 1.0f : value;
                value = (value + 1.0f) * 127.5f;
                int quantized = (int)(value + 0.5f);
                if (quantized < 0) quantized = 0;
                if (quantized > 255) quantized = 255;
                rgb[(size_t)channel * pixels + index] = (unsigned char)quantized;
            }
        }
    }
    ok = 1;
done:
    free(post);
    free(x);
    free(a);
    free(b);
    free(c);
    free(attended);
    free(scores);
    free(qkv);
    return ok;
}

/* ---------------------------------------------------------------- encoding */

/* The attention norm gamma of the encoder mid block is stored as [C,1,1]
 * (3 dims), which the generic gamma loader does not try for 768 channels. */
static float *q21_enc_gamma3(qzen_qwen21_vae *vae, const char *name, int channels, char *error,
                             size_t error_size) {
    char tensor[192];
    snprintf(tensor, sizeof(tensor), "%s.gamma", name);
    const uint64_t shape3[3] = {(uint64_t)channels, 1, 1};
    return q21_read_f32(&vae->header, tensor, 3, shape3, error, error_size);
}

static void q21_encoder_free(q21_encoder *enc) {
    q21_conv *convs[] = {&enc->conv_in, &enc->conv_out, &enc->quant_conv,
                         &enc->attn_qkv, &enc->attn_proj};
    for (size_t index = 0; index < sizeof(convs) / sizeof(convs[0]); index++) {
        free(convs[index]->weight);
        free(convs[index]->bias);
    }
    for (int block = 0; block < Q21_ENC_BLOCKS; block++) {
        for (int r = 0; r < 2; r++) {
            q21_resnet *res = &enc->down[block][r];
            free(res->conv1.weight);
            free(res->conv1.bias);
            free(res->conv2.weight);
            free(res->conv2.bias);
            free(res->shortcut.weight);
            free(res->shortcut.bias);
            free(res->gamma1);
            free(res->gamma2);
        }
        free(enc->down_conv[block].weight);
        free(enc->down_conv[block].bias);
    }
    for (int index = 0; index < 2; index++) {
        q21_resnet *res = &enc->mid[index];
        free(res->conv1.weight);
        free(res->conv1.bias);
        free(res->conv2.weight);
        free(res->conv2.bias);
        free(res->shortcut.weight);
        free(res->shortcut.bias);
        free(res->gamma1);
        free(res->gamma2);
    }
    free(enc->attn_gamma);
    free(enc->norm_out_gamma);
    memset(enc, 0, sizeof(*enc));
}

static int q21_encoder_init(qzen_qwen21_vae *vae, char *error, size_t error_size) {
    q21_encoder *enc = &vae->enc;
    if (enc->ready) return 1;
    static const int in_c[Q21_ENC_BLOCKS] = {96, 96, 192, 384, 768};
    static const int out_c[Q21_ENC_BLOCKS] = {96, 192, 384, 768, 768};
    static const int downsample[Q21_ENC_BLOCKS] = {1, 1, 1, 1, 0};
    static const int temporal[Q21_ENC_BLOCKS] = {0, 1, 1, 1, 0};
    for (int block = 0; block < Q21_ENC_BLOCKS; block++) {
        enc->in_c[block] = in_c[block];
        enc->out_c[block] = out_c[block];
        enc->downsample[block] = downsample[block];
        enc->temporal[block] = temporal[block];
    }
    char name[192];
    int ok = q21_vae_conv_load(vae, &enc->conv_in, "encoder.conv_in", in_c[0], Q21_ENC_INPUT,
                               3, error, error_size);
    for (int block = 0; ok && block < Q21_ENC_BLOCKS; block++) {
        for (int r = 0; ok && r < 2; r++) {
            const int block_in = r == 0 ? in_c[block] : out_c[block];
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.resnets.%d", block, r);
            ok = q21_vae_resnet_load(vae, &enc->down[block][r], name, block_in,
                                     out_c[block], error, error_size);
        }
        if (ok && downsample[block]) {
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.downsampler.resample.1",
                     block);
            ok = q21_vae_conv_load(vae, &enc->down_conv[block], name, out_c[block],
                                   out_c[block], 3, error, error_size);
        }
    }
    for (int index = 0; ok && index < 2; index++) {
        snprintf(name, sizeof(name), "encoder.mid_block.resnets.%d", index);
        ok = q21_vae_resnet_load(vae, &enc->mid[index], name, Q21_ENC_WIDE, Q21_ENC_WIDE,
                                 error, error_size);
    }
    if (ok) {
        enc->attn_gamma = q21_enc_gamma3(vae, "encoder.mid_block.attentions.0.norm",
                                         Q21_ENC_WIDE, error, error_size);
        ok = enc->attn_gamma != NULL &&
             q21_vae_conv_load(vae, &enc->attn_qkv, "encoder.mid_block.attentions.0.to_qkv",
                               3 * Q21_ENC_WIDE, Q21_ENC_WIDE, 1, error, error_size) &&
             q21_vae_conv_load(vae, &enc->attn_proj, "encoder.mid_block.attentions.0.proj",
                               Q21_ENC_WIDE, Q21_ENC_WIDE, 1, error, error_size);
    }
    if (ok) {
        enc->norm_out_gamma = q21_vae_gamma_load(vae, "encoder.norm_out", Q21_ENC_WIDE,
                                                 error, error_size);
        ok = enc->norm_out_gamma != NULL &&
             q21_vae_conv_load(vae, &enc->conv_out, "encoder.conv_out", Q21_ENC_OUT,
                               Q21_ENC_WIDE, 3, error, error_size) &&
             q21_vae_conv_load(vae, &enc->quant_conv, "quant_conv", Q21_ENC_OUT, Q21_ENC_OUT,
                               1, error, error_size);
    }
    if (!ok) {
        q21_encoder_free(enc);
        return 0;
    }
    enc->ready = 1;
    return 1;
}

int qzen_qwen21_vae_encode(qzen_qwen21_vae *vae, const unsigned char *rgb, int height, int width,
                         float *latent, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!vae || !rgb || !latent || height <= 0 || width <= 0) {
        snprintf(error, error_size, "invalid VAE encode request");
        return 0;
    }
    if (height % 16 || width % 16) {
        snprintf(error, error_size,
                 "the VAE encodes on a 16-pixel grid (%dx%d given)", width, height);
        return 0;
    }
    if (!q21_encoder_init(vae, error, error_size)) return 0;
    const q21_encoder *enc = &vae->enc;
    const int threads = vae->threads;

    int h = height, w = width;
    const size_t in_count = (size_t)Q21_ENC_INPUT * h * w;
    float *x = malloc(in_count * sizeof(float));
    float *a = NULL, *b = NULL, *c = NULL, *scratch = NULL;
    float *qkv = NULL, *attended = NULL, *scores = NULL;
    int ok = 0;
    if (!x) {
        snprintf(error, error_size, "out of memory for the %dx%d encode", width, height);
        return 0;
    }
    /* RGB8 -> [-1,1] plus the 4th (alpha) channel. WanGP passes
     * image.convert("RGBA") to this encoder; an opaque reference has alpha
     * 255, which maps to 255/127.5 - 1 = 1.0 here. The inpainting mask is NOT
     * a VAE input: it is binarised onto the latent grid and used by the
     * sampler's masked-denoising loop instead. */
    const size_t pixels = (size_t)h * w;
    for (size_t index = 0; index < pixels; index++) {
        for (int channel = 0; channel < 3; channel++) {
            x[(size_t)channel * pixels + index] =
                (float)rgb[(size_t)channel * pixels + index] * (2.0f / 255.0f) - 1.0f;
        }
        x[(size_t)3 * pixels + index] = 1.0f;
    }
    size_t wide = (size_t)Q21_ENC_WIDE * pixels;
    a = malloc(wide * sizeof(float));
    b = malloc(wide * sizeof(float));
    c = malloc(wide * sizeof(float));
    scratch = malloc(wide * sizeof(float));
    if (!a || !b || !c || !scratch) {
        snprintf(error, error_size, "out of memory for the encode stages");
        goto done;
    }
    q21_conv2d(&enc->conv_in, x, h, w, a, threads);
    q21_vae_stage_log("enc.conv_in", a, (size_t)enc->in_c[0] * pixels);
    free(x);
    x = NULL;

    for (int block = 0; block < Q21_ENC_BLOCKS; block++) {
        const int channels = enc->out_c[block];
        const int down = enc->downsample[block];
        /* factor_t: the encoder pads one zero frame in front of the middle
         * blocks' shortcut (temperal_downsample = [F,T,T,T]); factor_s halves
         * the canvas on every block that downsamples. */
        const int ft = (block != Q21_ENC_BLOCKS - 1 && enc->temporal[block]) ? 2 : 1;
        const int fs = down ? 2 : 1;
        const int out_h = down ? h / 2 : h;
        const int out_w = down ? w / 2 : w;
        /* The shortcut reads the block input, and the resnets below overwrite
         * `a` in place, so it has to be computed first. */
        float *shortcut = malloc((size_t)channels * out_h * out_w * sizeof(float));
        if (!shortcut) {
            snprintf(error, error_size, "out of memory for encode shortcut %d", block);
            goto done;
        }
        q21_avg_down_shortcut(enc->in_c[block], channels, ft, fs, a, h, w, shortcut, out_h,
                              out_w, threads);
        for (int r = 0; r < 2; r++) {
            /* resnet reads a, writes b, and needs two scratch planes of its own:
             * scratch for the normalised copy and c for conv1's output. */
            q21_vae_resnet(&enc->down[block][r], a, h, w, b, scratch, c, threads);
            memcpy(a, b, (size_t)channels * h * w * sizeof(float));
        }
        if (down) {
            /* The next block's resnet writes out_c[block + 1] planes into `a`
             * through the memcpy above, and out_c doubles while the canvas
             * quarters -- so the plane has to be sized for the *next* block,
             * not for the channels the downsampler just produced. Sizing it for
             * `channels` overran the heap on block 1 (192 planes into a 96-plane
             * buffer), which is what the first encode run crashed on. */
            const int next_c = block + 1 < Q21_ENC_BLOCKS ? enc->out_c[block + 1] : channels;
            const size_t grown_count = (size_t)(next_c > channels ? next_c : channels) *
                                       (size_t)out_h * (size_t)out_w;
            float *grown = malloc(grown_count * sizeof(float));
            if (!grown) {
                free(shortcut);
                snprintf(error, error_size, "out of memory for encode block %d", block);
                goto done;
            }
            q21_conv2d_stride2(&enc->down_conv[block], a, h, w, grown, out_h, out_w, threads);
            free(a);
            a = grown;
            h = out_h;
            w = out_w;
        }
        for (size_t index = 0; index < (size_t)channels * out_h * out_w; index++)
            a[index] += shortcut[index];
        free(shortcut);
        char stage[32];
        snprintf(stage, sizeof(stage), "enc.down.%d", block);
        q21_vae_stage_log(stage, a, (size_t)channels * h * w);
    }

    /* mid: resnet -> single head attention -> resnet, at 1/16 resolution */
    {
        const size_t plane = (size_t)h * w;
        const size_t count = (size_t)Q21_ENC_WIDE * plane;
        b = realloc(b, count * sizeof(float));
        scratch = realloc(scratch, count * sizeof(float));
        if (!b || !scratch) {
            snprintf(error, error_size, "out of memory for the encode mid block");
            goto done;
        }
        q21_vae_resnet(&enc->mid[0], a, h, w, b, scratch, c, threads);
        memcpy(a, b, count * sizeof(float));
        q21_vae_rms(a, Q21_ENC_WIDE, h, w, enc->attn_gamma, NULL, b);
        qkv = malloc(3 * count * sizeof(float));
        attended = malloc(count * sizeof(float));
        scores = malloc((size_t)threads * plane * sizeof(float));
        if (!qkv || !attended || !scores) {
            snprintf(error, error_size, "out of memory for the encode attention");
            goto done;
        }
        q21_conv2d(&enc->attn_qkv, b, h, w, qkv, threads);
        {
            const size_t stride = plane;
            q21_vae_attention_job job = {qkv,
                                         qkv + Q21_ENC_WIDE * stride,
                                         qkv + 2 * Q21_ENC_WIDE * stride,
                                         attended, scores, Q21_ENC_WIDE,
                                         (int)plane,
                                         1.0f / sqrtf((float)Q21_ENC_WIDE)};
            q21_parallel((int)plane, threads, q21_vae_attention_rows, &job);
        }
        free(qkv);
        qkv = NULL;
        q21_conv2d(&enc->attn_proj, attended, h, w, b, threads);
        for (size_t index = 0; index < count; index++) a[index] += b[index];
        free(attended);
        free(scores);
        attended = NULL;
        scores = NULL;
        q21_vae_resnet(&enc->mid[1], a, h, w, b, scratch, c, threads);
        memcpy(a, b, count * sizeof(float));
        q21_vae_stage_log("enc.mid", a, count);
    }

    /* norm_out RMS(768) -> SiLU -> conv_out 768->128 -> quant_conv 128->128 */
    {
        const size_t plane = (size_t)h * w;
        float *out = malloc((size_t)Q21_ENC_OUT * plane * sizeof(float));
        float *post = malloc((size_t)Q21_ENC_OUT * plane * sizeof(float));
        if (!out || !post) {
            free(out);
            free(post);
            snprintf(error, error_size, "out of memory for the encode tail");
            goto done;
        }
        q21_vae_rms(a, Q21_ENC_WIDE, h, w, enc->norm_out_gamma, NULL, b);
        q21_vae_silu(b, (size_t)Q21_ENC_WIDE * plane);
        q21_conv2d(&enc->conv_out, b, h, w, out, threads);
        q21_conv2d(&enc->quant_conv, out, h, w, post, threads);
        q21_vae_stage_log("enc.quant_conv", post, (size_t)Q21_ENC_OUT * plane);
        /* The sampler works in the normalised latent space: z = (mean - mu)/sd. */
        for (int channel = 0; channel < Q21_VAE_CHANNELS; channel++) {
            const float mean = vae->latent_mean ? vae->latent_mean[channel] : 0.0f;
            const float std = vae->latent_std ? vae->latent_std[channel] : 1.0f;
            const float inverse = std != 0.0f ? 1.0f / std : 1.0f;
            const float *src = post + (size_t)channel * plane;
            float *dst = latent + (size_t)channel * plane;
            for (size_t index = 0; index < plane; index++) {
                dst[index] = (src[index] - mean) * inverse;
            }
        }
        free(out);
        free(post);
    }
    ok = 1;
done:
    free(x);
    free(a);
    free(b);
    free(c);
    free(scratch);
    free(qkv);
    free(attended);
    free(scores);
    return ok;
}



typedef struct {
    qzen_gpu_tensor *weight, *bias;
    int out_c, in_c, k;
} q21_gpu_conv;

typedef struct {
    q21_gpu_conv conv1, conv2, shortcut;
    qzen_gpu_tensor *gamma1, *gamma2;
    int in_c, out_c;
} q21_gpu_resnet;

struct q21_vae_gpu {
    qzen_gpu *gpu;
    q21_gpu_conv post_quant, conv_in, conv_out;
    q21_gpu_resnet mid[2];
    q21_gpu_conv attn_qkv, attn_proj;
    qzen_gpu_tensor *attn_gamma;
    q21_gpu_resnet up[Q21_VAE_BLOCKS][Q21_VAE_RESNETS];
    q21_gpu_conv up_conv[Q21_VAE_BLOCKS];
    qzen_gpu_tensor *norm_out_gamma;
};

static int q21_vae_gpu_requested(void) {
    const char *value = getenv("QWEN21_VAE_GPU");
    return value && *value && strcmp(value, "0") != 0;
}

static int q21_gpu_conv_load(qzen_gpu *gpu, q21_gpu_conv *device,
                             const q21_conv *host) {
    memset(device, 0, sizeof(*device));
    device->out_c = host->out_c;
    device->in_c = host->in_c;
    device->k = host->k;
    if (!host->weight || !host->bias) return 1; /* empty slot */
    device->weight = qzen_gpu_tensor_from_f32(
        gpu, host->weight,
        (size_t)host->out_c * host->in_c * host->k * host->k);
    device->bias = qzen_gpu_tensor_from_f32(gpu, host->bias, (size_t)host->out_c);
    return device->weight && device->bias;
}

static void q21_gpu_conv_free(q21_gpu_conv *conv) {
    if (conv->weight) qzen_gpu_tensor_free(conv->weight);
    if (conv->bias) qzen_gpu_tensor_free(conv->bias);
    conv->weight = conv->bias = NULL;
}

static int q21_gpu_resnet_load(qzen_gpu *gpu, q21_gpu_resnet *device,
                               const q21_resnet *host) {
    memset(device, 0, sizeof(*device));
    device->in_c = host->in_c;
    device->out_c = host->out_c;
    device->gamma1 = qzen_gpu_tensor_from_f32(gpu, host->gamma1, (size_t)host->in_c);
    device->gamma2 = qzen_gpu_tensor_from_f32(gpu, host->gamma2, (size_t)host->out_c);
    return device->gamma1 && device->gamma2 &&
           q21_gpu_conv_load(gpu, &device->conv1, &host->conv1) &&
           q21_gpu_conv_load(gpu, &device->conv2, &host->conv2) &&
           q21_gpu_conv_load(gpu, &device->shortcut, &host->shortcut);
}

static void q21_gpu_resnet_free(q21_gpu_resnet *resnet) {
    if (resnet->gamma1) qzen_gpu_tensor_free(resnet->gamma1);
    if (resnet->gamma2) qzen_gpu_tensor_free(resnet->gamma2);
    resnet->gamma1 = resnet->gamma2 = NULL;
    q21_gpu_conv_free(&resnet->conv1);
    q21_gpu_conv_free(&resnet->conv2);
    q21_gpu_conv_free(&resnet->shortcut);
}

static void q21_vae_gpu_free(q21_vae_gpu *state) {
    if (!state) return;
    q21_gpu_conv_free(&state->post_quant);
    q21_gpu_conv_free(&state->conv_in);
    q21_gpu_conv_free(&state->conv_out);
    q21_gpu_conv_free(&state->attn_qkv);
    q21_gpu_conv_free(&state->attn_proj);
    if (state->attn_gamma) qzen_gpu_tensor_free(state->attn_gamma);
    if (state->norm_out_gamma) qzen_gpu_tensor_free(state->norm_out_gamma);
    for (int b = 0; b < Q21_VAE_BLOCKS; b++) {
        for (int r = 0; r < Q21_VAE_RESNETS; r++)
            q21_gpu_resnet_free(&state->up[b][r]);
        q21_gpu_conv_free(&state->up_conv[b]);
    }
    for (int i = 0; i < 2; i++) q21_gpu_resnet_free(&state->mid[i]);
    qzen_gpu_free(state->gpu);
    free(state);
}

static q21_vae_gpu *q21_vae_gpu_create(qzen_qwen21_vae *vae, char *error,
                                       size_t error_size) {
    q21_vae_gpu *state = calloc(1, sizeof(*state));
    if (!state) {
        snprintf(error, error_size, "cannot allocate the Qwen-Image VAE GPU state");
        return NULL;
    }
    state->gpu = qzen_gpu_create(NULL, error, error_size);
    if (!state->gpu) {
        free(state);
        return NULL;
    }
    qzen_gpu *gpu = state->gpu;
    int ok = q21_gpu_conv_load(gpu, &state->post_quant, &vae->post_quant) &&
             q21_gpu_conv_load(gpu, &state->conv_in, &vae->conv_in) &&
             q21_gpu_conv_load(gpu, &state->conv_out, &vae->conv_out) &&
             q21_gpu_conv_load(gpu, &state->attn_qkv, &vae->attn_qkv) &&
             q21_gpu_conv_load(gpu, &state->attn_proj, &vae->attn_proj) &&
             q21_gpu_resnet_load(gpu, &state->mid[0], &vae->mid[0]) &&
             q21_gpu_resnet_load(gpu, &state->mid[1], &vae->mid[1]);
    if (ok) {
        state->attn_gamma =
            qzen_gpu_tensor_from_f32(gpu, vae->attn_gamma, Q21_VAE_WIDE);
        state->norm_out_gamma = qzen_gpu_tensor_from_f32(
            gpu, vae->norm_out_gamma, (size_t)vae->channels[Q21_VAE_BLOCKS - 1]);
        ok = state->attn_gamma && state->norm_out_gamma;
    }
    for (int b = 0; ok && b < Q21_VAE_BLOCKS; b++) {
        for (int r = 0; ok && r < Q21_VAE_RESNETS; r++)
            ok = q21_gpu_resnet_load(gpu, &state->up[b][r], &vae->up[b][r]);
        if (ok) ok = q21_gpu_conv_load(gpu, &state->up_conv[b], &vae->up_conv[b]);
    }
    if (!ok) {
        if (!error[0])
            snprintf(error, error_size, "Qwen-Image VAE weight upload failed: %s",
                     qzen_gpu_error(gpu));
        q21_vae_gpu_free(state);
        return NULL;
    }
    return state;
}

static int q21_gpu_check(int result, const char *what, qzen_gpu *gpu, char *error,
                         size_t error_size) {
    if (!result)
        snprintf(error, error_size, "VAE GPU decode failed: %s: %s", what,
                 qzen_gpu_error(gpu));
    return result;
}

/* ResnetBlock2D on the device. output must not alias input: the shortcut is
 * written first and the main path then reads input again. */
static int q21_gpu_resnet_forward(qzen_gpu *gpu, const q21_gpu_resnet *res,
                                  uint32_t height, uint32_t width,
                                  qzen_gpu_tensor *input, qzen_gpu_tensor *normed,
                                  qzen_gpu_tensor *scratch, qzen_gpu_tensor *output,
                                  char *error, size_t error_size) {
    const uint32_t in_c = (uint32_t)res->in_c, out_c = (uint32_t)res->out_c;
    const size_t in_count = (size_t)in_c * height * width;
    const size_t out_count = (size_t)out_c * height * width;
    if (res->shortcut.weight) {
        if (!q21_gpu_check(qzen_gpu_conv2d_f32(gpu, output, input,
                                             res->shortcut.weight,
                                             res->shortcut.bias, out_c, in_c,
                                             height, width, 1, 0),
                           "resnet shortcut", gpu, error, error_size))
            return 0;
    } else if (!q21_gpu_check(qzen_gpu_copy_f32(gpu, output, 0, input, 0, out_count),
                              "resnet identity", gpu, error, error_size)) {
        return 0;
    }
    if (!q21_gpu_check(qzen_gpu_vae_rms_chw_f32(gpu, normed, input, res->gamma1,
                                              in_c, height, width),
                       "resnet norm1", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_silu_f32(gpu, normed, normed, (uint32_t)in_count),
                       "resnet silu1", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_conv2d_f32(gpu, scratch, normed, res->conv1.weight,
                                         res->conv1.bias, out_c, in_c, height,
                                         width, 3, 1),
                       "resnet conv1", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_vae_rms_chw_f32(gpu, normed, scratch, res->gamma2,
                                              out_c, height, width),
                       "resnet norm2", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_silu_f32(gpu, normed, normed, (uint32_t)out_count),
                       "resnet silu2", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_conv2d_f32(gpu, scratch, normed, res->conv2.weight,
                                         res->conv2.bias, out_c, out_c, height,
                                         width, 3, 1),
                       "resnet conv2", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_add_f32(gpu, output, output, scratch,
                                      (uint32_t)out_count),
                       "resnet residual", gpu, error, error_size))
        return 0;
    return 1;
}

/* Mid-block spatial attention. The single 1x1 conv emits qkv as [3C][h][w];
 * the single-head SDPA wants [positions][channels] rows, so each slice is
 * copied out and transposed in, then the result is transposed back onto the
 * residual. Everything here is transient, so the helper owns its scratch:
 * nine channel planes plus the [positions][positions] score matrix, which is
 * 64 MiB at a 1024 px output (64x64 latent, 4096 positions). */
static int q21_gpu_mid_attention(qzen_gpu *gpu, const q21_vae_gpu *state,
                                 uint32_t channels, uint32_t height,
                                 uint32_t width, qzen_gpu_tensor *x,
                                 qzen_gpu_tensor *normed, char *error,
                                 size_t error_size) {
    const uint32_t plane = height * width;
    const size_t channel_count = (size_t)channels * plane;
    const float scale = 1.0f / sqrtf((float)channels);
    qzen_gpu_tensor *qkv = qzen_gpu_tensor_new_f32(gpu, 3 * channel_count);
    qzen_gpu_tensor *q = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *k = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *v = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *attended = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *transposed = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *slice = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *projected = qzen_gpu_tensor_new_f32(gpu, channel_count);
    qzen_gpu_tensor *scores = qzen_gpu_tensor_new_f32(gpu, (size_t)plane * plane);
    int ok = qkv && q && k && v && attended && transposed && slice &&
             projected && scores;
    if (!ok)
        snprintf(error, error_size, "out of GPU memory for the mid attention");
    if (ok) {
        qzen_gpu_tensor *parts[3] = {q, k, v};
        ok = q21_gpu_check(qzen_gpu_vae_rms_chw_f32(gpu, normed, x,
                                                  state->attn_gamma, channels,
                                                  height, width),
                           "mid attention norm", gpu, error, error_size) &&
             q21_gpu_check(qzen_gpu_conv2d_f32(gpu, qkv, normed,
                                             state->attn_qkv.weight,
                                             state->attn_qkv.bias, 3 * channels,
                                             channels, height, width, 1, 0),
                           "mid attention qkv", gpu, error, error_size);
        for (uint32_t part = 0; ok && part < 3; part++) {
            ok = q21_gpu_check(
                     qzen_gpu_copy_f32(gpu, slice, 0, qkv,
                                     (size_t)part * channel_count,
                                     channel_count),
                     "mid attention slice", gpu, error, error_size) &&
                 q21_gpu_check(qzen_gpu_transpose_f32(gpu, parts[part], slice,
                                                    channels, plane),
                               "mid attention transpose", gpu, error,
                               error_size);
        }
    }
    if (ok)
        ok = q21_gpu_check(qzen_gpu_vae_sdpa_f32(gpu, attended, q, k, v, scores,
                                               plane, channels, scale),
                           "mid attention sdpa", gpu, error, error_size);
    if (ok)
        ok = q21_gpu_check(qzen_gpu_transpose_f32(gpu, transposed, attended,
                                                plane, channels),
                           "mid attention untranspose", gpu, error, error_size);
    if (ok)
        ok = q21_gpu_check(qzen_gpu_conv2d_f32(gpu, projected, transposed,
                                             state->attn_proj.weight,
                                             state->attn_proj.bias, channels,
                                             channels, height, width, 1, 0),
                           "mid attention proj", gpu, error, error_size);
    if (ok)
        ok = q21_gpu_check(qzen_gpu_add_f32(gpu, x, x, projected,
                                          (uint32_t)channel_count),
                           "mid attention residual", gpu, error, error_size);
    qzen_gpu_tensor_free(qkv);
    qzen_gpu_tensor_free(q);
    qzen_gpu_tensor_free(k);
    qzen_gpu_tensor_free(v);
    qzen_gpu_tensor_free(attended);
    qzen_gpu_tensor_free(transposed);
    qzen_gpu_tensor_free(slice);
    qzen_gpu_tensor_free(projected);
    qzen_gpu_tensor_free(scores);
    return ok;
}

/* Device decode. Buffer roles follow the CPU reference:
 *
 *   P/Q  ping-pong through the three resnets of each up block; after the
 *        stack P is spent and takes the nearest-2x staging for the upsampler
 *   N    normalisation output inside a resnet
 *   S    conv output inside a resnet, then the 4-channel tail
 *   V    DupUp3D staging of the block input (written before the resnets,
 *        because the ping-pong overwrites the block input on the second pass)
 *
 * All five are claimed at the largest stage count so the allocator sees one
 * shape; the peak live set is P, Q, N, S and V at block 3 (288->144 with an
 * upsampler), which is 139024 * latent_plane floats against a
 * 184320 * latent_plane claim. At a 1024 px output that is 3.0 GB, against
 * 0.75 GB at 512 px. */
static int q21_vae_gpu_decode(qzen_qwen21_vae *vae, q21_vae_gpu *state,
                              const float *latent, int height, int width,
                              unsigned char *rgb, char *error,
                              size_t error_size) {
    qzen_gpu *gpu = state->gpu;
    const uint32_t latent_h = (uint32_t)height, latent_w = (uint32_t)width;
    const size_t plane = (size_t)height * width;
    const uint32_t last_c = (uint32_t)vae->channels[Q21_VAE_BLOCKS - 1];
    size_t max_count = (size_t)Q21_VAE_WIDE * plane;
    {
        size_t p = plane;
        for (int block = 0; block < Q21_VAE_BLOCKS; block++) {
            const size_t in_c = (size_t)vae->width[block];
            const size_t out_c = (size_t)vae->channels[block];
            if (in_c * p > max_count) max_count = in_c * p;
            if (out_c * p > max_count) max_count = out_c * p;
            if (vae->upsamples[block]) {
                p *= 4;
                if (out_c * p > max_count) max_count = out_c * p;
            }
        }
    }
    qzen_gpu_tensor *pool[5];
    qzen_gpu_tensor *latent_dev = NULL;
    float *staged = NULL;
    int ok = 0;
    memset(pool, 0, sizeof(pool));
    for (int slot = 0; slot < 5; slot++) {
        pool[slot] = qzen_gpu_tensor_new_f32(gpu, max_count);
        if (!pool[slot]) {
            snprintf(error, error_size, "out of GPU memory for the %dx%d VAE decode",
                     height, width);
            goto done;
        }
    }
    qzen_gpu_tensor *p = pool[0], *q = pool[1], *normed = pool[2];
    qzen_gpu_tensor *scratch = pool[3], *dup_buf = pool[4];

    latent_dev =
        qzen_gpu_tensor_new_f32(gpu, (size_t)Q21_VAE_CHANNELS * plane);
    staged = malloc((size_t)Q21_VAE_CHANNELS * plane * sizeof(float));
    if (!latent_dev || !staged) {
        snprintf(error, error_size, "out of memory for the %dx%d VAE decode",
                 height, width);
        goto done;
    }
    memcpy(staged, latent, (size_t)Q21_VAE_CHANNELS * plane * sizeof(float));
    if (!qzen_gpu_tensor_write_f32(latent_dev, staged,
                                 (size_t)Q21_VAE_CHANNELS * plane)) {
        snprintf(error, error_size, "cannot upload the VAE latent: %s",
                 qzen_gpu_error(gpu));
        goto done;
    }
    if (!q21_gpu_check(qzen_gpu_begin(gpu), "qzen_gpu_begin", gpu, error,
                       error_size))
        goto done;

    /* post_quant 1x1 -> conv_in 3x3, both at latent resolution. */
    if (!q21_gpu_check(qzen_gpu_conv2d_f32(gpu, normed, latent_dev,
                                         state->post_quant.weight,
                                         state->post_quant.bias,
                                         Q21_VAE_CHANNELS, Q21_VAE_CHANNELS,
                                         latent_h, latent_w, 1, 0),
                       "post_quant_conv", gpu, error, error_size) ||
        !q21_gpu_check(qzen_gpu_conv2d_f32(gpu, p, normed, state->conv_in.weight,
                                         state->conv_in.bias, Q21_VAE_WIDE,
                                         Q21_VAE_CHANNELS, latent_h, latent_w,
                                         3, 1),
                       "decoder.conv_in", gpu, error, error_size))
        goto done;

    /* mid: resnet -> attention -> resnet, all at 1152 channels. */
    if (!q21_gpu_resnet_forward(gpu, &state->mid[0], latent_h, latent_w, p,
                                normed, scratch, q, error, error_size))
        goto done;
    if (!q21_gpu_mid_attention(gpu, state, Q21_VAE_WIDE, latent_h, latent_w, q,
                               normed, error, error_size))
        goto done;
    if (!q21_gpu_resnet_forward(gpu, &state->mid[1], latent_h, latent_w, q,
                                normed, scratch, p, error, error_size))
        goto done;

    uint32_t h = latent_h, w = latent_w;
    for (int block = 0; block < Q21_VAE_BLOCKS; block++) {
        const uint32_t in_c = (uint32_t)vae->width[block];
        const uint32_t out_c = (uint32_t)vae->channels[block];
        const uint32_t upsample = (uint32_t)vae->upsamples[block];
        const size_t out_plane = (size_t)h * w * (upsample ? 4 : 1);
        if (upsample &&
            !q21_gpu_check(qzen_gpu_vae_dupup_chw_f32(
                               gpu, dup_buf, p, in_c, out_c,
                               (uint32_t)vae->temporal[block], h, w),
                           "up_blocks dupup", gpu, error, error_size))
            goto done;
        if (!q21_gpu_resnet_forward(gpu, &state->up[block][0], h, w, p, normed,
                                    scratch, q, error, error_size) ||
            !q21_gpu_resnet_forward(gpu, &state->up[block][1], h, w, q, normed,
                                    scratch, p, error, error_size) ||
            !q21_gpu_resnet_forward(gpu, &state->up[block][2], h, w, p, normed,
                                    scratch, q, error, error_size))
            goto done;
        if (upsample) {
            const uint32_t high = h * 2, wide = w * 2;
            /* q holds the resnet stack, p is spent (its content is already in
             * dup_buf), so p takes the nearest-2x staging. */
            if (!q21_gpu_check(qzen_gpu_upsample_nearest2x_f32(gpu, p, q, out_c,
                                                             h, w),
                               "up_blocks upsample", gpu, error, error_size) ||
                !q21_gpu_check(qzen_gpu_conv2d_f32(gpu, q, p,
                                                 state->up_conv[block].weight,
                                                 state->up_conv[block].bias,
                                                 out_c, out_c, high, wide, 3,
                                                 1),
                               "up_blocks up_conv", gpu, error, error_size) ||
                !q21_gpu_check(qzen_gpu_add_f32(gpu, q, q, dup_buf,
                                              (uint32_t)(out_c * out_plane)),
                               "up_blocks dupup add", gpu, error,
                               error_size))
                goto done;
            h = high;
            w = wide;
        }
        /* q is the block output; swap so p carries it into the next block. */
        qzen_gpu_tensor *spent = p;
        p = q;
        q = spent;
    }

    /* norm_out RMS(144) -> SiLU -> conv_out 4 channels. */
    {
        const size_t pixels = (size_t)h * w;
        if (!q21_gpu_check(qzen_gpu_vae_rms_chw_f32(gpu, normed, p,
                                                  state->norm_out_gamma, last_c,
                                                  h, w),
                           "decoder.norm_out", gpu, error, error_size) ||
            !q21_gpu_check(qzen_gpu_silu_f32(gpu, normed, normed,
                                           (uint32_t)((size_t)last_c * pixels)),
                           "decoder.norm_out silu", gpu, error, error_size) ||
            !q21_gpu_check(qzen_gpu_conv2d_f32(gpu, scratch, normed,
                                             state->conv_out.weight,
                                             state->conv_out.bias, 4, last_c,
                                             h, w, 3, 1),
                           "decoder.conv_out", gpu, error, error_size) ||
            !q21_gpu_check(qzen_gpu_submit(gpu), "qzen_gpu_submit", gpu, error,
                           error_size))
            goto done;
        float *tail = malloc(4 * pixels * sizeof(float));
        if (!tail) {
            snprintf(error, error_size, "out of memory for the decode tail");
            goto done;
        }
        const int read = qzen_gpu_tensor_read_f32(scratch, tail, 4 * pixels);
        if (!read) {
            free(tail);
            snprintf(error, error_size, "cannot read the decoded pixels: %s",
                     qzen_gpu_error(gpu));
            goto done;
        }
        for (size_t index = 0; index < pixels; index++) {
            for (int channel = 0; channel < 3; channel++) {
                float value = tail[(size_t)channel * pixels + index];
                value = value < -1.0f ? -1.0f : value > 1.0f ? 1.0f : value;
                value = (value + 1.0f) * 127.5f;
                int quantized = (int)(value + 0.5f);
                if (quantized < 0) quantized = 0;
                if (quantized > 255) quantized = 255;
                rgb[(size_t)channel * pixels + index] = (unsigned char)quantized;
            }
        }
        free(tail);
    }
    ok = 1;
done:
    for (int slot = 0; slot < 5; slot++)
        if (pool[slot]) qzen_gpu_tensor_free(pool[slot]);
    if (latent_dev) qzen_gpu_tensor_free(latent_dev);
    free(staged);
    return ok;
}


static int q21_vae_conv_load(qzen_qwen21_vae *vae, q21_conv *conv, const char *name, int out_c,
                             int in_c, int k, char *error, size_t error_size) {
    const uint64_t weight_shape[4] = {(uint64_t)out_c, (uint64_t)in_c, (uint64_t)k, (uint64_t)k};
    const uint64_t bias_shape[1] = {(uint64_t)out_c};
    char tensor[192];
    memset(conv, 0, sizeof(*conv));
    snprintf(tensor, sizeof(tensor), "%s.weight", name);
    conv->weight = q21_read_f32(&vae->header, tensor, 4, weight_shape, error, error_size);
    if (!conv->weight) return 0;
    snprintf(tensor, sizeof(tensor), "%s.bias", name);
    conv->bias = q21_read_f32(&vae->header, tensor, 1, bias_shape, error, error_size);
    if (!conv->bias) {
        free(conv->weight);
        conv->weight = NULL;
        return 0;
    }
    conv->out_c = out_c;
    conv->in_c = in_c;
    conv->k = k;
    return 1;
}

static float *q21_vae_gamma_load(qzen_qwen21_vae *vae, const char *name, int channels, char *error,
                                 size_t error_size) {
    char tensor[192];
    snprintf(tensor, sizeof(tensor), "%s.gamma", name);
    if (channels == Q21_VAE_WIDE) {
        /* the norms are stored as [C, 1, 1, 1] (or [C, 1, 1] for attention) */
        const uint64_t shape4[4] = {(uint64_t)channels, 1, 1, 1};
        const qzen_st_tensor *found = qzen_st_find(&vae->header, tensor);
        if (found && found->ndim == 4) {
            return q21_read_f32(&vae->header, tensor, 4, shape4, error, error_size);
        }
        const uint64_t shape3[3] = {(uint64_t)channels, 1, 1};
        const uint64_t shape1[1] = {(uint64_t)channels};
        if (found && found->ndim == 3) {
            return q21_read_f32(&vae->header, tensor, 3, shape3, error, error_size);
        }
        return q21_read_f32(&vae->header, tensor, 1, shape1, error, error_size);
    }
    const qzen_st_tensor *found = qzen_st_find(&vae->header, tensor);
    if (found && found->ndim == 4) {
        const uint64_t shape4[4] = {(uint64_t)channels, 1, 1, 1};
        return q21_read_f32(&vae->header, tensor, 4, shape4, error, error_size);
    }
    const uint64_t shape1[1] = {(uint64_t)channels};
    return q21_read_f32(&vae->header, tensor, 1, shape1, error, error_size);
}

static int q21_vae_resnet_load(qzen_qwen21_vae *vae, q21_resnet *res, const char *prefix, int in_c,
                               int out_c, char *error, size_t error_size) {
    char name[192];
    memset(res, 0, sizeof(*res));
    res->in_c = in_c;
    res->out_c = out_c;
    snprintf(name, sizeof(name), "%s.norm1", prefix);
    res->gamma1 = q21_vae_gamma_load(vae, name, in_c, error, error_size);
    snprintf(name, sizeof(name), "%s.norm2", prefix);
    res->gamma2 = q21_vae_gamma_load(vae, name, out_c, error, error_size);
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    if (!res->gamma1 || !res->gamma2 ||
        !q21_vae_conv_load(vae, &res->conv1, name, out_c, in_c, 3, error, error_size)) {
        return 0;
    }
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    if (!q21_vae_conv_load(vae, &res->conv2, name, out_c, out_c, 3, error, error_size)) return 0;
    if (in_c != out_c) {
        snprintf(name, sizeof(name), "%s.conv_shortcut", prefix);
        if (!q21_vae_conv_load(vae, &res->shortcut, name, out_c, in_c, 1, error, error_size)) {
            return 0;
        }
    }
    return 1;
}

static int q21_vae_stats_load(qzen_qwen21_vae *vae, const char *checkpoint, char *error,
                              size_t error_size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s", checkpoint);
    char *slash = strrchr(path, '/');
    if (!slash) slash = strrchr(path, '\\');
    if (!slash) {
        snprintf(error, error_size, "%s: no directory for vae_config.json", checkpoint);
        return 0;
    }
    snprintf(slash + 1, sizeof(path) - (size_t)(slash + 1 - path), "vae_config.json");
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(error, error_size, "%s: cannot read the VAE config", path);
        return 0;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = malloc((size_t)length + 1);
    if (!text) {
        fclose(file);
        snprintf(error, error_size, "out of memory for the VAE config");
        return 0;
    }
    size_t got = fread(text, 1, (size_t)length, file);
    fclose(file);
    text[got] = '\0';
    cJSON *root = cJSON_Parse(text);
    free(text);
    if (!root) {
        snprintf(error, error_size, "%s: invalid JSON", path);
        return 0;
    }
    int ok = 0;
    const cJSON *mean = cJSON_GetObjectItemCaseSensitive(root, "latents_mean");
    const cJSON *std = cJSON_GetObjectItemCaseSensitive(root, "latents_std");
    if (!cJSON_IsArray(mean) || !cJSON_IsArray(std) ||
        cJSON_GetArraySize(mean) != Q21_VAE_CHANNELS ||
        cJSON_GetArraySize(std) != Q21_VAE_CHANNELS) {
        snprintf(error, error_size, "%s: latents_mean/latents_std are not %d values", path,
                 Q21_VAE_CHANNELS);
        cJSON_Delete(root);
        return 0;
    }
    vae->latent_mean = malloc(Q21_VAE_CHANNELS * sizeof(float));
    vae->latent_std = malloc(Q21_VAE_CHANNELS * sizeof(float));
    if (vae->latent_mean && vae->latent_std) {
        for (int index = 0; index < Q21_VAE_CHANNELS; index++) {
            vae->latent_mean[index] = (float)cJSON_GetArrayItem(mean, index)->valuedouble;
            vae->latent_std[index] = (float)cJSON_GetArrayItem(std, index)->valuedouble;
        }
        ok = 1;
    } else {
        snprintf(error, error_size, "out of memory for the latent statistics");
    }
    cJSON_Delete(root);
    return ok;
}

static void q21_vae_conv_free(q21_conv *conv) {
    free(conv->weight);
    free(conv->bias);
    memset(conv, 0, sizeof(*conv));
}

static void q21_vae_resnet_free(q21_resnet *res) {
    q21_vae_conv_free(&res->conv1);
    q21_vae_conv_free(&res->conv2);
    q21_vae_conv_free(&res->shortcut);
    free(res->gamma1);
    free(res->gamma2);
    memset(res, 0, sizeof(*res));
}

void qzen_qwen21_vae_close(qzen_qwen21_vae *vae) {
    if (!vae) return;
    if (vae->gpu) q21_vae_gpu_free(vae->gpu);
    q21_encoder_free(&vae->enc);
    q21_vae_conv_free(&vae->post_quant);
    q21_vae_conv_free(&vae->conv_in);
    q21_vae_conv_free(&vae->conv_out);
    q21_vae_conv_free(&vae->attn_qkv);
    q21_vae_conv_free(&vae->attn_proj);
    q21_vae_resnet_free(&vae->mid[0]);
    q21_vae_resnet_free(&vae->mid[1]);
    for (int block = 0; block < Q21_VAE_BLOCKS; block++) {
        for (int r = 0; r < Q21_VAE_RESNETS; r++) q21_vae_resnet_free(&vae->up[block][r]);
        q21_vae_conv_free(&vae->up_conv[block]);
    }
    free(vae->attn_gamma);
    free(vae->norm_out_gamma);
    free(vae->latent_mean);
    free(vae->latent_std);
    qzen_st_free_header(&vae->header);
    free(vae);
}

const float *qzen_qwen21_vae_latent_mean(const qzen_qwen21_vae *vae) {
    return vae ? vae->latent_mean : NULL;
}

const float *qzen_qwen21_vae_latent_std(const qzen_qwen21_vae *vae) {
    return vae ? vae->latent_std : NULL;
}

qzen_qwen21_vae *qzen_qwen21_vae_open(const char *checkpoint, int thread_count, char *error,
                                  size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!checkpoint) {
        snprintf(error, error_size, "no Qwen-Image 2.1 VAE checkpoint path");
        return NULL;
    }
    qzen_qwen21_vae *vae = calloc(1, sizeof(*vae));
    if (!vae) {
        snprintf(error, error_size, "out of memory for the VAE decoder");
        return NULL;
    }
    if (!qzen_st_read_header(checkpoint, &vae->header, error, error_size)) {
        free(vae);
        return NULL;
    }
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    int threads = thread_count > 0 ? thread_count : (int)(cores > 0 ? cores : 1);
    if (threads > Q21_MAX_THREADS) threads = Q21_MAX_THREADS;
    if (threads < 1) threads = 1;
    vae->threads = threads;

    /* dims = [144 * 8] + [144 * u for u in [8, 8, 4, 2, 1]] = [1152, 1152, 1152, 576,
     * 288, 144], so the up blocks are 1152->1152, 1152->1152, 1152->576, 576->288 and
     * 288->144; the first four upsample and the first three carry a time_conv. */
    static const int widths[Q21_VAE_BLOCKS] = {1152, 1152, 1152, 576, 288};
    static const int channels[Q21_VAE_BLOCKS] = {1152, 1152, 576, 288, 144};
    static const int upsamples[Q21_VAE_BLOCKS] = {1, 1, 1, 1, 0};
    static const int temporal[Q21_VAE_BLOCKS] = {1, 1, 1, 0, 0};
    for (int block = 0; block < Q21_VAE_BLOCKS; block++) {
        vae->width[block] = widths[block];
        vae->channels[block] = channels[block];
        vae->upsamples[block] = upsamples[block];
        vae->temporal[block] = temporal[block];
    }

    char name[192];
    int ok = q21_vae_conv_load(vae, &vae->post_quant, "post_quant_conv", Q21_VAE_CHANNELS,
                               Q21_VAE_CHANNELS, 1, error, error_size) &&
             q21_vae_conv_load(vae, &vae->conv_in, "decoder.conv_in", Q21_VAE_WIDE,
                               Q21_VAE_CHANNELS, 3, error, error_size) &&
             q21_vae_conv_load(vae, &vae->conv_out, "decoder.conv_out", 4,
                               channels[Q21_VAE_BLOCKS - 1], 3, error, error_size);
    if (!ok) {
        qzen_qwen21_vae_close(vae);
        return NULL;
    }
    for (int index = 0; ok && index < 2; index++) {
        snprintf(name, sizeof(name), "decoder.mid_block.resnets.%d", index);
        ok = q21_vae_resnet_load(vae, &vae->mid[index], name, Q21_VAE_WIDE, Q21_VAE_WIDE, error,
                                 error_size);
    }
    if (ok) {
        vae->attn_gamma =
            q21_vae_gamma_load(vae, "decoder.mid_block.attentions.0.norm", Q21_VAE_WIDE, error,
                               error_size);
        ok = vae->attn_gamma != NULL &&
             q21_vae_conv_load(vae, &vae->attn_qkv, "decoder.mid_block.attentions.0.to_qkv",
                               3 * Q21_VAE_WIDE, Q21_VAE_WIDE, 1, error, error_size) &&
             q21_vae_conv_load(vae, &vae->attn_proj, "decoder.mid_block.attentions.0.proj",
                               Q21_VAE_WIDE, Q21_VAE_WIDE, 1, error, error_size);
    }
    for (int block = 0; ok && block < Q21_VAE_BLOCKS; block++) {
        for (int r = 0; ok && r < Q21_VAE_RESNETS; r++) {
            const int in_c = r == 0 ? widths[block] : channels[block];
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.resnets.%d", block, r);
            ok = q21_vae_resnet_load(vae, &vae->up[block][r], name, in_c, channels[block], error,
                                     error_size);
        }
        if (ok && upsamples[block]) {
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.upsampler.resample.1", block);
            ok = q21_vae_conv_load(vae, &vae->up_conv[block], name, channels[block],
                                   channels[block], 3, error, error_size);
        }
    }
    if (ok) {
        vae->norm_out_gamma =
            q21_vae_gamma_load(vae, "decoder.norm_out", channels[Q21_VAE_BLOCKS - 1], error,
                               error_size);
        ok = vae->norm_out_gamma != NULL;
    }
    if (ok) ok = q21_vae_stats_load(vae, checkpoint, error, error_size);
    if (!ok) {
        qzen_qwen21_vae_close(vae);
        return NULL;
    }
    return vae;
}
