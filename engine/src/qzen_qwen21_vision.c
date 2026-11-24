/* Qwen3-VL-8B visual tower for Qwen-Image 2.1 identity references.
 *
 * CPU path is the numerical reference; the device path keeps the expensive
 * linears and SDPA on the GPU and uses the host only for layernorm, RoPE,
 * activations and patch gathering. All INT8 ConvRot weights use group 64
 * (comfy_quant metadata), unlike the language tower's group 256.
 */

#include "qzen_qwen21_vision.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qzen_gpu.h"
#include "qzen_qwen21_ops.h"

#define Q21V_HEADS 16
#define Q21V_HEAD_DIM 72
#define Q21V_QKV (Q21V_HEADS * Q21V_HEAD_DIM * 3)
#define Q21V_INTER 4304
#define Q21V_DEPTH 27
#define Q21V_GRID 48
#define Q21V_POS_ROWS 2304
#define Q21V_MERGED_IN (QZEN_QWEN21_VIS_HIDDEN * 4)
#define Q21V_WEIGHT_CAP (Q21V_MERGED_IN * Q21V_MERGED_IN)
#define Q21V_ATTN_SCALE 0.11785113019775792f /* 1/sqrt(72) */
#define Q21V_EPS 1e-6f

static const int q21v_deepstack_layers[QZEN_QWEN21_VIS_DEEPSTACK] = {8, 16, 24};

typedef struct {
    qzen_gpu *gpu;
    qzen_gpu_tensor *hidden;
    qzen_gpu_tensor *normed;
    qzen_gpu_tensor *qkv;
    qzen_gpu_tensor *q;
    qzen_gpu_tensor *k;
    qzen_gpu_tensor *v;
    qzen_gpu_tensor *attn;
    qzen_gpu_tensor *proj;
    qzen_gpu_tensor *mlp1;
    qzen_gpu_tensor *mlp2;
    qzen_gpu_tensor *patch;
    qzen_gpu_tensor *pos;
    qzen_gpu_tensor *gather;
    qzen_gpu_tensor *mout;
    qzen_gpu_tensor *weight;
    qzen_gpu_tensor *bias;
    uint32_t seq;
} q21v_gpu;

struct qzen_qwen21_vision {
    qzen_st_header *header;
    int threads;
    q21v_gpu *gpu;
};

/* ------------------------------------------------------------------ helpers */

static uint16_t q21v_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x8000u + ((bits >> 16) & 1u)) >> 16);
}

static float q21v_f32(uint16_t half) {
    uint32_t bits = (uint32_t)half << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float *q21v_read(const qzen_st_header *header, const char *name,
                        int ndim, const uint64_t *shape, uint32_t group,
                        char *error, size_t error_size) {
    const qzen_st_tensor *tensor = qzen_st_find(header, name);
    if (!q21_expect(tensor, name, ndim, shape, error, error_size)) return NULL;
    const qzen_st_tensor *scale = NULL;
    if (tensor->dtype == QZEN_DTYPE_I8) {
        char scale_name[256];
        q21_scale_name(name, scale_name, sizeof(scale_name));
        scale = qzen_st_find(header, scale_name);
        if (!scale) {
            snprintf(error, error_size, "%s: INT8 weight without %s", name,
                     scale_name);
            return NULL;
        }
    }
    float *values = NULL;
    if (!qzen_st_read_f32_convrot(header, tensor, header, scale,
                                  group ? group : 256, &values, error,
                                  error_size))
        return NULL;
    return values;
}

static uint16_t *q21v_halves(const float *values, size_t count) {
    uint16_t *half = malloc(count * sizeof(*half));
    if (!half) return NULL;
    for (size_t index = 0; index < count; index++)
        half[index] = q21v_bf16(values[index]);
    return half;
}

static void q21v_layer_norm(const float *input, const float *weight,
                            const float *bias, float *output, uint32_t width) {
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) sum += input[i];
    float mean = (float)(sum / width);
    double square = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        double delta = (double)input[i] - mean;
        square += delta * delta;
    }
    float inverse = 1.0f / sqrtf((float)(square / width) + Q21V_EPS);
    for (uint32_t i = 0; i < width; i++)
        output[i] = ((input[i] - mean) * inverse) * weight[i] + bias[i];
}

static float q21v_gelu_tanh_scalar(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static void q21v_gelu_tanh(float *values, size_t count) {
    for (size_t i = 0; i < count; i++) values[i] = q21v_gelu_tanh_scalar(values[i]);
}

static float q21v_gelu_scalar(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

static void q21v_gelu(float *values, size_t count) {
    for (size_t i = 0; i < count; i++) values[i] = q21v_gelu_scalar(values[i]);
}

/* Patch order matches vision_utils.py: block_row, block_col, in_row, in_col. */
static void q21v_patch_rc(int index, int merge_w, int *row, int *col) {
    int in_col = index % QZEN_QWEN21_VIS_MERGE;
    int in_row = (index / QZEN_QWEN21_VIS_MERGE) % QZEN_QWEN21_VIS_MERGE;
    int block_col = (index / (QZEN_QWEN21_VIS_MERGE * QZEN_QWEN21_VIS_MERGE)) %
                    merge_w;
    int block_row = index /
                    (QZEN_QWEN21_VIS_MERGE * QZEN_QWEN21_VIS_MERGE * merge_w);
    *row = block_row * QZEN_QWEN21_VIS_MERGE + in_row;
    *col = block_col * QZEN_QWEN21_VIS_MERGE + in_col;
}

static void q21v_patchify(const uint8_t *rgb, uint32_t width, uint32_t height,
                          float *patches) {
    int gh = (int)(height / QZEN_QWEN21_VIS_PATCH);
    int gw = (int)(width / QZEN_QWEN21_VIS_PATCH);
    int merge_w = gw / QZEN_QWEN21_VIS_MERGE;
    for (int patch = 0; patch < gh * gw; patch++) {
        int row, col;
        q21v_patch_rc(patch, merge_w, &row, &col);
        float *out = patches + (size_t)patch * 1536;
        for (int channel = 0; channel < 3; channel++) {
            const uint8_t *plane = rgb + (size_t)channel * width * height;
            for (int temporal = 0; temporal < 2; temporal++) {
                float *dst = out + ((size_t)channel * 2 + temporal) * 256;
                for (int py = 0; py < 16; py++) {
                    const uint8_t *src = plane +
                        (size_t)(row * 16 + py) * width + (size_t)col * 16;
                    for (int px = 0; px < 16; px++)
                        dst[py * 16 + px] = (float)src[px] * (1.0f / 127.5f) - 1.0f;
                }
            }
        }
    }
}

static void q21v_interp_pos(const float *pos, int gh, int gw, float *out) {
    int merge_w = gw / QZEN_QWEN21_VIS_MERGE;
    for (int patch = 0; patch < gh * gw; patch++) {
        int row, col;
        q21v_patch_rc(patch, merge_w, &row, &col);
        float hs = (float)row * (Q21V_GRID - 1) / (float)(gh > 1 ? gh - 1 : 1);
        float ws = (float)col * (Q21V_GRID - 1) / (float)(gw > 1 ? gw - 1 : 1);
        int h0 = (int)floorf(hs), w0 = (int)floorf(ws);
        int h1 = h0 + 1, w1 = w0 + 1;
        if (h0 < 0) h0 = 0;
        if (w0 < 0) w0 = 0;
        if (h1 >= Q21V_GRID) h1 = Q21V_GRID - 1;
        if (w1 >= Q21V_GRID) w1 = Q21V_GRID - 1;
        float wh1 = fabsf(hs - floorf(hs));
        float wh0 = 1.0f - wh1;
        float ww1 = fabsf(ws - floorf(ws));
        float ww0 = 1.0f - ww1;
        const float *r0 = pos + (size_t)h0 * Q21V_GRID * QZEN_QWEN21_VIS_HIDDEN;
        const float *r1 = pos + (size_t)h1 * Q21V_GRID * QZEN_QWEN21_VIS_HIDDEN;
        float *dst = out + (size_t)patch * QZEN_QWEN21_VIS_HIDDEN;
        for (int d = 0; d < QZEN_QWEN21_VIS_HIDDEN; d++) {
            float top = r0[w0 * QZEN_QWEN21_VIS_HIDDEN + d] * ww0 +
                        r0[w1 * QZEN_QWEN21_VIS_HIDDEN + d] * ww1;
            float bottom = r1[w0 * QZEN_QWEN21_VIS_HIDDEN + d] * ww0 +
                           r1[w1 * QZEN_QWEN21_VIS_HIDDEN + d] * ww1;
            dst[d] = top * wh0 + bottom * wh1;
        }
    }
}

static void q21v_rope(int gh, int gw, float *cos_out, float *sin_out) {
    int merge_w = gw / QZEN_QWEN21_VIS_MERGE;
    double inv[18];
    for (int i = 0; i < 18; i++)
        inv[i] = 1.0 / pow(10000.0, (double)(2 * i) / 36.0);
    for (int patch = 0; patch < gh * gw; patch++) {
        int row, col;
        q21v_patch_rc(patch, merge_w, &row, &col);
        float *c = cos_out + (size_t)patch * Q21V_HEAD_DIM;
        float *s = sin_out + (size_t)patch * Q21V_HEAD_DIM;
        for (int i = 0; i < 18; i++) {
            double ah = (double)row * inv[i];
            double aw = (double)col * inv[i];
            float ch = (float)cos(ah), sh = (float)sin(ah);
            float cw = (float)cos(aw), sw = (float)sin(aw);
            c[i] = c[i + 18] = c[i + 36] = c[i + 54] = ch;
            c[i + 18] = c[i + 54] = cw;
            s[i] = s[i + 36] = sh;
            s[i + 18] = s[i + 54] = sw;
        }
    }
}

static void q21v_rotate_qk(float *q, float *k, uint32_t seq,
                           const float *cos, const float *sin) {
    const int half = Q21V_HEAD_DIM / 2;
    for (uint32_t token = 0; token < seq; token++) {
        const float *c = cos + (size_t)token * Q21V_HEAD_DIM;
        const float *s = sin + (size_t)token * Q21V_HEAD_DIM;
        for (int head = 0; head < Q21V_HEADS; head++) {
            float *qv = q + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM;
            float *kv = k + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM;
            for (int d = 0; d < half; d++) {
                float q0 = qv[d], q1 = qv[d + half];
                float k0 = kv[d], k1 = kv[d + half];
                qv[d] = q0 * c[d] - q1 * s[d];
                qv[d + half] = q1 * c[d + half] + q0 * s[d + half];
                kv[d] = k0 * c[d] - k1 * s[d];
                kv[d + half] = k1 * c[d + half] + k0 * s[d + half];
            }
        }
    }
}

/* -------------------------------------------------------------- CPU forward */

static void q21v_matmul(float *out, const float *in, const float *weight,
                        const float *bias, uint32_t rows, uint32_t in_dim,
                        uint32_t out_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *x = in + (size_t)row * in_dim;
        float *y = out + (size_t)row * out_dim;
        for (uint32_t col = 0; col < out_dim; col++) {
            const float *w = weight + (size_t)col * in_dim;
            double sum = bias ? bias[col] : 0.0;
            for (uint32_t k = 0; k < in_dim; k++) sum += (double)x[k] * w[k];
            y[col] = (float)sum;
        }
    }
}

static int q21v_linear_cpu(qzen_qwen21_vision *v, const char *name,
                           const float *in, float *out, uint32_t rows,
                           uint32_t in_dim, uint32_t out_dim, uint32_t group,
                           char *error, size_t error_size) {
    uint64_t shape[2] = {out_dim, in_dim};
    uint64_t bias_shape[1] = {out_dim};
    float *weight = q21v_read(v->header, name, 2, shape, group, error, error_size);
    if (!weight) return 0;
    char bias_name[256];
    snprintf(bias_name, sizeof(bias_name), "%s.bias",
             strstr(name, ".weight") ? name : name);
    if (strstr(name, ".weight"))
        memcpy(bias_name + (size_t)(strstr(name, ".weight") - name), ".bias", 6);
    float *bias = q21v_read(v->header, bias_name, 1, bias_shape, 0,
                            error, error_size);
    if (!bias) {
        free(weight);
        return 0;
    }
    q21v_matmul(out, in, weight, bias, rows, in_dim, out_dim);
    free(weight);
    free(bias);
    return 1;
}

static int q21v_attention_cpu(float *out, const float *q, const float *k,
                              const float *v, uint32_t seq) {
    float *scores = malloc((size_t)seq * sizeof(*scores));
    if (!scores) return 0;
    for (uint32_t head = 0; head < Q21V_HEADS; head++) {
        for (uint32_t row = 0; row < seq; row++) {
            const float *qr = q + ((size_t)row * Q21V_HEADS + head) * Q21V_HEAD_DIM;
            float maximum = -INFINITY;
            for (uint32_t key = 0; key < seq; key++) {
                const float *kr = k + ((size_t)key * Q21V_HEADS + head) * Q21V_HEAD_DIM;
                double dot = 0.0;
                for (int d = 0; d < Q21V_HEAD_DIM; d++) dot += (double)qr[d] * kr[d];
                scores[key] = (float)(dot * Q21V_ATTN_SCALE);
                if (scores[key] > maximum) maximum = scores[key];
            }
            double total = 0.0;
            for (uint32_t key = 0; key < seq; key++) {
                scores[key] = expf(scores[key] - maximum);
                total += scores[key];
            }
            float *orr = out + ((size_t)row * Q21V_HEADS + head) * Q21V_HEAD_DIM;
            memset(orr, 0, Q21V_HEAD_DIM * sizeof(*orr));
            for (uint32_t key = 0; key < seq; key++) {
                const float *vr = v + ((size_t)key * Q21V_HEADS + head) * Q21V_HEAD_DIM;
                float p = (float)(scores[key] / total);
                for (int d = 0; d < Q21V_HEAD_DIM; d++) orr[d] += p * vr[d];
            }
        }
    }
    free(scores);
    return 1;
}

static int q21v_merger_cpu(qzen_qwen21_vision *vision, const char *prefix,
                           const float *hidden, uint32_t seq, int postshuffle,
                           float *out, char *error, size_t error_size) {
    uint32_t merged = seq / 4;
    float *gathered = malloc((size_t)merged * Q21V_MERGED_IN * sizeof(float));
    float *normed = malloc((size_t)merged * Q21V_MERGED_IN * sizeof(float));
    float *first = malloc((size_t)merged * Q21V_MERGED_IN * sizeof(float));
    if (!gathered || !normed || !first) {
        free(gathered); free(normed); free(first);
        snprintf(error, error_size, "out of memory for the vision merger");
        return 0;
    }
    float *source = (float *)hidden;
    if (!postshuffle) {
        char norm_name[256];
        uint64_t hidden_dim[1] = {QZEN_QWEN21_VIS_HIDDEN};
        snprintf(norm_name, sizeof(norm_name), "%s.norm.weight", prefix);
        float *nw = q21v_read(vision->header, norm_name, 1, hidden_dim, 0,
                              error, error_size);
        snprintf(norm_name, sizeof(norm_name), "%s.norm.bias", prefix);
        float *nb = q21v_read(vision->header, norm_name, 1, hidden_dim, 0,
                              error, error_size);
        if (!nw || !nb) {
            free(nw); free(nb);
            goto fail;
        }
        for (uint32_t i = 0; i < seq; i++)
            q21v_layer_norm(source + (size_t)i * QZEN_QWEN21_VIS_HIDDEN,
                            nw, nb, source + (size_t)i * QZEN_QWEN21_VIS_HIDDEN,
                            QZEN_QWEN21_VIS_HIDDEN);
        free(nw); free(nb);
    }
    for (uint32_t i = 0; i < merged; i++) {
        memcpy(gathered + (size_t)i * Q21V_MERGED_IN,
               source + (size_t)i * 4 * QZEN_QWEN21_VIS_HIDDEN,
               Q21V_MERGED_IN * sizeof(float));
    }
    uint64_t dim[1] = {Q21V_MERGED_IN};
    uint64_t out_dim[1] = {QZEN_QWEN21_VIS_OUT};
    char name[256];
    if (postshuffle) {
        snprintf(name, sizeof(name), "%s.norm.weight", prefix);
        float *w = q21v_read(vision->header, name, 1, dim, 0, error, error_size);
        snprintf(name, sizeof(name), "%s.norm.bias", prefix);
        float *b = q21v_read(vision->header, name, 1, dim, 0, error, error_size);
        if (!w || !b) { free(w); free(b); goto fail; }
        for (uint32_t i = 0; i < merged; i++)
            q21v_layer_norm(gathered + (size_t)i * Q21V_MERGED_IN, w, b,
                            normed + (size_t)i * Q21V_MERGED_IN, Q21V_MERGED_IN);
        free(w); free(b);
    } else {
        memcpy(normed, gathered, (size_t)merged * Q21V_MERGED_IN * sizeof(float));
    }
    snprintf(name, sizeof(name), "%s.linear_fc1.weight", prefix);
    if (!q21v_linear_cpu(vision, name, normed, first, merged, Q21V_MERGED_IN,
                         Q21V_MERGED_IN, 0, error, error_size)) goto fail;
    q21v_gelu(first, (size_t)merged * Q21V_MERGED_IN);
    snprintf(name, sizeof(name), "%s.linear_fc2.weight", prefix);
    if (!q21v_linear_cpu(vision, name, first, out, merged, Q21V_MERGED_IN,
                         QZEN_QWEN21_VIS_OUT, 0, error, error_size)) goto fail;
    free(gathered); free(normed); free(first);
    return 1;
fail:
    free(gathered); free(normed); free(first);
    return 0;
}

static int q21v_forward_cpu(qzen_qwen21_vision *vision,
                            const uint8_t *const *rgb, const uint32_t *widths,
                            const uint32_t *heights, int count, float **merged,
                            float *deep[QZEN_QWEN21_VIS_DEEPSTACK],
                            int *pad_count, char *error, size_t error_size) {
    size_t total = 0;
    for (int i = 0; i < count; i++) {
        if (widths[i] % QZEN_QWEN21_VIS_ALIGN ||
            heights[i] % QZEN_QWEN21_VIS_ALIGN) {
            snprintf(error, error_size, "visual reference %d is not 32-aligned", i);
            return 0;
        }
        total += (size_t)(widths[i] / 16) * (heights[i] / 16);
    }
    uint32_t seq = (uint32_t)total;
    uint32_t merged_tokens = seq / 4;
    float *hidden = calloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN, sizeof(float));
    float *normed = malloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
    float *qkv = malloc((size_t)seq * Q21V_QKV * sizeof(float));
    float *q = malloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
    float *k = malloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
    float *v = malloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
    float *attn = malloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
    float *proj = malloc((size_t)seq * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
    float *mlp = malloc((size_t)seq * Q21V_INTER * sizeof(float));
    float *cos = malloc((size_t)seq * Q21V_HEAD_DIM * sizeof(float));
    float *sin = malloc((size_t)seq * Q21V_HEAD_DIM * sizeof(float));
    int ok = 0;
    if (!hidden || !normed || !qkv || !q || !k || !v || !attn || !proj || !mlp ||
        !cos || !sin) {
        snprintf(error, error_size, "out of memory for the visual tower");
        goto done;
    }
    size_t offset = 0;
    uint64_t patch_shape[5] = {QZEN_QWEN21_VIS_HIDDEN, 3, 2, 16, 16};
    uint64_t pos_shape[2] = {Q21V_POS_ROWS, QZEN_QWEN21_VIS_HIDDEN};
    uint64_t vec[1] = {QZEN_QWEN21_VIS_HIDDEN};
    float *patch_weight = NULL, *patch_bias = NULL, *pos = NULL;
    patch_weight = q21v_read(vision->header, "visual.patch_embed.proj.weight",
                             5, patch_shape, 0, error, error_size);
    patch_bias = q21v_read(vision->header, "visual.patch_embed.proj.bias",
                           1, vec, 0, error, error_size);
    pos = q21v_read(vision->header, "visual.pos_embed.weight", 2, pos_shape,
                    0, error, error_size);
    if (!patch_weight || !patch_bias || !pos) goto done;
    for (int image = 0; image < count; image++) {
        size_t patches = (size_t)(widths[image] / 16) * (heights[image] / 16);
        float *raw = malloc(patches * 1536 * sizeof(float));
        float *projected = malloc(patches * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
        float *positions = malloc(patches * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
        if (!raw || !projected || !positions) {
            free(raw); free(projected); free(positions);
            snprintf(error, error_size, "out of memory for visual patches");
            goto done;
        }
        q21v_patchify(rgb[image], widths[image], heights[image], raw);
        q21v_matmul(projected, raw, patch_weight, patch_bias, (uint32_t)patches,
                    1536, QZEN_QWEN21_VIS_HIDDEN);
        q21v_interp_pos(pos, (int)(heights[image] / 16),
                        (int)(widths[image] / 16), positions);
        for (size_t i = 0; i < patches * QZEN_QWEN21_VIS_HIDDEN; i++)
            hidden[offset * QZEN_QWEN21_VIS_HIDDEN + i] =
                projected[i] + positions[i];
        q21v_rope((int)(heights[image] / 16), (int)(widths[image] / 16),
                  cos + offset * Q21V_HEAD_DIM, sin + offset * Q21V_HEAD_DIM);
        offset += patches;
        free(raw); free(projected); free(positions);
    }
    free(patch_weight); free(patch_bias); free(pos);
    patch_weight = patch_bias = pos = NULL;

    float *local_deep[QZEN_QWEN21_VIS_DEEPSTACK] = {NULL, NULL, NULL};
    char name[256];
    for (int layer = 0; layer < Q21V_DEPTH; layer++) {
        for (int i = 0; i < QZEN_QWEN21_VIS_HIDDEN; i++) normed[i] = 0;
        for (uint32_t token = 0; token < seq; token++) {
            float *src = hidden + (size_t)token * QZEN_QWEN21_VIS_HIDDEN;
            float *dst = normed + (size_t)token * QZEN_QWEN21_VIS_HIDDEN;
            uint64_t dim[1] = {QZEN_QWEN21_VIS_HIDDEN};
            char wname[256], bname[256];
            /* Norms are small; loading them per layer keeps peak RAM flat. */
            snprintf(wname, sizeof(wname), "visual.blocks.%d.norm1.weight", layer);
            snprintf(bname, sizeof(bname), "visual.blocks.%d.norm1.bias", layer);
            float *w = q21v_read(vision->header, wname, 1, dim, 0, error, error_size);
            float *b = q21v_read(vision->header, bname, 1, dim, 0, error, error_size);
            if (!w || !b) { free(w); free(b); goto done; }
            q21v_layer_norm(src, w, b, dst, QZEN_QWEN21_VIS_HIDDEN);
            free(w); free(b);
        }
        snprintf(name, sizeof(name), "visual.blocks.%d.attn.qkv", layer);
        if (!q21v_linear_cpu(vision, name, normed, qkv, seq,
                             QZEN_QWEN21_VIS_HIDDEN, Q21V_QKV, 64,
                             error, error_size)) goto done;
        for (uint32_t token = 0; token < seq; token++) {
            const float *row = qkv + (size_t)token * Q21V_QKV;
            for (int head = 0; head < Q21V_HEADS; head++) {
                memcpy(q + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM,
                       row + head * Q21V_HEAD_DIM, Q21V_HEAD_DIM * sizeof(float));
                memcpy(k + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM,
                       row + 1152 + head * Q21V_HEAD_DIM, Q21V_HEAD_DIM * sizeof(float));
                memcpy(v + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM,
                       row + 2304 + head * Q21V_HEAD_DIM, Q21V_HEAD_DIM * sizeof(float));
            }
        }
        q21v_rotate_qk(q, k, seq, cos, sin);
        if (!q21v_attention_cpu(attn, q, k, v, seq)) {
            snprintf(error, error_size, "out of memory for visual attention");
            goto done;
        }
        for (uint32_t token = 0; token < seq; token++) {
            float *dst = attn + (size_t)token * QZEN_QWEN21_VIS_HIDDEN;
            float reordered[QZEN_QWEN21_VIS_HIDDEN];
            for (int head = 0; head < Q21V_HEADS; head++)
                memcpy(reordered + head * Q21V_HEAD_DIM,
                       dst + head * Q21V_HEAD_DIM,
                       Q21V_HEAD_DIM * sizeof(float));
            memcpy(dst, reordered, sizeof(reordered));
        }
        snprintf(name, sizeof(name), "visual.blocks.%d.attn.proj", layer);
        if (!q21v_linear_cpu(vision, name, attn, proj, seq,
                             QZEN_QWEN21_VIS_HIDDEN, QZEN_QWEN21_VIS_HIDDEN, 64,
                             error, error_size)) goto done;
        for (size_t i = 0; i < (size_t)seq * QZEN_QWEN21_VIS_HIDDEN; i++)
            hidden[i] += proj[i];
        for (int slot = 0; slot < QZEN_QWEN21_VIS_DEEPSTACK; slot++) {
            if (q21v_deepstack_layers[slot] != layer) continue;
            local_deep[slot] = malloc((size_t)merged_tokens * QZEN_QWEN21_VIS_OUT *
                                   sizeof(float));
            char prefix[128];
            snprintf(prefix, sizeof(prefix), "visual.deepstack_merger_list.%d", slot);
            if (!local_deep[slot] ||
                !q21v_merger_cpu(vision, prefix, hidden, seq, 1, local_deep[slot],
                                 error, error_size)) goto done;
        }
        for (uint32_t token = 0; token < seq; token++) {
            uint64_t dim[1] = {QZEN_QWEN21_VIS_HIDDEN};
            char wname[256], bname[256];
            snprintf(wname, sizeof(wname), "visual.blocks.%d.norm2.weight", layer);
            snprintf(bname, sizeof(bname), "visual.blocks.%d.norm2.bias", layer);
            float *w = q21v_read(vision->header, wname, 1, dim, 0, error, error_size);
            float *b = q21v_read(vision->header, bname, 1, dim, 0, error, error_size);
            if (!w || !b) { free(w); free(b); goto done; }
            q21v_layer_norm(hidden + (size_t)token * QZEN_QWEN21_VIS_HIDDEN,
                            w, b, normed + (size_t)token * QZEN_QWEN21_VIS_HIDDEN,
                            QZEN_QWEN21_VIS_HIDDEN);
            free(w); free(b);
        }
        snprintf(name, sizeof(name), "visual.blocks.%d.mlp.linear_fc1", layer);
        if (!q21v_linear_cpu(vision, name, normed, mlp, seq,
                             QZEN_QWEN21_VIS_HIDDEN, Q21V_INTER, 64,
                             error, error_size)) goto done;
        q21v_gelu_tanh(mlp, (size_t)seq * Q21V_INTER);
        snprintf(name, sizeof(name), "visual.blocks.%d.mlp.linear_fc2", layer);
        if (!q21v_linear_cpu(vision, name, mlp, proj, seq,
                             Q21V_INTER, QZEN_QWEN21_VIS_HIDDEN, 16,
                             error, error_size)) goto done;
        for (size_t i = 0; i < (size_t)seq * QZEN_QWEN21_VIS_HIDDEN; i++)
            hidden[i] += proj[i];
    }
    *merged = malloc((size_t)merged_tokens * QZEN_QWEN21_VIS_OUT * sizeof(float));
    if (!*merged ||
        !q21v_merger_cpu(vision, "visual.merger", hidden, seq, 0, *merged,
                         error, error_size)) goto done;
    for (int i = 0; i < QZEN_QWEN21_VIS_DEEPSTACK; i++) {
        if (!local_deep[i]) {
            snprintf(error, error_size, "visual tower missing deepstack %d", i);
            goto done;
        }
        deep[i] = local_deep[i];
    }
    *pad_count = (int)merged_tokens;
    ok = 1;
done:
    if (!ok) {
        free(*merged); *merged = NULL;
        for (int i = 0; i < QZEN_QWEN21_VIS_DEEPSTACK; i++) { free(local_deep[i]); local_deep[i] = NULL; }
    }
    free(hidden); free(normed); free(qkv); free(q); free(k); free(v);
    free(attn); free(proj); free(mlp); free(cos); free(sin);
    free(patch_weight); free(patch_bias); free(pos);
    return ok;
}

/* -------------------------------------------------------------- GPU forward */

static void q21v_gpu_drop(qzen_gpu_tensor **tensor) {
    if (!tensor || !*tensor) return;
    qzen_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void q21v_gpu_free(q21v_gpu *ctx) {
    if (!ctx) return;
    q21v_gpu_drop(&ctx->hidden);
    q21v_gpu_drop(&ctx->normed);
    q21v_gpu_drop(&ctx->qkv);
    q21v_gpu_drop(&ctx->q);
    q21v_gpu_drop(&ctx->k);
    q21v_gpu_drop(&ctx->v);
    q21v_gpu_drop(&ctx->attn);
    q21v_gpu_drop(&ctx->proj);
    q21v_gpu_drop(&ctx->mlp1);
    q21v_gpu_drop(&ctx->mlp2);
    q21v_gpu_drop(&ctx->patch);
    q21v_gpu_drop(&ctx->pos);
    q21v_gpu_drop(&ctx->gather);
    q21v_gpu_drop(&ctx->mout);
    q21v_gpu_drop(&ctx->weight);
    q21v_gpu_drop(&ctx->bias);
    if (ctx->gpu) qzen_gpu_free(ctx->gpu);
    free(ctx);
}

#define Q21V_GPU_ALLOC(slot, elements)                                          \
    do {                                                                        \
        if (qzen_gpu_tensor_elements(ctx->slot) < (size_t)(elements)) {          \
            q21v_gpu_drop(&ctx->slot);                                          \
            ctx->slot = qzen_gpu_tensor_new_bf16(ctx->gpu, (elements));          \
        }                                                                       \
        if (!ctx->slot) {                                                       \
            snprintf(error, error_size, "out of device memory for " #slot);      \
            return 0;                                                           \
        }                                                                       \
    } while (0)

static int q21v_gpu_alloc(q21v_gpu *ctx, uint32_t seq, char *error,
                          size_t error_size) {
    size_t hidden = (size_t)seq * QZEN_QWEN21_VIS_HIDDEN;
    Q21V_GPU_ALLOC(hidden, hidden);
    Q21V_GPU_ALLOC(normed, hidden);
    Q21V_GPU_ALLOC(qkv, (size_t)seq * Q21V_QKV);
    Q21V_GPU_ALLOC(q, hidden);
    Q21V_GPU_ALLOC(k, hidden);
    Q21V_GPU_ALLOC(v, hidden);
    Q21V_GPU_ALLOC(attn, hidden);
    Q21V_GPU_ALLOC(proj, hidden);
    Q21V_GPU_ALLOC(mlp1, (size_t)seq * Q21V_INTER);
    Q21V_GPU_ALLOC(mlp2, hidden);
    Q21V_GPU_ALLOC(patch, (size_t)seq * 1536);
    Q21V_GPU_ALLOC(pos, hidden);
    Q21V_GPU_ALLOC(gather, (size_t)(seq / 4) * Q21V_MERGED_IN);
    Q21V_GPU_ALLOC(mout, (size_t)(seq / 4) * QZEN_QWEN21_VIS_OUT);
    Q21V_GPU_ALLOC(weight, Q21V_WEIGHT_CAP);
    Q21V_GPU_ALLOC(bias, Q21V_MERGED_IN);
    ctx->seq = seq;
    return 1;
}
#undef Q21V_GPU_ALLOC

static int q21v_gpu_linear(q21v_gpu *ctx, qzen_gpu_tensor *out,
                           qzen_gpu_tensor *in, const float *weight,
                           const float *bias, uint32_t rows, uint32_t in_dim,
                           uint32_t out_dim, char *error, size_t error_size) {
    uint16_t *w = q21v_halves(weight, (size_t)out_dim * in_dim);
    uint16_t *b = q21v_halves(bias, out_dim);
    int ok = 0;
    if (!w || !b) goto done;
    if (!qzen_gpu_tensor_write_bf16(ctx->weight, w, (size_t)out_dim * in_dim) ||
        !qzen_gpu_tensor_write_bf16(ctx->bias, b, out_dim) ||
        !qzen_gpu_linear_bf16(ctx->gpu, out, in, ctx->weight, ctx->bias,
                              rows, in_dim, out_dim)) {
        snprintf(error, error_size, "visual linear %ux%u failed: %s", out_dim,
                 in_dim, qzen_gpu_error(ctx->gpu));
        goto done;
    }
    ok = 1;
done:
    free(w); free(b);
    return ok;
}

static int q21v_gpu_norm(qzen_qwen21_vision *vision, q21v_gpu *ctx,
                         const char *prefix, int second, uint32_t seq,
                         char *error, size_t error_size) {
    size_t count = (size_t)seq * QZEN_QWEN21_VIS_HIDDEN;
    uint16_t *raw = malloc(count * sizeof(*raw));
    float *values = malloc(count * sizeof(*values));
    uint64_t dim[1] = {QZEN_QWEN21_VIS_HIDDEN};
    char name[256];
    float *weight = NULL, *bias = NULL;
    int ok = 0;
    if (!raw || !values) {
        snprintf(error, error_size, "out of memory for visual layernorm");
        goto done;
    }
    if (!qzen_gpu_tensor_read_bf16(ctx->hidden, raw, count)) {
        snprintf(error, error_size, "visual hidden read failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    for (size_t i = 0; i < count; i++) values[i] = q21v_f32(raw[i]);
    snprintf(name, sizeof(name), "%s.norm%s.weight", prefix, second ? "2" : "1");
    weight = q21v_read(vision->header, name, 1, dim, 0, error, error_size);
    if (!weight) goto done;
    snprintf(name, sizeof(name), "%s.norm%s.bias", prefix, second ? "2" : "1");
    bias = q21v_read(vision->header, name, 1, dim, 0, error, error_size);
    if (!bias) goto done;
    for (uint32_t token = 0; token < seq; token++)
        q21v_layer_norm(values + (size_t)token * QZEN_QWEN21_VIS_HIDDEN,
                        weight, bias,
                        values + (size_t)token * QZEN_QWEN21_VIS_HIDDEN,
                        QZEN_QWEN21_VIS_HIDDEN);
    for (size_t i = 0; i < count; i++) raw[i] = q21v_bf16(values[i]);
    if (!qzen_gpu_tensor_write_bf16(ctx->normed, raw, count)) {
        snprintf(error, error_size, "visual norm upload failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    ok = 1;
done:
    free(raw); free(values); free(weight); free(bias);
    return ok;
}

static int q21v_gpu_block(qzen_qwen21_vision *vision, int layer,
                          const float *cos, const float *sin, uint32_t seq,
                          char *error, size_t error_size) {
    q21v_gpu *ctx = vision->gpu;
    char name[256];
    uint64_t shape[2];
    uint64_t bias_shape[1];
    const uint64_t qkv_shape[2] = {Q21V_QKV, QZEN_QWEN21_VIS_HIDDEN};
    const uint64_t proj_shape[2] = {QZEN_QWEN21_VIS_HIDDEN, QZEN_QWEN21_VIS_HIDDEN};
    const uint64_t fc1_shape[2] = {Q21V_INTER, QZEN_QWEN21_VIS_HIDDEN};
    const uint64_t fc2_shape[2] = {QZEN_QWEN21_VIS_HIDDEN, Q21V_INTER};
    float *weight = NULL, *bias = NULL;
    uint16_t *qkv_raw = NULL, *q_raw = NULL, *k_raw = NULL;
    int ok = 0;

    snprintf(name, sizeof(name), "visual.blocks.%d", layer);
    if (!q21v_gpu_norm(vision, ctx, name, 0, seq, error, error_size)) goto done;

    snprintf(name, sizeof(name), "visual.blocks.%d.attn.qkv.weight", layer);
    shape[0] = qkv_shape[0]; shape[1] = qkv_shape[1];
    bias_shape[0] = qkv_shape[0];
    weight = q21v_read(vision->header, name, 2, shape, 64, error, error_size);
    if (!weight) goto done;
    snprintf(name, sizeof(name), "visual.blocks.%d.attn.qkv.bias", layer);
    bias = q21v_read(vision->header, name, 1, bias_shape, 0, error, error_size);
    if (!bias) goto done;
    if (!q21v_gpu_linear(ctx, ctx->qkv, ctx->normed, weight, bias, seq,
                         QZEN_QWEN21_VIS_HIDDEN, Q21V_QKV, error,
                         error_size)) goto done;
    free(weight); weight = NULL;
    free(bias); bias = NULL;

    size_t qkv_count = (size_t)seq * Q21V_QKV;
    size_t hidden_count = (size_t)seq * QZEN_QWEN21_VIS_HIDDEN;
    qkv_raw = malloc(qkv_count * sizeof(*qkv_raw));
    q_raw = malloc(hidden_count * sizeof(*q_raw));
    k_raw = malloc(hidden_count * sizeof(*k_raw));
    uint16_t *v_raw = malloc(hidden_count * sizeof(*v_raw));
    if (!qkv_raw || !q_raw || !k_raw || !v_raw) {
        free(v_raw);
        snprintf(error, error_size, "out of memory for visual qkv");
        goto done;
    }
    if (!qzen_gpu_tensor_read_bf16(ctx->qkv, qkv_raw, qkv_count)) {
        free(v_raw);
        snprintf(error, error_size, "visual qkv read failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    for (uint32_t token = 0; token < seq; token++) {
        const uint16_t *row = qkv_raw + (size_t)token * Q21V_QKV;
        for (int head = 0; head < Q21V_HEADS; head++) {
            memcpy(q_raw + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM,
                   row + head * Q21V_HEAD_DIM, Q21V_HEAD_DIM * sizeof(uint16_t));
            memcpy(k_raw + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM,
                   row + 1152 + head * Q21V_HEAD_DIM, Q21V_HEAD_DIM * sizeof(uint16_t));
            memcpy(v_raw + ((size_t)token * Q21V_HEADS + head) * Q21V_HEAD_DIM,
                   row + 2304 + head * Q21V_HEAD_DIM, Q21V_HEAD_DIM * sizeof(uint16_t));
        }
    }
    float *qf = malloc(hidden_count * sizeof(*qf));
    float *kf = malloc(hidden_count * sizeof(*kf));
    if (!qf || !kf) {
        free(qf); free(kf); free(v_raw);
        snprintf(error, error_size, "out of memory for visual rope");
        goto done;
    }
    for (size_t i = 0; i < hidden_count; i++) {
        qf[i] = q21v_f32(q_raw[i]);
        kf[i] = q21v_f32(k_raw[i]);
    }
    q21v_rotate_qk(qf, kf, seq, cos, sin);
    for (size_t i = 0; i < hidden_count; i++) {
        q_raw[i] = q21v_bf16(qf[i]);
        k_raw[i] = q21v_bf16(kf[i]);
    }
    free(qf); free(kf);
    if (!qzen_gpu_tensor_write_bf16(ctx->q, q_raw, hidden_count) ||
        !qzen_gpu_tensor_write_bf16(ctx->k, k_raw, hidden_count) ||
        !qzen_gpu_tensor_write_bf16(ctx->v, v_raw, hidden_count)) {
        free(v_raw);
        snprintf(error, error_size, "visual qkv upload failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    free(v_raw); v_raw = NULL;
    if (!qzen_gpu_sdpa_bf16(ctx->gpu, ctx->attn, ctx->q, ctx->k, ctx->v,
                            seq, Q21V_HEADS, Q21V_HEAD_DIM, Q21V_ATTN_SCALE)) {
        snprintf(error, error_size, "visual attention failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }

    snprintf(name, sizeof(name), "visual.blocks.%d.attn.proj.weight", layer);
    shape[0] = proj_shape[0]; shape[1] = proj_shape[1];
    bias_shape[0] = proj_shape[0];
    weight = q21v_read(vision->header, name, 2, shape, 64, error, error_size);
    if (!weight) goto done;
    snprintf(name, sizeof(name), "visual.blocks.%d.attn.proj.bias", layer);
    bias = q21v_read(vision->header, name, 1, bias_shape, 0, error, error_size);
    if (!bias) goto done;
    if (!q21v_gpu_linear(ctx, ctx->proj, ctx->attn, weight, bias, seq,
                         QZEN_QWEN21_VIS_HIDDEN, QZEN_QWEN21_VIS_HIDDEN,
                         error, error_size)) goto done;
    free(weight); weight = NULL;
    free(bias); bias = NULL;
    if (!qzen_gpu_add_bf16(ctx->gpu, ctx->hidden, ctx->hidden, ctx->proj,
                           (uint32_t)hidden_count)) {
        snprintf(error, error_size, "visual attention residual failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }

    snprintf(name, sizeof(name), "visual.blocks.%d", layer);
    if (!q21v_gpu_norm(vision, ctx, name, 1, seq, error, error_size)) goto done;
    snprintf(name, sizeof(name), "visual.blocks.%d.mlp.linear_fc1.weight", layer);
    shape[0] = fc1_shape[0]; shape[1] = fc1_shape[1];
    bias_shape[0] = fc1_shape[0];
    weight = q21v_read(vision->header, name, 2, shape, 64, error, error_size);
    if (!weight) goto done;
    snprintf(name, sizeof(name), "visual.blocks.%d.mlp.linear_fc1.bias", layer);
    bias = q21v_read(vision->header, name, 1, bias_shape, 0, error, error_size);
    if (!bias) goto done;
    if (!q21v_gpu_linear(ctx, ctx->mlp1, ctx->normed, weight, bias, seq,
                         QZEN_QWEN21_VIS_HIDDEN, Q21V_INTER, error,
                         error_size)) goto done;
    free(weight); weight = NULL;
    free(bias); bias = NULL;

    size_t mlp_count = (size_t)seq * Q21V_INTER;
    uint16_t *act = malloc(mlp_count * sizeof(*act));
    if (!act) {
        snprintf(error, error_size, "out of memory for visual MLP");
        goto done;
    }
    if (!qzen_gpu_tensor_read_bf16(ctx->mlp1, act, mlp_count)) {
        free(act);
        snprintf(error, error_size, "visual MLP read failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    for (size_t i = 0; i < mlp_count; i++) act[i] = q21v_bf16(q21v_gelu_tanh_scalar(q21v_f32(act[i])));
    if (!qzen_gpu_tensor_write_bf16(ctx->mlp1, act, mlp_count)) {
        free(act);
        snprintf(error, error_size, "visual MLP upload failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    free(act);

    snprintf(name, sizeof(name), "visual.blocks.%d.mlp.linear_fc2.weight", layer);
    shape[0] = fc2_shape[0]; shape[1] = fc2_shape[1];
    bias_shape[0] = fc2_shape[0];
    weight = q21v_read(vision->header, name, 2, shape, 16, error, error_size);
    if (!weight) goto done;
    snprintf(name, sizeof(name), "visual.blocks.%d.mlp.linear_fc2.bias", layer);
    bias = q21v_read(vision->header, name, 1, bias_shape, 0, error, error_size);
    if (!bias) goto done;
    if (!q21v_gpu_linear(ctx, ctx->mlp2, ctx->mlp1, weight, bias, seq,
                         Q21V_INTER, QZEN_QWEN21_VIS_HIDDEN, error,
                         error_size)) goto done;
    free(weight); weight = NULL;
    free(bias); bias = NULL;
    if (!qzen_gpu_add_bf16(ctx->gpu, ctx->hidden, ctx->hidden, ctx->mlp2,
                           (uint32_t)hidden_count)) {
        snprintf(error, error_size, "visual MLP residual failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    ok = 1;
done:
    free(weight); free(bias); free(qkv_raw); free(q_raw); free(k_raw);
    return ok;
}

static int q21v_gpu_merger(qzen_qwen21_vision *vision, const char *prefix,
                           int postshuffle, uint32_t seq, float *out,
                           char *error, size_t error_size) {
    q21v_gpu *ctx = vision->gpu;
    uint32_t merged = seq / 4;
    size_t hidden_count = (size_t)seq * QZEN_QWEN21_VIS_HIDDEN;
    size_t gather_count = (size_t)merged * Q21V_MERGED_IN;
    size_t out_count = (size_t)merged * QZEN_QWEN21_VIS_OUT;
    uint16_t *raw = malloc(hidden_count * sizeof(*raw));
    float *values = malloc(hidden_count * sizeof(*values));
    float *gathered = malloc(gather_count * sizeof(*gathered));
    uint16_t *half = malloc(gather_count * sizeof(*half));
    qzen_gpu_tensor *mid = NULL;
    uint64_t dim[1] = {Q21V_MERGED_IN};
    uint64_t hidden_dim[1] = {QZEN_QWEN21_VIS_HIDDEN};
    uint64_t out_dim[1] = {QZEN_QWEN21_VIS_OUT};
    float *weight = NULL, *bias = NULL;
    char name[256];
    int ok = 0;
    if (!raw || !values || !gathered || !half) {
        snprintf(error, error_size, "out of memory for the vision merger");
        goto done;
    }
    if (!qzen_gpu_tensor_read_bf16(ctx->hidden, raw, hidden_count)) {
        snprintf(error, error_size, "merger hidden read failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    for (size_t i = 0; i < hidden_count; i++) values[i] = q21v_f32(raw[i]);
    if (!postshuffle) {
        snprintf(name, sizeof(name), "%s.norm.weight", prefix);
        weight = q21v_read(vision->header, name, 1, hidden_dim, 0, error, error_size);
        snprintf(name, sizeof(name), "%s.norm.bias", prefix);
        bias = q21v_read(vision->header, name, 1, hidden_dim, 0, error, error_size);
        if (!weight || !bias) goto done;
        for (uint32_t i = 0; i < seq; i++)
            q21v_layer_norm(values + (size_t)i * QZEN_QWEN21_VIS_HIDDEN,
                            weight, bias,
                            values + (size_t)i * QZEN_QWEN21_VIS_HIDDEN,
                            QZEN_QWEN21_VIS_HIDDEN);
        free(weight); weight = NULL;
        free(bias); bias = NULL;
    }
    for (uint32_t i = 0; i < merged; i++)
        memcpy(gathered + (size_t)i * Q21V_MERGED_IN,
               values + (size_t)i * 4 * QZEN_QWEN21_VIS_HIDDEN,
               Q21V_MERGED_IN * sizeof(float));
    if (postshuffle) {
        snprintf(name, sizeof(name), "%s.norm.weight", prefix);
        weight = q21v_read(vision->header, name, 1, dim, 0, error, error_size);
        snprintf(name, sizeof(name), "%s.norm.bias", prefix);
        bias = q21v_read(vision->header, name, 1, dim, 0, error, error_size);
        if (!weight || !bias) goto done;
        for (uint32_t i = 0; i < merged; i++)
            q21v_layer_norm(gathered + (size_t)i * Q21V_MERGED_IN,
                            weight, bias,
                            gathered + (size_t)i * Q21V_MERGED_IN,
                            Q21V_MERGED_IN);
        free(weight); weight = NULL;
        free(bias); bias = NULL;
    }
    for (size_t i = 0; i < gather_count; i++) half[i] = q21v_bf16(gathered[i]);
    if (!qzen_gpu_tensor_write_bf16(ctx->gather, half, gather_count)) {
        snprintf(error, error_size, "merger input upload failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    mid = qzen_gpu_tensor_new_bf16(ctx->gpu, gather_count);
    if (!mid) {
        snprintf(error, error_size, "out of device memory for the merger");
        goto done;
    }
    uint64_t fc_shape[2] = {Q21V_MERGED_IN, Q21V_MERGED_IN};
    uint64_t fc_bias[1] = {Q21V_MERGED_IN};
    snprintf(name, sizeof(name), "%s.linear_fc1.weight", prefix);
    weight = q21v_read(vision->header, name, 2, fc_shape, 0, error, error_size);
    snprintf(name, sizeof(name), "%s.linear_fc1.bias", prefix);
    bias = q21v_read(vision->header, name, 1, fc_bias, 0, error, error_size);
    if (!weight || !bias) goto done;
    if (!q21v_gpu_linear(ctx, mid, ctx->gather, weight, bias, merged,
                         Q21V_MERGED_IN, Q21V_MERGED_IN, error, error_size))
        goto done;
    free(weight); weight = NULL;
    free(bias); bias = NULL;
    if (!qzen_gpu_tensor_read_bf16(mid, half, gather_count)) goto done;
    for (size_t i = 0; i < gather_count; i++)
        half[i] = q21v_bf16(q21v_gelu_scalar(q21v_f32(half[i])));
    if (!qzen_gpu_tensor_write_bf16(mid, half, gather_count)) goto done;
    uint64_t out_shape[2] = {QZEN_QWEN21_VIS_OUT, Q21V_MERGED_IN};
    uint64_t out_bias[1] = {QZEN_QWEN21_VIS_OUT};
    snprintf(name, sizeof(name), "%s.linear_fc2.weight", prefix);
    weight = q21v_read(vision->header, name, 2, out_shape, 0, error, error_size);
    snprintf(name, sizeof(name), "%s.linear_fc2.bias", prefix);
    bias = q21v_read(vision->header, name, 1, out_bias, 0, error, error_size);
    if (!weight || !bias) goto done;
    if (!q21v_gpu_linear(ctx, ctx->mout, mid, weight, bias, merged,
                         Q21V_MERGED_IN, QZEN_QWEN21_VIS_OUT, error,
                         error_size)) goto done;
    uint16_t *out_raw = malloc(out_count * sizeof(*out_raw));
    if (!out_raw) goto done;
    if (!qzen_gpu_tensor_read_bf16(ctx->mout, out_raw, out_count)) {
        free(out_raw);
        snprintf(error, error_size, "merger output read failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    for (size_t i = 0; i < out_count; i++) out[i] = q21v_f32(out_raw[i]);
    free(out_raw);
    ok = 1;
done:
    if (!ok && error && error_size && !error[0])
        snprintf(error, error_size, "visual merger failed");
    if (mid) qzen_gpu_tensor_free(mid);
    free(raw); free(values); free(gathered); free(half);
    free(weight); free(bias);
    return ok;
}

static int q21v_gpu_enabled(void) {
    const char *value = getenv("QWEN21_TEXT_GPU");
    if (!value || !value[0]) return 0;
    if (!strcmp(value, "0") || !strcmp(value, "false") || !strcmp(value, "no"))
        return 0;
    return 1;
}

static int q21v_forward_gpu(qzen_qwen21_vision *vision,
                            const uint8_t *const *rgb, const uint32_t *widths,
                            const uint32_t *heights, int count, float **merged,
                            float *deep[QZEN_QWEN21_VIS_DEEPSTACK],
                            int *pad_count, char *error, size_t error_size) {
    q21v_gpu *ctx = vision->gpu;
    size_t total = 0;
    for (int i = 0; i < count; i++) {
        if (widths[i] % QZEN_QWEN21_VIS_ALIGN ||
            heights[i] % QZEN_QWEN21_VIS_ALIGN) {
            snprintf(error, error_size, "visual reference %d is not 32-aligned", i);
            return 0;
        }
        total += (size_t)(widths[i] / 16) * (heights[i] / 16);
    }
    uint32_t seq = (uint32_t)total;
    uint32_t merged_tokens = seq / 4;
    if (!q21v_gpu_alloc(ctx, seq, error, error_size)) return 0;

    uint64_t patch_shape[5] = {QZEN_QWEN21_VIS_HIDDEN, 3, 2, 16, 16};
    uint64_t patch_bias_shape[1] = {QZEN_QWEN21_VIS_HIDDEN};
    uint64_t pos_shape[2] = {Q21V_POS_ROWS, QZEN_QWEN21_VIS_HIDDEN};
    float *patch_weight = NULL, *patch_bias = NULL, *pos = NULL;
    float *cos = malloc((size_t)seq * Q21V_HEAD_DIM * sizeof(float));
    float *sin = malloc((size_t)seq * Q21V_HEAD_DIM * sizeof(float));
    int ok = 0;
    if (!cos || !sin) {
        snprintf(error, error_size, "out of memory for visual RoPE");
        goto done;
    }
    patch_weight = q21v_read(vision->header, "visual.patch_embed.proj.weight",
                             5, patch_shape, 0, error, error_size);
    patch_bias = q21v_read(vision->header, "visual.patch_embed.proj.bias",
                           1, patch_bias_shape, 0, error, error_size);
    pos = q21v_read(vision->header, "visual.pos_embed.weight", 2, pos_shape,
                    0, error, error_size);
    if (!patch_weight || !patch_bias || !pos) goto done;

    size_t offset = 0;
    for (int image = 0; image < count; image++) {
        size_t patches = (size_t)(widths[image] / 16) * (heights[image] / 16);
        float *raw = malloc(patches * 1536 * sizeof(float));
        float *positions = malloc(patches * QZEN_QWEN21_VIS_HIDDEN * sizeof(float));
        uint16_t *half = malloc(patches * 1536 * sizeof(*half));
        if (!raw || !positions || !half) {
            free(raw); free(positions); free(half);
            snprintf(error, error_size, "out of memory for visual patches");
            goto done;
        }
        q21v_patchify(rgb[image], widths[image], heights[image], raw);
        for (size_t i = 0; i < patches * 1536; i++) half[i] = q21v_bf16(raw[i]);
        if (!qzen_gpu_tensor_write_bf16_range(ctx->patch, offset * 1536, half,
                                              (uint32_t)(patches * 1536))) {
            free(raw); free(positions); free(half);
            snprintf(error, error_size, "patch upload failed: %s",
                     qzen_gpu_error(ctx->gpu));
            goto done;
        }
        free(raw);
        q21v_interp_pos(pos, (int)(heights[image] / 16),
                        (int)(widths[image] / 16), positions);
        for (size_t i = 0; i < patches * QZEN_QWEN21_VIS_HIDDEN; i++)
            half[i] = q21v_bf16(positions[i]);
        if (!qzen_gpu_tensor_write_bf16_range(
                ctx->pos, offset * QZEN_QWEN21_VIS_HIDDEN, half,
                (uint32_t)(patches * QZEN_QWEN21_VIS_HIDDEN))) {
            free(positions); free(half);
            snprintf(error, error_size, "position upload failed: %s",
                     qzen_gpu_error(ctx->gpu));
            goto done;
        }
        free(positions); free(half);
        q21v_rope((int)(heights[image] / 16), (int)(widths[image] / 16),
                  cos + offset * Q21V_HEAD_DIM, sin + offset * Q21V_HEAD_DIM);
        offset += patches;
    }
    free(pos); pos = NULL;

    if (!q21v_gpu_linear(ctx, ctx->hidden, ctx->patch, patch_weight, patch_bias,
                         seq, 1536, QZEN_QWEN21_VIS_HIDDEN, error, error_size))
        goto done;
    free(patch_weight); patch_weight = NULL;
    free(patch_bias); patch_bias = NULL;
    if (!qzen_gpu_add_bf16(ctx->gpu, ctx->hidden, ctx->hidden, ctx->pos,
                           (uint32_t)((size_t)seq * QZEN_QWEN21_VIS_HIDDEN))) {
        snprintf(error, error_size, "position add failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }

    for (int layer = 0; layer < Q21V_DEPTH; layer++) {
        if (!q21v_gpu_block(vision, layer, cos, sin, seq, error, error_size))
            goto done;
        for (int slot = 0; slot < QZEN_QWEN21_VIS_DEEPSTACK; slot++) {
            if (q21v_deepstack_layers[slot] != layer) continue;
            deep[slot] = malloc((size_t)merged_tokens * QZEN_QWEN21_VIS_OUT *
                                sizeof(float));
            char prefix[128];
            snprintf(prefix, sizeof(prefix), "visual.deepstack_merger_list.%d", slot);
            if (!deep[slot] || !q21v_gpu_merger(vision, prefix, 1, seq,
                                                deep[slot], error, error_size))
                goto done;
        }
    }
    *merged = malloc((size_t)merged_tokens * QZEN_QWEN21_VIS_OUT * sizeof(float));
    if (!*merged || !q21v_gpu_merger(vision, "visual.merger", 0, seq, *merged,
                                     error, error_size)) goto done;
    *pad_count = (int)merged_tokens;
    ok = 1;
done:
    if (!ok) {
        free(*merged); *merged = NULL;
        for (int i = 0; i < QZEN_QWEN21_VIS_DEEPSTACK; i++) {
            free(deep[i]); deep[i] = NULL;
        }
    }
    free(cos); free(sin);
    free(patch_weight); free(patch_bias); free(pos);
    return ok;
}

qzen_qwen21_vision *qzen_qwen21_vision_open(qzen_st_header *header,
                                            int thread_count, char *error,
                                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!header) {
        snprintf(error, error_size, "no Qwen3-VL header for the visual tower");
        return NULL;
    }
    qzen_qwen21_vision *vision = calloc(1, sizeof(*vision));
    if (!vision) {
        snprintf(error, error_size, "out of memory for the visual tower");
        return NULL;
    }
    vision->header = header;
    vision->threads = thread_count > 0 ? thread_count : 4;
    if (vision->threads < 1) vision->threads = 1;
    if (q21v_gpu_enabled()) {
        char local[256] = {0};
        q21v_gpu *ctx = calloc(1, sizeof(*ctx));
        if (ctx) {
            ctx->gpu = qzen_gpu_create(NULL, local, sizeof(local));
            if (!ctx->gpu) {
                free(ctx);
                ctx = NULL;
            }
        }
        vision->gpu = ctx;
    }
    return vision;
}

void qzen_qwen21_vision_close(qzen_qwen21_vision *vision) {
    if (!vision) return;
    q21v_gpu_free(vision->gpu);
    free(vision);
}

int qzen_qwen21_vision_forward(qzen_qwen21_vision *vision,
                               const uint8_t *const *rgb,
                               const uint32_t *widths,
                               const uint32_t *heights,
                               int count,
                               float **merged,
                               float *deep[QZEN_QWEN21_VIS_DEEPSTACK],
                               int *pad_count,
                               char *error, size_t error_size) {
    if (!vision || !rgb || !widths || !heights || !merged || !deep || !pad_count ||
        count <= 0) {
        if (error && error_size)
            snprintf(error, error_size, "invalid visual tower request");
        return 0;
    }
    *merged = NULL;
    for (int i = 0; i < QZEN_QWEN21_VIS_DEEPSTACK; i++) deep[i] = NULL;
    *pad_count = 0;
    if (vision->gpu)
        return q21v_forward_gpu(vision, rgb, widths, heights, count, merged,
                                deep, pad_count, error, error_size);
    return q21v_forward_cpu(vision, rgb, widths, heights, count, merged, deep,
                            pad_count, error, error_size);
}
