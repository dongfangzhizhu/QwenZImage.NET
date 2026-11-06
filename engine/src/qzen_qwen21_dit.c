/* Qwen-Image 2.1 (7B) DiT: f32 CPU reference forward and INT8 ConvRot loader.
 *
 * Mirrors Wan2GP models/qwen21/transformer.py (QwenImage21Transformer2DModel):
 *   - joint sequence [text prefix ; image tokens], one stream of attention: the
 *     prefix is causal over itself, the target tokens see every key;
 *   - adaLN "continuous" modulation x * (1 + scale), residual += tanh(gate) * y,
 *     with two timestep rows (sigma for the target tokens, t = 0 for the prefix);
 *   - to_q/to_k/to_v -> 32 x 128 -> weighted RMSNorm -> complex (interleaved) RoPE
 *     with axes [16, 56, 56], theta 10000 -> SDPA(1/sqrt(128)) -> to_out.0;
 *   - SwiGLU (silu(gate_layer) * proj) then out; every linear is bias free.
 *
 * Weights: every block linear is comfy_quant_v1 int8_convrot, the same format as
 * the MiniMax-H3 checkpoint, so block tensors are read with qzen_st_read_f32()
 * (per-row F32 scale with the 1/16 fold, 256-point regular Hadamard un-rotation
 * along the input dimension). Everything runs in f32 so the CPU gate can be
 * compared with the torch reference at the 1e-6 level.
 */

#include "qzen_qwen21_dit.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "qzen_qwen21_ops.h"

static float q21_bf16_to_f32(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16u;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

/* TEMPORARY diagnostic (remove before committing): Q21_DUMP_DIR=<dir> and
 * Q21_DUMP_TAG=<tag> write every block stage to <dir>/<tag>_<stage>.f32 so the
 * CPU and GPU pipelines can be compared stage by stage. Inert without the
 * environment variable. */
static void q21_dump_f32(const char *stage, const float *data, size_t count) {
    const char *directory = getenv("Q21_DUMP_DIR");
    if (!directory || !data) return;
    const char *tag = getenv("Q21_DUMP_TAG");
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_%s.f32", directory, tag ? tag : "x",
             stage);
    FILE *file = fopen(path, "wb");
    if (!file) return;
    if (fwrite(data, sizeof(float), count, file) != count) {
        fclose(file);
        return;
    }
    fclose(file);
}

static void q21_dump_f32_tensor(const char *stage, qzen_gpu_tensor *tensor,
                                size_t count) {
    if (!getenv("Q21_DUMP_DIR") || !tensor) return;
    float *values = malloc(count * sizeof(float));
    if (!values) return;
    if (qzen_gpu_tensor_read_f32(tensor, values, count))
        q21_dump_f32(stage, values, count);
    free(values);
}

static void q21_dump_bf16(const char *stage, qzen_gpu_tensor *tensor,
                          size_t count) {
    if (!getenv("Q21_DUMP_DIR") || !tensor) return;
    uint16_t *bits = malloc(count * sizeof(uint16_t));
    float *values = malloc(count * sizeof(float));
    if (!bits || !values) {
        free(bits);
        free(values);
        return;
    }
    if (qzen_gpu_tensor_read_bf16(tensor, bits, count)) {
        for (size_t i = 0; i < count; i++) {
            union {
                uint32_t word;
                float as_float;
            } converted;
            converted.word = (uint32_t)bits[i] << 16u;
            values[i] = converted.as_float;
        }
        q21_dump_f32(stage, values, count);
    }
    free(bits);
    free(values);
}

#define Q21_HEADS 32
#define Q21_LAYERS 32
#define Q21_GROUPS 64 /* complex RoPE pairs: 8 (frame) + 28 (height) + 28 (width) */
#define Q21_ROPE_THETA 10000.0
#define Q21_ROPE_FRAME 16
#define Q21_ROPE_HEIGHT 56
#define Q21_ROPE_WIDTH 56
#define Q21_TIME_DIM 256
#define Q21_OUT_CHANNELS 64
#define Q21_IN_CHANNELS 64

struct q21_gpu_ctx;
static struct q21_gpu_ctx *q21_gpu_open(qzen_qwen21_model *model, char *error,
                                        size_t error_size);
static void q21_gpu_free(struct q21_gpu_ctx *ctx);
static int q21_forward_device(qzen_qwen21_model *model, const float *text,
                              int text_tokens,
                              const qzen_gpu_tensor *latent_device, int height,
                              int width, float sigma,
                              qzen_gpu_tensor *velocity_out, float *block0_out,
                              char *error, size_t error_size);
qzen_gpu_tensor *qzen_qwen21_device_velocity(qzen_qwen21_model *model, int slot,
                                         int image_tokens);

#include "qzen_gpu.h"

typedef struct {
    float *attn_q, *attn_k, *attn_v, *attn_out; /* [DIM][DIM] */
    float *mlp_gate, *mlp_proj;                 /* [MLP][DIM] */
    float *mlp_out;                             /* [DIM][MLP] */
    float norm_q[Q21_HEAD_DIM], norm_k[Q21_HEAD_DIM];
} q21_block;

struct qzen_qwen21_model {
    qzen_st_header header;
    int threads;
    float *img_in;     /* [DIM][64] */
    float *txt_in1;    /* [DIM][DIM] */
    float *txt_in2;    /* [DIM][DIM] */
    float *text_norm;  /* [DIM], the zero-centred parameter (weight + 1 at use) */
    float *time1;      /* [DIM][256] */
    float *time2;      /* [DIM][DIM] */
    float *modulation; /* [4 * DIM][DIM] */
    float *norm_out;   /* [DIM][DIM] */
    float *proj_out;   /* [64][DIM] */
    double weight_seconds;
    struct q21_gpu_ctx *gpu_ctx; /* QWEN21_GPU=1: resident INT8 weights */
};

/* ------------------------------------------------------------- activations */

/* RMSNorm over the last dimension with a dense weight (zero-centred callers pass
 * weight + 1 themselves). */
/* x *= 1 + scale, with the scale row chosen by region (the prefix uses the last
 * modulation row, the target tokens the sigma row). */
static void q21_modulate(float *restrict x, int tokens, int text_tokens, int dim,
                         const float *restrict text_scale, const float *restrict target_scale) {
    for (int t = 0; t < tokens; t++) {
        const float *scale = t < text_tokens ? text_scale : target_scale;
        float *row = x + (size_t)t * dim;
        for (int i = 0; i < dim; i++) row[i] *= 1.0f + scale[i];
    }
}

static void q21_gated_residual(float *restrict x, const float *restrict y, int tokens,
                               int text_tokens, int dim, const float *restrict text_gate,
                               const float *restrict target_gate) {
    for (int t = 0; t < tokens; t++) {
        const float *gate = t < text_tokens ? text_gate : target_gate;
        float *dst = x + (size_t)t * dim;
        const float *src = y + (size_t)t * dim;
        for (int i = 0; i < dim; i++) dst[i] += tanhf(gate[i]) * src[i];
    }
}

/* -------------------------------------------------------------------- rope */

/* Axes [16, 56, 56] -> 8 + 28 + 28 complex pairs. A text token t sits at
 * (frame, height, width) = (t, t, t); every image token sits at frame = the text
 * length with centred height/width indices in raster order (QwenImage21Rope). */
static void q21_rope_tables(int text_tokens, int height, int width, float *cos_tab,
                            float *sin_tab) {
    int tokens = text_tokens + height * width;
    double inverse[Q21_GROUPS];
    int axes[3] = {Q21_ROPE_FRAME, Q21_ROPE_HEIGHT, Q21_ROPE_WIDTH};
    int cursor = 0;
    for (int a = 0; a < 3; a++) {
        int dim = axes[a];
        for (int p = 0; p < dim / 2; p++) {
            inverse[cursor++] = exp(-log(Q21_ROPE_THETA) * (double)(2 * p) / (double)dim);
        }
    }
    const int frame_pairs = Q21_ROPE_FRAME / 2;
    const int height_pairs = Q21_ROPE_HEIGHT / 2;
    for (int t = 0; t < tokens; t++) {
        int frame = 0, high = 0, wide = 0;
        if (t < text_tokens) {
            frame = t;
            high = t;
            wide = t;
        } else {
            int index = t - text_tokens;
            frame = text_tokens;
            high = index / width - (height - height / 2);
            wide = index % width - (width - width / 2);
        }
        for (int p = 0; p < Q21_GROUPS; p++) {
            int value = p < frame_pairs ? frame : p < frame_pairs + height_pairs ? high : wide;
            double angle = (double)value * inverse[p];
            cos_tab[(size_t)t * Q21_GROUPS + p] = (float)cos(angle);
            sin_tab[(size_t)t * Q21_GROUPS + p] = (float)sin(angle);
        }
    }
}

/* Weighted RMSNorm over the head, then the rotation of channel pairs
 * (2p, 2p + 1): the interleaved complex multiply of kernels.complex_norm_rope. */
static void q21_norm_rope(float *x, int tokens, const float *weight, const float *cos_tab,
                          const float *sin_tab) {
    for (int t = 0; t < tokens; t++) {
        const float *cosine = cos_tab + (size_t)t * Q21_GROUPS;
        const float *sine = sin_tab + (size_t)t * Q21_GROUPS;
        for (int h = 0; h < Q21_HEADS; h++) {
            float *head = x + (size_t)t * Q21_HIDDEN + (size_t)h * Q21_HEAD_DIM;
            double sum = 0.0;
            for (int d = 0; d < Q21_HEAD_DIM; d++) sum += (double)head[d] * (double)head[d];
            float inverse = 1.0f / sqrtf((float)(sum / (double)Q21_HEAD_DIM) + Q21_EPS);
            for (int d = 0; d < Q21_HEAD_DIM; d++) head[d] = head[d] * inverse * weight[d];
            for (int p = 0; p < Q21_GROUPS; p++) {
                float first = head[2 * p];
                float second = head[2 * p + 1];
                head[2 * p] = first * cosine[p] - second * sine[p];
                head[2 * p + 1] = first * sine[p] + second * cosine[p];
            }
        }
    }
}

/* --------------------------------------------------------------- attention */

typedef struct {
    const float *q, *k, *v;
    float *out;
    float *scores; /* [threads][HEADS][tokens] */
    int tokens;
    int text_tokens;
    float scale;
} q21_attention_job;

static void q21_attention_queries(void *context, int slot, int start, int end) {
    q21_attention_job *job = context;
    float *scratch = job->scores + (size_t)slot * Q21_HEADS * (size_t)job->tokens;
    for (int q = start; q < end; q++) {
        int keys = q < job->text_tokens ? q + 1 : job->tokens;
        for (int h = 0; h < Q21_HEADS; h++) {
            const float *query = job->q + (size_t)q * Q21_HIDDEN + (size_t)h * Q21_HEAD_DIM;
            float *scores = scratch + (size_t)h * (size_t)job->tokens;
            float best = -INFINITY;
            for (int j = 0; j < keys; j++) {
                const float *key = job->k + (size_t)j * Q21_HIDDEN + (size_t)h * Q21_HEAD_DIM;
                float sum = 0.0f;
                for (int d = 0; d < Q21_HEAD_DIM; d++) sum += query[d] * key[d];
                sum *= job->scale;
                scores[j] = sum;
                if (sum > best) best = sum;
            }
            float total = 0.0f;
            for (int j = 0; j < keys; j++) {
                float value = expf(scores[j] - best);
                scores[j] = value;
                total += value;
            }
            float inverse = 1.0f / total;
            float *dst = job->out + (size_t)q * Q21_HIDDEN + (size_t)h * Q21_HEAD_DIM;
            for (int d = 0; d < Q21_HEAD_DIM; d++) dst[d] = 0.0f;
            for (int j = 0; j < keys; j++) {
                float weight = scores[j] * inverse;
                const float *value = job->v + (size_t)j * Q21_HIDDEN + (size_t)h * Q21_HEAD_DIM;
                for (int d = 0; d < Q21_HEAD_DIM; d++) dst[d] += weight * value[d];
            }
        }
    }
}

/* ---------------------------------------------------------------- weights */

static void q21_block_free(q21_block *block) {
    free(block->attn_q);
    free(block->attn_k);
    free(block->attn_v);
    free(block->attn_out);
    free(block->mlp_gate);
    free(block->mlp_proj);
    free(block->mlp_out);
    memset(block, 0, sizeof(*block));
}

static int q21_block_load(qzen_qwen21_model *model, int layer, q21_block *block, char *error,
                          size_t error_size) {
    const uint64_t dense[2] = {Q21_HIDDEN, Q21_HIDDEN};
    const uint64_t wide[2] = {Q21_FFN, Q21_HIDDEN};
    const uint64_t down[2] = {Q21_HIDDEN, Q21_FFN};
    const uint64_t vector[1] = {Q21_HEAD_DIM};
    const char *suffixes[7] = {"attn.to_q.weight",           "attn.to_k.weight",
                               "attn.to_v.weight",           "attn.to_out.0.weight",
                               "img_mlp.gate_layer.weight",  "img_mlp.proj.weight",
                               "img_mlp.out.weight"};
    const uint64_t *shapes[7] = {dense, dense, dense, dense, wide, wide, down};
    char name[160];
    memset(block, 0, sizeof(*block));
    double started = q21_now();
    for (int index = 0; index < 7; index++) {
        snprintf(name, sizeof(name), "transformer_blocks.%d.%s", layer, suffixes[index]);
        float *values = q21_read_f32(&model->header, name, 2, shapes[index], error, error_size);
        if (!values) {
            q21_block_free(block);
            return 0;
        }
        switch (index) {
            case 0: block->attn_q = values; break;
            case 1: block->attn_k = values; break;
            case 2: block->attn_v = values; break;
            case 3: block->attn_out = values; break;
            case 4: block->mlp_gate = values; break;
            case 5: block->mlp_proj = values; break;
            default: block->mlp_out = values; break;
        }
    }
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", layer);
    float *norm = q21_read_f32(&model->header, name, 1, vector, error, error_size);
    if (!norm) {
        q21_block_free(block);
        return 0;
    }
    memcpy(block->norm_q, norm, sizeof(block->norm_q));
    free(norm);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", layer);
    norm = q21_read_f32(&model->header, name, 1, vector, error, error_size);
    if (!norm) {
        q21_block_free(block);
        return 0;
    }
    memcpy(block->norm_k, norm, sizeof(block->norm_k));
    free(norm);
    model->weight_seconds += q21_now() - started;
    return 1;
}

/* ---------------------------------------------------------------- scratch */

typedef struct {
    int tokens;
    int threads;
    float *norm;          /* [tokens][DIM] */
    float *q, *k, *v;     /* [tokens][DIM] */
    float *attention;     /* [tokens][DIM] */
    float *projected;     /* [tokens][DIM] */
    float *gate, *up;     /* [tokens][MLP] */
    float *mlp;           /* [tokens][DIM] */
    float *scores;        /* [threads][HEADS][tokens] */
    float *cos_tab;       /* [tokens][GROUPS] */
    float *sin_tab;       /* [tokens][GROUPS] */
} q21_scratch;

static void q21_scratch_free(q21_scratch *scratch) {
    free(scratch->norm);
    free(scratch->q);
    free(scratch->k);
    free(scratch->v);
    free(scratch->attention);
    free(scratch->projected);
    free(scratch->gate);
    free(scratch->up);
    free(scratch->mlp);
    free(scratch->scores);
    free(scratch->cos_tab);
    free(scratch->sin_tab);
    memset(scratch, 0, sizeof(*scratch));
}

static int q21_scratch_init(q21_scratch *scratch, int tokens, int threads, char *error,
                            size_t error_size) {
    memset(scratch, 0, sizeof(*scratch));
    scratch->tokens = tokens;
    scratch->threads = threads;
    size_t dim = (size_t)tokens * Q21_HIDDEN;
    size_t wide = (size_t)tokens * Q21_FFN;
    scratch->norm = malloc(dim * sizeof(float));
    scratch->q = malloc(dim * sizeof(float));
    scratch->k = malloc(dim * sizeof(float));
    scratch->v = malloc(dim * sizeof(float));
    scratch->attention = malloc(dim * sizeof(float));
    scratch->projected = malloc(dim * sizeof(float));
    scratch->gate = malloc(wide * sizeof(float));
    scratch->up = malloc(wide * sizeof(float));
    scratch->mlp = malloc(dim * sizeof(float));
    scratch->scores = malloc((size_t)threads * Q21_HEADS * (size_t)tokens * sizeof(float));
    scratch->cos_tab = malloc((size_t)tokens * Q21_GROUPS * sizeof(float));
    scratch->sin_tab = malloc((size_t)tokens * Q21_GROUPS * sizeof(float));
    if (!scratch->norm || !scratch->q || !scratch->k || !scratch->v || !scratch->attention ||
        !scratch->projected || !scratch->gate || !scratch->up || !scratch->mlp ||
        !scratch->scores || !scratch->cos_tab || !scratch->sin_tab) {
        snprintf(error, error_size, "out of memory for %d tokens of DiT scratch", tokens);
        q21_scratch_free(scratch);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ block */

/* One QwenImage21TransformerBlock. module row 0 is the sigma row (target
 * tokens), row 1 the t = 0 row (text prefix); each row reads
 * [scale1 | gate1 | scale2 | gate2]. */
static void q21_block_run(const q21_block *block, float *hidden, int tokens, int text_tokens,
                          const float *module, q21_scratch *scratch, int threads) {
    const float *scale1_target = module;
    const float *gate1_target = module + Q21_HIDDEN;
    const float *scale2_target = module + 2 * Q21_HIDDEN;
    const float *gate2_target = module + 3 * Q21_HIDDEN;
    const float *scale1_text = module + 4 * Q21_HIDDEN;
    const float *gate1_text = module + 5 * Q21_HIDDEN;
    const float *scale2_text = module + 6 * Q21_HIDDEN;
    const float *gate2_text = module + 7 * Q21_HIDDEN;

    q21_layer_norm(hidden, tokens, Q21_HIDDEN, scratch->norm);
    q21_modulate(scratch->norm, tokens, text_tokens, Q21_HIDDEN, scale1_text, scale1_target);
    q21_dump_f32("ln1", scratch->norm, (size_t)tokens * Q21_HIDDEN);
    q21_linear(scratch->norm, tokens, block->attn_q, Q21_HIDDEN, Q21_HIDDEN, scratch->q, threads);
    q21_linear(scratch->norm, tokens, block->attn_k, Q21_HIDDEN, Q21_HIDDEN, scratch->k, threads);
    q21_linear(scratch->norm, tokens, block->attn_v, Q21_HIDDEN, Q21_HIDDEN, scratch->v, threads);
    q21_norm_rope(scratch->q, tokens, block->norm_q, scratch->cos_tab, scratch->sin_tab);
    q21_norm_rope(scratch->k, tokens, block->norm_k, scratch->cos_tab, scratch->sin_tab);
    q21_dump_f32("q", scratch->q, (size_t)tokens * Q21_HIDDEN);
    q21_dump_f32("k", scratch->k, (size_t)tokens * Q21_HIDDEN);
    q21_dump_f32("v", scratch->v, (size_t)tokens * Q21_HIDDEN);
    q21_attention_job attention = {scratch->q,
                                   scratch->k,
                                   scratch->v,
                                   scratch->attention,
                                   scratch->scores,
                                   tokens,
                                   text_tokens,
                                   1.0f / sqrtf((float)Q21_HEAD_DIM)};
    q21_parallel(tokens, threads, q21_attention_queries, &attention);
    q21_dump_f32("attn", scratch->attention, (size_t)tokens * Q21_HIDDEN);
    q21_linear(scratch->attention, tokens, block->attn_out, Q21_HIDDEN, Q21_HIDDEN,
               scratch->projected, threads);
    q21_dump_f32("projected", scratch->projected, (size_t)tokens * Q21_HIDDEN);
    q21_gated_residual(hidden, scratch->projected, tokens, text_tokens, Q21_HIDDEN, gate1_text,
                       gate1_target);
    q21_dump_f32("hidden1", hidden, (size_t)tokens * Q21_HIDDEN);

    q21_layer_norm(hidden, tokens, Q21_HIDDEN, scratch->norm);
    q21_modulate(scratch->norm, tokens, text_tokens, Q21_HIDDEN, scale2_text, scale2_target);
    q21_dump_f32("ln2", scratch->norm, (size_t)tokens * Q21_HIDDEN);
    q21_linear(scratch->norm, tokens, block->mlp_gate, Q21_FFN, Q21_HIDDEN, scratch->gate, threads);
    q21_linear(scratch->norm, tokens, block->mlp_proj, Q21_FFN, Q21_HIDDEN, scratch->up, threads);
    q21_swiglu(scratch->gate, scratch->up, (size_t)tokens * Q21_FFN);
    q21_dump_f32("mlp_act", scratch->gate, (size_t)tokens * Q21_FFN);
    q21_linear(scratch->gate, tokens, block->mlp_out, Q21_HIDDEN, Q21_FFN, scratch->mlp, threads);
    q21_dump_f32("mlp", scratch->mlp, (size_t)tokens * Q21_HIDDEN);
    q21_gated_residual(hidden, scratch->mlp, tokens, text_tokens, Q21_HIDDEN, gate2_text,
                       gate2_target);
    q21_dump_f32("hidden2", hidden, (size_t)tokens * Q21_HIDDEN);
}

/* ------------------------------------------------------ time conditioning */

typedef struct {
    float module[2 * 4 * Q21_HIDDEN]; /* [row][scale1 gate1 scale2 gate2] */
    float out_scale[2 * Q21_HIDDEN];  /* norm_out.linear(silu(temb)) per row */
} q21_conditioning;

/* Rows: sigma first, then t = 0 (the "causal condition" row of the prefix). */
static void q21_condition_one(qzen_qwen21_model *model, float sigma, float *module,
                              float *out_scale, int threads) {
    float sinusoid[Q21_TIME_DIM];
    const int half = Q21_TIME_DIM / 2;
    for (int p = 0; p < half; p++) {
        double frequency = exp(-log(Q21_ROPE_THETA) * (double)p / (double)half);
        double angle = (double)sigma * 1000.0 * frequency;
        sinusoid[p] = (float)cos(angle);
        sinusoid[half + p] = (float)sin(angle);
    }
    float projected[Q21_HIDDEN];
    q21_linear(sinusoid, 1, model->time1, Q21_HIDDEN, Q21_TIME_DIM, projected, threads);
    q21_silu(projected, Q21_HIDDEN);
    float embedding[Q21_HIDDEN];
    q21_linear(projected, 1, model->time2, Q21_HIDDEN, Q21_HIDDEN, embedding, threads);
    q21_silu(embedding, Q21_HIDDEN);
    q21_linear(embedding, 1, model->modulation, 4 * Q21_HIDDEN, Q21_HIDDEN, module, threads);
    q21_linear(embedding, 1, model->norm_out, Q21_HIDDEN, Q21_HIDDEN, out_scale, threads);
}

static void q21_condition(qzen_qwen21_model *model, float sigma, q21_conditioning *out,
                          int threads) {
    q21_condition_one(model, sigma, out->module, out->out_scale, threads);
    q21_condition_one(model, 0.0f, out->module + 4 * Q21_HIDDEN, out->out_scale + Q21_HIDDEN,
                      threads);
}

/* ------------------------------------------------------------------ model */

static int q21_model_read(qzen_qwen21_model *model, char *error, size_t error_size) {
    const uint64_t image[2] = {Q21_HIDDEN, Q21_IN_CHANNELS};
    const uint64_t dense[2] = {Q21_HIDDEN, Q21_HIDDEN};
    const uint64_t wide[2] = {4 * Q21_HIDDEN, Q21_HIDDEN};
    const uint64_t time[2] = {Q21_HIDDEN, Q21_TIME_DIM};
    const uint64_t out[2] = {Q21_OUT_CHANNELS, Q21_HIDDEN};
    const uint64_t vector[1] = {Q21_HIDDEN};
    struct {
        float **slot;
        const char *name;
        int ndim;
        const uint64_t *shape;
    } reads[] = {
        {&model->img_in, "img_in.weight", 2, image},
        {&model->txt_in1, "txt_in.in_layer.weight", 2, dense},
        {&model->txt_in2, "txt_in.out_layer.weight", 2, dense},
        {&model->text_norm, "txt_in.text_norm.weight", 1, vector},
        {&model->time1, "time_text_embed.timestep_embedder.linear_1.weight", 2, time},
        {&model->time2, "time_text_embed.timestep_embedder.linear_2.weight", 2, dense},
        {&model->modulation, "modulation.1.weight", 2, wide},
        {&model->norm_out, "norm_out.linear.weight", 2, dense},
        {&model->proj_out, "proj_out.weight", 2, out},
    };
    for (size_t index = 0; index < sizeof(reads) / sizeof(reads[0]); index++) {
        *reads[index].slot =
            q21_read_f32(&model->header, reads[index].name, reads[index].ndim, reads[index].shape, error,
                         error_size);
        if (!*reads[index].slot) return 0;
    }
    return 1;
}

qzen_qwen21_model *qzen_qwen21_open(const char *checkpoint, int thread_count, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!checkpoint) {
        snprintf(error, error_size, "no Qwen-Image 2.1 checkpoint path");
        return NULL;
    }
    qzen_qwen21_model *model = calloc(1, sizeof(*model));
    if (!model) {
        snprintf(error, error_size, "out of memory for the Qwen-Image 2.1 model");
        return NULL;
    }
    if (!qzen_st_read_header(checkpoint, &model->header, error, error_size)) {
        free(model);
        return NULL;
    }
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    int threads = thread_count > 0 ? thread_count : (int)(cores > 0 ? cores : 1);
    if (threads > Q21_MAX_THREADS) threads = Q21_MAX_THREADS;
    if (threads < 1) threads = 1;
    model->threads = threads;
    if (!q21_model_read(model, error, error_size)) {
        qzen_qwen21_close(model);
        return NULL;
    }
    const char *gpu_env = getenv("QWEN21_GPU");
    if (gpu_env && *gpu_env && strcmp(gpu_env, "0") != 0) {
        model->gpu_ctx = q21_gpu_open(model, error, error_size);
        if (!model->gpu_ctx) {
            qzen_qwen21_close(model);
            return NULL;
        }
    }
    return model;
}

void qzen_qwen21_close(qzen_qwen21_model *model) {
    if (!model) return;
    free(model->img_in);
    free(model->txt_in1);
    free(model->txt_in2);
    free(model->text_norm);
    free(model->time1);
    free(model->time2);
    free(model->modulation);
    free(model->norm_out);
    free(model->proj_out);
    if (model->gpu_ctx) q21_gpu_free(model->gpu_ctx);
    qzen_st_free_header(&model->header);
    free(model);
}

double qzen_qwen21_weight_seconds(const qzen_qwen21_model *model) {
    return model ? model->weight_seconds : 0.0;
}

/* --------------------------------------------------------------- forwards */

static float *q21_text_prefix(qzen_qwen21_model *model, const float *text, int text_tokens,
                              char *error, size_t error_size) {
    size_t count = (size_t)text_tokens * Q21_HIDDEN;
    float *first = malloc(count * sizeof(float));
    float *second = malloc(count * sizeof(float));
    float *centred = malloc((size_t)Q21_HIDDEN * sizeof(float));
    if (!first || !second || !centred) {
        snprintf(error, error_size, "out of memory for %d text tokens", text_tokens);
        free(first);
        free(second);
        free(centred);
        return NULL;
    }
    for (int index = 0; index < Q21_HIDDEN; index++) {
        centred[index] = model->text_norm[index] + 1.0f;
    }
    q21_rms_norm(text, text_tokens, Q21_HIDDEN, centred, first);
    q21_linear(first, text_tokens, model->txt_in1, Q21_HIDDEN, Q21_HIDDEN, second,
               model->threads);
    q21_gelu_tanh(second, count);
    q21_linear(second, text_tokens, model->txt_in2, Q21_HIDDEN, Q21_HIDDEN, first,
               model->threads);
    free(second);
    free(centred);
    return first;
}

int qzen_qwen21_forward(qzen_qwen21_model *model, const qzen_qwen21_request *request,
                      float *velocity, float *block0_out, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model || !request || !velocity) {
        snprintf(error, error_size, "invalid Qwen-Image 2.1 forward request");
        return 0;
    }
    const int text_tokens = request->text_tokens;
    const int image_tokens = request->height * request->width;
    const int tokens = text_tokens + image_tokens;
    const int sigma_count = request->sigma_count;
    if (!request->text || !request->latent || !request->sigmas || text_tokens <= 0 ||
        request->height <= 0 || request->width <= 0 || sigma_count <= 0) {
        snprintf(error, error_size, "Qwen-Image 2.1 forward needs text, latent and sigmas");
        return 0;
    }
    if (model->gpu_ctx) {
        const size_t plane = (size_t)image_tokens * Q21_OUT_CHANNELS;
        /* The request latent is host F32 [64][height][width]; upload it once
         * and let every sigma reuse the resident copy. */
        qzen_gpu_tensor *latent_device = qzen_gpu_tensor_from_f32(
            qzen_qwen21_device(model), request->latent,
            (size_t)image_tokens * Q21_IN_CHANNELS);
        if (!latent_device) {
            snprintf(error, error_size, "out of GPU memory for the latent");
            return 0;
        }
        int ok = 1;
        for (int s = 0; s < sigma_count && ok; s++) {
            qzen_gpu_tensor *velocity_device = qzen_qwen21_device_velocity(
                model, 0, image_tokens);
            if (!velocity_device) {
                snprintf(error, error_size, "out of GPU memory for the velocity");
                ok = 0;
                break;
            }
            float *block0_rows =
                block0_out ? block0_out + (size_t)s * tokens * Q21_HIDDEN : NULL;
            if (!q21_forward_device(model, request->text, text_tokens,
                                    latent_device, request->height,
                                    request->width,
                                    request->sigmas[s], velocity_device,
                                    block0_rows, error, error_size)) {
                ok = 0;
                break;
            }
            /* The device velocity buffer is BF16; widen it for the host
             * result. */
            uint16_t *staged = malloc(plane * sizeof(uint16_t));
            if (!staged ||
                !qzen_gpu_tensor_read_bf16(velocity_device, staged, plane)) {
                free(staged);
                snprintf(error, error_size, "velocity download failed");
                ok = 0;
                break;
            }
            for (size_t i = 0; i < plane; i++) {
                velocity[(size_t)s * plane + i] = q21_bf16_to_f32(staged[i]);
            }
            free(staged);
        }
        qzen_gpu_tensor_free(latent_device);
        return ok;
    }
    const int threads = model->threads;
    const size_t dim = (size_t)tokens * Q21_HIDDEN;
    q21_scratch scratch;
    memset(&scratch, 0, sizeof(scratch));
    float *prefix = NULL, *latent = NULL, *image = NULL, *hidden = NULL;
    q21_conditioning *conditions = NULL;
    int ok = 0;

    prefix = q21_text_prefix(model, request->text, text_tokens, error, error_size);
    if (!prefix) goto done;
    latent = malloc((size_t)image_tokens * Q21_IN_CHANNELS * sizeof(float));
    image = malloc((size_t)image_tokens * Q21_HIDDEN * sizeof(float));
    hidden = malloc((size_t)sigma_count * dim * sizeof(float));
    conditions = malloc((size_t)sigma_count * sizeof(*conditions));
    if (!latent || !image || !hidden || !conditions) {
        snprintf(error, error_size, "out of memory for %d image tokens", image_tokens);
        goto done;
    }
    for (int t = 0; t < image_tokens; t++) {
        for (int c = 0; c < Q21_IN_CHANNELS; c++) {
            latent[(size_t)t * Q21_IN_CHANNELS + c] = request->latent[(size_t)c * image_tokens + t];
        }
    }
    q21_linear(latent, image_tokens, model->img_in, Q21_HIDDEN, Q21_IN_CHANNELS, image, threads);
    for (int s = 0; s < sigma_count; s++) {
        float *rows = hidden + (size_t)s * dim;
        memcpy(rows, prefix, (size_t)text_tokens * Q21_HIDDEN * sizeof(float));
        memcpy(rows + (size_t)text_tokens * Q21_HIDDEN, image,
               (size_t)image_tokens * Q21_HIDDEN * sizeof(float));
        q21_condition(model, request->sigmas[s], &conditions[s], threads);
    }
    if (!q21_scratch_init(&scratch, tokens, threads, error, error_size)) goto done;
    q21_rope_tables(text_tokens, request->height, request->width, scratch.cos_tab,
                    scratch.sin_tab);
    for (int layer = 0; layer < Q21_LAYERS; layer++) {
        q21_block block;
        if (!q21_block_load(model, layer, &block, error, error_size)) goto done;
        for (int s = 0; s < sigma_count; s++) {
            float *rows = hidden + (size_t)s * dim;
            q21_block_run(&block, rows, tokens, text_tokens, conditions[s].module, &scratch,
                          threads);
            if (s == 0 && getenv("Q21_DUMP_DIR")) {
                char stage[32];
                snprintf(stage, sizeof(stage), "block%02d", layer);
                q21_dump_f32(stage, rows, dim);
            }
            if (layer == 0 && block0_out) {
                memcpy(block0_out + (size_t)s * dim, rows, dim * sizeof(float));
            }
        }
        q21_block_free(&block);
    }
    for (int s = 0; s < sigma_count; s++) {
        const float *rows = hidden + (size_t)s * dim + (size_t)text_tokens * Q21_HIDDEN;
        float *normed = scratch.norm + (size_t)text_tokens * Q21_HIDDEN;
        q21_layer_norm(rows, image_tokens, Q21_HIDDEN, normed);
        q21_modulate(normed, image_tokens, 0, Q21_HIDDEN, NULL, conditions[s].out_scale);
        q21_linear(normed, image_tokens, model->proj_out, Q21_OUT_CHANNELS, Q21_HIDDEN,
                   velocity + (size_t)s * image_tokens * Q21_OUT_CHANNELS, threads);
    }
    ok = 1;
done:
    free(prefix);
    free(latent);
    free(image);
    free(hidden);
    free(conditions);
    q21_scratch_free(&scratch);
    return ok;
}

int qzen_qwen21_block_forward(qzen_qwen21_model *model, int layer, const float *hidden,
                            int text_tokens, int height, int width, float sigma, float *out,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model || !hidden || !out || layer < 0 || layer >= Q21_LAYERS || text_tokens <= 0 ||
        height <= 0 || width <= 0) {
        snprintf(error, error_size, "invalid Qwen-Image 2.1 block request");
        return 0;
    }
    if (model->gpu_ctx) {
        return qzen_qwen21_block_forward_device(model, layer, hidden, text_tokens,
                                              height, width, sigma, out, error,
                                              error_size);
    }
    const int tokens = text_tokens + height * width;
    const size_t dim = (size_t)tokens * Q21_HIDDEN;
    q21_scratch scratch;
    memset(&scratch, 0, sizeof(scratch));
    q21_conditioning conditions;
    float *rows = malloc(dim * sizeof(float));
    int ok = 0;
    if (!rows) {
        snprintf(error, error_size, "out of memory for %d block tokens", tokens);
        return 0;
    }
    if (!q21_scratch_init(&scratch, tokens, model->threads, error, error_size)) {
        free(rows);
        return 0;
    }
    q21_rope_tables(text_tokens, height, width, scratch.cos_tab, scratch.sin_tab);
    memcpy(rows, hidden, dim * sizeof(float));
    q21_condition(model, sigma, &conditions, model->threads);
    q21_block block;
    if (q21_block_load(model, layer, &block, error, error_size)) {
        q21_block_run(&block, rows, tokens, text_tokens, conditions.module, &scratch,
                      model->threads);
        q21_block_free(&block);
        memcpy(out, rows, dim * sizeof(float));
        ok = 1;
    }
    free(rows);
    q21_scratch_free(&scratch);
    return ok;
}


/* ============================================================ GPU path ==
 *
 * QWEN21_GPU=1 keeps every block linear as resident INT8 ConvRot (the
 * checkpoint format is identical to the H3 DiT: pre-rotated rows, one F32
 * scale per output row), so a forward never touches the 7.26 GB checkpoint.
 * The block math runs on the existing qzen_gpu primitives:
 *
 *   LN(f32 carrier) + scale-only adaLN   qzen_gpu_layer_norm_adaln_f32in_bf16
 *     (true LayerNorm; the Wan RMS adaLN kernel does not apply; the shift
 *      slot reads a zero row of the modulation table)
 *   activation rotation                  qzen_gpu_convrot_rotate_bf16
 *   q/k/v/out projections                qzen_gpu_linear_int8_bf16 (three of
 *     them share one rotated activation; re-quantizing per projection is
 *     cheap next to the GEMM and keeps the Qwen split to_q/to_k/to_v
 *     weights untouched)
 *   per-head RMSNorm(128, weight)        qzen_gpu_head_rms_norm_bf16
 *   interleaved RoPE (pairs (2p, 2p+1))  qzen_gpu_qwen_rope_bf16 (new kernel;
 *     the existing rope kernels use the rotate-half convention)
 *   joint attention                      qzen_gpu_sdpa_flash_bf16 (non-causal,
 *     correct for the image tokens) + qzen_gpu_prefix_causal_bf16 (new kernel
 *     recomputing the tiny causal text-prefix rows)
 *   residual += tanh(gate) * branch      qzen_gpu_gate_f32res_bf16 on the F32
 *     carrier, with tanh folded into the modulation table on the host
 *   SwiGLU                               qzen_gpu_silu_mul_bf16
 *
 * The modulation table is [2 region rows][5 slots][4096] BF16 built on the
 * host by the same f32 conditioning path the CPU forward uses (rows:
 * 0 = sigma/target, 1 = t = 0/text; slots: scale1, gate1, scale2, gate2,
 * zeros). row_map sends text tokens to row 1. The final norm_out uses a
 * [2 slots] table (zeros, out_scale) and an all-zero row map. The device
 * forward runs one sigma per call; qzen_qwen21_forward loops sigmas over the
 * resident weights, so the multi-sigma API and the P0 block0 anchor keep
 * working unchanged.
 */

static uint16_t q21_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

typedef struct {
    qzen_gpu_tensor *q, *q_scale, *k, *k_scale, *v, *v_scale;
    qzen_gpu_tensor *out, *out_scale;
    qzen_gpu_tensor *gate, *gate_scale, *proj, *proj_scale, *down, *down_scale;
    qzen_gpu_tensor *norm_q, *norm_k; /* BF16 [128], uploaded verbatim */
} q21_gpu_block;

typedef struct q21_gpu_ctx {
    qzen_gpu *gpu;
    q21_gpu_block blocks[Q21_LAYERS];
    qzen_gpu_tensor *img_in;   /* BF16 [4096][64] */
    qzen_gpu_tensor *proj_out; /* BF16 [64][4096] */
    qzen_gpu_tensor *ones;     /* BF16 [4096]: affine-free LayerNorm weight */
    qzen_gpu_tensor *row_map;  /* U32 [tokens]: 0 target, 1 text */
    qzen_gpu_tensor *zero_map; /* U32 [tokens]: all zero (final norm) */
    qzen_gpu_tensor *rope_cos, *rope_sin; /* BF16 [tokens][64] */
    qzen_gpu_tensor *mod_table;   /* BF16 [2][5][4096] */
    qzen_gpu_tensor *final_table; /* BF16 [2][4096] */
    qzen_gpu_tensor *carrier;     /* F32 [tokens][4096], the residual */
    qzen_gpu_tensor *normed, *rot, *q, *k, *v, *attn, *projected; /* BF16 [tokens][4096] */
    qzen_gpu_tensor *act8, *act8_scales;
    qzen_gpu_tensor *mlp1, *mlp2, *mlp_act; /* BF16 [tokens][12288] */
    qzen_gpu_tensor *mlp8, *mlp8_scales;
    qzen_gpu_tensor *image_bf16; /* BF16 staging for the img_in projection */
    qzen_gpu_tensor *image_f32;  /* F32 staging for the carrier tail */
    qzen_gpu_tensor *velocity[2]; /* BF16 [image_tokens][64] device results */
    /* Geometry the scratch was sized for. The row map and the RoPE tables
     * depend on it, so a call with the same token count but a different
     * prefix length or canvas has to rebuild them. */
    int tokens, text_tokens, height, width, velocity_slot;
} q21_gpu;

static void q21_gpu_tensor_free(qzen_gpu_tensor **tensor) {
    if (*tensor) {
        qzen_gpu_tensor_free(*tensor);
        *tensor = NULL;
    }
}

static void q21_gpu_scratch_release(q21_gpu *ctx) {
    q21_gpu_tensor_free(&ctx->normed);
    q21_gpu_tensor_free(&ctx->rot);
    q21_gpu_tensor_free(&ctx->q);
    q21_gpu_tensor_free(&ctx->k);
    q21_gpu_tensor_free(&ctx->v);
    q21_gpu_tensor_free(&ctx->attn);
    q21_gpu_tensor_free(&ctx->projected);
    q21_gpu_tensor_free(&ctx->act8);
    q21_gpu_tensor_free(&ctx->act8_scales);
    q21_gpu_tensor_free(&ctx->mlp1);
    q21_gpu_tensor_free(&ctx->mlp2);
    q21_gpu_tensor_free(&ctx->mlp_act);
    q21_gpu_tensor_free(&ctx->mlp8);
    q21_gpu_tensor_free(&ctx->mlp8_scales);
    q21_gpu_tensor_free(&ctx->image_bf16);
    q21_gpu_tensor_free(&ctx->image_f32);
    q21_gpu_tensor_free(&ctx->row_map);
    q21_gpu_tensor_free(&ctx->zero_map);
    q21_gpu_tensor_free(&ctx->rope_cos);
    q21_gpu_tensor_free(&ctx->rope_sin);
    q21_gpu_tensor_free(&ctx->mod_table);
    q21_gpu_tensor_free(&ctx->final_table);
    q21_gpu_tensor_free(&ctx->carrier);
    ctx->tokens = 0;
}

static void q21_gpu_free(q21_gpu *ctx) {
    if (!ctx) return;
    for (int layer = 0; layer < Q21_LAYERS; layer++) {
        q21_gpu_block *block = &ctx->blocks[layer];
        q21_gpu_tensor_free(&block->q);
        q21_gpu_tensor_free(&block->q_scale);
        q21_gpu_tensor_free(&block->k);
        q21_gpu_tensor_free(&block->k_scale);
        q21_gpu_tensor_free(&block->v);
        q21_gpu_tensor_free(&block->v_scale);
        q21_gpu_tensor_free(&block->out);
        q21_gpu_tensor_free(&block->out_scale);
        q21_gpu_tensor_free(&block->gate);
        q21_gpu_tensor_free(&block->gate_scale);
        q21_gpu_tensor_free(&block->proj);
        q21_gpu_tensor_free(&block->proj_scale);
        q21_gpu_tensor_free(&block->down);
        q21_gpu_tensor_free(&block->down_scale);
        q21_gpu_tensor_free(&block->norm_q);
        q21_gpu_tensor_free(&block->norm_k);
    }
    q21_gpu_tensor_free(&ctx->img_in);
    q21_gpu_tensor_free(&ctx->proj_out);
    q21_gpu_tensor_free(&ctx->ones);
    q21_gpu_scratch_release(ctx);
    q21_gpu_tensor_free(&ctx->velocity[0]);
    q21_gpu_tensor_free(&ctx->velocity[1]);
    if (ctx->gpu) qzen_gpu_free(ctx->gpu);
    free(ctx);
}

/* Read one I8 ConvRot weight plus its F32 per-row scale straight from the
 * checkpoint into resident device tensors. The INT8 payload travels through
 * the process-wide pinned staging buffer (every payload length is even, so
 * it can move as uint16 pairs); the GEMM consumes both tensors verbatim. */
static int q21_gpu_load_int8(qzen_st_header *header, qzen_gpu *gpu, const char *name,
                             int rows, int columns, qzen_gpu_tensor **weight,
                             qzen_gpu_tensor **scales, char *error, size_t error_size) {
    const uint64_t shape[2] = {(uint64_t)rows, (uint64_t)columns};
    const qzen_st_tensor *tensor = qzen_st_find(header, name);
    if (!q21_expect(tensor, name, 2, shape, error, error_size)) return 0;
    if (tensor->dtype != QZEN_DTYPE_I8) {
        snprintf(error, error_size, "%s: expected an I8 ConvRot tensor", name);
        return 0;
    }
    char scale_name[256];
    q21_scale_name(name, scale_name, sizeof(scale_name));
    const uint64_t scale_shape[2] = {(uint64_t)rows, 1};
    const qzen_st_tensor *scale = qzen_st_find(header, scale_name);
    if (!q21_expect(scale, scale_name, 2, scale_shape, error, error_size)) return 0;
    size_t elements = (size_t)rows * columns;
    int8_t *payload = malloc(elements);
    float *scale_values = malloc((size_t)rows * sizeof(float));
    *weight = qzen_gpu_tensor_new_i8(gpu, elements);
    *scales = qzen_gpu_tensor_new_f32(gpu, rows);
    if (!payload || !scale_values || !*weight || !*scales) {
        snprintf(error, error_size, "%s: out of memory for the INT8 weight", name);
        goto fail;
    }
    if (!qzen_st_read_data(header, tensor, payload, elements, error, error_size) ||
        !qzen_st_read_data(header, scale, scale_values, (size_t)rows * sizeof(float),
                         error, error_size)) {
        goto fail;
    }
    if (!qzen_gpu_upload_i8_pinned(gpu, *weight, payload, elements)) {
        snprintf(error, error_size, "%s: pinned upload failed", name);
        goto fail;
    }
    if (!qzen_gpu_tensor_write_f32(*scales, scale_values, rows)) {
        snprintf(error, error_size, "%s: scale upload failed", name);
        goto fail;
    }
    free(payload);
    free(scale_values);
    return 1;
fail:
    free(payload);
    free(scale_values);
    q21_gpu_tensor_free(weight);
    q21_gpu_tensor_free(scales);
    return 0;
}

/* Upload a BF16 checkpoint tensor verbatim (the resident weights are stored
 * BF16, so no dequantisation happens on the host at all). */
static qzen_gpu_tensor *q21_gpu_load_bf16(qzen_st_header *header, qzen_gpu *gpu,
                                        const char *name, int ndim,
                                        const uint64_t *shape, char *error,
                                        size_t error_size) {
    const qzen_st_tensor *tensor = qzen_st_find(header, name);
    if (!q21_expect(tensor, name, ndim, shape, error, error_size)) return NULL;
    if (tensor->dtype != QZEN_DTYPE_BF16) {
        snprintf(error, error_size, "%s: expected a BF16 tensor", name);
        return NULL;
    }
    size_t elements = (size_t)qzen_st_tensor_elements(tensor);
    uint16_t *payload = malloc(elements * 2);
    qzen_gpu_tensor *result = qzen_gpu_tensor_new_bf16(gpu, elements);
    if (!payload || !result ||
        !qzen_st_read_data(header, tensor, payload, elements * 2, error, error_size) ||
        !qzen_gpu_upload_bf16_pinned(gpu, result, payload, elements)) {
        free(payload);
        q21_gpu_tensor_free(&result);
        snprintf(error, error_size, "%s: BF16 upload failed", name);
        return NULL;
    }
    free(payload);
    return result;
}

static int q21_gpu_load_block(q21_gpu *ctx, qzen_st_header *header, int layer,
                              q21_gpu_block *block, char *error, size_t error_size) {
    char name[192];
    memset(block, 0, sizeof(*block));
    struct {
        const char *suffix;
        int rows, columns;
        qzen_gpu_tensor **weight, **scales;
    } projections[] = {
        {"attn.to_q.weight", Q21_HIDDEN, Q21_HIDDEN, &block->q, &block->q_scale},
        {"attn.to_k.weight", Q21_HIDDEN, Q21_HIDDEN, &block->k, &block->k_scale},
        {"attn.to_v.weight", Q21_HIDDEN, Q21_HIDDEN, &block->v, &block->v_scale},
        {"attn.to_out.0.weight", Q21_HIDDEN, Q21_HIDDEN, &block->out, &block->out_scale},
        {"img_mlp.gate_layer.weight", Q21_FFN, Q21_HIDDEN, &block->gate, &block->gate_scale},
        {"img_mlp.proj.weight", Q21_FFN, Q21_HIDDEN, &block->proj, &block->proj_scale},
        {"img_mlp.out.weight", Q21_HIDDEN, Q21_FFN, &block->down, &block->down_scale},
    };
    for (size_t index = 0; index < sizeof(projections) / sizeof(projections[0]);
         index++) {
        snprintf(name, sizeof(name), "transformer_blocks.%d.%s", layer,
                 projections[index].suffix);
        if (!q21_gpu_load_int8(header, ctx->gpu, name, projections[index].rows,
                               projections[index].columns, projections[index].weight,
                               projections[index].scales, error, error_size)) {
            return 0;
        }
    }
    const uint64_t vector[1] = {Q21_HEAD_DIM};
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", layer);
    block->norm_q = q21_gpu_load_bf16(header, ctx->gpu, name, 1, vector, error,
                                      error_size);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", layer);
    block->norm_k = q21_gpu_load_bf16(header, ctx->gpu, name, 1, vector, error,
                                      error_size);
    return block->norm_q && block->norm_k;
}

static int q21_gpu_scratch_prepare(q21_gpu *ctx, int tokens, int text_tokens,
                                   int height, int width, char *error,
                                   size_t error_size) {
    if (ctx->tokens >= tokens && ctx->text_tokens == text_tokens &&
        ctx->height == height && ctx->width == width) {
        return 1;
    }
    q21_gpu_scratch_release(ctx);
    const size_t dim = (size_t)tokens * Q21_HIDDEN;
    const size_t wide = (size_t)tokens * Q21_FFN;
    /* The INT8 GEMM quantises the activation in place and needs room for the
     * row-padded tile: qzen_gpu_linear_int8_bf16 requires
     * quantized_input->elements >= padded_rows * input_dim and
     * input_scales->elements >= padded_rows, with padded_rows = round the
     * token count up to a multiple of 128. Allocating exactly tokens * dim
     * makes every call fail with "invalid INT8 linear request" as soon as
     * the token count is not a multiple of 128. */
    const size_t padded = ((size_t)tokens + 127u) & ~(size_t)127u;
    ctx->normed = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->rot = qzen_gpu_tensor_new_bf16(ctx->gpu, wide);
    ctx->q = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->k = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->v = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->attn = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->projected = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->act8 = qzen_gpu_tensor_new_i8(ctx->gpu, padded * Q21_HIDDEN);
    ctx->act8_scales = qzen_gpu_tensor_new_f32(ctx->gpu, padded);
    ctx->mlp1 = qzen_gpu_tensor_new_bf16(ctx->gpu, wide);
    ctx->mlp2 = qzen_gpu_tensor_new_bf16(ctx->gpu, wide);
    ctx->mlp_act = qzen_gpu_tensor_new_bf16(ctx->gpu, wide);
    ctx->mlp8 = qzen_gpu_tensor_new_i8(ctx->gpu, padded * Q21_FFN);
    ctx->mlp8_scales = qzen_gpu_tensor_new_f32(ctx->gpu, padded);
    ctx->image_bf16 = qzen_gpu_tensor_new_bf16(ctx->gpu, dim);
    ctx->image_f32 = qzen_gpu_tensor_new_f32(ctx->gpu, dim);
    ctx->rope_cos = qzen_gpu_tensor_new_bf16(ctx->gpu, (size_t)tokens * Q21_GROUPS);
    ctx->rope_sin = qzen_gpu_tensor_new_bf16(ctx->gpu, (size_t)tokens * Q21_GROUPS);
    ctx->carrier = qzen_gpu_tensor_new_f32(ctx->gpu, dim);
    ctx->mod_table = qzen_gpu_tensor_new_bf16(ctx->gpu, 2 * 5 * Q21_HIDDEN);
    ctx->final_table = qzen_gpu_tensor_new_bf16(ctx->gpu, 2 * Q21_HIDDEN);
    if (!ctx->normed || !ctx->rot || !ctx->q || !ctx->k || !ctx->v || !ctx->attn ||
        !ctx->projected || !ctx->act8 || !ctx->act8_scales || !ctx->mlp1 ||
        !ctx->mlp2 || !ctx->mlp_act || !ctx->mlp8 || !ctx->mlp8_scales ||
        !ctx->image_bf16 || !ctx->image_f32 || !ctx->rope_cos || !ctx->rope_sin ||
        !ctx->carrier || !ctx->mod_table || !ctx->final_table) {
        snprintf(error, error_size, "out of GPU memory for %d tokens of DiT scratch",
                 tokens);
        return 0;
    }
    /* row_map: text tokens read the t = 0 table row, image tokens the sigma
     * row; the final norm sees only image rows and always row 0. */
    uint32_t *map = malloc((size_t)tokens * sizeof(uint32_t));
    if (!map) {
        snprintf(error, error_size, "out of memory for the row map");
        return 0;
    }
    for (int t = 0; t < tokens; t++) map[t] = t < text_tokens ? 1u : 0u;
    qzen_gpu_tensor *uploaded = qzen_gpu_tensor_from_u32(ctx->gpu, map, tokens);
    if (!uploaded) {
        free(map);
        snprintf(error, error_size, "row map upload failed");
        return 0;
    }
    q21_gpu_tensor_free(&ctx->row_map);
    ctx->row_map = uploaded;
    for (int t = 0; t < tokens; t++) map[t] = 0u;
    uploaded = qzen_gpu_tensor_from_u32(ctx->gpu, map, tokens);
    free(map);
    if (!uploaded) {
        snprintf(error, error_size, "zero map upload failed");
        return 0;
    }
    q21_gpu_tensor_free(&ctx->zero_map);
    ctx->zero_map = uploaded;
    /* RoPE tables: the CPU builder produces F32; round to BF16 like every
     * other table this file feeds the GPU. */
    float *cosine = malloc((size_t)tokens * Q21_GROUPS * sizeof(float));
    float *sine = malloc((size_t)tokens * Q21_GROUPS * sizeof(float));
    uint16_t *staged = malloc((size_t)tokens * Q21_GROUPS * sizeof(uint16_t));
    if (!cosine || !sine || !staged) {
        free(cosine);
        free(sine);
        free(staged);
        snprintf(error, error_size, "out of memory for the RoPE tables");
        return 0;
    }
    q21_rope_tables(text_tokens, height, width, cosine, sine);
    for (size_t i = 0; i < (size_t)tokens * Q21_GROUPS; i++) {
        staged[i] = q21_f32_to_bf16(cosine[i]);
    }
    qzen_gpu_tensor *table = qzen_gpu_tensor_from_bf16(ctx->gpu, staged,
                                                   (size_t)tokens * Q21_GROUPS);
    if (!table) goto rope_fail;
    q21_gpu_tensor_free(&ctx->rope_cos);
    ctx->rope_cos = table;
    for (size_t i = 0; i < (size_t)tokens * Q21_GROUPS; i++) {
        staged[i] = q21_f32_to_bf16(sine[i]);
    }
    table = qzen_gpu_tensor_from_bf16(ctx->gpu, staged, (size_t)tokens * Q21_GROUPS);
    if (!table) goto rope_fail;
    q21_gpu_tensor_free(&ctx->rope_sin);
    ctx->rope_sin = table;
    free(cosine);
    free(sine);
    free(staged);
    ctx->tokens = tokens;
    ctx->text_tokens = text_tokens;
    ctx->height = height;
    ctx->width = width;
    return 1;
rope_fail:
    free(cosine);
    free(sine);
    free(staged);
    snprintf(error, error_size, "RoPE table upload failed");
    return 0;
}

/* One transformer block on the device. carrier is the F32 residual. */
static int q21_gpu_block_run(q21_gpu *ctx, const q21_gpu_block *block,
                             qzen_gpu_tensor *carrier, int tokens, int text_tokens,
                             char *error, size_t error_size) {
    const int dim = Q21_HIDDEN;
    const float eps = Q21_EPS;
#define Q21_OP(call, label)                                                    \
    do {                                                                       \
        if (!(call)) {                                                          \
            snprintf(error, error_size, "%s failed: %s", label,                \
                     qzen_gpu_error(ctx->gpu));                                   \
            return 0;                                                           \
        }                                                                       \
    } while (0)
    /* attention sub-block */
    Q21_OP(qzen_gpu_layer_norm_adaln_f32in_bf16(
               ctx->gpu, ctx->normed, carrier, ctx->ones, ctx->mod_table, NULL,
               ctx->row_map, tokens, dim, 5, 4, 0, eps),
           "qwen attention LN+scale");
    q21_dump_bf16("ln1", ctx->normed, (size_t)tokens * dim);
    Q21_OP(qzen_gpu_convrot_rotate_bf16(ctx->gpu, ctx->rot, ctx->normed, tokens, dim),
           "qwen attention ConvRot rotation");
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->q, ctx->act8, ctx->act8_scales,
                                   ctx->rot, block->q, block->q_scale, tokens, dim,
                                   dim, 0, 0),
           "qwen to_q int8");
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->k, ctx->act8, ctx->act8_scales,
                                   ctx->rot, block->k, block->k_scale, tokens, dim,
                                   dim, 0, 0),
           "qwen to_k int8");
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->v, ctx->act8, ctx->act8_scales,
                                   ctx->rot, block->v, block->v_scale, tokens, dim,
                                   dim, 0, 0),
           "qwen to_v int8");
    Q21_OP(qzen_gpu_head_rms_norm_bf16(ctx->gpu, ctx->q, block->norm_q, tokens,
                                     Q21_HEADS, Q21_HEAD_DIM, eps),
           "qwen q head rms");
    Q21_OP(qzen_gpu_head_rms_norm_bf16(ctx->gpu, ctx->k, block->norm_k, tokens,
                                     Q21_HEADS, Q21_HEAD_DIM, eps),
           "qwen k head rms");
    Q21_OP(qzen_gpu_qwen_rope_bf16(ctx->gpu, ctx->q, ctx->k, ctx->rope_cos,
                                 ctx->rope_sin, tokens, Q21_HEADS, Q21_HEAD_DIM),
           "qwen interleaved rope");
    q21_dump_bf16("q", ctx->q, (size_t)tokens * dim);
    q21_dump_bf16("k", ctx->k, (size_t)tokens * dim);
    q21_dump_bf16("v", ctx->v, (size_t)tokens * dim);
    Q21_OP(qzen_gpu_sdpa_flash_bf16(ctx->gpu, ctx->attn, ctx->q, ctx->k, ctx->v,
                                  tokens, tokens, Q21_HEADS, Q21_HEAD_DIM,
                                  1.0f / sqrtf((float)Q21_HEAD_DIM)),
           "qwen joint attention");
    if (text_tokens > 0) {
        Q21_OP(qzen_gpu_prefix_causal_bf16(ctx->gpu, ctx->attn, ctx->q, ctx->k,
                                         ctx->v, text_tokens, tokens, Q21_HEADS,
                                         Q21_HEAD_DIM,
                                         1.0f / sqrtf((float)Q21_HEAD_DIM)),
               "qwen causal prefix fixup");
    }
    q21_dump_bf16("attn", ctx->attn, (size_t)tokens * dim);
    /* to_out.0 is int8 ConvRot like every other block linear: its stored rows
     * are pre-rotated along the input dimension, so the attention output has
     * to go through the same Hadamard rotation before the GEMM. */
    Q21_OP(qzen_gpu_convrot_rotate_bf16(ctx->gpu, ctx->rot, ctx->attn, tokens, dim),
           "qwen attention out ConvRot rotation");
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->projected, ctx->act8,
                                   ctx->act8_scales, ctx->rot, block->out,
                                   block->out_scale, tokens, dim, dim, 0, 0),
           "qwen to_out int8");
    q21_dump_bf16("projected", ctx->projected, (size_t)tokens * dim);
    Q21_OP(qzen_gpu_gate_f32res_bf16(ctx->gpu, carrier, ctx->projected,
                                   ctx->mod_table, NULL, 0, ctx->row_map, tokens,
                                   dim, 5, 1),
           "qwen attention gate");
    q21_dump_f32_tensor("hidden1", carrier, (size_t)tokens * dim);
    /* FFN sub-block */
    Q21_OP(qzen_gpu_layer_norm_adaln_f32in_bf16(
               ctx->gpu, ctx->normed, carrier, ctx->ones, ctx->mod_table, NULL,
               ctx->row_map, tokens, dim, 5, 4, 2, eps),
           "qwen ffn LN+scale");
    q21_dump_bf16("ln2", ctx->normed, (size_t)tokens * dim);
    Q21_OP(qzen_gpu_convrot_rotate_bf16(ctx->gpu, ctx->rot, ctx->normed, tokens, dim),
           "qwen ffn ConvRot rotation");
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->mlp1, ctx->act8, ctx->act8_scales,
                                   ctx->rot, block->gate, block->gate_scale, tokens,
                                   dim, Q21_FFN, 0, 0),
           "qwen mlp gate int8");
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->mlp2, ctx->act8, ctx->act8_scales,
                                   ctx->rot, block->proj, block->proj_scale, tokens,
                                   dim, Q21_FFN, 0, 0),
           "qwen mlp proj int8");
    Q21_OP(qzen_gpu_silu_mul_bf16(ctx->gpu, ctx->mlp_act, ctx->mlp1, ctx->mlp2,
                                (size_t)tokens * Q21_FFN),
           "qwen swiglu");
    q21_dump_bf16("mlp_act", ctx->mlp_act, (size_t)tokens * Q21_FFN);
    Q21_OP(qzen_gpu_convrot_rotate_bf16(ctx->gpu, ctx->rot, ctx->mlp_act, tokens,
                                      Q21_FFN),
           "qwen ffn out ConvRot rotation");
    /* The down projection consumes the 12288-wide activation, so it needs the
     * wide quantisation buffer; act8 only has room for 4096-wide inputs. */
    Q21_OP(qzen_gpu_linear_int8_bf16(ctx->gpu, ctx->projected, ctx->mlp8,
                                   ctx->mlp8_scales, ctx->rot, block->down,
                                   block->down_scale, tokens, Q21_FFN, dim, 0, 0),
           "qwen mlp out int8");
    q21_dump_bf16("mlp", ctx->projected, (size_t)tokens * dim);
    Q21_OP(qzen_gpu_gate_f32res_bf16(ctx->gpu, carrier, ctx->projected,
                                   ctx->mod_table, NULL, 0, ctx->row_map, tokens,
                                   dim, 5, 3),
           "qwen ffn gate");
    q21_dump_f32_tensor("hidden2", carrier, (size_t)tokens * dim);
#undef Q21_OP
    return 1;
}

/* Host-side conditioning (exact CPU path) staged into the device tables. */
static int q21_gpu_stage_conditioning(q21_gpu *ctx, qzen_qwen21_model *model,
                                      float sigma, char *error, size_t error_size) {
    const size_t dim = Q21_HIDDEN;
    uint16_t *mod = malloc(2 * 5 * dim * sizeof(uint16_t));
    uint16_t *final = malloc(2 * dim * sizeof(uint16_t));
    if (!mod || !final) {
        free(mod);
        free(final);
        snprintf(error, error_size, "out of memory for the modulation tables");
        return 0;
    }
    q21_conditioning condition;
    q21_condition(model, sigma, &condition, model->threads);
    for (int region = 0; region < 2; region++) {
        const float *source = condition.module + region * 4 * Q21_HIDDEN;
        uint16_t *dst = mod + (size_t)region * 5 * dim;
        for (int i = 0; i < Q21_HIDDEN; i++) {
            dst[0 * dim + i] = q21_f32_to_bf16(source[0 * Q21_HIDDEN + i]);
            dst[1 * dim + i] = q21_f32_to_bf16(tanhf(source[1 * Q21_HIDDEN + i]));
            dst[2 * dim + i] = q21_f32_to_bf16(source[2 * Q21_HIDDEN + i]);
            dst[3 * dim + i] = q21_f32_to_bf16(tanhf(source[3 * Q21_HIDDEN + i]));
            dst[4 * dim + i] = 0; /* zero shift slot */
        }
    }
    for (int i = 0; i < Q21_HIDDEN; i++) {
        final[i] = 0; /* zero shift */
        final[dim + i] = q21_f32_to_bf16(condition.out_scale[i]);
    }
    int ok = qzen_gpu_tensor_write_bf16(ctx->mod_table, mod, 2 * 5 * dim) &&
             qzen_gpu_tensor_write_bf16(ctx->final_table, final, 2 * dim);
    free(mod);
    free(final);
    if (!ok) {
        snprintf(error, error_size, "modulation table upload failed");
        return 0;
    }
    return 1;
}

static struct q21_gpu_ctx *q21_gpu_open(qzen_qwen21_model *model, char *error,
                                        size_t error_size) {
    q21_gpu *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        snprintf(error, error_size, "cannot allocate the Qwen GPU context");
        return NULL;
    }
    ctx->gpu = qzen_gpu_create(NULL, error, error_size);
    if (!ctx->gpu) {
        free(ctx);
        return NULL;
    }
    for (int layer = 0; layer < Q21_LAYERS; layer++) {
        if (!q21_gpu_load_block(ctx, &model->header, layer, &ctx->blocks[layer],
                                error, error_size)) {
            q21_gpu_free(ctx);
            return NULL;
        }
    }
    const uint64_t image[2] = {Q21_HIDDEN, Q21_IN_CHANNELS};
    const uint64_t out[2] = {Q21_OUT_CHANNELS, Q21_HIDDEN};
    ctx->img_in = q21_gpu_load_bf16(&model->header, ctx->gpu, "img_in.weight", 2,
                                    image, error, error_size);
    ctx->proj_out = q21_gpu_load_bf16(&model->header, ctx->gpu, "proj_out.weight",
                                      2, out, error, error_size);
    if (ctx->img_in && ctx->proj_out) {
        uint16_t *ones = malloc(Q21_HIDDEN * sizeof(uint16_t));
        if (ones) {
            for (int i = 0; i < Q21_HIDDEN; i++) ones[i] = q21_f32_to_bf16(1.0f);
            qzen_gpu_tensor *uploaded =
                qzen_gpu_tensor_from_bf16(ctx->gpu, ones, Q21_HIDDEN);
            free(ones);
            ctx->ones = uploaded;
        }
    }
    if (!ctx->img_in || !ctx->proj_out || !ctx->ones) {
        if (!error[0]) snprintf(error, error_size, "resident weight upload failed");
        q21_gpu_free(ctx);
        return NULL;
    }
    return ctx;
}

/* --------------------------------------------------------------- LoRA ---- */

/* PEFT/Diffusers adapters for this checkpoint name the projections
 * "transformer.transformer_blocks.N.attn.<module>.lora.{down,up}.weight" with
 * down [rank, 4096] and up [4096, rank]. The official DMD2 two-step adapter
 * (qwen_image_2.1_DMD2-2steps.safetensors) touches to_q/to_k/to_v/to_out.0 on
 * every block at rank 32 with alpha 32, i.e. scale 1.0.
 *
 * The resident weights are INT8 ConvRot (W_rot = W @ H per 256-column group),
 * so the delta has to be rotated into that basis before it is added: merging
 * it in the plain basis mixes two coordinate systems and is silently wrong
 * (the same trap qzen_lora.c documents for the MiniMax checkpoint). The merge is
 * therefore: delta = up @ down * scale on the device, rotate, then requantise
 * each touched row against a fresh scale with qzen_gpu_lora_merge_i8. */

/* One number from the header text, by key name.
 *
 * The keys are not top-level: PEFT writes them into __metadata__ as a nested
 * JSON *string* under "lora_adapter_metadata", so the header bytes read
 *   "lora_adapter_metadata": "{\n ... \"transformer.lora_alpha\": 32, ... }"
 * i.e. the quote before the key is escaped. Searching for the quoted spelling
 * ("transformer.lora_alpha":) therefore matches nothing at all, which is why
 * this used to fall back to 1.0 for the DMD2 file -- correct by coincidence,
 * because alpha == r == 32 there, and silently wrong for any adapter where
 * they differ. Match the bare key instead and take the first number after the
 * colon, whatever the quoting around it.
 *
 * The character right after the key has to be a quote (top level) or a
 * backslash (escaped, nested), which is what keeps "transformer.r" from
 * matching "transformer.rank_pattern". */
static int q21_lora_number(const char *text, const char *key, double *out) {
    const char *at = text;
    while ((at = strstr(at, key)) != NULL) {
        const char *after = at + strlen(key);
        if (*after == '"' || *after == '\\') {
            const char *colon = strchr(after, ':');
            if (colon) {
                char *end = NULL;
                double value = strtod(colon + 1, &end);
                if (end != colon + 1 && value > 0.0) {
                    *out = value;
                    return 1;
                }
            }
        }
        at += strlen(key);
    }
    return 0;
}

/* alpha / r from the header metadata, 1.0 when the file carries neither (the
 * H3 adapters ship no alpha at all and are defined as scale 1). Non-uniform
 * alpha_pattern/rank_pattern tables are not honoured: they are empty in every
 * adapter this build has seen, and guessing a per-module ratio from a flat
 * file would be worse than saying so. */
double qzen_qwen21_lora_ratio(const char *path) {
    if (!path || !*path) return 1.0;
    FILE *file = fopen(path, "rb");
    if (!file) return 1.0;
    uint64_t header_bytes = 0;
    double ratio = 1.0;
    char *text = NULL;
    if (fread(&header_bytes, sizeof(header_bytes), 1, file) == 1 &&
        header_bytes > 0 && header_bytes < (16u << 20)) {
        text = malloc((size_t)header_bytes + 1);
        if (text && fread(text, 1, (size_t)header_bytes, file) ==
                        (size_t)header_bytes) {
            text[header_bytes] = '\0';
            double alpha = 0.0, rank = 0.0;
            if (q21_lora_number(text, "transformer.lora_alpha", &alpha) &&
                q21_lora_number(text, "transformer.r", &rank))
                ratio = alpha / rank;
        }
    }
    free(text);
    fclose(file);
    return ratio;
}

static const qzen_st_tensor *q21_lora_find(const qzen_st_header *header,
                                         const char *name, uint64_t rows,
                                         uint64_t columns, char *error,
                                         size_t error_size) {
    const qzen_st_tensor *tensor = qzen_st_find(header, name);
    if (!tensor) {
        snprintf(error, error_size, "the adapter has no %s", name);
        return NULL;
    }
    if (tensor->dtype != QZEN_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns) {
        snprintf(error, error_size,
                 "%s is %s [%llu, %llu], expected BF16 [%llu, %llu]", name,
                 qzen_dtype_name(tensor->dtype),
                 (unsigned long long)(tensor->ndim > 0 ? tensor->shape[0] : 0),
                 (unsigned long long)(tensor->ndim > 1 ? tensor->shape[1] : 0),
                 (unsigned long long)rows, (unsigned long long)columns);
        return NULL;
    }
    return tensor;
}

/* One projection: delta = up @ down * strength, rotated, merged in place. */
static int q21_lora_merge_module(q21_gpu *ctx, const qzen_st_header *header,
                                 const char *down_name, const char *up_name,
                                 qzen_gpu_tensor *weight, qzen_gpu_tensor *scales,
                                 float strength, char *error,
                                 size_t error_size) {
    const uint32_t columns = Q21_HIDDEN;
    const qzen_st_tensor *down = q21_lora_find(header, down_name, 32, columns,
                                             error, error_size);
    if (!down) return 0;
    const uint32_t rank = (uint32_t)down->shape[0];
    const qzen_st_tensor *up = q21_lora_find(header, up_name, columns, rank,
                                           error, error_size);
    if (!up) return 0;
    uint16_t *down_rows = malloc((size_t)rank * columns * sizeof(uint16_t));
    uint16_t *up_rows = malloc((size_t)columns * rank * sizeof(uint16_t));
    uint16_t *down_t = malloc((size_t)rank * columns * sizeof(uint16_t));
    if (!down_rows || !up_rows || !down_t) {
        free(down_rows);
        free(up_rows);
        free(down_t);
        snprintf(error, error_size, "out of memory for the adapter matrices");
        return 0;
    }
    int ok = qzen_st_read_data(header, down, down_rows,
                             (size_t)rank * columns * sizeof(uint16_t), error,
                             error_size) &&
             qzen_st_read_data(header, up, up_rows,
                             (size_t)columns * rank * sizeof(uint16_t), error,
                             error_size);
    if (ok) {
        /* The strength folds into the up matrix on the host, so the device
         * sees a plain GEMM: up * strength then down. */
        if (strength != 1.0f) {
            for (size_t i = 0; i < (size_t)columns * rank; i++)
                up_rows[i] = q21_f32_to_bf16(q21_bf16_to_f32(up_rows[i]) *
                                             strength);
        }
        /* qzen_gpu_linear_bf16 takes the weight as [out][in], so down (rank
         * major) is transposed once and shared by every chunk. */
        for (uint32_t row = 0; row < rank; row++)
            for (uint32_t column = 0; column < columns; column++)
                down_t[(size_t)column * rank + row] =
                    down_rows[(size_t)row * columns + column];
        qzen_gpu_tensor *weight_down = qzen_gpu_tensor_from_bf16(
            ctx->gpu, down_t, (size_t)columns * rank);
        if (!weight_down) {
            snprintf(error, error_size, "cannot upload the adapter: %s",
                     qzen_gpu_error(ctx->gpu));
            ok = 0;
        } else {
            const uint32_t chunk = 1024;
            for (uint32_t start = 0; start < columns && ok; start += chunk) {
                const uint32_t count =
                    columns - start < chunk ? columns - start : chunk;
                qzen_gpu_tensor *act = qzen_gpu_tensor_from_bf16(
                    ctx->gpu, up_rows + (size_t)start * rank,
                    (size_t)count * rank);
                qzen_gpu_tensor *delta =
                    qzen_gpu_tensor_new_bf16(ctx->gpu, (size_t)count * columns);
                if (!act || !delta) {
                    snprintf(error, error_size,
                             "out of GPU memory for the adapter delta: %s",
                             qzen_gpu_error(ctx->gpu));
                    qzen_gpu_tensor_free(act);
                    qzen_gpu_tensor_free(delta);
                    ok = 0;
                    break;
                }
                ok = qzen_gpu_linear_bf16(ctx->gpu, delta, act, weight_down, NULL,
                                        count, rank, columns);
                if (ok)
                    ok = qzen_gpu_convrot_rotate_bf16(ctx->gpu, delta, delta,
                                                    count, columns);
                if (ok)
                    ok = qzen_gpu_lora_merge_i8(ctx->gpu, weight, scales, delta,
                                              start, 0, count, columns);
                if (!ok)
                    snprintf(error, error_size, "adapter merge failed: %s",
                             qzen_gpu_error(ctx->gpu));
                qzen_gpu_tensor_free(act);
                qzen_gpu_tensor_free(delta);
            }
            qzen_gpu_tensor_free(weight_down);
        }
    }
    free(down_rows);
    free(up_rows);
    free(down_t);
    return ok;
}

int qzen_qwen21_apply_lora(qzen_qwen21_model *model, const char *path, float scale,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    q21_gpu *ctx = model ? model->gpu_ctx : NULL;
    if (!ctx) {
        snprintf(error, error_size,
                 "the qwen21 adapter merge needs the device DiT "
                 "(QWEN21_GPU=1)");
        return 0;
    }
    if (!path || !*path) {
        snprintf(error, error_size, "no adapter path");
        return 0;
    }
    if (!(scale > 0.0f)) scale = 1.0f;
    const double ratio = qzen_qwen21_lora_ratio(path);
    const float strength = (float)(ratio * (double)scale);
    qzen_st_header header;
    if (!qzen_st_read_header(path, &header, error, error_size)) return 0;
    static const char *const modules[4] = {"to_q", "to_k", "to_v", "to_out.0"};
    int ok = 1, merged = 0;
    const double started = q21_now();
    for (int layer = 0; layer < Q21_LAYERS && ok; layer++) {
        qzen_gpu_tensor *targets[4] = {ctx->blocks[layer].q, ctx->blocks[layer].k,
                                     ctx->blocks[layer].v,
                                     ctx->blocks[layer].out};
        qzen_gpu_tensor *scales[4] = {
            ctx->blocks[layer].q_scale, ctx->blocks[layer].k_scale,
            ctx->blocks[layer].v_scale, ctx->blocks[layer].out_scale};
        for (int module = 0; module < 4 && ok; module++) {
            char down_name[256], up_name[256];
            snprintf(down_name, sizeof(down_name),
                     "transformer.transformer_blocks.%d.attn.%s.lora.down.weight",
                     layer, modules[module]);
            snprintf(up_name, sizeof(up_name),
                     "transformer.transformer_blocks.%d.attn.%s.lora.up.weight",
                     layer, modules[module]);
            /* An adapter that only touches some projections is legal: a
             * missing pair is skipped, a malformed one is an error. */
            if (!qzen_st_find(&header, down_name) && !qzen_st_find(&header, up_name))
                continue;
            if (!q21_lora_merge_module(ctx, &header, down_name, up_name,
                                       targets[module], scales[module],
                                       strength, error, error_size)) {
                ok = 0;
                break;
            }
            merged++;
        }
    }
    /* Completeness gate: every down/up pair in the file has to have been
     * merged. A pair under a module this build does not know (to_add_out,
     * mlp.*, a renamed block index) would otherwise be skipped in silence and
     * the adapter would run half-applied -- which still produces a picture,
     * so nothing downstream would catch it. */
    int pairs = 0;
    for (size_t index = 0; index < header.tensor_count; index++) {
        const char *name = header.tensors[index].name;
        if (name && strstr(name, ".lora.down.weight")) pairs++;
    }
    qzen_st_free_header(&header);
    if (ok && pairs == 0) {
        /* A file with no attn pair at all (a different model's adapter, or a
         * plain checkpoint handed to --lora) would otherwise merge nothing
         * and still report success. */
        snprintf(error, error_size,
                 "the adapter has no attn.<module>.lora.down/up.weight tensors; "
                 "it is not a Qwen-Image 2.1 adapter");
        ok = 0;
    }
    if (ok && merged != pairs) {
        snprintf(error, error_size,
                 "the adapter has %d projections this build does not merge "
                 "(%d merged, only attn.to_q/to_k/to_v/to_out.0 are supported)",
                 pairs - merged, merged);
        ok = 0;
    }
    if (ok) {
        fprintf(stderr,
                "qwen21: adapter %s merged into %d/%d projections at scale "
                "%.3f (alpha/rank %.3f x request %.3f) in %.2fs\n",
                path, merged, pairs, strength, ratio, (double)scale,
                q21_now() - started);
        fflush(stderr);
    }
    return ok;
}

int qzen_qwen21_device_ready(const qzen_qwen21_model *model) {
    return model && model->gpu_ctx != NULL;
}

qzen_gpu *qzen_qwen21_device(const qzen_qwen21_model *model) {
    return model && model->gpu_ctx ? model->gpu_ctx->gpu : NULL;
}

/* Device velocity slot (0/1, alternating CFG buffers), grown on demand. The
 * tensor stays valid until the slot is grown again or the model closes. */
qzen_gpu_tensor *qzen_qwen21_device_velocity(qzen_qwen21_model *model, int slot,
                                         int image_tokens) {
    q21_gpu *ctx = model ? model->gpu_ctx : NULL;
    if (!ctx || slot < 0 || slot > 1 || image_tokens <= 0) return NULL;
    const size_t count = (size_t)image_tokens * Q21_OUT_CHANNELS;
    if (!ctx->velocity[slot] ||
        qzen_gpu_tensor_elements(ctx->velocity[slot]) < count) {
        q21_gpu_tensor_free(&ctx->velocity[slot]);
        ctx->velocity[slot] = qzen_gpu_tensor_new_bf16(ctx->gpu, count);
    }
    return ctx->velocity[slot];
}

/* Full device forward, one sigma. latent_device is [64][height][width] F32
 * channel major on the model's GPU and stays owned by the caller (the host
 * entry point uploads it once and loops sigmas over it); velocity_out
 * receives [image_tokens][64] BF16 (caller-owned, at least that many
 * elements). block0_out, when non-NULL, receives the joint hidden states
 * after block 0 (F32, host). */
static int q21_forward_device(qzen_qwen21_model *model, const float *text,
                              int text_tokens,
                              const qzen_gpu_tensor *latent_device, int height,
                              int width, float sigma,
                              qzen_gpu_tensor *velocity_out, float *block0_out,
                              char *error, size_t error_size) {
    q21_gpu *ctx = model->gpu_ctx;
    const int image_tokens = height * width;
    const int tokens = text_tokens + image_tokens;
    if (!q21_gpu_scratch_prepare(ctx, tokens, text_tokens, height, width, error,
                                 error_size)) {
        return 0;
    }
    if (!q21_gpu_stage_conditioning(ctx, model, sigma, error, error_size)) {
        return 0;
    }
    /* Text prefix: exact host path, uploaded into the carrier. */
    float *prefix = q21_text_prefix(model, text, text_tokens, error, error_size);
    if (!prefix) return 0;
    /* The latent arrives as channel-major F32 [64][height][width]; transpose
     * it into token-major rows, project with img_in, widen to F32 into the
     * carrier tail. */
    qzen_gpu_tensor *latent_tokens = qzen_gpu_tensor_new_f32(
        ctx->gpu, (size_t)image_tokens * Q21_IN_CHANNELS);
    if (!latent_tokens) {
        free(prefix);
        snprintf(error, error_size, "out of GPU memory for the latent staging");
        return 0;
    }
    int ok = 0;
    do {
        if (!qzen_gpu_transpose_f32(ctx->gpu, latent_tokens, latent_device,
                                  Q21_IN_CHANNELS, image_tokens)) {
            snprintf(error, error_size, "latent transpose failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        if (!qzen_gpu_cast_f32_to_bf16(ctx->gpu, ctx->image_bf16, latent_tokens,
                                     (size_t)image_tokens * Q21_IN_CHANNELS)) {
            snprintf(error, error_size, "latent cast failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        /* img_in widens 64 -> 4096 channels, so the output cannot alias the
         * input: row r writes past the input rows the GEMM has not read yet.
         * Stage the projection in normed (scratch, rewritten by block 0). */
        if (!qzen_gpu_linear_bf16(ctx->gpu, ctx->normed, ctx->image_bf16,
                                ctx->img_in, NULL, image_tokens, Q21_IN_CHANNELS,
                                Q21_HIDDEN)) {
            snprintf(error, error_size, "img_in projection failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        if (!qzen_gpu_cast_bf16_to_f32(ctx->gpu, ctx->image_f32, ctx->normed,
                                      (uint32_t)image_tokens * Q21_HIDDEN)) {
            snprintf(error, error_size, "img_in widen failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        if (!qzen_gpu_tensor_write_f32_range(ctx->carrier, 0, prefix,
                                           (size_t)text_tokens * Q21_HIDDEN)) {
            snprintf(error, error_size, "prefix upload failed");
            break;
        }
        if (!qzen_gpu_copy_d2d_f32(ctx->gpu, ctx->carrier,
                                 (size_t)text_tokens * Q21_HIDDEN, ctx->image_f32,
                                 0, (size_t)image_tokens * Q21_HIDDEN)) {
            snprintf(error, error_size, "carrier tail copy failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        int layer_failed = 0;
        for (int layer = 0; layer < Q21_LAYERS; layer++) {
            if (!q21_gpu_block_run(ctx, &ctx->blocks[layer], ctx->carrier, tokens,
                                   text_tokens, error, error_size)) {
                layer_failed = 1;
                break;
            }
            if (getenv("Q21_DUMP_DIR")) {
                char stage[32];
                snprintf(stage, sizeof(stage), "block%02d", layer);
                q21_dump_f32_tensor(stage, ctx->carrier, (size_t)tokens * Q21_HIDDEN);
            }
            if (layer == 0 && block0_out &&
                !qzen_gpu_tensor_read_f32(ctx->carrier, block0_out,
                                        (size_t)tokens * Q21_HIDDEN)) {
                snprintf(error, error_size, "block0 download failed");
                layer_failed = 1;
                break;
            }
        }
        if (layer_failed) break;
        /* final norm over the image rows, then proj_out */
        if (!qzen_gpu_copy_d2d_f32(ctx->gpu, ctx->image_f32, 0, ctx->carrier,
                                 (size_t)text_tokens * Q21_HIDDEN,
                                 (size_t)image_tokens * Q21_HIDDEN)) {
            snprintf(error, error_size, "final tail copy failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        if (!qzen_gpu_layer_norm_adaln_f32in_bf16(
                ctx->gpu, ctx->normed, ctx->image_f32, ctx->ones,
                ctx->final_table, NULL, ctx->zero_map, image_tokens, Q21_HIDDEN,
                2, 0, 1, Q21_EPS)) {
            snprintf(error, error_size, "final LN+scale failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        if (!qzen_gpu_linear_bf16(ctx->gpu, velocity_out, ctx->normed, ctx->proj_out,
                                NULL, image_tokens, Q21_HIDDEN,
                                Q21_OUT_CHANNELS)) {
            snprintf(error, error_size, "proj_out failed: %s",
                     qzen_gpu_error(ctx->gpu));
            break;
        }
        ok = 1;
    } while (0);
    qzen_gpu_tensor_free(latent_tokens);
    free(prefix);
    return ok;
}

int qzen_qwen21_forward_device(qzen_qwen21_model *model, const float *text,
                             int text_tokens, const qzen_gpu_tensor *latent_device,
                             int height, int width, float sigma,
                             qzen_gpu_tensor *velocity_out, float *block0_out,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model || !model->gpu_ctx || !text || !latent_device || !velocity_out ||
        text_tokens <= 0 || height <= 0 || width <= 0) {
        snprintf(error, error_size, "invalid qwen device forward request");
        return 0;
    }
    return q21_forward_device(model, text, text_tokens, latent_device, height,
                              width, sigma, velocity_out, block0_out, error,
                              error_size);
}

/* Single block on the device with host F32 in/out (gate localisation). */
int qzen_qwen21_block_forward_device(qzen_qwen21_model *model, int layer,
                                   const float *hidden, int text_tokens,
                                   int height, int width, float sigma, float *out,
                                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    q21_gpu *ctx = model ? model->gpu_ctx : NULL;
    if (!ctx || !hidden || !out || layer < 0 || layer >= Q21_LAYERS ||
        text_tokens <= 0 || height <= 0 || width <= 0) {
        snprintf(error, error_size, "invalid qwen device block request");
        return 0;
    }
    const int tokens = text_tokens + height * width;
    const size_t dim = (size_t)tokens * Q21_HIDDEN;
    if (!q21_gpu_scratch_prepare(ctx, tokens, text_tokens, height, width, error,
                                 error_size)) {
        return 0;
    }
    if (!q21_gpu_stage_conditioning(ctx, model, sigma, error, error_size)) {
        return 0;
    }
    if (!qzen_gpu_tensor_write_f32(ctx->carrier, hidden, dim)) {
        snprintf(error, error_size, "block input upload failed");
        return 0;
    }
    if (!q21_gpu_block_run(ctx, &ctx->blocks[layer], ctx->carrier, tokens,
                           text_tokens, error, error_size)) {
        return 0;
    }
    if (!qzen_gpu_tensor_read_f32(ctx->carrier, out, dim)) {
        snprintf(error, error_size, "block output download failed");
        return 0;
    }
    return 1;
}
