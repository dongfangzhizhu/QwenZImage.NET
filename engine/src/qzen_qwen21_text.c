/* Qwen3-VL-8B language tower: f32 CPU reference forward and the t2i template.
 *
 * Mirrors models/qwen21/text_layers.py (Qwen3VLTextDecoderLayer):
 *   x = x + attn(rms(x, input_layernorm))      with q/k normed over head_dim
 *   x = x + down(silu(gate(rms(x, post_layernorm))) * up(...))
 * 32 query heads over 8 KV heads, causal, scale 1/sqrt(128), rotate_half RoPE at
 * theta 5e6 (text-only mRoPE: the three position axes all equal the token
 * index). Embeddings come from the BF16 table row by row, block linears from the
 * same int8_convrot reader the DiT uses.
 *
 * The DiT consumes the last layer output BEFORE model.norm, and the pipeline
 * drops the first 14 tokens (the system turn) of the chat template.
 */

#include "qzen_qwen21_text.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "qzen_qwen21_ops.h"

/* The device forward (QWEN21_TEXT_GPU=1) lives at the bottom of this file; the
 * f32 CPU path above it stays the numerical gate and the fallback. */
struct q21_text_gpu;
static int q21_text_gpu_enabled(void);
static struct q21_text_gpu *q21_text_gpu_open(char *error, size_t error_size);
static void q21_text_gpu_free(struct q21_text_gpu *ctx);
static int q21_text_gpu_forward(qzen_qwen21_text *model, const uint32_t *ids,
                                size_t count, float *hidden_out, float *trace,
                                char *error, size_t error_size);

#include "qzen_gpu.h"

#define Q21_TEXT_HEADS 32
#define Q21_TEXT_KV_HEADS 8
#define Q21_TEXT_KV_DIM (Q21_TEXT_KV_HEADS * Q21_HEAD_DIM)
#define Q21_TEXT_GROUPS 64 /* RoPE pairs over head_dim 128 */
#define Q21_TEXT_THETA 5000000.0
#define Q21_TEXT_GROUP_SIZE 4 /* query heads per KV head */
#define Q21_TEXT_VOCAB 151936

static const char q21_text_system[] =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
static const char q21_text_user[] = "<|im_start|>user\n";
static const char q21_text_tail[] = "<|im_end|>\n<|im_start|>assistant\n";

typedef struct {
    float input_norm[Q21_HIDDEN];
    float post_norm[Q21_HIDDEN];
    float q_norm[Q21_HEAD_DIM];
    float k_norm[Q21_HEAD_DIM];
} q21_text_norms;

typedef struct {
    float *q_proj;   /* [4096][4096] */
    float *k_proj;   /* [1024][4096] */
    float *v_proj;   /* [1024][4096] */
    float *o_proj;   /* [4096][4096] */
    float *gate;     /* [12288][4096] */
    float *up;       /* [12288][4096] */
    float *down;     /* [4096][12288] */
} q21_text_block;

struct qzen_qwen21_text {
    qzen_st_header header;
    int threads;
    const qzen_st_tensor *embed; /* BF16 [vocab][4096], read row by row */
    q21_text_norms *norms;     /* [QZEN_QWEN21_TEXT_LAYERS] */
    double weight_seconds;
    struct q21_text_gpu *gpu;  /* QWEN21_TEXT_GPU=1: INT8 weights stay INT8 */
    int gpu_tried;             /* the open was attempted (and may have failed) */
};

static void q21_text_block_free(q21_text_block *block) {
    free(block->q_proj);
    free(block->k_proj);
    free(block->v_proj);
    free(block->o_proj);
    free(block->gate);
    free(block->up);
    free(block->down);
    memset(block, 0, sizeof(*block));
}

static int q21_text_block_load(qzen_qwen21_text *model, int layer, q21_text_block *block,
                               char *error, size_t error_size) {
    const uint64_t query[2] = {Q21_HIDDEN, Q21_HIDDEN};
    const uint64_t kv[2] = {Q21_TEXT_KV_HEADS * Q21_HEAD_DIM, Q21_HIDDEN};
    const uint64_t wide[2] = {Q21_FFN, Q21_HIDDEN};
    const uint64_t down[2] = {Q21_HIDDEN, Q21_FFN};
    const char *suffixes[7] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                               "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                               "mlp.gate_proj.weight",     "mlp.up_proj.weight",
                               "mlp.down_proj.weight"};
    const uint64_t *shapes[7] = {query, kv, kv, query, wide, wide, down};
    char name[160];
    memset(block, 0, sizeof(*block));
    double started = q21_now();
    for (int index = 0; index < 7; index++) {
        snprintf(name, sizeof(name), "language_model.layers.%d.%s", layer, suffixes[index]);
        float *values = q21_read_f32(&model->header, name, 2, shapes[index], error, error_size);
        if (!values) {
            q21_text_block_free(block);
            return 0;
        }
        switch (index) {
            case 0: block->q_proj = values; break;
            case 1: block->k_proj = values; break;
            case 2: block->v_proj = values; break;
            case 3: block->o_proj = values; break;
            case 4: block->gate = values; break;
            case 5: block->up = values; break;
            default: block->down = values; break;
        }
    }
    model->weight_seconds += q21_now() - started;
    return 1;
}

static int q21_text_norms_load(qzen_qwen21_text *model, char *error, size_t error_size) {
    const uint64_t hidden[1] = {Q21_HIDDEN};
    const uint64_t head[1] = {Q21_HEAD_DIM};
    char name[160];
    model->norms = calloc(QZEN_QWEN21_TEXT_LAYERS, sizeof(*model->norms));
    if (!model->norms) {
        snprintf(error, error_size, "out of memory for the language tower norms");
        return 0;
    }
    for (int layer = 0; layer < QZEN_QWEN21_TEXT_LAYERS; layer++) {
        q21_text_norms *norms = &model->norms[layer];
        struct {
            float *slot;
            const char *suffix;
            int ndim;
            const uint64_t *shape;
        } reads[4] = {
            {norms->input_norm, "input_layernorm.weight", 1, hidden},
            {norms->post_norm, "post_attention_layernorm.weight", 1, hidden},
            {norms->q_norm, "self_attn.q_norm.weight", 1, head},
            {norms->k_norm, "self_attn.k_norm.weight", 1, head},
        };
        for (int index = 0; index < 4; index++) {
            snprintf(name, sizeof(name), "language_model.layers.%d.%s", layer,
                     reads[index].suffix);
            float *values = q21_read_f32(&model->header, name, reads[index].ndim,
                                         reads[index].shape, error, error_size);
            if (!values) return 0;
            memcpy(reads[index].slot, values,
                   (size_t)(reads[index].shape[0]) * sizeof(float));
            free(values);
        }
    }
    return 1;
}

/* ------------------------------------------------------------ chat template */

int qzen_qwen21_text_prompt(const char *prompt, char *out, size_t size, char *error,
                          size_t error_size) {
    const char *text = (prompt && prompt[0]) ? prompt : " ";
    size_t needed = sizeof(q21_text_system) + sizeof(q21_text_user) +
                    sizeof(q21_text_tail) + strlen(text);
    if (size < needed) {
        snprintf(error, error_size, "prompt template needs %zu bytes, got %zu", needed, size);
        return 0;
    }
    snprintf(out, size, "%s%s%s%s", q21_text_system, q21_text_user, text, q21_text_tail);
    return 1;
}

int qzen_qwen21_text_encode(const qzen_tokenizer *tokenizer, const char *prompt, uint32_t **ids,
                          size_t *count, char *error, size_t error_size) {
    if (!tokenizer || !ids || !count) {
        snprintf(error, error_size, "Qwen3-VL encode needs a tokenizer");
        return 0;
    }
    size_t capacity = strlen(prompt ? prompt : "") + 512;
    char *text = malloc(capacity);
    if (!text) {
        snprintf(error, error_size, "out of memory for the prompt template");
        return 0;
    }
    if (!qzen_qwen21_text_prompt(prompt, text, capacity, error, error_size)) {
        free(text);
        return 0;
    }
    uint32_t *encoded = NULL;
    size_t total = 0;
    if (!qzen_tokenizer_encode(tokenizer, text, 0, &encoded, &total, error, error_size)) {
        free(text);
        return 0;
    }
    free(text);
    if (total <= QZEN_QWEN21_TEXT_DROP) {
        snprintf(error, error_size, "templated prompt tokenised to %zu tokens, expected more "
                                    "than the %d token system prefix",
                 total, QZEN_QWEN21_TEXT_DROP);
        qzen_tokenizer_ids_free(encoded);
        return 0;
    }
    size_t kept = total - QZEN_QWEN21_TEXT_DROP;
    uint32_t *trimmed = malloc(kept * sizeof(uint32_t));
    if (!trimmed) {
        snprintf(error, error_size, "out of memory for %zu prompt tokens", kept);
        qzen_tokenizer_ids_free(encoded);
        return 0;
    }
    memcpy(trimmed, encoded + QZEN_QWEN21_TEXT_DROP, kept * sizeof(uint32_t));
    qzen_tokenizer_ids_free(encoded);
    *ids = trimmed;
    *count = kept;
    return 1;
}

/* ------------------------------------------------------------------ forward */

static void q21_text_rope_tables(int tokens, float *cosine, float *sine) {
    double inverse[Q21_TEXT_GROUPS];
    for (int pair = 0; pair < Q21_TEXT_GROUPS; pair++) {
        inverse[pair] = exp(-log(Q21_TEXT_THETA) * (double)(2 * pair) / (double)Q21_HEAD_DIM);
    }
    for (int token = 0; token < tokens; token++) {
        for (int pair = 0; pair < Q21_TEXT_GROUPS; pair++) {
            double angle = (double)token * inverse[pair];
            cosine[(size_t)token * Q21_TEXT_GROUPS + pair] = (float)cos(angle);
            sine[(size_t)token * Q21_TEXT_GROUPS + pair] = (float)sin(angle);
        }
    }
}

/* rotate_half rotation: the second half of the head feeds the first. The same
 * cos/sin row serves every head. */
static void q21_text_rope_apply(float *x, int tokens, int heads, const float *cosine,
                                const float *sine) {
    const int half = Q21_HEAD_DIM / 2;
    for (int token = 0; token < tokens; token++) {
        const float *cos_row = cosine + (size_t)token * Q21_TEXT_GROUPS;
        const float *sin_row = sine + (size_t)token * Q21_TEXT_GROUPS;
        for (int head = 0; head < heads; head++) {
            float *values = x + ((size_t)token * heads + head) * Q21_HEAD_DIM;
            for (int d = 0; d < half; d++) {
                float first = values[d];
                float second = values[d + half];
                float c = cos_row[d];
                float s = sin_row[d];
                values[d] = first * c - second * s;
                values[d + half] = second * c + first * s;
            }
        }
    }
}

/* Weighted RMSNorm over head_dim (q_norm / k_norm). */
static void q21_text_head_norm(float *x, int tokens, int heads, const float *weight) {
    for (int token = 0; token < tokens; token++) {
        for (int head = 0; head < heads; head++) {
            float *values = x + ((size_t)token * heads + head) * Q21_HEAD_DIM;
            double sum = 0.0;
            for (int d = 0; d < Q21_HEAD_DIM; d++) sum += (double)values[d] * (double)values[d];
            float inverse = 1.0f / sqrtf((float)(sum / (double)Q21_HEAD_DIM) + Q21_EPS);
            for (int d = 0; d < Q21_HEAD_DIM; d++) values[d] = values[d] * inverse * weight[d];
        }
    }
}

typedef struct {
    const float *q, *k, *v;
    float *out;
    float *scores; /* [threads][HEADS][tokens] */
    int tokens;
    float scale;
} q21_text_attention_job;

static void q21_text_attention_rows(void *context, int slot, int start, int end) {
    q21_text_attention_job *job = context;
    float *scratch = job->scores + (size_t)slot * Q21_TEXT_HEADS * (size_t)job->tokens;
    for (int token = start; token < end; token++) {
        int keys = token + 1;
        for (int head = 0; head < Q21_TEXT_HEADS; head++) {
            int source = head / Q21_TEXT_GROUP_SIZE;
            const float *query =
                job->q + ((size_t)token * Q21_TEXT_HEADS + head) * Q21_HEAD_DIM;
            float *scores = scratch + (size_t)head * (size_t)job->tokens;
            float best = -INFINITY;
            for (int j = 0; j < keys; j++) {
                const float *key = job->k + ((size_t)j * Q21_TEXT_KV_HEADS + source) * Q21_HEAD_DIM;
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
            float *dst = job->out + ((size_t)token * Q21_TEXT_HEADS + head) * Q21_HEAD_DIM;
            for (int d = 0; d < Q21_HEAD_DIM; d++) dst[d] = 0.0f;
            for (int j = 0; j < keys; j++) {
                float weight = scores[j] * inverse;
                const float *value =
                    job->v + ((size_t)j * Q21_TEXT_KV_HEADS + source) * Q21_HEAD_DIM;
                for (int d = 0; d < Q21_HEAD_DIM; d++) dst[d] += weight * value[d];
            }
        }
    }
}

static void q21_text_block_run(const q21_text_block *block, const q21_text_norms *norms,
                               float *hidden, int tokens, float *normed, float *q, float *k,
                               float *v, float *attended, float *gate_out, float *up_out,
                               float *projected, float *scores, const float *cosine,
                               const float *sine, int threads) {
    /* attention half */
    q21_rms_norm(hidden, tokens, Q21_HIDDEN, norms->input_norm, normed);
    q21_linear(normed, tokens, block->q_proj, Q21_HIDDEN, Q21_HIDDEN, q, threads);
    q21_linear(normed, tokens, block->k_proj, Q21_TEXT_KV_HEADS * Q21_HEAD_DIM, Q21_HIDDEN, k,
               threads);
    q21_linear(normed, tokens, block->v_proj, Q21_TEXT_KV_HEADS * Q21_HEAD_DIM, Q21_HIDDEN, v,
               threads);
    q21_text_head_norm(q, tokens, Q21_TEXT_HEADS, norms->q_norm);
    q21_text_head_norm(k, tokens, Q21_TEXT_KV_HEADS, norms->k_norm);
    q21_text_rope_apply(q, tokens, Q21_TEXT_HEADS, cosine, sine);
    q21_text_rope_apply(k, tokens, Q21_TEXT_KV_HEADS, cosine, sine);
    q21_text_attention_job attention = {q,     k,     v,     attended, scores, tokens,
                                        1.0f / sqrtf((float)Q21_HEAD_DIM)};
    q21_parallel(tokens, threads, q21_text_attention_rows, &attention);
    q21_linear(attended, tokens, block->o_proj, Q21_HIDDEN, Q21_HIDDEN, projected, threads);
    for (size_t index = 0; index < (size_t)tokens * Q21_HIDDEN; index++) {
        hidden[index] += projected[index];
    }

    /* MLP half */
    q21_rms_norm(hidden, tokens, Q21_HIDDEN, norms->post_norm, normed);
    q21_linear(normed, tokens, block->gate, Q21_FFN, Q21_HIDDEN, gate_out, threads);
    q21_linear(normed, tokens, block->up, Q21_FFN, Q21_HIDDEN, up_out, threads);
    q21_swiglu(gate_out, up_out, (size_t)tokens * Q21_FFN);
    q21_linear(gate_out, tokens, block->down, Q21_HIDDEN, Q21_FFN, projected, threads);
    for (size_t index = 0; index < (size_t)tokens * Q21_HIDDEN; index++) {
        hidden[index] += projected[index];
    }
}

int qzen_qwen21_text_forward(qzen_qwen21_text *model, const uint32_t *ids, size_t count,
                           float *hidden_out, float *trace, char *error,
                           size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model || !ids || !count || !hidden_out) {
        snprintf(error, error_size, "invalid Qwen3-VL text forward request");
        return 0;
    }
    /* QWEN21_TEXT_GPU=1 moves the whole stack to the device. A context that
     * cannot be created (no GPU, driver error) is not fatal: the f32 CPU path
     * is slow but exact, so the request still succeeds. A failure *inside* the
     * device forward is returned instead of being silently rerun on the CPU,
     * because 78 s of unexpected wall time is harder to diagnose than an
     * error string. */
    if (!model->gpu_tried) {
        model->gpu_tried = 1;
        if (q21_text_gpu_enabled()) {
            model->gpu = q21_text_gpu_open(error, error_size);
            if (!model->gpu) {
                fprintf(stderr,
                        "QwenZImageNative qwen21 text: device path unavailable (%s); "
                        "falling back to the f32 CPU reference\n",
                        error);
                if (error && error_size) error[0] = '\0';
            }
        }
    }
    if (model->gpu) {
        return q21_text_gpu_forward(model, ids, count, hidden_out, trace, error,
                                    error_size);
    }
    const int tokens = (int)count;
    const int threads = model->threads;
    const size_t hidden_values = (size_t)tokens * Q21_HIDDEN;
    float *hidden = malloc(hidden_values * sizeof(float));
    float *normed = malloc(hidden_values * sizeof(float));
    float *q = malloc(hidden_values * sizeof(float));
    float *k = malloc((size_t)tokens * Q21_TEXT_KV_HEADS * Q21_HEAD_DIM * sizeof(float));
    float *v = malloc((size_t)tokens * Q21_TEXT_KV_HEADS * Q21_HEAD_DIM * sizeof(float));
    float *attended = malloc(hidden_values * sizeof(float));
    float *projected = malloc(hidden_values * sizeof(float));
    float *gate_out = malloc((size_t)tokens * Q21_FFN * sizeof(float));
    float *up_out = malloc((size_t)tokens * Q21_FFN * sizeof(float));
    float *scores = malloc((size_t)threads * Q21_TEXT_HEADS * (size_t)tokens * sizeof(float));
    float *cosine = malloc((size_t)tokens * Q21_TEXT_GROUPS * sizeof(float));
    float *sine = malloc((size_t)tokens * Q21_TEXT_GROUPS * sizeof(float));
    int ok = 0;
    if (!hidden || !normed || !q || !k || !v || !attended || !projected || !gate_out ||
        !up_out || !scores || !cosine || !sine) {
        snprintf(error, error_size, "out of memory for %d text tokens", tokens);
        goto done;
    }
    for (int token = 0; token < tokens; token++) {
        if (ids[token] >= Q21_TEXT_VOCAB) {
            snprintf(error, error_size, "token id %u is outside the %d row vocabulary",
                     ids[token], Q21_TEXT_VOCAB);
            goto done;
        }
        if (!q21_read_rows_f32(&model->header, model->embed, ids[token], 1, Q21_HIDDEN,
                               hidden + (size_t)token * Q21_HIDDEN, error, error_size)) {
            goto done;
        }
    }
    q21_text_rope_tables(tokens, cosine, sine);
    for (int layer = 0; layer < QZEN_QWEN21_TEXT_LAYERS; layer++) {
        q21_text_block block;
        if (!q21_text_block_load(model, layer, &block, error, error_size)) goto done;
        q21_text_block_run(&block, &model->norms[layer], hidden, tokens, normed, q, k, v,
                           attended, gate_out, up_out, projected, scores, cosine, sine,
                           threads);
        q21_text_block_free(&block);
        if (trace) {
            memcpy(trace + (size_t)layer * Q21_HIDDEN,
                   hidden + (size_t)(tokens - 1) * Q21_HIDDEN, Q21_HIDDEN * sizeof(float));
        }
    }
    memcpy(hidden_out, hidden, hidden_values * sizeof(float));
    ok = 1;
done:
    free(hidden);
    free(normed);
    free(q);
    free(k);
    free(v);
    free(attended);
    free(projected);
    free(gate_out);
    free(up_out);
    free(scores);
    free(cosine);
    free(sine);
    return ok;
}

qzen_qwen21_text *qzen_qwen21_text_open(const char *checkpoint, int thread_count, char *error,
                                    size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!checkpoint) {
        snprintf(error, error_size, "no Qwen3-VL checkpoint path");
        return NULL;
    }
    qzen_qwen21_text *model = calloc(1, sizeof(*model));
    if (!model) {
        snprintf(error, error_size, "out of memory for the Qwen3-VL language tower");
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
    model->embed = qzen_st_find(&model->header, "language_model.embed_tokens.weight");
    const uint64_t embed_shape[2] = {Q21_TEXT_VOCAB, Q21_HIDDEN};
    if (!q21_expect(model->embed, "language_model.embed_tokens.weight", 2, embed_shape, error,
                    error_size) ||
        !q21_text_norms_load(model, error, error_size)) {
        qzen_qwen21_text_close(model);
        return NULL;
    }
    return model;
}

void qzen_qwen21_text_close(qzen_qwen21_text *model) {
    if (!model) return;
    q21_text_gpu_free(model->gpu);
    free(model->norms);
    qzen_st_free_header(&model->header);
    free(model);
}

double qzen_qwen21_text_weight_seconds(const qzen_qwen21_text *model) {
    return model ? model->weight_seconds : 0.0;
}

int qzen_qwen21_text_on_device(const qzen_qwen21_text *model) {
    return model && model->gpu ? 1 : 0;
}

/* --- device forward path (QWEN21_TEXT_GPU=1) ------------------------------
 *
 * The CPU path above is the numerical reference, and it is slow for one
 * structural reason: every block linear is INT8 ConvRot, so the CPU widens
 * 6.9 GB of weights to F32 on every forward (78 s for a 26-token prompt on 32
 * cores, of which 77 s is the widening). The device path keeps the weights in
 * their stored INT8 form and rotates the *activation* instead - the same trade
 * the DiT makes:
 *
 *   RMSNorm over the F32 carrier   qzen_gpu_cvt_f32_bf16 + qzen_gpu_rms_norm_bf16
 *   activation rotation            qzen_gpu_convrot_rotate_bf16
 *   q/k/v/o/gate/up/down           qzen_gpu_linear_int8_bf16
 *   q/k head RMSNorm + RoPE        qzen_gpu_text_qk_rope_bf16 (rotate_half, GQA)
 *   causal attention               qzen_gpu_gqa_causal_bf16 (32 query / 8 KV)
 *   SwiGLU                         qzen_gpu_silu_mul_bf16
 *   residual += branch             qzen_gpu_gate_f32res_bf16 with a ones table
 *
 * Only one layer's weights are on the device at a time. All 36 layers have the
 * same shapes, so the seven weight tensors are allocated once and overwritten
 * from the checkpoint as the walk descends the stack: 193 MB of weights plus
 * ~60 MB of scratch for a 26-token prompt, which leaves the DiT's 15 GB alone
 * instead of the 6.9 GB a resident copy would cost.
 */

typedef struct {
    qzen_gpu_tensor *q, *q_scale;       /* [4096][4096] INT8 + [4096] F32 rows */
    qzen_gpu_tensor *k, *k_scale;       /* [1024][4096] */
    qzen_gpu_tensor *v, *v_scale;       /* [1024][4096] */
    qzen_gpu_tensor *o, *o_scale;       /* [4096][4096] */
    qzen_gpu_tensor *gate, *gate_scale; /* [12288][4096] */
    qzen_gpu_tensor *up, *up_scale;     /* [12288][4096] */
    qzen_gpu_tensor *down, *down_scale; /* [4096][12288] */
    qzen_gpu_tensor *input_norm;        /* BF16 [4096] */
    qzen_gpu_tensor *post_norm;         /* BF16 [4096] */
    qzen_gpu_tensor *q_norm;            /* BF16 [128] */
    qzen_gpu_tensor *k_norm;            /* BF16 [128] */
} q21_text_gpu_weights;

typedef struct q21_text_gpu {
    qzen_gpu *gpu;
    int tokens;                   /* scratch was sized for this many tokens */
    q21_text_gpu_weights weights; /* one layer at a time, overwritten in place */
    qzen_gpu_tensor *carrier;       /* F32 [tokens][4096], the residual */
    qzen_gpu_tensor *x;             /* BF16 [tokens][4096]: carrier for the norm */
    qzen_gpu_tensor *normed;        /* BF16 [tokens][4096] */
    qzen_gpu_tensor *rot;           /* BF16 [tokens][12288]: rotated activation */
    qzen_gpu_tensor *q_raw, *k_raw; /* BF16 before the head norm */
    qzen_gpu_tensor *q, *k;         /* BF16 after head norm + RoPE */
    qzen_gpu_tensor *v;             /* BF16 [tokens][1024] */
    qzen_gpu_tensor *attn;          /* BF16 [tokens][4096] */
    qzen_gpu_tensor *projected;     /* BF16 [tokens][4096] */
    qzen_gpu_tensor *mlp1, *mlp2, *mlp_act; /* BF16 [tokens][12288] */
    qzen_gpu_tensor *act8, *act8_scales;    /* INT8 scratch for 4096-wide inputs */
    qzen_gpu_tensor *mlp8, *mlp8_scales;    /* INT8 scratch for 12288-wide inputs */
    qzen_gpu_tensor *rope_cos, *rope_sin;   /* BF16 [tokens][64] */
    qzen_gpu_tensor *ones;          /* BF16 [4096] all 1.0: plain residual add */
    qzen_gpu_tensor *row_map;       /* U32 [tokens] all 0 */
    double weight_seconds;        /* checkpoint -> device time, this forward */
} q21_text_gpu;

static int q21_text_gpu_enabled(void) {
    const char *value = getenv("QWEN21_TEXT_GPU");
    if (!value || !value[0]) return 0;
    if (!strcmp(value, "0") || !strcmp(value, "false") || !strcmp(value, "no")) return 0;
    return 1;
}

static uint16_t q21_text_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void q21_text_gpu_drop(qzen_gpu_tensor **tensor) {
    if (*tensor) {
        qzen_gpu_tensor_free(*tensor);
        *tensor = NULL;
    }
}

static void q21_text_gpu_weights_free(q21_text_gpu_weights *w) {
    q21_text_gpu_drop(&w->q);
    q21_text_gpu_drop(&w->q_scale);
    q21_text_gpu_drop(&w->k);
    q21_text_gpu_drop(&w->k_scale);
    q21_text_gpu_drop(&w->v);
    q21_text_gpu_drop(&w->v_scale);
    q21_text_gpu_drop(&w->o);
    q21_text_gpu_drop(&w->o_scale);
    q21_text_gpu_drop(&w->gate);
    q21_text_gpu_drop(&w->gate_scale);
    q21_text_gpu_drop(&w->up);
    q21_text_gpu_drop(&w->up_scale);
    q21_text_gpu_drop(&w->down);
    q21_text_gpu_drop(&w->down_scale);
    q21_text_gpu_drop(&w->input_norm);
    q21_text_gpu_drop(&w->post_norm);
    q21_text_gpu_drop(&w->q_norm);
    q21_text_gpu_drop(&w->k_norm);
}

static int q21_text_gpu_weights_alloc(q21_text_gpu *ctx, char *error,
                                      size_t error_size) {
    qzen_gpu *gpu = ctx->gpu;
    q21_text_gpu_weights *w = &ctx->weights;
    w->q = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_HIDDEN * Q21_HIDDEN);
    w->q_scale = qzen_gpu_tensor_new_f32(gpu, Q21_HIDDEN);
    w->k = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_TEXT_KV_DIM * Q21_HIDDEN);
    w->k_scale = qzen_gpu_tensor_new_f32(gpu, Q21_TEXT_KV_DIM);
    w->v = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_TEXT_KV_DIM * Q21_HIDDEN);
    w->v_scale = qzen_gpu_tensor_new_f32(gpu, Q21_TEXT_KV_DIM);
    w->o = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_HIDDEN * Q21_HIDDEN);
    w->o_scale = qzen_gpu_tensor_new_f32(gpu, Q21_HIDDEN);
    w->gate = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_FFN * Q21_HIDDEN);
    w->gate_scale = qzen_gpu_tensor_new_f32(gpu, Q21_FFN);
    w->up = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_FFN * Q21_HIDDEN);
    w->up_scale = qzen_gpu_tensor_new_f32(gpu, Q21_FFN);
    w->down = qzen_gpu_tensor_new_i8(gpu, (size_t)Q21_HIDDEN * Q21_FFN);
    w->down_scale = qzen_gpu_tensor_new_f32(gpu, Q21_HIDDEN);
    w->input_norm = qzen_gpu_tensor_new_bf16(gpu, Q21_HIDDEN);
    w->post_norm = qzen_gpu_tensor_new_bf16(gpu, Q21_HIDDEN);
    w->q_norm = qzen_gpu_tensor_new_bf16(gpu, Q21_HEAD_DIM);
    w->k_norm = qzen_gpu_tensor_new_bf16(gpu, Q21_HEAD_DIM);
    if (!w->q || !w->q_scale || !w->k || !w->k_scale || !w->v || !w->v_scale ||
        !w->o || !w->o_scale || !w->gate || !w->gate_scale || !w->up ||
        !w->up_scale || !w->down || !w->down_scale || !w->input_norm ||
        !w->post_norm || !w->q_norm || !w->k_norm) {
        snprintf(error, error_size,
                 "out of GPU memory for one Qwen3-VL layer of INT8 weights");
        return 0;
    }
    return 1;
}

static void q21_text_gpu_scratch_free(q21_text_gpu *ctx) {
    q21_text_gpu_drop(&ctx->carrier);
    q21_text_gpu_drop(&ctx->x);
    q21_text_gpu_drop(&ctx->normed);
    q21_text_gpu_drop(&ctx->rot);
    q21_text_gpu_drop(&ctx->q_raw);
    q21_text_gpu_drop(&ctx->k_raw);
    q21_text_gpu_drop(&ctx->q);
    q21_text_gpu_drop(&ctx->k);
    q21_text_gpu_drop(&ctx->v);
    q21_text_gpu_drop(&ctx->attn);
    q21_text_gpu_drop(&ctx->projected);
    q21_text_gpu_drop(&ctx->mlp1);
    q21_text_gpu_drop(&ctx->mlp2);
    q21_text_gpu_drop(&ctx->mlp_act);
    q21_text_gpu_drop(&ctx->act8);
    q21_text_gpu_drop(&ctx->act8_scales);
    q21_text_gpu_drop(&ctx->mlp8);
    q21_text_gpu_drop(&ctx->mlp8_scales);
    q21_text_gpu_drop(&ctx->rope_cos);
    q21_text_gpu_drop(&ctx->rope_sin);
    q21_text_gpu_drop(&ctx->ones);
    q21_text_gpu_drop(&ctx->row_map);
    ctx->tokens = 0;
}

/* RoPE tables, the plain-add gate row and the row map are the only inputs that
 * depend on the token count, so they are rebuilt with the scratch. */
static int q21_text_gpu_scratch_alloc(q21_text_gpu *ctx, int tokens, char *error,
                                      size_t error_size) {
    q21_text_gpu_scratch_free(ctx);
    const size_t dim = (size_t)tokens * Q21_HIDDEN;
    const size_t wide = (size_t)tokens * Q21_FFN;
    const size_t kv = (size_t)tokens * Q21_TEXT_KV_DIM;
    /* qzen_gpu_linear_int8_bf16 quantises the activation in place and wants both
     * the payload and the scale vector padded to a multiple of 128 rows. */
    const size_t padded = ((size_t)tokens + 127u) & ~(size_t)127u;
    qzen_gpu *gpu = ctx->gpu;
    ctx->carrier = qzen_gpu_tensor_new_f32(gpu, dim);
    ctx->x = qzen_gpu_tensor_new_bf16(gpu, dim);
    ctx->normed = qzen_gpu_tensor_new_bf16(gpu, dim);
    ctx->rot = qzen_gpu_tensor_new_bf16(gpu, wide);
    ctx->q_raw = qzen_gpu_tensor_new_bf16(gpu, dim);
    ctx->k_raw = qzen_gpu_tensor_new_bf16(gpu, kv);
    ctx->q = qzen_gpu_tensor_new_bf16(gpu, dim);
    ctx->k = qzen_gpu_tensor_new_bf16(gpu, kv);
    ctx->v = qzen_gpu_tensor_new_bf16(gpu, kv);
    ctx->attn = qzen_gpu_tensor_new_bf16(gpu, dim);
    ctx->projected = qzen_gpu_tensor_new_bf16(gpu, dim);
    ctx->mlp1 = qzen_gpu_tensor_new_bf16(gpu, wide);
    ctx->mlp2 = qzen_gpu_tensor_new_bf16(gpu, wide);
    ctx->mlp_act = qzen_gpu_tensor_new_bf16(gpu, wide);
    ctx->act8 = qzen_gpu_tensor_new_i8(gpu, padded * Q21_HIDDEN);
    ctx->act8_scales = qzen_gpu_tensor_new_f32(gpu, padded);
    ctx->mlp8 = qzen_gpu_tensor_new_i8(gpu, padded * Q21_FFN);
    ctx->mlp8_scales = qzen_gpu_tensor_new_f32(gpu, padded);
    ctx->rope_cos = qzen_gpu_tensor_new_bf16(gpu, (size_t)tokens * Q21_TEXT_GROUPS);
    ctx->rope_sin = qzen_gpu_tensor_new_bf16(gpu, (size_t)tokens * Q21_TEXT_GROUPS);
    ctx->ones = qzen_gpu_tensor_new_bf16(gpu, Q21_HIDDEN);
    if (!ctx->carrier || !ctx->x || !ctx->normed || !ctx->rot || !ctx->q_raw ||
        !ctx->k_raw || !ctx->q || !ctx->k || !ctx->v || !ctx->attn ||
        !ctx->projected || !ctx->mlp1 || !ctx->mlp2 || !ctx->mlp_act ||
        !ctx->act8 || !ctx->act8_scales || !ctx->mlp8 || !ctx->mlp8_scales ||
        !ctx->rope_cos || !ctx->rope_sin || !ctx->ones) {
        snprintf(error, error_size, "out of GPU memory for %d tokens of text scratch",
                 tokens);
        return 0;
    }
    float *cosine = malloc((size_t)tokens * Q21_TEXT_GROUPS * sizeof(float));
    float *sine = malloc((size_t)tokens * Q21_TEXT_GROUPS * sizeof(float));
    uint16_t *staged = malloc((size_t)tokens * Q21_TEXT_GROUPS * sizeof(uint16_t));
    uint16_t *one_row = malloc(Q21_HIDDEN * sizeof(uint16_t));
    uint32_t *map = calloc((size_t)tokens, sizeof(uint32_t));
    int ok = 0;
    if (!cosine || !sine || !staged || !one_row || !map) {
        snprintf(error, error_size, "out of memory for the %d token tables", tokens);
        goto done;
    }
    q21_text_rope_tables(tokens, cosine, sine);
    for (size_t index = 0; index < (size_t)tokens * Q21_TEXT_GROUPS; index++) {
        staged[index] = q21_text_bf16(cosine[index]);
    }
    if (!qzen_gpu_tensor_write_bf16(ctx->rope_cos, staged,
                                  (size_t)tokens * Q21_TEXT_GROUPS)) {
        snprintf(error, error_size, "RoPE cosine upload failed");
        goto done;
    }
    for (size_t index = 0; index < (size_t)tokens * Q21_TEXT_GROUPS; index++) {
        staged[index] = q21_text_bf16(sine[index]);
    }
    if (!qzen_gpu_tensor_write_bf16(ctx->rope_sin, staged,
                                  (size_t)tokens * Q21_TEXT_GROUPS)) {
        snprintf(error, error_size, "RoPE sine upload failed");
        goto done;
    }
    for (int index = 0; index < Q21_HIDDEN; index++) one_row[index] = q21_text_bf16(1.0f);
    if (!qzen_gpu_tensor_write_bf16(ctx->ones, one_row, Q21_HIDDEN)) {
        snprintf(error, error_size, "residual gate row upload failed");
        goto done;
    }
    qzen_gpu_tensor *uploaded = qzen_gpu_tensor_from_u32(gpu, map, (size_t)tokens);
    if (!uploaded) {
        snprintf(error, error_size, "row map upload failed");
        goto done;
    }
    ctx->row_map = uploaded;
    ctx->tokens = tokens;
    ok = 1;
done:
    free(cosine);
    free(sine);
    free(staged);
    free(one_row);
    free(map);
    return ok;
}

/* One I8 ConvRot weight plus its per-row F32 scale, straight into the resident
 * device tensors. The payload never becomes F32 on the host. */
static int q21_text_gpu_upload_int8(qzen_st_header *header, qzen_gpu *gpu, const char *name,
                                    int rows, int columns, qzen_gpu_tensor *weight,
                                    qzen_gpu_tensor *scales, char *error,
                                    size_t error_size) {
    const uint64_t shape[2] = {(uint64_t)rows, (uint64_t)columns};
    const qzen_st_tensor *tensor = qzen_st_find(header, name);
    if (!q21_expect(tensor, name, 2, shape, error, error_size)) return 0;
    if (tensor->dtype != QZEN_DTYPE_I8) {
        snprintf(error, error_size, "%s: expected an I8 ConvRot tensor, got %s", name,
                 qzen_dtype_name(tensor->dtype));
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
    if (!payload || !scale_values) {
        free(payload);
        free(scale_values);
        snprintf(error, error_size, "%s: out of memory for %zu INT8 bytes", name,
                 elements);
        return 0;
    }
    int ok = qzen_st_read_data(header, tensor, payload, elements, error, error_size) &&
             qzen_gpu_upload_i8_pinned(gpu, weight, payload, elements) &&
             qzen_st_read_data(header, scale, scale_values, (size_t)rows * sizeof(float),
                             error, error_size) &&
             qzen_gpu_tensor_write_f32(scales, scale_values, (size_t)rows);
    free(payload);
    free(scale_values);
    if (!ok) snprintf(error, error_size, "%s: INT8 upload failed", name);
    return ok;
}

/* Norm weights are small; they are read as F32 (the checkpoint may hold them as
 * F32 or BF16) and rounded once on the way to the device. */
static int q21_text_gpu_upload_norm(qzen_st_header *header, qzen_gpu_tensor *tensor,
                                    const char *name, int width, char *error,
                                    size_t error_size) {
    const uint64_t shape[1] = {(uint64_t)width};
    float *values = q21_read_f32(header, name, 1, shape, error, error_size);
    if (!values) return 0;
    uint16_t *halves = malloc((size_t)width * sizeof(uint16_t));
    if (!halves) {
        free(values);
        snprintf(error, error_size, "%s: out of memory for the BF16 copy", name);
        return 0;
    }
    for (int index = 0; index < width; index++) halves[index] = q21_text_bf16(values[index]);
    int ok = qzen_gpu_tensor_write_bf16(tensor, halves, (size_t)width);
    free(halves);
    free(values);
    if (!ok) snprintf(error, error_size, "%s: device upload failed", name);
    return ok;
}

static int q21_text_gpu_layer_load(q21_text_gpu *ctx, qzen_st_header *header, int layer,
                                   char *error, size_t error_size) {
    q21_text_gpu_weights *w = &ctx->weights;
    struct {
        const char *suffix;
        int rows, columns;
        qzen_gpu_tensor *weight, *scales;
    } projections[7] = {
        {"self_attn.q_proj.weight", Q21_HIDDEN, Q21_HIDDEN, NULL, NULL},
        {"self_attn.k_proj.weight", Q21_TEXT_KV_DIM, Q21_HIDDEN, NULL, NULL},
        {"self_attn.v_proj.weight", Q21_TEXT_KV_DIM, Q21_HIDDEN, NULL, NULL},
        {"self_attn.o_proj.weight", Q21_HIDDEN, Q21_HIDDEN, NULL, NULL},
        {"mlp.gate_proj.weight", Q21_FFN, Q21_HIDDEN, NULL, NULL},
        {"mlp.up_proj.weight", Q21_FFN, Q21_HIDDEN, NULL, NULL},
        {"mlp.down_proj.weight", Q21_HIDDEN, Q21_FFN, NULL, NULL},
    };
    projections[0].weight = w->q;    projections[0].scales = w->q_scale;
    projections[1].weight = w->k;    projections[1].scales = w->k_scale;
    projections[2].weight = w->v;    projections[2].scales = w->v_scale;
    projections[3].weight = w->o;    projections[3].scales = w->o_scale;
    projections[4].weight = w->gate; projections[4].scales = w->gate_scale;
    projections[5].weight = w->up;   projections[5].scales = w->up_scale;
    projections[6].weight = w->down; projections[6].scales = w->down_scale;
    char name[192];
    for (size_t index = 0; index < sizeof(projections) / sizeof(projections[0]);
         index++) {
        snprintf(name, sizeof(name), "language_model.layers.%d.%s", layer,
                 projections[index].suffix);
        if (!q21_text_gpu_upload_int8(header, ctx->gpu, name, projections[index].rows,
                                      projections[index].columns,
                                      projections[index].weight,
                                      projections[index].scales, error, error_size)) {
            return 0;
        }
    }
    struct {
        qzen_gpu_tensor *tensor;
        const char *suffix;
        int width;
    } norms[4] = {
        {w->input_norm, "input_layernorm.weight", Q21_HIDDEN},
        {w->post_norm, "post_attention_layernorm.weight", Q21_HIDDEN},
        {w->q_norm, "self_attn.q_norm.weight", Q21_HEAD_DIM},
        {w->k_norm, "self_attn.k_norm.weight", Q21_HEAD_DIM},
    };
    for (size_t index = 0; index < sizeof(norms) / sizeof(norms[0]); index++) {
        snprintf(name, sizeof(name), "language_model.layers.%d.%s", layer,
                 norms[index].suffix);
        if (!q21_text_gpu_upload_norm(header, norms[index].tensor, name,
                                      norms[index].width, error, error_size)) {
            return 0;
        }
    }
    return 1;
}

static int q21_text_gpu_block(q21_text_gpu *ctx, int tokens, char *error,
                              size_t error_size) {
    const q21_text_gpu_weights *w = &ctx->weights;
    qzen_gpu *gpu = ctx->gpu;
    const uint32_t rows = (uint32_t)tokens;
    const float scale = 1.0f / sqrtf((float)Q21_HEAD_DIM);
#define Q21_TEXT_OP(call, label)                                                  \
    do {                                                                          \
        if (!(call)) {                                                            \
            snprintf(error, error_size, "%s failed: %s", label,                   \
                     qzen_gpu_error(gpu));                                          \
            return 0;                                                             \
        }                                                                         \
    } while (0)
    Q21_TEXT_OP(qzen_gpu_cvt_f32_bf16(gpu, ctx->x, ctx->carrier,
                                    rows * (uint32_t)Q21_HIDDEN),
                "carrier widen");
    Q21_TEXT_OP(qzen_gpu_rms_norm_bf16(gpu, ctx->normed, ctx->x, w->input_norm, rows,
                                     Q21_HIDDEN, Q21_EPS),
                "input RMSNorm");
    Q21_TEXT_OP(qzen_gpu_convrot_rotate_bf16(gpu, ctx->rot, ctx->normed, rows, Q21_HIDDEN),
                "attention rotation");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->q_raw, ctx->act8, ctx->act8_scales,
                                        ctx->rot, w->q, w->q_scale, rows, Q21_HIDDEN,
                                        Q21_HIDDEN, 0, 0),
                "to_q");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->k_raw, ctx->act8, ctx->act8_scales,
                                        ctx->rot, w->k, w->k_scale, rows, Q21_HIDDEN,
                                        Q21_TEXT_KV_DIM, 0, 0),
                "to_k");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->v, ctx->act8, ctx->act8_scales,
                                        ctx->rot, w->v, w->v_scale, rows, Q21_HIDDEN,
                                        Q21_TEXT_KV_DIM, 0, 0),
                "to_v");
    /* q/k head norm + rotate_half RoPE cannot run in place: the kernel reads
     * both halves of a head to write either half, so the pre-norm GEMM output
     * is kept and the rotated result lands in q/k. */
    Q21_TEXT_OP(qzen_gpu_text_qk_rope_bf16(gpu, ctx->q, ctx->k, ctx->q_raw, ctx->k_raw,
                                         w->q_norm, w->k_norm, ctx->rope_cos,
                                         ctx->rope_sin, rows, Q21_TEXT_HEADS,
                                         Q21_TEXT_KV_HEADS, Q21_HEAD_DIM, Q21_EPS),
                "q/k RMSNorm + RoPE");
    Q21_TEXT_OP(qzen_gpu_gqa_causal_bf16(gpu, ctx->attn, ctx->q, ctx->k, ctx->v, rows,
                                       Q21_TEXT_HEADS, Q21_TEXT_KV_HEADS,
                                       Q21_HEAD_DIM, scale),
                "causal attention");
    Q21_TEXT_OP(qzen_gpu_convrot_rotate_bf16(gpu, ctx->rot, ctx->attn, rows, Q21_HIDDEN),
                "o_proj rotation");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->projected, ctx->act8,
                                        ctx->act8_scales, ctx->rot, w->o, w->o_scale,
                                        rows, Q21_HIDDEN, Q21_HIDDEN, 0, 0),
                "to_o");
    Q21_TEXT_OP(qzen_gpu_gate_f32res_bf16(gpu, ctx->carrier, ctx->projected, ctx->ones,
                                        NULL, 0, ctx->row_map, rows, Q21_HIDDEN, 1, 0),
                "attention residual");
    Q21_TEXT_OP(qzen_gpu_cvt_f32_bf16(gpu, ctx->x, ctx->carrier,
                                    rows * (uint32_t)Q21_HIDDEN),
                "carrier widen");
    Q21_TEXT_OP(qzen_gpu_rms_norm_bf16(gpu, ctx->normed, ctx->x, w->post_norm, rows,
                                     Q21_HIDDEN, Q21_EPS),
                "post RMSNorm");
    Q21_TEXT_OP(qzen_gpu_convrot_rotate_bf16(gpu, ctx->rot, ctx->normed, rows, Q21_HIDDEN),
                "mlp rotation");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->mlp1, ctx->act8, ctx->act8_scales,
                                        ctx->rot, w->gate, w->gate_scale, rows,
                                        Q21_HIDDEN, Q21_FFN, 0, 0),
                "gate_proj");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->mlp2, ctx->act8, ctx->act8_scales,
                                        ctx->rot, w->up, w->up_scale, rows, Q21_HIDDEN,
                                        Q21_FFN, 0, 0),
                "up_proj");
    Q21_TEXT_OP(qzen_gpu_silu_mul_bf16(gpu, ctx->mlp_act, ctx->mlp1, ctx->mlp2,
                                     rows * (uint32_t)Q21_FFN),
                "SwiGLU");
    Q21_TEXT_OP(qzen_gpu_convrot_rotate_bf16(gpu, ctx->rot, ctx->mlp_act, rows, Q21_FFN),
                "down rotation");
    Q21_TEXT_OP(qzen_gpu_linear_int8_bf16(gpu, ctx->projected, ctx->mlp8,
                                        ctx->mlp8_scales, ctx->rot, w->down,
                                        w->down_scale, rows, Q21_FFN, Q21_HIDDEN, 0, 0),
                "down_proj");
    Q21_TEXT_OP(qzen_gpu_gate_f32res_bf16(gpu, ctx->carrier, ctx->projected, ctx->ones,
                                        NULL, 0, ctx->row_map, rows, Q21_HIDDEN, 1, 0),
                "mlp residual");
#undef Q21_TEXT_OP
    return 1;
}

static struct q21_text_gpu *q21_text_gpu_open(char *error, size_t error_size) {
    q21_text_gpu *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        snprintf(error, error_size, "out of memory for the device text context");
        return NULL;
    }
    ctx->gpu = qzen_gpu_create(NULL, error, error_size);
    if (!ctx->gpu) {
        free(ctx);
        return NULL;
    }
    if (!q21_text_gpu_weights_alloc(ctx, error, error_size)) {
        q21_text_gpu_free(ctx);
        return NULL;
    }
    return ctx;
}

static void q21_text_gpu_free(struct q21_text_gpu *ctx) {
    if (!ctx) return;
    q21_text_gpu_scratch_free(ctx);
    q21_text_gpu_weights_free(&ctx->weights);
    if (ctx->gpu) qzen_gpu_free(ctx->gpu);
    free(ctx);
}

static int q21_text_gpu_forward(qzen_qwen21_text *model, const uint32_t *ids, size_t count,
                                float *hidden_out, float *trace, char *error,
                                size_t error_size) {
    q21_text_gpu *ctx = model->gpu;
    const int tokens = (int)count;
    const size_t dim = (size_t)tokens * Q21_HIDDEN;
    const double started = q21_now();
    if (ctx->tokens != tokens &&
        !q21_text_gpu_scratch_alloc(ctx, tokens, error, error_size)) {
        return 0;
    }
    float *embedding = malloc(dim * sizeof(float));
    if (!embedding) {
        snprintf(error, error_size, "out of memory for %d token embeddings", tokens);
        return 0;
    }
    int ok = 0;
    for (int token = 0; token < tokens; token++) {
        if (ids[token] >= Q21_TEXT_VOCAB) {
            snprintf(error, error_size, "token id %u is outside the %d row vocabulary",
                     ids[token], Q21_TEXT_VOCAB);
            goto done;
        }
        if (!q21_read_rows_f32(&model->header, model->embed, ids[token], 1, Q21_HIDDEN,
                               embedding + (size_t)token * Q21_HIDDEN, error,
                               error_size)) {
            goto done;
        }
    }
    if (!qzen_gpu_tensor_write_f32(ctx->carrier, embedding, dim)) {
        snprintf(error, error_size, "embedding upload failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    ctx->weight_seconds = 0.0;
    for (int layer = 0; layer < QZEN_QWEN21_TEXT_LAYERS; layer++) {
        const double load_started = q21_now();
        if (!q21_text_gpu_layer_load(ctx, &model->header, layer, error, error_size)) {
            goto done;
        }
        ctx->weight_seconds += q21_now() - load_started;
        if (!q21_text_gpu_block(ctx, tokens, error, error_size)) goto done;
        if (trace) {
            /* Reading a row synchronises the stream, so the per-layer trace is
             * only collected when the caller asked for it (tests, diagnosis). */
            if (!qzen_gpu_tensor_read_f32_range(ctx->carrier,
                                              (size_t)(tokens - 1) * Q21_HIDDEN,
                                              trace + (size_t)layer * Q21_HIDDEN,
                                              Q21_HIDDEN)) {
                snprintf(error, error_size, "layer %d trace read failed", layer);
                goto done;
            }
        }
    }
    if (!qzen_gpu_submit(ctx->gpu)) {
        snprintf(error, error_size, "device submit failed: %s", qzen_gpu_error(ctx->gpu));
        goto done;
    }
    if (!qzen_gpu_tensor_read_f32(ctx->carrier, hidden_out, dim)) {
        snprintf(error, error_size, "hidden state read failed: %s",
                 qzen_gpu_error(ctx->gpu));
        goto done;
    }
    ok = 1;
    /* Same field on both paths: the CPU side accumulates dequantisation time,
     * the device side the checkpoint -> device time. Without this the test
     * harness reports "0.0 s of it layer weight reads" for a forward that spent
     * 3.8 s of its 3.9 s reading. */
    model->weight_seconds += ctx->weight_seconds;
    fprintf(stderr,
            "QwenZImageNative qwen21 text: %d tokens on the device, %d layers, %.2fs "
            "(%.2fs of it weight upload)\n",
            tokens, QZEN_QWEN21_TEXT_LAYERS, q21_now() - started, ctx->weight_seconds);
    fflush(stderr);
done:
    free(embedding);
    return ok;
}

