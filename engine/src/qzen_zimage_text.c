#include "qzen_zimage_text.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* --- small helpers --------------------------------------------------------- */

static float ztext_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void ztext_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Read a plain BF16 tensor [dim0, dim1] (or 1-D [count]) into a BF16 buffer. */
static int ztext_read_bf16(const qzen_weight_store *store, const char *name,
                           uint64_t dim0, uint64_t dim1, uint16_t *out,
                           char *error, size_t error_size) {
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        ztext_fail(error, error_size, "required Qwen3 weight is absent: %s", name);
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
        ztext_fail(error, error_size,
                  "weight %s is %s[%d] with shape [%llu, %llu], expected BF16 [%llu, %llu]",
                  name, qzen_dtype_name(tensor->dtype), tensor->ndim,
                  (unsigned long long)tensor->shape[0],
                  (unsigned long long)tensor->shape[1],
                  (unsigned long long)dim0, (unsigned long long)dim1);
        return 0;
    }
    return qzen_st_read_data(header, tensor, out, (size_t)elements * 2,
                           error, error_size);
}

/* Read a plain BF16 vector into F32 (RMSNorm weights). */
static int ztext_read_bf16_f32(const qzen_weight_store *store, const char *name,
                               uint64_t count, float *out,
                               char *error, size_t error_size) {
    uint16_t *staged = malloc((size_t)count * 2);
    if (!staged) {
        ztext_fail(error, error_size, "cannot stage %s", name);
        return 0;
    }
    int ok = ztext_read_bf16(store, name, count, 1, staged, error, error_size);
    if (ok) {
        for (uint64_t index = 0; index < count; index++) {
            out[index] = ztext_bf16_to_f32(staged[index]);
        }
    }
    free(staged);
    return ok;
}

qzen_ztext_config qzen_ztext_default_config(void) {
    qzen_ztext_config config;
    config.layers = 36;
    config.dim = 2560;
    config.heads = 32;
    config.kv_heads = 8;
    config.head_dim = 128;
    config.ffn_dim = 9728;
    config.vocab = 151936;
    config.norm_eps = 1e-6f;
    config.rope_theta = 1e6f;
    config.max_length = 512;
    return config;
}

/* --- model ----------------------------------------------------------------- */

typedef struct {
    qzen_zimage_linear q_proj;  /* [heads * head_dim, dim] */
    qzen_zimage_linear k_proj;  /* [kv_heads * head_dim, dim] */
    qzen_zimage_linear v_proj;  /* [kv_heads * head_dim, dim] */
    qzen_zimage_linear o_proj;  /* [dim, heads * head_dim] */
    qzen_zimage_linear gate_proj; /* [ffn_dim, dim] */
    qzen_zimage_linear up_proj;   /* [ffn_dim, dim] */
    qzen_zimage_linear down_proj; /* [dim, ffn_dim] */
    float q_norm[128];        /* RMSNorm over head_dim, eps = norm_eps */
    float k_norm[128];
    float input_ln[2560];     /* RMSNorm over dim */
    float post_attention_ln[2560];
} qzen_ztext_layer;

struct qzen_ztext_model {
    qzen_ztext_config config;
    uint16_t *embed_tokens;   /* [vocab, dim] BF16, tie_word_embeddings */
    qzen_ztext_layer *layers;
    uint32_t layer_count;
};

#define ZTEXT_Q_NORM_DIM 128
#define ZTEXT_LN_DIM 2560

static int ztext_load_layer(const qzen_weight_store *store, uint32_t index,
                            const qzen_ztext_config *config,
                            qzen_ztext_layer *layer,
                            char *error, size_t error_size) {
    char name[192];
    const char *prefix = "model.layers.";
    /* All Linears are quanto qint8 pairs, no bias (attention_bias false). */
    static const struct {
        const char *sub;
        uint32_t rows;
        uint32_t cols;
    } linears[7] = {
        { "self_attn.q_proj.weight", 0, 0 },
        { "self_attn.k_proj.weight", 0, 0 },
        { "self_attn.v_proj.weight", 0, 0 },
        { "self_attn.o_proj.weight", 0, 0 },
        { "mlp.gate_proj.weight", 0, 0 },
        { "mlp.up_proj.weight", 0, 0 },
        { "mlp.down_proj.weight", 0, 0 }
    };
    uint32_t q_rows = config->heads * config->head_dim;
    uint32_t kv_rows = config->kv_heads * config->head_dim;
    uint32_t rows[7] = { q_rows, kv_rows, kv_rows, config->dim,
                         config->ffn_dim, config->ffn_dim, config->dim };
    /* Attention runs at heads * head_dim = 4096 wide, not hidden 2560; o_proj
     * and down_proj map back to the hidden width. */
    uint32_t cols[7] = { config->dim, config->dim, config->dim, q_rows,
                         config->dim, config->dim, config->ffn_dim };
    qzen_zimage_linear *targets[7] = {
        &layer->q_proj, &layer->k_proj, &layer->v_proj, &layer->o_proj,
        &layer->gate_proj, &layer->up_proj, &layer->down_proj
    };
    for (int entry = 0; entry < 7; entry++) {
        snprintf(name, sizeof(name), "%s%u.%s", prefix, index, linears[entry].sub);
        if (!qzen_zimage_linear_load_quanto(store, name, rows[entry], cols[entry], 0,
                                          targets[entry], error, error_size)) {
            return 0;
        }
    }
    snprintf(name, sizeof(name), "%s%u.self_attn.q_norm.weight", prefix, index);
    if (!ztext_read_bf16_f32(store, name, ZTEXT_Q_NORM_DIM, layer->q_norm,
                             error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s%u.self_attn.k_norm.weight", prefix, index);
    if (!ztext_read_bf16_f32(store, name, ZTEXT_Q_NORM_DIM, layer->k_norm,
                             error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s%u.input_layernorm.weight", prefix, index);
    if (!ztext_read_bf16_f32(store, name, ZTEXT_LN_DIM, layer->input_ln,
                             error, error_size)) return 0;
    snprintf(name, sizeof(name), "%s%u.post_attention_layernorm.weight", prefix, index);
    if (!ztext_read_bf16_f32(store, name, ZTEXT_LN_DIM, layer->post_attention_ln,
                             error, error_size)) return 0;
    return 1;
}

static void ztext_free_layer(qzen_ztext_layer *layer) {
    qzen_zimage_linear *linears[7] = {
        &layer->q_proj, &layer->k_proj, &layer->v_proj, &layer->o_proj,
        &layer->gate_proj, &layer->up_proj, &layer->down_proj
    };
    for (int entry = 0; entry < 7; entry++) qzen_zimage_linear_free(linears[entry]);
}

qzen_ztext_model *qzen_ztext_model_load(const qzen_weight_store *store,
                                    const qzen_ztext_config *config,
                                    void (*progress)(uint32_t layer, uint32_t total),
                                    char *error, size_t error_size) {
    qzen_ztext_model *model = calloc(1, sizeof(*model));
    if (!model) {
        ztext_fail(error, error_size, "cannot allocate Qwen3 model");
        return NULL;
    }
    model->config = *config;
    model->layer_count = config->layers;
    model->embed_tokens =
        malloc((size_t)config->vocab * config->dim * sizeof(uint16_t));
    model->layers = calloc(config->layers, sizeof(qzen_ztext_layer));
    if (!model->embed_tokens || !model->layers) {
        ztext_fail(error, error_size, "cannot allocate Qwen3 weights");
        qzen_ztext_model_free(model);
        return NULL;
    }
    if (!ztext_read_bf16(store, "model.embed_tokens.weight", config->vocab,
                         config->dim, model->embed_tokens, error, error_size)) {
        qzen_ztext_model_free(model);
        return NULL;
    }
    for (uint32_t index = 0; index < config->layers; index++) {
        if (!ztext_load_layer(store, index, config, &model->layers[index],
                              error, error_size)) {
            qzen_ztext_model_free(model);
            return NULL;
        }
        if (progress) progress(index + 1, config->layers);
    }
    return model;
}

void qzen_ztext_model_free(qzen_ztext_model *model) {
    if (!model) return;
    free(model->embed_tokens);
    if (model->layers) {
        for (uint32_t index = 0; index < model->layer_count; index++) {
            ztext_free_layer(&model->layers[index]);
        }
    }
    free(model->layers);
    free(model);
}

/* --- chat template + tokenizer ---------------------------------------------- */

char *qzen_ztext_format_prompt(const char *prompt) {
    static const char head[] = "<|im_start|>user\n";
    static const char tail[] = "<|im_end|>\n<|im_start|>assistant\n";
    size_t prompt_len = strlen(prompt);
    size_t total = sizeof(head) - 1 + prompt_len + sizeof(tail) - 1 + 1;
    char *text = malloc(total);
    if (!text) return NULL;
    memcpy(text, head, sizeof(head) - 1);
    memcpy(text + sizeof(head) - 1, prompt, prompt_len);
    memcpy(text + sizeof(head) - 1 + prompt_len, tail, sizeof(tail));
    return text;
}

int qzen_ztext_build_inputs(const qzen_ztext_config *config,
                          const qzen_tokenizer *tokenizer,
                          const char *prompt, int32_t *ids, uint8_t *mask,
                          uint32_t *real_len, char *error, size_t error_size) {
    char *formatted = qzen_ztext_format_prompt(prompt);
    if (!formatted) {
        ztext_fail(error, error_size, "cannot allocate chat-formatted prompt");
        return 0;
    }
    uint32_t *raw = NULL;
    size_t count = 0;
    int ok = qzen_tokenizer_encode(tokenizer, formatted, 0, &raw, &count,
                                 error, error_size);
    free(formatted);
    if (!ok) return 0;
    if (count > config->max_length) {
        /* truncation=True: HF truncates to max_length (right padding side). */
        count = config->max_length;
    }
    for (uint32_t position = 0; position < config->max_length; position++) {
        int real = position < count;
        ids[position] = real ? (int32_t)raw[position] : (int32_t)QZEN_PAD_TOKEN_ID;
        mask[position] = real ? 1 : 0;
    }
    free(raw);
    if (real_len) *real_len = (uint32_t)count;
    return 1;
}

/* --- forward ----------------------------------------------------------------- */

/* out[rows, out_dim] = input[rows, in_dim] @ weight^T, BF16 weight widened per
 * element (same numerics as qzen_zimage_linear_forward, OpenMP over rows). */
static void ztext_linear_forward(const qzen_zimage_linear *linear,
                                 const float *input, uint32_t rows,
                                 float *output) {
    const uint32_t in = linear->in;
    const uint32_t out = linear->out;
    {
        int64_t row;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row = 0; row < (int64_t)rows; row++) {
            const float *x = input + (size_t)row * in;
            float *y = output + (size_t)row * out;
            for (uint32_t o = 0; o < out; o++) {
                const uint16_t *w = linear->weight + (size_t)o * in;
                float sum = 0.0f;
                for (uint32_t c = 0; c < in; c++) {
                    sum += x[c] * ztext_bf16_to_f32(w[c]);
                }
                y[o] = sum;
            }
        }
    }
}

/* out = rms(in) * weight over the last width elements of each row. */
static void ztext_rms_forward(const float *weight, uint32_t width, float eps,
                              const float *input, uint32_t rows, float *output) {
    {
        int64_t row;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row = 0; row < (int64_t)rows; row++) {
            const float *x = input + (size_t)row * width;
            float *y = output + (size_t)row * width;
            double sum = 0.0;
            for (uint32_t c = 0; c < width; c++) sum += (double)x[c] * x[c];
            float inverse = 1.0f / sqrtf((float)(sum / width) + eps);
            for (uint32_t c = 0; c < width; c++) y[c] = x[c] * inverse * weight[c];
        }
    }
}

/* GPT-NeoX style rotary: pair (i, i + half) rotates with frequency i,
 * q_embed = q * cos + rotate_half(q) * sin with cos/sin covering head_dim. */
static void ztext_rope_tables(const qzen_ztext_config *config, uint32_t seq,
                              float *cos, float *sin) {
    uint32_t half = config->head_dim / 2;
    for (uint32_t position = 0; position < seq; position++) {
        for (uint32_t i = 0; i < half; i++) {
            float exponent = (float)(2 * i) / (float)config->head_dim;
            float frequency = 1.0f / powf(config->rope_theta, exponent);
            float angle = (float)position * frequency;
            cos[position * half + i] = cosf(angle);
            sin[position * half + i] = sinf(angle);
        }
    }
}

static void ztext_apply_rope(const float *cos, const float *sin,
                             uint32_t head_dim, uint32_t rows, float *x) {
    uint32_t half = head_dim / 2;
    for (uint32_t row = 0; row < rows; row++) {
        const float *row_cos = cos + (size_t)row * half;
        const float *row_sin = sin + (size_t)row * half;
        float *v = x + (size_t)row * head_dim;
        for (uint32_t i = 0; i < half; i++) {
            float a = v[i];
            float b = v[i + half];
            v[i] = a * row_cos[i] - b * row_sin[i];
            v[i + half] = b * row_cos[i] + a * row_sin[i];
        }
    }
}

int qzen_ztext_forward(const qzen_ztext_model *model, const int32_t *ids,
                     uint32_t seq, uint32_t max_layers, float *out,
                     char *error, size_t error_size) {
    const qzen_ztext_config *config = &model->config;
    const uint32_t dim = config->dim;
    const uint32_t head_dim = config->head_dim;
    const uint32_t heads = config->heads;
    const uint32_t kv_heads = config->kv_heads;
    const uint32_t ffn = config->ffn_dim;
    const uint32_t q_rows = heads * head_dim;
    const uint32_t kv_rows = kv_heads * head_dim;
    const uint32_t group = heads / kv_heads;   /* GQA: 4 Q heads per KV head */
    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint32_t run_layers =
        (max_layers == 0 || max_layers > config->layers) ? config->layers : max_layers;

    size_t plane = (size_t)seq * dim;
    float *x = malloc(plane * sizeof(float));
    float *normed = malloc(plane * sizeof(float));
    float *q = malloc((size_t)seq * q_rows * sizeof(float));
    float *k = malloc((size_t)seq * kv_rows * sizeof(float));
    float *v = malloc((size_t)seq * kv_rows * sizeof(float));
    float *attn = malloc((size_t)seq * q_rows * sizeof(float));
    float *branch = malloc(plane * sizeof(float));
    float *ffn_a = malloc((size_t)seq * ffn * sizeof(float));
    float *ffn_b = malloc((size_t)seq * ffn * sizeof(float));
    float *rope_cos = malloc((size_t)seq * (head_dim / 2) * sizeof(float));
    float *rope_sin = malloc((size_t)seq * (head_dim / 2) * sizeof(float));
    if (!x || !normed || !q || !k || !v || !attn || !branch || !ffn_a ||
        !ffn_b || !rope_cos || !rope_sin) {
        ztext_fail(error, error_size, "out of host memory in Qwen3 forward");
        free(x); free(normed); free(q); free(k); free(v); free(attn);
        free(branch); free(ffn_a); free(ffn_b);
        free(rope_cos); free(rope_sin);
        return 0;
    }

    ztext_rope_tables(config, seq, rope_cos, rope_sin);

    /* Embedding lookup widened to F32. */
    for (uint32_t position = 0; position < seq; position++) {
        int32_t id = ids[position];
        if (id < 0 || (uint32_t)id >= config->vocab) {
            ztext_fail(error, error_size, "token id %d out of range", id);
            free(x); free(normed); free(q); free(k); free(v); free(attn);
            free(branch); free(ffn_a); free(ffn_b);
            free(rope_cos); free(rope_sin);
            return 0;
        }
        const uint16_t *row = model->embed_tokens + (size_t)id * dim;
        float *dst = x + (size_t)position * dim;
        for (uint32_t c = 0; c < dim; c++) dst[c] = ztext_bf16_to_f32(row[c]);
    }

    for (uint32_t layer_index = 0; layer_index < run_layers; layer_index++) {
        const qzen_ztext_layer *layer = &model->layers[layer_index];

        /* attention branch: rms -> q/k/v -> qk norm -> rope -> causal GQA */
        ztext_rms_forward(layer->input_ln, dim, config->norm_eps, x, seq, normed);
        ztext_linear_forward(&layer->q_proj, normed, seq, q);
        ztext_linear_forward(&layer->k_proj, normed, seq, k);
        ztext_linear_forward(&layer->v_proj, normed, seq, v);

        /* Per-head q/k RMSNorm (weight over head_dim), then RoPE. */
        for (uint32_t position = 0; position < seq; position++) {
            const float *cos_row = rope_cos + (size_t)position * (head_dim / 2);
            const float *sin_row = rope_sin + (size_t)position * (head_dim / 2);
            for (uint32_t h = 0; h < heads; h++) {
                float *q_head = q + (size_t)position * q_rows + (size_t)h * head_dim;
                float sum = 0.0;
                for (uint32_t c = 0; c < head_dim; c++) sum += (double)q_head[c] * q_head[c];
                float inverse = 1.0f / sqrtf((float)(sum / head_dim) + config->norm_eps);
                for (uint32_t c = 0; c < head_dim; c++) q_head[c] *= inverse * layer->q_norm[c];
                ztext_apply_rope(cos_row, sin_row, head_dim, 1, q_head);
            }
            for (uint32_t h = 0; h < kv_heads; h++) {
                float *k_head = k + (size_t)position * kv_rows + (size_t)h * head_dim;
                float sum = 0.0;
                for (uint32_t c = 0; c < head_dim; c++) sum += (double)k_head[c] * k_head[c];
                float inverse = 1.0f / sqrtf((float)(sum / head_dim) + config->norm_eps);
                for (uint32_t c = 0; c < head_dim; c++) k_head[c] *= inverse * layer->k_norm[c];
                ztext_apply_rope(cos_row, sin_row, head_dim, 1, k_head);
            }
        }

        /* Causal GQA attention: Q head h attends KV head h / group.  One
         * softmax scratch row per head: threads must not share scores. */
        float *head_scores = malloc((size_t)heads * seq * sizeof(float));
        if (!head_scores) {
            ztext_fail(error, error_size, "out of host memory in attention");
            free(x); free(normed); free(q); free(k); free(v); free(attn);
            free(branch); free(ffn_a); free(ffn_b); free(rope_cos); free(rope_sin);
            return 0;
        }
        {
            int64_t h;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (h = 0; h < (int64_t)heads; h++) {
                float *scores = head_scores + (size_t)h * seq;
                const float *k_head = k + (size_t)(h / group) * head_dim;
                const float *v_head = v + (size_t)(h / group) * head_dim;
                for (uint32_t position = 0; position < seq; position++) {
                    const float *q_head = q + (size_t)position * q_rows + (size_t)h * head_dim;
                    float peak = -INFINITY;
                    for (uint32_t past = 0; past <= position; past++) {
                        const float *k_row = k_head + (size_t)past * kv_rows;
                        float dot = 0.0f;
                        for (uint32_t c = 0; c < head_dim; c++) dot += q_head[c] * k_row[c];
                        dot *= scale;
                        scores[past] = dot;
                        if (dot > peak) peak = dot;
                    }
                    float total = 0.0f;
                    for (uint32_t past = 0; past <= position; past++) {
                        scores[past] = expf(scores[past] - peak);
                        total += scores[past];
                    }
                    float *out_row = attn + (size_t)position * q_rows + (size_t)h * head_dim;
                    for (uint32_t c = 0; c < head_dim; c++) out_row[c] = 0.0f;
                    for (uint32_t past = 0; past <= position; past++) {
                        float weight = scores[past] / total;
                        const float *v_row = v_head + (size_t)past * kv_rows;
                        for (uint32_t c = 0; c < head_dim; c++) out_row[c] += weight * v_row[c];
                    }
                }
            }
        }
        free(head_scores);
        ztext_linear_forward(&layer->o_proj, attn, seq, branch);
        for (size_t index = 0; index < plane; index++) x[index] += branch[index];

        /* MLP branch: rms -> silu(gate) * up -> down. */
        ztext_rms_forward(layer->post_attention_ln, dim, config->norm_eps, x, seq, normed);
        ztext_linear_forward(&layer->gate_proj, normed, seq, ffn_a);
        ztext_linear_forward(&layer->up_proj, normed, seq, ffn_b);
        for (size_t index = 0; index < (size_t)seq * ffn; index++) {
            float gate = ffn_a[index];
            ffn_a[index] = gate / (1.0f + expf(-gate)) * ffn_b[index];
        }
        ztext_linear_forward(&layer->down_proj, ffn_a, seq, branch);
        for (size_t index = 0; index < plane; index++) x[index] += branch[index];
    }

    /* hidden_states[-2]: last processed layer output before the final norm. */
    memcpy(out, x, plane * sizeof(float));
    free(x); free(normed); free(q); free(k); free(v); free(attn);
    free(branch); free(ffn_a); free(ffn_b);
    free(rope_cos); free(rope_sin);
    return 1;
}