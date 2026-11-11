#include "qzen_zimage_pipeline.h"
#include "qzen_quanto.h"
#include "qzen_weights.h"
#include "qzen_zimage_text.h"
#include "qzen_tokenizer.h"
#include "qzen_safetensors.h"
#include "qzen_zimage_control.h"
#include "qzen_zimage_vae.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Coarse phase timers for the GPU path (ZPIPE_TRACE=1). */
static double zpipe_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int zpipe_trace_timers(void) { return getenv("ZPIPE_TRACE") != NULL; }

/* --- small helpers ---------------------------------------------------------- */

static void zpipe_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float zpipe_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Read a plain BF16 tensor [dim0, dim1] (or [dim0]) into F32. */
static int zpipe_read_bf16_f32(const qzen_weight_store *store, const char *name,
                               uint64_t dim0, uint64_t dim1, float *out,
                               char *error, size_t error_size) {
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        zpipe_fail(error, error_size, "required Z-Image weight is absent: %s", name);
        return 0;
    }
    uint64_t elements = dim0 * dim1;
    int shape_ok = tensor->dtype == QZEN_DTYPE_BF16;
    if (shape_ok && tensor->ndim == 1) {
        shape_ok = (dim1 == 1 && tensor->shape[0] == dim0) ||
                   (dim0 == 1 && tensor->shape[0] == dim1);
    } else if (shape_ok && tensor->ndim == 2) {
        shape_ok = tensor->shape[0] == dim0 && tensor->shape[1] == dim1;
        /* pad tokens store as [1, dim] rows */
        shape_ok = shape_ok || (dim1 == 1 && tensor->shape[0] == 1 &&
                                tensor->shape[1] == dim0);
    } else {
        shape_ok = 0;
    }
    if (!shape_ok) {
        zpipe_fail(error, error_size,
                   "weight %s is %s[%d] with shape [%llu, %llu], expected BF16 [%llu, %llu]",
                   name, qzen_dtype_name(tensor->dtype), tensor->ndim,
                   (unsigned long long)tensor->shape[0],
                   (unsigned long long)tensor->shape[1],
                   (unsigned long long)dim0, (unsigned long long)dim1);
        return 0;
    }
    uint16_t *staged = malloc((size_t)elements * 2);
    if (!staged) {
        zpipe_fail(error, error_size, "cannot stage %s", name);
        return 0;
    }
    if (!qzen_st_read_data(header, tensor, staged, (size_t)elements * 2, error, error_size)) {
        free(staged);
        return 0;
    }
    for (uint64_t index = 0; index < elements; index++) {
        out[index] = zpipe_bf16_to_f32(staged[index]);
    }
    free(staged);
    return 1;
}

/* --- pipeline struct --------------------------------------------------------- */

struct qzen_zimage_pipeline {
    qzen_zpipe_config config;
    qzen_zimage_config dit_config;
    qzen_zimage_t_embedder t_embedder;
    qzen_zimage_linear x_embedder;   /* Linear(64 -> dim), bias */
    qzen_zimage_rms cap_norm;        /* RMSNorm(2560), eps 1e-5 */
    qzen_zimage_linear cap_linear;   /* Linear(2560 -> dim), bias */
    float *x_pad_token;            /* [dim] */
    float *cap_pad_token;          /* [dim] */
    qzen_zimage_block *noise_refiner[2];
    qzen_zimage_block *context_refiner[2];
    qzen_zimage_block *layers[30];
    qzen_zimage_final_layer final_layer;
    /* optional GPU forward path (NULL falls back to the CPU reference). */
    qzen_zimage_gpu *gpu;
    /* optional text encoder (qzen_zpipe_attach_text) */
    qzen_ztext_model *text_model;
    qzen_tokenizer *tokenizer;
    qzen_ztext_config text_config;
};

/* Upload every GPU-resident weight.  The host copies stay in place: the
 * adaLN modulation Linears keep running on the host and the CPU path remains
 * available as a fallback (ZIMAGE_GPU=0 skips the upload entirely). */
static int zpipe_upload_gpu(qzen_zimage_pipeline *pipe, char *error,
                            size_t error_size) {
    const char *switch_env = getenv("ZIMAGE_GPU");
    if (switch_env && strcmp(switch_env, "0") == 0) return 1;
    const qzen_zimage_config *dit = &pipe->dit_config;
    pipe->gpu = qzen_zimage_gpu_create(dit, error, error_size);
    if (!pipe->gpu) {
        if (error) *error = '\0';
        return 1; /* no CUDA device: stay on the CPU path */
    }
    int ok = qzen_zimage_t_embedder_upload(&pipe->t_embedder, pipe->gpu, error,
                                         error_size) &&
             qzen_zimage_linear_upload(&pipe->x_embedder, pipe->gpu, error,
                                     error_size) &&
             qzen_zimage_rms_upload(&pipe->cap_norm, pipe->gpu, error,
                                  error_size) &&
             qzen_zimage_linear_upload(&pipe->cap_linear, pipe->gpu, error,
                                     error_size) &&
             qzen_zimage_final_layer_upload(&pipe->final_layer, pipe->gpu, error,
                                          error_size);
    for (uint32_t index = 0; ok && index < 2; index++)
        ok = qzen_zimage_block_upload(pipe->noise_refiner[index], pipe->gpu,
                                    error, error_size);
    for (uint32_t index = 0; ok && index < 2; index++)
        ok = qzen_zimage_block_upload(pipe->context_refiner[index], pipe->gpu,
                                    error, error_size);
    for (uint32_t index = 0; ok && index < 30; index++)
        ok = qzen_zimage_block_upload(pipe->layers[index], pipe->gpu, error,
                                    error_size);
    if (!ok) {
        qzen_zimage_gpu_free(pipe->gpu);
        pipe->gpu = NULL;
        return 0;
    }
    return 1;
}

qzen_zpipe_config qzen_zpipe_default_config(void) {
    qzen_zpipe_config config;
    config.steps = 8;
    config.shift = 3.0f;
    config.seq_multiple = 32;
    return config;
}

qzen_zimage_pipeline *qzen_zpipe_load(const char *dit_dir,
                                  const qzen_zpipe_config *config,
                                  char *error, size_t error_size) {
    qzen_weight_store *store = qzen_weight_store_open(dit_dir, error, error_size);
    if (!store) return NULL;
    qzen_zimage_pipeline *pipe = calloc(1, sizeof(qzen_zimage_pipeline));
    if (!pipe) {
        qzen_weight_store_free(store);
        zpipe_fail(error, error_size, "cannot allocate qzen_zimage_pipeline");
        return NULL;
    }
    pipe->config = config ? *config : qzen_zpipe_default_config();
    pipe->dit_config = qzen_zimage_default_config();
    const qzen_zimage_config *dit = &pipe->dit_config;
    int ok = 1;
    double load_trace_t0 = zpipe_now();

    if (ok) ok = qzen_zimage_t_embedder_load(store, dit, &pipe->t_embedder, error, error_size);
    if (ok) ok = qzen_zimage_linear_load(store, "all_x_embedder.2-1.weight", dit->dim,
                                       64, 1, &pipe->x_embedder, error, error_size);
    if (ok) ok = qzen_zimage_rms_load(store, "cap_embedder.0.weight", 2560, 1e-5f,
                                    &pipe->cap_norm, error, error_size);
    if (ok) ok = qzen_zimage_linear_load(store, "cap_embedder.1.weight", dit->dim,
                                       2560, 1, &pipe->cap_linear, error, error_size);
    if (ok) {
        pipe->x_pad_token = malloc(dit->dim * sizeof(float));
        pipe->cap_pad_token = malloc(dit->dim * sizeof(float));
        if (!pipe->x_pad_token || !pipe->cap_pad_token) {
            zpipe_fail(error, error_size, "cannot allocate pad tokens");
            ok = 0;
        }
    }
    if (ok) ok = zpipe_read_bf16_f32(store, "x_pad_token", dit->dim, 1,
                                     pipe->x_pad_token, error, error_size);
    if (ok) ok = zpipe_read_bf16_f32(store, "cap_pad_token", dit->dim, 1,
                                     pipe->cap_pad_token, error, error_size);
    for (uint32_t index = 0; ok && index < 2; index++) {
        pipe->noise_refiner[index] = qzen_zimage_block_load(store, "noise_refiner", index,
                                                          dit, error, error_size);
        ok = pipe->noise_refiner[index] != NULL;
    }
    for (uint32_t index = 0; ok && index < 2; index++) {
        pipe->context_refiner[index] = qzen_zimage_block_load(store, "context_refiner", index,
                                                            dit, error, error_size);
        ok = pipe->context_refiner[index] != NULL;
    }
    for (uint32_t index = 0; ok && index < 30; index++) {
        pipe->layers[index] = qzen_zimage_block_load(store, "layers", index,
                                                   dit, error, error_size);
        ok = pipe->layers[index] != NULL;
    }
    if (ok) ok = qzen_zimage_final_layer_load(store, dit, 64, &pipe->final_layer,
                                            error, error_size);
    qzen_weight_store_free(store);
    double load_trace_t1 = zpipe_now();
    if (!zpipe_upload_gpu(pipe, error, error_size)) {
        qzen_zpipe_free(pipe);
        return NULL;
    }
    double load_trace_t2 = zpipe_now();
    if (getenv("ZPIPE_TRACE"))
        fprintf(stderr,
                "load phases: weights (read+dequant) %.2f s, GPU upload %.2f s\n",
                load_trace_t1 - load_trace_t0, load_trace_t2 - load_trace_t1);
    if (getenv("ZPIPE_TRACE")) {
        fprintf(stderr, "post-load x_embedder W[0..2]=%.6f %.6f %.6f bias[0]=%.6f\n",
                zpipe_bf16_to_f32(pipe->x_embedder.weight[0]),
                zpipe_bf16_to_f32(pipe->x_embedder.weight[1]),
                zpipe_bf16_to_f32(pipe->x_embedder.weight[2]),
                pipe->x_embedder.bias[0]);
        fprintf(stderr, "DiT GPU path: %s\n",
                pipe->gpu ? "enabled" : "disabled (CPU reference)");
    }
    if (!ok) {
        qzen_zpipe_free(pipe);
        return NULL;
    }
    return pipe;
}

void qzen_zpipe_free(qzen_zimage_pipeline *pipe) {
    if (!pipe) return;
    qzen_zpipe_release_text(pipe);
    qzen_zimage_t_embedder_free(&pipe->t_embedder);
    qzen_zimage_linear_free(&pipe->x_embedder);
    qzen_zimage_rms_free(&pipe->cap_norm);
    qzen_zimage_linear_free(&pipe->cap_linear);
    free(pipe->x_pad_token);
    free(pipe->cap_pad_token);
    for (uint32_t index = 0; index < 2; index++) {
        qzen_zimage_block_free(pipe->noise_refiner[index]);
        qzen_zimage_block_free(pipe->context_refiner[index]);
    }
    for (uint32_t index = 0; index < 30; index++) {
        qzen_zimage_block_free(pipe->layers[index]);
    }
    qzen_zimage_final_layer_free(&pipe->final_layer);
    if (pipe->gpu) qzen_zimage_gpu_free(pipe->gpu);
    /* The CPU fallback's scratch is process-wide, so it outlives the blocks
     * that filled it; give it back with the last of them (PLAN §21). */
    qzen_zimage_cpu_scratch_release();
    free(pipe);
}

/* --- text -------------------------------------------------------------------- */

int qzen_zpipe_attach_text(qzen_zimage_pipeline *pipe, const char *qwen_dir,
                         char *error, size_t error_size) {
    if (pipe->text_model) return 1;
    char tokenizer_path[512];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", qwen_dir);
    pipe->tokenizer = qzen_tokenizer_load(tokenizer_path, error, error_size);
    if (!pipe->tokenizer) return 0;
    pipe->text_config = qzen_ztext_default_config();
    qzen_weight_store *store = qzen_weight_store_open(qwen_dir, error, error_size);
    if (!store) return 0;
    pipe->text_model = qzen_ztext_model_load(store, &pipe->text_config, NULL,
                                           error, error_size);
    qzen_weight_store_free(store);
    return pipe->text_model != NULL;
}

void qzen_zpipe_release_text(qzen_zimage_pipeline *pipe) {
    if (pipe->text_model) {
        qzen_ztext_model_free(pipe->text_model);
        pipe->text_model = NULL;
    }
    if (pipe->tokenizer) {
        qzen_tokenizer_free(pipe->tokenizer);
        pipe->tokenizer = NULL;
    }
}

/* The text encoder comes from zimage_dir, not from the DiT file, so the two
 * Nunchaku tiers share one instance.  Freeing the pipeline to change tiers
 * would drop it and pay 71-79 s to read and upload it again -- see
 * serve_ensure_model, which detaches before the free and adopts after the load. */
int qzen_zpipe_detach_text(qzen_zimage_pipeline *pipe, qzen_ztext_model **model,
                         qzen_tokenizer **tokenizer, qzen_ztext_config *config) {
    if (!pipe || !pipe->text_model) return 0;
    if (model) *model = pipe->text_model;
    if (tokenizer) *tokenizer = pipe->tokenizer;
    if (config) *config = pipe->text_config;
    pipe->text_model = NULL;
    pipe->tokenizer = NULL;
    memset(&pipe->text_config, 0, sizeof(pipe->text_config));
    return 1;
}

void qzen_zpipe_adopt_text(qzen_zimage_pipeline *pipe, qzen_ztext_model *model,
                         qzen_tokenizer *tokenizer,
                         const qzen_ztext_config *config) {
    if (!pipe || !model) return;
    qzen_zpipe_release_text(pipe); /* a pipeline holds at most one encoder */
    pipe->text_model = model;
    pipe->tokenizer = tokenizer;
    if (config) pipe->text_config = *config;
}

void qzen_zpipe_release_detached_text(qzen_ztext_model *model,
                                    qzen_tokenizer *tokenizer) {
    if (model) qzen_ztext_model_free(model);
    if (tokenizer) qzen_tokenizer_free(tokenizer);
}

int qzen_zpipe_encode_text(qzen_zimage_pipeline *pipe, const char *prompt,
                         float **cap_feats, uint32_t *cap_len,
                         char *error, size_t error_size) {
    if (!pipe->text_model) {
        zpipe_fail(error, error_size, "text encoder not attached");
        return 0;
    }
    uint32_t max_length = pipe->text_config.max_length;
    int32_t *ids = malloc(max_length * sizeof(int32_t));
    uint8_t *mask = malloc(max_length);
    uint32_t real_len = 0;
    if (!ids || !mask) {
        free(ids); free(mask);
        zpipe_fail(error, error_size, "cannot allocate token buffers");
        return 0;
    }
    if (!qzen_ztext_build_inputs(&pipe->text_config, pipe->tokenizer, prompt, ids, mask,
                               &real_len, error, error_size)) {
        free(ids); free(mask);
        return 0;
    }
    if (real_len == 0 || real_len > max_length) {
        free(ids); free(mask);
        zpipe_fail(error, error_size, "the prompt produced %u tokens", real_len);
        return 0;
    }
    /* The tokenizer right-pads to max_length, but Qwen3's attention is causal
     * over positions <= p, so a padded row can never reach a real one.  Running
     * the forward over real_len positions therefore returns the same numbers
     * for the rows we consume -- and costs a fraction of the work.  Measured
     * on the host path (§22): 512 positions took 46 s for a 13-token prompt,
     * 13 positions take 1.2 s, and the cap features are bit-identical. */
    float *features = malloc((size_t)real_len * pipe->text_config.dim * sizeof(float));
    if (!features) {
        free(ids); free(mask);
        zpipe_fail(error, error_size, "cannot allocate cap features");
        return 0;
    }
    if (!qzen_ztext_forward(pipe->text_model, ids, real_len, 0, features, error,
                          error_size)) {
        free(ids); free(mask); free(features);
        return 0;
    }
    free(ids); free(mask);
    *cap_feats = features;
    *cap_len = real_len;
    return 1;
}

/* --- RNG ---------------------------------------------------------------------- */

static uint64_t zpipe_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t zpipe_rotl(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

/* One xoshiro256++ draw; state is updated in place. */
static uint64_t zpipe_next(uint64_t s[4]) {
    uint64_t result = zpipe_rotl(s[0] + s[3], 23) + s[3];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = zpipe_rotl(s[3], 45);
    return result;
}

/* Uniform in (0, 1) with 53-bit precision. */
static double zpipe_uniform(uint64_t s[4]) {
    return ((double)(zpipe_next(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

void qzen_zpipe_init_latents(uint64_t seed, uint32_t channels, uint32_t lh,
                           uint32_t lw, float *latents) {
    uint64_t sm = seed ^ 0x8000000000000001ULL;
    uint64_t s[4];
    for (uint32_t index = 0; index < 4; index++) s[index] = zpipe_splitmix64(&sm);
    size_t count = (size_t)channels * lh * lw;
    int have_spare = 0;
    float spare = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float value;
        if (have_spare) {
            value = spare;
            have_spare = 0;
        } else {
            /* Box-Muller: two uniforms -> two standard normals. */
            double u1 = zpipe_uniform(s);
            double u2 = zpipe_uniform(s);
            double radius = sqrt(-2.0 * log(u1));
            double angle = 2.0 * 3.14159265358979323846 * u2;
            value = (float)(radius * cos(angle));
            spare = (float)(radius * sin(angle));
            have_spare = 1;
        }
        latents[index] = value;
    }
}
/* --- DiT forward -------------------------------------------------------------- */

void qzen_zpipe_sigmas(uint32_t steps, float shift, float *sigmas) {
    /* Descending: sigma((steps-k)/steps) for k = 0..steps-1, then 0.
     * sigma(u) = shift * u / (1 + (shift - 1) * u). */
    for (uint32_t k = 0; k < steps; k++) {
        float u = (float)(steps - k) / (float)steps;
        sigmas[k] = shift * u / (1.0f + (shift - 1.0f) * u);
    }
    sigmas[steps] = 0.0f;
}

/* Host-side preamble shared by the CPU and GPU paths: token geometry,
 * patchify, RoPE ids/tables and the timestep embedding. */
typedef struct {
    uint32_t dim;
    uint32_t h_tokens, w_tokens;
    uint32_t x_real, x_pad, seq_x;
    uint32_t cap_pad, seq_cap, seq;
    uint32_t half;
    uint32_t latent_h, latent_w;
    float *patches;  /* [seq_x, 64] */
    int32_t *ids;    /* [seq, 3] */
    float *rope_cos; /* [seq, half] */
    float *rope_sin;
    float *t_freqs; /* [256] */
    float *adaln;   /* [256] */
} zpipe_prep;

static int zpipe_prepare(const qzen_zimage_pipeline *pipe, const float *latents,
                         uint32_t latent_h, uint32_t latent_w,
                         uint32_t cap_len, float tau, zpipe_prep *prep,
                         char *error, size_t error_size) {
    memset(prep, 0, sizeof(*prep));
    const qzen_zimage_config *dit = &pipe->dit_config;
    prep->dim = dit->dim;
    prep->half = dit->head_dim / 2;
    prep->latent_h = latent_h;
    prep->latent_w = latent_w;
    prep->h_tokens = latent_h / 2;
    prep->w_tokens = latent_w / 2;
    prep->x_real = prep->h_tokens * prep->w_tokens;
    prep->x_pad = (pipe->config.seq_multiple -
                   prep->x_real % pipe->config.seq_multiple) %
                  pipe->config.seq_multiple;
    prep->seq_x = prep->x_real + prep->x_pad;
    prep->cap_pad = (pipe->config.seq_multiple -
                     cap_len % pipe->config.seq_multiple) %
                    pipe->config.seq_multiple;
    prep->seq_cap = cap_len + prep->cap_pad;
    prep->seq = prep->seq_x + prep->seq_cap;
    uint32_t seq = prep->seq;
    uint32_t seq_x = prep->seq_x;
    uint32_t seq_cap = prep->seq_cap;
    uint32_t x_real = prep->x_real;
    uint32_t half = prep->half;
    uint32_t h_tokens = prep->h_tokens, w_tokens = prep->w_tokens;

    prep->patches = malloc((size_t)seq_x * 64 * sizeof(float));
    prep->ids = malloc((size_t)seq * 3 * sizeof(int32_t));
    prep->rope_cos = malloc((size_t)seq * half * sizeof(float));
    prep->rope_sin = malloc((size_t)seq * half * sizeof(float));
    prep->t_freqs = malloc(256 * sizeof(float));
    prep->adaln = malloc(256 * sizeof(float));
    if (!prep->patches || !prep->ids || !prep->rope_cos || !prep->rope_sin ||
        !prep->t_freqs || !prep->adaln) {
        zpipe_fail(error, error_size, "cannot allocate DiT forward buffers");
        return 0;
    }

    /* patch features [x_real, 64]: token = (h * w_tokens + w), feature =
     * (ph * 2 + pw) * 16 + c, matching "(f h w) (pf ph pw c)". */
    for (uint32_t h = 0; h < h_tokens; h++) {
        for (uint32_t w = 0; w < w_tokens; w++) {
            float *row = prep->patches + ((size_t)h * w_tokens + w) * 64;
            for (uint32_t ph = 0; ph < 2; ph++) {
                for (uint32_t pw = 0; pw < 2; pw++) {
                    const float *plane_base =
                        latents + (size_t)((2 * h + ph) * latent_w + 2 * w + pw);
                    float *cell = row + (size_t)(ph * 2 + pw) * 16;
                    for (uint32_t c = 0; c < 16; c++) {
                        cell[c] = plane_base[(size_t)c * latent_h * latent_w];
                    }
                }
            }
        }
    }

    /* RoPE ids: caption axis0 = 1..seq_cap, image axis0 = seq_cap + 1 with
     * (h, w) on axes 1/2; pad rows (0, 0, 0). */
    for (uint32_t i = 0; i < seq_x; i++) {
        if (i < x_real) {
            uint32_t h = i / w_tokens, w = i % w_tokens;
            prep->ids[(size_t)i * 3 + 0] = (int32_t)(seq_cap + 1);
            prep->ids[(size_t)i * 3 + 1] = (int32_t)h;
            prep->ids[(size_t)i * 3 + 2] = (int32_t)w;
        } else {
            prep->ids[(size_t)i * 3 + 0] = 0;
            prep->ids[(size_t)i * 3 + 1] = 0;
            prep->ids[(size_t)i * 3 + 2] = 0;
        }
    }
    for (uint32_t i = 0; i < seq_cap; i++) {
        prep->ids[(size_t)(seq_x + i) * 3 + 0] = (int32_t)(i + 1);
        prep->ids[(size_t)(seq_x + i) * 3 + 1] = 0;
        prep->ids[(size_t)(seq_x + i) * 3 + 2] = 0;
    }
    qzen_zimage_rope_tables(dit, prep->ids, seq, prep->rope_cos, prep->rope_sin);

    /* timestep -> adaLN vector (stays a host MLP: two tiny Linears). */
    qzen_zimage_timestep_embedding(tau * dit->t_scale, 256, prep->t_freqs);
    qzen_zimage_t_embedder_forward(&pipe->t_embedder, prep->t_freqs, 1,
                                 prep->adaln);
    if (getenv("ZPIPE_TRACE")) {
        fprintf(stderr, "t_freq[0..2]=%.6f %.6f %.6f adaln[0..2]=%.6f %.6f %.6f\n",
                prep->t_freqs[0], prep->t_freqs[1], prep->t_freqs[2],
                prep->adaln[0], prep->adaln[1], prep->adaln[2]);
    }
    return 1;
}

static void zpipe_prep_free(zpipe_prep *prep) {
    free(prep->patches);
    free(prep->ids);
    free(prep->rope_cos);
    free(prep->rope_sin);
    free(prep->t_freqs);
    free(prep->adaln);
    memset(prep, 0, sizeof(*prep));
}

/* [rows, 64] patch features -> [16, latent_h, latent_w] velocity, real rows. */
static void zpipe_unpatchify(const float *branch, uint32_t x_real,
                             uint32_t w_tokens, uint32_t latent_h,
                             uint32_t latent_w, float *velocity) {
    for (uint32_t token = 0; token < x_real; token++) {
        uint32_t h = token / w_tokens, w = token % w_tokens;
        const float *row = branch + (size_t)token * 64;
        for (uint32_t ph = 0; ph < 2; ph++) {
            for (uint32_t pw = 0; pw < 2; pw++) {
                float *out =
                    velocity + (size_t)((2 * h + ph) * latent_w + 2 * w + pw);
                const float *cell = row + (size_t)(ph * 2 + pw) * 16;
                for (uint32_t c = 0; c < 16; c++) {
                    out[(size_t)c * latent_h * latent_w] = cell[c];
                }
            }
        }
    }
}

/* --- Z-Image Turbo Fun ControlNet Union 2.1 ------------------------------
 *
 * One denoise step precomputes every ControlNet skip connection on the host,
 * then the ordinary CPU/GPU DiT forward adds them with context_scale. This is
 * the WanGP v2.x path (prepare_forward_control_2_0_refiner -> dedicated
 * control noise refiners -> control_layers prime pass ->
 * prepare_forward_control_2_0_layers -> control_layers hint pass) specialised
 * for the single-sample, guidance_scale = 0 case the C engine implements.
 *
 * Running the hint path on the host keeps the CUDA change to one new kernel
 * (BF16 scaled add) and avoids uploading a second 7 GiB transformer when the
 * text encoder is still resident. When the ControlNet weights have been
 * uploaded, each control block is evaluated on the device through the
 * host-in/host-out block wrapper; otherwise the CPU reference forward runs.
 */

typedef struct {
    float *noise[QZEN_ZIMAGE_CONTROL_REFINERS]; /* [seq_x, dim] */
    float *main[QZEN_ZIMAGE_CONTROL_LAYERS];    /* [seq, dim] */
    uint32_t layers, refiners, seq_x, seq, dim;
    float scale;
} zpipe_control_hints;

static void zpipe_control_hints_free(zpipe_control_hints *hints) {
    if (!hints) return;
    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_REFINERS; i++)
        free(hints->noise[i]);
    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_LAYERS; i++)
        free(hints->main[i]);
    memset(hints, 0, sizeof(*hints));
}

static void zpipe_control_add(float *dst, const float *src, size_t count,
                              float alpha) {
    for (size_t i = 0; i < count; i++) dst[i] += alpha * src[i];
}

static int zpipe_control_block(const qzen_zimage_pipeline *pipe,
                               const qzen_zimage_block *block, int use_gpu,
                               float *x, uint32_t seq, const float *adaln,
                               const float *rope_cos, const float *rope_sin,
                               char *error, size_t error_size) {
    if (use_gpu && pipe->gpu)
        return qzen_zimage_block_forward_gpu_host(block, pipe->gpu, x, seq,
                                                  adaln, rope_cos, rope_sin,
                                                  error, error_size);
    return qzen_zimage_block_forward(block, x, seq, adaln, rope_cos, rope_sin,
                                     error, error_size);
}

/* out = linear(x), in place (scratch has rows * linear->out floats). */
static void zpipe_control_proj(const qzen_zimage_linear *linear, float *x,
                               uint32_t rows, float *scratch) {
    qzen_zimage_linear_forward(linear, x, rows, scratch);
    memcpy(x, scratch, (size_t)rows * linear->out * sizeof(float));
}

static int zpipe_control_hints_compute(const qzen_zimage_pipeline *pipe,
    const zpipe_prep *prep, const float *cap_feats, uint32_t cap_len,
    qzen_zimage_control *control, const float *control_context,
    float context_scale, zpipe_control_hints *hints,
    char *error, size_t error_size) {
    memset(hints, 0, sizeof(*hints));
    const uint32_t dim = prep->dim, half = prep->half;
    const uint32_t seq = prep->seq, seq_x = prep->seq_x;
    const uint32_t seq_cap = prep->seq_cap;
    const uint32_t layers = qzen_zimage_control_layers(control);
    const uint32_t refiners = qzen_zimage_control_refiners(control);
    const uint32_t in_dim = qzen_zimage_control_in_dim(control);
    if (!layers || layers > QZEN_ZIMAGE_CONTROL_LAYERS || !refiners ||
        refiners > QZEN_ZIMAGE_CONTROL_REFINERS ||
        (in_dim != QZEN_ZIMAGE_CONTROL_IN_DIM_V2)) {
        zpipe_fail(error, error_size,
                   "unsupported ControlNet Union variant: in_dim=%u, %u layers, "
                   "%u dedicated noise refiners (expected v2.x: 33/15/2)",
                   in_dim, layers, refiners);
        return 0;
    }
    if (!control_context || !cap_feats || !isfinite(context_scale)) {
        zpipe_fail(error, error_size, "invalid ControlNet context or scale");
        return 0;
    }

    const int ctrl_gpu = pipe->gpu && qzen_zimage_control_uploaded(control);
    const int main_gpu = pipe->gpu != NULL;
    const size_t x_count = (size_t)seq_x * dim;
    const size_t full_count = (size_t)seq * dim;
    const uint32_t patch_features = in_dim * 4;

    float *ctrl_patches = calloc((size_t)seq_x * patch_features, sizeof(float));
    float *c = malloc(full_count * sizeof(float));
    float *x_init = malloc(x_count * sizeof(float));
    float *x_nr = malloc(x_count * sizeof(float));
    float *cap_buf = malloc((size_t)seq_cap * dim * sizeof(float));
    float *normed = malloc((size_t)(cap_len ? cap_len : 1) * 2560 * sizeof(float));
    float *scratch = malloc(full_count * sizeof(float));
    float *x_unified = malloc(full_count * sizeof(float));
    if (!ctrl_patches || !c || !x_init || !x_nr || !cap_buf || !normed ||
        !scratch || !x_unified) {
        free(ctrl_patches); free(c); free(x_init); free(x_nr);
        free(cap_buf); free(normed); free(scratch); free(x_unified);
        zpipe_fail(error, error_size, "cannot allocate ControlNet hint buffers");
        return 0;
    }

    /* Patchify the control context exactly like the main latents, but with
     * in_dim planes per 2x2 cell: token = (h * w_tokens + w), feature =
     * (ph * 2 + pw) * in_dim + c. Padded rows stay zero until after the
     * embedder, when they are replaced by the main DiT's x_pad_token. */
    const uint32_t lh = prep->latent_h, lw = prep->latent_w;
    for (uint32_t h = 0; h < prep->h_tokens; h++) {
        for (uint32_t w = 0; w < prep->w_tokens; w++) {
            float *row = ctrl_patches +
                         ((size_t)h * prep->w_tokens + w) * patch_features;
            for (uint32_t ph = 0; ph < 2; ph++) {
                for (uint32_t pw = 0; pw < 2; pw++) {
                    const size_t pixel =
                        (size_t)(2 * h + ph) * lw + 2 * w + pw;
                    float *cell = row + ((size_t)(ph * 2 + pw) * in_dim);
                    for (uint32_t ch = 0; ch < in_dim; ch++)
                        cell[ch] = control_context[(size_t)ch * lh * lw + pixel];
                }
            }
        }
    }

    /* Only the real patch rows carry data; the padded rows are overwritten
     * with x_pad_token below, so never read the uninitialised tail. */
    qzen_zimage_linear_forward(qzen_zimage_control_embedder(control),
                               ctrl_patches, prep->x_real, c);
    qzen_zimage_linear_forward(&pipe->x_embedder, prep->patches,
                               prep->x_real, x_init);
    for (uint32_t i = prep->x_real; i < seq_x; i++) {
        memcpy(c + (size_t)i * dim, pipe->x_pad_token, dim * sizeof(float));
        memcpy(x_init + (size_t)i * dim, pipe->x_pad_token,
               dim * sizeof(float));
    }
    free(ctrl_patches);
    ctrl_patches = NULL;

    /* Context-refined caption rows. WanGP uses cap_embedded_list[0] *after*
     * the context refiners when it assembles the main-layer hint sequence. */
    qzen_zimage_rms_forward(&pipe->cap_norm, cap_feats, cap_len, normed);
    qzen_zimage_linear_forward(&pipe->cap_linear, normed, cap_len, cap_buf);
    for (uint32_t i = cap_len; i < seq_cap; i++)
        memcpy(cap_buf + (size_t)i * dim, pipe->cap_pad_token,
               dim * sizeof(float));
    free(normed);
    normed = NULL;
    for (uint32_t index = 0; index < 2; index++) {
        if (!zpipe_control_block(pipe, pipe->context_refiner[index], main_gpu,
                                 cap_buf, seq_cap, NULL,
                                 prep->rope_cos + (size_t)seq_x * half,
                                 prep->rope_sin + (size_t)seq_x * half,
                                 error, error_size))
            goto fail;
    }

    /* Dedicated control noise refiners: before_proj(c) + x at block 0, one
     * control block, then after_proj(c) becomes this layer's skip. */
    for (uint32_t r = 0; r < refiners; r++) {
        const qzen_zimage_linear *before =
            qzen_zimage_control_before_proj(control, 1, r);
        if (before) {
            zpipe_control_proj(before, c, seq_x, scratch);
            zpipe_control_add(c, x_init, x_count, 1.0f);
        }
        if (!zpipe_control_block(pipe,
                                 qzen_zimage_control_refiner(control, r),
                                 ctrl_gpu, c, seq_x, prep->adaln,
                                 prep->rope_cos, prep->rope_sin,
                                 error, error_size))
            goto fail;
        float *skip = malloc(x_count * sizeof(float));
        if (!skip) {
            zpipe_fail(error, error_size, "cannot allocate ControlNet noise hint");
            goto fail;
        }
        qzen_zimage_linear_forward(
            qzen_zimage_control_after_proj(control, 1, r), c, seq_x, skip);
        hints->noise[r] = skip;
    }

    /* The main noise refiners, with their skips applied, form the `x` that
     * control_layers.0 adds to the main-phase hint sequence. */
    memcpy(x_nr, x_init, x_count * sizeof(float));
    for (uint32_t index = 0; index < 2 && index < refiners; index++) {
        if (!zpipe_control_block(pipe, pipe->noise_refiner[index], main_gpu,
                                 x_nr, seq_x, prep->adaln,
                                 prep->rope_cos, prep->rope_sin,
                                 error, error_size))
            goto fail;
        zpipe_control_add(x_nr, hints->noise[index], x_count, context_scale);
    }

    /* v2.x "finish processing" pass: advance the hint state through all
     * control layers on the image section before the main layers run. */
    for (uint32_t j = 0; j < layers; j++) {
        const qzen_zimage_linear *before =
            qzen_zimage_control_before_proj(control, 0, j);
        if (before) {
            zpipe_control_proj(before, c, seq_x, scratch);
            zpipe_control_add(c, x_init, x_count, 1.0f);
        }
        if (!zpipe_control_block(pipe, qzen_zimage_control_layer(control, j),
                                 ctrl_gpu, c, seq_x, prep->adaln,
                                 prep->rope_cos, prep->rope_sin,
                                 error, error_size))
            goto fail;
    }

    /* Main-phase hints: [control context | refined caption] over the unified
     * sequence, with the noise-refined image + caption as the `x` added by
     * control_layers.0. Every layer produces one skip for main layer 2*j. */
    memcpy(c + x_count, cap_buf, (size_t)seq_cap * dim * sizeof(float));
    memcpy(x_unified, x_nr, x_count * sizeof(float));
    memcpy(x_unified + x_count, cap_buf,
           (size_t)seq_cap * dim * sizeof(float));
    for (uint32_t j = 0; j < layers; j++) {
        const qzen_zimage_linear *before =
            qzen_zimage_control_before_proj(control, 0, j);
        if (before) {
            zpipe_control_proj(before, c, seq, scratch);
            zpipe_control_add(c, x_unified, full_count, 1.0f);
        }
        if (!zpipe_control_block(pipe, qzen_zimage_control_layer(control, j),
                                 ctrl_gpu, c, seq, prep->adaln,
                                 prep->rope_cos, prep->rope_sin,
                                 error, error_size))
            goto fail;
        float *skip = malloc(full_count * sizeof(float));
        if (!skip) {
            zpipe_fail(error, error_size, "cannot allocate ControlNet layer hint");
            goto fail;
        }
        qzen_zimage_linear_forward(
            qzen_zimage_control_after_proj(control, 0, j), c, seq, skip);
        hints->main[j] = skip;
    }

    hints->layers = layers;
    hints->refiners = refiners;
    hints->seq_x = seq_x;
    hints->seq = seq;
    hints->dim = dim;
    hints->scale = context_scale;
    free(c); free(x_init); free(x_nr); free(cap_buf);
    free(scratch); free(x_unified);
    return 1;

fail:
    free(ctrl_patches);
    free(c); free(x_init); free(x_nr); free(cap_buf);
    free(normed); free(scratch); free(x_unified);
    zpipe_control_hints_free(hints);
    return 0;
}

float *qzen_zpipe_control_context_ex(const qzen_zimage_control *control,
                                     const float *control_latent,
                                     const float *keep_mask,
                                     const float *inpaint_latent,
                                     uint32_t latent_h, uint32_t latent_w,
                                     char *error, size_t error_size) {
    const uint32_t in_dim = qzen_zimage_control_in_dim(control);
    const int has_inpaint = (keep_mask != NULL) || (inpaint_latent != NULL);
    if (!control_latent || !latent_h || !latent_w ||
        (in_dim != QZEN_ZIMAGE_CONTROL_IN_DIM_V2 &&
         in_dim != QZEN_ZIMAGE_CONTROL_IN_DIM_V1)) {
        zpipe_fail(error, error_size, "invalid ControlNet control latent");
        return NULL;
    }
    if (has_inpaint && (!keep_mask || !inpaint_latent)) {
        zpipe_fail(error, error_size,
                   "inpaint needs both keep_mask and inpaint_latent");
        return NULL;
    }
    if (has_inpaint && in_dim != QZEN_ZIMAGE_CONTROL_IN_DIM_V2) {
        zpipe_fail(error, error_size,
                   "inpainting requires 33-channel Union 2.x weights "
                   "(16-channel v1 packs cannot inpaint)");
        return NULL;
    }
    const size_t plane = (size_t)latent_h * latent_w;
    float *context = calloc((size_t)in_dim * plane, sizeof(float));
    if (!context) {
        zpipe_fail(error, error_size, "cannot allocate ControlNet context");
        return NULL;
    }
    /* [control_latent(16) | keep_mask(1) | inpaint_latent(16)] -- WanGP
     * models/z_image/pipeline_z_image.py, control_in_dim == 33 branch.
     * Without a mask the last 17 planes stay at zero, exactly as before. */
    memcpy(context, control_latent, 16 * plane * sizeof(float));
    if (has_inpaint) {
        memcpy(context + 16 * plane, keep_mask, plane * sizeof(float));
        memcpy(context + 17 * plane, inpaint_latent,
               16 * plane * sizeof(float));
    }
    return context;
}

float *qzen_zpipe_control_context(const qzen_zimage_control *control,
                                  const float *control_latent,
                                  uint32_t latent_h, uint32_t latent_w,
                                  char *error, size_t error_size) {
    return qzen_zpipe_control_context_ex(control, control_latent, NULL, NULL,
                                         latent_h, latent_w, error, error_size);
}

int qzen_zpipe_control_prepare(qzen_zimage_pipeline *pipe,
                               qzen_zimage_control *control,
                               char *error, size_t error_size) {
    if (!pipe || !control) {
        zpipe_fail(error, error_size, "control prepare needs a pipeline and weights");
        return 0;
    }
    if (!pipe->gpu) return 1; /* CPU hint path needs no uploads */
    char local[512];
    if (!qzen_zimage_control_upload(control, pipe->gpu, local, sizeof(local))) {
        /* Device upload is an optimisation, not a requirement. */
        if (getenv("ZPIPE_TRACE"))
            fprintf(stderr, "ControlNet GPU upload unavailable (%s); "
                            "using the CPU hint path\n", local);
        if (error && error_size) error[0] = '\0';
    }
    return 1;
}

/* --- Hint injection ----------------------------------------------------------
 *
 * WanGP's BaseZImageTransformerBlock.forward adds each ControlNet skip AFTER
 * its own attention / feed-forward pass, with alpha = control_context_scale.
 * control_layers_places is [0,2,...,28] for Union 2.x, so only even main
 * layers receive a hint; the two dedicated control noise refiners map 1:1 to
 * the main DiT's noise_refiner.0/1. A NULL `hints` leaves the ordinary t2i
 * forward bit-for-bit unchanged (qzen_zpipe_denoise() always passes NULL).
 */
static uint16_t zpipe_f32_to_bf16(float value);

static const float *zpipe_hint_for(const zpipe_control_hints *hints,
                                   int main_phase, uint32_t index,
                                   size_t *count, float *scale) {
    if (!hints) return NULL;
    if (main_phase) {
        if ((index & 1u) != 0u) return NULL;
        uint32_t j = index / 2u;
        if (j >= hints->layers || !hints->main[j]) return NULL;
        *count = (size_t)hints->seq * hints->dim;
        *scale = hints->scale;
        return hints->main[j];
    }
    if (index >= hints->refiners || !hints->noise[index]) return NULL;
    *count = (size_t)hints->seq_x * hints->dim;
    *scale = hints->scale;
    return hints->noise[index];
}

static void zpipe_apply_hint_cpu(float *x, const zpipe_control_hints *hints,
                                 int main_phase, uint32_t index) {
    size_t count;
    float scale;
    const float *hint = zpipe_hint_for(hints, main_phase, index, &count, &scale);
    if (hint) zpipe_control_add(x, hint, count, scale);
}

/* `scratch` / `staged` are allocated lazily on the first injection with
 * `capacity` elements, which must cover the largest hint (the unified
 * sequence). Every early return in the GPU forward happens before the first
 * injection, so those buffers never leak on an error path. */
static int zpipe_apply_hint_gpu(const qzen_zimage_pipeline *pipe,
                                qzen_gpu_tensor *dst,
                                qzen_gpu_tensor **scratch, uint16_t **staged,
                                size_t capacity,
                                const zpipe_control_hints *hints,
                                int main_phase, uint32_t index,
                                char *error, size_t error_size) {
    size_t count;
    float scale;
    const float *hint = zpipe_hint_for(hints, main_phase, index, &count, &scale);
    if (!hint) return 1;
    qzen_gpu *gpu = qzen_zimage_gpu_handle(pipe->gpu);
    if (!*scratch) {
        *scratch = qzen_gpu_tensor_new_bf16(gpu, capacity);
        *staged = (uint16_t *)malloc(capacity * sizeof(uint16_t));
    }
    if (!*scratch || !*staged || count > capacity) {
        zpipe_fail(error, error_size,
                   "cannot allocate the ControlNet hint GPU buffer");
        return 0;
    }
    for (size_t i = 0; i < count; i++)
        (*staged)[i] = zpipe_f32_to_bf16(hint[i]);
    if (!qzen_gpu_tensor_write_bf16_range(*scratch, 0, *staged, count) ||
        !qzen_gpu_scale_add_bf16(gpu, dst, dst, *scratch,
                                 (uint32_t)count, scale)) {
        zpipe_fail(error, error_size, "ControlNet hint injection failed: %s",
                   qzen_gpu_error(gpu));
        return 0;
    }
    return 1;
}

/* --- CPU reference path ------------------------------------------------------ */

static int zpipe_dit_forward_cpu(const qzen_zimage_pipeline *pipe,
                                 const zpipe_prep *prep,
                                 const float *cap_feats, uint32_t cap_len,
                                 const zpipe_control_hints *hints,
                                 float *velocity, char *error,
                                 size_t error_size) {
    const uint32_t dim = prep->dim;
    const uint32_t seq = prep->seq, seq_x = prep->seq_x, seq_cap = prep->seq_cap;
    const uint32_t x_real = prep->x_real;
    const uint32_t half = prep->half;
    const float *adaln = prep->adaln;
    const float *rope_cos = prep->rope_cos;
    const float *rope_sin = prep->rope_sin;
    const float *patches = prep->patches;

    float *x = malloc((size_t)seq * dim * sizeof(float));
    float *cap = malloc((size_t)seq_cap * dim * sizeof(float));
    float *normed_cap = malloc((size_t)cap_len * 2560 * sizeof(float));
    float *branch = malloc((size_t)seq * 64 * sizeof(float));
    if (!x || !cap || !normed_cap || !branch) {
        zpipe_fail(error, error_size, "cannot allocate DiT forward buffers");
        free(x); free(cap); free(normed_cap); free(branch);
        return 0;
    }

    /* embeddings. */
    qzen_zimage_linear_forward(&pipe->x_embedder, patches, x_real, x);
    for (uint32_t i = x_real; i < seq_x; i++) {
        memcpy(x + (size_t)i * dim, pipe->x_pad_token, dim * sizeof(float));
    }
    if (cap_len) {
        qzen_zimage_rms_forward(&pipe->cap_norm, cap_feats, cap_len, normed_cap);
        qzen_zimage_linear_forward(&pipe->cap_linear, normed_cap, cap_len, cap);
    }
    for (uint32_t i = cap_len; i < seq_cap; i++) {
        memcpy(cap + (size_t)i * dim, pipe->cap_pad_token, dim * sizeof(float));
    }

    /* refiners. */
    for (uint32_t index = 0; index < 2; index++) {
        if (!qzen_zimage_block_forward(pipe->noise_refiner[index], x, seq_x, adaln,
                                     rope_cos, rope_sin, error, error_size)) {
            goto fail;
        }
        zpipe_apply_hint_cpu(x, hints, 0, index);
    }
    for (uint32_t index = 0; index < 2; index++) {
        if (!qzen_zimage_block_forward(pipe->context_refiner[index], cap, seq_cap,
                                     NULL, rope_cos + (size_t)seq_x * half,
                                     rope_sin + (size_t)seq_x * half, error,
                                     error_size)) {
            goto fail;
        }
    }

    /* unified [x | cap]. */
    memcpy(x + (size_t)seq_x * dim, cap, (size_t)seq_cap * dim * sizeof(float));
    for (uint32_t index = 0; index < 30; index++) {
        if (!qzen_zimage_block_forward(pipe->layers[index], x, seq, adaln,
                                     rope_cos, rope_sin, error, error_size)) {
            goto fail;
        }
        zpipe_apply_hint_cpu(x, hints, 1, index);
    }

    /* final layer + unpatchify over the real image rows. */
    qzen_zimage_final_layer_forward(&pipe->final_layer, x, seq, adaln, branch);
    zpipe_unpatchify(branch, x_real, prep->w_tokens, prep->latent_h,
                     prep->latent_w, velocity);
    free(x); free(cap); free(normed_cap); free(branch);
    return 1;

fail:
    free(x); free(cap); free(normed_cap); free(branch);
    return 0;
}

/* --- GPU path ----------------------------------------------------------------- */

static uint16_t zpipe_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int zpipe_dit_forward_gpu(const qzen_zimage_pipeline *pipe,
                                 const zpipe_prep *prep,
                                 const float *cap_feats, uint32_t cap_len,
                                 const zpipe_control_hints *hints,
                                 float *velocity, char *error,
                                 size_t error_size) {
    qzen_zimage_gpu *zgu = pipe->gpu;
    /* Lazily allocated on the first hint injection; see zpipe_apply_hint_gpu. */
    qzen_gpu_tensor *hint_scratch = NULL;
    uint16_t *hint_staged = NULL;
    qzen_gpu *gpu = qzen_zimage_gpu_handle(zgu);
    const uint32_t dim = prep->dim, half = prep->half;
    const uint32_t seq = prep->seq, seq_x = prep->seq_x, seq_cap = prep->seq_cap;
    const uint32_t x_real = prep->x_real, x_pad = prep->x_pad;
    const uint32_t cap_pad = prep->cap_pad;

    double trace_t0 = zpipe_now();
    int trace = zpipe_trace_timers();
    if (!qzen_zimage_gpu_reserve(zgu, seq, error, error_size)) return 0;
    if (!qzen_zimage_gpu_set_rope(zgu, prep->rope_cos, prep->rope_sin, seq, half,
                                error, error_size))
        return 0;
    /* The caption section runs on its own tensor, so its RoPE table must
     * start at the caption's first row (row seq_x of the unified sequence). */
    if (!qzen_gpu_tensor_write_f32_range(qzen_zimage_gpu_rope_cos_cap(zgu), 0,
                                       prep->rope_cos + (size_t)seq_x * half,
                                       (size_t)seq_cap * half) ||
        !qzen_gpu_tensor_write_f32_range(qzen_zimage_gpu_rope_sin_cap(zgu), 0,
                                       prep->rope_sin + (size_t)seq_x * half,
                                       (size_t)seq_cap * half)) {
        zpipe_fail(error, error_size, "cannot upload caption RoPE tables");
        return 0;
    }

    qzen_gpu_tensor *img = qzen_zimage_gpu_img(zgu);
    qzen_gpu_tensor *cap = qzen_zimage_gpu_cap(zgu);
    qzen_gpu_tensor *full = qzen_zimage_gpu_full(zgu);
    qzen_gpu_tensor *out64 = qzen_zimage_gpu_out64(zgu);
    qzen_gpu_tensor *rope_cos = qzen_zimage_gpu_rope_cos(zgu);
    qzen_gpu_tensor *rope_sin = qzen_zimage_gpu_rope_sin(zgu);
    qzen_gpu_tensor *rope_cos_cap = qzen_zimage_gpu_rope_cos_cap(zgu);
    qzen_gpu_tensor *rope_sin_cap = qzen_zimage_gpu_rope_sin_cap(zgu);
    qzen_gpu_tensor *modulation = qzen_zimage_gpu_modulation(zgu);

    /* Host staging: BF16 patch features, BF16 caption features and the pad
     * rows (x_pad rows of x_pad_token followed by cap_pad rows of
     * cap_pad_token). */
    uint16_t *patches_bf16 = malloc((size_t)x_real * 64 * 2);
    uint16_t *cap_bf16 = malloc((size_t)cap_len * 2560 * 2);
    uint16_t *pads = malloc((size_t)(x_pad + cap_pad) * dim * 2);
    float *branch = malloc((size_t)seq * 64 * sizeof(float));
    if (!patches_bf16 || !cap_bf16 || (!pads && (x_pad || cap_pad)) || !branch) {
        zpipe_fail(error, error_size, "cannot allocate GPU staging buffers");
        free(patches_bf16); free(cap_bf16); free(pads); free(branch);
        return 0;
    }
    for (size_t index = 0; index < (size_t)x_real * 64; index++)
        patches_bf16[index] = zpipe_f32_to_bf16(prep->patches[index]);
    for (size_t index = 0; index < (size_t)cap_len * 2560; index++)
        cap_bf16[index] = zpipe_f32_to_bf16(cap_feats[index]);
    for (uint32_t i = 0; i < x_pad; i++)
        for (uint32_t c = 0; c < dim; c++)
            pads[(size_t)i * dim + c] = zpipe_f32_to_bf16(pipe->x_pad_token[c]);
    for (uint32_t i = 0; i < cap_pad; i++)
        for (uint32_t c = 0; c < dim; c++)
            pads[(size_t)(x_pad + i) * dim + c] =
                zpipe_f32_to_bf16(pipe->cap_pad_token[c]);

    qzen_gpu_tensor *patches_gpu =
        qzen_gpu_tensor_from_bf16(gpu, patches_bf16, (size_t)x_real * 64);
    qzen_gpu_tensor *cap_feats_gpu =
        qzen_gpu_tensor_from_bf16(gpu, cap_bf16, (size_t)cap_len * 2560);
    free(patches_bf16);
    free(cap_bf16);
    if (!patches_gpu || !cap_feats_gpu) {
        qzen_gpu_tensor_free(patches_gpu);
        qzen_gpu_tensor_free(cap_feats_gpu);
        free(pads); free(branch);
        zpipe_fail(error, error_size, "cannot upload GPU inputs: %s",
                   qzen_gpu_error(gpu));
        return 0;
    }
    /* Pad rows never overlap the GEMM outputs, so they are written up front. */
    if (x_pad && !qzen_gpu_tensor_write_bf16_range(img, (size_t)x_real * dim,
                                                 pads, (size_t)x_pad * dim)) {
        qzen_gpu_tensor_free(patches_gpu);
        qzen_gpu_tensor_free(cap_feats_gpu);
        free(pads); free(branch);
        zpipe_fail(error, error_size, "cannot write image pad rows: %s",
                   qzen_gpu_error(gpu));
        return 0;
    }
    if (cap_pad &&
        !qzen_gpu_tensor_write_bf16_range(cap, (size_t)cap_len * dim,
                                        pads + (size_t)x_pad * dim,
                                        (size_t)cap_pad * dim)) {
        qzen_gpu_tensor_free(patches_gpu);
        qzen_gpu_tensor_free(cap_feats_gpu);
        free(pads); free(branch);
        zpipe_fail(error, error_size, "cannot write caption pad rows: %s",
                   qzen_gpu_error(gpu));
        return 0;
    }
    free(pads);

    double trace_t1 = zpipe_now();
    int ok = qzen_gpu_begin(gpu);
    /* image section: all_x_embedder.2-1 over the patch features. */
    ok = ok && qzen_gpu_linear_bf16(gpu, img, patches_gpu,
                                  pipe->x_embedder.gpu_weight,
                                  pipe->x_embedder.gpu_bias, x_real, 64, dim);
    qzen_gpu_tensor_free(patches_gpu);
    /* caption section: cap_embedder = RMSNorm(2560) + Linear(2560 -> dim).
     * The RMSNorm runs in place on the staged features. */
    ok = ok && qzen_gpu_rms_norm_bf16(gpu, cap_feats_gpu, cap_feats_gpu,
                                    pipe->cap_norm.gpu_weight, cap_len, 2560,
                                    pipe->cap_norm.eps);
    ok = ok && qzen_gpu_linear_bf16(gpu, cap, cap_feats_gpu,
                                  pipe->cap_linear.gpu_weight,
                                  pipe->cap_linear.gpu_bias, cap_len, 2560, dim);
    qzen_gpu_tensor_free(cap_feats_gpu);
    /* refiners. ControlNet noise hints are added right after the matching
     * main noise refiner, with alpha = control_context_scale. */
    for (uint32_t index = 0; index < 2; index++) {
        ok = ok && qzen_zimage_block_modulation_gpu(pipe->noise_refiner[index],
                                                  zgu, prep->adaln, error,
                                                  error_size) &&
             qzen_zimage_block_forward_gpu(pipe->noise_refiner[index], zgu, img,
                                         seq_x, modulation, rope_cos, rope_sin,
                                         error, error_size);
        if (ok && !zpipe_apply_hint_gpu(pipe, img, &hint_scratch, &hint_staged,
                                        (size_t)seq * dim, hints, 0, index,
                                        error, error_size))
            ok = 0;
    }
    for (uint32_t index = 0; index < 2; index++) {
        ok = ok && qzen_zimage_block_forward_gpu(pipe->context_refiner[index], zgu,
                                               cap, seq_cap, NULL, rope_cos_cap,
                                               rope_sin_cap, error, error_size);
    }
    /* unified [x | cap]. */
    double trace_t2 = zpipe_now();
    ok = ok && qzen_gpu_copy_bf16(gpu, full, 0, img, 0, (size_t)seq_x * dim) &&
         qzen_gpu_copy_bf16(gpu, full, (size_t)seq_x * dim, cap, 0,
                          (size_t)seq_cap * dim);
    double trace_t3 = zpipe_now();
    for (uint32_t index = 0; index < 30; index++) {
        ok = ok && qzen_zimage_block_modulation_gpu(pipe->layers[index], zgu,
                                                  prep->adaln, error,
                                                  error_size) &&
             qzen_zimage_block_forward_gpu(pipe->layers[index], zgu, full, seq,
                                         modulation, rope_cos, rope_sin, error,
                                         error_size);
        /* Union 2.x control_layers_places = [0,2,...,28]: one skip per even
         * main layer, added after that layer's forward. */
        if (ok && !zpipe_apply_hint_gpu(pipe, full, &hint_scratch, &hint_staged,
                                        (size_t)seq * dim, hints, 1, index,
                                        error, error_size))
            ok = 0;
    }
    /* final layer over the unified sequence. */
    double trace_t4 = zpipe_now();
    ok = ok && qzen_zimage_final_layer_forward_gpu(&pipe->final_layer, zgu, full,
                                                 seq, prep->adaln, out64, error,
                                                 error_size);
    double trace_t5 = zpipe_now();
    if (ok) {
        uint16_t *out_bf16 = malloc((size_t)seq * 64 * 2);
        if (!out_bf16) {
            zpipe_fail(error, error_size, "cannot allocate output staging");
            ok = 0;
        } else {
            ok = qzen_gpu_tensor_read_bf16(out64, out_bf16, (size_t)seq * 64) &&
                 qzen_gpu_submit(gpu);
            if (ok) {
                for (size_t index = 0; index < (size_t)seq * 64; index++)
                    branch[index] = zpipe_bf16_to_f32(out_bf16[index]);
            }
            free(out_bf16);
        }
    } else {
        (void)qzen_gpu_submit(gpu);
    }
    if (!ok && (!error || !*error))
        zpipe_fail(error, error_size, "Z-Image GPU DiT forward failed: %s",
                   qzen_gpu_error(gpu));
    double trace_t6 = zpipe_now();
    if (trace)
        fprintf(stderr,
                "gpu phases: setup %.3f embedders+refiners %.3f concat %.3f "
                "main30 %.3f final %.3f download %.3f total %.3f\n",
                trace_t1 - trace_t0, trace_t2 - trace_t1, trace_t3 - trace_t2,
                trace_t4 - trace_t3, trace_t5 - trace_t4, trace_t6 - trace_t5,
                trace_t6 - trace_t0);
    if (ok)
        zpipe_unpatchify(branch, x_real, prep->w_tokens, prep->latent_h,
                         prep->latent_w, velocity);
    free(branch);
    qzen_gpu_tensor_free(hint_scratch);
    free(hint_staged);
    return ok;
}

int qzen_zpipe_dit_forward(const qzen_zimage_pipeline *pipe, const float *latents,
                         uint32_t latent_h, uint32_t latent_w,
                         const float *cap_feats, uint32_t cap_len, float tau,
                         float *velocity, char *error, size_t error_size) {
    zpipe_prep prep;
    if (!zpipe_prepare(pipe, latents, latent_h, latent_w, cap_len, tau, &prep,
                       error, error_size))
        return 0;
    int ok = pipe->gpu
                 ? zpipe_dit_forward_gpu(pipe, &prep, cap_feats, cap_len, NULL,
                                         velocity, error, error_size)
                 : zpipe_dit_forward_cpu(pipe, &prep, cap_feats, cap_len, NULL,
                                         velocity, error, error_size);
    zpipe_prep_free(&prep);
    return ok;
}

int qzen_zpipe_denoise(const qzen_zimage_pipeline *pipe, const float *cap_feats,
                     uint32_t cap_len, float *latents, uint32_t latent_h,
                     uint32_t latent_w,
                     void (*progress)(uint32_t step, uint32_t steps),
                     char *error, size_t error_size) {
    uint32_t steps = pipe->config.steps;
    float *sigmas = malloc((steps + 1) * sizeof(float));
    float *velocity = malloc((size_t)16 * latent_h * latent_w * sizeof(float));
    if (!sigmas || !velocity) {
        free(sigmas); free(velocity);
        zpipe_fail(error, error_size, "cannot allocate denoise buffers");
        return 0;
    }
    qzen_zpipe_sigmas(steps, pipe->config.shift, sigmas);
    for (uint32_t step = 0; step < steps; step++) {
        float tau = 1.0f - sigmas[step];
        if (!qzen_zpipe_dit_forward(pipe, latents, latent_h, latent_w, cap_feats,
                                  cap_len, tau, velocity, error, error_size)) {
            free(sigmas); free(velocity);
            return 0;
        }
        /* Euler: x += (sigma_next - sigma) * (-v). */
        float delta = sigmas[step + 1] - sigmas[step];
        size_t count = (size_t)16 * latent_h * latent_w;
        for (size_t index = 0; index < count; index++) {
            latents[index] -= delta * velocity[index];
        }
        if (progress) progress(step + 1, steps);
    }
    free(sigmas); free(velocity);
    return 1;
}

/* Same Euler schedule as qzen_zpipe_denoise(), but every step first recomputes
 * the ControlNet skips from the CURRENT noisy latents (WanGP recomputes the
 * control context once and the skips per step, because adaLN and x_init depend
 * on tau and x_t). Plain t2i stays on qzen_zpipe_denoise() and never pays for
 * hint computation. */
int qzen_zpipe_denoise_control(const qzen_zimage_pipeline *pipe,
                     qzen_zimage_control *control,
                     const float *control_context, float context_scale,
                     const float *cap_feats, uint32_t cap_len,
                     float *latents, uint32_t latent_h, uint32_t latent_w,
                     void (*progress)(uint32_t step, uint32_t steps),
                     char *error, size_t error_size) {
    if (!pipe || !control || !control_context || !isfinite(context_scale)) {
        zpipe_fail(error, error_size, "invalid ControlNet denoise request");
        return 0;
    }
    const uint32_t steps = pipe->config.steps;
    float *sigmas = malloc((steps + 1) * sizeof(float));
    float *velocity = malloc((size_t)16 * latent_h * latent_w * sizeof(float));
    if (!sigmas || !velocity) {
        free(sigmas); free(velocity);
        zpipe_fail(error, error_size, "cannot allocate denoise buffers");
        return 0;
    }
    qzen_zpipe_sigmas(steps, pipe->config.shift, sigmas);
    for (uint32_t step = 0; step < steps; step++) {
        const float tau = 1.0f - sigmas[step];
        zpipe_prep prep;
        zpipe_control_hints hints;
        if (!zpipe_prepare(pipe, latents, latent_h, latent_w, cap_len, tau,
                           &prep, error, error_size))
            goto fail;
        if (!zpipe_control_hints_compute(pipe, &prep, cap_feats, cap_len,
                                         control, control_context,
                                         context_scale, &hints,
                                         error, error_size)) {
            zpipe_prep_free(&prep);
            goto fail;
        }
        int ok = pipe->gpu
            ? zpipe_dit_forward_gpu(pipe, &prep, cap_feats, cap_len, &hints,
                                    velocity, error, error_size)
            : zpipe_dit_forward_cpu(pipe, &prep, cap_feats, cap_len, &hints,
                                    velocity, error, error_size);
        zpipe_control_hints_free(&hints);
        zpipe_prep_free(&prep);
        if (!ok) goto fail;

        /* Euler: x += (sigma_next - sigma) * (-v). */
        const float delta = sigmas[step + 1] - sigmas[step];
        const size_t count = (size_t)16 * latent_h * latent_w;
        for (size_t index = 0; index < count; index++)
            latents[index] -= delta * velocity[index];
        if (progress) progress(step + 1, steps);
    }
    free(sigmas); free(velocity);
    return 1;

fail:
    free(sigmas); free(velocity);
    return 0;
}
/* --- PNG (8-bit RGB, stored deflate) ------------------------------------------ */

static uint32_t zpipe_crc_table[256];

static void zpipe_crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (uint32_t k = 0; k < 8; k++) {
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        }
        zpipe_crc_table[n] = c;
    }
}

static uint32_t zpipe_crc_update(uint32_t running, const uint8_t *data, size_t count) {
    for (size_t index = 0; index < count; index++) {
        running = zpipe_crc_table[(running ^ data[index]) & 0xFF] ^ (running >> 8);
    }
    return running;
}

static void zpipe_store_u32(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)(value >> 24);
    out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8);
    out[3] = (uint8_t)value;
}

static int zpipe_write_chunk(FILE *file, const char *type, const uint8_t *data,
                             uint32_t length) {
    uint8_t header[8];
    uint8_t crc_bytes[4];
    zpipe_store_u32(header, length);
    memcpy(header + 4, type, 4);
    if (fwrite(header, 1, 8, file) != 8) return 0;
    if (length && fwrite(data, 1, length, file) != length) return 0;
    uint32_t crc = zpipe_crc_update(0xFFFFFFFFu, (const uint8_t *)type, 4);
    crc = zpipe_crc_update(crc, data, length);
    zpipe_store_u32(crc_bytes, crc ^ 0xFFFFFFFFu);
    if (fwrite(crc_bytes, 1, 4, file) != 4) return 0;
    return 1;
}

static void zpipe_store_u16(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)(value & 0xFF);
    out[1] = (uint8_t)(value >> 8);
}

int qzen_png_write_rgb8(const char *path, uint32_t width, uint32_t height,
                      const uint8_t *rgb, char *error, size_t error_size) {
    static int crc_ready = 0;
    if (!crc_ready) {
        zpipe_crc_init();
        crc_ready = 1;
    }
    size_t raw_size = (size_t)height * (width * 3 + 1);
    size_t deflate_blocks = (raw_size + 65534) / 65535;   /* >= 1 */
    uint8_t *raw = malloc(raw_size);
    size_t idat_size = 2 + deflate_blocks * 5 + raw_size + 4;
    uint8_t *idat = malloc(idat_size);
    FILE *file = fopen(path, "wb");
    if (!raw || !idat || !file) {
        free(raw); free(idat);
        if (file) fclose(file);
        zpipe_fail(error, error_size, "cannot write PNG %s", path);
        return 0;
    }
    /* filter byte 0 per scanline. */
    for (uint32_t y = 0; y < height; y++) {
        raw[(size_t)y * (width * 3 + 1)] = 0;
        memcpy(raw + (size_t)y * (width * 3 + 1) + 1, rgb + (size_t)y * width * 3,
               (size_t)width * 3);
    }
    /* zlib stream: 0x78 0x01 + stored deflate blocks + adler32. */
    size_t offset = 0;
    idat[offset++] = 0x78;
    idat[offset++] = 0x01;
    size_t remaining = raw_size;
    const uint8_t *cursor = raw;
    while (remaining > 0) {
        size_t block = remaining > 65535 ? 65535 : remaining;
        int final_block = block == remaining;
        idat[offset++] = (uint8_t)(final_block ? 1 : 0);
        zpipe_store_u16(idat + offset, (uint32_t)block);
        offset += 2;
        zpipe_store_u16(idat + offset, (uint32_t)(~block & 0xFFFF));
        offset += 2;
        memcpy(idat + offset, cursor, block);
        offset += block;
        cursor += block;
        remaining -= block;
    }
    uint32_t a = 1, b = 0;
    for (size_t index = 0; index < raw_size; index++) {
        a = (a + raw[index]) % 65521u;
        b = (b + a) % 65521u;
    }
    zpipe_store_u32(idat + offset, (b << 16) | a);
    offset += 4;

    static const uint8_t signature[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    int ok = fwrite(signature, 1, 8, file) == 8;
    uint8_t ihdr[13];
    zpipe_store_u32(ihdr, width);
    zpipe_store_u32(ihdr + 4, height);
    ihdr[8] = 8;   /* bit depth */
    ihdr[9] = 2;   /* color type: truecolor RGB */
    ihdr[10] = 0;  /* compression */
    ihdr[11] = 0;  /* filter */
    ihdr[12] = 0;  /* interlace */
    ok = ok && zpipe_write_chunk(file, "IHDR", ihdr, 13);
    ok = ok && zpipe_write_chunk(file, "IDAT", idat, (uint32_t)offset);
    ok = ok && zpipe_write_chunk(file, "IEND", NULL, 0);
    if (fclose(file) != 0) ok = 0;
    free(raw);
    free(idat);
    if (!ok) {
        zpipe_fail(error, error_size, "short write on %s", path);
        return 0;
    }
    return 1;
}

int qzen_png_write_pixels_f32(const char *path, const float *pixels,
                            uint32_t width, uint32_t height,
                            char *error, size_t error_size) {
    /* pixels are planar [3][H][W]; PNG rows are interleaved RGB. */
    size_t plane = (size_t)width * height;
    uint8_t *rgb = malloc(plane * 3);
    if (!rgb) {
        zpipe_fail(error, error_size, "cannot allocate RGB buffer");
        return 0;
    }
    const size_t plane_offset[3] = {0, plane, 2 * plane};
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t *out = rgb + ((size_t)y * width + x) * 3;
            for (uint32_t c = 0; c < 3; c++) {
                float value = pixels[plane_offset[c] + (size_t)y * width + x];
                value = value * 0.5f + 0.5f;
                if (value < 0.0f) value = 0.0f;
                if (value > 1.0f) value = 1.0f;
                out[c] = (uint8_t)(value * 255.0f + 0.5f);
            }
        }
    }
    int ok = qzen_png_write_rgb8(path, width, height, rgb, error, error_size);
    free(rgb);
    return ok;
}
