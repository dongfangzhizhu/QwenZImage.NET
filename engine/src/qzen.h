/* Public API for the QwenZImageNative-metal MiniMax-H3 inference engine. */
#ifndef QZEN_H
#define QZEN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QZEN_VERSION "0.2.3"
#define QZEN_DEFAULT_WIDTH 864
#define QZEN_DEFAULT_HEIGHT 480
#define QZEN_DEFAULT_FRAMES 56
#define QZEN_DEFAULT_STEPS 20
#define QZEN_DEFAULT_DIT_LAYERS 50
#define QZEN_MIN_DIT_LAYERS 35

typedef struct qzen_ctx qzen_ctx;
typedef struct qzen_result qzen_result;

typedef struct {
    size_t embedding_entries;
    size_t embedding_bytes;
    int prepared_dit;
    int video_decoder;
    /* Cumulative DiT weight loads on this context; rebinds do not count. */
    unsigned long long dit_loads;
} qzen_cache_info;

typedef enum {
    QZEN_REFERENCE_IMAGE = 1,
    QZEN_REFERENCE_VIDEO = 2,
    QZEN_REFERENCE_AUDIO = 3,
    QZEN_REFERENCE_VIDEO_AUDIO = 4
} qzen_reference_kind;

typedef struct {
    qzen_reference_kind kind;
    const char *path;
    const char *audio_path;
    int include_embedded_audio;
} qzen_reference;

typedef enum {
    QZEN_REFERENCE_IMAGE_MATCH = 0,
    QZEN_REFERENCE_IMAGE_MAX = 1
} qzen_reference_image_size;

typedef struct {
    int width;
    int height;
    int stride;
    const uint8_t *rgb;
    int frame_index;
    int frame_count;
    /* Non-negative only for an intermediate denoising preview. */
    int denoise_step;
    int denoise_steps;
} qzen_frame;

typedef int (*qzen_frame_callback)(const qzen_frame *frame, void *opaque);
typedef int (*qzen_progress_callback)(const char *phase, int completed, int total,
                                    void *opaque);

typedef struct {
    int width;
    int height;
    int frames;
    int steps;
    uint64_t seed;
    const char *output_path;
    const char *first_frame;
    const char *last_frame;
    const qzen_reference *references;
    size_t reference_count;
    qzen_reference_image_size reference_image_size;
    /* Evaluate one of every N denoiser steps. 1 is the close-reference path,
     * 2 is the validated fast path, and 3 is the aggressive fast path. */
    int denoise_reuse;
    /* Number of gate-ranked DiT residual blocks to retain. 50 is exact,
     * 45 is the validated fast setting, and 40 is more aggressive. */
    int dit_layers;
    /* Recompute the transformer core every N denoiser steps while refreshing
     * the timestep head each step. 1 is exact, 4 fast, and 6 aggressive. */
    int core_reuse;
    /* Off by default. Pools adjacent horizontal video tokens in middle DiT
     * blocks (full-res residual). Faster long-video wall clock, with a
     * visible quality drop (fox-fast ~17.8 dB PSNR / 0.72 SSIM vs off). */
    int token_reduction;
    /* Off by default. Sol-Attn-style sparse SDPA on long sequences: keep
     * important KV tiles, approximate the rest from pooled K/V. Not
     * bit-identical. Override τ with QZEN_SOL_ATTN_TAU (default 0.5). */
    int sol_attn;
    /* Off by default. Run DiT block 0 every step and skip the rest when its
     * residual relative L2 stays under QZEN_FBC_REL (default 0.10). Early steps
     * (QZEN_FBC_WARMUP, default 4), late steps (QZEN_FBC_TAIL, default 4), and a
     * streak cap (QZEN_FBC_MAX_STREAK, default 4) stay dense. Not bit-identical. */
    int fbc;
    /* Use one int8 activation scale per FC2 row and the M5 full-K kernel.
     * Faster, but more numerically aggressive than grouped int8. */
    int use_int8_row_fc2;
    /* Restore the released spatial RoPE grid at 256x256. The default applies
     * a visually validated half-scale grid only at that native canvas. */
    int use_reference_rope;
    /* Keep only two original BF16 DiT blocks in memory and overlap reading the
     * next block from the checkpoint with execution of the current block. */
    int ssd_streaming;
    /* Optional lower internal model canvas. Both must be zero (exact output
     * canvas) or valid same-aspect dimensions no larger than width/height. */
    int render_width;
    int render_height;
    /* Force the portable close-reference BF16/MPS MLP implementation instead
     * of the fastest validated native MLP supported by the current GPU. */
    int use_slower_bf16_mlp;
    /* Force the portable close-reference BF16 QKV projection. */
    int use_slower_bf16_qkv;
    /* Force the portable BF16 attention-output projection. */
    int use_slower_bf16_attention_output;
    /* Materialize row-major BF16 after SDPA before int8 quantization. */
    int use_slower_row_major_attention_output;
    /* Keep int8 projection-input quantization as standalone kernels. */
    int use_slower_unfused_int8_inputs;
    /* Keep Q/K norm and RoPE as a separate kernel after int8 QKV. */
    int use_slower_unfused_qkv_rope;
    /* Force scalar BF16 loads in the fused Q/K RMS reducer. */
    int use_slower_scalar_qkv_rms;
    /* Reread int8 dequantization scales from device memory per output. */
    int use_slower_uncached_int8_scales;
    /* Use the generic runtime-bound FC1 TensorOps K loop. */
    int use_slower_dynamic_fc1_k;
    /* Force the original 256-thread FC2 grouped activation quantizer. */
    int use_slower_grouped_quantizer;
    /* Decode and deliver one representative frame after every Euler step. */
    int preview_denoise;
    qzen_frame_callback on_frame;
    qzen_progress_callback on_progress;
    void *callback_opaque;
    /* Run the PDD 8-step distilled release: the fixed 8-step schedule, the
     * per-step fused output heads, and (unless QZEN_LORA is already set) the
     * matching adaptor merged into the INT8 DiT weights. Forces steps to 8. */
    int pdd8;
    /* Keep the denoiser mapped after this request so the next one only pays
     * for the VAEs. Off by default: a one-shot run releases the DiT before
     * the VAE decode, which is strictly the smaller peak. QZEN_KEEP_DIT=1 also
     * turns it on so a resident server can be driven without new plumbing. */
    int keep_dit;
} qzen_params;

#define QZEN_PARAMS_DEFAULT { \
    QZEN_DEFAULT_WIDTH, QZEN_DEFAULT_HEIGHT, QZEN_DEFAULT_FRAMES, QZEN_DEFAULT_STEPS, \
    UINT64_C(42), NULL, NULL, NULL, NULL, 0, QZEN_REFERENCE_IMAGE_MATCH, \
    1, QZEN_DEFAULT_DIT_LAYERS, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, \
    0, 0 \
}

typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    /* Apple: the GPU family number (9 for M3, 10 for M4...).  CUDA: the
     * compute capability as major*10+minor, so an RTX 4090 reads 89.  The
     * field kept its Metal name when the CUDA backend landed, because every
     * reader wants the same thing from it -- which generation is this. */
    int apple_gpu_family;
    /* Apple: supportsFamily:MTLGPUFamilyMetal4, i.e. M5-class.  CUDA:
     * sm_120 and up, i.e. Blackwell.  Both answer "does this GPU have the
     * newer tensor path", which is how QwenZImageNative.c uses it. */
    int metal4;
    int unified_memory;
    /* The compute capability the GPU code in this binary was compiled for,
     * 0 when the build system did not record one (always 0 on Apple).  On
     * CUDA a mismatch means no cubin matches the device and every launch
     * fails, so qzen_device_arch_state() compares the two. */
    int build_arch;
} qzen_device_info;

typedef struct {
    uint64_t bytes;
    uint64_t tensor_bytes;
    size_t files;
    size_t tensors;
} qzen_component_info;

typedef struct {
    qzen_component_info text_encoder;
    qzen_component_info fl2va_transformer;
    qzen_component_info ref2va_transformer;
    qzen_component_info video_vae;
    qzen_component_info audio_vae;
} qzen_model_info;

struct qzen_result {
    int width;
    int height;
    int frames;
    int fps;
    int sample_rate;
    uint64_t seed;
};

/* Load model metadata and initialize the Metal device. Weights remain unmapped. */
qzen_ctx *qzen_load_dir(const char *model_dir);
void qzen_free(qzen_ctx *ctx);

const char *qzen_last_error(const qzen_ctx *ctx);
const qzen_device_info *qzen_device(const qzen_ctx *ctx);
const qzen_model_info *qzen_model(const qzen_ctx *ctx);

/* Interactive-session reuse. Disabled by default so one-shot callers retain
 * the original phase-by-phase memory lifetime. */
void qzen_cache_set_enabled(qzen_ctx *ctx, int enabled);
/* Directory for the on-disk text embedding cache used by the resident server:
 * a repeated prompt with the same anchors replays the BF16 rows instead of
 * streaming the encoder again. Returns 1 when the directory is usable. */
int qzen_set_text_cache_dir(qzen_ctx *ctx, const char *dir);
void qzen_cache_clear(qzen_ctx *ctx);
void qzen_cache_get_info(const qzen_ctx *ctx, qzen_cache_info *info);

/* Generate media, delivering decoded frames incrementally through on_frame. */
qzen_result *qzen_generate(qzen_ctx *ctx, const char *prompt,
                       const qzen_params *params);
void qzen_result_free(qzen_result *result);

/* Load the denoiser for a resident service before the first request, using a
 * minimal text-only layout at the given parameters. The first real request
 * rebinds the layout-derived state against the same weights, so this only
 * pays the weight read once. Returns 1 when the DiT is resident afterwards
 * (an already-matching resident DiT counts) and 0 with a message on failure;
 * callers may ignore the failure and let the first request load it. */
int qzen_dit_preload(qzen_ctx *ctx, const qzen_params *params,
                   char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
