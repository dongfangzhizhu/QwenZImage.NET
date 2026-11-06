#ifndef QZEN_QWEN21_DIT_H
#define QZEN_QWEN21_DIT_H

/* Qwen-Image 2.1 (7B) single-stream DiT: CPU reference forward.
 *
 * Phase 0 numeric gate for the Wan2GP models/qwen21 reference: an f32 CPU
 * implementation of the 32-block joint-attention transformer, reading the
 * INT8 ConvRot checkpoint through the same qzen_safetensors dequantiser the
 * MiniMax-H3 DiT uses. Layout and modulation rows follow transformer.py:
 * the sequence is [text prefix ; image tokens], the prefix modulates from the
 * t=0 row and the image tokens from the sigma row.
 */

#include <stddef.h>
#include "qzen_gpu.h"

typedef struct qzen_qwen21_model qzen_qwen21_model;

typedef struct qzen_qwen21_request {
    const float *text;   /* [text_tokens][4096] text encoder rows */
    int text_tokens;
    const float *latent; /* [64][height][width], channel major */
    int height;
    int width;
    const float *sigmas; /* [sigma_count] in 0..1 (scheduler sigma) */
    int sigma_count;
} qzen_qwen21_request;

/* Open a single-file checkpoint and read the resident (non-block) weights.
 * thread_count <= 0 selects the online CPU count, capped at QZEN_QWEN21_THREADS. */
qzen_qwen21_model *qzen_qwen21_open(const char *checkpoint, int thread_count,
                                char *error, size_t error_size);
void qzen_qwen21_close(qzen_qwen21_model *model);

/* Full forward for every sigma in one pass over the block weights.
 * velocity:   [sigma_count][height * width][64] on return.
 * block0_out: optional [sigma_count][text_tokens + height * width][4096], the
 *             joint hidden states after transformer_blocks.0 (gate anchor). */
int qzen_qwen21_forward(qzen_qwen21_model *model, const qzen_qwen21_request *request,
                      float *velocity, float *block0_out,
                      char *error, size_t error_size);

/* One block only, for localising a gate failure: hidden [tokens][4096] in and
 * out, tokens split at text_tokens, modulation rows from sigma. */
int qzen_qwen21_block_forward(qzen_qwen21_model *model, int layer,
                            const float *hidden, int text_tokens, int height, int width,
                            float sigma, float *out,
                            char *error, size_t error_size);

/* Telemetry: seconds spent reading/dequantising block weights, and the
 * per-block weight bytes (rows * columns * 4 once dequantised). */
double qzen_qwen21_weight_seconds(const qzen_qwen21_model *model);

/* --- Device path (QWEN21_GPU=1 at open) --------------------------------
 * The 32 blocks stay resident as INT8 ConvRot (~7.0 GB); a forward never
 * re-reads the checkpoint and the hidden carrier never leaves the GPU. */
int qzen_qwen21_device_ready(const qzen_qwen21_model *model);
qzen_gpu *qzen_qwen21_device(const qzen_qwen21_model *model);
/* One-sigma forward with the latent already on the device (channel-major
 * F32 [64][height][width]); velocity_out receives [image_tokens][64] BF16.
 * block0_out (optional, host F32 [tokens][4096]) anchors block 0. */
int qzen_qwen21_forward_device(qzen_qwen21_model *model, const float *text,
                             int text_tokens, const qzen_gpu_tensor *latent_device,
                             int height, int width, float sigma,
                             qzen_gpu_tensor *velocity_out, float *block0_out,
                             char *error, size_t error_size);
/* Device velocity scratch slots (0/1) for the sampler's CFG pair; grown on
 * demand, valid until the next grow or close. */
qzen_gpu_tensor *qzen_qwen21_device_velocity(qzen_qwen21_model *model, int slot,
                                         int image_tokens);

/* Merge a PEFT/Diffusers adapter into the resident INT8 ConvRot projections
 * (attn.to_q/to_k/to_v/to_out.0 on every block). scale multiplies the file's
 * own alpha/rank, so 1.0 is "as published". Device path only: the CPU
 * reference keeps its F32 weights and is used for numeric gates, not serving.
 * Returns 1 on success; error names the tensor that failed. */
int qzen_qwen21_apply_lora(qzen_qwen21_model *model, const char *path, float scale,
                         char *error, size_t error_size);
/* The adapter's own alpha/rank, read from the safetensors header without
 * mapping any tensor. 1.0 when the file carries no alpha or no rank (the H3
 * adapters define themselves as scale 1). PEFT nests the pair inside
 * __metadata__."lora_adapter_metadata", so the key is escaped in the raw
 * header text; the parser matches the bare key and survives both spellings. */
double qzen_qwen21_lora_ratio(const char *path);
/* One block on the device with host F32 in/out (gate localisation). */
int qzen_qwen21_block_forward_device(qzen_qwen21_model *model, int layer,
                                   const float *hidden, int text_tokens,
                                   int height, int width, float sigma, float *out,
                                   char *error, size_t error_size);


#endif
