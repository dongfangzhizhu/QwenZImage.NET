#ifndef QZEN_ZIMAGE_PIPELINE_H
#define QZEN_ZIMAGE_PIPELINE_H

#include "qzen_zimage_dit.h"
#include "qzen_zimage_text.h"

#include <stddef.h>
#include <stdint.h>

/* Z-Image Turbo text -> denoise -> decode pipeline (Wan2GP
 * models/z_image/pipeline_z_image.py, the non-unified Flow Match Euler path
 * with guidance_scale = 0 as used by the Turbo checkpoint).
 *
 * Sampling (task spec, matching FlowMatchEulerDiscreteScheduler with
 * num_train_timesteps 1000 and shift 3.0):
 *
 *   sigma(u) = shift * u / (1 + (shift - 1) * u),  u = k / N
 *   sigmas = [sigma(1), sigma((N-1)/N), ..., sigma(1/N), 0]   (N = steps = 8)
 *
 * so sigma descends from 1.0 to 0 in N Euler steps:
 *
 *   tau_i = 1 - sigmas[i]                 (normalized timestep, DiT input)
 *   v_i   = DiT(x_i, tau_i, cap_feats)    (velocity head output)
 *   x_{i+1} = x_i + (sigmas[i+1] - sigmas[i]) * (-v_i)
 *
 * The DiT driver assembles the full forward per z_image_transformer2d.py:
 * patchify 2x2 (64 ch per patch) -> all_x_embedder.2-1, pad rows replaced by
 * x_pad_token; cap_embedder (RMSNorm(2560) + Linear(2560->3840)), pad rows
 * replaced by cap_pad_token; 2 noise refiners (modulated, x section), 2
 * context refiners (unmodulated, cap section), concat [x | cap], 30 main
 * layers, FinalLayer, unpatchify.  RoPE position ids: caption axis0 = 1..,
 * image axis0 = cap_padded_len + 1 with (h, w) on axes 1/2, pad rows (0,0,0),
 * all padded to seq_multiple = 32.
 *
 * Everything runs on the CPU in F32 (weights BF16 widened per element),
 * OpenMP parallel via qzen_zimage_dit.c.  A 1024x1024 8-step run takes on the
 * order of an hour; the CUDA port is a later phase.
 */

typedef struct {
    uint32_t steps;        /* 8 (Turbo) */
    float shift;           /* 3.0 */
    uint32_t seq_multiple; /* 32 (SEQ_MULTI_OF) */
} qzen_zpipe_config;

qzen_zpipe_config qzen_zpipe_default_config(void);

typedef struct qzen_zimage_pipeline qzen_zimage_pipeline;
typedef struct qzen_zimage_control qzen_zimage_control;

/* Load the quanto qint8 DiT checkpoint (plus its plain-BF16 refiners,
 * embedders, pad tokens and final layer) from the single-file directory
 * "dit_dir".  ~12.3 GiB resident BF16. */
qzen_zimage_pipeline *qzen_zpipe_load(const char *dit_dir,
                                  const qzen_zpipe_config *config,
                                  char *error, size_t error_size);
void qzen_zpipe_free(qzen_zimage_pipeline *pipe);

/* Load the Qwen3-4B text encoder + tokenizer from "qwen_dir" (the directory
 * holding qwen3_quanto_bf16_int8.safetensors and tokenizer.json).  ~7.8 GiB.
 * qzen_zpipe_release_text frees it again (the cap features survive). */
int qzen_zpipe_attach_text(qzen_zimage_pipeline *pipe, const char *qwen_dir,
                         char *error, size_t error_size);
void qzen_zpipe_release_text(qzen_zimage_pipeline *pipe);

/* Hand the encoder over without freeing it, and take it back.  The encoder is
 * shared by every DiT pack (it lives in zimage_dir), so a tier switch frees the
 * DiT but should keep this: re-reading and re-uploading it costs 71-79 s on the
 * same file.  A detached pair nobody adopts is freed with
 * qzen_zpipe_release_detached_text. */
int qzen_zpipe_detach_text(qzen_zimage_pipeline *pipe, qzen_ztext_model **model,
                         qzen_tokenizer **tokenizer, qzen_ztext_config *config);
void qzen_zpipe_adopt_text(qzen_zimage_pipeline *pipe, qzen_ztext_model *model,
                         qzen_tokenizer *tokenizer,
                         const qzen_ztext_config *config);
void qzen_zpipe_release_detached_text(qzen_ztext_model *model,
                                    qzen_tokenizer *tokenizer);

/* Chat template + tokenize + Qwen3 forward; cap_feats receives a malloc-ed
 * [cap_len, 2560] F32 array of the last-layer pre-final-norm rows (the
 * real, un-padded tokens only). */
int qzen_zpipe_encode_text(qzen_zimage_pipeline *pipe, const char *prompt,
                         float **cap_feats, uint32_t *cap_len,
                         char *error, size_t error_size);

/* One DiT velocity evaluation.  latents [16, lh, lw] F32 (DiT/normalized
 * space), tau the normalized timestep, velocity [16, lh, lw] receives the
 * model output. */
int qzen_zpipe_dit_forward(const qzen_zimage_pipeline *pipe, const float *latents,
                         uint32_t latent_h, uint32_t latent_w,
                         const float *cap_feats, uint32_t cap_len, float tau,
                         float *velocity, char *error, size_t error_size);

/* The N+1 sigma schedule (descending, terminating at 0) for steps/shift. */
void qzen_zpipe_sigmas(uint32_t steps, float shift, float *sigmas);

/* Deterministic latent init: standard normal [channels, lh, lw] from a
 * xoshiro256++ / splitmix64 stream seeded by "seed" (Box-Muller).  This RNG
 * is QwenZImageNative-specific: torch checkpoints of the same seed do NOT match, so the
 * numerics gates feed dumped latents instead. */
void qzen_zpipe_init_latents(uint64_t seed, uint32_t channels, uint32_t lh,
                           uint32_t lw, float *latents);

/* Full denoise loop over the 8 sigma intervals, in place on latents. */
int qzen_zpipe_denoise(const qzen_zimage_pipeline *pipe, const float *cap_feats,
                     uint32_t cap_len, float *latents, uint32_t latent_h,
                     uint32_t latent_w,
                     void (*progress)(uint32_t step, uint32_t steps),
                     char *error, size_t error_size);

/* --- Z-Image Turbo Fun ControlNet Union (additive ABI v1, stage 16) -------
 *
 * Prepare the ControlNet weights for this pipeline. On a GPU build this
 * uploads them into the DiT's GPU context; on failure (e.g. not enough VRAM)
 * the hint path falls back to the CPU block forward. Host weights always stay
 * resident. `control` is owned by the caller and must outlive every
 * qzen_zpipe_denoise_control() call. */
int qzen_zpipe_control_prepare(qzen_zimage_pipeline *pipe,
                               qzen_zimage_control *control,
                               char *error, size_t error_size);

/* Build the 33-channel (v2.x) or 16-channel (v1) ControlNet context from a
 * VAE-encoded, normalized control latent [16, lh, lw]. Returns a malloc-ed
 * [in_dim, lh, lw] F32 array, or NULL. v2.x concatenates
 * [control_latent(16) | keep_mask(1, all zeros) | inpaint_latent(16, zeros)],
 * exactly like WanGP pipeline_z_image.py with no inpaint mask. */
float *qzen_zpipe_control_context(const qzen_zimage_control *control,
                                  const float *control_latent,
                                  uint32_t latent_h, uint32_t latent_w,
                                  char *error, size_t error_size);

/* Denoise with ControlNet conditioning. `control_context` is the array above
 * (caller-owned); `context_scale` scales every skip connection (WanGP's
 * control_context_scale, 0.65 for Union2.1). Plain t2i uses
 * qzen_zpipe_denoise() and never pays for any of this. */
int qzen_zpipe_denoise_control(const qzen_zimage_pipeline *pipe,
                     qzen_zimage_control *control,
                     const float *control_context, float context_scale,
                     const float *cap_feats, uint32_t cap_len,
                     float *latents, uint32_t latent_h, uint32_t latent_w,
                     void (*progress)(uint32_t step, uint32_t steps),
                     char *error, size_t error_size);

/* Minimal PNG writer (8-bit RGB, stored deflate, no external deps). */
int qzen_png_write_rgb8(const char *path, uint32_t width, uint32_t height,
                      const uint8_t *rgb, char *error, size_t error_size);

/* [-1, 1] F32 pixels [3, height, width] -> clamped RGB8 PNG. */
int qzen_png_write_pixels_f32(const char *path, const float *pixels,
                            uint32_t width, uint32_t height,
                            char *error, size_t error_size);

#endif
