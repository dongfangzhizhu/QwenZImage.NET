#ifndef QZEN_QWEN21_VAE_H
#define QZEN_QWEN21_VAE_H

/* Qwen-Image 2.1 VAE decoder: f32 CPU reference (AutoencoderKLQwenImage21,
 * models/qwen21/vae.py). Image path only: the 3-D convolutions lose their
 * single frame, so every layer is a plain 2-D convolution.
 *
 *   post_quant_conv 64->64 (1x1)
 *   conv_in 64->1152 (3x3) -> mid: resnet, single-head spatial attention
 *   (scale 1/sqrt(1152)), resnet
 *   -> 5 up blocks 1152 -> 1152 -> 576 -> 288 -> 144 -> 144, 3 resnets each,
 *      the first four upsampling (nearest x2 then 3x3 conv) and every upsampled
 *      block adding the DupUp3D channel shuffle of its input
 *   -> norm_out RMS(144) -> SiLU -> conv_out 144->4 (3x3)
 *
 * Normalisation is QwenImage21RMS_norm: x * sqrt(C) / ||x||_2(C) * gamma + bias,
 * not a GroupNorm. The checkpoint is plain F32, so nothing is dequantised here.
 */

#include <stddef.h>
#include <stdint.h>

#define QZEN_QWEN21_VAE_CHANNELS 64

typedef struct qzen_qwen21_vae qzen_qwen21_vae;

qzen_qwen21_vae *qzen_qwen21_vae_open(const char *checkpoint, int thread_count, char *error,
                                  size_t error_size);
void qzen_qwen21_vae_close(qzen_qwen21_vae *vae);

/* Decode a denormalised latent [64][height][width] into RGB8
 * [3][16*height][16*width] (four nearest-2x up blocks): clamp to [-1, 1]
 * then (x + 1) * 127.5, rounding to the nearest integer. The encoder below
 * is the exact inverse, hence the /16 latent grid used by the ABI. */
int qzen_qwen21_vae_decode(qzen_qwen21_vae *vae, const float *latent, int height, int width,
                         unsigned char *rgb, char *error, size_t error_size);

/* Encode RGB8 [3][height][width] into the sampler's normalised latent
 * [64][height/16][width/16]: the encoder is the mirror of the decoder and
 * lives in the same checkpoint, loaded on the first call. The 4th input
 * channel is the RGBA alpha of the reference (WanGP feeds image.convert("RGBA")
 * into the VAE); an opaque edit reference therefore sets it to 1 everywhere.
 * The user's inpainting mask does NOT enter the VAE -- it is applied on the
 * latent grid by the sampler (see qzen_qwen21_sample.latent_mask).
 * Deterministic: the returned latent is the posterior mean, so an edit does
 * not resample the reference. */
int qzen_qwen21_vae_encode(qzen_qwen21_vae *vae, const unsigned char *rgb, int height, int width,
                         float *latent, char *error, size_t error_size);

/* Per-channel latent statistics from vae_config.json: the sampler denormalises
 * with z * std + mean before decoding. */
const float *qzen_qwen21_vae_latent_mean(const qzen_qwen21_vae *vae);
const float *qzen_qwen21_vae_latent_std(const qzen_qwen21_vae *vae);

#endif
