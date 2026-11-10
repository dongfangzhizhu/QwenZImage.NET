#ifndef QZEN_ZIMAGE_VAE_H
#define QZEN_ZIMAGE_VAE_H

#include <stddef.h>
#include <stdint.h>

/* Conv AutoencoderKL decoder for Z-Image Turbo (Wan2GP
 * models/z_image/autoencoder_kl.py, a thin wrapper over diffusers'
 * AutoencoderKL).  The decoder serves every text-to-image request; the
 * encoder is an additional, lazily loaded path used only to turn a ControlNet
 * guide image into its normalized latent (qzen_zvae_encode).
 *
 * Architecture (ZImageTurbo_VAE_bf16_config.json):
 *   conv_in    Conv2d(16 -> 512, 3x3, pad 1)
 *   mid_block  ResnetBlock2D -> Attention (1 head, dim 512, GroupNorm 32,
 *              residual connection, bias) -> ResnetBlock2D
 *   4x UpDecoderBlock2D with block_out_channels [512, 512, 256, 128] (the
 *   reverse of the config's [128, 256, 512, 512]), each with
 *   layers_per_block + 1 = 3 ResnetBlock2D (GroupNorm 32 eps 1e-6 -> SiLU ->
 *   conv3x3 -> GroupNorm -> SiLU -> conv3x3 + 1x1 shortcut when the channel
 *   count changes) followed by Upsample2D (nearest x2 + conv3x3) on every
 *   block except the last
 *   conv_norm_out GroupNorm(32, 128) -> SiLU -> conv_out Conv2d(128 -> 3)
 *
 * The checkpoint is plain BF16 (no quantization, ~160 MB); weights are
 * widened to F32 at load.  Two decoders share this file: an F32 CPU reference
 * with OpenMP, and a device mirror built from the same host weights on the
 * first decode (qzen_gpu_conv2d_f32 / group_norm_chw / upsample_nearest2x /
 * vae_sdpa_f32).  Which one runs is ZIMAGE_VAE_GPU: unset = device when there
 * is one and CPU otherwise, "0" = the reference path, "1" = demand the device.
 * At 512 px they measure 0.33 s and 22.18 s (§23), and at 1024 px they agree
 * to 115.88 dB (rms 3e-6).
 *
 * The decoder consumes the *normalized* latents produced by the DiT and
 * undoes the pipeline scaling itself:
 *
 *   pixels = decode(latents / scaling_factor + shift_factor)
 *
 * with scaling_factor 0.3611 and shift_factor 0.1159, returning F32 RGB in
 * [-1, 1] (the same range diffusers' VaeImageProcessor expects before
 * (x / 2 + 0.5).clamp(0, 1)).
 */

typedef struct {
    uint32_t latent_channels;   /* 16 */
    uint32_t out_channels;      /* 3 */
    uint32_t up_channels[4];    /* per up block: 512, 512, 256, 128 */
    uint32_t layers_per_block;  /* 2 (=> 3 resnets per up block) */
    uint32_t groups;            /* GroupNorm groups, 32 */
    float eps;                  /* GroupNorm eps, 1e-6 */
    float scaling_factor;       /* 0.3611 */
    float shift_factor;         /* 0.1159 */
} qzen_zvae_config;

qzen_zvae_config qzen_zvae_default_config(void);

typedef struct qzen_zvae qzen_zvae;

/* Load every decoder tensor as F32 from the BF16 safetensors file(s) in
 * "directory" (a single-file directory is expected, mirroring the DiT fixture
 * layout).  ~330 MB resident. */
qzen_zvae *qzen_zvae_load(const char *directory, const qzen_zvae_config *config,
                      char *error, size_t error_size);
void qzen_zvae_free(qzen_zvae *vae);

/* latents: [latent_channels, latent_h, latent_w] F32, normalized (DiT space).
 * pixels:  [out_channels, latent_h * 8, latent_w * 8] F32 in [-1, 1]. */
int qzen_zvae_decode(const qzen_zvae *vae, const float *latents,
                   uint32_t latent_h, uint32_t latent_w, float *pixels,
                   char *error, size_t error_size);

/* Encoder side (ControlNet). Planar RGB8 [3, height, width] in 0..255 ->
 * normalized latent [latent_channels, height/8, width/8] in DiT space:
 *
 *   pixels = (rgb / 255) * 2 - 1
 *   mean   = encoder(pixels)  (posterior mean, first 16 of the 32 channels)
 *   latent = (mean - shift_factor) * scaling_factor
 *
 * The encoder weights live in the same vae.safetensors as the decoder and are
 * loaded lazily on the first encode, so decoding an ordinary text-to-image
 * request costs nothing extra. width/height must be non-zero multiples of 8.
 * `latent` is caller-owned and holds 16 * (width/8) * (height/8) floats. */
int qzen_zvae_encode(qzen_zvae *vae, const uint8_t *rgb,
                     uint32_t width, uint32_t height, float *latent,
                     char *error, size_t error_size);

#endif
