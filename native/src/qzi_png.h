#ifndef QZI_PNG_H
#define QZI_PNG_H

#include <stddef.h>
#include <stdint.h>

/* Minimal 8-bit RGB / RGBA PNG encoder: stored (uncompressed) deflate blocks,
 * no external dependencies. Used by the deterministic reference backend so the
 * whole managed stack can be exercised without a GPU or model weights, and by
 * the real backend for both output formats.
 *
 * `rgb` is interleaved [H][W][channels]. RGBA uses PNG colour type 6, RGB
 * colour type 2. Returns a malloc'd buffer, or NULL on failure (with a
 * message in `error`). The caller frees it with free() -- the ABI layer hands
 * it to qzi_image_release(), which does exactly that. */
uint8_t *qzi_png_encode_rgb8(const uint8_t *rgb, uint32_t width, uint32_t height,
                             size_t *out_size, char *error, size_t error_cap);

/* Same, but 4 interleaved channels (R,G,B,A) and PNG colour type 6. WanGP's
 * rgba = Enabled path keeps the VAE decoder's 4th channel as generated alpha. */
uint8_t *qzi_png_encode_rgba8(const uint8_t *rgba, uint32_t width, uint32_t height,
                              size_t *out_size, char *error, size_t error_cap);

#endif /* QZI_PNG_H */
