#ifndef QZEN_QWEN21_PIPELINE_H
#define QZEN_QWEN21_PIPELINE_H

/* Qwen-Image 2.1 sampling: sigma schedules, CFG and the Euler step, then the VAE.
 *
 * Base model (models/qwen21/pipeline.py + scheduler_config.json):
 *   sigmas = linspace(1, 1/N, N), dynamic exponential shift anchored on the IMAGE
 *   token count only, then the terminal stretch to shift_terminal = 0.02, then 0.
 * Turbo: the eight released sigmas, no shift and no terminal stretch, then 0.
 *   The last turbo step from 0.414568 straight to 0 is by design.
 * Euler: x += (sigma_next - sigma) * v with the raw proj_out velocity (no sign
 * flip). CFG: v = v_neg + guidance * (v_pos - v_neg); base runs guidance 4.0
 * against the blank-prompt embedding, turbo runs guidance 1.0 (single pass).
 */

#include <stddef.h>
#include <stdint.h>

#include "qzen_qwen21_dit.h"
#include "qzen_qwen21_vae.h"

#define QZEN_QWEN21_TURBO_STEPS 8

/* Base schedule: steps sigmas plus the trailing zero. image_tokens is the number
 * of target tokens (height * width), which is what drives the shift. */
int qzen_qwen21_sigmas_base(int steps, int image_tokens, float *sigmas);

/* The eight released turbo sigmas plus the trailing zero. */
int qzen_qwen21_sigmas_turbo(float *sigmas);

/* Index of the first schedule point at or below `strength`, i.e. where an edit
 * starts. strength 1.0 gives 0 (text-to-image); a strength below the last
 * sigma gives steps - 1 (one step of work), never 0. */
int qzen_qwen21_edit_start(const float *sigmas, int steps, float strength);

typedef struct {
    const float *text; /* [text_tokens][4096], positive prompt */
    int text_tokens;
    const float *negative; /* [negative_tokens][4096] or NULL */
    int negative_tokens;
    const float *latent; /* [64][height][width], standard normal sample */
    int height;
    int width;
    const float *sigmas; /* [steps + 1] */
    int steps;
    float guidance;
    /* Image editing (img2img on the same weights): the reference image's
     * normalised latent [64][height][width] and how much of the schedule to
     * run. strength 1.0 means "ignore the reference" and is bit-identical to a
     * text-to-image request, so the existing gates keep covering the t2i path.
     * The start point is x = (1 - sigma) * init + sigma * noise, the same flow
     * parameterisation the Euler loop integrates. */
    const float *init_latent;
    float strength;
} qzen_qwen21_sample;

/* Phase progress for the resident service: "dit" with (step, steps) before each
 * denoise step, then "vae" with (0, 1) before the decode. */
typedef void (*qzen_qwen21_progress)(const char *phase, int completed, int total,
                                   void *opaque);

/* Denoise and decode; rgb is [3][4*height][4*width]. on_progress may be NULL. */
int qzen_qwen21_pipeline_run_ex(qzen_qwen21_model *dit, qzen_qwen21_vae *vae,
                              const qzen_qwen21_sample *sample, unsigned char *rgb,
                              qzen_qwen21_progress on_progress, void *opaque,
                              char *error, size_t error_size);

/* Same without progress reporting. */
int qzen_qwen21_pipeline_run(qzen_qwen21_model *dit, qzen_qwen21_vae *vae,
                           const qzen_qwen21_sample *sample, unsigned char *rgb, char *error,
                           size_t error_size);

#endif
