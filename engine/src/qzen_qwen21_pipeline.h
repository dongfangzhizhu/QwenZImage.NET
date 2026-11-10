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

/* WanGP denoising_strength -> first denoise step to re-run:
 * min(int(steps * (1 - strength)), steps - 1). strength 1.0 gives 0
 * (text-to-image), the smallest strengths still leave one step of work.
 * `sigmas` is accepted for upstream source compatibility but not used here. */
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
    /* WanGP-style masked inpainting (mode 0, "Masked Denoising"). The mask is
     * one byte per LATENT cell [height*width]: 1 = regenerate (white in the
     * user's mask), 0 = keep the reference trajectory (black). During the
     * first ceil((steps - first_step) * masking_strength) denoise steps the
     * 0 cells are reset to
     *     known = (1 - sigmas[step+1]) * init_latent + sigmas[step+1] * noise,
     * exactly like models/qwen21/pipeline.py. Requires init_latent.
     * Both fields are appended, so existing positional initialisers keep their
     * meaning and a NULL / 0 pair is a plain img2img edit. */
    const uint8_t *latent_mask;
    float masking_strength;
    /* Advanced WanGP editing, additive in the same struct. All four fields
     * default to zero/NULL for existing positional initialisers.
     *
     * WanGP's edit pipeline is a CONDITION-PREFIX model: the reference latent
     * is concatenated in front of the noised target and the DiT attends to it,
     * so outpainting can start from pure noise and still keep the source. This
     * engine is an img2img sampler instead -- the reference IS the starting
     * latent -- so Red Canvas outpainting is approximated by:
     *   - step_offset = 0 (the full schedule, like WanGP's red margins);
     *   - masked cells (margins + painted interior) start from full noise,
     *     kept cells start at sigmas[first_step] (retain the source);
     *   - mask == 0 cells are pinned to the known trajectory, which is what
     *     preserves the source content without a condition prefix.
     * See qzen_qwen21_pipeline.c for the exact formulas. */
    int outpainting;        /* 1 = Red Canvas margins, run from step 0 */
    int lanpaint_steps;     /* 0 = off; 2/5/10/15 inner Langevin iterations */
    uint64_t lanpaint_seed; /* deterministic stream for the Langevin noise */
    unsigned char *alpha;   /* optional [1][16*height][16*width] RGBA plane */
} qzen_qwen21_sample;

/* Phase progress for the resident service: "dit" with (step, steps) before each
 * denoise step, then "vae" with (0, 1) before the decode. */
typedef void (*qzen_qwen21_progress)(const char *phase, int completed, int total,
                                   void *opaque);

/* Denoise and decode; rgb is [3][16*height][16*width] (the DiT grid is already
 * at 1/16 resolution, so this is the requested canvas). on_progress may be NULL. */
int qzen_qwen21_pipeline_run_ex(qzen_qwen21_model *dit, qzen_qwen21_vae *vae,
                              const qzen_qwen21_sample *sample, unsigned char *rgb,
                              qzen_qwen21_progress on_progress, void *opaque,
                              char *error, size_t error_size);

/* Same without progress reporting. */
int qzen_qwen21_pipeline_run(qzen_qwen21_model *dit, qzen_qwen21_vae *vae,
                           const qzen_qwen21_sample *sample, unsigned char *rgb, char *error,
                           size_t error_size);

#endif
