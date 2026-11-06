/* Qwen-Image 2.1 sampling loop: schedule, CFG, Euler, latent denormalisation,
 * VAE decode. See qzen_qwen21_pipeline.h for the formulas and their sources. */

#include "qzen_qwen21_pipeline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q21_BASE_IMAGE_SEQ_LEN 256
#define Q21_MAX_IMAGE_SEQ_LEN 8192
#define Q21_BASE_SHIFT 0.5
#define Q21_MAX_SHIFT 0.9
#define Q21_SHIFT_TERMINAL 0.02
#define Q21_LATENT_CHANNELS 64

static const float q21_turbo_sigmas[QZEN_QWEN21_TURBO_STEPS] = {
    1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f,
};

int qzen_qwen21_sigmas_turbo(float *sigmas) {
    if (!sigmas) return 0;
    for (int index = 0; index < QZEN_QWEN21_TURBO_STEPS; index++) {
        sigmas[index] = q21_turbo_sigmas[index];
    }
    sigmas[QZEN_QWEN21_TURBO_STEPS] = 0.0f;
    return QZEN_QWEN21_TURBO_STEPS + 1;
}

/* Edit start: the first schedule point at or below `strength`. strength 1.0
 * lands on sigmas[0] == 1.0, where the blend below is exactly the noise, so a
 * text-to-image request takes the same path with nothing added to the result.
 * When no sigma is low enough (a strength below the last step's sigma) the
 * answer is the last step: an edit that starts at the last step is one step of
 * work, which is the closest thing to "change almost nothing" the schedule can
 * offer. Returning 0 there would silently run the whole schedule instead. */
int qzen_qwen21_edit_start(const float *sigmas, int steps, float strength) {
    if (!sigmas || steps <= 0) return 0;
    if (!(strength > 0.0f) || strength > 1.0f) strength = 1.0f;
    for (int step = 0; step < steps; step++) {
        if (sigmas[step] <= strength) return step;
    }
    return steps - 1;
}

int qzen_qwen21_sigmas_base(int steps, int image_tokens, float *sigmas) {
    if (!sigmas || steps <= 0 || image_tokens <= 0) return 0;
    const double slope = (Q21_MAX_SHIFT - Q21_BASE_SHIFT) /
                         (double)(Q21_MAX_IMAGE_SEQ_LEN - Q21_BASE_IMAGE_SEQ_LEN);
    const double mu = (double)(image_tokens - Q21_BASE_IMAGE_SEQ_LEN) * slope + Q21_BASE_SHIFT;
    const double shift = exp(mu);
    double shifted[64];
    if (steps >= (int)(sizeof(shifted) / sizeof(shifted[0]))) {
        return 0;
    }
    for (int index = 0; index < steps; index++) {
        double base = steps == 1 ? 1.0 : 1.0 - (double)index * (1.0 - 1.0 / (double)steps) /
                                                     (double)(steps - 1);
        shifted[index] = shift * base / (1.0 + (shift - 1.0) * base);
    }
    const double scale = (1.0 - shifted[steps - 1]) / (1.0 - Q21_SHIFT_TERMINAL);
    for (int index = 0; index < steps; index++) {
        sigmas[index] = (float)(1.0 - (1.0 - shifted[index]) / scale);
    }
    sigmas[steps] = 0.0f;
    return steps + 1;
}

/* One DiT evaluation for one prompt, returning velocity [tokens][64]. */
static int q21_velocity(qzen_qwen21_model *dit, const float *text, int text_tokens,
                        const float *latent_grid, int height, int width, float sigma,
                        float *velocity, char *error, size_t error_size) {
    const qzen_qwen21_request request = {text, text_tokens, latent_grid, height, width, &sigma, 1};
    return qzen_qwen21_forward(dit, &request, velocity, NULL, error, error_size);
}

int qzen_qwen21_pipeline_run_ex(qzen_qwen21_model *dit, qzen_qwen21_vae *vae,
                              const qzen_qwen21_sample *sample, unsigned char *rgb,
                              qzen_qwen21_progress on_progress, void *opaque,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !vae || !sample || !rgb || !sample->text || !sample->latent || !sample->sigmas ||
        sample->steps <= 0 || sample->height <= 0 || sample->width <= 0) {
        snprintf(error, error_size, "invalid Qwen-Image 2.1 sampling request");
        return 0;
    }
    const int tokens = sample->height * sample->width;
    const float *mean = qzen_qwen21_vae_latent_mean(vae);
    const float *std = qzen_qwen21_vae_latent_std(vae);
    int ok = 0;
    float *grid = malloc((size_t)Q21_LATENT_CHANNELS * tokens * sizeof(float));
    float *tokens_major = malloc((size_t)tokens * Q21_LATENT_CHANNELS * sizeof(float));
    float *positive = malloc((size_t)tokens * Q21_LATENT_CHANNELS * sizeof(float));
    float *negative = sample->negative && sample->negative_tokens > 0
                          ? malloc((size_t)tokens * Q21_LATENT_CHANNELS * sizeof(float))
                          : NULL;
    if (!grid || !tokens_major || !positive ||
        (sample->negative && sample->negative_tokens > 0 && !negative)) {
        snprintf(error, error_size, "out of memory for the %dx%d sampler", sample->height,
                 sample->width);
        goto done;
    }
    memcpy(grid, sample->latent, (size_t)Q21_LATENT_CHANNELS * tokens * sizeof(float));
    /* Edit start: pick the first schedule point at or below `strength`, then
     * blend the reference latent into the noise at that sigma. strength 1.0
     * lands on sigmas[0] == 1.0, where the blend is exactly the noise, so a
     * text-to-image request takes the same code path with no arithmetic added
     * to the result (0 * init + 1 * noise). */
    int first_step = 0;
    if (sample->init_latent) {
        first_step = qzen_qwen21_edit_start(sample->sigmas, sample->steps, sample->strength);
        const float sigma = sample->sigmas[first_step];
        for (size_t index = 0; index < (size_t)Q21_LATENT_CHANNELS * tokens; index++) {
            grid[index] = (1.0f - sigma) * sample->init_latent[index] +
                          sigma * sample->latent[index];
        }
    }
    for (int step = first_step; step < sample->steps; step++) {
        const float sigma = sample->sigmas[step];
        const float next = sample->sigmas[step + 1];
        if (on_progress)
            on_progress("dit", step - first_step, sample->steps - first_step, opaque);
        /* channel major -> token major for the DiT */
        for (int token = 0; token < tokens; token++) {
            for (int channel = 0; channel < Q21_LATENT_CHANNELS; channel++) {
                tokens_major[(size_t)token * Q21_LATENT_CHANNELS + channel] =
                    grid[(size_t)channel * tokens + token];
            }
        }
        if (!q21_velocity(dit, sample->text, sample->text_tokens, grid, sample->height,
                          sample->width, sigma, positive, error, error_size)) {
            goto done;
        }
        if (negative) {
            if (!q21_velocity(dit, sample->negative, sample->negative_tokens, grid, sample->height,
                              sample->width, sigma, negative, error, error_size)) {
                goto done;
            }
            for (size_t index = 0; index < (size_t)tokens * Q21_LATENT_CHANNELS; index++) {
                positive[index] =
                    negative[index] + sample->guidance * (positive[index] - negative[index]);
            }
        }
        const float delta = next - sigma;
        for (int token = 0; token < tokens; token++) {
            for (int channel = 0; channel < Q21_LATENT_CHANNELS; channel++) {
                const size_t index = (size_t)token * Q21_LATENT_CHANNELS + channel;
                grid[(size_t)channel * tokens + token] += delta * positive[index];
            }
        }
    }
    /* z * std + mean, channel major, then the decoder */
    for (int channel = 0; channel < Q21_LATENT_CHANNELS; channel++) {
        const float centre = mean ? mean[channel] : 0.0f;
        const float spread = std ? std[channel] : 1.0f;
        for (int token = 0; token < tokens; token++) {
            grid[(size_t)channel * tokens + token] =
                grid[(size_t)channel * tokens + token] * spread + centre;
        }
    }
    if (on_progress) on_progress("vae", 0, 1, opaque);
    if (!qzen_qwen21_vae_decode(vae, grid, sample->height, sample->width, rgb, error,
                              error_size)) {
        goto done;
    }
    if (on_progress) on_progress("vae", 1, 1, opaque);
    ok = 1;
done:
    free(grid);
    free(tokens_major);
    free(positive);
    free(negative);
    return ok;
}

int qzen_qwen21_pipeline_run(qzen_qwen21_model *dit, qzen_qwen21_vae *vae,
                           const qzen_qwen21_sample *sample, unsigned char *rgb, char *error,
                           size_t error_size) {
    return qzen_qwen21_pipeline_run_ex(dit, vae, sample, rgb, NULL, NULL, error,
                                     error_size);
}
