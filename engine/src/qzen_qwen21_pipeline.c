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

/* Edit start = WanGP's denoising_strength rule:
 *     first_step = min(int(steps * (1 - strength)), steps - 1)
 * strength 1.0 is text-to-image (no step skipped); smaller values start that
 * many steps closer to the end. The earlier upstream rule searched for the
 * first sigma <= strength, which is NOT equivalent: the turbo schedule is
 * highly non-linear, so strength 0.75 landed on step 6 of 8 (only two steps
 * of work) instead of WanGP's step 2. Keeping the step-based rule also makes
 * the reference backend's phase trace agree with the real engine.
 * `sigmas` is kept in the signature for source compatibility with the vendored
 * upstream API; the schedule is still used for the blend and the Euler loop. */
int qzen_qwen21_edit_start(const float *sigmas, int steps, float strength) {
    (void)sigmas;
    if (steps <= 0) return 0;
    if (!(strength > 0.0f) || strength > 1.0f) strength = 1.0f;
    int first_step = (int)((double)steps * (1.0 - (double)strength));
    if (first_step >= steps) first_step = steps - 1;
    if (first_step < 0) first_step = 0;
    return first_step;
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

/* ---------------------------------------------------------------- LanPaint */
/* C port of WanGP shared/lanpaint.py (Algorithms 1-4): the Bidirectional
 * Guided (BiG) score drives a few Langevin iterations before each sampler
 * step, with each iteration solved exactly as a stochastic harmonic
 * oscillator. No extra weights are needed -- this is purely a sampler change.
 *
 * The Python code packs the latent into 2x2 spatial blocks first, but every
 * operation involved (target, oscillate, mask, randn) is elementwise, so
 * pack/unpack is a permutation that only reorders the RNG stream. Running
 * directly on the channel-major grid is statistically identical.
 *
 * With WanGP's defaults (Friction=15, Lambda=8, Beta=1, StepSize=0.15),
 * d = 1 - 4a/g = 1 - 4/Friction^2 > 0 for BOTH regions, so l1/l2 are real
 * and the .real parts of _oscillator() reduce exactly to the closed form
 * below. The complex underdamped branch is unreachable and clamped here. */

typedef struct {
    double uu, up, pu, pp, su, c, sp;
} q21_osc;

typedef struct {
    uint64_t state;
    double spare;
    int has_spare;
} q21_rng;

static double q21_pick(double vx, double vy, double mask) {
    return mask * (vx - vy) + vy;
}

static double q21_phi(double z) {
    if (fabs(z) > 1e-6) return (exp(z) - 1.0) / z;
    return 1.0 + z / 2.0 + z * z / 6.0;
}

static q21_osc q21_oscillator(double a, double g, double f) {
    double d = 1.0 - 4.0 * a / g;
    if (d <= 1e-8) d = 1e-8; /* critical damping is the continuous limit */
    const double root = sqrt(d);
    const double l1 = -f * g * (1.0 - root) / 2.0;
    const double l2 = -f * g * (1.0 + root) / 2.0;
    const double e1 = exp(l1), e2 = exp(l2);
    const double dl = l1 - l2;
    q21_osc o;
    o.uu = (l1 * e2 - l2 * e1) / dl;
    o.up = f * (e1 - e2) / dl;
    o.pu = -g * a * f * (e1 - e2) / dl;
    o.pp = (l1 * e1 - l2 * e2) / dl;
    const double var_u = 2.0 * f * a / d *
        (q21_phi(2.0 * l1) - 2.0 * q21_phi(l1 + l2) + q21_phi(2.0 * l2));
    const double cov = a / d * (e1 - e2) * (e1 - e2);
    const double sum = l1 + l2;
    const double var_p = 2.0 * a / (f * d) *
        (l1 * (e1 * e1 - 1.0) / 2.0 -
         2.0 * l1 * l2 * (e1 * e2 - 1.0) / sum +
         l2 * (e2 * e2 - 1.0) / 2.0);
    o.su = sqrt(fmax(var_u, 0.0));
    o.c = o.su > 0.0 ? cov / o.su : 0.0;
    o.sp = sqrt(fmax(var_p - o.c * o.c, 0.0));
    return o;
}

static double q21_rng_normal(q21_rng *rng) {
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint64_t a = rng->state;
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint64_t b = rng->state;
    const double u1 = (double)((a >> 11) + 1) / 9007199254740993.0;
    const double u2 = (double)(b >> 11) / 9007199254740992.0;
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.283185307179586 * u2;
    rng->spare = radius * sin(angle);
    rng->has_spare = 1;
    return radius * cos(angle);
}

/* target(r): the BiG score's oscillator mean m = C/A. WanGP forces
 * cfg_BIG = 1 for Flux-style models, so v_big is the raw conditional
 * prediction while v is the ordinary CFG combination. */
static int q21_lp_target(qzen_qwen21_model *dit, const qzen_qwen21_sample *sample,
                         const float *r, float sigma, int tokens,
                         float *positive, float *negative, float *m,
                         char *error, size_t error_size) {
    if (!q21_velocity(dit, sample->text, sample->text_tokens, r, sample->height,
                      sample->width, sigma, positive, error, error_size))
        return 0;
    const int has_neg = negative != NULL;
    if (has_neg &&
        !q21_velocity(dit, sample->negative, sample->negative_tokens, r,
                      sample->height, sample->width, sigma, negative, error,
                      error_size))
        return 0;
    const double s = (double)sigma;
    const double signal = 1.0 - s;
    const double lam = 8.0;
    const double known_scale = lam / (1.0 + lam) * s;
    for (int c = 0; c < Q21_LATENT_CHANNELS; c++) {
        for (int t = 0; t < tokens; t++) {
            const size_t ti = (size_t)t * Q21_LATENT_CHANNELS + c;
            const size_t ci = (size_t)c * tokens + t;
            const double cond = (double)positive[ti];
            const double v = has_neg
                ? (double)negative[ti] +
                      (double)sample->guidance * (cond - (double)negative[ti])
                : cond;
            const double x0 = (double)r[ci] - s * v;
            const double eps = (double)r[ci] + (1.0 - s) * cond;
            const double generated = signal * x0;
            const double kept = signal * (double)sample->init_latent[ci] +
                                known_scale * eps;
            const double mask = sample->latent_mask[t] ? 1.0 : 0.0;
            m[ci] = (float)q21_pick(generated, kept, mask);
        }
    }
    return 1;
}

/* One exact oscillator transition with region-dependent coefficients. */
static void q21_lp_oscillate(float *r, float *p, const float *m,
                             const uint8_t *mask, int tokens,
                             const q21_osc *ox, const q21_osc *oy,
                             double scale_y, double s, q21_rng *rng) {
    for (int c = 0; c < Q21_LATENT_CHANNELS; c++) {
        for (int t = 0; t < tokens; t++) {
            const size_t i = (size_t)c * tokens + t;
            const double mv = mask[t] ? 1.0 : 0.0;
            const double uu = q21_pick(ox->uu, oy->uu, mv);
            const double up = q21_pick(ox->up, oy->up, mv);
            const double pu = q21_pick(ox->pu, oy->pu, mv);
            const double pp = q21_pick(ox->pp, oy->pp, mv);
            const double su = q21_pick(ox->su, oy->su * scale_y, mv) * s;
            const double cc = q21_pick(ox->c, oy->c * scale_y, mv) * s;
            const double sp = q21_pick(ox->sp, oy->sp * scale_y, mv) * s;
            const double u = (double)r[i] - (double)m[i];
            const double n1 = q21_rng_normal(rng);
            const double n2 = q21_rng_normal(rng);
            const double next_r = (double)m[i] + uu * u +
                                  up * (double)p[i] + su * n1;
            const double next_p = pu * u + pp * (double)p[i] +
                                  cc * n1 + sp * n2;
            r[i] = (float)next_r;
            p[i] = (float)next_p;
        }
    }
}

/* NSteps Langevin iterations on `grid` before one Euler step. r/p/m/m_new
 * are caller-owned scratch buffers of [64][tokens]. */
static int q21_lp_refine(qzen_qwen21_model *dit, const qzen_qwen21_sample *sample,
                         float *grid, int tokens, float sigma, int step,
                         float *r, float *p, float *m, float *m_new,
                         float *positive, float *negative,
                         char *error, size_t error_size) {
    const double eta = 0.15;
    const double friction = 15.0;
    const double lam = 8.0;
    const double s = (double)sigma;
    const double signal = 1.0 - s;
    const double ax = eta, gx = friction * friction * eta;
    const double ay = eta * (1.0 + lam), gy = friction * friction * ay;
    const double scale_y = 1.0 / sqrt(1.0 + lam);
    const q21_osc o1x = q21_oscillator(ax, gx, 1.0);
    const q21_osc o1y = q21_oscillator(ay, gy, 1.0);
    const q21_osc ohx = q21_oscillator(ax, gx, 0.5);
    const q21_osc ohy = q21_oscillator(ay, gy, 0.5);
    const double sqrt_ag_x = sqrt(ax * gx);
    const double sqrt_ag_y = sqrt(ay * gy) * scale_y;
    const double ag_x = ax * gx, ag_y = ay * gy;
    q21_rng rng = {sample->lanpaint_seed ^
                       ((uint64_t)step + 1u) * 0x9E3779B97F4A7C15ULL,
                   0.0, 0};

    /* r = lerp(clean + s * noise, x, mask); the known region restarts from
     * its pre-diffused value on every sampler step. */
    for (int c = 0; c < Q21_LATENT_CHANNELS; c++) {
        for (int t = 0; t < tokens; t++) {
            const size_t i = (size_t)c * tokens + t;
            const double known = signal * (double)sample->init_latent[i] +
                                 s * (double)sample->latent[i];
            const double mv = sample->latent_mask[t] ? 1.0 : 0.0;
            r[i] = (float)q21_pick((double)grid[i], known, mv);
        }
    }
    int have_m = 0;
    for (int iteration = 0; iteration < sample->lanpaint_steps; iteration++) {
        if (!have_m) {
            /* First order: draw p from its stationary law, one full step. */
            if (!q21_lp_target(dit, sample, r, sigma, tokens, positive,
                               negative, m, error, error_size))
                return 0;
            for (int c = 0; c < Q21_LATENT_CHANNELS; c++) {
                for (int t = 0; t < tokens; t++) {
                    const size_t i = (size_t)c * tokens + t;
                    const double mv = sample->latent_mask[t] ? 1.0 : 0.0;
                    p[i] = (float)(q21_pick(sqrt_ag_x, sqrt_ag_y, mv) * s *
                                   q21_rng_normal(&rng));
                }
            }
            q21_lp_oscillate(r, p, m, sample->latent_mask, tokens, &o1x, &o1y,
                             scale_y, s, &rng);
            have_m = 1;
        } else {
            /* Second order Strang splitting with one midpoint evaluation. */
            q21_lp_oscillate(r, p, m, sample->latent_mask, tokens, &ohx, &ohy,
                             scale_y, s, &rng);
            if (!q21_lp_target(dit, sample, r, sigma, tokens, positive,
                               negative, m_new, error, error_size))
                return 0;
            for (int c = 0; c < Q21_LATENT_CHANNELS; c++) {
                for (int t = 0; t < tokens; t++) {
                    const size_t i = (size_t)c * tokens + t;
                    const double mv = sample->latent_mask[t] ? 1.0 : 0.0;
                    p[i] = (float)((double)p[i] +
                                   q21_pick(ag_x, ag_y, mv) *
                                       ((double)m_new[i] - (double)m[i]));
                    m[i] = m_new[i];
                }
            }
            q21_lp_oscillate(r, p, m, sample->latent_mask, tokens, &ohx, &ohy,
                             scale_y, s, &rng);
        }
    }
    memcpy(grid, r, (size_t)Q21_LATENT_CHANNELS * tokens * sizeof(float));
    return 1;
}
/* ------------------------------------------------------------ /LanPaint */

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
    if (sample->latent_mask && !sample->init_latent) {
        snprintf(error, error_size,
                 "a latent mask requires an encoded reference image");
        return 0;
    }
    if (sample->lanpaint_steps > 0 &&
        (!sample->latent_mask || !sample->init_latent)) {
        snprintf(error, error_size,
                 "LanPaint requires both an edit mask and an encoded reference");
        return 0;
    }
    const int tokens = sample->height * sample->width;
    const float *mean = qzen_qwen21_vae_latent_mean(vae);
    const float *std = qzen_qwen21_vae_latent_std(vae);
    int ok = 0;
    const size_t sample_bytes =
        (size_t)Q21_LATENT_CHANNELS * tokens * sizeof(float);
    float *grid = malloc(sample_bytes);
    /* tokens_major doubles as the LanPaint momentum scratch. The sampler's
     * DiT consumes the channel-major grid directly, so repurposing this
     * historical staging buffer costs nothing. */
    float *tokens_major = malloc(sample_bytes);
    float *positive = malloc(sample_bytes);
    float *negative = sample->negative && sample->negative_tokens > 0
                          ? malloc(sample_bytes)
                          : NULL;
    float *lp_r = NULL, *lp_m = NULL, *lp_mnew = NULL;
    if (sample->lanpaint_steps > 0) {
        lp_r = malloc(sample_bytes);
        lp_m = malloc(sample_bytes);
        lp_mnew = malloc(sample_bytes);
    }
    if (!grid || !tokens_major || !positive ||
        (sample->negative && sample->negative_tokens > 0 && !negative) ||
        (sample->lanpaint_steps > 0 && (!lp_r || !lp_m || !lp_mnew))) {
        snprintf(error, error_size, "out of memory for the %dx%d sampler", sample->height,
                 sample->width);
        goto done;
    }
    memcpy(grid, sample->latent, sample_bytes);
    /* Edit start is WanGP's step ratio (qzen_qwen21_edit_start), not a sigma
     * threshold: blend the reference latent into the noise at that schedule
     * point. strength 1.0 starts at step 0 where sigmas[0] == 1.0, so the blend
     * is exactly the noise and a text-to-image request takes the same code
     * path with no arithmetic added (0 * init + 1 * noise). */
    const int outpainting = sample->outpainting && sample->init_latent != NULL;
    const int lanpaint = sample->lanpaint_steps > 0;
    int first_step = 0;
    if (sample->init_latent) {
        first_step = qzen_qwen21_edit_start(sample->sigmas, sample->steps, sample->strength);
        if (outpainting) {
            /* Red Canvas on an img2img sampler: WanGP starts from pure noise
             * because its DiT sees the red canvas as a condition PREFIX. We
             * have no prefix, so instead we start the masked cells (new
             * margins + painted interior) from full noise at sigmas[0] and
             * the kept source cells at sigmas[first_step]. That gives the
             * margins the whole schedule while still preserving the source. */
            const float keep = sample->sigmas[first_step];
            const float full = sample->sigmas[0];
            for (int channel = 0; channel < Q21_LATENT_CHANNELS; channel++) {
                for (int token = 0; token < tokens; token++) {
                    const size_t index = (size_t)channel * tokens + token;
                    const float sigma = (sample->latent_mask &&
                                         sample->latent_mask[token]) ? full : keep;
                    grid[index] = (1.0f - sigma) * sample->init_latent[index] +
                                  sigma * sample->latent[index];
                }
            }
        } else {
            const float sigma = sample->sigmas[first_step];
            for (size_t index = 0; index < (size_t)Q21_LATENT_CHANNELS * tokens; index++) {
                grid[index] = (1.0f - sigma) * sample->init_latent[index] +
                              sigma * sample->latent[index];
            }
        }
    }
    /* WanGP: step_offset = 0 for instruction (Red Canvas) outpainting, so the
     * red margins always run the full schedule. LanPaint forces strength 1.0,
     * hence first_step == 0 and this is a no-op for that mode. */
    const int step_offset = outpainting ? 0 : first_step;
    /* WanGP masked_steps = ceil((steps - first_step) * masking_strength):
     * for that many steps the BLACK (mask == 0) latent cells are pinned to the
     * reference trajectory, while the WHITE cells are free to be regenerated.
     * The ABI layer normalises masking_strength to 1.0 when a mask is present;
     * the fallback keeps direct engine callers from silently dropping it. */
    int masked_steps = 0;
    if (sample->latent_mask) {
        float fraction = sample->masking_strength;
        if (!(fraction > 0.0f) || fraction > 1.0f) fraction = 1.0f;
        const int remaining = sample->steps - first_step;
        masked_steps = (int)ceil((double)remaining * (double)fraction);
        if (masked_steps > remaining) masked_steps = remaining;
        if (masked_steps < 1) masked_steps = 1;
    }
    for (int step = step_offset; step < sample->steps; step++) {
        const float sigma = sample->sigmas[step];
        const float next = sample->sigmas[step + 1];
        if (on_progress)
            on_progress("dit", step - step_offset, sample->steps - step_offset, opaque);
        /* WanGP refines the inpainted region with LanPaint before every
         * sampler step except the last one. Each inner iteration costs one
         * (turbo) or two (base CFG) DiT evaluations. */
        if (lanpaint && step < sample->steps - 1 &&
            !q21_lp_refine(dit, sample, grid, tokens, sigma, step,
                           lp_r, tokens_major, lp_m, lp_mnew,
                           positive, negative, error, error_size)) {
            goto done;
        }
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
        /* WanGP: known = original * (1 - sigma_next) + noise * sigma_next;
         * latents = known * (1 - mask) + latents * mask. The grid is channel
         * major, the mask is token major [height*width]. Only the first
         * masked_steps iterations pin the unmasked cells; after that the whole
         * image evolves freely so the regenerated area blends in. */
        if (sample->latent_mask && step < first_step + masked_steps) {
            const float keep = 1.0f - next;
            for (int token = 0; token < tokens; token++) {
                if (sample->latent_mask[token]) continue;
                for (int channel = 0; channel < Q21_LATENT_CHANNELS; channel++) {
                    const size_t index = (size_t)channel * tokens + token;
                    grid[index] = keep * sample->init_latent[index] +
                                  next * sample->latent[index];
                }
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
    if (!qzen_qwen21_vae_decode_ex(vae, grid, sample->height, sample->width, rgb,
                                   sample->alpha, error, error_size)) {
        goto done;
    }
    if (on_progress) on_progress("vae", 1, 1, opaque);
    ok = 1;
done:
    free(lp_mnew);
    free(lp_m);
    free(lp_r);
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
