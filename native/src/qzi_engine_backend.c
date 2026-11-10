/*
 * qzi_engine_backend.c -- real Qwen-Image 2.1 / Z-Image backend.
 *
 * Compiled into QwenZImageNative.dll / libQwenZImageNative.so only when
 * -DQZI_WITH_CUDA_ENGINE=ON. It implements the qzi_backend vtable by calling
 * the vendored C/CUDA engine in engine/ directly -- in-process, no HTTP, no
 * Go, no JSON hop. qzi.c still owns validation, single-flight serialisation,
 * progress plumbing and buffer ownership.
 *
 * Orchestration is a direct port of the upstream resident service's
 * serve_execute_qwen21() / serve_execute_zimage() paths, minus the HTTP
 * server state: the DiT stays resident across calls, the text tower is
 * attached lazily, and PNG bytes are produced in memory rather than written
 * to a temp file and read back.
 */

#include "qzi.h"
#include "qzi_backend.h"
#include "qzi_internal.h"
#include "qzi_png.h"

#include "qzen_config.h"
#include "qzen_tokenizer.h"
#include "qzen_qwen21_dit.h"
#include "qzen_qwen21_text.h"
#include "qzen_qwen21_vae.h"
#include "qzen_qwen21_pipeline.h"
#include "qzen_zimage_pipeline.h"
#include "qzen_zimage_vae.h"
#include "qzen_zimage_control.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
/* On Windows the vendored engine's win32 shim supplies <dirent.h> (it is
 * force-included via /FI by qzen_apply_platform_flags and lives in the shim's
 * include directory). Keep the include unconditional: resolve_zimage_pack()
 * enumerates SVDQ packs on every platform. */
#include <dirent.h>

#define QZEN_PATH_CAP 1024
#define QZEN_ERR_CAP  512

/* ------------------------------------------------------------------ state */

typedef struct qzen_state {
    /* Paths resolved from QwenZImage.config.json / the qzi_engine_config. */
    char qwen21_dir[QZEN_PATH_CAP];
    char qwen21_lora_dir[QZEN_PATH_CAP];
    char zimage_dir[QZEN_PATH_CAP];
    char zimage_nunchaku_dir[QZEN_PATH_CAP];
    char zimage_control_dir[QZEN_PATH_CAP];  /* ControlNet Union 2.1 weights */
    char text_cache_dir[QZEN_PATH_CAP];
    int  device_id;

    /* Qwen-Image 2.1 pack. */
    char q_turbo[QZEN_PATH_CAP];
    char q_base[QZEN_PATH_CAP];
    char q_vae[QZEN_PATH_CAP];
    char q_text[QZEN_PATH_CAP];
    char q_tokenizer[QZEN_PATH_CAP];
    int  q_paths_resolved;
    qzen_tokenizer     *tokenizer;
    qzen_qwen21_text   *text;
    qzen_qwen21_model  *dit;
    qzen_qwen21_vae    *vae;
    int   tier_resident;        /* -1 none, 0 base, 1 turbo */
    /* QZI_LORA_NAME_MAX + 1: never truncate an adapter name, otherwise two
     * different adapters sharing a prefix would look "resident" and the
     * wrong weights would stay merged. */
    char  lora_resident[QZI_LORA_NAME_MAX + 1];
    float lora_scale_resident;
    float *uncond;
    int    uncond_tokens;

    /* Z-Image pack. */
    char z_dit[QZEN_PATH_CAP];
    char z_vae[QZEN_PATH_CAP];
    char z_text[QZEN_PATH_CAP];
    char z_pack_file[QZEN_PATH_CAP];
    char z_pack_id[64];
    int  z_pack;                /* 0 none, -1 bf16, 1 nunchaku resident */
    qzen_zimage_pipeline *zpipe;
    int  z_text_resident;
    /* ControlNet Union weights and the DiT they were uploaded to. Releasing
     * the DiT invalidates the device copies, so both are dropped together. */
    qzen_zimage_control *z_control;
    qzen_zimage_pipeline *z_control_pipe;

    qzi_model_kind resident;
    uint64_t served;
    double   started;
} qzen_state;

/* --------------------------------------------------------------- utilities */

static void copy_path(char *out, size_t cap, const char *value)
{
    if (!value || !*value) { out[0] = '\0'; return; }
    snprintf(out, cap, "%s", value);
}

static int join_path(char *out, size_t cap, const char *dir, const char *name)
{
    int n = snprintf(out, cap, "%s/%s", dir, name);
    return n > 0 && (size_t)n < cap;
}

static int is_regular(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static int is_directory(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
}

/* A pack may be laid out as <pack>/<component>/ or as flat files directly in
 * <pack>. Prefer the subdirectory, fall back to the pack itself -- the same
 * rule the upstream service uses. */
static int component_dir(const char *pack, const char *name,
                         char *out, size_t cap)
{
    char sub[QZEN_PATH_CAP];
    if (!join_path(sub, sizeof(sub), pack, name)) return 0;
    if (is_directory(sub)) return join_path(out, cap, pack, name);
    return copy_path(out, cap, pack), 1;
}

static int first_file(const char *dir, const char *const *names, size_t count,
                      char *out, size_t cap)
{
    for (size_t i = 0; i < count; i++) {
        char path[QZEN_PATH_CAP];
        if (!join_path(path, sizeof(path), dir, names[i])) continue;
        if (is_regular(path)) return copy_path(out, cap, path), 1;
    }
    return 0;
}

/* The engine's GPU paths are opt-in through environment switches. An explicit
 * 0 in the environment is respected so small cards can stay on the CPU. */
static void enable_env(const char *name, const char *value)
{
    const char *current = getenv(name);
    if (current && *current) return;
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void enable_gpu_switches(void)
{
    static const char *const names[] = {
        "QWEN21_GPU", "QWEN21_VAE_GPU", "QWEN21_TEXT_GPU",
        "ZIMAGE_GPU", "ZIMAGE_VAE_GPU"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        enable_env(names[i], "1");
}

/* splitmix64 + Box-Muller, byte-for-byte the upstream sampler's noise so a
 * given seed reproduces the service's image. */
static void fill_noise(uint64_t seed, float *out, size_t count)
{
    uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    for (size_t i = 0; i < count; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u1 = (double)((state >> 11) + 1) / 9007199254740993.0;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (double)(state >> 11) / 9007199254740992.0;
        double radius = sqrt(-2.0 * log(u1));
        out[i] = (float)(radius * cos(6.283185307179586 * u2));
    }
}

/* Planar RGB8 [3][H][W] -> interleaved [H][W][3] for the PNG encoder. */
static void pack_planar_rgb8(const uint8_t *planar, uint32_t width,
                             uint32_t height, uint8_t *packed)
{
    size_t pixels = (size_t)width * height;
    for (size_t i = 0; i < pixels; i++) {
        packed[i * 3 + 0] = planar[i];
        packed[i * 3 + 1] = planar[pixels + i];
        packed[i * 3 + 2] = planar[2 * pixels + i];
    }
}

/* ------------------------------------------------------------ image edits */

/* WanGP's cover fit: scale the reference to FILL the output canvas, then crop
 * the overflow equally from both sides (shared/utils/utils.py:693-749 and the
 * upstream service's H3_IMAGE_FIT_COVER). Stretching a 4:3 photo onto a square
 * canvas would change the composition before the model ever sees it.
 * `src` is planar [channels][sw*sh] uint8; `dst` is [channels][dw*dh].
 * Bilinear sampling keeps this dependency-free (no libpng / ffmpeg here). */
static int resample_planar_into(const uint8_t *src, uint32_t sw, uint32_t sh,
                                uint8_t *dst, uint32_t canvas_w, uint32_t canvas_h,
                                uint32_t dst_x, uint32_t dst_y,
                                uint32_t dw, uint32_t dh, uint32_t channels)
{
    if (!src || !dst || !sw || !sh || !dw || !dh || !channels ||
        dst_x + dw > canvas_w || dst_y + dh > canvas_h) return 0;
    const double scale = fmax((double)dw / (double)sw,
                              (double)dh / (double)sh);
    const double crop_x = ((double)sw * scale - (double)dw) * 0.5;
    const double crop_y = ((double)sh * scale - (double)dh) * 0.5;
    const size_t src_plane = (size_t)sw * sh;
    const size_t dst_plane = (size_t)canvas_w * canvas_h;

    for (uint32_t y = 0; y < dh; y++) {
        double sy = ((double)y + crop_y + 0.5) / scale - 0.5;
        if (sy < 0.0) sy = 0.0;
        if (sy > (double)sh - 1.0) sy = (double)sh - 1.0;
        const uint32_t y0 = (uint32_t)sy;
        const uint32_t y1 = y0 + 1 < sh ? y0 + 1 : y0;
        const double fy = sy - (double)y0;
        for (uint32_t x = 0; x < dw; x++) {
            double sx = ((double)x + crop_x + 0.5) / scale - 0.5;
            if (sx < 0.0) sx = 0.0;
            if (sx > (double)sw - 1.0) sx = (double)sw - 1.0;
            const uint32_t x0 = (uint32_t)sx;
            const uint32_t x1 = x0 + 1 < sw ? x0 + 1 : x0;
            const double fx = sx - (double)x0;
            const double w00 = (1.0 - fx) * (1.0 - fy);
            const double w10 = fx * (1.0 - fy);
            const double w01 = (1.0 - fx) * fy;
            const double w11 = fx * fy;
            const size_t dst_index =
                (size_t)(dst_y + y) * canvas_w + dst_x + x;
            for (uint32_t c = 0; c < channels; c++) {
                const uint8_t *plane = src + (size_t)c * src_plane;
                const double value =
                    (double)plane[(size_t)y0 * sw + x0] * w00 +
                    (double)plane[(size_t)y0 * sw + x1] * w10 +
                    (double)plane[(size_t)y1 * sw + x0] * w01 +
                    (double)plane[(size_t)y1 * sw + x1] * w11;
                dst[(size_t)c * dst_plane + dst_index] =
                    (uint8_t)(value < 0.0 ? 0.0 : value > 255.0 ? 255.0 : value + 0.5);
            }
        }
    }
    return 1;
}

/* Cover-fit into a canvas-filling buffer (the original edit path). */
static int resample_planar_cover(const uint8_t *src, uint32_t sw, uint32_t sh,
                                 uint32_t dw, uint32_t dh, uint32_t channels,
                                 uint8_t *dst)
{
    return resample_planar_into(src, sw, sh, dst, dw, dh, 0, 0, dw, dh, channels);
}

/* WanGP downsamples the painted mask to (width/16, height/16) with LANCZOS
 * and thresholds at >64 (`latent_mask = ... .gt(64)`). A 16x16 area mean is
 * the equivalent contraction for a hard brush mask, with no resampling
 * library required. WHITE (255) means "regenerate", BLACK means "keep". */
static int downsample_mask(const uint8_t *canvas_mask, uint32_t w, uint32_t h,
                           int latent_w, int latent_h, uint8_t *out)
{
    if (!canvas_mask || !out || latent_w <= 0 || latent_h <= 0 ||
        w != (uint32_t)latent_w * 16u || h != (uint32_t)latent_h * 16u)
        return 0;
    for (int ly = 0; ly < latent_h; ly++) {
        for (int lx = 0; lx < latent_w; lx++) {
            uint64_t sum = 0;
            for (uint32_t dy = 0; dy < 16; dy++) {
                const uint8_t *row = canvas_mask +
                    (size_t)(ly * 16 + (int)dy) * w + (size_t)lx * 16;
                for (uint32_t dx = 0; dx < 16; dx++) sum += row[dx];
            }
            out[(size_t)ly * latent_w + lx] =
                (uint8_t)((sum / 256) > 64 ? 1 : 0);
        }
    }
    return 1;
}

/* Planar RGB8 [3][H][W] + alpha plane [H][W] -> interleaved RGBA8. */
static void pack_planar_rgba8(const uint8_t *planar, const uint8_t *alpha,
                              uint32_t width, uint32_t height, uint8_t *packed)
{
    const size_t pixels = (size_t)width * height;
    for (size_t i = 0; i < pixels; i++) {
        packed[i * 4 + 0] = planar[i];
        packed[i * 4 + 1] = planar[pixels + i];
        packed[i * 4 + 2] = planar[2 * pixels + i];
        packed[i * 4 + 3] = alpha[i];
    }
}

/* Planar F32 [-1,1] [3][H][W] -> interleaved RGB8, matching
 * qzen_png_write_pixels_f32's clamp/round exactly. */
static void pack_pixels_f32(const float *pixels, uint32_t width,
                            uint32_t height, uint8_t *rgb)
{
    size_t plane = (size_t)width * height;
    const size_t offset[3] = {0, plane, 2 * plane};
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t *out = rgb + ((size_t)y * width + x) * 3;
            for (uint32_t c = 0; c < 3; c++) {
                float value = pixels[offset[c] + (size_t)y * width + x];
                value = value * 0.5f + 0.5f;
                if (value < 0.0f) value = 0.0f;
                if (value > 1.0f) value = 1.0f;
                out[c] = (uint8_t)(value * 255.0f + 0.5f);
            }
        }
    }
}

static qzi_status emit_png_ex(const uint8_t *pixels, uint32_t width,
                              uint32_t height, uint32_t channels,
                              qzi_image *out, char *error, size_t error_cap)
{
    size_t size = 0;
    uint8_t *png = channels == 4u
        ? qzi_png_encode_rgba8(pixels, width, height, &size, error, error_cap)
        : qzi_png_encode_rgb8(pixels, width, height, &size, error, error_cap);
    if (!png) return QZI_ERR_ENGINE;
    out->data = png;
    out->size = size;
    out->width = width;
    out->height = height;
    out->token = (uint64_t)(uintptr_t)png;
    return QZI_OK;
}

static qzi_status emit_png(const uint8_t *rgb, uint32_t width, uint32_t height,
                           qzi_image *out, char *error, size_t error_cap)
{
    return emit_png_ex(rgb, width, height, 3u, out, error, error_cap);
}

/* --------------------------------------------------------------- progress */

typedef struct progress_bridge {
    const qzi_callbacks *callbacks;
    double start;
    int cancelled;
} progress_bridge;

static void bridge_emit(progress_bridge *bridge, const char *phase,
                        int32_t completed, int32_t total)
{
    qzi_emit_progress(bridge->callbacks, phase, completed, total, bridge->start);
    if (qzi_cancelled(bridge->callbacks)) bridge->cancelled = 1;
}

static void qwen21_progress(const char *phase, int completed, int total,
                            void *opaque)
{
    progress_bridge *bridge = (progress_bridge *)opaque;
    if (!bridge) return;
    bridge_emit(bridge, phase, (int32_t)completed, (int32_t)total);
}

/* qzen_zpipe_denoise's progress callback has no opaque pointer. The qzi ABI
 * serialises generation (single-flight), so exactly one denoise can be in
 * flight per process and a single static bridge is sufficient. */
static progress_bridge *g_zimage_bridge;

static void zimage_progress(uint32_t step, uint32_t steps)
{
    progress_bridge *bridge = g_zimage_bridge;
    if (!bridge) return;
    bridge_emit(bridge, "dit", (int32_t)step, (int32_t)steps);
}

/* ------------------------------------------------------- model residency */

static void release_zimage(qzen_state *s)
{
    if (s->zpipe) {
        if (s->z_text_resident) qzen_zpipe_release_text(s->zpipe);
        qzen_zpipe_free(s->zpipe);
        s->zpipe = NULL;
    }
    s->z_text_resident = 0;
    s->z_pack = 0;
    s->z_pack_id[0] = '\0';
    if (s->z_control) {
        qzen_zimage_control_free(s->z_control);
        s->z_control = NULL;
    }
    s->z_control_pipe = NULL;
}

static void release_qwen21(qzen_state *s)
{
    if (s->text) { qzen_qwen21_text_close(s->text); s->text = NULL; }
    if (s->tokenizer) { qzen_tokenizer_free(s->tokenizer); s->tokenizer = NULL; }
    if (s->dit) { qzen_qwen21_close(s->dit); s->dit = NULL; }
    if (s->vae) { qzen_qwen21_vae_close(s->vae); s->vae = NULL; }
    free(s->uncond);
    s->uncond = NULL;
    s->uncond_tokens = 0;
    s->tier_resident = -1;
    s->lora_resident[0] = '\0';
    s->lora_scale_resident = 0.0f;
}

static qzi_status resolve_qwen21_pack(qzen_state *s, char *error, size_t cap)
{
    if (s->q_paths_resolved) return QZI_OK;
    if (!*s->qwen21_dir) {
        qzi_set_error(error, cap,
            "qwen21_dir is not configured (see QwenZImage.config.json)");
        return QZI_ERR_NOT_CONFIGURED;
    }
    char dit_dir[QZEN_PATH_CAP], vae_dir[QZEN_PATH_CAP], text_dir[QZEN_PATH_CAP];
    if (!component_dir(s->qwen21_dir, "dit", dit_dir, sizeof(dit_dir)) ||
        !component_dir(s->qwen21_dir, "vae", vae_dir, sizeof(vae_dir)) ||
        !component_dir(s->qwen21_dir, "text_encoder", text_dir,
                       sizeof(text_dir))) {
        qzi_set_error(error, cap, "the qwen21 pack path is too long");
        return QZI_ERR_ENGINE;
    }
    static const char *const turbo_names[] = {
        "qwen_image_21_turbo_7B_int8_convrot.safetensors",
        "dit_turbo.safetensors", "turbo.safetensors",
        "dit_turbo_int8.safetensors"};
    static const char *const base_names[] = {
        "qwen_image_21_7B_int8_convrot.safetensors", "dit_base.safetensors",
        "base.safetensors", "dit_base_int8.safetensors"};
    static const char *const vae_names[] = {"qwen_image_21_vae.safetensors",
                                            "vae.safetensors"};
    static const char *const text_names[] = {
        "Qwen3-VL-8B-Instruct_int8_convrot.safetensors",
        "text_encoder.safetensors", "text_encoder_int8.safetensors"};
    static const char *const tokenizer_names[] = {"tokenizer.json",
                                                  "tokenizer_legacy.json"};

    first_file(dit_dir, turbo_names, 4, s->q_turbo, sizeof(s->q_turbo));
    first_file(dit_dir, base_names, 4, s->q_base, sizeof(s->q_base));
    if (!*s->q_turbo && !*s->q_base) {
        qzi_set_error_fmt(error, cap,
            "qwen21 pack has no DiT checkpoint in %s", dit_dir);
        return QZI_ERR_NOT_CONFIGURED;
    }
    if (!first_file(vae_dir, vae_names, 2, s->q_vae, sizeof(s->q_vae))) {
        qzi_set_error_fmt(error, cap, "qwen21 pack is missing %s/%s",
                          vae_dir, vae_names[0]);
        return QZI_ERR_NOT_CONFIGURED;
    }
    if (!first_file(text_dir, text_names, 3, s->q_text, sizeof(s->q_text))) {
        qzi_set_error_fmt(error, cap, "qwen21 pack is missing %s/%s",
                          text_dir, text_names[0]);
        return QZI_ERR_NOT_CONFIGURED;
    }
    if (!first_file(text_dir, tokenizer_names, 2, s->q_tokenizer,
                    sizeof(s->q_tokenizer))) {
        qzi_set_error_fmt(error, cap, "qwen21 pack is missing %s/%s",
                          text_dir, tokenizer_names[0]);
        return QZI_ERR_NOT_CONFIGURED;
    }
    s->q_paths_resolved = 1;
    return QZI_OK;
}

/* The ABI already guarantees `lora` is a bare file name. Keep the check here
 * too: the backend is the last line of defence before joining a user-supplied
 * name onto a directory. Mirrors qzi.c's no_path_separators() exactly. */
static int adapter_name_is_bare(const char *name)
{
    const unsigned char *p;
    if (!name || !*name) return 0;
    if (strlen(name) > QZI_LORA_NAME_MAX) return 0;
    if (strchr(name, '/') || strchr(name, '\\') || strchr(name, ':')) return 0;
    if (strstr(name, "..")) return 0;
    for (p = (const unsigned char *)name; *p; p++)
        if (*p < 0x20u || *p == 0x7fu) return 0;
    return 1;
}

/* Drops every trace of a resident DiT. `qzen_qwen21_apply_lora()` merges
 * adapters into the INT8 weights *in place*, so a failed merge can leave a
 * half-applied model behind. Keeping such a DiT would make the next request
 * (especially one with no LoRA) silently render with corrupted weights. */
static void forget_qwen21_dit(qzen_state *s)
{
    if (s->dit) { qzen_qwen21_close(s->dit); s->dit = NULL; }
    s->tier_resident = -1;
    s->lora_resident[0] = '\0';
    s->lora_scale_resident = 0.0f;
    /* Deliberately keep s->resident: the text tower / VAE may still be
     * resident, and qzen_generate_zimage() uses this flag to release them on
     * a model switch. s->dit == NULL is what forces the next reload. */
}

static qzi_status ensure_qwen21_dit(qzen_state *s, int turbo,
                                    const char *lora, float lora_scale,
                                    progress_bridge *bridge,
                                    char *error, size_t cap)
{
    const char *tier_path = turbo ? s->q_turbo : s->q_base;
    const char *wanted = lora && *lora ? lora : "";
    /* qzi_generate_qwen21() already normalises the 0 sentinel to 1.0, but
     * keep the backend correct on its own: 0 and 1.0 must not look like two
     * different adapters and trigger a multi-GB reload. */
    const float scale = lora_scale > 0.0f ? lora_scale : 1.0f;
    char lora_path[QZEN_PATH_CAP] = "";
    int resident, same_adapter, need_reload;

    if (!*tier_path) {
        qzi_set_error_fmt(error, cap,
            "the qwen21 pack has no %s checkpoint", turbo ? "turbo" : "base");
        return QZI_ERR_UNSUPPORTED;
    }

    /* Validate and resolve the adapter *before* loading multi-GB weights, so
     * a bad name or a missing directory fails immediately. */
    if (*wanted) {
        if (!adapter_name_is_bare(wanted)) {
            qzi_set_error_fmt(error, cap, "invalid adapter name \"%s\"",
                              wanted);
            return QZI_ERR_INVALID_ARGUMENT;
        }
        if (!*s->qwen21_lora_dir) {
            qzi_set_error(error, cap,
                "qwen21_lora_dir is not configured (see QwenZImage.config.json)");
            return QZI_ERR_NOT_CONFIGURED;
        }
        if (!join_path(lora_path, sizeof(lora_path), s->qwen21_lora_dir,
                       wanted)) {
            qzi_set_error(error, cap, "the LoRA path is too long");
            return QZI_ERR_ENGINE;
        }
    }

    resident = s->lora_resident[0] != '\0';
    same_adapter = (*wanted == '\0' && !resident) ||
                   (*wanted && resident &&
                    !strcmp(wanted, s->lora_resident) &&
                    scale == s->lora_scale_resident);
    need_reload = !s->dit || s->tier_resident != turbo ||
                  (resident && !same_adapter) ||
                  (!resident && *wanted);

    if (need_reload) {
        /* Clear the cache first: if qzen_qwen21_open() fails, no stale
         * "adapter is resident" state may survive. */
        forget_qwen21_dit(s);
        bridge_emit(bridge, "model load", 0, 1);
        char load_error[QZEN_ERR_CAP] = "";
        s->dit = qzen_qwen21_open(tier_path, 0, load_error,
                                  sizeof(load_error));
        if (!s->dit) {
            qzi_set_error_fmt(error, cap, "qwen21 DiT: %s", load_error);
            return QZI_ERR_ENGINE;
        }
        s->tier_resident = turbo;
        free(s->uncond);
        s->uncond = NULL;
        s->uncond_tokens = 0;
    }
    if (*wanted && (need_reload || !same_adapter)) {
        char load_error[QZEN_ERR_CAP] = "";
        if (!qzen_qwen21_apply_lora(s->dit, lora_path, scale,
                                    load_error, sizeof(load_error))) {
            forget_qwen21_dit(s);
            qzi_set_error_fmt(error, cap, "qwen21 LoRA: %s", load_error);
            return QZI_ERR_ENGINE;
        }
        snprintf(s->lora_resident, sizeof(s->lora_resident), "%s", wanted);
        s->lora_scale_resident = scale;
    }
    if (!s->vae) {
        char load_error[QZEN_ERR_CAP] = "";
        s->vae = qzen_qwen21_vae_open(s->q_vae, 0, load_error,
                                      sizeof(load_error));
        if (!s->vae) {
            qzi_set_error_fmt(error, cap, "qwen21 VAE: %s", load_error);
            return QZI_ERR_ENGINE;
        }
    }
    if (!s->tokenizer) {
        char load_error[QZEN_ERR_CAP] = "";
        s->tokenizer = qzen_tokenizer_load(s->q_tokenizer, load_error,
                                           sizeof(load_error));
        if (!s->tokenizer) {
            qzi_set_error_fmt(error, cap, "qwen21 tokenizer: %s", load_error);
            return QZI_ERR_ENGINE;
        }
    }
    s->resident = QZI_MODEL_QWEN21;
    return QZI_OK;
}

static int encode_text(qzen_state *s, const char *prompt,
                       float **rows, int *count, char *error, size_t cap)
{
    uint32_t *ids = NULL;
    size_t tokens = 0;
    if (!qzen_qwen21_text_encode(s->tokenizer, prompt, &ids, &tokens,
                                 error, cap))
        return 0;
    if (!tokens || tokens > 4096) {
        qzen_tokenizer_ids_free(ids);
        qzi_set_error_fmt(error, cap,
            "the prompt tokenises to %zu rows; qwen21 accepts 1..4096", tokens);
        return 0;
    }
    float *hidden = (float *)malloc(tokens * QZEN_QWEN21_TEXT_DIM *
                                    sizeof(float));
    if (!hidden) {
        qzen_tokenizer_ids_free(ids);
        qzi_set_error(error, cap, "out of memory for text rows");
        return 0;
    }
    if (!s->text) {
        s->text = qzen_qwen21_text_open(s->q_text, 0, error, cap);
        if (!s->text) {
            qzen_tokenizer_ids_free(ids);
            free(hidden);
            return 0;
        }
    }
    if (!qzen_qwen21_text_forward(s->text, ids, tokens, hidden, NULL,
                                  error, cap)) {
        qzen_tokenizer_ids_free(ids);
        free(hidden);
        return 0;
    }
    qzen_tokenizer_ids_free(ids);
    *rows = hidden;
    *count = (int)tokens;
    return 1;
}

/* ------------------------------------------------- zimage pack resolution */

static qzi_status resolve_zimage_pack(qzen_state *s, const char *checkpoint,
                                      char *error, size_t cap)
{
    if (!*s->zimage_dir) {
        qzi_set_error(error, cap,
            "zimage_dir is not configured (see QwenZImage.config.json)");
        return QZI_ERR_NOT_CONFIGURED;
    }
    if (!component_dir(s->zimage_dir, "dit", s->z_dit, sizeof(s->z_dit)) ||
        !component_dir(s->zimage_dir, "vae", s->z_vae, sizeof(s->z_vae)) ||
        !component_dir(s->zimage_dir, "text_encoder", s->z_text,
                       sizeof(s->z_text))) {
        qzi_set_error(error, cap, "the zimage pack path is too long");
        return QZI_ERR_ENGINE;
    }
    char tokenizer_path[QZEN_PATH_CAP];
    if (!join_path(tokenizer_path, sizeof(tokenizer_path), s->z_text,
                   "tokenizer.json") || !is_regular(tokenizer_path)) {
        qzi_set_error_fmt(error, cap, "zimage pack is missing %s/tokenizer.json",
                          s->z_text);
        return QZI_ERR_NOT_CONFIGURED;
    }

    int bf16 = !checkpoint || !*checkpoint || !strcmp(checkpoint, "bf16");
    s->z_pack_file[0] = '\0';
    snprintf(s->z_pack_id, sizeof(s->z_pack_id), "%s",
             bf16 ? "bf16" : "nunchaku");

    if (bf16) return QZI_OK;

    if (!*s->zimage_nunchaku_dir) {
        qzi_set_error(error, cap, "zimage_nunchaku_dir is not configured "
                                  "(see QwenZImage.config.json)");
        return QZI_ERR_UNSUPPORTED;
    }

    /* Enumerate svdq-*.safetensors under zimage_nunchaku_dir. qzi.c rejects
     * every checkpoint containing "fp4" (int4 packs only), so this resolver
     * handles just "nunchaku" (first int4 pack) and "nunchaku-r<rank>";
     * a single .safetensors path is also accepted. */
    char found[QZEN_PATH_CAP] = "";
    char best_name[256] = "";
    char rank[32] = "";
    if (strncmp(checkpoint, "nunchaku-r", 10) == 0) {
        snprintf(rank, sizeof(rank), "%s", checkpoint + 10);
    }
    char pattern[64];
    if (*rank)
        snprintf(pattern, sizeof(pattern), "_r%s-", rank);
    else
        snprintf(pattern, sizeof(pattern), "%s", "svdq-int4");

    if (is_regular(s->zimage_nunchaku_dir) &&
        strstr(s->zimage_nunchaku_dir, ".safetensors")) {
        snprintf(found, sizeof(found), "%s", s->zimage_nunchaku_dir);
    } else if (is_directory(s->zimage_nunchaku_dir)) {
        DIR *stream = opendir(s->zimage_nunchaku_dir);
        if (!stream) {
            qzi_set_error_fmt(error, cap, "cannot list %s",
                              s->zimage_nunchaku_dir);
            return QZI_ERR_ENGINE;
        }
        struct dirent *entry;
        while ((entry = readdir(stream)) != NULL) {
            const char *name = entry->d_name;
            size_t length = strlen(name);
            if (strncmp(name, "svdq-", 5) != 0) continue;
            if (length < 14 || strcmp(name + length - 12, ".safetensors") != 0)
                continue;
            if (*rank && !strstr(name, pattern)) continue;
            if (!*rank && strncmp(name, "svdq-int4", 9) != 0) continue;
            /* Deterministic choice: lexicographically first matching pack. */
            if (!*best_name || strcmp(name, best_name) < 0)
                snprintf(best_name, sizeof(best_name), "%s", name);
        }
        closedir(stream);
        if (*best_name)
            snprintf(found, sizeof(found), "%s/%s",
                     s->zimage_nunchaku_dir, best_name);
    }
    if (!*found) {
        qzi_set_error_fmt(error, cap,
            "checkpoint \"%s\" is not available under %s",
            checkpoint, s->zimage_nunchaku_dir);
        return QZI_ERR_UNSUPPORTED;
    }
    copy_path(s->z_pack_file, sizeof(s->z_pack_file), found);
    snprintf(s->z_pack_id, sizeof(s->z_pack_id), "%s", checkpoint);
    return QZI_OK;
}

/* ------------------------------------------------------------ vtable: init */

static qzi_status qzen_init(void **out_context,
                            const qzi_engine_config *config,
                            char *error, size_t error_cap)
{
    if (!out_context || !config) return QZI_ERR_INVALID_ARGUMENT;
    *out_context = NULL;

    qzen_config file;
    char config_error[QZEN_ERR_CAP] = "";
    int rc = qzen_config_load(NULL, &file, config_error, sizeof(config_error));
    if (rc < 0) {
        qzi_set_error_fmt(error, error_cap, "config: %s", config_error);
        return QZI_ERR_NOT_CONFIGURED;
    }
    /* Environment overrides have the same precedence as on the managed side,
     * and are the only channel for qwen21_lora_dir / zimage_nunchaku_dir set
     * in code (the C ABI has no fields for them). */
    if (qzen_config_apply_environment(&file, config_error,
                                      sizeof(config_error)) < 0) {
        qzi_set_error_fmt(error, error_cap, "config: %s", config_error);
        return QZI_ERR_NOT_CONFIGURED;
    }

    qzen_state *s = (qzen_state *)calloc(1, sizeof(*s));
    if (!s) {
        qzi_set_error(error, error_cap, "out of memory creating backend state");
        return QZI_ERR_OUT_OF_MEMORY;
    }
    const char *qwen = config->qwen21_dir && *config->qwen21_dir
                           ? config->qwen21_dir
                           : qzen_config_field(&file, "qwen21_dir");
    const char *zimg = config->zimage_dir && *config->zimage_dir
                           ? config->zimage_dir
                           : qzen_config_field(&file, "zimage_dir");
    const char *cache = config->cache_dir && *config->cache_dir
                            ? config->cache_dir
                            : qzen_config_field(&file, "text_cache_dir");
    copy_path(s->qwen21_dir, sizeof(s->qwen21_dir), qwen);
    copy_path(s->zimage_dir, sizeof(s->zimage_dir), zimg);
    copy_path(s->text_cache_dir, sizeof(s->text_cache_dir), cache);
    copy_path(s->qwen21_lora_dir, sizeof(s->qwen21_lora_dir),
              qzen_config_field(&file, "qwen21_lora_dir"));
    copy_path(s->zimage_nunchaku_dir, sizeof(s->zimage_nunchaku_dir),
              qzen_config_field(&file, "zimage_nunchaku_dir"));
    /* The C ABI v1 has no ControlNet path field: it arrives through the
     * config file or QZI_ZIMAGE_CONTROL_DIR, both of which the managed side
     * publishes in a create-scoped environment block. */
    copy_path(s->zimage_control_dir, sizeof(s->zimage_control_dir),
              qzen_config_field(&file, "zimage_control_dir"));
    s->device_id = config->device_id;
    s->tier_resident = -1;
    s->started = qzi_now_seconds();

    /* Respect an explicit device choice through the CUDA runtime's own
     * visibility variable, and enable the GPU kernels before any load. */
    if (s->device_id >= 0) {
        char device[32];
        snprintf(device, sizeof(device), "%d", s->device_id);
        if (!getenv("CUDA_VISIBLE_DEVICES")) enable_env("CUDA_VISIBLE_DEVICES",
                                                        device);
    }
    enable_gpu_switches();

    *out_context = s;
    return QZI_OK;
}

static void qzen_shutdown(void *context)
{
    qzen_state *s = (qzen_state *)context;
    if (!s) return;
    release_zimage(s);
    release_qwen21(s);
    free(s);
}

/* ---------------------------------------------------------- vtable: qwen21 */

/* Shared Qwen-Image 2.1 path. `edit` is NULL for plain text-to-image. With an
 * edit request the reference (and its optional mask) is centre-crop cover-fitted
 * to the output canvas, the reference is VAE-encoded into the normalised latent
 * that starts the sampler, and the mask is contracted onto the 1/16 latent grid
 * for WanGP's masked-denoising loop. One implementation keeps text encoding,
 * model residency, the sigma schedule, progress phases and PNG emission
 * identical between t2i and img2img. */
/* WanGP models/qwen21/pipeline.py: RED_OUTPAINTING_PROMPT, appended to the
 * user prompt whenever the Red Canvas border method is active. */
#define QZEN_RED_OUTPAINTING_PROMPT \
    "Remove the red paddings on the sides and show what's behind them."

/* WanGP model_modes -> LanPaint NSteps ({2:2, 3:5, 4:10, 5:15}). */
static int qzen_lanpaint_iterations(uint32_t model_mode)
{
    switch (model_mode) {
    case QZI_EDIT_MODE_LANPAINT_2:  return 2;
    case QZI_EDIT_MODE_LANPAINT_5:  return 5;
    case QZI_EDIT_MODE_LANPAINT_10: return 10;
    case QZI_EDIT_MODE_LANPAINT_15: return 15;
    default:                        return 0;
    }
}

static qzi_status qzen_generate_qwen21_impl(void *context,
                                            const qzi_qwen21_request *base,
                                            const qzi_qwen21_edit_request *edit,
                                            const qzi_qwen21_edit_ex_request *adv,
                                            const qzi_callbacks *callbacks,
                                            qzi_image *out_image,
                                            char *error, size_t error_cap)
{
    qzen_state *s = (qzen_state *)context;
    progress_bridge bridge = {callbacks, qzi_now_seconds(), 0};
    char local[QZEN_ERR_CAP] = "";
    const qzi_qwen21_edit_request *edit_ref = adv ? &adv->base : edit;
    const qzi_qwen21_request *request = edit_ref ? &edit_ref->base : base;
    const int turbo = !(request->tier && *request->tier &&
                        strcmp(request->tier, "base") == 0);
    /* qzi.c resolves the zero sentinels before calling a backend; repeat it
     * here so an integrator calling the vtable directly observes the same
     * effective values (and so does the reference backend). */
    const int steps = request->steps > 0 ? request->steps : (turbo ? 8 : 40);
    const float guidance = request->guidance > 0.0f
                               ? request->guidance : (turbo ? 1.0f : 4.0f);
    const int outpainting = adv && (adv->margin_top || adv->margin_bottom ||
                                    adv->margin_left || adv->margin_right);
    const int lanpaint_steps = adv ? qzen_lanpaint_iterations(adv->model_mode) : 0;
    /* WanGP forces denoising_strength = masking_strength = 1.0 for LanPaint. */
    const float strength = !edit_ref ? 1.0f
                         : lanpaint_steps ? 1.0f
                         : (edit_ref->strength > 0.0f ? edit_ref->strength : 0.75f);
    const float masking_strength = (!edit_ref || !edit_ref->mask) ? 0.0f
        : lanpaint_steps ? 1.0f
        : (edit_ref->masking_strength > 0.0f ? edit_ref->masking_strength : 1.0f);
    float *text = NULL, *negative = NULL;
    int text_tokens = 0, negative_tokens = 0, negative_owned = 0;
    float *latent = NULL, *init_latent = NULL;
    uint8_t *rgb = NULL, *packed = NULL, *alpha = NULL;
    uint8_t *reference = NULL, *canvas_mask = NULL, *latent_mask = NULL;
    uint8_t *red_canvas = NULL;
    char *prompt_owned = NULL;
    qzi_status status = QZI_ERR_ENGINE;

    /* qzi.c validates on the way in, but the backend is also a public
     * native/native contract: never read caller-owned pixels before checking
     * the dimensions that describe them. */
    status = adv ? qzi_validate_qwen21_edit_ex(adv, error, error_cap)
          : edit_ref ? qzi_validate_qwen21_edit(edit_ref, error, error_cap)
                     : qzi_validate_qwen21(request, error, error_cap);
    if (status != QZI_OK) return status;

    status = resolve_qwen21_pack(s, error, error_cap);
    if (status != QZI_OK) return status;
    if (s->resident == QZI_MODEL_ZIMAGE) release_zimage(s);
    status = ensure_qwen21_dit(s, turbo, request->lora,
                               request->lora_scale, &bridge,
                               error, error_cap);
    if (status != QZI_OK) return status;

    bridge_emit(&bridge, "text", 0, 1);
    /* WanGP: input_prompt.rstrip().rstrip(".") + ". " +
     * RED_OUTPAINTING_PROMPT, only when the instruction is not already there. */
    const char *prompt = request->prompt;
    if (outpainting && !strstr(prompt, QZEN_RED_OUTPAINTING_PROMPT)) {
        size_t length = strlen(prompt);
        while (length > 0 && (unsigned char)prompt[length - 1] <= ' ') length--;
        while (length > 0 && prompt[length - 1] == '.') length--;
        const char *suffix = ". " QZEN_RED_OUTPAINTING_PROMPT;
        const size_t suffix_len = strlen(suffix);
        prompt_owned = (char *)malloc(length + suffix_len + 1u);
        if (!prompt_owned) {
            qzi_set_error(error, error_cap,
                          "out of memory for the outpainting prompt");
            status = QZI_ERR_OUT_OF_MEMORY;
            goto done;
        }
        memcpy(prompt_owned, prompt, length);
        memcpy(prompt_owned + length, suffix, suffix_len + 1u);
        prompt = prompt_owned;
    }
    if (!encode_text(s, prompt, &text, &text_tokens,
                     local, sizeof(local))) {
        qzi_set_error_fmt(error, error_cap, "text encode: %s", local);
        status = QZI_ERR_ENGINE;
        goto done;
    }
    if (!turbo) {
        if (request->negative_prompt && *request->negative_prompt) {
            if (!encode_text(s, request->negative_prompt, &negative,
                             &negative_tokens, local, sizeof(local))) {
                qzi_set_error_fmt(error, error_cap,
                                  "negative prompt: %s", local);
                status = QZI_ERR_ENGINE;
                goto done;
            }
            negative_owned = 1;
        } else if (s->uncond && s->uncond_tokens > 0) {
            negative = s->uncond;
            negative_tokens = s->uncond_tokens;
        } else {
            if (!encode_text(s, " ", &negative, &negative_tokens,
                             local, sizeof(local))) {
                qzi_set_error_fmt(error, error_cap,
                                  "negative prompt: %s", local);
                status = QZI_ERR_ENGINE;
                goto done;
            }
            s->uncond = negative;
            s->uncond_tokens = negative_tokens;
            negative_owned = 0;
        }
    }
    if (bridge.cancelled || qzi_cancelled(callbacks)) {
        qzi_set_error(error, error_cap, "cancelled during text encoding");
        status = QZI_ERR_CANCELLED;
        goto done;
    }

    const uint32_t canvas_w = (uint32_t)request->width;
    const uint32_t canvas_h = (uint32_t)request->height;
    const int latent_w = request->width / 16;
    const int latent_h = request->height / 16;
    const size_t plane = (size_t)latent_h * latent_w;
    const size_t canvas_pixels = (size_t)canvas_w * canvas_h;
    latent = (float *)malloc(QZEN_QWEN21_VAE_CHANNELS * plane *
                             sizeof(float));
    rgb = (uint8_t *)malloc(canvas_pixels * 3u);
    if (adv && adv->rgba) alpha = (uint8_t *)malloc(canvas_pixels);
    if (!latent || !rgb || (adv && adv->rgba && !alpha)) {
        qzi_set_error(error, error_cap, "out of memory for the latent/RGB buffers");
        status = QZI_ERR_OUT_OF_MEMORY;
        goto done;
    }

    if (edit_ref) {
        /* The ABI hands over planar RGB8 at the reference's own size. WanGP
         * cover-fits it to the output canvas first; when the sizes already
         * match, pass the caller's buffer straight through so VAE encoding
         * sees the original pixels without a resampling blur. Red Canvas
         * outpainting instead cover-fits into the INNER source rectangle and
         * surrounds it with opaque red (FF0000). */
        const uint32_t margin_left = outpainting ? adv->margin_left : 0u;
        const uint32_t margin_top = outpainting ? adv->margin_top : 0u;
        const uint32_t inner_w = outpainting
            ? canvas_w - adv->margin_left - adv->margin_right : canvas_w;
        const uint32_t inner_h = outpainting
            ? canvas_h - adv->margin_top - adv->margin_bottom : canvas_h;
        const uint8_t *ref_planar = edit_ref->reference_rgb;
        const uint8_t *mask_canvas = edit_ref->mask;
        const int needs_fit = outpainting ||
                              edit_ref->reference_width != canvas_w ||
                              edit_ref->reference_height != canvas_h;

        bridge_emit(&bridge, "image encode", 0, 1);
        if (bridge.cancelled || qzi_cancelled(callbacks)) {
            qzi_set_error(error, error_cap,
                          "cancelled during reference encoding");
            status = QZI_ERR_CANCELLED;
            goto done;
        }
        init_latent = (float *)malloc(QZEN_QWEN21_VAE_CHANNELS * plane *
                                      sizeof(float));
        if (!init_latent) {
            qzi_set_error(error, error_cap,
                          "out of memory for the encoded reference latent");
            status = QZI_ERR_OUT_OF_MEMORY;
            goto done;
        }
        if (outpainting) {
            /* The mask marks the red margins as "regenerate" (255) and the
             * source rectangle as "keep" (0), except where the user painted a
             * white brush stroke. Margins are 32-aligned by the ABI, so they
             * land exactly on latent-cell boundaries after the /16 grid. */
            red_canvas = (uint8_t *)malloc(canvas_pixels * 3u);
            canvas_mask = (uint8_t *)malloc(canvas_pixels);
            if (!red_canvas || !canvas_mask) {
                qzi_set_error(error, error_cap,
                              "out of memory for the red outpainting canvas");
                status = QZI_ERR_OUT_OF_MEMORY;
                goto done;
            }
            memset(red_canvas, 0, canvas_pixels * 3u);
            memset(red_canvas, 255, canvas_pixels); /* planar R channel */
            memset(canvas_mask, 255, canvas_pixels);
            if (!resample_planar_into(edit_ref->reference_rgb,
                                      edit_ref->reference_width,
                                      edit_ref->reference_height,
                                      red_canvas, canvas_w, canvas_h,
                                      margin_left, margin_top,
                                      inner_w, inner_h, 3u)) {
                qzi_set_error(error, error_cap,
                    "cannot cover-fit the reference into the outpainting source rectangle");
                status = QZI_ERR_ENGINE;
                goto done;
            }
            ref_planar = red_canvas;
            for (uint32_t y = 0; y < inner_h; y++)
                memset(canvas_mask + (size_t)(margin_top + y) * canvas_w +
                       margin_left, 0, inner_w);
            if (edit_ref->mask &&
                !resample_planar_into(edit_ref->mask,
                                      edit_ref->reference_width,
                                      edit_ref->reference_height,
                                      canvas_mask, canvas_w, canvas_h,
                                      margin_left, margin_top,
                                      inner_w, inner_h, 1u)) {
                qzi_set_error(error, error_cap,
                    "cannot cover-fit the edit mask into the outpainting source rectangle");
                status = QZI_ERR_ENGINE;
                goto done;
            }
            mask_canvas = canvas_mask;
        } else if (needs_fit) {
            reference = (uint8_t *)malloc(canvas_pixels * 3u);
            if (!reference) {
                qzi_set_error(error, error_cap,
                              "out of memory for the resampled reference");
                status = QZI_ERR_OUT_OF_MEMORY;
                goto done;
            }
            if (!resample_planar_cover(edit_ref->reference_rgb,
                                       edit_ref->reference_width,
                                       edit_ref->reference_height,
                                       canvas_w, canvas_h, 3u, reference)) {
                qzi_set_error(error, error_cap,
                              "cannot cover-fit the reference image");
                status = QZI_ERR_ENGINE;
                goto done;
            }
            ref_planar = reference;
            if (edit_ref->mask) {
                /* The painted mask lives at reference resolution and must follow
                 * exactly the same crop as the reference, or the white "regenerate"
                 * region would be offset from what the user actually sees. */
                canvas_mask = (uint8_t *)malloc(canvas_pixels);
                if (!canvas_mask) {
                    qzi_set_error(error, error_cap,
                                  "out of memory for the resampled mask");
                    status = QZI_ERR_OUT_OF_MEMORY;
                    goto done;
                }
                if (!resample_planar_cover(edit_ref->mask,
                                           edit_ref->reference_width,
                                           edit_ref->reference_height,
                                           canvas_w, canvas_h, 1u,
                                           canvas_mask)) {
                    qzi_set_error(error, error_cap,
                                  "cannot cover-fit the edit mask");
                    status = QZI_ERR_ENGINE;
                    goto done;
                }
                mask_canvas = canvas_mask;
            }
        }

        /* The VAE's 4th input channel is opaque alpha, not the inpainting
         * mask: qzen_qwen21_vae_encode() fills it internally. The returned
         * latent is already normalised with the checkpoint's mean/std, which
         * is the space the Euler sampler integrates. */
        local[0] = '\0';
        if (!qzen_qwen21_vae_encode(s->vae, ref_planar, (int)canvas_h,
                                    (int)canvas_w, init_latent, local,
                                    sizeof(local))) {
            qzi_set_error_fmt(error, error_cap,
                              "reference VAE encode: %s", local);
            status = QZI_ERR_ENGINE;
            goto done;
        }
        if (mask_canvas) {
            latent_mask = (uint8_t *)malloc(plane);
            if (!latent_mask) {
                qzi_set_error(error, error_cap,
                              "out of memory for the latent mask");
                status = QZI_ERR_OUT_OF_MEMORY;
                goto done;
            }
            if (!downsample_mask(mask_canvas, canvas_w, canvas_h,
                                 latent_w, latent_h, latent_mask)) {
                qzi_set_error(error, error_cap,
                              "cannot downsample the edit mask to the latent grid");
                status = QZI_ERR_ENGINE;
                goto done;
            }
        }
        bridge_emit(&bridge, "image encode", 1, 1);
        if (bridge.cancelled || qzi_cancelled(callbacks)) {
            qzi_set_error(error, error_cap,
                          "cancelled during reference encoding");
            status = QZI_ERR_CANCELLED;
            goto done;
        }
    }

    fill_noise(request->seed, latent, QZEN_QWEN21_VAE_CHANNELS * plane);

    float sigmas[80];
    int sigma_count = turbo
        ? qzen_qwen21_sigmas_turbo(sigmas)
        : qzen_qwen21_sigmas_base(steps, (int)plane, sigmas);
    if (!sigma_count || sigma_count != steps + 1) {
        qzi_set_error_fmt(error, error_cap,
            "the qwen21 sigma schedule has %d entries for %d steps",
            sigma_count, steps);
        status = QZI_ERR_ENGINE;
        goto done;
    }

    /* LanPaint's Langevin noise must be reproducible for a given seed and must
     * not disturb fill_noise()'s stream, so it gets its own mixed seed. The
     * four trailing fields are additive: a zeroed tail is exactly the old t2i
     * / edit / ControlNet behaviour. */
    const uint64_t lanpaint_seed =
        ((uint64_t)request->seed * 0x9E3779B97F4A7C15ULL) ^
        0x243F6A8885A308D3ULL;
    qzen_qwen21_sample sample = {
        text, text_tokens, negative, negative_tokens, latent,
        latent_h, latent_w, sigmas, steps, guidance,
        init_latent, strength, latent_mask, masking_strength,
        outpainting, lanpaint_steps, lanpaint_seed, alpha
    };
    local[0] = '\0';
    if (!qzen_qwen21_pipeline_run_ex(s->dit, s->vae, &sample, rgb,
                                     qwen21_progress, &bridge,
                                     local, sizeof(local))) {
        qzi_set_error_fmt(error, error_cap, "sample: %s", local);
        status = QZI_ERR_ENGINE;
        goto done;
    }
    if (bridge.cancelled || qzi_cancelled(callbacks)) {
        qzi_set_error(error, error_cap, "cancelled during denoising");
        status = QZI_ERR_CANCELLED;
        goto done;
    }

    bridge_emit(&bridge, "encode", 0, 1);
    /* WanGP's rgba = Enabled keeps conv_out's 4th channel as model-generated
     * alpha and writes PNG colour type 6; everything else stays RGB / type 2. */
    const uint32_t channels = (adv && adv->rgba) ? 4u : 3u;
    packed = (uint8_t *)malloc(canvas_pixels * channels);
    if (!packed) {
        qzi_set_error(error, error_cap, "out of memory for the PNG");
        status = QZI_ERR_OUT_OF_MEMORY;
        goto done;
    }
    if (channels == 4u)
        pack_planar_rgba8(rgb, alpha, canvas_w, canvas_h, packed);
    else
        pack_planar_rgb8(rgb, canvas_w, canvas_h, packed);
    status = emit_png_ex(packed, canvas_w, canvas_h, channels,
                         out_image, error, error_cap);
    if (status == QZI_OK) s->served++;

done:
    free(latent_mask);
    free(canvas_mask);
    free(reference);
    free(red_canvas);
    free(alpha);
    free(prompt_owned);
    free(init_latent);
    free(packed);
    free(rgb);
    free(latent);
    if (negative_owned) free(negative);
    free(text);
    return status;
}

static qzi_status qzen_generate_qwen21(void *context,
                                       const qzi_qwen21_request *request,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap)
{
    return qzen_generate_qwen21_impl(context, request, NULL, NULL, callbacks,
                                     out_image, error, error_cap);
}

static qzi_status qzen_generate_qwen21_edit(void *context,
                                            const qzi_qwen21_edit_request *request,
                                            const qzi_callbacks *callbacks,
                                            qzi_image *out_image,
                                            char *error, size_t error_cap)
{
    return qzen_generate_qwen21_impl(context, NULL, request, NULL, callbacks,
                                     out_image, error, error_cap);
}

/* Third additive ABI v1 slot: advanced editing still goes through the very
 * same qzen_generate_qwen21_impl orchestration -- no second pipeline, no
 * second copy of the 7B weights. Only the request envelope differs. */
static qzi_status qzen_generate_qwen21_edit_ex(void *context,
                                               const qzi_qwen21_edit_ex_request *request,
                                               const qzi_callbacks *callbacks,
                                               qzi_image *out_image,
                                               char *error, size_t error_cap)
{
    return qzen_generate_qwen21_impl(context, NULL, NULL, request, callbacks,
                                     out_image, error, error_cap);
}

/* ---------------------------------------------------------- vtable: zimage */

/* The Union 2.1 checkpoint is a separate ~3.7 GiB safetensors file, loaded
 * lazily on the first ControlNet request. A missing path is a configuration
 * error (not an engine failure) so callers get an actionable message. */
static qzi_status ensure_zimage_control(qzen_state *s,
                                        progress_bridge *bridge,
                                        char *error, size_t error_cap)
{
    if (s->z_control) return QZI_OK;
    if (!*s->zimage_control_dir) {
        qzi_set_error(error, error_cap,
            "zimage_control_dir is not configured: set it in "
            "QwenZImage.config.json (or QZI_ZIMAGE_CONTROL_DIR) to the "
            "Z-Image-Turbo-Fun-Controlnet-Union2.1 safetensors file");
        return QZI_ERR_NOT_CONFIGURED;
    }
    bridge_emit(bridge, "model load", 0, 1);
    char local[QZEN_ERR_CAP] = "";
    qzen_zimage_config dit = qzen_zimage_default_config();
    s->z_control = qzen_zimage_control_load(s->zimage_control_dir, &dit,
                                            local, sizeof(local));
    if (!s->z_control) {
        qzi_set_error_fmt(error, error_cap, "ControlNet weights: %s", local);
        return QZI_ERR_ENGINE;
    }
    s->z_control_pipe = NULL;
    return QZI_OK;
}

/* Shared Z-Image path. `control` is NULL for plain text-to-image; with it we
 * cover-fit + VAE-encode the guide image into a 33-channel control context,
 * lazily load the Union 2.1 weights, and run the hint-conditioned denoise
 * loop. Text encoding, model residency, VAE decode and PNG emission are
 * identical, so t2i keeps its exact behaviour and byte-level determinism. */
static qzi_status qzen_generate_zimage_impl(void *context,
                                       const qzi_zimage_request *base,
                                       const qzi_zimage_control_request *control,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap)
{
    const qzi_zimage_request *request = control ? &control->base : base;
    qzen_state *s = (qzen_state *)context;
    progress_bridge bridge = {callbacks, qzi_now_seconds(), 0};
    char local[QZEN_ERR_CAP] = "";
    float *cap = NULL, *latents = NULL, *pixels = NULL;
    float *control_latent = NULL, *control_context = NULL;
    uint8_t *rgb = NULL, *guide = NULL;
    qzen_zvae *vae = NULL;
    uint32_t cap_len = 0;
    qzi_status status = QZI_ERR_ENGINE;

    /* Defensive: the ABI layer already validates and normalises, but a custom
     * backend registration can call this function directly. */
    if (control) {
        status = qzi_validate_zimage_control(control, error, error_cap);
        if (status != QZI_OK) return status;
    }

    char previous_id[64];
    snprintf(previous_id, sizeof(previous_id), "%s", s->z_pack_id);
    status = resolve_zimage_pack(s, request->checkpoint, error, error_cap);
    if (status != QZI_OK) return status;
    if (s->resident == QZI_MODEL_QWEN21) release_qwen21(s);

    char dit_path[QZEN_PATH_CAP];
    char pack_id[64];
    snprintf(dit_path, sizeof(dit_path), "%s",
             *s->z_pack_file ? s->z_pack_file : s->z_dit);
    snprintf(pack_id, sizeof(pack_id), "%s", s->z_pack_id);
    const int want_pack = *s->z_pack_file ? 1 : -1;
    if (!s->zpipe || s->z_pack != want_pack ||
        strcmp(previous_id, pack_id) != 0) {
        release_zimage(s);
        bridge_emit(&bridge, "model load", 0, 1);
        local[0] = '\0';
        s->zpipe = qzen_zpipe_load(dit_path, NULL, local, sizeof(local));
        if (!s->zpipe) {
            qzi_set_error_fmt(error, error_cap, "zimage DiT: %s", local);
            return QZI_ERR_ENGINE;
        }
        s->z_pack = want_pack;
        snprintf(s->z_pack_id, sizeof(s->z_pack_id), "%s", pack_id);
    }
    s->resident = QZI_MODEL_ZIMAGE;

    bridge_emit(&bridge, "text", 0, 1);
    if (!s->z_text_resident) {
        local[0] = '\0';
        if (!qzen_zpipe_attach_text(s->zpipe, s->z_text, local,
                                    sizeof(local))) {
            qzi_set_error_fmt(error, error_cap, "text encoder: %s", local);
            status = QZI_ERR_ENGINE;
            goto done;
        }
        s->z_text_resident = 1;
    }
    local[0] = '\0';
    if (!qzen_zpipe_encode_text(s->zpipe, request->prompt, &cap, &cap_len,
                                local, sizeof(local))) {
        qzi_set_error_fmt(error, error_cap, "text encode: %s", local);
        status = QZI_ERR_ENGINE;
        goto done;
    }
    if (bridge.cancelled || qzi_cancelled(callbacks)) {
        qzi_set_error(error, error_cap, "cancelled during text encoding");
        status = QZI_ERR_CANCELLED;
        goto done;
    }

    const uint32_t width = (uint32_t)request->width;
    const uint32_t height = (uint32_t)request->height;
    const uint32_t lh = height / 8, lw = width / 8;
    latents = (float *)malloc((size_t)16 * lh * lw * sizeof(float));
    if (!latents) {
        qzi_set_error(error, error_cap, "out of memory for latents");
        status = QZI_ERR_OUT_OF_MEMORY;
        goto done;
    }
    qzen_zpipe_init_latents(request->seed, 16, lh, lw, latents);

    if (control) {
        status = ensure_zimage_control(s, &bridge, error, error_cap);
        if (status != QZI_OK) goto done;

        bridge_emit(&bridge, "image encode", 0, 1);
        const uint8_t *guide_rgb = control->control_rgb;
        if (control->control_width != width ||
            control->control_height != height) {
            guide = (uint8_t *)malloc((size_t)3 * width * height);
            if (!guide) {
                qzi_set_error(error, error_cap,
                              "out of memory for the cover-fit control image");
                status = QZI_ERR_OUT_OF_MEMORY;
                goto done;
            }
            if (!resample_planar_cover(control->control_rgb,
                                       control->control_width,
                                       control->control_height,
                                       width, height, 3, guide)) {
                qzi_set_error(error, error_cap,
                              "cannot cover-fit the control image");
                status = QZI_ERR_ENGINE;
                goto done;
            }
            guide_rgb = guide;
        }

        local[0] = '\0';
        vae = qzen_zvae_load(s->z_vae, NULL, local, sizeof(local));
        if (!vae) {
            qzi_set_error_fmt(error, error_cap, "vae: %s", local);
            status = QZI_ERR_ENGINE;
            goto done;
        }
        control_latent = (float *)malloc((size_t)16 * lh * lw * sizeof(float));
        if (!control_latent) {
            qzi_set_error(error, error_cap,
                          "out of memory for the control latent");
            status = QZI_ERR_OUT_OF_MEMORY;
            goto done;
        }
        local[0] = '\0';
        if (!qzen_zvae_encode(vae, guide_rgb, width, height, control_latent,
                              local, sizeof(local))) {
            qzi_set_error_fmt(error, error_cap,
                              "control image VAE encode: %s", local);
            status = QZI_ERR_ENGINE;
            goto done;
        }
        local[0] = '\0';
        control_context = qzen_zpipe_control_context(s->z_control,
                                                     control_latent, lh, lw,
                                                     local, sizeof(local));
        if (!control_context) {
            qzi_set_error_fmt(error, error_cap, "control context: %s", local);
            status = QZI_ERR_ENGINE;
            goto done;
        }
        /* The Union weights are uploaded into the DiT's GPU context. A
         * checkpoint switch frees that DiT, so re-upload for the new one. */
        if (s->z_control_pipe != s->zpipe) {
            local[0] = '\0';
            if (!qzen_zpipe_control_prepare(s->zpipe, s->z_control,
                                            local, sizeof(local))) {
                qzi_set_error_fmt(error, error_cap,
                                  "ControlNet prepare: %s", local);
                status = QZI_ERR_ENGINE;
                goto done;
            }
            s->z_control_pipe = s->zpipe;
        }
        bridge_emit(&bridge, "image encode", 1, 1);
    }

    g_zimage_bridge = &bridge;
    local[0] = '\0';
    /* ABI normalises the 0 sentinel to 0.65; keep the backend correct when a
     * custom caller hands us an unnormalised request. */
    const float context_scale = control && control->control_scale > 0.0f
                                    ? control->control_scale : 0.65f;
    int denoised = control
        ? qzen_zpipe_denoise_control(s->zpipe, s->z_control, control_context,
                                     context_scale, cap, cap_len, latents,
                                     lh, lw, zimage_progress,
                                     local, sizeof(local))
        : qzen_zpipe_denoise(s->zpipe, cap, cap_len, latents,
                             lh, lw, zimage_progress,
                             local, sizeof(local));
    g_zimage_bridge = NULL;
    if (!denoised) {
        qzi_set_error_fmt(error, error_cap, "denoise: %s", local);
        status = QZI_ERR_ENGINE;
        goto done;
    }
    if (bridge.cancelled || qzi_cancelled(callbacks)) {
        qzi_set_error(error, error_cap, "cancelled during denoising");
        status = QZI_ERR_CANCELLED;
        goto done;
    }

    bridge_emit(&bridge, "vae", 0, 1);
    /* A ControlNet request already loaded the VAE for the guide image; reuse
     * it instead of re-reading ~330 MB before decoding. */
    if (!vae) {
        local[0] = '\0';
        vae = qzen_zvae_load(s->z_vae, NULL, local, sizeof(local));
        if (!vae) {
            qzi_set_error_fmt(error, error_cap, "vae: %s", local);
            status = QZI_ERR_ENGINE;
            goto done;
        }
    }
    pixels = (float *)malloc((size_t)3 * width * height * sizeof(float));
    if (!pixels) {
        qzi_set_error(error, error_cap, "out of memory for decoded pixels");
        status = QZI_ERR_OUT_OF_MEMORY;
        goto done;
    }
    local[0] = '\0';
    if (!qzen_zvae_decode(vae, latents, lh, lw, pixels, local,
                          sizeof(local))) {
        qzi_set_error_fmt(error, error_cap, "decode: %s", local);
        status = QZI_ERR_ENGINE;
        goto done;
    }
    qzen_zvae_free(vae);
    vae = NULL;

    bridge_emit(&bridge, "encode", 0, 1);
    rgb = (uint8_t *)malloc((size_t)3 * width * height);
    if (!rgb) {
        qzi_set_error(error, error_cap, "out of memory for the PNG");
        status = QZI_ERR_OUT_OF_MEMORY;
        goto done;
    }
    pack_pixels_f32(pixels, width, height, rgb);
    status = emit_png(rgb, width, height, out_image, error, error_cap);
    if (status == QZI_OK) { s->served++; s->resident = QZI_MODEL_ZIMAGE; }

done:
    if (vae) qzen_zvae_free(vae);
    free(rgb);
    free(guide);
    free(pixels);
    free(latents);
    free(control_latent);
    free(control_context);
    free(cap);
    return status;
}

static qzi_status qzen_generate_zimage(void *context,
                                       const qzi_zimage_request *request,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap)
{
    return qzen_generate_zimage_impl(context, request, NULL, callbacks,
                                     out_image, error, error_cap);
}

static qzi_status qzen_generate_zimage_control(
                                       void *context,
                                       const qzi_zimage_control_request *request,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap)
{
    return qzen_generate_zimage_impl(context, NULL, request, callbacks,
                                     out_image, error, error_cap);
}

/* ------------------------------------------------- vtable: health / unload */

static qzi_status qzen_health(void *context, qzi_health *out,
                              char *error, size_t error_cap)
{
    (void)error; (void)error_cap;
    qzen_state *s = (qzen_state *)context;
    if (!s || !out) return QZI_ERR_INVALID_ARGUMENT;
    out->device_id = s->device_id;
    out->reference_backend = 0;
    /* The ABI layer fills resident_model, served, busy and uptime. The
     * backend only needs to declare that this is NOT the noise reference. */
    return QZI_OK;
}

static qzi_status qzen_unload(void *context, char *error, size_t error_cap)
{
    (void)error; (void)error_cap;
    qzen_state *s = (qzen_state *)context;
    if (!s) return QZI_ERR_INVALID_ARGUMENT;
    /* Release re-creatable state; never the DiT, which costs minutes. */
    if (s->text) { qzen_qwen21_text_close(s->text); s->text = NULL; }
    free(s->uncond);
    s->uncond = NULL;
    s->uncond_tokens = 0;
    if (s->zpipe && s->z_text_resident) {
        qzen_zpipe_release_text(s->zpipe);
        s->z_text_resident = 0;
    }
    return QZI_OK;
}

static void qzen_release_image(void *context, qzi_image *image)
{
    (void)context;
    if (!image || !image->data) return;
    free((void *)image->data);
    image->data = NULL;
    image->size = 0;
    image->token = 0;
}

static const qzi_backend qzen_vtable = {
    QZI_ABI_VERSION,
    "QwenZImageNative",
    qzen_init,
    qzen_shutdown,
    qzen_generate_qwen21,
    qzen_generate_zimage,
    qzen_health,
    qzen_unload,
    qzen_release_image,
    /* Additive ABI v1 slots: MUST stay last and in this order so older vtable
     * initialisers keep every existing offset. */
    qzen_generate_qwen21_edit,
    qzen_generate_zimage_control,
    qzen_generate_qwen21_edit_ex
};

const qzi_backend *qzi_real_backend(void)
{
    return &qzen_vtable;
}
