/*
 * qzi.c -- reference implementation of the QwenZImage.NET C ABI.
 *
 * This file is the control plane, not the model: it owns engine lifetime,
 * the single-flight lock, argument validation, progress/cancellation plumbing,
 * error messages, counters and output-buffer ownership. The actual pixels come
 * from a qzi_backend vtable. The built-in backend is the deterministic
 * reference implementation in qzi_stub.c; a real CUDA engine registers its own
 * vtable with qzi_register_backend() before creating an engine.
 *
 * Build: see native/CMakeLists.txt. Exported symbols are exactly the qzi_*
 * functions declared in include/qzi.h.
 */

#include "qzi.h"
#include "qzi_backend.h"
#include "qzi_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
typedef CRITICAL_SECTION qzi_mutex;
#else
#  include <pthread.h>
#  include <time.h>
typedef pthread_mutex_t qzi_mutex;
#endif

#define QZI_MAX_PROMPT_BYTES 16384u
#define QZI_MAX_CANVAS_SIDE  2048
#define QZI_MAX_PIXELS       (2048LL * 2048LL)
#define QZI_MAX_REFERENCE_SIDE 4096u
#define QZI_EDIT_DEFAULT_STRENGTH 0.75f
#define QZI_CONTROL_DEFAULT_SCALE 0.65f
#define QZI_CONTROL_MAX_SCALE    2.0f
#define QZI_CONTROL_MIN_SCALE    0.05f

struct qzi_engine {
    qzi_backend     backend;
    void           *context;
    qzi_mutex       mutex;
    double          started;
    uint64_t        served;
    uint64_t        rejected;
    int             busy;
    qzi_model_kind  resident;
};

static qzi_backend g_backend;
static int         g_backend_selected = 0;

/* ---- small platform shims ---------------------------------------------- */

static void mutex_init(qzi_mutex *m)
{
#if defined(_WIN32)
    InitializeCriticalSection(m);
#else
    pthread_mutex_init(m, NULL);
#endif
}

static void mutex_destroy(qzi_mutex *m)
{
#if defined(_WIN32)
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}

static void mutex_unlock(qzi_mutex *m)
{
#if defined(_WIN32)
    LeaveCriticalSection(m);
#else
    pthread_mutex_unlock(m);
#endif
}

/* Non-blocking: returns 1 when the lock was acquired. */
static int mutex_try_lock(qzi_mutex *m)
{
#if defined(_WIN32)
    return TryEnterCriticalSection(m) ? 1 : 0;
#else
    return pthread_mutex_trylock(m) == 0 ? 1 : 0;
#endif
}

double qzi_now_seconds(void)
{
#if defined(_WIN32)
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

/* ---- shared helpers ---------------------------------------------------- */

void qzi_set_error(char *error, size_t error_cap, const char *text)
{
    size_t n;
    if (!error || error_cap == 0) return;
    error[0] = '\0';
    if (!text) return;
    n = strlen(text);
    if (n >= error_cap) n = error_cap - 1;
    memcpy(error, text, n);
    error[n] = '\0';
}

void qzi_set_error_fmt(char *error, size_t error_cap, const char *fmt, ...)
{
    va_list ap;
    if (!error || error_cap == 0) return;
    va_start(ap, fmt);
    (void)vsnprintf(error, error_cap, fmt, ap);
    va_end(ap);
    error[error_cap - 1] = '\0';
}

uint64_t qzi_fnv1a(const char *text)
{
    uint64_t h = 14695981039346656037ULL;
    const unsigned char *p = (const unsigned char *)(text ? text : "");
    while (*p) {
        h ^= *p++;
        h *= 1099511628211ULL;
    }
    return h;
}

int qzi_cancelled(const qzi_callbacks *callbacks)
{
    if (!callbacks || !callbacks->should_cancel) return 0;
    return callbacks->should_cancel(callbacks->user) ? 1 : 0;
}

void qzi_emit_progress(const qzi_callbacks *callbacks, const char *phase,
                       int32_t completed, int32_t total, double start)
{
    if (!callbacks || !callbacks->progress) return;
    callbacks->progress(phase, completed, total,
                        qzi_now_seconds() - start, callbacks->user);
}

static int valid_prompt(const char *prompt)
{
    size_t n = 0;
    if (!prompt || prompt[0] == '\0') return 0;
    while (n <= QZI_MAX_PROMPT_BYTES && prompt[n] != '\0') n++;
    return n > 0 && n <= QZI_MAX_PROMPT_BYTES;
}

/* Case-insensitive "fp4" test; the managed validator uses
 * StringComparison.OrdinalIgnoreCase, so native must match it exactly. */
static int contains_fp4(const char *text)
{
    if (!text) return 0;
    for (; *text; text++) {
        if ((*text == 'f' || *text == 'F') &&
            (text[1] == 'p' || text[1] == 'P') && text[2] == '4') return 1;
    }
    return 0;
}

static int no_path_separators(const char *name)
{
    size_t length;
    const unsigned char *p;
    if (!name || !*name) return 0;
    length = strlen(name);
    if (length > QZI_LORA_NAME_MAX) return 0;
    if (strchr(name, '/') || strchr(name, '\\') || strchr(name, ':')) return 0;
    if (strstr(name, "..")) return 0;
    for (p = (const unsigned char *)name; *p; p++)
        if (*p < 0x20u || *p == 0x7fu) return 0;
    return 1;
}

static qzi_status check_canvas(int32_t width, int32_t height, int32_t multiple,
                               char *error, size_t error_cap)
{
    if (width <= 0 || height <= 0 || width % multiple || height % multiple ||
        width > QZI_MAX_CANVAS_SIDE || height > QZI_MAX_CANVAS_SIDE) {
        qzi_set_error_fmt(error, error_cap,
                          "width and height must be multiples of %d in 1..%d",
                          multiple, QZI_MAX_CANVAS_SIDE);
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if ((int64_t)width * (int64_t)height > QZI_MAX_PIXELS) {
        qzi_set_error(error, error_cap, "canvas exceeds the 2048*2048 pixel limit");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    return QZI_OK;
}

qzi_status qzi_validate_qwen21(const qzi_qwen21_request *request,
                               char *error, size_t error_cap)
{
    int turbo = 1;
    qzi_status s;
    if (!request) {
        qzi_set_error(error, error_cap, "qwen21 request is NULL");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->flags != 0) {
        qzi_set_error(error, error_cap, "qwen21 request flags must be 0");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (!valid_prompt(request->prompt)) {
        qzi_set_error(error, error_cap,
                      "prompt is required and must be at most 16384 UTF-8 bytes");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->tier && *request->tier) {
        if (strcmp(request->tier, "base") == 0) turbo = 0;
        else if (strcmp(request->tier, "turbo") != 0) {
            qzi_set_error(error, error_cap, "tier must be \"turbo\" or \"base\"");
            return QZI_ERR_INVALID_ARGUMENT;
        }
    }
    s = check_canvas(request->width, request->height, 32, error, error_cap);
    if (s != QZI_OK) return s;
    if (turbo) {
        if (request->negative_prompt && *request->negative_prompt) {
            qzi_set_error(error, error_cap,
                          "negative_prompt is not supported on the turbo tier");
            return QZI_ERR_INVALID_ARGUMENT;
        }
        /* 0 means "caller left the default to us". */
        if (request->steps != 0 && request->steps != 8) {
            qzi_set_error(error, error_cap,
                          "turbo tier requires steps == 8");
            return QZI_ERR_INVALID_ARGUMENT;
        }
        if (request->guidance != 0.0f && request->guidance != 1.0f) {
            qzi_set_error(error, error_cap,
                          "turbo tier requires guidance == 1.0");
            return QZI_ERR_INVALID_ARGUMENT;
        }
    } else {
        if (request->steps < 0 || request->steps > 50) {
            qzi_set_error(error, error_cap, "base tier steps must be 1..50");
            return QZI_ERR_INVALID_ARGUMENT;
        }
        /* NaN/Inf fail the closed interval below and are rejected, not
         * handed to the CFG branch. 0.0f is the "tier default" sentinel. */
        if (request->guidance != 0.0f &&
            !(request->guidance >= 1.0f && request->guidance <= 20.0f)) {
            qzi_set_error(error, error_cap,
                          "base tier guidance must be finite and in 1..20");
            return QZI_ERR_INVALID_ARGUMENT;
        }
    }
    {
        int has_lora = request->lora && *request->lora;
        float scale = request->lora_scale;
        if (has_lora && !no_path_separators(request->lora)) {
            qzi_set_error_fmt(error, error_cap,
                "lora must be a bare file name of at most %u bytes "
                "(no path separators, ':', '..' or control characters)",
                (unsigned)QZI_LORA_NAME_MAX);
            return QZI_ERR_INVALID_ARGUMENT;
        }
        /* NaN/Inf fail both comparisons below, so they are rejected too.
         * 0.0f is the ABI's "engine default" sentinel, not a strength. */
        if (scale != 0.0f) {
            if (!has_lora) {
                qzi_set_error(error, error_cap,
                    "lora_scale requires lora; otherwise it would be silently dropped");
                return QZI_ERR_INVALID_ARGUMENT;
            }
            if (!(scale >= 0.05f && scale <= 4.0f)) {
                qzi_set_error(error, error_cap,
                    "lora_scale must be finite and in 0.05..4 (or 0 for the default 1.0)");
                return QZI_ERR_INVALID_ARGUMENT;
            }
        }
    }
    return QZI_OK;
}

/* Resolves the ABI's zero-valued "tier default" sentinels once, before any
 * backend sees the request. Without this, a C caller leaving steps/guidance
 * zeroed would get a different schedule from a .NET caller using the same
 * defaults, and 0/1.0 lora_scale would look like an adapter change. */
static void qzi_normalize_qwen21(const qzi_qwen21_request *request,
                                 qzi_qwen21_request *out)
{
    int turbo = !(request->tier && *request->tier &&
                  strcmp(request->tier, "base") == 0);
    *out = *request;
    if (!out->tier || !*out->tier) out->tier = "turbo";
    if (out->steps == 0) out->steps = turbo ? 8 : 40;
    if (out->guidance == 0.0f) out->guidance = turbo ? 1.0f : 4.0f;
    if (out->lora && *out->lora && out->lora_scale == 0.0f)
        out->lora_scale = 1.0f;
}

qzi_status qzi_validate_qwen21_edit(const qzi_qwen21_edit_request *request,
                                    char *error, size_t error_cap)
{
    qzi_status s;
    if (!request) {
        qzi_set_error(error, error_cap, "qwen21 edit request is NULL");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->flags != 0) {
        qzi_set_error(error, error_cap, "qwen21 edit request flags must be 0");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    s = qzi_validate_qwen21(&request->base, error, error_cap);
    if (s != QZI_OK) return s;
    if (!request->reference_rgb) {
        qzi_set_error(error, error_cap,
                      "reference_rgb is required for image editing");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->reference_width == 0 || request->reference_height == 0 ||
        request->reference_width > QZI_MAX_REFERENCE_SIDE ||
        request->reference_height > QZI_MAX_REFERENCE_SIDE ||
        (int64_t)request->reference_width * (int64_t)request->reference_height >
            QZI_MAX_PIXELS) {
        qzi_set_error_fmt(error, error_cap,
            "reference image must be 1..%u pixels per side and at most 2048*2048 total pixels",
            (unsigned)QZI_MAX_REFERENCE_SIDE);
        return QZI_ERR_INVALID_ARGUMENT;
    }
    /* NaN/Inf fail both comparisons and are rejected, not coerced. */
    if (request->strength != 0.0f &&
        !(request->strength >= 0.05f && request->strength <= 1.0f)) {
        qzi_set_error(error, error_cap,
            "strength must be finite and in 0.05..1.0 (or 0 for the engine default 0.75)");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->masking_strength != 0.0f &&
        !(request->masking_strength >= 0.05f &&
          request->masking_strength <= 1.0f)) {
        qzi_set_error(error, error_cap,
            "masking_strength must be finite and in 0.05..1.0 (or 0 for 1.0)");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (!request->mask && request->masking_strength != 0.0f) {
        qzi_set_error(error, error_cap,
            "masking_strength requires mask; otherwise it would be silently dropped");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    return QZI_OK;
}

/* Resolve the two edit sentinels once, before any backend sees the request,
 * so C and .NET callers agree on the effective values. */
static void qzi_normalize_qwen21_edit(const qzi_qwen21_edit_request *request,
                                      qzi_qwen21_edit_request *out)
{
    qzi_normalize_qwen21(&request->base, &out->base);
    out->reference_rgb = request->reference_rgb;
    out->reference_width = request->reference_width;
    out->reference_height = request->reference_height;
    out->mask = request->mask;
    out->strength = request->strength == 0.0f
                        ? QZI_EDIT_DEFAULT_STRENGTH : request->strength;
    out->masking_strength = request->mask
                                ? (request->masking_strength == 0.0f
                                       ? 1.0f : request->masking_strength)
                                : 0.0f;
    out->flags = 0;
}

/* Advanced editing validation. Everything in the embedded edit request is
 * checked by qzi_validate_qwen21_edit() first, so the extras only need to
 * validate their own invariants. Outpainting margins follow WanGP's
 * OUTPAINTING_ALIGNMENT = 32 and its "keep at least a 32 px source" rule. */
qzi_status qzi_validate_qwen21_edit_ex(const qzi_qwen21_edit_ex_request *request,
                                       char *error, size_t error_cap)
{
    qzi_status s;
    if (!request) {
        qzi_set_error(error, error_cap, "qwen21 advanced edit request is NULL");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->flags != 0) {
        qzi_set_error(error, error_cap,
                      "qwen21 advanced edit request flags must be 0");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    s = qzi_validate_qwen21_edit(&request->base, error, error_cap);
    if (s != QZI_OK) return s;

    const uint32_t alignment = QZI_OUTPAINT_ALIGNMENT;
    const uint32_t m[4] = {request->margin_top, request->margin_bottom,
                           request->margin_left, request->margin_right};
    for (int i = 0; i < 4; i++) {
        if (m[i] % alignment != 0) {
            qzi_set_error_fmt(error, error_cap,
                "outpainting margins must be multiples of %u pixels",
                (unsigned)alignment);
            return QZI_ERR_INVALID_ARGUMENT;
        }
    }
    const int64_t source_w = (int64_t)request->base.base.width -
                             (int64_t)request->margin_left -
                             (int64_t)request->margin_right;
    const int64_t source_h = (int64_t)request->base.base.height -
                             (int64_t)request->margin_top -
                             (int64_t)request->margin_bottom;
    if (source_w < (int64_t)alignment || source_h < (int64_t)alignment) {
        qzi_set_error_fmt(error, error_cap,
            "outpainting must leave at least a %ux%u source rectangle",
            (unsigned)alignment, (unsigned)alignment);
        return QZI_ERR_INVALID_ARGUMENT;
    }
    const int outpainting = request->margin_top || request->margin_bottom ||
                            request->margin_left || request->margin_right;

    if (request->model_mode != QZI_EDIT_MODE_MASKED &&
        request->model_mode != QZI_EDIT_MODE_LANPAINT_2 &&
        request->model_mode != QZI_EDIT_MODE_LANPAINT_5 &&
        request->model_mode != QZI_EDIT_MODE_LANPAINT_10 &&
        request->model_mode != QZI_EDIT_MODE_LANPAINT_15) {
        qzi_set_error(error, error_cap,
            "model_mode must be 0 (Masked Denoising) or 2/3/4/5 (LanPaint "
            "2/5/10/15 steps)");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->model_mode != QZI_EDIT_MODE_MASKED) {
        if (!request->base.mask) {
            qzi_set_error(error, error_cap,
                "LanPaint requires mask; it refines a masked edit region");
            return QZI_ERR_INVALID_ARGUMENT;
        }
        if (outpainting) {
            /* WanGP only constructs LanPaint when NOT instruction_outpainting.
             * Reject instead of silently dropping the selected method. */
            qzi_set_error(error, error_cap,
                "LanPaint and outpainting are mutually exclusive (WanGP disables "
                "LanPaint for Red Canvas outpainting)");
            return QZI_ERR_INVALID_ARGUMENT;
        }
    }
    if (request->rgba > 1u) {
        qzi_set_error(error, error_cap, "rgba must be 0 or 1");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    return QZI_OK;
}

/* Resolve the embedded edit sentinels, then apply WanGP's LanPaint override
 * (denoising_strength = masking_strength = 1.0) so both backends see the same
 * effective values. */
static void qzi_normalize_qwen21_edit_ex(const qzi_qwen21_edit_ex_request *request,
                                         qzi_qwen21_edit_ex_request *out)
{
    qzi_normalize_qwen21_edit(&request->base, &out->base);
    out->margin_top = request->margin_top;
    out->margin_bottom = request->margin_bottom;
    out->margin_left = request->margin_left;
    out->margin_right = request->margin_right;
    out->model_mode = request->model_mode;
    out->rgba = request->rgba;
    out->flags = 0;
    if (out->model_mode != QZI_EDIT_MODE_MASKED) {
        out->base.strength = 1.0f;
        out->base.masking_strength = 1.0f;
    }
}

qzi_status qzi_validate_zimage(const qzi_zimage_request *request,
                               char *error, size_t error_cap)
{
    qzi_status s;
    if (!request) {
        qzi_set_error(error, error_cap, "zimage request is NULL");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->flags != 0) {
        qzi_set_error(error, error_cap, "zimage request flags must be 0");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (!valid_prompt(request->prompt)) {
        qzi_set_error(error, error_cap,
                      "prompt is required and must be at most 16384 UTF-8 bytes");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    s = check_canvas(request->width, request->height, 16, error, error_cap);
    if (s != QZI_OK) return s;
    if (request->checkpoint && *request->checkpoint) {
        const char *c = request->checkpoint;
        if (strcmp(c, "bf16") != 0 && strcmp(c, "nunchaku") != 0 &&
            strncmp(c, "nunchaku-", 9) != 0) {
            qzi_set_error_fmt(error, error_cap,
                              "unknown zimage checkpoint \"%s\"", c);
            return QZI_ERR_UNSUPPORTED;
        }
        if (contains_fp4(c)) {
            qzi_set_error(error, error_cap,
                          "svdq-fp4 packs are not supported (int4 only)");
            return QZI_ERR_UNSUPPORTED;
        }
    }
    return QZI_OK;
}

/* ---- Z-Image ControlNet Union (additive ABI v1) -------------------------- */

/* The Union checkpoint is trained on pre-processed control images; the type
 * is descriptive metadata. Normalising it here (case-insensitively) means a
 * C caller, the .NET wrapper and the reference backend all agree on one
 * canonical string, so "PoSe" and "pose" produce byte-identical results. */
static int qzi_ascii_ieq(const char *a, const char *b)
{
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++, cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca + 32);
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb + 32);
        if (ca != cb) return 0;
    }
    return *a == *b;
}

static const char *qzi_normalize_control_type(const char *type)
{
    static const char *const names[] = {
        "auto", "pose", "canny", "depth", "scribble", "raw"
    };
    if (!type || !*type) return names[0];
    if (strlen(type) > 32) return NULL;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (qzi_ascii_ieq(type, names[i])) return names[i];
    }
    return NULL;
}

qzi_status qzi_validate_zimage_control(const qzi_zimage_control_request *request,
                                       char *error, size_t error_cap)
{
    qzi_status s;
    if (!request) {
        qzi_set_error(error, error_cap, "zimage control request is NULL");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->flags != 0) {
        qzi_set_error(error, error_cap, "zimage control request flags must be 0");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    s = qzi_validate_zimage(&request->base, error, error_cap);
    if (s != QZI_OK) return s;
    if (!request->control_rgb) {
        qzi_set_error(error, error_cap,
                      "control_rgb is required for ControlNet generation");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->control_width == 0 || request->control_height == 0 ||
        request->control_width > QZI_MAX_REFERENCE_SIDE ||
        request->control_height > QZI_MAX_REFERENCE_SIDE ||
        (int64_t)request->control_width * (int64_t)request->control_height >
            QZI_MAX_PIXELS) {
        qzi_set_error_fmt(error, error_cap,
            "control image must be 1..%u pixels per side and at most 2048*2048 total pixels",
            (unsigned)QZI_MAX_REFERENCE_SIDE);
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (!qzi_normalize_control_type(request->control_type)) {
        qzi_set_error(error, error_cap,
            "control_type must be null/auto/pose/canny/depth/scribble/raw");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    /* NaN/Inf fail both comparisons and are rejected, not coerced. */
    if (request->control_scale != 0.0f &&
        !(request->control_scale >= QZI_CONTROL_MIN_SCALE &&
          request->control_scale <= QZI_CONTROL_MAX_SCALE)) {
        qzi_set_error(error, error_cap,
            "control_scale must be finite and in 0.05..2.0 (or 0 for the engine default 0.65)");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    return QZI_OK;
}

/* Resolve the scale sentinel and canonicalise the type once, before any
 * backend sees the request, so C and .NET callers agree on effective values. */
static void qzi_normalize_zimage_control(const qzi_zimage_control_request *request,
                                         qzi_zimage_control_request *out)
{
    *out = *request;
    out->control_type = qzi_normalize_control_type(request->control_type);
    out->control_scale = request->control_scale == 0.0f
                             ? QZI_CONTROL_DEFAULT_SCALE
                             : request->control_scale;
    out->flags = 0;
}

qzi_status qzi_validate_zimage_inpaint(const qzi_zimage_inpaint_request *request,
                                       char *error, size_t error_cap)
{
    qzi_status s;
    if (!request) {
        qzi_set_error(error, error_cap, "zimage inpaint request is NULL");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->flags != 0) {
        qzi_set_error(error, error_cap, "zimage inpaint request flags must be 0");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    s = qzi_validate_zimage_control(&request->base, error, error_cap);
    if (s != QZI_OK) return s;
    if (!request->mask) {
        qzi_set_error(error, error_cap,
                      "mask is required for Z-Image inpainting");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (request->mask_width == 0 || request->mask_height == 0 ||
        request->mask_width > QZI_MAX_REFERENCE_SIDE ||
        request->mask_height > QZI_MAX_REFERENCE_SIDE ||
        (int64_t)request->mask_width * (int64_t)request->mask_height >
            QZI_MAX_PIXELS) {
        qzi_set_error_fmt(error, error_cap,
            "inpaint mask must be 1..%u pixels per side and at most 2048*2048 total pixels",
            (unsigned)QZI_MAX_REFERENCE_SIDE);
        return QZI_ERR_INVALID_ARGUMENT;
    }
    return QZI_OK;
}

/* Same sentinel / type normalisation as a plain ControlNet request; the mask
 * plane and its dimensions are passed through verbatim (each is cover-fitted
 * to the output canvas by the backend). */
static void qzi_normalize_zimage_inpaint(const qzi_zimage_inpaint_request *request,
                                         qzi_zimage_inpaint_request *out)
{
    qzi_normalize_zimage_control(&request->base, &out->base);
    out->mask = request->mask;
    out->mask_width = request->mask_width;
    out->mask_height = request->mask_height;
    out->flags = 0;
}

/* ---- backend registration ---------------------------------------------- */

#ifdef QZI_HAS_REAL_BACKEND
/* Implemented by src/qzi_engine_backend.c, linked into this same shared
 * library only when -DQZI_WITH_CUDA_ENGINE=ON. Compile-time selection avoids
 * constructor-order hazards on Windows and keeps the reference build free of
 * any engine symbols. qzi_register_backend(NULL) still restores the
 * compiled-in default backend for in-process comparison. */
extern const qzi_backend *qzi_real_backend(void);
#endif

static void ensure_backend(void)
{
    if (g_backend_selected) return;
#ifdef QZI_HAS_REAL_BACKEND
    g_backend = *qzi_real_backend();
#else
    g_backend = qzi_reference_vtable;
#endif
    g_backend_selected = 1;
}

QZI_API uint32_t qzi_abi_version(void)
{
    return QZI_ABI_VERSION;
}

QZI_API uint32_t qzi_abi_features(void)
{
    return QZI_FEATURE_QWEN21_EDIT | QZI_FEATURE_ZIMAGE_CONTROL |
           QZI_FEATURE_QWEN21_EDIT_ADV | QZI_FEATURE_ZIMAGE_INPAINT;
}

QZI_API size_t qzi_qwen21_edit_layout(void)
{
    return sizeof(qzi_qwen21_edit_request);
}

QZI_API size_t qzi_zimage_control_layout(void)
{
    return sizeof(qzi_zimage_control_request);
}

QZI_API size_t qzi_qwen21_edit_ex_layout(void)
{
    return sizeof(qzi_qwen21_edit_ex_request);
}

QZI_API size_t qzi_zimage_inpaint_layout(void)
{
    return sizeof(qzi_zimage_inpaint_request);
}

QZI_API const char *qzi_status_text(qzi_status status)
{
    switch (status) {
    case QZI_OK:                   return "ok";
    case QZI_ERR_INVALID_ARGUMENT: return "invalid argument";
    case QZI_ERR_NOT_CONFIGURED:   return "model pack not configured";
    case QZI_ERR_UNSUPPORTED:      return "unsupported feature or checkpoint";
    case QZI_ERR_BUSY:             return "a generation owns the device";
    case QZI_ERR_CANCELLED:        return "cancelled";
    case QZI_ERR_OUT_OF_MEMORY:    return "out of memory";
    case QZI_ERR_ENGINE:           return "engine failure";
    case QZI_ERR_INTERNAL:         return "internal error";
    default:                       return "unknown status";
    }
}

QZI_API qzi_status qzi_abi_layout(size_t *sizes, size_t count)
{
    if (!sizes || count < QZI_LAYOUT_SLOTS) return QZI_ERR_INVALID_ARGUMENT;
    sizes[QZI_LAYOUT_ENGINE_CONFIG]  = sizeof(qzi_engine_config);
    sizes[QZI_LAYOUT_QWEN21_REQUEST] = sizeof(qzi_qwen21_request);
    sizes[QZI_LAYOUT_ZIMAGE_REQUEST] = sizeof(qzi_zimage_request);
    sizes[QZI_LAYOUT_CALLBACKS]      = sizeof(qzi_callbacks);
    sizes[QZI_LAYOUT_IMAGE]          = sizeof(qzi_image);
    sizes[QZI_LAYOUT_HEALTH]         = sizeof(qzi_health);
    return QZI_OK;
}

QZI_API qzi_status qzi_register_backend(const qzi_backend *backend)
{
    if (!backend) {
        g_backend_selected = 0;
        ensure_backend();
        return QZI_OK;
    }
    if (backend->abi_version != QZI_ABI_VERSION || !backend->name ||
        !backend->init || !backend->shutdown ||
        !backend->generate_qwen21 || !backend->generate_zimage ||
        !backend->release_image) {
        return QZI_ERR_INVALID_ARGUMENT;
    }
    g_backend = *backend;
    g_backend_selected = 1;
    return QZI_OK;
}

/* ---- engine lifetime --------------------------------------------------- */

QZI_API qzi_status qzi_engine_create(const qzi_engine_config *config,
                                     qzi_engine **out_engine,
                                     char *error, size_t error_cap)
{
    qzi_engine *engine;
    qzi_status status;

    if (!config || !out_engine || !error || error_cap == 0) {
        qzi_set_error(error, error_cap, "config, out_engine and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    *out_engine = NULL;
    error[0] = '\0';
    if (config->abi_version != QZI_ABI_VERSION) {
        qzi_set_error_fmt(error, error_cap,
                          "ABI mismatch: caller %u, native %u",
                          config->abi_version, QZI_ABI_VERSION);
        return QZI_ERR_UNSUPPORTED;
    }
    if (config->flags != 0 || config->device_id < -1) {
        qzi_set_error(error, error_cap, "invalid engine config");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    ensure_backend();

    engine = (qzi_engine *)calloc(1, sizeof(*engine));
    if (!engine) {
        qzi_set_error(error, error_cap, "engine allocation failed");
        return QZI_ERR_OUT_OF_MEMORY;
    }
    engine->backend = g_backend;
    engine->started = qzi_now_seconds();
    mutex_init(&engine->mutex);

    status = engine->backend.init(&engine->context, config, error, error_cap);
    if (status != QZI_OK) {
        mutex_destroy(&engine->mutex);
        free(engine);
        if (error[0] == '\0')
            qzi_set_error(error, error_cap, qzi_status_text(status));
        return status;
    }
    *out_engine = engine;
    return QZI_OK;
}

QZI_API void qzi_engine_destroy(qzi_engine *engine)
{
    if (!engine) return;
    if (engine->backend.shutdown) engine->backend.shutdown(engine->context);
    mutex_destroy(&engine->mutex);
    free(engine);
}

/* ---- health / unload --------------------------------------------------- */

QZI_API qzi_status qzi_engine_health(qzi_engine *engine, qzi_health *out,
                                     char *error, size_t error_cap)
{
    qzi_status status;
    if (!engine || !out) {
        qzi_set_error(error, error_cap, "engine and out are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    out->resident_model = engine->resident;
    out->busy = engine->busy;
    out->queue_depth = 0;
    out->device_id = -1;
    out->free_vram_mib = -1;
    out->total_vram_mib = -1;
    out->served = engine->served;
    out->rejected = engine->rejected;
    out->uptime_seconds = qzi_now_seconds() - engine->started;
    out->reference_backend = 0;
    if (engine->backend.health) {
        status = engine->backend.health(engine->context, out, error, error_cap);
        if (status != QZI_OK) return status;
    }
    return QZI_OK;
}

QZI_API qzi_status qzi_engine_unload(qzi_engine *engine,
                                     char *error, size_t error_cap)
{
    qzi_status status;
    if (!engine) {
        qzi_set_error(error, error_cap, "engine is required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    if (!mutex_try_lock(&engine->mutex)) {
        qzi_set_error(error, error_cap,
                      "a generation owns the device; unload refused while busy");
        return QZI_ERR_BUSY;
    }
    status = engine->backend.unload
                 ? engine->backend.unload(engine->context, error, error_cap)
                 : QZI_OK;
    mutex_unlock(&engine->mutex);
    /* DiT stays resident by design; only re-creatable caches are released. */
    return status;
}

/* ---- generation -------------------------------------------------------- */

/* Two shared lock helpers keep the public entry points free of casts while
 * preserving the single-flight contract (one GPU, one resident model). */
static int generation_enter(qzi_engine *engine, char *error, size_t error_cap)
{
    if (!mutex_try_lock(&engine->mutex)) {
        qzi_set_error(error, error_cap,
                      "a generation owns the device; concurrent generate refused");
        engine->rejected++;
        return 0;
    }
    engine->busy = 1;
    return 1;
}

static qzi_status generation_leave(qzi_engine *engine, qzi_status status,
                                   qzi_image *out_image, qzi_model_kind model,
                                   char *error, size_t error_cap)
{
    engine->busy = 0;
    if (status == QZI_OK) {
        if (!out_image->data || out_image->size == 0) {
            qzi_set_error(error, error_cap, "backend returned an empty image");
            engine->rejected++;
            status = QZI_ERR_ENGINE;
        } else {
            out_image->token = (uint64_t)(uintptr_t)out_image->data;
            engine->served++;
            engine->resident = model;
        }
    } else {
        engine->rejected++;
        if (error[0] == '\0') qzi_set_error(error, error_cap, qzi_status_text(status));
    }
    mutex_unlock(&engine->mutex);
    return status;
}

QZI_API qzi_status qzi_generate_qwen21(qzi_engine *engine,
                                       const qzi_qwen21_request *request,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap)
{
    qzi_status status;
    qzi_qwen21_request normalized;
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_qwen21(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    qzi_normalize_qwen21(request, &normalized);
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_qwen21(engine->context, &normalized,
                                             callbacks, out_image, error,
                                             error_cap);
    return generation_leave(engine, status, out_image, QZI_MODEL_QWEN21,
                            error, error_cap);
}

QZI_API qzi_status qzi_generate_zimage(qzi_engine *engine,
                                       const qzi_zimage_request *request,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap)
{
    qzi_status status;
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_zimage(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_zimage(engine->context, request, callbacks,
                                             out_image, error, error_cap);
    return generation_leave(engine, status, out_image, QZI_MODEL_ZIMAGE,
                            error, error_cap);
}

QZI_API qzi_status qzi_generate_qwen21_edit(qzi_engine *engine,
                                             const qzi_qwen21_edit_request *request,
                                             const qzi_callbacks *callbacks,
                                             qzi_image *out_image,
                                             char *error, size_t error_cap)
{
    qzi_status status;
    qzi_qwen21_edit_request normalized;
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_qwen21_edit(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    if (!engine->backend.generate_qwen21_edit) {
        engine->rejected++;
        qzi_set_error(error, error_cap,
                      "the registered native backend does not implement image editing");
        return QZI_ERR_UNSUPPORTED;
    }
    qzi_normalize_qwen21_edit(request, &normalized);
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_qwen21_edit(engine->context, &normalized,
                                                  callbacks, out_image, error,
                                                  error_cap);
    return generation_leave(engine, status, out_image, QZI_MODEL_QWEN21,
                            error, error_cap);
}

QZI_API qzi_status qzi_generate_zimage_control(qzi_engine *engine,
                                                const qzi_zimage_control_request *request,
                                                const qzi_callbacks *callbacks,
                                                qzi_image *out_image,
                                                char *error, size_t error_cap)
{
    qzi_status status;
    qzi_zimage_control_request normalized;
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_zimage_control(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    if (!engine->backend.generate_zimage_control) {
        engine->rejected++;
        qzi_set_error(error, error_cap,
                      "the registered native backend does not implement Z-Image ControlNet");
        return QZI_ERR_UNSUPPORTED;
    }
    qzi_normalize_zimage_control(request, &normalized);
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_zimage_control(engine->context, &normalized,
                                                     callbacks, out_image, error,
                                                     error_cap);
    return generation_leave(engine, status, out_image, QZI_MODEL_ZIMAGE,
                            error, error_cap);
}

QZI_API qzi_status qzi_generate_zimage_inpaint(qzi_engine *engine,
                                                const qzi_zimage_inpaint_request *request,
                                                const qzi_callbacks *callbacks,
                                                qzi_image *out_image,
                                                char *error, size_t error_cap)
{
    qzi_status status;
    qzi_zimage_inpaint_request normalized;
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_zimage_inpaint(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    if (!engine->backend.generate_zimage_inpaint) {
        engine->rejected++;
        qzi_set_error(error, error_cap,
            "the registered native backend does not implement Z-Image inpainting");
        return QZI_ERR_UNSUPPORTED;
    }
    qzi_normalize_zimage_inpaint(request, &normalized);
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_zimage_inpaint(engine->context, &normalized,
                                                     callbacks, out_image, error,
                                                     error_cap);
    return generation_leave(engine, status, out_image, QZI_MODEL_ZIMAGE,
                            error, error_cap);
}

QZI_API qzi_status qzi_generate_qwen21_edit_ex(qzi_engine *engine,
                                               const qzi_qwen21_edit_ex_request *request,
                                               const qzi_callbacks *callbacks,
                                               qzi_image *out_image,
                                               char *error, size_t error_cap)
{
    qzi_status status;
    qzi_qwen21_edit_ex_request normalized;
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_qwen21_edit_ex(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    if (!engine->backend.generate_qwen21_edit_ex) {
        engine->rejected++;
        qzi_set_error(error, error_cap,
            "the registered native backend does not implement advanced image editing");
        return QZI_ERR_UNSUPPORTED;
    }
    qzi_normalize_qwen21_edit_ex(request, &normalized);
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_qwen21_edit_ex(engine->context, &normalized,
                                                     callbacks, out_image, error,
                                                     error_cap);
    return generation_leave(engine, status, out_image, QZI_MODEL_QWEN21,
                            error, error_cap);
}

QZI_API void qzi_image_release(qzi_engine *engine, qzi_image *image)
{
    if (!engine || !image || !image->data) return;
    if (engine->backend.release_image)
        engine->backend.release_image(engine->context, image);
    memset(image, 0, sizeof(*image));
}
