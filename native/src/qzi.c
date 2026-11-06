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

static int no_path_separators(const char *name)
{
    if (strchr(name, '/') || strchr(name, '\\')) return 0;
    if (strstr(name, "..")) return 0;
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
        if (request->guidance < 0.0f ||
            (request->guidance > 0.0f &&
             (request->guidance < 1.0f || request->guidance > 20.0f))) {
            qzi_set_error(error, error_cap, "base tier guidance must be 1..20");
            return QZI_ERR_INVALID_ARGUMENT;
        }
    }
    if (request->lora && *request->lora) {
        if (!no_path_separators(request->lora)) {
            qzi_set_error(error, error_cap,
                          "lora must be a bare file name without path separators");
            return QZI_ERR_INVALID_ARGUMENT;
        }
        if (request->lora_scale > 0.0f &&
            (request->lora_scale < 0.05f || request->lora_scale > 4.0f)) {
            qzi_set_error(error, error_cap, "lora_scale must be 0.05..4");
            return QZI_ERR_INVALID_ARGUMENT;
        }
    }
    return QZI_OK;
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
        if (strstr(c, "fp4") != NULL) {
            qzi_set_error(error, error_cap,
                          "svdq-fp4 packs are not supported (int4 only)");
            return QZI_ERR_UNSUPPORTED;
        }
    }
    return QZI_OK;
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
    if (!engine || !out_image || !error || error_cap == 0) {
        qzi_set_error(error, error_cap,
                      "engine, out_image and error buffer are required");
        return QZI_ERR_INVALID_ARGUMENT;
    }
    memset(out_image, 0, sizeof(*out_image));
    error[0] = '\0';
    status = qzi_validate_qwen21(request, error, error_cap);
    if (status != QZI_OK) { engine->rejected++; return status; }
    if (!generation_enter(engine, error, error_cap)) return QZI_ERR_BUSY;
    status = engine->backend.generate_qwen21(engine->context, request, callbacks,
                                             out_image, error, error_cap);
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

QZI_API void qzi_image_release(qzi_engine *engine, qzi_image *image)
{
    if (!engine || !image || !image->data) return;
    if (engine->backend.release_image)
        engine->backend.release_image(engine->context, image);
    memset(image, 0, sizeof(*image));
}
