/*
 * custom_backend_template.c -- starting point for an alternative engine backend.
 *
 * This file is NOT compiled by native/CMakeLists.txt. It is a template that
 * shows the exact shape of a qzi_backend implementation. It deliberately does
 * copy any vendored engine source: the TODO regions below are where an
 * integrator wires in its own model-loading / denoise / VAE calls. For the
 * real backend shipped with this repository, see native/src/qzi_engine_backend.c.
 *
 * Until those TODOs are filled in, every generate_* call transparently
 * delegates to the bundled deterministic reference backend, so the vtable is
 * immediately testable against the same ABI, progress phases and buffer
 * ownership. Define QZI_TEMPLATE_USE_REFERENCE_FALLBACK=0 to remove the
 * fallback once the real engine is wired up.
 *
 * Build example (linking against the bundled qzi shared library):
 *   cc -std=c11 -I native/include -shared -fPIC \
 *      integration/custom_backend_template.c -L artifacts/native/<platform> \
 *      -lQwenZImageNative -o libQwenZImageCustom.so
 */

#include "qzi.h"
#include "qzi_backend.h"

#include <stdlib.h>
#include <string.h>

#ifndef QZI_TEMPLATE_USE_REFERENCE_FALLBACK
#  define QZI_TEMPLATE_USE_REFERENCE_FALLBACK 1
#endif

/* Per-engine state. qzi.c owns this pointer and passes it to every callback. */
typedef struct custom_state {
    char *qwen21_dir;
    char *zimage_dir;
    char *cache_dir;
    int   device_id;
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    void *reference_context;
#endif
} custom_state;

static void set_error(char *error, size_t cap, const char *message)
{
    if (!error || cap == 0) return;
    size_t n = strlen(message);
    if (n >= cap) n = cap - 1;
    memcpy(error, message, n);
    error[n] = '\0';
}

static char *copy_string(const char *text)
{
    if (!text) return NULL;
    size_t n = strlen(text) + 1;
    char *copy = (char *)malloc(n);
    if (copy) memcpy(copy, text, n);
    return copy;
}

/* qzi.c validates the ABI version before calling init. Resolve paths and
 * check the device here, but do NOT mmap multi-GB weights: load them lazily on
 * the first generate so a bad path fails on the first real request. */
static qzi_status custom_init(void **out_context, const qzi_engine_config *config,
                          char *error, size_t error_cap)
{
    if (!out_context || !config) return QZI_ERR_INVALID_ARGUMENT;
    *out_context = NULL;

    custom_state *state = (custom_state *)calloc(1, sizeof(*state));
    if (!state) {
        set_error(error, error_cap, "out of memory creating backend state");
        return QZI_ERR_OUT_OF_MEMORY;
    }
    state->device_id = config->device_id;
    state->qwen21_dir = copy_string(config->qwen21_dir);
    state->zimage_dir = copy_string(config->zimage_dir);
    state->cache_dir  = copy_string(config->cache_dir);

    /* TODO: verify the model packs exist, select the CUDA device, create the
     * text-embedding cache, etc. Report problems through `error`. */

#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    const qzi_backend *reference = qzi_reference_backend();
    qzi_status s = reference->init(&state->reference_context, config,
                                   error, error_cap);
    if (s != QZI_OK) {
        custom_state *tmp = state;
        /* shutdown is not called for a failed init, so clean up here */
        free(tmp->qwen21_dir); free(tmp->zimage_dir); free(tmp->cache_dir);
        free(tmp);
        return s;
    }
#endif

    *out_context = state;
    return QZI_OK;
}

static void custom_shutdown(void *context)
{
    custom_state *state = (custom_state *)context;
    if (!state) return;
    /* TODO: unmap weights, destroy CUDA streams, free the text cache. */
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    qzi_reference_backend()->shutdown(state->reference_context);
#endif
    free(state->qwen21_dir);
    free(state->zimage_dir);
    free(state->cache_dir);
    free(state);
}

/*
 * Both generate_* implementations must:
 *   1. emit "model load" / "text" / "dit" (one tick per denoise step) /
 *      "vae" / "encode" through callbacks->progress when it is non-NULL
 *      (image edits additionally emit "image encode" for the reference VAE
 *      pass, see the optional generate_qwen21_edit slot below);
 *   2. poll callbacks->should_cancel between phases and steps and return
 *      QZI_ERR_CANCELLED if it asks to stop;
 *   3. write a complete PNG into out_image->data / ->size / ->width /
 *      ->height / ->token, using memory that release_image() will free;
 *   4. write a human-readable message into `error` on every failure.
 * qzi.c already performs validation and single-flight serialisation, so a
 * backend never sees two concurrent generates and does not need to repeat the
 * canvas / tier / checkpoint checks (it may still do so defensively).
 */
static qzi_status custom_generate_qwen21(void *context,
                                     const qzi_qwen21_request *request,
                                     const qzi_callbacks *callbacks,
                                     qzi_image *out_image,
                                     char *error, size_t error_cap)
{
    custom_state *state = (custom_state *)context;
    (void)state; (void)request; (void)callbacks; (void)error; (void)error_cap;

    /* TODO: wire in the real Qwen-Image 2.1 pipeline, e.g.
     *   - resolve state->qwen21_dir + request->tier (turbo/base) + request->lora
     *   - text encode (progress "text", cancel check)
     *   - denoise with request->steps / guidance / seed (progress "dit")
     *   - VAE decode into RGB8 (progress "vae")
     *   - PNG encode (progress "encode"), assign out_image->*
     * See native/src/qzi_engine_backend.c and engine/src/qzen_qwen21_pipeline.h
     * for a complete implementation; keep the DiT resident across calls.
     */
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    return qzi_reference_backend()->generate_qwen21(
        state->reference_context, request, callbacks, out_image, error, error_cap);
#else
    set_error(error, error_cap, "custom Qwen21 backend is not implemented yet");
    return QZI_ERR_ENGINE;
#endif
}

static qzi_status custom_generate_zimage(void *context,
                                     const qzi_zimage_request *request,
                                     const qzi_callbacks *callbacks,
                                     qzi_image *out_image,
                                     char *error, size_t error_cap)
{
    custom_state *state = (custom_state *)context;
    (void)state; (void)request; (void)callbacks; (void)error; (void)error_cap;

    /* TODO: wire in the real Z-Image Turbo 8-step schedule, e.g.
     *   - state->zimage_dir + request->checkpoint (bf16 / nunchaku / pack id)
     *   - qzen_zpipe_load / qzen_zpipe_encode_text / qzen_zpipe_denoise /
     *     qzen_zvae_decode (see engine/src/qzen_zimage_pipeline.h)
     *   - there is no CFG branch: no negative prompt, no guidance
     * Orchestration reference: native/src/qzi_engine_backend.c.
     */
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    return qzi_reference_backend()->generate_zimage(
        state->reference_context, request, callbacks, out_image, error, error_cap);
#else
    set_error(error, error_cap, "custom Z-Image backend is not implemented yet");
    return QZI_ERR_ENGINE;
#endif
}

static qzi_status custom_health(void *context, qzi_health *out,
                            char *error, size_t error_cap)
{
    custom_state *state = (custom_state *)context;
    (void)state; (void)error; (void)error_cap;
    if (!out) return QZI_ERR_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));

    /* TODO: report the real resident model, device id and free/total VRAM.
     * qzi.c fills in busy / queue_depth / served / rejected / uptime. */
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    qzi_status s = qzi_reference_backend()->health(
        state->reference_context, out, error, error_cap);
    if (s != QZI_OK) return s;
#endif
    out->reference_backend = 0;   /* this is a real engine */
    return QZI_OK;
}

static qzi_status custom_unload(void *context, char *error, size_t error_cap)
{
    custom_state *state = (custom_state *)context;
    (void)state; (void)error; (void)error_cap;
    /* TODO: drop re-creatable state (text tower / caches). Never free the DiT:
     * reloading it costs minutes. qzi.c refuses this call while busy. */
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    return qzi_reference_backend()->unload(
        state->reference_context, error, error_cap);
#else
    return QZI_OK;
#endif
}

static void custom_release_image(void *context, qzi_image *image)
{
    custom_state *state = (custom_state *)context;
    /* TODO: free whatever generate_* allocated behind image->data. The ABI
     * layer zeroes the struct afterwards; qzi_image_release is idempotent for
     * an already-zeroed image. */
#if QZI_TEMPLATE_USE_REFERENCE_FALLBACK
    qzi_reference_backend()->release_image(state->reference_context, image);
#else
    (void)state;
    free((void *)image->data);
    image->data = NULL;
    image->size = 0;
    image->token = 0;
#endif
}

static const qzi_backend custom_backend_vtable = {
    QZI_ABI_VERSION,
    "custom-engine",
    custom_init,
    custom_shutdown,
    custom_generate_qwen21,
    custom_generate_zimage,
    custom_health,
    custom_unload,
    custom_release_image,
    /* Optional, additive in ABI v1. Leaving it NULL keeps t2i working and
     * makes qzi_generate_qwen21_edit() answer QZI_ERR_UNSUPPORTED instead of
     * calling through a null pointer. Implement it (see qzi_backend.h) to
     * support Qwen-Image 2.1 image editing. */
    NULL,
    /* Optional, additive in ABI v1: Z-Image Turbo Fun ControlNet Union. NULL
     * makes qzi_generate_zimage_control() answer QZI_ERR_UNSUPPORTED while
     * every older generate path keeps working. */
    NULL,
    /* Optional, additive in ABI v1: advanced Qwen-Image 2.1 editing
     * (outpainting / LanPaint / RGBA). NULL makes
     * qzi_generate_qwen21_edit_ex() answer QZI_ERR_UNSUPPORTED. */
    NULL,
    /* Optional, additive in ABI v1: Z-Image Turbo Fun ControlNet Union 2.x
     * mask inpainting. NULL makes qzi_generate_zimage_inpaint() answer
     * QZI_ERR_UNSUPPORTED while every older generate path keeps working.
     * All four optional slots stay at the END of the vtable, in this order,
     * so an integrator that recompiles against qzi_backend.h keeps every
     * existing slot at the same offset. */
    NULL
};

/* Call this BEFORE qzi_engine_create(). The vtable is copied, so a static
 * const is fine. This module is a *client* of QwenZImageNative.dll, so the declaration is
 * intentionally plain (no QZI_API): on Windows the qzi_* declarations in the
 * header are then dllimport, which is exactly what linking against
 * QwenZImageNative.dll needs. Export this function from your own shared library if the host loads
 * it dynamically. An engine that replaces qzi.c entirely should instead
 * export the qzi_* functions from qzi.h itself and skip registration. */
qzi_status qzi_register_custom_backend(void)
{
    return qzi_register_backend(&custom_backend_vtable);
}
