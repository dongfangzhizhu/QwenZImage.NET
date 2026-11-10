/*
 * qzi_backend.h -- pluggable generation backend for the reference ABI.
 *
 * The C ABI in qzi.h is the managed/native contract. This header is the
 * native/native contract: a backend implements one qzi_backend vtable and
 * either calls qzi_register_backend(&vtable) or is selected at compile time
 * (see QZI_HAS_REAL_BACKEND in qzi.c). The whole managed stack works unchanged.
 *
 * The vtable is copied by qzi_register_backend, so it may live on the stack.
 * All callbacks run on the calling thread except where noted. Every function
 * that can fail writes a NUL-terminated message into `error`.
 */

#ifndef QZI_BACKEND_H
#define QZI_BACKEND_H

#include "qzi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qzi_backend {
    uint32_t     abi_version;      /* QZI_ABI_VERSION */
    const char  *name;             /* static literal, e.g. "QwenZImageNative" */

    /* Called once by qzi_engine_create. The backend allocates its own
     * per-engine state and writes it to *out_context; qzi.c owns the pointer
     * and passes it to every other callback, then to shutdown(). Resolve model
     * directories and validate the device here, but do NOT mmap multi-GB
     * weights yet -- do that lazily in generate_* so a bad path fails on the
     * first real request, like the resident inference backend. */
    qzi_status (*init)(void **out_context, const qzi_engine_config *config,
                       char *error, size_t error_cap);
    void       (*shutdown)(void *context);

    /* Both generate_* calls are synchronous and may take minutes. They must
     * report progress, poll should_cancel between phases/steps, and honour
     * the single-flight contract (the ABI layer already serialises callers,
     * so a backend never sees two concurrent generates). */
    qzi_status (*generate_qwen21)(void *context,
                                  const qzi_qwen21_request *request,
                                  const qzi_callbacks *callbacks,
                                  qzi_image *out_image,
                                  char *error, size_t error_cap);
    qzi_status (*generate_zimage)(void *context,
                                 const qzi_zimage_request *request,
                                 const qzi_callbacks *callbacks,
                                 qzi_image *out_image,
                                 char *error, size_t error_cap);

    qzi_status (*health)(void *context, qzi_health *out,
                         char *error, size_t error_cap);
    /* Release re-creatable resident state, never the DiT (minute-scale reload). */
    qzi_status (*unload)(void *context, char *error, size_t error_cap);
    /* Free the buffer behind out_image->data. The ABI layer zeroes the struct
     * afterwards. */
    void       (*release_image)(void *context, qzi_image *image);

    /* Optional, additive in ABI v1: Qwen-Image 2.1 image editing. A backend
     * that leaves this NULL reports QZI_ERR_UNSUPPORTED from
     * qzi_generate_qwen21_edit() while t2i keeps working unchanged. This field
     * was LAST before ControlNet; both optional fields stay at the END of the
     * vtable so an integrator that recompiles against this header (required)
     * keeps every existing slot at the same offset. */
    qzi_status (*generate_qwen21_edit)(void *context,
                                       const qzi_qwen21_edit_request *request,
                                       const qzi_callbacks *callbacks,
                                       qzi_image *out_image,
                                       char *error, size_t error_cap);

    /* Optional, additive in ABI v1: Z-Image Turbo Fun ControlNet Union. NULL
     * makes qzi_generate_zimage_control() report QZI_ERR_UNSUPPORTED while the
     * three older generate paths keep working unchanged. */
    qzi_status (*generate_zimage_control)(void *context,
                                          const qzi_zimage_control_request *request,
                                          const qzi_callbacks *callbacks,
                                          qzi_image *out_image,
                                          char *error, size_t error_cap);

    /* Optional, additive in ABI v1: advanced Qwen-Image 2.1 editing
     * (outpainting / LanPaint / RGBA). NULL makes
     * qzi_generate_qwen21_edit_ex() report QZI_ERR_UNSUPPORTED while all four
     * older generate paths keep working unchanged. Always the LAST slot. */
    qzi_status (*generate_qwen21_edit_ex)(void *context,
                                          const qzi_qwen21_edit_ex_request *request,
                                          const qzi_callbacks *callbacks,
                                          qzi_image *out_image,
                                          char *error, size_t error_cap);
} qzi_backend;

/* Filled in by the built-in deterministic reference backend. Exposed so an
 * integrator can wrap it (e.g. to compare outputs against a real engine). */
QZI_API const qzi_backend *qzi_reference_backend(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QZI_BACKEND_H */
