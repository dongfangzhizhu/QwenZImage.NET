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
} qzi_backend;

/* Filled in by the built-in deterministic reference backend. Exposed so an
 * integrator can wrap it (e.g. to compare outputs against a real engine). */
QZI_API const qzi_backend *qzi_reference_backend(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QZI_BACKEND_H */
