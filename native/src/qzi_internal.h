/* Shared helpers for the reference ABI implementation. Not part of the public
 * contract -- engine integrators do not need this header. */

#ifndef QZI_INTERNAL_H
#define QZI_INTERNAL_H

#include "qzi.h"
#include "qzi_backend.h"

void        qzi_set_error(char *error, size_t error_cap, const char *text);
void        qzi_set_error_fmt(char *error, size_t error_cap, const char *fmt, ...);
double      qzi_now_seconds(void);
int         qzi_cancelled(const qzi_callbacks *callbacks);
void        qzi_emit_progress(const qzi_callbacks *callbacks, const char *phase,
                              int32_t completed, int32_t total, double start);
uint64_t    qzi_fnv1a(const char *text);
qzi_status  qzi_validate_qwen21(const qzi_qwen21_request *request,
                                char *error, size_t error_cap);
qzi_status  qzi_validate_qwen21_edit(const qzi_qwen21_edit_request *request,
                                     char *error, size_t error_cap);
qzi_status  qzi_validate_qwen21_edit_ex(const qzi_qwen21_edit_ex_request *request,
                                        char *error, size_t error_cap);
qzi_status  qzi_validate_zimage(const qzi_zimage_request *request,
                                char *error, size_t error_cap);
qzi_status  qzi_validate_zimage_control(const qzi_zimage_control_request *request,
                                        char *error, size_t error_cap);
qzi_status  qzi_validate_zimage_inpaint(const qzi_zimage_inpaint_request *request,
                                        char *error, size_t error_cap);

/* The built-in deterministic backend vtable (defined in qzi_stub.c). */
extern const qzi_backend qzi_reference_vtable;

#endif /* QZI_INTERNAL_H */
