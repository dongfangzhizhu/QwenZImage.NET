/* Internal helper shared by the qzi control layer, the reference backend and
 * the vendored engine's config loader. NOT part of the public C ABI v1:
 * qzi_getenv() is a hidden symbol inside QwenZImageNative, not an export.
 *
 * Why this exists instead of plain getenv():
 *   On Windows the C runtime keeps a private snapshot of the process
 *   environment, taken when the CRT initialises. .NET's
 *   Environment.SetEnvironmentVariable() calls the Win32 API directly, so a
 *   QZI_* value published by managed code after startup is invisible to the
 *   native DLL's getenv(). Reading through GetEnvironmentVariableA() sees the
 *   live Win32 environment, which is exactly what QziClient scopes around
 *   qzi_engine_create(). On POSIX setenv()/getenv() already share `environ`.
 */
#ifndef QZI_ENV_H
#define QZI_ENV_H

#include <stddef.h>

/* Copies the NUL-terminated value of `name` into `buffer`.
 * Returns 1 when found, 0 when unset, -1 when the value does not fit
 * (buffer is left empty so callers can report a clean error). */
int qzi_getenv(const char *name, char *buffer, size_t capacity);

#endif /* QZI_ENV_H */
