/* Portable environment lookup. See native/include/qzi_env.h for the rationale
 * (Windows CRT snapshots the environment and cannot see .NET's Win32 writes).
 */

#include "qzi_env.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

int qzi_getenv(const char *name, char *buffer, size_t capacity)
{
    if (!name || !*name || !buffer || capacity == 0) return 0;
    buffer[0] = '\0';

#ifdef _WIN32
    {
        /* A zero-length value and a missing variable both make the copying
         * call return 0, so ask for the required size first. */
        DWORD needed = GetEnvironmentVariableA(name, NULL, 0);
        DWORD written;
        if (needed == 0) return 0; /* ERROR_ENVVAR_NOT_FOUND */
        if (needed > (DWORD)capacity) {
            buffer[0] = '\0';
            return -1;
        }
        written = GetEnvironmentVariableA(name, buffer, (DWORD)capacity);
        if (written >= (DWORD)capacity) {
            buffer[0] = '\0';
            return -1;
        }
        return 1; /* written == 0 means "set but empty", which is still found */
    }
#else
    {
        const char *value = getenv(name);
        size_t length;
        if (!value) return 0;
        length = strlen(value);
        if (length >= capacity) {
            buffer[0] = '\0';
            return -1;
        }
        memcpy(buffer, value, length + 1u);
        return 1;
    }
#endif
}
