/* Win32 implementations of the POSIX calls qzen_platform.h maps onto.
 *
 * Everything here is small on purpose: the engine's use of each call is
 * narrow, and a narrow shim is one that can be reasoned about.  The two that
 * are not trivial are pread (must keep 16 loader lanes independent, see
 * qzen_gpu.cu's read fan-out) and sigaction (Ctrl+C has to cancel a running
 * generation the way SIGINT does).
 */
#include "qzen_platform.h"

#include <windows.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

/* This file implements the POSIX names qzen_platform.h #defines onto the shims,
 * so every call it makes to the CRT's own version has to escape the macro
 * first: `getenv` here would otherwise expand to qzen_win_getenv and recurse
 * until the stack is gone.  After the includes, because that is where the
 * macro comes from. */
#undef getenv

/* ------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------- */

int qzen_win_mkdir(const char *path) {
    if (!path || !*path) {
        errno = EINVAL;
        return -1;
    }
    return _mkdir(path);
}

int qzen_win_setenv(const char *name, const char *value, int overwrite) {
    if (!name || !*name || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value ? value : "") == 0 ? 0 : -1;
}

int qzen_win_unsetenv(const char *name) {
    if (!name || !*name) {
        errno = EINVAL;
        return -1;
    }
    /* _putenv_s with an empty value removes the variable. */
    return _putenv_s(name, "") == 0 ? 0 : -1;
}

/* The engine asks for HOME when it wants ~/.cache; Windows spells that
 * USERPROFILE.  Doing the translation here keeps every call site unchanged. */
const char *qzen_win_getenv(const char *name) {
    const char *value = getenv(name);
    if (value && *value) return value;
    if (name && strcmp(name, "HOME") == 0) {
        value = getenv("USERPROFILE");
        if (value && *value) return value;
    }
    return value;
}

int qzen_win_getpid(void) {
    return (int)GetCurrentProcessId();
}

/* The engine writes UTF-8 to stderr/stdout (see /utf-8 in the build flags).
 * A console created with the system ANSI code page would decode those bytes as
 * GBK/CP1252 and mangle both the usage text and the Wan negative prompt, so
 * the console is switched over at start-up.  Fails silently when there is no
 * console (output redirected to a file or a pipe), which is the common case
 * for the WebUI, and is exactly when nothing needs converting. */
void qzen_win_console_init(void) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
}

/* ------------------------------------------------------------------------
 * open
 * ---------------------------------------------------------------------- */

/* Always binary.  The engine only opens checkpoints, cache blobs and temporary
 * binaries through open(), and the CRT's text default would rewrite CRLF and
 * stop at 0x1A.  The mode argument is only meaningful with _O_CREAT, which is
 * also the only case where POSIX callers pass a third argument. */
int qzen_win_open(const char *path, int flags, ...) {
    int mode = 0;
    if (flags & _O_CREAT) {
        va_list arguments;
        va_start(arguments, flags);
        mode = va_arg(arguments, int);
        va_end(arguments);
    }
    return _open(path, flags | _O_BINARY, mode);
}

/* ------------------------------------------------------------------------
 * Positional I/O
 * ---------------------------------------------------------------------- */

/* ReadFile with an explicit offset is the only way to keep several loader
 * lanes on one descriptor: a shared CRT file pointer would serialise them, and
 * the whole point of the 16-way fan-out is to keep the NVMe queue deep.
 *
 * The handle comes from _open(), i.e. it is not FILE_FLAG_OVERLAPPED, so the
 * call stays synchronous while still honouring the offset.  tests/test_win_pread
 * pins that behaviour down. */
ssize_t qzen_win_pread(int fd, void *buffer, size_t bytes, int64_t offset) {
    if (fd < 0 || !buffer) {
        errno = EINVAL;
        return -1;
    }
    HANDLE handle = (HANDLE)_get_osfhandle(fd);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    if (bytes == 0) return 0;

    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.Offset = (DWORD)((uint64_t)offset & 0xFFFFFFFFu);
    overlapped.OffsetHigh = (DWORD)((uint64_t)offset >> 32);

    /* ReadFile takes a DWORD length; the engine never asks for more than a
     * slice, but clamp rather than truncate silently. */
    DWORD request = bytes > 0x40000000u ? 0x40000000u : (DWORD)bytes;
    DWORD read = 0;
    if (!ReadFile(handle, buffer, request, &read, &overlapped)) {
        DWORD error = GetLastError();
        if (error == ERROR_HANDLE_EOF) return 0;
        errno = (error == ERROR_ACCESS_DENIED) ? EACCES : EIO;
        return -1;
    }
    return (ssize_t)read;
}

ssize_t qzen_win_pwrite(int fd, const void *buffer, size_t bytes, int64_t offset) {
    if (fd < 0 || !buffer) {
        errno = EINVAL;
        return -1;
    }
    HANDLE handle = (HANDLE)_get_osfhandle(fd);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    if (bytes == 0) return 0;

    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.Offset = (DWORD)((uint64_t)offset & 0xFFFFFFFFu);
    overlapped.OffsetHigh = (DWORD)((uint64_t)offset >> 32);

    DWORD request = bytes > 0x40000000u ? 0x40000000u : (DWORD)bytes;
    DWORD written = 0;
    if (!WriteFile(handle, buffer, request, &written, &overlapped)) {
        errno = EIO;
        return -1;
    }
    return (ssize_t)written;
}

/* ------------------------------------------------------------------------
 * Time
 * ---------------------------------------------------------------------- */

unsigned int qzen_win_sleep(unsigned int seconds) {
    Sleep(seconds * 1000u);
    return 0;
}

int qzen_win_usleep(unsigned int microseconds) {
    /* Sleep rounds down to the timer granularity; round up so a 200 us wait
     * does not become a busy spin. */
    DWORD milliseconds = microseconds / 1000u;
    if (microseconds % 1000u) milliseconds++;
    Sleep(milliseconds);
    return 0;
}

int qzen_win_clock_gettime(int clock_id, struct timespec *result) {
    if (!result) {
        errno = EINVAL;
        return -1;
    }
    if (clock_id == CLOCK_REALTIME) {
        FILETIME file_time;
        GetSystemTimePreciseAsFileTime(&file_time);
        ULARGE_INTEGER ticks;
        ticks.LowPart = file_time.dwLowDateTime;
        ticks.HighPart = file_time.dwHighDateTime;
        /* 100 ns ticks since 1601-01-01; convert to the Unix epoch. */
        const uint64_t epoch_offset = 116444736000000000ull;
        uint64_t since_epoch = ticks.QuadPart - epoch_offset;
        result->tv_sec = (time_t)(since_epoch / 10000000ull);
        result->tv_nsec = (long)((since_epoch % 10000000ull) * 100ull);
        return 0;
    }

    static LARGE_INTEGER frequency;
    static int have_frequency = 0;
    if (!have_frequency) {
        QueryPerformanceFrequency(&frequency);
        have_frequency = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    result->tv_sec = (time_t)(counter.QuadPart / frequency.QuadPart);
    result->tv_nsec = (long)(((counter.QuadPart % frequency.QuadPart) * 1000000000ll) /
                             frequency.QuadPart);
    return 0;
}

/* ------------------------------------------------------------------------
 * Machine facts
 * ---------------------------------------------------------------------- */

long qzen_win_sysconf(int name) {
    switch (name) {
        case _SC_NPROCESSORS_ONLN:
            return (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        case _SC_PAGESIZE: {
            SYSTEM_INFO info;
            GetSystemInfo(&info);
            return (long)info.dwPageSize;
        }
        case _SC_PHYS_PAGES: {
            MEMORYSTATUSEX status;
            memset(&status, 0, sizeof(status));
            status.dwLength = sizeof(status);
            if (!GlobalMemoryStatusEx(&status)) return -1;
            SYSTEM_INFO info;
            GetSystemInfo(&info);
            return (long)(status.ullTotalPhys / info.dwPageSize);
        }
        default:
            errno = EINVAL;
            return -1;
    }
}

/* ------------------------------------------------------------------------
 * Paths
 * ---------------------------------------------------------------------- */

/* realpath: absolute, with "." and ".." resolved.  It does not resolve
 * junctions, the same way realpath on Linux does not cross a bind mount.
 *
 * Two things go beyond _fullpath, both because this is what the server's
 * containment check compares prefixes of:
 *
 * - the wide API, so a path outside the ANSI code page survives (a model
 *   directory under a Chinese user name is the ordinary case here);
 * - GetLongPathNameW, because Windows can name one directory two ways
 *   ("C:/Users/ADMINI~1/AppData/Local/Temp" and "C:/Users/Administrator/...")
 *   and %TEMP% on this account is the short one.  Without this step a caller
 *   who typed one spelling is refused for a path the engine spells the other
 *   way, which is the one thing a containment test must not do.
 *
 * GetLongPathNameW only answers for a path that exists, and the target here
 * usually does not (an output file).  So the deepest existing ancestor is
 * canonicalised and the missing tail re-attached: the answer is then the same
 * for a file that exists and one that is about to.  _fullpath made the same
 * promise by never failing, which is why callers in this tree treat a missing
 * leaf as resolvable. */
static int qzen_win_canonical_wide(const wchar_t *full, wchar_t *out) {
    wchar_t work[PATH_MAX];
    if (wcslen(full) >= PATH_MAX) return 0;
    wcscpy(work, full);
    /* The part already cut off the end, always with its leading separator, so
     * it can be concatenated onto whatever ancestor turns out to exist. */
    wchar_t tail[PATH_MAX];
    tail[0] = 0;
    for (;;) {
        /* "C:" means "the current directory on C:", which is not the drive
         * root this is looking for. */
        if (wcslen(work) == 2 && work[1] == L':') {
            work[2] = L'\\';
            work[3] = 0;
        }
        DWORD length = GetLongPathNameW(work, out, PATH_MAX);
        if (length > 0 && length < PATH_MAX) {
            size_t used = wcslen(out);
            const wchar_t *append = tail;
            /* "C:\" + "\dir" would double the separator. */
            if (used > 0 && out[used - 1] == L'\\' && append[0] == L'\\') {
                append++;
            }
            if (used + wcslen(append) >= PATH_MAX) return 0;
            wcscat(out, append);
            return 1;
        }
        wchar_t *slash = wcsrchr(work, L'\\');
        if (!slash) return 0;
        if (slash[1] == 0) {          /* trailing separator: just drop it */
            *slash = 0;
            continue;
        }
        if (wcslen(slash + 1) + 1 + wcslen(tail) >= PATH_MAX) return 0;
        wchar_t grown[PATH_MAX];
        if (swprintf(grown, PATH_MAX, L"\\%s%s", slash + 1, tail) < 0) return 0;
        wcscpy(tail, grown);
        *slash = 0;
        if (work[0] == 0) return 0;
    }
}

char *qzen_win_realpath(const char *path, char *resolved) {
    if (!path) {
        errno = EINVAL;
        return NULL;
    }
    wchar_t wide[PATH_MAX];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, PATH_MAX) <= 0) {
        errno = ENOENT;
        return NULL;
    }
    wchar_t full[PATH_MAX];
    DWORD length = GetFullPathNameW(wide, PATH_MAX, full, NULL);
    if (length == 0 || length >= PATH_MAX) {
        errno = ENOENT;
        return NULL;
    }
    wchar_t canonical[PATH_MAX];
    if (!qzen_win_canonical_wide(full, canonical)) {
        errno = ENOENT;
        return NULL;
    }
    char buffer[PATH_MAX];
    if (WideCharToMultiByte(CP_UTF8, 0, canonical, -1, buffer, sizeof(buffer),
                            NULL, NULL) <= 0) {
        errno = ENOENT;
        return NULL;
    }
    if (resolved) {
        memcpy(resolved, buffer, strlen(buffer) + 1);
        return resolved;
    }
    return _strdup(buffer);
}

/* mkstemp: the templates in the engine look like "/tmp/QwenZImageNative-terminal-XXXXXX".
 * A leading /tmp is rewritten to the real temp directory, then _mktemp_s picks
 * the suffix and _open with _O_EXCL makes the claim atomic. */
/* POSIX writes the generated name back in place, and promises the result is
 * the same length as the template.  The %TEMP% translation breaks that promise
 * (it is longer), so the caller's buffer is assumed to be PATH_MAX -- which is
 * what every caller in this tree uses.  Keeping the assumption in one helper
 * means a future caller with a tight buffer fails here instead of silently
 * truncating the path. */
static void qzen_win_write_back(char *template_name, const char *generated) {
    snprintf(template_name, PATH_MAX, "%s", generated);
}

int qzen_win_mkstemp(char *template_name) {
    if (!template_name) {
        errno = EINVAL;
        return -1;
    }
    char path[PATH_MAX];
    const char *source = template_name;
    if (strncmp(source, "/tmp/", 5) == 0) {
        const char *temp = getenv("TEMP");
        if (!temp || !*temp) temp = getenv("TMP");
        if (!temp || !*temp) temp = ".";
        snprintf(path, sizeof(path), "%s/%s", temp, source + 5);
        source = path;
    }
    char working[PATH_MAX];
    snprintf(working, sizeof(working), "%s", source);

    if (_mktemp_s(working, strlen(working) + 1) != 0) {
        errno = EEXIST;
        return -1;
    }
    int fd = _open(working, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY, 0600);
    if (fd < 0) return -1;
    qzen_win_write_back(template_name, working);
    return fd;
}

/* ------------------------------------------------------------------------
 * nanosleep
 * ---------------------------------------------------------------------- */

/* The one caller is the accept() retry pause (10 ms), so millisecond
 * granularity is the right resolution: Sleep rounds up, which keeps the
 * server from spinning if a sub-millisecond pause is ever requested. */
int qzen_win_nanosleep(const struct timespec *request,
                     struct timespec *remaining) {
    if (remaining) {
        remaining->tv_sec = 0;
        remaining->tv_nsec = 0;
    }
    if (!request || request->tv_sec < 0 || request->tv_nsec < 0) {
        errno = EINVAL;
        return -1;
    }
    int64_t milliseconds = (int64_t)request->tv_sec * 1000 +
                           (request->tv_nsec + 999999) / 1000000;
    if (milliseconds <= 0) return 0;
    Sleep((DWORD)(milliseconds > 0x7fffffff ? 0x7fffffff : milliseconds));
    return 0;
}

/* ------------------------------------------------------------------------
 * mkstemps
 * ---------------------------------------------------------------------- */

/* Same contract as mkstemp but with `suffix_length` characters that stay in
 * place: the terminal PNG writer asks for "QwenZImageNative-XXXXXX.png" with a 4-character
 * suffix.  _mktemp_s only substitutes a trailing XXXXXX, so the suffix is
 * lifted off, the placeholder expanded, and the suffix put back. */
int qzen_win_mkstemps(char *template_name, int suffix_length) {
    if (!template_name || suffix_length < 0) {
        errno = EINVAL;
        return -1;
    }
    char path[PATH_MAX];
    const char *source = template_name;
    if (strncmp(source, "/tmp/", 5) == 0) {
        const char *temp = getenv("TEMP");
        if (!temp || !*temp) temp = getenv("TMP");
        if (!temp || !*temp) temp = ".";
        snprintf(path, sizeof(path), "%s/%s", temp, source + 5);
        source = path;
    }
    size_t total = strlen(source);
    if ((size_t)suffix_length >= total) {
        errno = EINVAL;
        return -1;
    }
    char suffix[PATH_MAX];
    snprintf(suffix, sizeof(suffix), "%s", source + total - suffix_length);
    char working[PATH_MAX];
    snprintf(working, sizeof(working), "%.*s",
             (int)(total - (size_t)suffix_length), source);
    if (_mktemp_s(working, strlen(working) + 1) != 0) {
        errno = EEXIST;
        return -1;
    }
    char final_path[PATH_MAX];
    snprintf(final_path, sizeof(final_path), "%s%s", working, suffix);
    int fd = _open(final_path, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY, 0600);
    if (fd < 0) return -1;
    qzen_win_write_back(template_name, final_path);
    return fd;
}

/* ------------------------------------------------------------------------
 * Randomness
 * ---------------------------------------------------------------------- */

/* arc4random_buf's only caller is the CLI's seed generator.  rand_s is the
 * UCRT's CSPRNG (it wraps RtlGenRandom), so a loop over it is equivalent for
 * a 64-bit seed; a failure leaves the buffer zeroed and the caller still gets
 * a deterministic-but-valid seed rather than an uninitialised one. */
int qzen_win_random_bytes(void *buffer, size_t bytes) {
    unsigned char *cursor = (unsigned char *)buffer;
    size_t remaining = bytes;
    while (remaining) {
        unsigned int value = 0;
        if (rand_s(&value) != 0) {
            memset(cursor, 0, remaining);
            errno = EIO;
            return -1;
        }
        size_t chunk = remaining < sizeof(value) ? remaining : sizeof(value);
        memcpy(cursor, &value, chunk);
        cursor += chunk;
        remaining -= chunk;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * mkdtemp
 * ---------------------------------------------------------------------- */

/* The CLI puts its session directory under /tmp, which qzen_win_mkstemp already
 * translates to %TEMP%; mkdtemp needs the same translation plus a directory
 * create.  _mktemp_s only substitutes the XXXXXX, so the retry loop that POSIX
 * implementations get from O_EXCL has to be explicit here.
 *
 * Returns the template pointer on success and NULL on failure, like POSIX.  An
 * int return (0 on success) was silently inverted by every `if (!mkdtemp(...))`
 * caller -- qzen_cli.c builds its session directory that way, so on Windows the
 * interactive CLI always reported "cannot create session directory" and quit. */
char *qzen_win_mkdtemp(char *template_name) {
    if (!template_name) {
        errno = EINVAL;
        return NULL;
    }
    char path[PATH_MAX];
    const char *source = template_name;
    if (strncmp(source, "/tmp/", 5) == 0) {
        const char *temp = getenv("TEMP");
        if (!temp || !*temp) temp = getenv("TMP");
        if (!temp || !*temp) temp = ".";
        snprintf(path, sizeof(path), "%s/%s", temp, source + 5);
        source = path;
    }
    char working[PATH_MAX];
    for (int attempt = 0; attempt < 100; attempt++) {
        snprintf(working, sizeof(working), "%s", source);
        if (_mktemp_s(working, strlen(working) + 1) != 0) {
            errno = EEXIST;
            return NULL;
        }
        if (_mkdir(working) == 0) {
            qzen_win_write_back(template_name, working);
            return template_name;
        }
        if (errno != EEXIST) return NULL;
    }
    errno = EEXIST;
    return NULL;
}

/* ------------------------------------------------------------------------
 * Shell hand-off
 * ---------------------------------------------------------------------- */

/* `QwenZImageNative --open` (and the CLI's post-generate step) launches the platform viewer
 * on the finished video.  POSIX has xdg-open/open; Windows has no such
 * program, only the shell's file association, reached through cmd's builtin
 * `start`.  The empty title argument is required: without it `start` would
 * treat the quoted path as the window title when it contains a space. */
int qzen_win_open_path(const char *path) {
    if (!path || !*path) {
        errno = EINVAL;
        return -1;
    }
    wchar_t wide[PATH_MAX];
    int length = MultiByteToWideChar(CP_UTF8, 0, path, -1, wide,
                                     (int)(sizeof(wide) / sizeof(wide[0])));
    if (length <= 0) {
        errno = EINVAL;
        return -1;
    }
    intptr_t child = _wspawnlp(_P_NOWAIT, L"cmd.exe", L"cmd.exe", L"/c",
                               L"start", L"\"\"", wide, NULL);
    if (child == -1) return -1;
    return 0;
}

/* ------------------------------------------------------------------------
 * memmem
 * ---------------------------------------------------------------------- */

void *qzen_win_memmem(const void *haystack, size_t haystack_length,
                    const void *needle, size_t needle_length) {
    if (!haystack || !needle || needle_length == 0) return NULL;
    if (haystack_length < needle_length) return NULL;
    const unsigned char *start = (const unsigned char *)haystack;
    const unsigned char *last = start + haystack_length - needle_length;
    for (const unsigned char *cursor = start; cursor <= last; cursor++) {
        if (cursor[0] == ((const unsigned char *)needle)[0] &&
            memcmp(cursor, needle, needle_length) == 0) {
            return (void *)cursor;
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------------
 * Signals
 * ---------------------------------------------------------------------- */

/* MSVC's <signal.h> has SIGINT and SIGTERM but no sigaction, and Windows has
 * no asynchronous signal delivery to the main thread.  Console control events
 * are the closest equivalent: Ctrl+C, Ctrl+Break and console close arrive on a
 * separate thread, and the server's handler only closes the listen socket and
 * flips a flag, so the weaker delivery guarantee is fine. */
static void (*qzen_signal_handlers[16])(int);
static int qzen_console_handler_installed = 0;

static BOOL WINAPI qzen_win_console_handler(DWORD event) {
    int signal_number = 0;
    switch (event) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
            signal_number = SIGINT;
            break;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            signal_number = SIGTERM;
            break;
        default:
            return FALSE;
    }
    void (*handler)(int) = qzen_signal_handlers[signal_number & 15];
    if (handler && handler != SIG_IGN && handler != SIG_DFL) {
        handler(signal_number);
        return TRUE;
    }
    return FALSE;
}

int sigaction(int signal_number, const struct sigaction *action,
              struct sigaction *old_action) {
    if (signal_number < 0 || signal_number > 15) {
        errno = EINVAL;
        return -1;
    }
    if (old_action) {
        old_action->sa_handler = qzen_signal_handlers[signal_number];
        memset(&old_action->sa_mask, 0, sizeof(old_action->sa_mask));
        old_action->sa_flags = 0;
    }
    if (action) {
        qzen_signal_handlers[signal_number] = action->sa_handler;
        if (!qzen_console_handler_installed) {
            SetConsoleCtrlHandler(qzen_win_console_handler, TRUE);
            qzen_console_handler_installed = 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * Terminal stubs (linenoise.c)
 * ---------------------------------------------------------------------- */

int tcgetattr(int fd, struct termios *termios_state) {
    (void)fd;
    (void)termios_state;
    errno = ENOTTY;
    return -1;
}

int tcsetattr(int fd, int actions, const struct termios *termios_state) {
    (void)fd;
    (void)actions;
    (void)termios_state;
    errno = ENOTTY;
    return -1;
}

int ioctl(int fd, unsigned long request, ...) {
    (void)fd;
    (void)request;
    errno = ENOTTY;
    return -1;
}
