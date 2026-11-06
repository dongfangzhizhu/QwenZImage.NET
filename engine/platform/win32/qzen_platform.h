/* Win32 compatibility layer for the POSIX surface the engine is written to.
 *
 * The engine targets POSIX: pthreads, sockets, pread, dirent, posix_spawn.  A
 * Windows build would otherwise need #ifdef in every one of the 82 source
 * files, so instead CMake force-includes this header (/FI) into every
 * translation unit and links platform/win32/*.c, which implements the missing
 * calls on top of Win32.  The Linux build never sees this file.
 *
 * Rules for this header:
 *   - it runs before every other #include in the TU, so it must not pull in
 *     <windows.h>: that defines min/max/interface/ERROR and breaks CUDA.  Only
 *     the .c files behind it include <windows.h>.
 *   - it must stay C and C++ clean: nvcc compiles qzen_gpu.cu with it.
 *   - it must not change the meaning of anything the Linux build relies on.
 */
#ifndef QZEN_PLATFORM_H
#define QZEN_PLATFORM_H

#if defined(_WIN32)

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif
/* Enables rand_s() in <stdlib.h>; qzen_win_random_bytes() backs the
 * arc4random_buf the CLI seeds its RNG with. */
#ifndef _CRT_RAND_S
#define _CRT_RAND_S 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#include <process.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Types the engine expects from <sys/types.h>
 * ---------------------------------------------------------------------- */

/* MSVC has no ssize_t.  intptr_t is the right width: the engine uses it for
 * read()/send() results and for buffer deltas. */
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif

/* off_t is `long` in UCRT, i.e. 32 bits, and the safetensors reader seeks to
 * byte offsets inside 7 GB checkpoints.  Redefining the name here (after the
 * CRT headers are already processed) makes every use in the engine 64-bit.
 * The static assert below is the tripwire: if a future CRT header reintroduces
 * a 32-bit off_t, the build fails instead of reading the wrong bytes. */
#define off_t int64_t

/* posix_spawn cannot hand back a pid on Windows: CreateProcess returns a
 * HANDLE, and the engine only ever passes that value back to waitpid.  pid_t
 * is therefore pointer-width here, wide enough to carry the handle.  POSIX
 * says int; nothing in the engine formats a pid, and the one place that
 * initialises one to -1 still works. */
#ifndef _PID_T_DEFINED
#define _PID_T_DEFINED
typedef intptr_t pid_t;
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* UCRT has no symbolic descriptor names; the engine uses them for the three
 * standard streams (linenoise writes to STDOUT_FILENO, qzen_ffmpeg dup2s
 * STDOUT_FILENO onto a pipe).  The values match the CRT's own numbering. */
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

/* write() takes an unsigned count on UCRT; the engine clamps its chunks with
 * SSIZE_MAX the way POSIX requires.  PTRDIFF_MAX is the same bound. */
#ifndef SSIZE_MAX
#define SSIZE_MAX INTPTR_MAX
#endif

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISLNK
#define S_ISLNK(mode) 0
#endif

/* ------------------------------------------------------------------------
 * POSIX names UCRT does not export
 * ---------------------------------------------------------------------- */

/* Same semantics, different spelling. */
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define strtok_r strtok_s
#define isatty _isatty
#define fseeko _fseeki64
#define ftello _ftelli64

/* 32-bit struct stat would truncate the size of a 7 GB checkpoint, so the name
 * is redirected to the 64-bit entry point; `struct stat` becomes
 * `struct _stat64`, whose st_size is __int64. */
#define stat _stat64
#define fstat _fstat64

/* Implemented in qzen_win_posix.c. */
int qzen_win_mkdir(const char *path);
int qzen_win_setenv(const char *name, const char *value, int overwrite);
int qzen_win_unsetenv(const char *name);
ssize_t qzen_win_pread(int fd, void *buffer, size_t bytes, int64_t offset);
ssize_t qzen_win_pwrite(int fd, const void *buffer, size_t bytes, int64_t offset);
unsigned int qzen_win_sleep(unsigned int seconds);
int qzen_win_usleep(unsigned int microseconds);
long qzen_win_sysconf(int name);
int qzen_win_mkstemp(char *template_name);
char *qzen_win_realpath(const char *path, char *resolved);
int qzen_win_clock_gettime(int clock_id, struct timespec *result);
void *qzen_win_memmem(const void *haystack, size_t haystack_length,
                    const void *needle, size_t needle_length);
int qzen_win_getpid(void);
const char *qzen_win_getenv(const char *name);
/* Replaces posix_spawnp("xdg-open"/"open", path): hands the file to the shell's
 * default handler.  Returns 0 when the shell accepted it. */
int qzen_win_open_path(const char *path);
int qzen_win_random_bytes(void *buffer, size_t bytes);
/* POSIX mkdtemp returns the template pointer on success, NULL on failure.  The
 * engine's callers test it as a pointer, so this shim must too. */
char *qzen_win_mkdtemp(char *template_name);
int qzen_win_mkstemps(char *template_name, int suffix_length);
int qzen_win_nanosleep(const struct timespec *request, struct timespec *remaining);
/* close() serves both CRT descriptors and sockets in the engine; the shim
 * decides which namespace the value belongs to (see qzen_win_socket.c). */
int qzen_win_close(intptr_t fd);
/* open() has to force _O_BINARY: the CRT defaults to text mode, which turns
 * CRLF into LF on read and stops at the first 0x1A, and every caller in the
 * engine is reading or writing a checkpoint or a cache blob. */
int qzen_win_open(const char *path, int flags, ...);
/* Switches the console to UTF-8 so the engine's UTF-8 literals print. */
void qzen_win_console_init(void);

#define mkdir(path, ...) qzen_win_mkdir((path))
#define setenv(name, value, overwrite) qzen_win_setenv((name), (value), (overwrite))
#define unsetenv(name) qzen_win_unsetenv((name))
#define pread qzen_win_pread
#define pwrite qzen_win_pwrite
#define sleep qzen_win_sleep
#define usleep qzen_win_usleep
#define sysconf qzen_win_sysconf
#define mkstemp qzen_win_mkstemp
#define mkdtemp qzen_win_mkdtemp
#define mkstemps(template_name, suffix_length) \
    qzen_win_mkstemps((template_name), (suffix_length))
#define nanosleep(request, remaining) qzen_win_nanosleep((request), (remaining))
#define close(fd) qzen_win_close((intptr_t)(fd))
#define open(...) qzen_win_open(__VA_ARGS__)
#define arc4random_buf(buffer, bytes) qzen_win_random_bytes((buffer), (bytes))
#define realpath qzen_win_realpath
#define clock_gettime qzen_win_clock_gettime
#define memmem qzen_win_memmem
#define getpid qzen_win_getpid
#define getenv qzen_win_getenv

/* _SC_* selectors for qzen_win_sysconf.  Values are arbitrary but stable. */
#define _SC_PAGESIZE 1
#define _SC_PHYS_PAGES 2
#define _SC_NPROCESSORS_ONLN 3

/* ftruncate takes a 64-bit length; _chsize_s is the only 64-bit spelling. */
#define ftruncate(fd, length) _chsize_s((fd), (int64_t)(length))

/* _pipe is the UCRT pipe; the engine only ever wants binary mode. */
#define pipe(fds) _pipe((fds), 65536, _O_BINARY)

/* ------------------------------------------------------------------------
 * Compiler extensions
 * ---------------------------------------------------------------------- */

/* Two uses: visibility (meaningless here) and printf format checking (MSVC
 * cannot check gnu_printf anyway). */
#define __attribute__(spec)

/* qzen_gpu.cu keeps a per-thread cached file descriptor. */
#define __thread __declspec(thread)

/* ------------------------------------------------------------------------
 * GCC atomic builtins: qzen_serve.c publishes progress through a seqlock and
 * qzen_text_encoder.c announces itself once, both with the __atomic_* family.
 * MSVC has no such builtins, so they are mapped onto the _Interlocked
 * intrinsics.  All of the engine's counters are 32-bit ints, which is exactly
 * what the long-typed intrinsics operate on; the memory-order argument is
 * dropped because x86-64 already gives every one of these a full fence.
 * ---------------------------------------------------------------------- */
#if defined(_MSC_VER)
#include <intrin.h>

#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5

#define __sync_fetch_and_add(pointer, value) \
    ((long)_InterlockedExchangeAdd((volatile long *)(pointer), (long)(value)))
#define __atomic_exchange_n(pointer, value, order) \
    ((long)_InterlockedExchange((volatile long *)(pointer), (long)(value)))
/* _InterlockedExchangeAdd returns the *old* value; __atomic_add_fetch promises
 * the new one, hence the extra add. */
#define __atomic_add_fetch(pointer, value, order) \
    ((long)_InterlockedExchangeAdd((volatile long *)(pointer), (long)(value)) + \
     (long)(value))
#define __atomic_load_n(pointer, order) (*(volatile long *)(pointer))
#define __atomic_store_n(pointer, value, order) \
    ((void)_InterlockedExchange((volatile long *)(pointer), (long)(value)))
#endif /* _MSC_VER */

/* ------------------------------------------------------------------------
 * sigaction: the engine installs a SIGINT/SIGTERM handler to cancel the
 * in-flight generation.  MSVC only has signal(), so the shim below maps the
 * console control events (Ctrl+C, Ctrl+Break, close) onto the same handler.
 * ---------------------------------------------------------------------- */

#ifndef QZEN_WIN_SIGSET_T
#define QZEN_WIN_SIGSET_T
typedef struct {
    unsigned long bits[2];
} sigset_t;
#endif

struct sigaction {
    void (*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
};

/* Declared under its POSIX name rather than #defined to a shim: the engine
 * writes `struct sigaction ignore;`, and a function-like rename would rewrite
 * the struct tag too and leave the type undefined.  Nothing in the CRT or the
 * SDK declares sigaction, so the name is free. */
int sigaction(int signal_number, const struct sigaction *action,
              struct sigaction *old_action);

/* Windows has no SIGPIPE: a write to a closed pipe returns EPIPE from write()
 * instead of raising.  The engine still brackets its FFmpeg pipes with
 * sigaction(SIGPIPE, SIG_IGN) to survive a viewer that exits early, so the
 * number has to exist and be accepted; the shim records it and never
 * delivers it.  13 is unused by UCRT (SIGBREAK is 21). */
#ifndef SIGPIPE
#define SIGPIPE 13
#endif

/* sigset_t is inert here: the console handler runs synchronously, so there is
 * no window in which a mask would matter. */
#define sigemptyset(set) ((void)(set))
#define sigfillset(set) ((void)(set))
#define sigaddset(set, signal_number) ((void)(set), (void)(signal_number), 0)
#define sigdelset(set, signal_number) ((void)(set), (void)(signal_number), 0)

/* ------------------------------------------------------------------------
 * Sanity checks: these catch a CRT that disagrees with the assumptions above
 * at compile time rather than at the first 4 GB tensor.
 * ---------------------------------------------------------------------- */
#ifdef __cplusplus
static_assert(sizeof(off_t) == 8, "off_t must be 64-bit on Win32");
static_assert(sizeof(ssize_t) == 8, "ssize_t must be pointer width");
#else
_Static_assert(sizeof(off_t) == 8, "off_t must be 64-bit on Win32");
_Static_assert(sizeof(ssize_t) == 8, "ssize_t must be pointer width");
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _WIN32 */
#endif /* QZEN_PLATFORM_H */
