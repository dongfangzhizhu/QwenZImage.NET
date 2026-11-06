/* posix_spawn / waitpid over CreateProcessW.
 *
 * The engine spawns FFmpeg and FFprobe as filters: it hands the child one end
 * of a _pipe() and reads or writes the other end, then waits.  That is the
 * only pattern this file has to support, which is why the file-action list is
 * a fixed array of two dup2s plus two closes instead of a general builder.
 *
 * Two details carry the whole design:
 *
 *   - the child's std handle has to be the *same numeric value* as the
 *     parent's, because handle inheritance copies table entries verbatim.
 *     That is why STARTUPINFO is filled from _get_osfhandle() rather than
 *     from anything the child would have to resolve.
 *
 *   - a handle that is not marked inheritable is not copied into the child at
 *     all, which is exactly what posix_spawn's "close in child" means.  So the
 *     close actions are implemented by clearing HANDLE_FLAG_INHERIT and
 *     restoring it after CreateProcess returns.  Getting this wrong is what
 *     makes a filter hang: the child would keep the write end of its own
 *     stdin pipe open and never see EOF.
 */
#include "qzen_platform.h"

#include <windows.h>
#include <spawn.h>
#include <sys/wait.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * File actions
 * ---------------------------------------------------------------------- */

int posix_spawn_file_actions_init(posix_spawn_file_actions_t *actions) {
    if (!actions) return EINVAL;
    memset(actions, 0, sizeof(*actions));
    return 0;
}

int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t *actions) {
    if (!actions) return EINVAL;
    memset(actions, 0, sizeof(*actions));
    return 0;
}

int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t *actions,
                                     int fd, int new_fd) {
    if (!actions || fd < 0 || new_fd < 0 || new_fd > 2) return EINVAL;
    if (actions->count >= QZEN_WIN_SPAWN_ACTIONS_MAX) return EINVAL;
    qzen_win_spawn_dup *dup = &actions->dups[actions->count++];
    dup->source_fd = fd;
    dup->target_fd = new_fd;
    dup->used = 1;
    return 0;
}

int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t *actions,
                                      int fd) {
    if (!actions || fd < 0) return EINVAL;
    if (actions->close_count >= (int)(sizeof(actions->close_fds) /
                                      sizeof(actions->close_fds[0]))) {
        return EINVAL;
    }
    actions->close_fds[actions->close_count++] = fd;
    return 0;
}

/* ------------------------------------------------------------------------
 * Command line
 * ---------------------------------------------------------------------- */

/* UTF-8 -> UTF-16.  Returns a malloc'd buffer, or NULL when the input is not
 * valid UTF-8 (or empty). */
static wchar_t *wide_from_utf8(const char *text) {
    if (!text) return NULL;
    int length = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
    if (length <= 0) return NULL;
    wchar_t *wide = (wchar_t *)malloc((size_t)length * sizeof(wchar_t));
    if (!wide) return NULL;
    if (MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, length) <= 0) {
        free(wide);
        return NULL;
    }
    return wide;
}

/* Appends one argument using the quoting rules CommandLineToArgvW (and hence
 * every CRT program) parses back:  a run of N backslashes followed by a quote
 * becomes 2N backslashes plus an escaped quote, and a run at the end of the
 * argument doubles.  Without this an FFmpeg path with a space silently splits
 * into two arguments. */
static int append_argument(wchar_t *buffer, size_t capacity, size_t *length,
                           const wchar_t *argument) {
    size_t needed = 0;
    int quote = argument[0] == L'\0';
    for (const wchar_t *scan = argument; *scan; scan++) {
        if (*scan == L' ' || *scan == L'\t' || *scan == L'"') quote = 1;
        needed++;
    }
    /* Worst case every character is a backslash before a quote. */
    if (*length + needed * 2 + 3 >= capacity) return 0;
    if (*length) buffer[(*length)++] = L' ';
    if (quote) buffer[(*length)++] = L'"';
    for (const wchar_t *scan = argument; *scan; scan++) {
        if (*scan == L'\\') {
            size_t run = 0;
            while (scan[run] == L'\\') run++;
            if (scan[run] == L'"' || scan[run] == L'\0') {
                for (size_t index = 0; index < run * 2; index++) {
                    buffer[(*length)++] = L'\\';
                }
            } else {
                for (size_t index = 0; index < run; index++) {
                    buffer[(*length)++] = L'\\';
                }
            }
            scan += run - 1;
        } else if (*scan == L'"') {
            buffer[(*length)++] = L'\\';
            buffer[(*length)++] = L'"';
        } else {
            buffer[(*length)++] = *scan;
        }
    }
    if (quote) buffer[(*length)++] = L'"';
    buffer[*length] = L'\0';
    return 1;
}

/* Builds the environment block CreateProcessW wants: "K=V\0K=V\0\0", in
 * UTF-16.  NULL means "inherit the parent's", which is the common case. */
static wchar_t *build_environment(char *const envp[]) {
    if (!envp) return NULL;
    size_t total = 0;
    for (char *const *entry = envp; *entry; entry++) {
        wchar_t *wide = wide_from_utf8(*entry);
        if (!wide) continue;
        total += wcslen(wide) + 1;
        free(wide);
    }
    if (!total) return NULL;
    wchar_t *block = (wchar_t *)calloc(total + 1, sizeof(wchar_t));
    if (!block) return NULL;
    wchar_t *cursor = block;
    for (char *const *entry = envp; *entry; entry++) {
        wchar_t *wide = wide_from_utf8(*entry);
        if (!wide) continue;
        size_t length = wcslen(wide);
        memcpy(cursor, wide, length * sizeof(wchar_t));
        cursor += length;
        *cursor++ = L'\0';
        free(wide);
    }
    *cursor = L'\0';
    return block;
}

/* Does the caller want the environment it already has?
 *
 * posix_spawn's contract is "the child gets envp".  Every call site in the
 * engine passes `environ`, for which the Win32 equivalent is lpEnvironment =
 * NULL (inherit) -- and that is not a shortcut, it is the only form that works
 * everywhere.  It matters because this host's CreateProcessW rejects a UTF-16
 * block outright: with lpEnvironment = L"A=B\0\0" it returns
 * ERROR_INVALID_PARAMETER, while the same bytes handed to CreateProcessA (i.e.
 * an ANSI block) are accepted, and CreateProcessW accepts that ANSI block too.
 * Measured here with cmd.exe and ffmpeg.exe as the child, in every combination
 * of CREATE_NO_WINDOW and bInheritHandles.
 *
 * So: inherit when the caller passed the inherited environment, and build a
 * real UTF-16 block (the documented form) only for a custom one.  A custom
 * block that this API refuses fails loudly instead of being re-encoded by
 * guess -- nothing in the engine passes one, and silently handing a child a
 * different environment than asked for would be worse than an error. */
static int wants_inherited_environment(char *const envp[]) {
    extern char **environ;
    return !envp || envp == environ;
}

/* ------------------------------------------------------------------------
 * Spawn
 * ---------------------------------------------------------------------- */

/* Programs without a directory component are looked up the way execvp does:
 * on PATH, with the executable extension appended. */
static wchar_t *resolve_program(const wchar_t *program) {
    if (!program) return NULL;
    if (wcschr(program, L'/') || wcschr(program, L'\\')) {
        size_t length = wcslen(program);
        wchar_t *copy = (wchar_t *)malloc((length + 1) * sizeof(wchar_t));
        if (copy) memcpy(copy, program, (length + 1) * sizeof(wchar_t));
        return copy;
    }
    wchar_t located[MAX_PATH * 2];
    DWORD found = SearchPathW(NULL, program, L".exe",
                              (DWORD)(sizeof(located) / sizeof(located[0])),
                              located, NULL);
    if (found == 0 || found >= sizeof(located) / sizeof(located[0])) return NULL;
    size_t length = wcslen(located);
    wchar_t *copy = (wchar_t *)malloc((length + 1) * sizeof(wchar_t));
    if (copy) memcpy(copy, located, (length + 1) * sizeof(wchar_t));
    return copy;
}

/* Tracks the inherit flag of one handle so it can be put back afterwards. */
typedef struct {
    HANDLE handle;
    DWORD flags;
} inherit_state;

int posix_spawnp(pid_t *pid_out, const char *program,
                 const posix_spawn_file_actions_t *actions,
                 const void *attrp, char *const argv[], char *const envp[]) {
    (void)attrp;
    if (!pid_out || !program || !argv || !argv[0]) return EINVAL;
    *pid_out = -1;

    wchar_t *wide_program = wide_from_utf8(program);
    wchar_t *executable = resolve_program(wide_program);
    free(wide_program);
    if (!executable) {
        /* The engine turns a non-zero return into strerror(code). */
        return ENOENT;
    }

    /* 32k characters is the CreateProcessW limit; refusing earlier keeps the
     * failure inside this function instead of as a truncated command line. */
    wchar_t command_line[32768];
    size_t command_length = 0;
    command_line[0] = L'\0';
    int ok = append_argument(command_line,
                             sizeof(command_line) / sizeof(command_line[0]),
                             &command_length, executable);
    for (char *const *entry = argv + 1; ok && *entry; entry++) {
        wchar_t *wide = wide_from_utf8(*entry);
        if (!wide) {
            ok = 0;
            break;
        }
        ok = append_argument(command_line,
                             sizeof(command_line) / sizeof(command_line[0]),
                             &command_length, wide);
        free(wide);
    }
    free(executable);
    if (!ok) return EINVAL;

    STARTUPINFOW startup;
    PROCESS_INFORMATION child;
    memset(&startup, 0, sizeof(startup));
    memset(&child, 0, sizeof(child));
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;

    /* Default every standard stream to ours; a dup2 action then overrides the
     * ones the caller actually redirects. */
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    inherit_state restore[QZEN_WIN_SPAWN_ACTIONS_MAX + 2];
    int restore_count = 0;
    int failure = 0;
    /* Handles that the dup2 actions handed to the child.  A later "close this
     * fd" action must not touch them: on POSIX dup2 makes a *copy* at the new
     * slot, so closing the source leaves the child's copy valid, but here the
     * STARTUPINFO field and the source descriptor are the same HANDLE, and
     * clearing its inherit flag would take the child's stdin away.  ffmpeg
     * then fails with "Bad file descriptor" while the parent still holds a
     * perfectly good write end. */
    HANDLE kept[8];
    int kept_count = 0;

    for (int index = 0; index < (actions ? actions->count : 0); index++) {
        const qzen_win_spawn_dup *dup = &actions->dups[index];
        intptr_t raw = _get_osfhandle(dup->source_fd);
        if (raw == -1) {
            failure = EBADF;
            break;
        }
        HANDLE handle = (HANDLE)raw;
        if (kept_count < (int)(sizeof(kept) / sizeof(kept[0])))
            kept[kept_count++] = handle;
        if (restore_count < (int)(sizeof(restore) / sizeof(restore[0]))) {
            DWORD flags = 0;
            if (GetHandleInformation(handle, &flags)) {
                restore[restore_count].handle = handle;
                restore[restore_count].flags = flags;
                restore_count++;
            }
        }
        SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        if (dup->target_fd == 0) startup.hStdInput = handle;
        else if (dup->target_fd == 1) startup.hStdOutput = handle;
        else startup.hStdError = handle;
    }

    /* The three defaults above are only inherited if they are inheritable;
     * a console handle already is, a redirected one may not be. */
    if (!failure) {
        HANDLE standards[3] = {startup.hStdInput, startup.hStdOutput,
                               startup.hStdError};
        for (int index = 0; index < 3; index++) {
            if (!standards[index] || standards[index] == INVALID_HANDLE_VALUE) {
                continue;
            }
            DWORD flags = 0;
            if (!GetHandleInformation(standards[index], &flags)) continue;
            if (flags & HANDLE_FLAG_INHERIT) continue;
            if (restore_count < (int)(sizeof(restore) / sizeof(restore[0]))) {
                restore[restore_count].handle = standards[index];
                restore[restore_count].flags = flags;
                restore_count++;
            }
            SetHandleInformation(standards[index], HANDLE_FLAG_INHERIT,
                                 HANDLE_FLAG_INHERIT);
        }
    }

    for (int index = 0; !failure && actions &&
                        index < actions->close_count; index++) {
        intptr_t raw = _get_osfhandle(actions->close_fds[index]);
        if (raw == -1) continue;
        HANDLE handle = (HANDLE)raw;
        int kept_by_child = 0;
        for (int other = 0; other < kept_count; other++)
            if (kept[other] == handle) kept_by_child = 1;
        if (kept_by_child) continue;
        DWORD flags = 0;
        if (!GetHandleInformation(handle, &flags)) continue;
        if (restore_count < (int)(sizeof(restore) / sizeof(restore[0]))) {
            restore[restore_count].handle = handle;
            restore[restore_count].flags = flags;
            restore_count++;
        }
        /* Not inherited == closed in the child. */
        SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
    }

    wchar_t *environment = wants_inherited_environment(envp)
                               ? NULL
                               : build_environment(envp);
    BOOL created = FALSE;
    if (!failure) {
        /* CREATE_NO_WINDOW: the engine is started from a terminal *and* from
         * the WebUI's background process; without it every FFmpeg invocation
         * would flash a console window on the desktop.  The filter talks over
         * pipes, so it never needs a console. */
        created = CreateProcessW(NULL, command_line, NULL, NULL, TRUE,
                                 CREATE_NO_WINDOW, environment, NULL, &startup,
                                 &child);
    }
    free(environment);

    for (int index = restore_count - 1; index >= 0; index--) {
        SetHandleInformation(restore[index].handle, HANDLE_FLAG_INHERIT,
                             restore[index].flags);
    }

    if (failure) return failure;
    if (!created) {
        DWORD last = GetLastError();
        /* The caller only sees ENOENT/EACCES/EIO, and "EIO" says nothing about
         * which of CreateProcessW's forty failures happened. One line here is
         * the difference between a five-minute fix and an afternoon. */
        fprintf(stderr, "QwenZImageNative: CreateProcessW failed (%lu) for %ls\n",
                (unsigned long)last, command_line);
        if (last == ERROR_FILE_NOT_FOUND || last == ERROR_PATH_NOT_FOUND) {
            return ENOENT;
        }
        if (last == ERROR_ACCESS_DENIED) return EACCES;
        return EIO;
    }

    CloseHandle(child.hThread);
    /* The handle stays open on purpose: waitpid owns it from here. */
    *pid_out = (pid_t)(intptr_t)child.hProcess;
    return 0;
}

/* ------------------------------------------------------------------------
 * Wait
 * ---------------------------------------------------------------------- */

pid_t waitpid(pid_t pid, int *status, int options) {
    (void)options;
    if (pid <= 0) {
        errno = ECHILD;
        return -1;
    }
    HANDLE process = (HANDLE)(intptr_t)pid;
    DWORD waited = WaitForSingleObject(process, INFINITE);
    if (waited != WAIT_OBJECT_0) {
        errno = ECHILD;
        return -1;
    }
    DWORD code = 0;
    if (!GetExitCodeProcess(process, &code)) {
        CloseHandle(process);
        errno = ECHILD;
        return -1;
    }
    CloseHandle(process);
    if (status) *status = QZEN_WIN_EXIT_STATUS(code);
    return pid;
}
