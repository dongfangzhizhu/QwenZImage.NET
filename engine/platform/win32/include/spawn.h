/* <spawn.h>: the posix_spawn subset qzen_ffmpeg.c and qzen_terminal.c use.
 *
 * Implemented over CreateProcess in qzen_win_spawn.c.  The file actions only
 * need to rewire stdin/stdout onto a pipe, which maps onto STARTUPINFO's
 * hStdInput/hStdOutput. */
#ifndef QZEN_WIN_SPAWN_H
#define QZEN_WIN_SPAWN_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QZEN_WIN_SPAWN_ACTIONS_MAX 4

typedef struct {
    int source_fd;   /* our end of the pipe */
    int target_fd;   /* the child's 0 or 1 */
    int used;
} qzen_win_spawn_dup;

typedef struct {
    qzen_win_spawn_dup dups[QZEN_WIN_SPAWN_ACTIONS_MAX];
    int count;
    int close_fds[2];
    int close_count;
} posix_spawn_file_actions_t;

int posix_spawn_file_actions_init(posix_spawn_file_actions_t *actions);
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t *actions);
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t *actions,
                                     int fd, int new_fd);
int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t *actions,
                                      int fd);
int posix_spawnp(pid_t *pid_out, const char *program,
                 const posix_spawn_file_actions_t *actions,
                 const void *attrp, char *const argv[], char *const envp[]);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_SPAWN_H */
