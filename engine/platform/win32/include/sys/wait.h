/* <sys/wait.h>: waitpid and the W* status macros, over WaitForSingleObject.
 *
 * The pid the spawn shim reports is the process HANDLE, which is why pid_t is
 * pointer-width on this platform (see qzen_platform.h).  Keeping the handle
 * rather than a process id means waitpid needs no lookup and cannot lose the
 * child to pid reuse. */
#ifndef QZEN_WIN_SYS_WAIT_H
#define QZEN_WIN_SYS_WAIT_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors the POSIX status encoding so WIFEXITED/WEXITSTATUS keep working. */
#define QZEN_WIN_EXIT_STATUS(code) (((code) & 0xff) << 8)

#define WIFEXITED(status) (((status) & 0x7f) == 0)
#define WEXITSTATUS(status) (((status) & 0xff00) >> 8)
#define WIFSIGNALED(status) 0
#define WTERMSIG(status) 0

/* `options` is accepted and ignored: the engine only ever passes 0, and the
 * spawn shim closes the handle once the status has been reported, so WNOHANG
 * would have nothing left to poll. */
pid_t waitpid(pid_t pid, int *status, int options);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_SYS_WAIT_H */
