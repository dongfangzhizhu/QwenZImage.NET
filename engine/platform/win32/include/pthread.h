/* The pthread subset the engine uses, implemented over Win32 in
 * qzen_win_thread.c.
 *
 * The engine touches 17 symbols, all of them mutex / condition variable /
 * thread create-join, so a ~200 line shim is cheaper than pulling in
 * pthreads-win32 and giving up the single-binary property.  The public types
 * are opaque storage sized for the Win32 primitive behind them, which keeps
 * <windows.h> out of every translation unit (it defines min/max and breaks
 * CUDA).
 */
#ifndef QZEN_WIN_PTHREAD_H
#define QZEN_WIN_PTHREAD_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* _beginthreadex returns a uintptr_t; the engine only ever stores and joins. */
typedef void *pthread_t;

/* CRITICAL_SECTION is 40 bytes on x64 (5 pointers).  opaque[5] is this shim's
 * "already initialised" flag, which is what makes a statically zeroed mutex
 * (PTHREAD_MUTEX_INITIALIZER) work without a constructor. */
typedef struct {
    void *opaque[6];
} pthread_mutex_t;

/* CONDITION_VARIABLE is one pointer and its initialised state is all-zero, so
 * no flag is needed. */
typedef struct {
    void *opaque[2];
} pthread_cond_t;

#define PTHREAD_CREATE_JOINABLE 0
#define PTHREAD_CREATE_DETACHED 1

typedef struct {
    int detach_state;
} pthread_attr_t;

#define PTHREAD_MUTEX_INITIALIZER {{0}}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start)(void *), void *argument);
int pthread_join(pthread_t thread, void **result);
int pthread_detach(pthread_t thread);
void pthread_exit(void *result);
pthread_t pthread_self(void);

int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_trylock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);

int pthread_cond_init(pthread_cond_t *cond, const void *attr);
int pthread_cond_destroy(pthread_cond_t *cond);
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);

int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_destroy(pthread_attr_t *attr);
int pthread_attr_setdetachstate(pthread_attr_t *attr, int state);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_PTHREAD_H */
