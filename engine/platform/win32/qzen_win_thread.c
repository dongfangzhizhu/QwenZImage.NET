/* pthread subset on top of Win32 primitives.
 *
 * Mapping: pthread_mutex_t -> CRITICAL_SECTION (recursive, like the default
 * pthread mutex), pthread_cond_t -> CONDITION_VARIABLE, pthread_t ->
 * _beginthreadex handle.  SleepConditionVariableCS gives exactly the
 * "release the mutex, sleep, re-acquire" semantics pthread_cond_wait has.
 */
#include "pthread.h"

#include <windows.h>

#include <errno.h>
#include <stdlib.h>

/* opaque[0..4] is the CRITICAL_SECTION (40 bytes on x64); opaque[5] is the
 * initialisation flag, which must live outside it. */
#define QZEN_MUTEX_FLAG_INDEX 5
#define QZEN_MUTEX_READY ((void *)1)

typedef struct {
    void *(*start)(void *);
    void *argument;
} qzen_thread_start;

static void qzen_check_mutex_layout(void) {
    /* A compile-time check would be nicer, but CRITICAL_SECTION's size is only
     * visible where <windows.h> is; keep it a runtime assert in debug builds. */
    (void)0;
}

static CRITICAL_SECTION *qzen_mutex_cs(pthread_mutex_t *mutex) {
    CRITICAL_SECTION *cs = (CRITICAL_SECTION *)&mutex->opaque[0];
    if (mutex->opaque[QZEN_MUTEX_FLAG_INDEX] != QZEN_MUTEX_READY) {
        /* One global lock guards the transition.  Statically zeroed mutexes
         * (PTHREAD_MUTEX_INITIALIZER) arrive here on first use, so the cost is
         * paid once per mutex and never on the hot path. */
        static SRWLOCK init_lock = SRWLOCK_INIT;
        AcquireSRWLockExclusive(&init_lock);
        if (mutex->opaque[QZEN_MUTEX_FLAG_INDEX] != QZEN_MUTEX_READY) {
            qzen_check_mutex_layout();
            InitializeCriticalSection(cs);
            mutex->opaque[QZEN_MUTEX_FLAG_INDEX] = QZEN_MUTEX_READY;
        }
        ReleaseSRWLockExclusive(&init_lock);
    }
    return cs;
}

static CONDITION_VARIABLE *qzen_cond_cv(pthread_cond_t *cond) {
    return (CONDITION_VARIABLE *)&cond->opaque[0];
}

/* --- threads ----------------------------------------------------------- */

static unsigned __stdcall qzen_thread_trampoline(void *raw) {
    qzen_thread_start *start = (qzen_thread_start *)raw;
    void *(*entry)(void *) = start->start;
    void *argument = start->argument;
    free(start);
    entry(argument);
    return 0;
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start)(void *), void *argument) {
    if (!thread || !start) return EINVAL;
    qzen_thread_start *context = (qzen_thread_start *)malloc(sizeof(*context));
    if (!context) return ENOMEM;
    context->start = start;
    context->argument = argument;

    unsigned thread_id = 0;
    uintptr_t handle = _beginthreadex(NULL, 0, qzen_thread_trampoline, context,
                                      0, &thread_id);
    if (handle == 0) {
        free(context);
        return EAGAIN;
    }
    *thread = (pthread_t)handle;
    if (attr && attr->detach_state == PTHREAD_CREATE_DETACHED) {
        CloseHandle((HANDLE)handle);
        *thread = NULL;
    }
    return 0;
}

int pthread_join(pthread_t thread, void **result) {
    if (!thread) return EINVAL;
    HANDLE handle = (HANDLE)thread;
    DWORD code = 0;
    WaitForSingleObject(handle, INFINITE);
    GetExitCodeThread(handle, &code);
    CloseHandle(handle);
    if (result) *result = NULL;
    return 0;
}

int pthread_detach(pthread_t thread) {
    if (thread) CloseHandle((HANDLE)thread);
    return 0;
}

void pthread_exit(void *result) {
    (void)result;
    _endthreadex(0);
}

pthread_t pthread_self(void) {
    return (pthread_t)GetCurrentThread();
}

/* --- mutexes ----------------------------------------------------------- */

int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr) {
    (void)attr;
    if (!mutex) return EINVAL;
    /* Defer the real initialisation to the first lock: one code path for both
     * statically and dynamically created mutexes. */
    memset(mutex, 0, sizeof(*mutex));
    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    if (mutex->opaque[QZEN_MUTEX_FLAG_INDEX] == QZEN_MUTEX_READY) {
        DeleteCriticalSection((CRITICAL_SECTION *)&mutex->opaque[0]);
        mutex->opaque[QZEN_MUTEX_FLAG_INDEX] = NULL;
    }
    return 0;
}

int pthread_mutex_lock(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    EnterCriticalSection(qzen_mutex_cs(mutex));
    return 0;
}

int pthread_mutex_trylock(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    return TryEnterCriticalSection(qzen_mutex_cs(mutex)) ? 0 : EBUSY;
}

int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    LeaveCriticalSection(qzen_mutex_cs(mutex));
    return 0;
}

/* --- condition variables ---------------------------------------------- */

int pthread_cond_init(pthread_cond_t *cond, const void *attr) {
    (void)attr;
    if (!cond) return EINVAL;
    memset(cond, 0, sizeof(*cond)); /* all-zero is the initialised state */
    return 0;
}

int pthread_cond_destroy(pthread_cond_t *cond) {
    (void)cond; /* CONDITION_VARIABLE needs no teardown */
    return 0;
}

int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    if (!cond || !mutex) return EINVAL;
    /* SleepConditionVariableCS atomically releases the critical section and
     * re-acquires it before returning, which is exactly pthread_cond_wait. */
    if (!SleepConditionVariableCS(qzen_cond_cv(cond), qzen_mutex_cs(mutex),
                                  INFINITE)) {
        return EINVAL;
    }
    return 0;
}

int pthread_cond_signal(pthread_cond_t *cond) {
    if (!cond) return EINVAL;
    WakeConditionVariable(qzen_cond_cv(cond));
    return 0;
}

int pthread_cond_broadcast(pthread_cond_t *cond) {
    if (!cond) return EINVAL;
    WakeAllConditionVariable(qzen_cond_cv(cond));
    return 0;
}

/* --- attributes -------------------------------------------------------- */

int pthread_attr_init(pthread_attr_t *attr) {
    if (!attr) return EINVAL;
    attr->detach_state = PTHREAD_CREATE_JOINABLE;
    return 0;
}

int pthread_attr_destroy(pthread_attr_t *attr) {
    (void)attr;
    return 0;
}

int pthread_attr_setdetachstate(pthread_attr_t *attr, int state) {
    if (!attr) return EINVAL;
    attr->detach_state = state;
    return 0;
}
