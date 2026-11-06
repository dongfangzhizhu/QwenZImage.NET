/* Winsock2 implementation of the socket calls <sys/socket.h> maps onto.
 *
 * The engine's server is a tiny accept/recv/send loop written against POSIX
 * conventions: -1 with errno on failure, EINTR retried, close() for teardown.
 * Winsock differs in exactly three places, and all three are handled here:
 *
 *   - SOCKET is pointer-width and INVALID_SOCKET is ~0, not -1;
 *   - failures come back as SOCKET_ERROR with WSAGetLastError() instead of
 *     errno, and there is no EINTR (a blocking recv is not interrupted by a
 *     console control event);
 *   - sockets are released with closesocket(), not close().  The server calls
 *     close() on its listening socket, so close() itself is redirected here
 *     and decides which namespace the descriptor belongs to.
 */
#include "qzen_platform.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <sys/socket.h>

#include <errno.h>
#include <limits.h>

/* <sys/socket.h> renames recv/send/closesocket onto the shims below, so the
 * Winsock calls this file has to make are spelled through an #undef.  This
 * has to come *after* the include, not before it: the header defines the
 * macros, so an earlier #undef would simply be undone. */
#undef recv
#undef send
#undef closesocket

static int qzen_socket_ready = 0;

/* Winsock needs an explicit process-wide start-up before the first socket
 * call; it is idempotent, so the server calls it unconditionally on entry and
 * the recv/send wrappers call it defensively. */
int qzen_win_socket_startup(void) {
    if (qzen_socket_ready) return 1;
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 0;
    qzen_socket_ready = 1;
    return 1;
}

/* Maps the Winsock error onto errno so the server's strerror() messages and
 * its `errno == EINTR` retry loops keep working.  Returns the errno value. */
int qzen_win_socket_last_error(void) {
    int mapped;
    switch (WSAGetLastError()) {
        case WSAEINTR: mapped = EINTR; break;
        case WSAEWOULDBLOCK: mapped = EWOULDBLOCK; break;
        case WSAECONNRESET: mapped = ECONNRESET; break;
        case WSAECONNABORTED: mapped = ECONNABORTED; break;
        case WSAETIMEDOUT: mapped = ETIMEDOUT; break;
        case WSAENOTSOCK: mapped = ENOTSOCK; break;
        case WSAEADDRINUSE: mapped = EADDRINUSE; break;
        case WSAEADDRNOTAVAIL: mapped = EADDRNOTAVAIL; break;
        case WSAEACCES: mapped = EACCES; break;
        case WSAENOBUFS: mapped = ENOBUFS; break;
        case WSAEMSGSIZE: mapped = EMSGSIZE; break;
        case WSAEINVAL: mapped = EINVAL; break;
        default: mapped = EIO; break;
    }
    errno = mapped;
    return mapped;
}

ssize_t qzen_win_recv(intptr_t fd, void *buffer, size_t bytes, int flags) {
    if (!qzen_win_socket_startup()) {
        errno = EIO;
        return -1;
    }
    /* Winsock's length argument is a 32-bit int; the engine never asks for
     * more than a request body, so the clamp is a formality. */
    int chunk = bytes > (size_t)INT_MAX ? INT_MAX : (int)bytes;
    int received = recv((SOCKET)fd, (char *)buffer, chunk, flags);
    if (received == SOCKET_ERROR) {
        qzen_win_socket_last_error();
        return -1;
    }
    return (ssize_t)received;
}

ssize_t qzen_win_send(intptr_t fd, const void *buffer, size_t bytes, int flags) {
    if (!qzen_win_socket_startup()) {
        errno = EIO;
        return -1;
    }
    int chunk = bytes > (size_t)INT_MAX ? INT_MAX : (int)bytes;
    int sent = send((SOCKET)fd, (const char *)buffer, chunk, flags);
    if (sent == SOCKET_ERROR) {
        qzen_win_socket_last_error();
        return -1;
    }
    return (ssize_t)sent;
}

int qzen_win_socket_close(intptr_t fd) {
    if (fd == (intptr_t)INVALID_SOCKET || fd < 0) {
        errno = EBADF;
        return -1;
    }
    if (closesocket((SOCKET)fd) == 0) return 0;
    qzen_win_socket_last_error();
    return -1;
}

int qzen_win_socket_shutdown_write(intptr_t fd) {
    if (shutdown((SOCKET)fd, SD_SEND) == 0) return 0;
    qzen_win_socket_last_error();
    return -1;
}

/* close() has to serve both CRT descriptors (the FFmpeg pipes) and sockets.
 * closesocket() is tried first because it is the only one that refuses a
 * descriptor from the other namespace: a CRT fd answers WSAENOTSOCK and is
 * left untouched, which makes the fallback below safe in both directions. */
int qzen_win_close(intptr_t fd) {
    if (fd < 0) {
        errno = EBADF;
        return -1;
    }
    if (qzen_win_socket_startup() && closesocket((SOCKET)fd) == 0) return 0;
    if (WSAGetLastError() == WSAENOTSOCK) return _close((int)fd);
    qzen_win_socket_last_error();
    return -1;
}
