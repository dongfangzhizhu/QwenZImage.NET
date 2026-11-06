/* <sys/ioctl.h>: only TIOCGWINSZ, which linenoise uses to learn the window
 * width.  The shim answers from the real console buffer so the prompt wraps
 * where the user expects. */
#ifndef QZEN_WIN_SYS_IOCTL_H
#define QZEN_WIN_SYS_IOCTL_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};

#define TIOCGWINSZ 0x5413

int ioctl(int fd, unsigned long request, ...);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_SYS_IOCTL_H */
