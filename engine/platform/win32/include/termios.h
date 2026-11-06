/* <termios.h>: enough of the interface for linenoise.c to compile.
 *
 * Windows has no termios.  tcgetattr() fails on purpose, which makes linenoise
 * report "not a terminal" and fall back instead of pretending it switched the
 * console into raw mode.  The interactive prompt is not the supported entry
 * point on Windows anyway - the subcommands are. */
#ifndef QZEN_WIN_TERMIOS_H
#define QZEN_WIN_TERMIOS_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char cc_t;
typedef unsigned int speed_t;
typedef unsigned int tcflag_t;

#define NCCS 32

struct termios {
    tcflag_t c_iflag;
    tcflag_t c_oflag;
    tcflag_t c_cflag;
    tcflag_t c_lflag;
    cc_t c_cc[NCCS];
};

/* Input modes. */
#define BRKINT 0x0002
#define ICRNL 0x0100
#define INPCK 0x0010
#define ISTRIP 0x0020
#define IXON 0x0400

/* Output modes. */
#define OPOST 0x0001

/* Control modes. */
#define CS8 0x0030

/* Local modes. */
#define ECHO 0x0008
#define ICANON 0x0002
#define IEXTEN 0x8000
#define ISIG 0x0001

/* Control characters. */
#define VMIN 6
#define VTIME 5

#define TCSANOW 0
#define TCSAFLUSH 2

int tcgetattr(int fd, struct termios *termios_state);
int tcsetattr(int fd, int actions, const struct termios *termios_state);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_TERMIOS_H */
