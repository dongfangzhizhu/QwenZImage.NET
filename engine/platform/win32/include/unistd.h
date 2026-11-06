/* POSIX names the engine expects from <unistd.h>.
 *
 * Most of them (read, write, open, close, unlink, access, dup, getcwd) already
 * exist in the UCRT under their POSIX spelling once _CRT_DECLARE_NONSTDC_NAMES
 * is on, so this header only has to pull in the right CRT header; the handful
 * that are genuinely missing are declared in qzen_platform.h. */
#ifndef QZEN_WIN_UNISTD_H
#define QZEN_WIN_UNISTD_H

#include "qzen_platform.h"

#include <io.h>
#include <direct.h>
#include <process.h>

#endif /* QZEN_WIN_UNISTD_H */
