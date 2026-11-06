#ifndef QZEN_DEVICE_H
#define QZEN_DEVICE_H

#include "qzen.h"

int qzen_device_probe(qzen_device_info *info, char *error, size_t error_size);

/* Does the GPU code in this binary match the device it is running on?
 * 1: it does.  0: it does not, and every kernel launch will fail.  -1: there
 * is nothing to compare, either because the build did not record an
 * architecture (Apple builds never do) or because no device was found. */
int qzen_device_arch_state(const qzen_device_info *info);

#endif
