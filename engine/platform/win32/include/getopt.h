/* <getopt.h> for Windows: getopt_long plus the three globals, matching the
 * glibc behaviour main.c relies on (clustered short options, "--name=value",
 * "--name value", "--" terminator, optind after the loop). */
#ifndef QZEN_WIN_GETOPT_H
#define QZEN_WIN_GETOPT_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define no_argument 0
#define required_argument 1
#define optional_argument 2

struct option {
    const char *name;
    int has_arg;
    int *flag;
    int val;
};

extern char *optarg;
extern int optind;
extern int opterr;
extern int optopt;

int getopt(int argc, char *const argv[], const char *optstring);
int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *long_options, int *long_index);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_GETOPT_H */
