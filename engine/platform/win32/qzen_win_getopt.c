#include "getopt.h"

#include <stdio.h>
#include <string.h>

char *optarg = NULL;
int optind = 1;
int opterr = 1;
int optopt = 0;

/* Where the next character of a clustered short option ("-abc") resumes. */
static int qzen_short_cursor = 0;

static void qzen_getopt_reset(void) {
    qzen_short_cursor = 0;
}

/* Handles one short option group.  Returns the option character, ':' when an
 * argument is missing, or '?' for an unknown option - the same contract as
 * glibc, so main.c's switch stays portable. */
static int qzen_short_option(int argc, char *const argv[], const char *optstring) {
    const char *current = argv[optind];
    char option = current[qzen_short_cursor];
    const char *spec = strchr(optstring, option);

    if (!spec || option == ':') {
        optopt = (unsigned char)option;
        if (qzen_short_cursor == 1) optind++;
        qzen_getopt_reset();
        if (opterr) {
            fprintf(stderr, "unrecognized option '-%c'\n", option);
        }
        return '?';
    }

    /* Advance past this option before consuming a possible argument. */
    qzen_short_cursor++;
    int takes_argument = spec[1] == ':';

    if (!takes_argument) {
        if (current[qzen_short_cursor] == '\0') {
            optind++;
            qzen_getopt_reset();
        }
        return (unsigned char)option;
    }

    if (current[qzen_short_cursor] != '\0') {
        /* "-dvalue" */
        optarg = (char *)(current + qzen_short_cursor);
        optind++;
        qzen_getopt_reset();
        return (unsigned char)option;
    }
    if (optind + 1 < argc) {
        /* "-d value" */
        optarg = argv[optind + 1];
        optind += 2;
        qzen_getopt_reset();
        return (unsigned char)option;
    }
    optopt = (unsigned char)option;
    optind++;
    qzen_getopt_reset();
    if (opterr) {
        fprintf(stderr, "option '-%c' requires an argument\n", option);
    }
    return optstring[0] == ':' ? ':' : '?';
}

int getopt(int argc, char *const argv[], const char *optstring) {
    optarg = NULL;
    while (optind < argc) {
        const char *current = argv[optind];
        if (current[0] != '-' || current[1] == '\0') return -1; /* not an option */
        if (strcmp(current, "--") == 0) {
            optind++;
            return -1;
        }
        if (qzen_short_cursor == 0) qzen_short_cursor = 1;
        return qzen_short_option(argc, argv, optstring);
    }
    return -1;
}

int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *long_options, int *long_index) {
    optarg = NULL;

    while (optind < argc) {
        const char *current = argv[optind];

        if (current[0] != '-' || current[1] == '\0') return -1;
        if (strcmp(current, "--") == 0) {
            optind++;
            return -1;
        }

        /* Long form. */
        if (current[1] == '-') {
            const char *name = current + 2;
            const char *equals = strchr(name, '=');
            size_t name_length = equals ? (size_t)(equals - name) : strlen(name);

            int match = -1;
            for (int index = 0; long_options && long_options[index].name;
                 index++) {
                if (strlen(long_options[index].name) == name_length &&
                    strncmp(long_options[index].name, name, name_length) == 0) {
                    match = index;
                    break;
                }
            }
            if (match < 0) {
                if (opterr) {
                    fprintf(stderr, "unrecognized option '--%.*s'\n",
                            (int)name_length, name);
                }
                optind++;
                return '?';
            }

            const struct option *entry = &long_options[match];
            if (entry->has_arg == required_argument) {
                if (equals) {
                    optarg = (char *)(equals + 1);
                } else if (optind + 1 < argc) {
                    optarg = argv[optind + 1];
                    optind++;
                } else {
                    if (opterr) {
                        fprintf(stderr, "option '--%s' requires an argument\n",
                                entry->name);
                    }
                    optind++;
                    return optstring[0] == ':' ? ':' : '?';
                }
            } else if (entry->has_arg == optional_argument) {
                if (equals) optarg = (char *)(equals + 1);
            } else if (equals) {
                if (opterr) {
                    fprintf(stderr, "option '--%s' takes no argument\n",
                            entry->name);
                }
                optind++;
                return '?';
            }

            optind++;
            if (long_index) *long_index = match;
            if (entry->flag) {
                *entry->flag = entry->val;
                continue; /* flag-only options report nothing */
            }
            return entry->val;
        }

        /* Short form, possibly clustered. */
        if (qzen_short_cursor == 0) qzen_short_cursor = 1;
        return qzen_short_option(argc, argv, optstring);
    }
    return -1;
}
