/* <dirent.h> over FindFirstFile/FindNextFile.
 *
 * The engine only ever reads d_name (to pick *.safetensors and *.json out of a
 * pack directory), so d_type is filled in on a best-effort basis and never
 * relied on.  d_name is MAX_PATH wide because that is what FindData carries. */
#ifndef QZEN_WIN_DIRENT_H
#define QZEN_WIN_DIRENT_H

#include "qzen_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DT_UNKNOWN 0
#define DT_DIR 4
#define DT_REG 8

struct dirent {
    char d_name[260];
    int d_type;
};

typedef struct qzen_win_dir DIR;

DIR *opendir(const char *path);
struct dirent *readdir(DIR *directory);
int closedir(DIR *directory);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_DIRENT_H */
