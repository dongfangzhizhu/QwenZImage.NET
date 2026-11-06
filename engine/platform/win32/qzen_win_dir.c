#include "dirent.h"

#include <windows.h>

#include <stdlib.h>
#include <string.h>

struct qzen_win_dir {
    HANDLE handle;
    int first;
    WIN32_FIND_DATAA data;
    struct dirent entry;
};

DIR *opendir(const char *path) {
    if (!path || !*path) return NULL;

    size_t length = strlen(path);
    /* FindFirstFile wants a wildcard, and "dir" without a trailing separator
     * has to become "dir\*". */
    char *pattern = (char *)malloc(length + 3);
    if (!pattern) return NULL;
    memcpy(pattern, path, length);
    if (path[length - 1] == '\\' || path[length - 1] == '/') {
        memcpy(pattern + length, "*", 2);
    } else {
        memcpy(pattern + length, "\\*", 3);
    }

    DIR *directory = (DIR *)calloc(1, sizeof(*directory));
    if (!directory) {
        free(pattern);
        return NULL;
    }
    directory->handle = FindFirstFileA(pattern, &directory->data);
    free(pattern);
    if (directory->handle == INVALID_HANDLE_VALUE) {
        free(directory);
        return NULL;
    }
    directory->first = 1;
    return directory;
}

struct dirent *readdir(DIR *directory) {
    if (!directory) return NULL;
    if (!directory->first) {
        if (!FindNextFileA(directory->handle, &directory->data)) return NULL;
    }
    directory->first = 0;

    snprintf(directory->entry.d_name, sizeof(directory->entry.d_name), "%s",
             directory->data.cFileName);
    directory->entry.d_type =
        (directory->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            ? DT_DIR
            : DT_REG;
    return &directory->entry;
}

int closedir(DIR *directory) {
    if (!directory) return -1;
    FindClose(directory->handle);
    free(directory);
    return 0;
}
