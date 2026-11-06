/*
 * qzen_config.c -- QwenZImage.config.json reader for the native engine.
 *
 * Derived from the upstream project's config loader (MIT, see UPSTREAM.md),
 * rewritten for the QwenZImageNative key set and the QZI_CONFIG environment
 * variable. Only path-valued keys are read here; numeric settings such as
 * device_id and thread_count belong to the managed QziClientOptions and are
 * deliberately ignored by this loader.
 */

#include "qzen_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "third_party/cJSON.h"

typedef struct {
    const char *key;
    size_t offset;
} config_field;

#define CONFIG_FIELD(name) {#name, offsetof(qzen_config, name)}

static const config_field config_fields[] = {
    CONFIG_FIELD(qwen21_dir),
    CONFIG_FIELD(qwen21_lora_dir),
    CONFIG_FIELD(zimage_dir),
    CONFIG_FIELD(zimage_nunchaku_dir),
    CONFIG_FIELD(text_cache_dir),
};

enum { CONFIG_FIELD_COUNT =
           (int)(sizeof(config_fields) / sizeof(config_fields[0])) };

static const char *home_dir(void) {
    static const char *resolved;
    if (resolved) return resolved;
    const char *home = getenv("HOME");
#if defined(_WIN32)
    if (!home || !*home) home = getenv("USERPROFILE");
#endif
    resolved = home && *home ? home : "";
    return resolved;
}

static char *config_slot(qzen_config *config, int index) {
    return (char *)config + config_fields[index].offset;
}

static int find_field(const char *key) {
    for (int index = 0; index < CONFIG_FIELD_COUNT; index++)
        if (!strcmp(config_fields[index].key, key)) return index;
    return -1;
}

/* Directory of the running module. Used so a config shipped next to the DLL
 * (or next to the host application) wins over the current working directory. */
static const char *program_dir(void) {
    static char directory[QZEN_CONFIG_PATH_MAX];
    static int state; /* 0 = untried, 1 = found, -1 = unavailable */
    if (state) return state > 0 ? directory : "";
    state = -1;
#if defined(_WIN32)
    wchar_t wide[QZEN_CONFIG_PATH_MAX];
    DWORD units = GetModuleFileNameW(NULL, wide,
                                     (DWORD)(sizeof(wide) / sizeof(wide[0])));
    if (units > 0 && units < sizeof(wide) / sizeof(wide[0])) {
        int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, (int)units, directory,
                                        (int)sizeof(directory) - 1, NULL, NULL);
        if (bytes > 0) {
            directory[bytes] = '\0';
            char *slash = strrchr(directory, '\\');
            char *forward = strrchr(directory, '/');
            if (forward && (!slash || forward > slash)) slash = forward;
            if (slash && slash != directory) {
                *slash = '\0';
                state = 1;
                for (char *cursor = directory; *cursor; cursor++)
                    if (*cursor == '\\') *cursor = '/';
            }
        }
    }
#else
    char link[QZEN_CONFIG_PATH_MAX * 2];
    ssize_t length = readlink("/proc/self/exe", link, sizeof(link) - 1);
    if (length > 0) {
        link[length] = '\0';
        char *slash = strrchr(link, '/');
        if (slash && slash != link) {
            *slash = '\0';
            if (snprintf(directory, sizeof(directory), "%s", link) > 0) state = 1;
        }
    }
#endif
    return state > 0 ? directory : "";
}

int qzen_config_expand_home(const char *path, char *out, size_t capacity) {
    if (!path || !out || !capacity) return 0;
    if (path[0] == '~' &&
        (path[1] == '\0' || path[1] == '/' || path[1] == '\\')) {
        const char *home = home_dir();
        if (*home) {
            int written = snprintf(out, capacity, "%s%s", home, path + 1);
            return written > 0 && (size_t)written < capacity;
        }
    }
    int written = snprintf(out, capacity, "%s", path);
    return written > 0 && (size_t)written < capacity;
}

static void report(char *error, size_t cap, const char *origin,
                   const char *key, const char *detail) {
    if (!error || !cap) return;
    if (key)
        snprintf(error, cap, "%s: \"%s\": %s", origin, key, detail);
    else
        snprintf(error, cap, "%s: %s", origin, detail);
}

static const char *platform_section(void) {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "osx";
#else
    return "linux";
#endif
}

/* Applies one JSON object. Top-level platform sections are objects rather
 * than paths; the caller applies the active section separately. */
static int apply_object(const cJSON *object, qzen_config *config,
                        const char *origin, int top_level,
                        char *error, size_t error_cap) {
    const cJSON *item = NULL;
    cJSON_ArrayForEach(item, object) {
        const char *key = item->string;
        if (!key) continue;
        if (top_level &&
            (!strcmp(key, "windows") || !strcmp(key, "linux") ||
             !strcmp(key, "osx") || !strcmp(key, "macos")))
            continue;
        if (key[0] == '_' || key[0] == '$') continue;
        int index = find_field(key);
        /* Managed-side-only numeric settings and future keys are ignored:
         * QziConfig.Load is the strict validator, this loader is the reader. */
        if (index < 0) continue;
        if (!cJSON_IsString(item) || !item->valuestring) {
            report(error, error_cap, origin, key, "must be a string path");
            return -1;
        }
        if (!*item->valuestring) continue;
        if (!qzen_config_expand_home(item->valuestring,
                                     config_slot(config, index),
                                     QZEN_CONFIG_PATH_MAX)) {
            report(error, error_cap, origin, key, "path is too long");
            return -1;
        }
    }
    return 0;
}

static int file_exists(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static char *read_file(const char *path, char *error, size_t error_cap) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        report(error, error_cap, path, NULL, "cannot be opened");
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        report(error, error_cap, path, NULL, "cannot be sized");
        return NULL;
    }
    long size = ftell(file);
    if (size < 0 || size > QZEN_CONFIG_FILE_MAX) {
        fclose(file);
        report(error, error_cap, path, NULL, "is empty or larger than 1 MiB");
        return NULL;
    }
    rewind(file);
    char *text = (char *)malloc((size_t)size + 1);
    if (!text) {
        fclose(file);
        report(error, error_cap, path, NULL, "out of memory");
        return NULL;
    }
    size_t read = fread(text, 1, (size_t)size, file);
    fclose(file);
    if (read != (size_t)size) {
        free(text);
        report(error, error_cap, path, NULL, "cannot be read");
        return NULL;
    }
    text[read] = '\0';
    return text;
}

int qzen_config_load(const char *explicit_path, qzen_config *out,
                     char *error, size_t error_cap) {
    if (error && error_cap) error[0] = '\0';
    if (!out) {
        report(error, error_cap, "config", NULL, "no destination");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (!explicit_path || !*explicit_path) explicit_path = getenv("QZI_CONFIG");

    char path[QZEN_CONFIG_PATH_MAX * 2];
    if (explicit_path && *explicit_path) {
        if (!qzen_config_expand_home(explicit_path, path, sizeof(path))) {
            report(error, error_cap, "config", NULL, "path is too long");
            return -1;
        }
        if (!file_exists(path)) {
            report(error, error_cap, path, NULL,
                   "not found (given by --config or QZI_CONFIG)");
            return -1;
        }
    } else {
        char candidates[3][QZEN_CONFIG_PATH_MAX];
        int count = 0;
        const char *program = program_dir();
        if (*program && count < 3) {
            snprintf(candidates[count++], QZEN_CONFIG_PATH_MAX,
                     "%s/QwenZImage.config.json", program);
        }
        if (count < 3)
            snprintf(candidates[count++], QZEN_CONFIG_PATH_MAX,
                     "%s", "QwenZImage.config.json");
        const char *home = home_dir();
        if (*home && count < 3)
            snprintf(candidates[count++], QZEN_CONFIG_PATH_MAX,
                     "%s/.config/QwenZImage/config.json", home);
        path[0] = '\0';
        for (int index = 0; index < count; index++) {
            if (!file_exists(candidates[index])) continue;
            snprintf(path, sizeof(path), "%s", candidates[index]);
            break;
        }
        if (!path[0]) return 0;
    }

    char *text = read_file(path, error, error_cap);
    if (!text) return -1;
    cJSON *root = cJSON_Parse(text);
    free(text);
    if (!root) {
        report(error, error_cap, path, NULL, "is not valid JSON");
        return -1;
    }
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        report(error, error_cap, path, NULL, "must contain a JSON object");
        return -1;
    }
    snprintf(out->source, sizeof(out->source), "%s", path);

    int status = apply_object(root, out, path, 1, error, error_cap);
    if (status == 0) {
        const cJSON *section = cJSON_GetObjectItemCaseSensitive(
            root, platform_section());
        if (section && cJSON_IsObject(section))
            status = apply_object(section, out, path, 0, error, error_cap);
    }
    cJSON_Delete(root);
    if (status != 0) {
        out->source[0] = '\0';
        return -1;
    }
    return 1;
}

const char *qzen_config_field(const qzen_config *config, const char *key) {
    if (!config || !key) return "";
    int index = find_field(key);
    if (index < 0) return "";
    return (const char *)config + config_fields[index].offset;
}
