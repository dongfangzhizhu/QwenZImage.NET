/*
 * qzen_config.h -- QwenZImage.config.json reader for the native engine.
 *
 * The managed QziConfig loader parses the same file and passes the three core
 * directories through the qzi C ABI. This small C loader reads it again for
 * the fields the ABI intentionally does not carry: the LoRA directory, the
 * SVDQuant (Nunchaku) repacks and the platform overrides. Both sides resolve
 * the file with the same search order, so they can never disagree about which
 * pack is in use.
 *
 * This file is part of the renamed engine vendored under engine/. See
 * engine/UPSTREAM.md for the original project and its MIT license.
 */

#ifndef QZEN_CONFIG_H
#define QZEN_CONFIG_H

#include <stddef.h>

#define QZEN_CONFIG_PATH_MAX 1024
#define QZEN_CONFIG_FILE_MAX (1 << 20)

typedef struct qzen_config {
    char qwen21_dir[QZEN_CONFIG_PATH_MAX];
    char qwen21_lora_dir[QZEN_CONFIG_PATH_MAX];
    char zimage_dir[QZEN_CONFIG_PATH_MAX];
    char zimage_nunchaku_dir[QZEN_CONFIG_PATH_MAX];
    char text_cache_dir[QZEN_CONFIG_PATH_MAX];
    char source[QZEN_CONFIG_PATH_MAX];
} qzen_config;

/* Loads the first config file found.
 *
 * explicit_path overrides the search; when NULL, QZI_CONFIG is consulted,
 * then <exe dir>/QwenZImage.config.json, ./QwenZImage.config.json and
 * ~/.config/QwenZImage/config.json. Values may start with ~. Top-level keys
 * apply to every platform; a "windows" / "linux" / "osx" object overrides
 * them on that OS. Keys beginning with _ or $ are comments.
 *
 * Returns 1 when a file was read, 0 when no file exists (not an error),
 * -1 when an explicitly named file is missing or the JSON is unusable.
 * On -1, `error` receives a NUL-terminated message.
 */
int qzen_config_load(const char *explicit_path, qzen_config *out,
                     char *error, size_t error_cap);

/* Applies QZI_* environment overrides on top of a loaded config. The managed
 * QziConfig loader follows the same precedence (environment > file), and
 * QziClient publishes code-level values through these same variables for the
 * keys the C ABI cannot carry (qwen21_lora_dir / zimage_nunchaku_dir).
 * Values may start with ~. Returns 0, or -1 when a value is too long. */
int qzen_config_apply_environment(qzen_config *config, char *error,
                                  size_t error_cap);

/* Returns the value of a known key, or "" when unset/unknown. */
const char *qzen_config_field(const qzen_config *config, const char *key);

/* Expands a leading ~ into the user's home directory. Returns 0 on overflow. */
int qzen_config_expand_home(const char *path, char *out, size_t capacity);

#endif /* QZEN_CONFIG_H */
