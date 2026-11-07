/* Native ABI smoke test: no GPU, no weights, no external dependencies.
 * Run by ctest (see native/CMakeLists.txt) and by the CI workflow.
 *
 * With the deterministic reference backend it exercises the whole contract,
 * including byte-identical generation and buffer ownership. With the real
 * C/CUDA backend (QZI_WITH_CUDA_ENGINE=ON) there are no model weights in CI,
 * so it only checks the parts that never touch the device: ABI layout, engine
 * creation, argument validation and the backend flag. Generation with real
 * weights is exercised by the WebUI / Sample instead.
 */

/* setenv/unsetenv need the POSIX feature macro before any system header. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "qzi.h"
#include "qzi_env.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static int failures = 0;

#define CHECK(cond, label) do {                                        \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL: %s (line %d)\n", label, __LINE__);     \
        failures++;                                                    \
    }                                                                  \
} while (0)

static int is_png(const qzi_image *image)
{
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    return image->size > 33 && memcmp(image->data, sig, 8) == 0;
}

static int model_load_seen = 0;

static void on_progress(const char *phase, int32_t completed, int32_t total,
                        double elapsed, void *user)
{
    (void)completed; (void)total; (void)elapsed; (void)user;
    if (phase && strcmp(phase, "model load") == 0) model_load_seen++;
}

/* LoRA validation is duplicated in qzi.c, the managed RequestValidator and
 * the real backend. These cases run before the vtable is called, so they are
 * valid on both backends (no model weights needed). Returns the number of
 * rejected requests issued, so main() can assert the health counters. */
static int check_lora_validation(qzi_engine *engine,
                                 const qzi_qwen21_request *base,
                                 char *error, size_t error_cap)
{
    char long_name[QZI_LORA_NAME_MAX + 2];
    qzi_image image;
    int index, rejected = 0;
    struct {
        const char *lora;
        float scale;
    } cases[] = {
        {"../escape.safetensors", 1.0f},
        {"C:escape.safetensors", 1.0f},
        {"a\\b.safetensors", 1.0f},
        {NULL, 1.0f},                 /* filled with long_name below */
        {NULL, 2.0f},                 /* scale without an adapter */
        {"a.safetensors", 5.0f},
        {"a.safetensors", -1.0f},
        {"a.safetensors", 0.01f},
        {"a.safetensors", 4.01f},
    };
    memset(&image, 0, sizeof(image));
    memset(long_name, 'a', sizeof(long_name) - 1);
    long_name[sizeof(long_name) - 1] = '\0';
    cases[3].lora = long_name;       /* QZI_LORA_NAME_MAX + 1 bytes */

    for (index = 0; index < (int)(sizeof(cases) / sizeof(cases[0])); index++) {
        qzi_qwen21_request r = *base;
        r.lora = cases[index].lora;
        r.lora_scale = cases[index].scale;
        if (qzi_generate_qwen21(engine, &r, NULL, &image, error,
                                error_cap) != QZI_ERR_INVALID_ARGUMENT) {
            fprintf(stderr, "FAIL: lora validation case %d (%s)\n",
                    index, error[0] ? error : "no error");
            failures++;
        }
        rejected++;
    }
    return rejected;
}

/* QziClient scopes QZI_* variables around qzi_engine_create(). On Windows the
 * host changes the Win32 environment, which the native DLL's CRT getenv()
 * cannot see (it holds a startup snapshot). Pin down that qzi_getenv() does. */
static void check_scoped_environment(void)
{
    const char *name = "QZI_SMOKE_SCOPED";
    char buffer[64];
    char tiny[4];

#ifdef _WIN32
#define SET_ENV(v) SetEnvironmentVariableA(name, v)
#define CLEAR_ENV() SetEnvironmentVariableA(name, NULL)
#else
#define SET_ENV(v) setenv(name, v, 1)
#define CLEAR_ENV() unsetenv(name)
#endif

    CLEAR_ENV();
    CHECK(qzi_getenv(name, buffer, sizeof(buffer)) == 0,
          "unset scoped env reads as unset");

    SET_ENV("live");
    CHECK(qzi_getenv(name, buffer, sizeof(buffer)) == 1 &&
          strcmp(buffer, "live") == 0,
          "scoped env value visible to native config reader");
    SET_ENV("much-longer-than-a-four-byte-buffer");
    CHECK(qzi_getenv(name, tiny, sizeof(tiny)) == -1 && tiny[0] == '\0',
          "oversized env value is reported, not truncated");
    CLEAR_ENV();
    CHECK(qzi_getenv(name, buffer, sizeof(buffer)) == 0,
          "env removal visible to native config reader");

#undef SET_ENV
#undef CLEAR_ENV
}

int main(void)
{
    qzi_engine_config config;
    qzi_qwen21_request qwen;
    qzi_zimage_request zimage;
    qzi_image a, b, c;
    qzi_callbacks callbacks;
    qzi_health health;
    qzi_engine *engine = NULL;
    char error[QZI_ERROR_CAPACITY];
    size_t sizes[QZI_LAYOUT_SLOTS];
    qzi_status status;
    int equal = 0;
    int reference = 0;
    int extra_rejected = 0;

    memset(&config, 0, sizeof(config));
    memset(&qwen, 0, sizeof(qwen));
    memset(&zimage, 0, sizeof(zimage));
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&c, 0, sizeof(c));
    memset(&callbacks, 0, sizeof(callbacks));
    memset(&health, 0, sizeof(health));

    check_scoped_environment();

    CHECK(qzi_abi_version() == QZI_ABI_VERSION, "abi version");
    CHECK(strcmp(qzi_status_text(QZI_ERR_BUSY),
                 "a generation owns the device") == 0, "status text");
    CHECK(qzi_abi_layout(sizes, QZI_LAYOUT_SLOTS) == QZI_OK, "abi layout");
    CHECK(sizes[QZI_LAYOUT_ENGINE_CONFIG] == sizeof(qzi_engine_config),
          "engine config size");
    CHECK(sizes[QZI_LAYOUT_QWEN21_REQUEST] == sizeof(qzi_qwen21_request),
          "qwen21 request size");
    CHECK(sizes[QZI_LAYOUT_IMAGE] == sizeof(qzi_image), "image size");

    config.abi_version = QZI_ABI_VERSION;
    config.qwen21_dir = ".";
    config.zimage_dir = ".";
    config.cache_dir = ".";
    config.device_id = -1;
    config.thread_count = 0;
    status = qzi_engine_create(&config, &engine, error, sizeof(error));
    CHECK(status == QZI_OK && engine != NULL, "engine create");
    if (status != QZI_OK) {
        fprintf(stderr, "  create error: %s\n", error);
        return 1;
    }

    CHECK(qzi_engine_health(engine, &health, error, sizeof(error)) == QZI_OK,
          "initial health");
    reference = health.reference_backend == 1;
    printf("qzi_abi_smoke: backend = %s\n",
           reference ? "deterministic reference" : "real C/CUDA engine");

    qwen.prompt = "a red fox in snow";
    qwen.width = 64;
    qwen.height = 64;
    qwen.steps = 8;
    qwen.guidance = 1.0f;
    qwen.lora_scale = 0.0f;   /* 0 = ABI default (1.0); non-zero requires lora */
    qwen.seed = 42;

    zimage.prompt = "a red fox in snow";
    zimage.width = 64;
    zimage.height = 64;
    zimage.seed = 42;
    zimage.checkpoint = "bf16";

    if (reference) {
        status = qzi_generate_qwen21(engine, &qwen, NULL, &a, error,
                                     sizeof(error));
        CHECK(status == QZI_OK, "qwen21 generate");
        if (status != QZI_OK) fprintf(stderr, "  qwen21 error: %s\n", error);
        CHECK(is_png(&a), "qwen21 png signature");
        CHECK(a.width == 64 && a.height == 64, "qwen21 dimensions");
        CHECK(a.token != 0, "qwen21 release token");

        status = qzi_generate_qwen21(engine, &qwen, NULL, &b, error,
                                     sizeof(error));
        CHECK(status == QZI_OK, "qwen21 second generate");
        if (status == QZI_OK && a.size == b.size)
            equal = memcmp(a.data, b.data, a.size) == 0;
        CHECK(equal, "identical request is byte-identical");

        /* LoRA and its strength are part of the request identity. Without a
         * GPU these bytes are the only way to prove the parameter is plumbed
         * all the way through the ABI instead of being silently dropped. */
        callbacks.progress = on_progress;
        qwen.lora = "dmd2.safetensors";
        qwen.lora_scale = 0.0f;        /* ABI default -> effective 1.0 */
        model_load_seen = 0;
        status = qzi_generate_qwen21(engine, &qwen, &callbacks, &c, error,
                                     sizeof(error));
        CHECK(status == QZI_OK, "qwen21 lora generate");
        if (status != QZI_OK) fprintf(stderr, "  lora error: %s\n", error);
        CHECK(status != QZI_OK ||
              (a.size == c.size && memcmp(a.data, c.data, a.size) != 0),
              "lora changes output bytes");
        CHECK(model_load_seen == 1, "adapter switch emits model load phase");

        qzi_image_release(engine, &b);   /* free the byte-identical baseline */
        qwen.lora_scale = 2.0f;
        model_load_seen = 0;
        status = qzi_generate_qwen21(engine, &qwen, &callbacks, &b, error,
                                     sizeof(error));
        CHECK(status == QZI_OK, "qwen21 lora scale generate");
        CHECK(status != QZI_OK ||
              (b.size == c.size && memcmp(b.data, c.data, b.size) != 0),
              "lora_scale changes output bytes");
        CHECK(model_load_seen == 1, "scale change emits model load phase");

        qwen.lora = NULL;
        qwen.lora_scale = 0.0f;
        callbacks.progress = NULL;
        qzi_image_release(engine, &c);
        qzi_image_release(engine, &b);
        CHECK(b.data == NULL && b.token == 0, "image release zeroes struct");
        qzi_image_release(engine, &a);

        status = qzi_generate_zimage(engine, &zimage, NULL, &a, error,
                                     sizeof(error));
        CHECK(status == QZI_OK, "zimage generate");
        if (status != QZI_OK) fprintf(stderr, "  zimage error: %s\n", error);
        CHECK(is_png(&a), "zimage png signature");
        qzi_image_release(engine, &a);
    } else {
        printf("qzi_abi_smoke: generation skipped (real backend needs model "
               "weights configured in QwenZImage.config.json)\n");
    }

    /* Validation is enforced on the native side too, not just in .NET. These
     * two requests are rejected by qzi.c before the vtable is called, so they
     * work on a real backend built without weights installed. */
    qwen.width = 100; /* not a multiple of 32 */
    status = qzi_generate_qwen21(engine, &qwen, NULL, &a, error,
                                 sizeof(error));
    CHECK(status == QZI_ERR_INVALID_ARGUMENT, "qwen21 alignment validation");
    CHECK(strlen(error) > 0, "native error message populated");
    qwen.width = 64;

    zimage.checkpoint = "svdq-fp4-r256";
    status = qzi_generate_zimage(engine, &zimage, NULL, &a, error,
                                 sizeof(error));
    CHECK(status == QZI_ERR_UNSUPPORTED, "fp4 checkpoint rejected");
    extra_rejected = 0;
    zimage.checkpoint = "SVDQ-FP4-r256";   /* managed side is case-insensitive */
    status = qzi_generate_zimage(engine, &zimage, NULL, &a, error,
                                 sizeof(error));
    CHECK(status == QZI_ERR_UNSUPPORTED, "uppercase fp4 checkpoint rejected");
    extra_rejected++;
    zimage.checkpoint = "bf16";

    /* Every LoRA validation case is rejected in qzi.c, before any backend
     * call, so these run on the real CUDA build too. */
    extra_rejected += check_lora_validation(engine, &qwen, error,
                                            sizeof(error));

    CHECK(qzi_engine_health(engine, &health, error, sizeof(error)) == QZI_OK,
          "health");
    if (reference) {
        /* 2 baseline qwen21 + 2 LoRA qwen21 + 1 zimage generation. */
        CHECK(health.served == 5, "served counter");
        CHECK(health.rejected == 2 + extra_rejected, "rejected counter");
        CHECK(health.reference_backend == 1, "reference backend flag");
        CHECK(health.resident_model == QZI_MODEL_ZIMAGE, "resident model");
    } else {
        CHECK(health.served == 0, "served counter");
        CHECK(health.rejected == 2 + extra_rejected, "rejected counter");
        CHECK(health.reference_backend == 0, "real backend flag");
    }
    CHECK(qzi_engine_unload(engine, error, sizeof(error)) == QZI_OK, "unload");

    qzi_engine_destroy(engine);

    if (failures) {
        fprintf(stderr, "qzi_abi_smoke: %d failure(s)\n", failures);
        return 1;
    }
    printf("qzi_abi_smoke: OK (%s)\n",
           reference ? "full generation contract"
                     : "ABI + validation only, real backend");
    return 0;
}
