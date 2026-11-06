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

#include "qzi.h"

#include <stdio.h>
#include <string.h>

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

int main(void)
{
    qzi_engine_config config;
    qzi_qwen21_request qwen;
    qzi_zimage_request zimage;
    qzi_image a, b;
    qzi_health health;
    qzi_engine *engine = NULL;
    char error[QZI_ERROR_CAPACITY];
    size_t sizes[QZI_LAYOUT_SLOTS];
    qzi_status status;
    int equal = 0;
    int reference = 0;

    memset(&config, 0, sizeof(config));
    memset(&qwen, 0, sizeof(qwen));
    memset(&zimage, 0, sizeof(zimage));
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&health, 0, sizeof(health));

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
    qwen.lora_scale = 1.0f;
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
    zimage.checkpoint = "bf16";

    CHECK(qzi_engine_health(engine, &health, error, sizeof(error)) == QZI_OK,
          "health");
    if (reference) {
        CHECK(health.served == 3, "served counter");
        CHECK(health.rejected == 2, "rejected counter");
        CHECK(health.reference_backend == 1, "reference backend flag");
        CHECK(health.resident_model == QZI_MODEL_ZIMAGE, "resident model");
    } else {
        CHECK(health.served == 0, "served counter");
        CHECK(health.rejected == 2, "rejected counter");
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
