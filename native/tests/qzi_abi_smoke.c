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
#include "qzi_backend.h"
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
static int image_encode_seen = 0;

static void on_progress(const char *phase, int32_t completed, int32_t total,
                        double elapsed, void *user)
{
    (void)completed; (void)total; (void)elapsed; (void)user;
    if (phase && strcmp(phase, "model load") == 0) model_load_seen++;
    if (phase && strcmp(phase, "image encode") == 0) image_encode_seen++;
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

/* Image editing is an additive ABI v1 extension. These requests are rejected
 * in qzi.c before the vtable is called, so they run on both backends and on
 * CI without model weights. Returns the number of rejected requests. */
#define EXPECT_EDIT_INVALID(call, label) do {                          \
    if ((call) != QZI_ERR_INVALID_ARGUMENT) {                          \
        fprintf(stderr, "FAIL: edit validation %s (%s)\n", label,     \
                error[0] ? error : "no error");                        \
        failures++;                                                    \
    }                                                                  \
    rejected++;                                                        \
} while (0)

static int check_edit_validation(qzi_engine *engine,
                                 const qzi_qwen21_request *base,
                                 char *error, size_t error_cap)
{
    static uint8_t rgb[3 * 64 * 64];
    static uint8_t mask[64 * 64];
    qzi_qwen21_edit_request e;
    qzi_image image;
    int rejected = 0;

    memset(&image, 0, sizeof(image));
    memset(&e, 0, sizeof(e));
    e.base = *base;
    e.reference_rgb = rgb;
    e.reference_width = 64;
    e.reference_height = 64;
    e.strength = 0.5f;

    e.reference_rgb = NULL;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "reference_rgb required");
    e.reference_rgb = rgb;
    e.reference_width = 0;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "reference dimensions");
    e.reference_width = 64;
    e.strength = 2.0f;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "strength upper bound");
    e.strength = 0.01f;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "strength lower bound");
    e.strength = 0.5f;
    e.masking_strength = 0.5f;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "masking_strength requires mask");
    e.mask = mask;
    e.masking_strength = 2.0f;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "masking_strength bound");
    e.mask = NULL;
    e.masking_strength = 0.0f;
    e.flags = 1;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "reserved flags");
    e.flags = 0;
    e.base.width = 100;
    EXPECT_EDIT_INVALID(qzi_generate_qwen21_edit(engine, &e, NULL, &image,
                                                 error, error_cap),
                        "base canvas validation");
    e.base.width = 64;
    return rejected;
}

#undef EXPECT_EDIT_INVALID

/* With the deterministic backend the edit request must be observable in the
 * bytes: a different reference, a mask, or a different strength each change
 * the result. Three successful generations, called only for `reference`. */
static void check_edit_generation(qzi_engine *engine,
                                  const qzi_qwen21_request *base,
                                  char *error, size_t error_cap)
{
    static uint8_t rgb[3 * 64 * 64];
    static uint8_t mask[64 * 64];
    qzi_qwen21_edit_request e;
    qzi_image x, y, z;
    qzi_status status;

    memset(&x, 0, sizeof(x));
    memset(&y, 0, sizeof(y));
    memset(&z, 0, sizeof(z));
    memset(rgb, 0x11, sizeof(rgb));
    memset(mask, 0, sizeof(mask));
    memset(&e, 0, sizeof(e));
    e.base = *base;
    e.reference_rgb = rgb;
    e.reference_width = 64;
    e.reference_height = 64;
    e.strength = 0.5f;

    status = qzi_generate_qwen21_edit(engine, &e, NULL, &x, error, error_cap);
    CHECK(status == QZI_OK, "qwen21 edit generate");
    if (status != QZI_OK) fprintf(stderr, "  edit error: %s\n", error);
    CHECK(is_png(&x), "edit png signature");

    memset(rgb, 0xEE, sizeof(rgb));
    status = qzi_generate_qwen21_edit(engine, &e, NULL, &y, error, error_cap);
    CHECK(status == QZI_OK, "qwen21 edit different reference");
    CHECK(status != QZI_OK ||
          (x.size == y.size && memcmp(x.data, y.data, x.size) != 0),
          "reference image changes output bytes");

    memset(mask, 255, sizeof(mask) / 2);
    e.mask = mask;
    e.masking_strength = 0.5f;
    status = qzi_generate_qwen21_edit(engine, &e, NULL, &z, error, error_cap);
    CHECK(status == QZI_OK, "qwen21 edit with mask");
    CHECK(status != QZI_OK ||
          (y.size == z.size && memcmp(y.data, z.data, y.size) != 0),
          "mask changes output bytes");

    qzi_image_release(engine, &x);
    qzi_image_release(engine, &y);
    qzi_image_release(engine, &z);
}

/* Advanced Qwen-Image 2.1 editing is the THIRD additive ABI v1 extension:
 * Red Canvas outpainting, LanPaint model modes and RGBA output. Validation
 * happens in qzi.c before the vtable is called, so these cases also run on
 * the real CUDA build without weights. Returns the number of rejected calls. */
#define EXPECT_ADV_INVALID(call, label) do {                           \
    if ((call) != QZI_ERR_INVALID_ARGUMENT) {                          \
        fprintf(stderr, "FAIL: advanced edit validation %s (%s)\n",   \
                label, error[0] ? error : "no error");                 \
        failures++;                                                    \
    }                                                                  \
    rejected++;                                                        \
} while (0)

static int check_edit_ex_validation(qzi_engine *engine,
                                    const qzi_qwen21_request *base,
                                    char *error, size_t error_cap)
{
    static uint8_t rgb[3 * 64 * 64];
    static uint8_t mask[64 * 64];
    qzi_qwen21_edit_ex_request e;
    qzi_image image;
    int rejected = 0;

    memset(&image, 0, sizeof(image));
    memset(&e, 0, sizeof(e));
    memset(rgb, 0x22, sizeof(rgb));
    memset(mask, 255, sizeof(mask));
    e.base.base = *base;
    e.base.reference_rgb = rgb;
    e.base.reference_width = 64;
    e.base.reference_height = 64;
    e.base.strength = 0.5f;

    e.flags = 1;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "reserved flags");
    e.flags = 0;
    e.margin_top = 16;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "margin alignment");
    e.margin_top = 64;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "source rectangle too small");
    e.margin_top = 0;
    e.model_mode = 7;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "unknown model mode");
    e.model_mode = QZI_EDIT_MODE_LANPAINT_5;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "LanPaint requires mask");
    e.base.mask = mask;
    e.base.masking_strength = 1.0f;
    e.margin_left = 32;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "LanPaint and outpainting are exclusive");
    e.margin_left = 0;
    e.rgba = 2;
    EXPECT_ADV_INVALID(qzi_generate_qwen21_edit_ex(engine, &e, NULL, &image,
                                                   error, error_cap),
                       "rgba must be 0 or 1");
    return rejected;
}

#undef EXPECT_ADV_INVALID

/* The deterministic backend mixes every advanced knob into its PNG tag and
 * can emit colour type 6, so the whole path is observable without a GPU. */
static void check_edit_ex_generation(qzi_engine *engine,
                                     const qzi_qwen21_request *base,
                                     char *error, size_t error_cap)
{
    static uint8_t rgb[3 * 64 * 64];
    static uint8_t mask[64 * 64];
    qzi_qwen21_edit_ex_request e;
    qzi_image a, b, c, d;
    qzi_status status;

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&c, 0, sizeof(c));
    memset(&d, 0, sizeof(d));
    memset(rgb, 0x44, sizeof(rgb));
    memset(mask, 255, sizeof(mask));
    memset(&e, 0, sizeof(e));
    e.base.base = *base;
    e.base.reference_rgb = rgb;
    e.base.reference_width = 64;
    e.base.reference_height = 64;
    e.base.strength = 0.5f;

    status = qzi_generate_qwen21_edit_ex(engine, &e, NULL, &a, error, error_cap);
    CHECK(status == QZI_OK, "advanced edit generate");
    if (status != QZI_OK) fprintf(stderr, "  adv edit error: %s\n", error);
    CHECK(is_png(&a), "advanced edit png signature");
    CHECK(a.data && a.size > 26 && a.data[25] == 2, "advanced edit RGB colour type");

    e.margin_top = 32;
    status = qzi_generate_qwen21_edit_ex(engine, &e, NULL, &b, error, error_cap);
    CHECK(status == QZI_OK, "outpainting generate");
    CHECK(status != QZI_OK ||
          (a.size == b.size && memcmp(a.data, b.data, a.size) != 0),
          "outpainting margins change output bytes");

    e.margin_top = 0;
    e.rgba = 1;
    status = qzi_generate_qwen21_edit_ex(engine, &e, NULL, &c, error, error_cap);
    CHECK(status == QZI_OK, "RGBA advanced edit generate");
    CHECK(is_png(&c), "RGBA png signature");
    CHECK(status == QZI_OK && c.data && c.size > 26 && c.data[25] == 6,
          "RGBA output uses PNG colour type 6");
    CHECK(status == QZI_OK && c.size > a.size, "RGBA output is larger than RGB");

    e.rgba = 0;
    e.model_mode = QZI_EDIT_MODE_LANPAINT_5;
    e.base.mask = mask;
    e.base.masking_strength = 1.0f;
    status = qzi_generate_qwen21_edit_ex(engine, &e, NULL, &d, error, error_cap);
    CHECK(status == QZI_OK, "LanPaint generate");
    CHECK(status != QZI_OK ||
          (a.size == d.size && memcmp(a.data, d.data, a.size) != 0),
          "LanPaint mode changes output bytes");

    qzi_image_release(engine, &a);
    qzi_image_release(engine, &b);
    qzi_image_release(engine, &c);
    qzi_image_release(engine, &d);
}

/* Z-Image ControlNet is the second additive ABI v1 extension. These requests
 * are rejected in qzi.c before the vtable is called, so they run on both
 * backends without model weights. Returns the number of rejected requests. */
#define EXPECT_CONTROL_INVALID(call, label) do {                       \
    if ((call) != QZI_ERR_INVALID_ARGUMENT) {                          \
        fprintf(stderr, "FAIL: control validation %s (%s)\n", label,  \
                error[0] ? error : "no error");                        \
        failures++;                                                    \
    }                                                                  \
    rejected++;                                                        \
} while (0)

static int check_control_validation(qzi_engine *engine,
                                    const qzi_zimage_request *base,
                                    char *error, size_t error_cap)
{
    static uint8_t rgb[3 * 64 * 64];
    static char long_type[64];
    qzi_zimage_control_request c;
    qzi_image image;
    int rejected = 0;

    memset(&image, 0, sizeof(image));
    memset(&c, 0, sizeof(c));
    memset(rgb, 0x22, sizeof(rgb));
    memset(long_type, 'x', sizeof(long_type) - 1);
    long_type[sizeof(long_type) - 1] = '\0';
    c.base = *base;
    c.control_rgb = rgb;
    c.control_width = 64;
    c.control_height = 64;
    c.control_type = "pose";
    c.control_scale = 0.0f;

    c.control_rgb = NULL;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "control_rgb required");
    c.control_rgb = rgb;
    c.control_width = 0;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "control dimensions");
    c.control_width = 4097;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "control side bound");
    c.control_height = 4096;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "control pixel budget");
    c.control_width = 64;
    c.control_height = 64;
    c.control_type = "hog";
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "unknown control type");
    c.control_type = long_type;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "overlong control type");
    c.control_type = "pose";
    c.control_scale = 0.01f;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "scale lower bound");
    c.control_scale = 2.1f;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "scale upper bound");
    {
        /* NaN must be rejected, not coerced; build the bit pattern without a
         * compile-time division by zero (MSVC warns on 0.0f / 0.0f). */
        uint32_t nan_bits = 0x7fc00000u;
        float nan_value = 0.0f;
        memcpy(&nan_value, &nan_bits, sizeof(nan_value));
        c.control_scale = nan_value;
    }
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "non-finite scale");
    c.control_scale = 0.65f;
    c.flags = 1;
    EXPECT_CONTROL_INVALID(qzi_generate_zimage_control(engine, &c, NULL, &image,
                                                       error, error_cap),
                           "reserved flags");
    c.flags = 0;
    c.base.checkpoint = "svdq-fp4-r256";
    if (qzi_generate_zimage_control(engine, &c, NULL, &image, error,
                                    error_cap) != QZI_ERR_UNSUPPORTED) {
        fprintf(stderr, "FAIL: control base checkpoint validation (%s)\n",
                error[0] ? error : "no error");
        failures++;
    }
    rejected++;
    c.base.checkpoint = "bf16";
    return rejected;
}

#undef EXPECT_CONTROL_INVALID

/* With the deterministic backend a ControlNet request must be observable in
 * the bytes: it differs from plain t2i, and the control image / scale / type
 * each change the result. Case-insensitive types ("PoSe" vs "pose") must be
 * byte-identical because qzi.c canonicalises them. Six successful generations
 * (one plain + five control), called only for `reference`. */
static void check_control_generation(qzi_engine *engine,
                                     const qzi_zimage_request *base,
                                     char *error, size_t error_cap)
{
    static uint8_t rgb[3 * 64 * 64];
    qzi_zimage_control_request c;
    qzi_callbacks callbacks;
    qzi_image p, x, y, z, w, v;
    qzi_status status;

    memset(&p, 0, sizeof(p)); memset(&x, 0, sizeof(x));
    memset(&y, 0, sizeof(y)); memset(&z, 0, sizeof(z));
    memset(&w, 0, sizeof(w)); memset(&v, 0, sizeof(v));
    memset(&callbacks, 0, sizeof(callbacks));
    memset(rgb, 0x22, sizeof(rgb));
    memset(&c, 0, sizeof(c));
    c.base = *base;
    c.control_rgb = rgb;
    c.control_width = 64;
    c.control_height = 64;
    c.control_type = "pose";
    c.control_scale = 0.0f;   /* ABI default -> effective 0.65 */
    callbacks.progress = on_progress;

    status = qzi_generate_zimage(engine, base, NULL, &p, error, error_cap);
    CHECK(status == QZI_OK, "control baseline t2i generate");

    image_encode_seen = 0;
    model_load_seen = 0;
    status = qzi_generate_zimage_control(engine, &c, &callbacks, &x, error,
                                         error_cap);
    CHECK(status == QZI_OK, "zimage control generate");
    if (status != QZI_OK) fprintf(stderr, "  control error: %s\n", error);
    CHECK(is_png(&x), "control png signature");
    CHECK(status != QZI_OK ||
          (p.size == x.size && memcmp(p.data, x.data, p.size) != 0),
          "control differs from plain t2i");
    CHECK(image_encode_seen == 2, "control reports image encode phase");
    CHECK(model_load_seen == 1, "control pack switch emits model load phase");

    c.control_type = "PoSe";
    status = qzi_generate_zimage_control(engine, &c, NULL, &y, error, error_cap);
    CHECK(status == QZI_OK, "zimage control uppercase type");
    CHECK(status != QZI_OK ||
          (x.size == y.size && memcmp(x.data, y.data, x.size) == 0),
          "control type is case-insensitive");

    memset(rgb, 0x99, sizeof(rgb));
    c.control_type = "pose";
    status = qzi_generate_zimage_control(engine, &c, NULL, &z, error, error_cap);
    CHECK(status == QZI_OK, "zimage control different image");
    CHECK(status != QZI_OK ||
          (x.size == z.size && memcmp(x.data, z.data, x.size) != 0),
          "control image changes output bytes");

    c.control_scale = 1.25f;
    status = qzi_generate_zimage_control(engine, &c, NULL, &w, error, error_cap);
    CHECK(status == QZI_OK, "zimage control scale");
    CHECK(status != QZI_OK ||
          (z.size == w.size && memcmp(z.data, w.data, z.size) != 0),
          "control scale changes output bytes");

    c.control_type = "canny";
    status = qzi_generate_zimage_control(engine, &c, NULL, &v, error, error_cap);
    CHECK(status == QZI_OK, "zimage control type");
    CHECK(status != QZI_OK ||
          (w.size == v.size && memcmp(w.data, v.data, w.size) != 0),
          "control type changes output bytes");

    qzi_image_release(engine, &p);
    qzi_image_release(engine, &x);
    qzi_image_release(engine, &y);
    qzi_image_release(engine, &z);
    qzi_image_release(engine, &w);
    qzi_image_release(engine, &v);
}

/* A backend built against the pre-edit qzi_backend.h has no edit slot.
 * Registering one must keep text-to-image intact, and the additive edit entry
 * point must answer QZI_ERR_UNSUPPORTED instead of calling a NULL pointer.
 * This is what keeps an old v1 plugin loadable by the new library. */
static qzi_status legacy_init(void **context, const qzi_engine_config *config,
                              char *error, size_t cap)
{
    (void)config; (void)error; (void)cap;
    *context = NULL;
    return QZI_OK;
}

static void legacy_shutdown(void *context) { (void)context; }

static qzi_status legacy_generate_qwen21(void *context,
                                          const qzi_qwen21_request *request,
                                          const qzi_callbacks *callbacks,
                                          qzi_image *image, char *error, size_t cap)
{
    (void)context; (void)request; (void)callbacks; (void)image;
    snprintf(error, cap, "legacy backend: generation not exercised here");
    return QZI_ERR_ENGINE;
}

static qzi_status legacy_generate_zimage(void *context,
                                         const qzi_zimage_request *request,
                                         const qzi_callbacks *callbacks,
                                         qzi_image *image, char *error, size_t cap)
{
    (void)context; (void)request; (void)callbacks; (void)image;
    snprintf(error, cap, "legacy backend: generation not exercised here");
    return QZI_ERR_ENGINE;
}

static qzi_status legacy_health(void *context, qzi_health *out,
                                char *error, size_t cap)
{
    (void)context; (void)out; (void)error; (void)cap;
    return QZI_OK;
}

static qzi_status legacy_unload(void *context, char *error, size_t cap)
{
    (void)context; (void)error; (void)cap;
    return QZI_OK;
}

static void legacy_release(void *context, qzi_image *image)
{
    (void)context; (void)image;
}

static void check_legacy_backend(const qzi_engine_config *config,
                                 const qzi_qwen21_request *base,
                                 char *error, size_t cap)
{
    static uint8_t rgb[3 * 64 * 64];
    qzi_backend legacy;
    qzi_qwen21_edit_request e;
    qzi_qwen21_edit_ex_request x;
    qzi_zimage_control_request c;
    qzi_zimage_request z;
    qzi_image image;
    qzi_engine *engine = NULL;
    qzi_status status;

    memset(&legacy, 0, sizeof(legacy));
    memset(&e, 0, sizeof(e));
    memset(&x, 0, sizeof(x));
    memset(&c, 0, sizeof(c));
    memset(&z, 0, sizeof(z));
    memset(&image, 0, sizeof(image));
    memset(rgb, 0x33, sizeof(rgb));
    z.prompt = "a red fox in snow";
    z.width = 64;
    z.height = 64;
    z.seed = 42;
    c.base = z;
    c.control_rgb = rgb;
    c.control_width = 64;
    c.control_height = 64;
    c.control_type = "pose";
    c.control_scale = 0.65f;
    legacy.abi_version = QZI_ABI_VERSION;
    legacy.name = "legacy-v1";
    legacy.init = legacy_init;
    legacy.shutdown = legacy_shutdown;
    legacy.generate_qwen21 = legacy_generate_qwen21;
    legacy.generate_zimage = legacy_generate_zimage;
    legacy.health = legacy_health;
    legacy.unload = legacy_unload;
    legacy.release_image = legacy_release;
    legacy.generate_qwen21_edit = NULL;
    legacy.generate_zimage_control = NULL;
    legacy.generate_qwen21_edit_ex = NULL;

    CHECK(qzi_register_backend(&legacy) == QZI_OK, "register legacy backend");
    status = qzi_engine_create(config, &engine, error, cap);
    CHECK(status == QZI_OK && engine != NULL, "legacy backend engine create");
    if (status == QZI_OK && engine) {
        e.base = *base;
        e.reference_rgb = rgb;
        e.reference_width = 64;
        e.reference_height = 64;
        e.strength = 0.5f;
        x.base = e;
        CHECK(qzi_generate_qwen21_edit(engine, &e, NULL, &image, error, cap) ==
                  QZI_ERR_UNSUPPORTED,
              "legacy backend rejects image editing");
        CHECK(qzi_generate_zimage_control(engine, &c, NULL, &image, error, cap) ==
                  QZI_ERR_UNSUPPORTED,
              "legacy backend rejects Z-Image ControlNet");
        CHECK(qzi_generate_qwen21_edit_ex(engine, &x, NULL, &image, error, cap) ==
                  QZI_ERR_UNSUPPORTED,
              "legacy backend rejects advanced image editing");
        qzi_engine_destroy(engine);
    }
    CHECK(qzi_register_backend(NULL) == QZI_OK, "restore default backend");
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
    CHECK((qzi_abi_features() & QZI_FEATURE_QWEN21_EDIT) != 0,
          "image editing feature advertised");
    CHECK((qzi_abi_features() & QZI_FEATURE_ZIMAGE_CONTROL) != 0,
          "Z-Image ControlNet feature advertised");
    CHECK((qzi_abi_features() & QZI_FEATURE_QWEN21_EDIT_ADV) != 0,
          "advanced image editing feature advertised");
    CHECK(qzi_qwen21_edit_layout() == sizeof(qzi_qwen21_edit_request),
          "edit request layout size");
    CHECK(qzi_zimage_control_layout() == sizeof(qzi_zimage_control_request),
          "control request layout size");
    CHECK(qzi_qwen21_edit_ex_layout() == sizeof(qzi_qwen21_edit_ex_request),
          "advanced edit request layout size");
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

        check_control_generation(engine, &zimage, error, sizeof(error));
        check_edit_generation(engine, &qwen, error, sizeof(error));
        check_edit_ex_generation(engine, &qwen, error, sizeof(error));
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
    extra_rejected += check_edit_validation(engine, &qwen, error,
                                            sizeof(error));
    extra_rejected += check_control_validation(engine, &zimage, error,
                                               sizeof(error));
    extra_rejected += check_edit_ex_validation(engine, &qwen, error,
                                               sizeof(error));

    CHECK(qzi_engine_health(engine, &health, error, sizeof(error)) == QZI_OK,
          "health");
    if (reference) {
        /* 2 baseline + 2 LoRA + 1 zimage + 6 control (1 plain t2i + 5
         * control) + 3 image-edit + 4 advanced-edit generations. */
        CHECK(health.served == 18, "served counter");
        CHECK(health.rejected == 2 + extra_rejected, "rejected counter");
        CHECK(health.reference_backend == 1, "reference backend flag");
        /* The last four generations are qwen21 advanced edits, so the resident
         * model is QWEN21 (not ZIMAGE from the earlier zimage job). */
        CHECK(health.resident_model == QZI_MODEL_QWEN21, "resident model");
    } else {
        CHECK(health.served == 0, "served counter");
        CHECK(health.rejected == 2 + extra_rejected, "rejected counter");
        CHECK(health.reference_backend == 0, "real backend flag");
    }
    CHECK(qzi_engine_unload(engine, error, sizeof(error)) == QZI_OK, "unload");

    qzi_engine_destroy(engine);

    /* Additive-ABI compatibility: a backend without the new edit slot must
     * stay loadable and report UNSUPPORTED rather than crashing. */
    check_legacy_backend(&config, &qwen, error, sizeof(error));

    if (failures) {
        fprintf(stderr, "qzi_abi_smoke: %d failure(s)\n", failures);
        return 1;
    }
    printf("qzi_abi_smoke: OK (%s)\n",
           reference ? "full generation contract"
                     : "ABI + validation only, real backend");
    return 0;
}
