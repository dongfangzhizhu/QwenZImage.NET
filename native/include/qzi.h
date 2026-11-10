/*
 * qzi.h -- QwenZImage.NET stable C ABI (version 1).
 *
 * This header is the contract between a native image engine (C/CUDA) and the
 * QwenZImage.NET managed wrapper. It deliberately contains NO engine-specific
 * types: any implementation that exports these functions can be loaded by the
 * NuGet package. QwenZImage.NET ships a complete reference implementation of
 * this ABI in native/src/qzi.c with a pluggable backend (see qzi_backend.h).
 *
 * Design rules, all of them load-bearing for P/Invoke:
 *   1. Plain C linkage, no C++ exceptions crossing the boundary, no STL.
 *   2. Every struct is a POD with a fixed layout; qzi_abi_layout() reports the
 *      sizeof() of each struct so the managed side can assert its own layout.
 *   3. Caller owns all input memory. Strings are NUL-terminated UTF-8 and are
 *      only read during the call.
 *   4. The engine owns the output image buffer. qzi_image_release() must be
 *      called exactly once for every successful qzi_image.
 *   5. Every fallible call returns qzi_status and, on failure, writes a
 *      NUL-terminated human-readable message into the caller's error buffer.
 *   6. /generate is synchronous and single-flight: a second concurrent call
 *      returns QZI_ERR_BUSY. It may block for minutes.
 *   7. Phase strings passed to the progress callback are static, NUL-terminated
 *      literals owned by the engine. They must not be freed.
 */

#ifndef QZI_H
#define QZI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(QZI_BUILDING_DLL)
#    define QZI_API __declspec(dllexport)
#  else
#    define QZI_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define QZI_API __attribute__((visibility("default")))
#else
#  define QZI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QZI_ABI_VERSION 1u
#define QZI_ERROR_CAPACITY 512u
#define QZI_LAYOUT_SLOTS 6u

/* Capability bits reported by qzi_abi_features(). A v1 library built before
 * image editing / ControlNet does not export that symbol at all; callers should
 * probe for the export first (NativeLibrary.TryGetExport on .NET) and treat its
 * absence as "no optional features". Both bits are additive: QZI_ABI_VERSION
 * stays 1 and every pre-existing struct and export keeps its exact layout. */
#define QZI_FEATURE_QWEN21_EDIT    0x1u
#define QZI_FEATURE_ZIMAGE_CONTROL 0x2u

/* Maximum byte length of a LoRA adapter file name (excluding the NUL).
 * Both sides of the ABI reject longer names instead of truncating them. */
#define QZI_LORA_NAME_MAX 255u

/* Layout slot indices reported by qzi_abi_layout(). */
#define QZI_LAYOUT_ENGINE_CONFIG  0u
#define QZI_LAYOUT_QWEN21_REQUEST 1u
#define QZI_LAYOUT_ZIMAGE_REQUEST 2u
#define QZI_LAYOUT_CALLBACKS      3u
#define QZI_LAYOUT_IMAGE          4u
#define QZI_LAYOUT_HEALTH         5u

typedef enum qzi_status {
    QZI_OK                   = 0,
    QZI_ERR_INVALID_ARGUMENT = 1,  /* malformed request / failed validation   */
    QZI_ERR_NOT_CONFIGURED   = 2,  /* model pack missing / engine not created */
    QZI_ERR_UNSUPPORTED      = 3,  /* tier, checkpoint or feature unavailable */
    QZI_ERR_BUSY             = 4,  /* another generation owns the device     */
    QZI_ERR_CANCELLED        = 5,  /* cancellation callback asked to stop    */
    QZI_ERR_OUT_OF_MEMORY    = 6,
    QZI_ERR_ENGINE           = 7,  /* model load, denoise, VAE, device error */
    QZI_ERR_INTERNAL         = 8
} qzi_status;

typedef enum qzi_model_kind {
    QZI_MODEL_NONE    = 0,
    QZI_MODEL_QWEN21  = 1,
    QZI_MODEL_ZIMAGE  = 2
} qzi_model_kind;

/* Opaque engine instance. Created by qzi_engine_create, freed by
 * qzi_engine_destroy. Not thread-safe for concurrent generation; the ABI
 * serialises generation with QZI_ERR_BUSY by design (one GPU, one resident
 * model). */
typedef struct qzi_engine qzi_engine;

typedef struct qzi_engine_config {
    uint32_t     abi_version;   /* must be QZI_ABI_VERSION */
    const char  *qwen21_dir;    /* NUL UTF-8, may be NULL */
    const char  *zimage_dir;    /* NUL UTF-8, may be NULL */
    const char  *cache_dir;     /* NUL UTF-8, may be NULL */
    int32_t      device_id;     /* -1 = auto */
    int32_t      thread_count;  /*  0 = auto */
    uint32_t     flags;         /* reserved, must be 0 */
} qzi_engine_config;

/* Qwen-Image 2.1 text-to-image request. Rules mirror the vendored engine:
 * width/height are multiples of 32 in [32, 2048], w*h <= 2048*2048;
 * tier "turbo" forces steps == 8 and guidance == 1.0 and rejects
 * negative_prompt; "base" allows steps 1..50, guidance 1..20.
 * A zero steps / guidance means "use the tier default" (8/1.0 or 40/4.0);
 * qzi_generate_qwen21() resolves those defaults before the backend sees the
 * request, so both backends observe the same effective values.
 *
 * lora is a bare adapter file name resolved inside qwen21_lora_dir: at most
 * QZI_LORA_NAME_MAX bytes, containing no '/', '\\', ':', '..' or control
 * characters. LoRA works on both tiers. lora_scale == 0.0f means "engine
 * default" (1.0) and is the only value allowed without lora; any other value
 * requires lora, must be finite and must lie in [0.05, 4]. Invalid values are
 * rejected, never silently coerced. */
typedef struct qzi_qwen21_request {
    const char  *prompt;           /* required, non-empty, <= 16384 UTF-8 bytes */
    const char  *negative_prompt;  /* NULL or UTF-8; base tier only */
    const char  *tier;             /* NULL = "turbo", or "turbo" / "base" */
    const char  *lora;             /* NULL or a bare adapter file name */
    int32_t      width;
    int32_t      height;
    int32_t      steps;
    float        guidance;
    float        lora_scale;
    uint64_t     seed;
    uint32_t     flags;            /* reserved, must be 0 */
} qzi_qwen21_request;

/* Qwen-Image 2.1 image editing (img2img) request. Added in ABI v1 as a
 * purely additive extension: qzi_qwen21_request and every existing export keep
 * their exact layout, so a v1 library that predates editing still serves t2i
 * callers. Use qzi_abi_features() to test for QZI_FEATURE_QWEN21_EDIT before
 * calling qzi_generate_qwen21_edit().
 *
 * The reference image is planar RGB8: channel 0 occupies ref_w*ref_h bytes,
 * then channel 1, then channel 2. It is resampled to the output canvas with a
 * centre-cropped cover fit before VAE encoding. `mask` is optional, one byte
 * per reference pixel (255 = regenerate this pixel, 0 = keep the reference);
 * it is downsampled to the 1/16 latent grid and thresholded at 64, exactly like
 * WanGP's input_masks. Editable area is the WHITE part of the mask.
 *
 * `strength` is WanGP's denoising_strength: 1.0 starts from pure noise (a
 * text-to-image result), smaller values start closer to the reference. 0.0 is
 * the "use the engine default" sentinel (0.75). `masking_strength` is WanGP's
 * fraction of the remaining schedule during which unmasked latent cells are
 * restored to the reference trajectory; it requires `mask` and defaults to 1.0.
 * Both are validated, never silently coerced. */
typedef struct qzi_qwen21_edit_request {
    qzi_qwen21_request base;             /* the ordinary t2i request */
    const uint8_t     *reference_rgb;    /* required: [3][ref_w*ref_h] planar */
    uint32_t           reference_width;
    uint32_t           reference_height;
    const uint8_t     *mask;             /* NULL or [ref_w*ref_h] 8-bit */
    float              strength;         /* 0 sentinel, else 0.05..1.0 */
    float              masking_strength; /* 0 sentinel, else 0.05..1.0 */
    uint32_t           flags;            /* reserved, must be 0 */
} qzi_qwen21_edit_request;

/* Z-Image (Turbo / Nunchaku) text-to-image request. Rules mirror the engine:
 * width/height are multiples of 16 in [16, 2048], w*h <= 2048*2048;
 * checkpoint is NULL/"bf16" (default), "nunchaku", or a pack id such as
 * "nunchaku-r256". Turbo uses a fixed 8-step schedule; there is no CFG
 * branch, so no negative prompt and no steps/guidance fields. */
typedef struct qzi_zimage_request {
    const char  *prompt;       /* required, non-empty, <= 16384 UTF-8 bytes */
    const char  *checkpoint;   /* NULL/"bf16"/"nunchaku"/<pack id> */
    int32_t      width;
    int32_t      height;
    uint64_t     seed;
    uint32_t     flags;        /* reserved, must be 0 */
} qzi_zimage_request;

/* Z-Image Turbo Fun ControlNet Union (WanGP z_image_control2_1). Additive in
 * ABI v1, exactly like qzi_qwen21_edit_request: no existing struct or export
 * changes. Use qzi_abi_features() to test QZI_FEATURE_ZIMAGE_CONTROL before
 * calling qzi_generate_zimage_control().
 *
 * The control image is planar RGB8 (channel 0 occupies cw*ch bytes, then 1
 * and 2). It is resampled to the output canvas with a centre-cropped cover
 * fit, VAE-encoded to a 16-channel 1/8 latent, padded to the 33-channel
 * control context (control latent | zero keep-mask | zero inpaint latent) and
 * patchified with the DiT's 2x2 patch embedder. WanGP's "Union" weights are
 * trained on preprocessed control images (pose / canny / depth / scribble);
 * `control_type` is descriptive metadata only, "raw" means the caller already
 * supplies a Z-Image-format control image. This library does NOT run an edge /
 * pose / depth estimator -- doing that upstream keeps the C ABI dependency
 * free and matches WanGP's guide_preprocessing "V" path.
 *
 * `control_scale` is WanGP's control_context_scale (v2.1 default 0.65, v1/v2
 * 0.75). 0.0f is the "use engine default" sentinel; any other value must be
 * finite and lie in [0.05, 2.0]. Invalid values are rejected, never coerced. */
typedef struct qzi_zimage_control_request {
    qzi_zimage_request base;              /* the ordinary t2i request */
    const uint8_t     *control_rgb;       /* required: [3][cw*ch] planar */
    uint32_t           control_width;
    uint32_t           control_height;
    const char        *control_type;      /* NULL/"auto"/"pose"/"canny"/"depth"/"scribble"/"raw" */
    float              control_scale;     /* 0 sentinel, else 0.05..2.0 */
    uint32_t           flags;             /* reserved, must be 0 */
} qzi_zimage_control_request;

/* Progress callback. `phase` is a static literal: "model load", "text",
 * "image encode" (image edits / ControlNet: VAE-encoding the reference or
 * control image), "dit"
 * (completed/total = denoise steps actually re-run), "vae", "encode".
 * The callback runs on the generation thread; it must be fast and must not
 * call back into the engine. */
typedef void (*qzi_progress_fn)(const char *phase, int32_t completed,
                                int32_t total, double elapsed_seconds,
                                void *user);

/* Cancellation callback. The engine polls it between phases/steps; a non-zero
 * return aborts the generation with QZI_ERR_CANCELLED. May be NULL. Runs on
 * the generation thread. */
typedef int (*qzi_cancel_fn)(void *user);

typedef struct qzi_callbacks {
    qzi_progress_fn  progress;
    qzi_cancel_fn    should_cancel;
    void            *user;       /* passed verbatim to both callbacks */
} qzi_callbacks;

/* Output image. `data` points to a complete PNG file (image/png) owned by the
 * engine until qzi_image_release() is called with the same `token`. */
typedef struct qzi_image {
    const uint8_t *data;
    size_t         size;
    uint32_t       width;
    uint32_t       height;
    uint64_t       token;
} qzi_image;

typedef struct qzi_health {
    qzi_model_kind  resident_model;
    int32_t         busy;
    int32_t         queue_depth;
    int32_t         device_id;
    int64_t         free_vram_mib;
    int64_t         total_vram_mib;
    uint64_t        served;
    uint64_t        rejected;
    double          uptime_seconds;
    int32_t         reference_backend; /* 1 = deterministic reference backend */
} qzi_health;

/* ---- ABI surface ------------------------------------------------------- */

QZI_API uint32_t    qzi_abi_version(void);
QZI_API const char *qzi_status_text(qzi_status status);

/* Additive capability query. QZI_ABI_VERSION stays 1: no existing struct or
 * export changed, so this is the only way to distinguish an editing-capable
 * v1 library from an older one. The bit describes what this LIBRARY implements,
 * not the registered backend: a custom backend with a NULL edit slot still
 * answers QZI_ERR_UNSUPPORTED from qzi_generate_qwen21_edit(). */
QZI_API uint32_t    qzi_abi_features(void);

/* sizeof(qzi_qwen21_edit_request) / sizeof(qzi_zimage_control_request), so a
 * managed wrapper can reject a struct-packing mismatch before passing a
 * pointer across the ABI. Additive in v1: qzi_abi_layout() keeps its original
 * six slots and old libraries do not export these symbols at all. */
QZI_API size_t      qzi_qwen21_edit_layout(void);
QZI_API size_t      qzi_zimage_control_layout(void);

/* Reports sizeof() for the six POD structs above. Used by the managed wrapper
 * to fail fast on a struct-packing mismatch instead of corrupting memory. */
QZI_API qzi_status  qzi_abi_layout(size_t *sizes, size_t count);

/* Registers (or replaces) the generation backend. Must be called before
 * qzi_engine_create. NULL restores the built-in reference backend. The
 * pointed-to vtable is copied. See qzi_backend.h. */
struct qzi_backend;
QZI_API qzi_status  qzi_register_backend(const struct qzi_backend *backend);

QZI_API qzi_status  qzi_engine_create(const qzi_engine_config *config,
                                      qzi_engine **out_engine,
                                      char *error, size_t error_cap);
QZI_API void        qzi_engine_destroy(qzi_engine *engine);

QZI_API qzi_status  qzi_engine_health(qzi_engine *engine, qzi_health *out,
                                      char *error, size_t error_cap);
QZI_API qzi_status  qzi_engine_unload(qzi_engine *engine,
                                      char *error, size_t error_cap);

QZI_API qzi_status  qzi_generate_qwen21(qzi_engine *engine,
                                         const qzi_qwen21_request *request,
                                         const qzi_callbacks *callbacks,
                                         qzi_image *out_image,
                                         char *error, size_t error_cap);
QZI_API qzi_status  qzi_generate_zimage(qzi_engine *engine,
                                        const qzi_zimage_request *request,
                                        const qzi_callbacks *callbacks,
                                        qzi_image *out_image,
                                        char *error, size_t error_cap);

/* Qwen-Image 2.1 image editing. Same single-flight, progress, cancellation
 * and buffer-ownership rules as qzi_generate_qwen21(). Returns
 * QZI_ERR_UNSUPPORTED when the registered backend has no editing slot.
 * The reference/mask buffers are only read during the call and remain owned
 * by the caller. */
QZI_API qzi_status  qzi_generate_qwen21_edit(qzi_engine *engine,
                                             const qzi_qwen21_edit_request *request,
                                             const qzi_callbacks *callbacks,
                                             qzi_image *out_image,
                                             char *error, size_t error_cap);

/* Z-Image Turbo ControlNet Union generation. Same single-flight, progress,
 * cancellation and buffer-ownership rules as qzi_generate_zimage(). Returns
 * QZI_ERR_UNSUPPORTED when the registered backend has no control slot, and
 * QZI_ERR_NOT_CONFIGURED when the real backend has no ControlNet weights.
 * The control_rgb buffer is only read during the call and remains owned by
 * the caller. */
QZI_API qzi_status  qzi_generate_zimage_control(qzi_engine *engine,
                                                const qzi_zimage_control_request *request,
                                                const qzi_callbacks *callbacks,
                                                qzi_image *out_image,
                                                char *error, size_t error_cap);

/* Releases the buffer behind out_image and zeroes the struct. Idempotent for
 * a zeroed image; calling twice with the same token is undefined. */
QZI_API void        qzi_image_release(qzi_engine *engine, qzi_image *image);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QZI_H */
