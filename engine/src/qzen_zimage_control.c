/* Z-Image Turbo Fun ControlNet Union 2.1 weight loader.
 *
 * The checkpoint's tensors are named exactly like the main DiT's, prefixed
 * with "control_", so qzen_zimage_block_load() can load both the quantized
 * "control_layers.N.*" group and the plain-BF16 "control_noise_refiner.N.*"
 * group unchanged: that loader auto-detects the quanto qint8 (weight._data /
 * weight._scale) pairs vs. plain BF16.  Only the projections and the single
 * feature embedder need dedicated handling here. */

#include "qzen_zimage_control.h"

#include "qzen_weights.h"
#include "qzen_safetensors.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qzen_zimage_control {
    /* calloc() leaves every field zero, so _free() is safe on a partially
     * loaded state: unloaded Linears are all-NULL and unloaded blocks are
     * NULL pointers. */
    qzen_zimage_linear embedder;

    uint32_t in_dim;      /* derived from the embedder width */
    uint32_t layers;      /* control_layers blocks actually loaded */
    uint32_t refiners;    /* control_noise_refiner blocks actually loaded */
    int uploaded;

    qzen_zimage_block *layer_blocks[QZEN_ZIMAGE_CONTROL_LAYERS];
    qzen_zimage_block *refiner_blocks[QZEN_ZIMAGE_CONTROL_REFINERS];

    qzen_zimage_linear layer_after[QZEN_ZIMAGE_CONTROL_LAYERS];
    qzen_zimage_linear layer_before[QZEN_ZIMAGE_CONTROL_LAYERS];
    int layer_has_before[QZEN_ZIMAGE_CONTROL_LAYERS];

    qzen_zimage_linear refiner_after[QZEN_ZIMAGE_CONTROL_REFINERS];
    qzen_zimage_linear refiner_before[QZEN_ZIMAGE_CONTROL_REFINERS];
    int refiner_has_before[QZEN_ZIMAGE_CONTROL_REFINERS];
};

static void ctl_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

qzen_zimage_control *qzen_zimage_control_load(const char *path,
                                             const qzen_zimage_config *config,
                                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !config) {
        ctl_fail(error, error_size,
                 "ControlNet load needs a checkpoint path and a DiT config");
        return NULL;
    }

    qzen_zimage_control *control = calloc(1, sizeof(*control));
    if (!control) {
        ctl_fail(error, error_size, "cannot allocate Z-Image ControlNet state");
        return NULL;
    }

    qzen_weight_store *store = NULL;
    const qzen_st_header *embed_header = NULL;
    const qzen_st_tensor *embed = NULL;
    uint32_t in_dim = 0;
    char name[224];
    char probe[224];
    char probe_quant[224];

    store = qzen_weight_store_open(path, error, error_size);
    if (!store) goto failure;

    /* The embedder's second dimension is 4 * in_dim.  Deriving in_dim from the
     * weight (132 / 4 = 33 for v2.x, 64 / 4 = 16 for v1) tells the variants
     * apart from the checkpoint itself instead of a guessed config value. */
    embed = qzen_weight_find(store, "control_all_x_embedder.2-1.weight",
                             &embed_header);
    if (!embed) {
        ctl_fail(error, error_size,
                 "checkpoint lacks control_all_x_embedder.2-1.weight; this is "
                 "not a Z-Image Turbo Fun ControlNet Union file");
        goto failure;
    }
    if (embed->ndim != 2 || embed->shape[0] != config->dim ||
        embed->shape[1] % 4 != 0) {
        ctl_fail(error, error_size,
                 "control_all_x_embedder.2-1.weight must be BF16[dim, in_dim*4] "
                 "with dim=%u: got %s ndim=%d shape=[%llu, %llu]",
                 config->dim, qzen_dtype_name(embed->dtype), embed->ndim,
                 (unsigned long long)embed->shape[0],
                 (unsigned long long)(embed->ndim > 1 ? embed->shape[1] : 0));
        goto failure;
    }
    in_dim = (uint32_t)(embed->shape[1] / 4);
    if (in_dim != QZEN_ZIMAGE_CONTROL_IN_DIM_V1 &&
        in_dim != QZEN_ZIMAGE_CONTROL_IN_DIM_V2) {
        ctl_fail(error, error_size,
                 "unsupported ControlNet variant: embedder in_dim=%u (weight "
                 "[dim, %llu]); expected %u for v1 or %u for v2.x",
                 in_dim, (unsigned long long)embed->shape[1],
                 QZEN_ZIMAGE_CONTROL_IN_DIM_V1,
                 QZEN_ZIMAGE_CONTROL_IN_DIM_V2);
        goto failure;
    }
    control->in_dim = in_dim;
    /* This engine implements the v2.x hint path (dedicated control noise
     * refiners + 15 control layers). A v1 Union checkpoint has no
     * control_noise_refiner.* tensors, so say so now instead of failing later
     * with a confusing missing-tensor message. */
    if (in_dim == QZEN_ZIMAGE_CONTROL_IN_DIM_V1) {
        ctl_fail(error, error_size,
                 "this is a ControlNet Union v1 checkpoint (in_dim=16); only "
                 "the v2/v2.1 Union checkpoints (in_dim=33, with dedicated "
                 "control noise refiners) are supported. Use e.g. "
                 "Z-Image-Turbo-Fun-Controlnet-Union2.1_*.safetensors");
        goto failure;
    }

    /* Guard the layer count up front: a different ControlNet release has a
     * different number of control_layers, and block_load's per-tensor errors
     * are far less actionable than "wrong variant".  Accept either the plain
     * BF16 or the quanto qint8 spelling of the last layer's to_q weight. */
    snprintf(probe, sizeof(probe), "control_layers.%u.attention.to_q.weight",
             QZEN_ZIMAGE_CONTROL_LAYERS - 1);
    snprintf(probe_quant, sizeof(probe_quant), "%s._data", probe);
    if (!qzen_weight_find(store, probe, NULL) &&
        !qzen_weight_find(store, probe_quant, NULL)) {
        ctl_fail(error, error_size,
                 "ControlNet checkpoint has no control_layers.%u.* tensors; "
                 "expected the %u-layer Z-Image Turbo Fun ControlNet Union 2.1 "
                 "(looked for %s or %s). The file is a different variant.",
                 QZEN_ZIMAGE_CONTROL_LAYERS - 1, QZEN_ZIMAGE_CONTROL_LAYERS,
                 probe, probe_quant);
        goto failure;
    }

    if (!qzen_zimage_linear_load(store, "control_all_x_embedder.2-1.weight",
                                 config->dim, in_dim * 4, 1, &control->embedder,
                                 error, error_size))
        goto failure;

    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_LAYERS; i++) {
        control->layer_blocks[i] = qzen_zimage_block_load(
            store, "control_layers", i, config, error, error_size);
        if (!control->layer_blocks[i]) goto failure;
        control->layers++;
    }
    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_REFINERS; i++) {
        control->refiner_blocks[i] = qzen_zimage_block_load(
            store, "control_noise_refiner", i, config, error, error_size);
        if (!control->refiner_blocks[i]) goto failure;
        control->refiners++;
    }

    /* control_layers projections are quanto qint8; the dedicated refiners are
     * plain BF16.  Every projection is [dim, dim] with a bias. */
    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_LAYERS; i++) {
        snprintf(name, sizeof(name), "control_layers.%u.after_proj.weight", i);
        if (!qzen_zimage_linear_load_quanto(store, name, config->dim,
                                            config->dim, 1,
                                            &control->layer_after[i],
                                            error, error_size))
            goto failure;
    }
    snprintf(name, sizeof(name), "control_layers.0.before_proj.weight");
    if (!qzen_zimage_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                        &control->layer_before[0],
                                        error, error_size))
        goto failure;
    control->layer_has_before[0] = 1;

    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_REFINERS; i++) {
        snprintf(name, sizeof(name),
                 "control_noise_refiner.%u.after_proj.weight", i);
        if (!qzen_zimage_linear_load(store, name, config->dim, config->dim, 1,
                                     &control->refiner_after[i],
                                     error, error_size))
            goto failure;
    }
    snprintf(name, sizeof(name), "control_noise_refiner.0.before_proj.weight");
    if (!qzen_zimage_linear_load(store, name, config->dim, config->dim, 1,
                                 &control->refiner_before[0],
                                 error, error_size))
        goto failure;
    control->refiner_has_before[0] = 1;

    qzen_weight_store_free(store);
    return control;

failure:
    if (store) qzen_weight_store_free(store);
    qzen_zimage_control_free(control);
    return NULL;
}

void qzen_zimage_control_free(qzen_zimage_control *control) {
    if (!control) return;
    qzen_zimage_linear_free(&control->embedder);
    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_LAYERS; i++) {
        qzen_zimage_block_free(control->layer_blocks[i]);
        qzen_zimage_linear_free(&control->layer_after[i]);
        qzen_zimage_linear_free(&control->layer_before[i]);
    }
    for (uint32_t i = 0; i < QZEN_ZIMAGE_CONTROL_REFINERS; i++) {
        qzen_zimage_block_free(control->refiner_blocks[i]);
        qzen_zimage_linear_free(&control->refiner_after[i]);
        qzen_zimage_linear_free(&control->refiner_before[i]);
    }
    free(control);
}

uint32_t qzen_zimage_control_in_dim(const qzen_zimage_control *control) {
    return control ? control->in_dim : 0;
}

uint32_t qzen_zimage_control_layers(const qzen_zimage_control *control) {
    return control ? control->layers : 0;
}

uint32_t qzen_zimage_control_refiners(const qzen_zimage_control *control) {
    return control ? control->refiners : 0;
}

const qzen_zimage_linear *qzen_zimage_control_embedder(
    const qzen_zimage_control *control) {
    return control ? &control->embedder : NULL;
}

qzen_zimage_block *qzen_zimage_control_layer(qzen_zimage_control *control,
                                             uint32_t index) {
    if (!control || index >= control->layers) return NULL;
    return control->layer_blocks[index];
}

qzen_zimage_block *qzen_zimage_control_refiner(qzen_zimage_control *control,
                                               uint32_t index) {
    if (!control || index >= control->refiners) return NULL;
    return control->refiner_blocks[index];
}

const qzen_zimage_linear *qzen_zimage_control_before_proj(
    const qzen_zimage_control *control, int refiner, uint32_t index) {
    if (!control) return NULL;
    if (refiner) {
        if (index >= control->refiners ||
            !control->refiner_has_before[index])
            return NULL;
        return &control->refiner_before[index];
    }
    if (index >= control->layers || !control->layer_has_before[index])
        return NULL;
    return &control->layer_before[index];
}

const qzen_zimage_linear *qzen_zimage_control_after_proj(
    const qzen_zimage_control *control, int refiner, uint32_t index) {
    if (!control) return NULL;
    if (refiner) {
        if (index >= control->refiners) return NULL;
        return &control->refiner_after[index];
    }
    if (index >= control->layers) return NULL;
    return &control->layer_after[index];
}

int qzen_zimage_control_upload(qzen_zimage_control *control,
                               qzen_zimage_gpu *zgu,
                               char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!control || !zgu) {
        ctl_fail(error, error_size,
                 "ControlNet upload needs a loaded control state and a GPU context");
        return 0;
    }

    if (!qzen_zimage_linear_upload(&control->embedder, zgu,
                                   error, error_size))
        goto failure;

    for (uint32_t i = 0; i < control->layers; i++) {
        if (!qzen_zimage_block_upload(control->layer_blocks[i], zgu,
                                      error, error_size))
            goto failure;
    }
    for (uint32_t i = 0; i < control->refiners; i++) {
        if (!qzen_zimage_block_upload(control->refiner_blocks[i], zgu,
                                      error, error_size))
            goto failure;
    }

    /* Upload every projection that was actually loaded.  after_proj exists for
     * all 17 blocks; before_proj only for index 0 of each group (the flag
     * gates it), so no empty/zeroed Linear is ever handed to the uploader. */
    for (uint32_t i = 0; i < control->layers; i++) {
        if (!qzen_zimage_linear_upload(&control->layer_after[i], zgu,
                                       error, error_size))
            goto failure;
        if (control->layer_has_before[i] &&
            !qzen_zimage_linear_upload(&control->layer_before[i], zgu,
                                       error, error_size))
            goto failure;
    }
    for (uint32_t i = 0; i < control->refiners; i++) {
        if (!qzen_zimage_linear_upload(&control->refiner_after[i], zgu,
                                       error, error_size))
            goto failure;
        if (control->refiner_has_before[i] &&
            !qzen_zimage_linear_upload(&control->refiner_before[i], zgu,
                                       error, error_size))
            goto failure;
    }

    control->uploaded = 1;
    return 1;

failure:
    /* Host BF16 weights are never touched by the uploaders, so they stay
     * intact for the CPU fallback even after a partial GPU upload. */
    control->uploaded = 0;
    return 0;
}

int qzen_zimage_control_uploaded(const qzen_zimage_control *control) {
    return control ? control->uploaded : 0;
}
