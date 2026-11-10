#ifndef QZEN_ZIMAGE_CONTROL_H
#define QZEN_ZIMAGE_CONTROL_H

/* Z-Image Turbo Fun ControlNet Union (v2.1) weight store.
 *
 * Reference: D:/AI/Wan2GP/models/z_image/z_image_transformer2d.py and
 * configs/z_image_control2_1.json. The checkpoint is a single safetensors
 * file whose tensors are named exactly like the main DiT's, prefixed with
 * "control_":
 *
 *   control_all_x_embedder.2-1.weight   BF16 [3840, 132]  (33 * 2 * 2)
 *   control_layers.{0..14}.*            quanto I8 + BF16 norms/bias
 *   control_noise_refiner.{0..1}.*      plain BF16
 *   control_layers.0.before_proj.*      (block_id 0 only)
 *   control_noise_refiner.0.before_proj.*
 *   control_layers.N.after_proj.*       (one per control layer)
 *   control_noise_refiner.N.after_proj.*
 *
 * There are no pad tokens in this file: the ControlNet reuses the main DiT's
 * x_pad_token / cap_pad_token, which is why the hint computation lives in
 * qzen_zimage_pipeline.c and only the weights live here.
 *
 * The 15 control layers map to main DiT layers 0,2,4,...,28
 * (control_layers_places); the 2 dedicated noise refiners map 1:1 to the main
 * noise_refiner.0/1 blocks (use_separate_control_refiner = true in v2.1).
 */

#include "qzen_zimage_dit.h"

#include <stddef.h>
#include <stdint.h>

#define QZEN_ZIMAGE_CONTROL_LAYERS   15u
#define QZEN_ZIMAGE_CONTROL_REFINERS  2u
#define QZEN_ZIMAGE_CONTROL_IN_DIM_V2 33u
#define QZEN_ZIMAGE_CONTROL_IN_DIM_V1 16u

typedef struct qzen_zimage_control qzen_zimage_control;

/* Open a single .safetensors file (or a directory of shards) and load every
 * ControlNet tensor as host BF16/F32. `in_dim` is derived from the embedder's
 * shape (132 / 4 = 33 for v2.x, 64 / 4 = 16 for v1) so v1 and v2.x are
 * distinguished by the weights, not by a guessed config. */
qzen_zimage_control *qzen_zimage_control_load(const char *path,
                                             const qzen_zimage_config *config,
                                             char *error, size_t error_size);
void qzen_zimage_control_free(qzen_zimage_control *control);

uint32_t qzen_zimage_control_in_dim(const qzen_zimage_control *control);
uint32_t qzen_zimage_control_layers(const qzen_zimage_control *control);
uint32_t qzen_zimage_control_refiners(const qzen_zimage_control *control);

const qzen_zimage_linear *qzen_zimage_control_embedder(const qzen_zimage_control *control);
qzen_zimage_block *qzen_zimage_control_layer(qzen_zimage_control *control,
                                             uint32_t index);
qzen_zimage_block *qzen_zimage_control_refiner(qzen_zimage_control *control,
                                               uint32_t index);

/* NULL for every index except 0 (block_id 0 owns before_proj). `refiner`
 * selects the dedicated control noise refiner when non-zero. */
const qzen_zimage_linear *qzen_zimage_control_before_proj(
    const qzen_zimage_control *control, int refiner, uint32_t index);
const qzen_zimage_linear *qzen_zimage_control_after_proj(
    const qzen_zimage_control *control, int refiner, uint32_t index);

/* Upload the blocks and projections to the Z-Image GPU context. Optional:
 * when it fails the caller can still run the hint path on the CPU. Returns 1
 * on success, 0 with an error message otherwise (and leaves the host weights
 * intact). */
int qzen_zimage_control_upload(qzen_zimage_control *control,
                               qzen_zimage_gpu *zgu,
                               char *error, size_t error_size);
int qzen_zimage_control_uploaded(const qzen_zimage_control *control);

#endif
