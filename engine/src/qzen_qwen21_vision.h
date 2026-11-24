#ifndef QZEN_QWEN21_VISION_H
#define QZEN_QWEN21_VISION_H

/* Qwen3-VL-8B visual tower used by Qwen-Image 2.1 image references.
 *
 * Mirrors models/qwen21/text_layers.py::Qwen3VLVisionModel and
 * models/qwen21/processor.py: patchify 32-aligned RGB into 16x16 temporal
 * patches, add interpolated pos_embed, run 27 axial-RoPE vision blocks, and
 * produce the final merger plus three deepstack mergers. The merged rows are
 * written into the language tower's <|image_pad|> slots; deepstack rows are
 * added back after language layers 0/1/2.
 *
 * The tower borrows the Qwen3-VL safetensors header from the language tower.
 * No ABI struct or exported symbol changes: this is an engine-internal
 * component used only by the identity-reference backend path.
 */

#include <stddef.h>
#include <stdint.h>

#include "qzen_safetensors.h"

#define QZEN_QWEN21_VIS_HIDDEN 1152
#define QZEN_QWEN21_VIS_OUT 4096
#define QZEN_QWEN21_VIS_PATCH 16
#define QZEN_QWEN21_VIS_MERGE 2
#define QZEN_QWEN21_VIS_ALIGN (QZEN_QWEN21_VIS_PATCH * QZEN_QWEN21_VIS_MERGE)
#define QZEN_QWEN21_VIS_DEEPSTACK 3

typedef struct qzen_qwen21_vision qzen_qwen21_vision;

/* `header` stays owned by the caller (the language tower) and must outlive
 * the vision model. When QWEN21_TEXT_GPU=1 the tower creates its own short
 * lived CUDA context for the forward; the context is freed by close(). */
qzen_qwen21_vision *qzen_qwen21_vision_open(qzen_st_header *header,
                                            int thread_count,
                                            char *error, size_t error_size);
void qzen_qwen21_vision_close(qzen_qwen21_vision *vision);

/* Each reference is planar RGB8 [3][width*height]. Width/height must be
 * multiples of 32. On success:
 *   *merged   = [total_pads][4096] F32 (caller frees)
 *   *deep[i]  = [total_pads][4096] F32 (caller frees, i=0..2)
 *   *pad_count= total number of <|image_pad|> slots across all images
 * total_pads per image = (width/16)*(height/16)/4. */
int qzen_qwen21_vision_forward(qzen_qwen21_vision *vision,
                               const uint8_t *const *rgb,
                               const uint32_t *widths,
                               const uint32_t *heights,
                               int count,
                               float **merged,
                               float *deep[QZEN_QWEN21_VIS_DEEPSTACK],
                               int *pad_count,
                               char *error, size_t error_size);

#endif
