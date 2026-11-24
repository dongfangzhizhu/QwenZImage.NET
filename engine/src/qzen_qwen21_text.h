#ifndef QZEN_QWEN21_TEXT_H
#define QZEN_QWEN21_TEXT_H

/* Qwen3-VL-8B language tower for Qwen-Image 2.1: CPU f32 reference forward plus
 * the t2i chat template the Wan2GP pipeline feeds it.
 *
 * Mirrors models/qwen21/text_layers.py (Qwen3VLTextDecoderLayer) and
 * models/qwen21/prompt.py: pre-norm RMSNorm, 32 query heads / 8 KV heads with
 * q_norm and k_norm over head_dim, rotate_half RoPE at theta 5e6, causal
 * attention at 1/sqrt(128). Text-only mRoPE degenerates to 1D RoPE because all
 * three position axes hold the same value.
 *
 * The Qwen-Image 2.1 DiT consumes the LAST decoder layer output BEFORE
 * model.norm (the pipeline captures it with a forward hook on text_model.norm),
 * and drops the 14-token system prefix of the chat template.
 */

#include <stddef.h>
#include <stdint.h>

#include "qzen_tokenizer.h"
#include "qzen_safetensors.h"

#define QZEN_QWEN21_TEXT_DIM 4096
#define QZEN_QWEN21_TEXT_LAYERS 36
#define QZEN_QWEN21_TEXT_DROP 14 /* system prefix tokens of the t2i template */

typedef struct qzen_qwen21_text qzen_qwen21_text;

qzen_qwen21_text *qzen_qwen21_text_open(const char *checkpoint, int thread_count,
                                    char *error, size_t error_size);
void qzen_qwen21_text_close(qzen_qwen21_text *model);

/* The t2i chat template around prompt; an empty prompt becomes one space, as the
 * pipeline does before formatting. */
int qzen_qwen21_text_prompt(const char *prompt, char *out, size_t size, char *error,
                          size_t error_size);

/* Template, tokenise, drop the 14-token system prefix. The caller releases *ids
 * with qzen_tokenizer_ids_free(). */
int qzen_qwen21_text_encode(const qzen_tokenizer *tokenizer, const char *prompt,
                          uint32_t **ids, size_t *count, char *error, size_t error_size);

/* ids -> hidden [count][QZEN_QWEN21_TEXT_DIM], the last decoder layer output before
 * model.norm. trace (may be NULL) receives [QZEN_QWEN21_TEXT_LAYERS][DIM], the last
 * token after each layer, for localising a mismatch. */
int qzen_qwen21_text_forward(qzen_qwen21_text *model, const uint32_t *ids, size_t count,
                           float *hidden, float *trace, char *error, size_t error_size);

/* Identity-reference template: one <imageN><|vision_start|> block per
 * reference, with pad_counts[i] consecutive <|image_pad|> slots. `pad_pos`
 * receives the trimmed-token index of every pad slot in encounter order. */
int qzen_qwen21_text_encode_vision(const qzen_tokenizer *tokenizer,
                                   const char *prompt, const int *pad_counts,
                                   int image_count, uint32_t **ids, size_t *count,
                                   int32_t **pad_pos, char *error, size_t error_size);

/* Same forward as qzen_qwen21_text_forward(), but overwrites the listed token
 * rows with `visual` ([pad_n][4096]) before layer 0 and adds `deep[i]` after
 * text layers i=0..2. Both CPU and device paths are supported. */
int qzen_qwen21_text_forward_vision(qzen_qwen21_text *model, const uint32_t *ids,
                                    size_t count, const float *visual,
                                    const int32_t *pad_pos, int pad_n,
                                    const float *const deep[3], float *hidden,
                                    char *error, size_t error_size);

/* Same as qzen_qwen21_text_forward_vision(), but also supplies the merged
 * image grid (grid_h[i] x grid_w[i]) for each reference. Wan2GP uses those
 * grids to assign Qwen3-VL 3D mRoPE positions to <|image_pad|> tokens; plain
 * sequential positions are only correct for text-only prompts. pad_counts,
 * grid_h, grid_w and image_count must describe the same pad_pos groups. */
int qzen_qwen21_text_forward_vision_ex(qzen_qwen21_text *model,
                                       const uint32_t *ids, size_t count,
                                       const float *visual,
                                       const int32_t *pad_pos, int pad_n,
                                       const float *const deep[3],
                                       const int *pad_counts,
                                       const int *grid_h, const int *grid_w,
                                       int image_count,
                                       float *hidden, char *error,
                                       size_t error_size);

/* Seconds spent moving layer weights: reading and dequantising them on the CPU
 * path, reading and uploading them on the device path. */
double qzen_qwen21_text_weight_seconds(const qzen_qwen21_text *model);

/* Borrowed checkpoint header, used by the visual tower to read visual.* from
 * the same safetensors file. The pointer stays valid until text_close(). */
qzen_st_header *qzen_qwen21_text_header(qzen_qwen21_text *model);

/* 1 when the last forward ran on the device (QWEN21_TEXT_GPU=1 and the CUDA
 * context came up), 0 while the f32 CPU reference is in charge. The two paths
 * do not agree to the same tolerance: the device keeps the weights INT8 and
 * quantises activations, which is a ~5e-3 cosine on the hidden states against
 * the torch reference, while the CPU path is at 1e-5. */
int qzen_qwen21_text_on_device(const qzen_qwen21_text *model);

#endif
