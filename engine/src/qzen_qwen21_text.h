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

/* Seconds spent moving layer weights: reading and dequantising them on the CPU
 * path, reading and uploading them on the device path. */
double qzen_qwen21_text_weight_seconds(const qzen_qwen21_text *model);

/* 1 when the last forward ran on the device (QWEN21_TEXT_GPU=1 and the CUDA
 * context came up), 0 while the f32 CPU reference is in charge. The two paths
 * do not agree to the same tolerance: the device keeps the weights INT8 and
 * quantises activations, which is a ~5e-3 cosine on the hidden states against
 * the torch reference, while the CPU path is at 1e-5. */
int qzen_qwen21_text_on_device(const qzen_qwen21_text *model);

#endif
