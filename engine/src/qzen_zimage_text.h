#ifndef QZEN_ZIMAGE_TEXT_H
#define QZEN_ZIMAGE_TEXT_H

#include "qzen_tokenizer.h"
#include "qzen_zimage_dit.h"

#include <stddef.h>
#include <stdint.h>

/* Qwen3-4B text encoder for Z-Image Turbo (Wan2GP models/z_image, T5 replaced
 * by Qwen3ForCausalLM used as a feature extractor).
 *
 * Pipeline stage (pipeline_z_image.py _encode_prompt):
 *
 *   formatted = chat_template([{role: user, content: prompt}],
 *                             add_generation_prompt=True, enable_thinking=True)
 *   ids, mask = tokenizer(formatted, padding="max_length",
 *                         max_length=512, truncation=True)
 *   cap_feats = model(input_ids, attention_mask, output_hidden_states=True)
 *                   .hidden_states[-2][mask]      # real tokens only
 *
 * so the consumed features are the LAST LAYER output taken BEFORE the final
 * RMSNorm (hidden_states[-2]), restricted to the un-padded positions.
 *
 * Architecture (ckpts/Qwen3/config.json): 36 layers, hidden 2560, 32 Q heads x
 * 128 / 8 KV heads x 128 (GQA), FFN SwiGLU 9728, RMSNorm eps 1e-6 everywhere
 * (including the per-head q_norm / k_norm over head_dim), RoPE theta 1e6 in
 * GPT-NeoX pair layout (i, i + head_dim/2), causal attention, no biases,
 * tie_word_embeddings (embed_tokens doubles as lm_head, unused here).
 *
 * Weights are the quanto qint8 text checkpoint (Wan2GP quanto_bf16_int8):
 * every Linear is a "<name>.weight._data" I8 [out, in] + "<name>.weight._scale"
 * BF16 [out, 1] pair dequantized to BF16 at load via qzen_quanto; norms and
 * embed_tokens are plain BF16.  Forward math is F32 with BF16 weights widened
 * per element, matching the Phase 0 CPU reference path.  The GEMM is OpenMP
 * parallel (link targets with -fopenmp).
 *
 * Padding: the tokenizer right-pads, and causal attention means padded rows
 * never influence real rows, so callers pass the number of real tokens as seq
 * and the padded tail is never computed.  qzen_zpipe_encode_text relies on this:
 * the prompt is tokenized at max_length (the tokenizer contract) but the
 * forward runs real_len positions, which is bit-identical for the rows it
 * returns and skips 500 of the 512 rows of GEMM work.
 */

typedef struct {
    uint32_t layers;          /* 36 */
    uint32_t dim;             /* 2560 */
    uint32_t heads;           /* 32 */
    uint32_t kv_heads;        /* 8 */
    uint32_t head_dim;        /* 128 */
    uint32_t ffn_dim;         /* 9728 */
    uint32_t vocab;           /* 151936 */
    float norm_eps;           /* 1e-6 (input/post-attention/q/k norms) */
    float rope_theta;         /* 1e6 */
    uint32_t max_length;      /* 512, tokenizer padding="max_length" */
} qzen_ztext_config;

qzen_ztext_config qzen_ztext_default_config(void);

typedef struct qzen_ztext_model qzen_ztext_model;

/* Load all 36 layers from a quanto qint8 Qwen3 checkpoint (~7.8 GiB resident
 * BF16 after dequantization).  The store is opened over the directory holding
 * qwen3_quanto_bf16_int8.safetensors.  The progress callback, when non-NULL,
 * is invoked after each loaded layer (index, total). */
qzen_ztext_model *qzen_ztext_model_load(const qzen_weight_store *store,
                                    const qzen_ztext_config *config,
                                    void (*progress)(uint32_t layer,
                                                     uint32_t total),
                                    char *error, size_t error_size);
void qzen_ztext_model_free(qzen_ztext_model *model);

/* Qwen3 chat template for a single user message with a generation prompt
 * (enable_thinking=True leaves the assistant prefix bare):
 *
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 *
 * Returns a malloc-ed NUL-terminated string, released with free(). */
char *qzen_ztext_format_prompt(const char *prompt);

/* Encode the chat-formatted prompt with the Qwen3 BPE tokenizer (tokenizer.json,
 * special tokens included) and right-pad with <|endoftext|> (151643) to
 * config->max_length.  ids and mask are caller buffers of max_length
 * int32/uint8; real_len receives the number of un-padded tokens.  The
 * tokenizer is qzen_tokenizer.h's opaque handle over Qwen3 tokenizer.json. */
int qzen_ztext_build_inputs(const qzen_ztext_config *config,
                          const qzen_tokenizer *tokenizer,
                          const char *prompt, int32_t *ids, uint8_t *mask,
                          uint32_t *real_len,
                          char *error, size_t error_size);

/* Decoder forward over ids[seq].  out is [seq, dim] F32 receiving the last
 * processed layer output BEFORE the final RMSNorm (all layers when
 * max_layers == 0, otherwise the first max_layers layers -- used by the
 * two-layer numerics gate).  The model keeps no state between calls. */
int qzen_ztext_forward(const qzen_ztext_model *model, const int32_t *ids,
                     uint32_t seq, uint32_t max_layers, float *out,
                     char *error, size_t error_size);

#endif
