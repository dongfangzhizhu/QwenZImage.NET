#ifndef QZEN_TOKENIZER_H
#define QZEN_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define QZEN_PAD_TOKEN_ID UINT32_C(151643)

typedef struct qzen_tokenizer qzen_tokenizer;

qzen_tokenizer *qzen_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size);
void qzen_tokenizer_free(qzen_tokenizer *tokenizer);

/* The caller owns *ids and releases it with qzen_tokenizer_ids_free(). */
int qzen_tokenizer_encode(const qzen_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size);
void qzen_tokenizer_ids_free(uint32_t *ids);

/* Look up an added/special token id (for example "<|image_pad|>"). */
int qzen_tokenizer_token_id(const qzen_tokenizer *tokenizer, const char *token,
                            uint32_t *id);

/* Encode and additionally report which output positions hold `marker`.
 * *slots has *count entries; non-marker positions are -1, marker positions
 * are the zero-based indices into *ids. The caller frees both arrays:
 * ids with qzen_tokenizer_ids_free(), slots with free(). */
int qzen_tokenizer_encode_spans(const qzen_tokenizer *tokenizer, const char *utf8,
                                const char *marker, uint32_t **ids, int32_t **slots,
                                size_t *count, char *error, size_t error_size);

/* The caller owns the returned UTF-8 string and releases it with free(). */
char *qzen_tokenizer_decode(const qzen_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size);

#endif
