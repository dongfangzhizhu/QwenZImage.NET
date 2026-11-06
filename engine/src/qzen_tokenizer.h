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

/* The caller owns the returned UTF-8 string and releases it with free(). */
char *qzen_tokenizer_decode(const qzen_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size);

#endif
