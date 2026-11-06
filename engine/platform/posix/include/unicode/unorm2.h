/* ICU <unicode/unorm2.h> subset: the NFC normaliser the tokenizer asks for.
 * ICU convention: dest may be NULL with capacity 0 to ask for the length. */
#ifndef QZEN_POSIX_ICU_UNORM2_H
#define QZEN_POSIX_ICU_UNORM2_H

#include "unicode/utypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct UNormalizer2 UNormalizer2;

const UNormalizer2 *unorm2_getNFCInstance(UErrorCode *status);

int32_t unorm2_normalize(const UNormalizer2 *normalizer, const UChar *src,
                         int32_t length, UChar *dest, int32_t capacity,
                         UErrorCode *status);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_POSIX_ICU_UNORM2_H */
