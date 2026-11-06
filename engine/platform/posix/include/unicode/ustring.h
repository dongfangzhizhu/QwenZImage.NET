/* ICU <unicode/ustring.h> subset: the two UTF-8 <-> UTF-16 calls the
 * tokenizer makes. srcLength < 0 means NUL-terminated, as in ICU. */
#ifndef QZEN_POSIX_ICU_USTRING_H
#define QZEN_POSIX_ICU_USTRING_H

#include "unicode/utypes.h"

#ifdef __cplusplus
extern "C" {
#endif

void u_strFromUTF8(UChar *dest, int32_t destCapacity, int32_t *pDestLength,
                   const char *src, int32_t srcLength, UErrorCode *pErrorCode);

void u_strToUTF8(char *dest, int32_t destCapacity, int32_t *pDestLength,
                 const UChar *src, int32_t srcLength, UErrorCode *pErrorCode);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_POSIX_ICU_USTRING_H */
