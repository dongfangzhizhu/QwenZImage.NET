/* ICU <unicode/ustring.h> subset: the two UTF-8 <-> UTF-16 calls the tokenizer
 * makes.  Backed by MultiByteToWideChar / WideCharToMultiByte (CP_UTF8), which
 * are the same conversions ICU's fast path performs. */
#ifndef QZEN_WIN_ICU_USTRING_H
#define QZEN_WIN_ICU_USTRING_H

#include "unicode/utypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* srcLength < 0 means NUL-terminated, as in ICU.  dest may be NULL with
 * destCapacity 0 to ask for the required length. */
void u_strFromUTF8(UChar *dest, int32_t destCapacity, int32_t *pDestLength,
                   const char *src, int32_t srcLength, UErrorCode *pErrorCode);

void u_strToUTF8(char *dest, int32_t destCapacity, int32_t *pDestLength,
                 const UChar *src, int32_t srcLength, UErrorCode *pErrorCode);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_ICU_USTRING_H */
