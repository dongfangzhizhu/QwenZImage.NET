/* ICU <unicode/uchar.h> subset, implemented on Win32 (see utypes.h). */
#ifndef QZEN_WIN_ICU_UCHAR_H
#define QZEN_WIN_ICU_UCHAR_H

#include "unicode/utypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* General category of one code point.  Backed by GetStringTypeW: the C1_ALPHA
 * and C1_DIGIT bits decide letter vs number, iswupper/iswlower picks the case,
 * and everything else falls into the punctuation/symbol buckets the tokenizer
 * only ever compares against for "not a letter, not a number". */
int8_t u_charType(UChar32 value);

/* Unicode White_Space.  GetStringTypeW's C1_SPACE covers the ASCII and
 * Latin-1 spaces plus the Unicode separators Windows knows about; the engine
 * adds U+001C..U+001F itself. */
UBool u_isUWhiteSpace(UChar32 value);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_WIN_ICU_UCHAR_H */
