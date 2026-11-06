/* ICU <unicode/uchar.h> subset needed by qzen_tokenizer.c on POSIX. */
#ifndef QZEN_POSIX_ICU_UCHAR_H
#define QZEN_POSIX_ICU_UCHAR_H

#include "unicode/utypes.h"

#ifdef __cplusplus
extern "C" {
#endif

int8_t u_charType(UChar32 value);
UBool u_isUWhiteSpace(UChar32 value);

#ifdef __cplusplus
}
#endif

#endif /* QZEN_POSIX_ICU_UCHAR_H */
