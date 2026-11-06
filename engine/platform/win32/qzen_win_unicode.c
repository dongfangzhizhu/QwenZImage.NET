/* The Win32 side of the ICU surface qzen_tokenizer.c uses (see
 * platform/win32/include/unicode/*.h for what and why).
 *
 * Everything here is a thin call onto the platform: NormalizeString for NFC,
 * MultiByteToWideChar / WideCharToMultiByte for the UTF-8 conversions, and
 * GetStringTypeW for the character properties.  No Unicode tables are shipped.
 */
#include "unicode/uchar.h"
#include "unicode/unorm2.h"
#include "unicode/ustring.h"

#include <windows.h>
#include <wchar.h>

#include <stdlib.h>
#include <string.h>

/* One code point as UTF-16, without the terminator.  Returns the unit count. */
static int to_utf16(UChar32 value, WCHAR units[2]) {
    if (value < 0 || value > 0x10FFFF) return 0;
    if (value <= 0xFFFF) {
        if (value >= 0xD800 && value <= 0xDFFF) return 0; /* lone surrogate */
        units[0] = (WCHAR)value;
        return 1;
    }
    value -= 0x10000;
    units[0] = (WCHAR)(0xD800 + (value >> 10));
    units[1] = (WCHAR)(0xDC00 + (value & 0x3FF));
    return 2;
}

int8_t u_charType(UChar32 value) {
    WCHAR units[2];
    WORD type = 0;
    const int count = to_utf16(value, units);
    if (count == 0) return (int8_t)U_UNASSIGNED;
    if (!GetStringTypeW(CT_CTYPE1, units, count, &type))
        return (int8_t)U_UNASSIGNED;
    if (type & C1_ALPHA) {
        if (iswlower((wint_t)value)) return (int8_t)U_LOWERCASE_LETTER;
        if (iswupper((wint_t)value)) return (int8_t)U_UPPERCASE_LETTER;
        return (int8_t)U_OTHER_LETTER;
    }
    if (type & C1_DIGIT) return (int8_t)U_DECIMAL_DIGIT_NUMBER;
    if (type & C1_SPACE) return (int8_t)U_SPACE_SEPARATOR;
    if (type & C1_PUNCT) return (int8_t)U_OTHER_PUNCTUATION;
    return (int8_t)U_OTHER_SYMBOL;
}

UBool u_isUWhiteSpace(UChar32 value) {
    WCHAR units[2];
    WORD type = 0;
    const int count = to_utf16(value, units);
    if (count == 0) return 0;
    if (!GetStringTypeW(CT_CTYPE1, units, count, &type)) return 0;
    return (type & C1_SPACE) != 0;
}

/* --------------------------------------------------------------------------
 * NFC
 * ------------------------------------------------------------------------ */

/* ICU hands out a pointer to an opaque normaliser that the caller only ever
 * passes straight back to unorm2_normalize.  Completing the type here (the
 * header only forward-declares it) gives the singleton somewhere to live. */
struct UNormalizer2 {
    int unused;
};

static const struct UNormalizer2 nfc_singleton = {0};
static const UNormalizer2 *const nfc_instance = &nfc_singleton;

const UNormalizer2 *unorm2_getNFCInstance(UErrorCode *status) {
    if (status) *status = U_ZERO_ERROR;
    return nfc_instance;
}

int32_t unorm2_normalize(const UNormalizer2 *normalizer, const UChar *src,
                         int32_t length, UChar *dest, int32_t capacity,
                         UErrorCode *status) {
    (void)normalizer;
    if (status) *status = U_ZERO_ERROR;
    if (!src || length < 0) {
        if (status) *status = U_INVALID_CHAR_FOUND;
        return 0;
    }
    const int needed = NormalizeString(NormalizationC, (LPCWSTR)src, length,
                                       NULL, 0);
    if (needed <= 0) {
        if (status) *status = U_INVALID_CHAR_FOUND;
        return 0;
    }
    if (!dest || capacity < needed) {
        if (status) *status = U_BUFFER_OVERFLOW_ERROR;
        return needed;
    }
    const int written = NormalizeString(NormalizationC, (LPCWSTR)src, length,
                                        (LPWSTR)dest, capacity);
    if (written <= 0) {
        if (status) *status = U_INVALID_CHAR_FOUND;
        return 0;
    }
    if (capacity > written) dest[written] = 0;
    return written;
}

/* --------------------------------------------------------------------------
 * UTF-8 <-> UTF-16
 * ------------------------------------------------------------------------ */

void u_strFromUTF8(UChar *dest, int32_t destCapacity, int32_t *pDestLength,
                   const char *src, int32_t srcLength, UErrorCode *pErrorCode) {
    if (pErrorCode) *pErrorCode = U_ZERO_ERROR;
    if (pDestLength) *pDestLength = 0;
    if (!src) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (srcLength < 0) srcLength = (int32_t)strlen(src);
    if (srcLength == 0) {
        if (dest && destCapacity > 0) dest[0] = 0;
        return;
    }
    const int needed = MultiByteToWideChar(CP_UTF8, 0, src, srcLength, NULL, 0);
    if (needed <= 0) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (pDestLength) *pDestLength = needed;
    if (!dest || destCapacity < needed) {
        if (pErrorCode) *pErrorCode = U_BUFFER_OVERFLOW_ERROR;
        return;
    }
    if (MultiByteToWideChar(CP_UTF8, 0, src, srcLength, (LPWSTR)dest,
                            destCapacity) != needed) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (destCapacity > needed) dest[needed] = 0;
}

void u_strToUTF8(char *dest, int32_t destCapacity, int32_t *pDestLength,
                 const UChar *src, int32_t srcLength, UErrorCode *pErrorCode) {
    if (pErrorCode) *pErrorCode = U_ZERO_ERROR;
    if (pDestLength) *pDestLength = 0;
    if (!src || srcLength < 0) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (srcLength == 0) {
        if (dest && destCapacity > 0) dest[0] = 0;
        return;
    }
    const int needed = WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)src, srcLength,
                                           NULL, 0, NULL, NULL);
    if (needed <= 0) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (pDestLength) *pDestLength = needed;
    if (!dest || destCapacity < needed) {
        if (pErrorCode) *pErrorCode = U_BUFFER_OVERFLOW_ERROR;
        return;
    }
    if (WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)src, srcLength, dest,
                            destCapacity, NULL, NULL) != needed) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (destCapacity > needed) dest[needed] = 0;
}
