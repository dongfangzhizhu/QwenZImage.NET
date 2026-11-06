/* The POSIX side of the ICU surface qzen_tokenizer.c uses (see the headers
 * under platform/posix/include/unicode/ for what and why).
 *
 * A shipped Linux runtime must not carry a DT_NEEDED entry for a versioned
 * libicuuc.so.NN: Ubuntu 22.04 has .so.70, Debian 12 .so.72, newer distros
 * .so.7x, and a binary linked against any one of them fails to load on the
 * others. So we dlopen whatever ICU the host happens to have, resolve the six
 * C entry points by name (ICU renames them with a major-version suffix, e.g.
 * u_charType_72) and use a built-in fallback when there is no ICU at all.
 *
 * The fallback is deliberately conservative: exact UTF-8 <-> UTF-16, an
 * ASCII/CJK-oriented category + white-space table, and identity NFC. That
 * covers the prompts the bundled tokenizers see on an ICU-less host; on any
 * normal distro ICU is present and every call is the real thing. This is the
 * same "approximate the platform, never link a versioned Unicode library"
 * policy the Win32 shim implements with GetStringTypeW/NormalizeString.
 */
#include "unicode/uchar.h"
#include "unicode/unorm2.h"
#include "unicode/ustring.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Runtime ICU resolution
 * ------------------------------------------------------------------------ */

typedef int8_t (*qzen_char_type_fn)(UChar32);
typedef UBool (*qzen_is_white_fn)(UChar32);
typedef const UNormalizer2 *(*qzen_get_nfc_fn)(UErrorCode *);
typedef int32_t (*qzen_normalize_fn)(const UNormalizer2 *, const UChar *,
                                     int32_t, UChar *, int32_t, UErrorCode *);
typedef void (*qzen_from_utf8_fn)(UChar *, int32_t, int32_t *, const char *,
                                  int32_t, UErrorCode *);
typedef void (*qzen_to_utf8_fn)(char *, int32_t, int32_t *, const UChar *,
                                int32_t, UErrorCode *);

typedef struct {
    qzen_char_type_fn char_type;
    qzen_is_white_fn is_white;
    qzen_get_nfc_fn get_nfc;
    qzen_normalize_fn normalize;
    qzen_from_utf8_fn from_utf8;
    qzen_to_utf8_fn to_utf8;
} qzen_icu_api;

/* Zeroed == "no ICU, use the fallback". Installed in one shot by pthread_once,
 * so no reader can ever observe a half-filled table. */
static qzen_icu_api qzen_icu;
static pthread_once_t qzen_icu_once = PTHREAD_ONCE_INIT;

_Static_assert(sizeof(void *) == sizeof(qzen_char_type_fn),
               "POSIX dlsym requires object/function pointers of equal size");

static void *qzen_dlsym_icu(void *handle, const char *name) {
    char symbol[96];
    void *address = dlsym(handle, name); /* ICU built with renaming disabled */
    if (address) return address;
    /* Debian/Ubuntu/Fedora ship u_charType_72, u_strFromUTF8_72, ... */
    for (int major = 90; major >= 40; --major) {
        snprintf(symbol, sizeof(symbol), "%s_%d", name, major);
        address = dlsym(handle, symbol);
        if (address) return address;
    }
    return NULL;
}

static void *qzen_dlopen_icu(void) {
    const int flags = RTLD_LAZY | RTLD_LOCAL;
    char soname[64];
    void *handle = dlopen("libicuuc.so", flags); /* -dev package */
    if (!handle) handle = dlopen("libicuuc.dylib", flags);
    for (int major = 90; major >= 40 && !handle; --major) {
        snprintf(soname, sizeof(soname), "libicuuc.so.%d", major);
        handle = dlopen(soname, flags);
        if (handle) break;
        snprintf(soname, sizeof(soname), "libicuuc.%d.dylib", major);
        handle = dlopen(soname, flags);
    }
    return handle;
}

/* memcpy avoids the ISO C forbidding a void* -> function pointer cast; it is
 * exactly what dlsym's own documentation recommends. */
static void qzen_set_fn(void *slot, size_t size, void *address) {
    if (address) memcpy(slot, &address, size);
}

static void qzen_icu_init(void) {
    void *handle = qzen_dlopen_icu();
    if (!handle) return;
    qzen_icu_api found;
    memset(&found, 0, sizeof(found));
    qzen_set_fn(&found.char_type, sizeof(found.char_type),
                qzen_dlsym_icu(handle, "u_charType"));
    qzen_set_fn(&found.is_white, sizeof(found.is_white),
                qzen_dlsym_icu(handle, "u_isUWhiteSpace"));
    qzen_set_fn(&found.get_nfc, sizeof(found.get_nfc),
                qzen_dlsym_icu(handle, "unorm2_getNFCInstance"));
    qzen_set_fn(&found.normalize, sizeof(found.normalize),
                qzen_dlsym_icu(handle, "unorm2_normalize"));
    qzen_set_fn(&found.from_utf8, sizeof(found.from_utf8),
                qzen_dlsym_icu(handle, "u_strFromUTF8"));
    qzen_set_fn(&found.to_utf8, sizeof(found.to_utf8),
                qzen_dlsym_icu(handle, "u_strToUTF8"));
    /* All six or nothing: mixing real and fallback conversions would produce
     * inconsistent UTF-16 lengths. The handle stays open for the process. */
    if (found.char_type && found.is_white && found.get_nfc &&
        found.normalize && found.from_utf8 && found.to_utf8) {
        memcpy(&qzen_icu, &found, sizeof(qzen_icu));
    }
}

/* --------------------------------------------------------------------------
 * Fallback: Unicode property approximation
 * ------------------------------------------------------------------------ */

static int qzen_is_white_fallback(UChar32 c) {
    return c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D ||
           c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

/* Only qzen_letter() and qzen_number() consume the category, so the fallback
 * needs to agree with ICU on "is this a letter / a number / neither". The
 * ranges below cover ASCII, Latin/Greek/Cyrillic, CJK, kana, Hangul and the
 * fullwidth forms that the bundled tokenizers actually see. */
static int8_t qzen_char_type_fallback(UChar32 c) {
    if (c < 0) return (int8_t)U_UNASSIGNED;
    if (c < 0x80) {
        if (c >= 'A' && c <= 'Z') return (int8_t)U_UPPERCASE_LETTER;
        if (c >= 'a' && c <= 'z') return (int8_t)U_LOWERCASE_LETTER;
        if (c >= '0' && c <= '9') return (int8_t)U_DECIMAL_DIGIT_NUMBER;
        if (qzen_is_white_fallback(c)) return (int8_t)U_SPACE_SEPARATOR;
        if (c < 0x20 || c == 0x7F) return (int8_t)U_CONTROL_CHAR;
        return (int8_t)U_OTHER_PUNCTUATION;
    }
    if (c >= 0xD800 && c <= 0xDFFF) return (int8_t)U_SURROGATE;
    if (c >= 0xE000 && c <= 0xF8FF) return (int8_t)U_PRIVATE_USE_CHAR;
    if (qzen_is_white_fallback(c)) return (int8_t)U_SPACE_SEPARATOR;
    if ((c >= 0xFF10 && c <= 0xFF19) || (c >= 0x0660 && c <= 0x0669) ||
        (c >= 0x06F0 && c <= 0x06F9))
        return (int8_t)U_DECIMAL_DIGIT_NUMBER;
    if ((c >= 0x00C0 && c <= 0x02FF && c != 0x00D7 && c != 0x00F7) ||
        (c >= 0x0370 && c <= 0x1FFF) || (c >= 0x2C00 && c <= 0x2FEF) ||
        (c >= 0x3040 && c <= 0x30FF) || (c >= 0x3400 && c <= 0x4DBF) ||
        (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7A3) ||
        (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFF21 && c <= 0xFF3A) ||
        (c >= 0xFF41 && c <= 0xFF5A) || (c >= 0x20000 && c <= 0x3FFFD))
        return (int8_t)U_OTHER_LETTER;
    if ((c >= 0x2160 && c <= 0x2188)) return (int8_t)U_LETTER_NUMBER;
    if ((c >= 0x2000 && c <= 0x206F) || (c >= 0x3001 && c <= 0x303F) ||
        (c >= 0xFE10 && c <= 0xFE6F) || (c >= 0xFF01 && c <= 0xFF0F) ||
        (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) ||
        (c >= 0xFF5B && c <= 0xFF65))
        return (int8_t)U_OTHER_PUNCTUATION;
    return (int8_t)U_OTHER_SYMBOL;
}

/* --------------------------------------------------------------------------
 * Fallback: UTF-8 <-> UTF-16 (exact)
 * ------------------------------------------------------------------------ */

static int qzen_utf8_next(const unsigned char *src, int32_t length,
                          int32_t *offset, UChar32 *out) {
    const int32_t i = *offset;
    const unsigned char lead = src[i];
    int extra;
    UChar32 cp;
    if (lead < 0x80) { cp = lead; extra = 0; }
    else if ((lead & 0xE0) == 0xC0) { cp = lead & 0x1Fu; extra = 1; }
    else if ((lead & 0xF0) == 0xE0) { cp = lead & 0x0Fu; extra = 2; }
    else if ((lead & 0xF8) == 0xF0) { cp = lead & 0x07u; extra = 3; }
    else return 0;
    if (i + extra >= length) return 0;
    for (int k = 1; k <= extra; ++k) {
        const unsigned char tail = src[i + k];
        if ((tail & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (UChar32)(tail & 0x3Fu);
    }
    if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
        (extra == 3 && cp < 0x10000) || cp > 0x10FFFF ||
        (cp >= 0xD800 && cp <= 0xDFFF))
        return 0;
    *offset = i + extra + 1;
    *out = cp;
    return 1;
}

static int qzen_utf16_units(UChar32 cp, UChar out[2]) {
    if (cp <= 0xFFFF) { out[0] = (UChar)cp; return 1; }
    cp -= 0x10000;
    out[0] = (UChar)(0xD800 + (cp >> 10));
    out[1] = (UChar)(0xDC00 + (cp & 0x3FF));
    return 2;
}

static int qzen_utf8_units(UChar32 cp, unsigned char out[4]) {
    if (cp <= 0x7F) {
        out[0] = (unsigned char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        out[0] = (unsigned char)(0xC0 | (cp >> 6));
        out[1] = (unsigned char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        out[0] = (unsigned char)(0xE0 | (cp >> 12));
        out[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (unsigned char)(0xF0 | (cp >> 18));
    out[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

static int qzen_utf16_next(const UChar *src, int32_t length, int32_t *offset,
                           UChar32 *out) {
    const int32_t i = *offset;
    UChar32 cp = src[i];
    if (cp >= 0xD800 && cp <= 0xDBFF) {
        if (i + 1 >= length) return 0;
        const UChar32 low = src[i + 1];
        if (low < 0xDC00 || low > 0xDFFF) return 0;
        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
        *offset = i + 2;
    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
        return 0; /* unpaired low surrogate */
    } else {
        *offset = i + 1;
    }
    *out = cp;
    return 1;
}

/* --------------------------------------------------------------------------
 * Public ICU surface
 * ------------------------------------------------------------------------ */

int8_t u_charType(UChar32 value) {
    pthread_once(&qzen_icu_once, qzen_icu_init);
    if (qzen_icu.char_type) return qzen_icu.char_type(value);
    return qzen_char_type_fallback(value);
}

UBool u_isUWhiteSpace(UChar32 value) {
    pthread_once(&qzen_icu_once, qzen_icu_init);
    if (qzen_icu.is_white) return qzen_icu.is_white(value);
    return (UBool)qzen_is_white_fallback(value);
}

/* ICU hands out an opaque pointer the caller only passes straight back. With
 * real ICU we return its singleton; otherwise our own marker. Completing the
 * forward declaration from unorm2.h gives the marker somewhere to live. */
struct UNormalizer2 { int unused; };
static const struct UNormalizer2 qzen_fallback_nfc = {0};

const UNormalizer2 *unorm2_getNFCInstance(UErrorCode *status) {
    pthread_once(&qzen_icu_once, qzen_icu_init);
    if (qzen_icu.get_nfc) return qzen_icu.get_nfc(status);
    if (status) *status = U_ZERO_ERROR;
    return &qzen_fallback_nfc;
}

int32_t unorm2_normalize(const UNormalizer2 *normalizer, const UChar *src,
                         int32_t length, UChar *dest, int32_t capacity,
                         UErrorCode *status) {
    pthread_once(&qzen_icu_once, qzen_icu_init);
    if (qzen_icu.normalize)
        return qzen_icu.normalize(normalizer, src, length, dest, capacity,
                                  status);
    /* Identity NFC. Correct for already-composed text (ASCII and CJK, i.e.
     * virtually every prompt); decomposed Latin sequences are left as-is on
     * an ICU-less host rather than being mangled by a partial composition
     * table. */
    (void)normalizer;
    if (status) *status = U_ZERO_ERROR;
    if (!src || length < 0) {
        if (status) *status = U_INVALID_CHAR_FOUND;
        return 0;
    }
    if (!dest || capacity < length) {
        if (status) *status = U_BUFFER_OVERFLOW_ERROR;
        return length;
    }
    memcpy(dest, src, (size_t)length * sizeof(UChar));
    if (capacity > length) dest[length] = 0;
    return length;
}

void u_strFromUTF8(UChar *dest, int32_t destCapacity, int32_t *pDestLength,
                   const char *src, int32_t srcLength, UErrorCode *pErrorCode) {
    pthread_once(&qzen_icu_once, qzen_icu_init);
    if (qzen_icu.from_utf8) {
        qzen_icu.from_utf8(dest, destCapacity, pDestLength, src, srcLength,
                           pErrorCode);
        return;
    }
    if (pErrorCode) *pErrorCode = U_ZERO_ERROR;
    if (pDestLength) *pDestLength = 0;
    if (!src) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    if (srcLength < 0) srcLength = (int32_t)strlen(src);
    const unsigned char *bytes = (const unsigned char *)src;
    int32_t needed = 0, offset = 0;
    while (offset < srcLength) {
        UChar32 cp;
        if (!qzen_utf8_next(bytes, srcLength, &offset, &cp)) {
            if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
            if (pDestLength) *pDestLength = needed;
            return;
        }
        UChar units[2];
        needed += qzen_utf16_units(cp, units);
    }
    if (pDestLength) *pDestLength = needed;
    if (!dest || destCapacity < needed) {
        if (pErrorCode) *pErrorCode = U_BUFFER_OVERFLOW_ERROR;
        return;
    }
    offset = 0;
    int32_t written = 0;
    while (offset < srcLength) {
        UChar32 cp;
        qzen_utf8_next(bytes, srcLength, &offset, &cp); /* already validated */
        written += qzen_utf16_units(cp, dest + written);
    }
    if (destCapacity > written) dest[written] = 0;
}

void u_strToUTF8(char *dest, int32_t destCapacity, int32_t *pDestLength,
                 const UChar *src, int32_t srcLength, UErrorCode *pErrorCode) {
    pthread_once(&qzen_icu_once, qzen_icu_init);
    if (qzen_icu.to_utf8) {
        qzen_icu.to_utf8(dest, destCapacity, pDestLength, src, srcLength,
                         pErrorCode);
        return;
    }
    if (pErrorCode) *pErrorCode = U_ZERO_ERROR;
    if (pDestLength) *pDestLength = 0;
    if (!src || srcLength < 0) {
        if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
        return;
    }
    int32_t needed = 0, offset = 0;
    while (offset < srcLength) {
        UChar32 cp;
        if (!qzen_utf16_next(src, srcLength, &offset, &cp)) {
            if (pErrorCode) *pErrorCode = U_INVALID_CHAR_FOUND;
            if (pDestLength) *pDestLength = needed;
            return;
        }
        unsigned char units[4];
        needed += qzen_utf8_units(cp, units);
    }
    if (pDestLength) *pDestLength = needed;
    if (!dest || destCapacity < needed) {
        if (pErrorCode) *pErrorCode = U_BUFFER_OVERFLOW_ERROR;
        return;
    }
    offset = 0;
    int32_t written = 0;
    while (offset < srcLength) {
        UChar32 cp;
        qzen_utf16_next(src, srcLength, &offset, &cp); /* already validated */
        written += qzen_utf8_units(cp, (unsigned char *)dest + written);
    }
    if (destCapacity > written) dest[written] = 0;
}
