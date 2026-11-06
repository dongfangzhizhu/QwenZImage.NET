/* Minimal ICU surface shared by the POSIX and Win32 compatibility backends.
 *
 * qzen_tokenizer.c needs exactly six ICU calls: the general category of a
 * code point, the Unicode white-space property, NFC normalisation, and
 * UTF-8 <-> UTF-16 conversion. On POSIX these are resolved from whatever
 * libicuuc the host happens to ship (see qzen_posix_unicode.c), with a
 * built-in ASCII fallback when none is present. That keeps the shipped
 * libQwenZImageNative.so free of a versioned libicuuc.so.NN dependency.
 *
 * The names, types and enum values mirror ICU 7x so the engine source stays
 * byte-identical across platforms. */
#ifndef QZEN_POSIX_ICU_TYPES_H
#define QZEN_POSIX_ICU_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int8_t UBool;
typedef uint16_t UChar;
typedef int32_t UChar32;
typedef int32_t UErrorCode;

#define U_ZERO_ERROR 0
#define U_BUFFER_OVERFLOW_ERROR 15
#define U_INVALID_CHAR_FOUND 10

#define U_SUCCESS(code) ((code) <= 0)
#define U_FAILURE(code) ((code) > 0)

typedef enum {
    U_UNASSIGNED = 0,
    U_UPPERCASE_LETTER = 1,
    U_LOWERCASE_LETTER = 2,
    U_TITLECASE_LETTER = 3,
    U_MODIFIER_LETTER = 4,
    U_OTHER_LETTER = 5,
    U_NON_SPACING_MARK = 6,
    U_ENCLOSING_MARK = 7,
    U_COMBINING_SPACING_MARK = 8,
    U_DECIMAL_DIGIT_NUMBER = 9,
    U_LETTER_NUMBER = 10,
    U_OTHER_NUMBER = 11,
    U_SPACE_SEPARATOR = 12,
    U_LINE_SEPARATOR = 13,
    U_PARAGRAPH_SEPARATOR = 14,
    U_CONTROL_CHAR = 15,
    U_FORMAT_CHAR = 16,
    U_PRIVATE_USE_CHAR = 17,
    U_SURROGATE = 18,
    U_DASH_PUNCTUATION = 19,
    U_START_PUNCTUATION = 20,
    U_END_PUNCTUATION = 21,
    U_CONNECTOR_PUNCTUATION = 22,
    U_OTHER_PUNCTUATION = 23,
    U_MATH_SYMBOL = 24,
    U_CURRENCY_SYMBOL = 25,
    U_MODIFIER_SYMBOL = 26,
    U_OTHER_SYMBOL = 27,
    U_INITIAL_PUNCTUATION = 28,
    U_FINAL_PUNCTUATION = 29
} UCharCategory;

#ifdef __cplusplus
}
#endif

#endif /* QZEN_POSIX_ICU_TYPES_H */
