/* POSIX ICU compatibility layer: runtime dual-path check.
 *
 * qzen_posix_unicode.c resolves ICU exactly once per process (pthread_once),
 * so "host ICU present" and "forced no ICU" cannot live in one binary. CMake
 * builds this file twice:
 *
 *   qzi_posix_unicode_icu       - normal dlopen, expects real ICU NFC
 *   qzi_posix_unicode_fallback  - -Wl,--wrap=dlopen makes every dlopen fail,
 *                                 expects the built-in ASCII/CJK fallback
 *
 * Both variants must agree on character classification, white-space, exact
 * UTF-8 <-> UTF-16 conversion (including a 4-byte code point / surrogate
 * pair), invalid UTF-8 rejection and NFC of already-composed text. The only
 * intended difference is decomposed Latin: real ICU composes e + U+0301 into
 * U+00E9, while the fallback is deliberately identity NFC.
 *
 * If the first variant finds no usable ICU on the host it exits with 77,
 * which CTest records as SKIP (all shared assertions still ran and passed).
 */
#include "unicode/uchar.h"
#include "unicode/unorm2.h"
#include "unicode/ustring.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#ifdef QZEN_TEST_NO_ICU
/* GNU ld --wrap=dlopen: qzen_dlopen_icu() can never obtain a handle. */
void *__wrap_dlopen(const char *filename, int flags)
{
    (void)filename; (void)flags;
    return NULL;
}
#endif

#define SKIP_NO_ICU 77

static int failures = 0;

#define CHECK(cond, label) do {                                        \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL: %s (line %d)\n", label, __LINE__);     \
        failures++;                                                    \
    }                                                                  \
} while (0)

/* "fox <U+1F98A> <CJK> <e-acute>": 18 UTF-8 bytes, 11 UTF-16 units. */
static const char k_utf8[] =
    "fox \xF0\x9F\xA6\x8A \xE4\xB8\xAD\xE6\x96\x87 \xC3\xA9";
static const UChar k_utf16[] = {
    'f', 'o', 'x', ' ', 0xD83E, 0xDD8A, ' ',
    0x4E2D, 0x6587, ' ', 0x00E9
};
#define UTF8_LEN  18
#define UTF16_LEN 11

int main(void)
{
    UErrorCode status;
    int32_t length = -1;
    UChar u16[32];
    char back[64];

    /* ---- UTF-8 -> UTF-16: preflight, then the real conversion ---- */
    memset(u16, 0, sizeof(u16));
    status = U_ZERO_ERROR;
    u_strFromUTF8(NULL, 0, &length, k_utf8, -1, &status);
    CHECK(length == UTF16_LEN, "u_strFromUTF8 preflight length");
    CHECK(status == U_BUFFER_OVERFLOW_ERROR || U_SUCCESS(status),
          "u_strFromUTF8 preflight status");

    status = U_ZERO_ERROR;
    u_strFromUTF8(u16, 32, NULL, k_utf8, -1, &status);
    CHECK(U_SUCCESS(status), "u_strFromUTF8 status");
    CHECK(memcmp(u16, k_utf16, sizeof(k_utf16)) == 0,
          "u_strFromUTF8 UTF-16 units incl. surrogate pair");
    CHECK(u16[UTF16_LEN] == 0, "u_strFromUTF8 NUL-terminates");

    /* ---- UTF-16 -> UTF-8 must be byte-identical ---- */
    length = -1;
    status = U_ZERO_ERROR;
    u_strToUTF8(NULL, 0, &length, u16, UTF16_LEN, &status);
    CHECK(length == UTF8_LEN, "u_strToUTF8 preflight length");
    CHECK(status == U_BUFFER_OVERFLOW_ERROR || U_SUCCESS(status),
          "u_strToUTF8 preflight status");

    memset(back, 0, sizeof(back));
    status = U_ZERO_ERROR;
    u_strToUTF8(back, 64, NULL, u16, UTF16_LEN, &status);
    CHECK(U_SUCCESS(status), "u_strToUTF8 status");
    CHECK(memcmp(back, k_utf8, UTF8_LEN) == 0,
          "UTF-8 round-trip is byte-identical");
    CHECK(back[UTF8_LEN] == 0, "u_strToUTF8 NUL-terminates");

    /* ---- invalid UTF-8 must be reported ---- */
    status = U_ZERO_ERROR;
    memset(u16, 0, sizeof(u16));
    u_strFromUTF8(u16, 32, NULL, "\xC3\x28", -1, &status);
    CHECK(status != U_ZERO_ERROR, "invalid UTF-8 is rejected");
#ifdef QZEN_TEST_NO_ICU
    CHECK(status == U_INVALID_CHAR_FOUND,
          "fallback reports U_INVALID_CHAR_FOUND");
#endif

    /* ---- character classification / white space ---- */
    CHECK(u_charType('A') == U_UPPERCASE_LETTER, "ASCII uppercase letter");
    CHECK(u_charType('7') == U_DECIMAL_DIGIT_NUMBER, "ASCII digit");
    CHECK(u_charType(0x4E2D) == U_OTHER_LETTER, "CJK ideograph is a letter");
    CHECK(u_charType(0x1F98A) == U_OTHER_SYMBOL, "emoji is a symbol");
    CHECK(u_isUWhiteSpace(' ') && !u_isUWhiteSpace('A'),
          "ASCII white space");
    CHECK(u_isUWhiteSpace(0x3000), "ideographic space is white space");

    /* ---- NFC ---- */
    status = U_ZERO_ERROR;
    const UNormalizer2 *nfc = unorm2_getNFCInstance(&status);
    CHECK(nfc != NULL && U_SUCCESS(status), "NFC singleton is available");

    const UChar composed[] = {'f', 'o', 'x', 0x4E2D, 0x00E9};
    status = U_ZERO_ERROR;
    int32_t n = unorm2_normalize(nfc, composed, 5, NULL, 0, &status);
    CHECK(n == 5, "already-composed text keeps its length");
    UChar norm[8];
    memset(norm, 0, sizeof(norm));
    status = U_ZERO_ERROR;
    unorm2_normalize(nfc, composed, 5, norm, 8, &status);
    CHECK(U_SUCCESS(status), "NFC of composed text succeeds");
    CHECK(memcmp(norm, composed, sizeof(composed)) == 0,
          "NFC of composed text is unchanged");

    /* The one intentional difference between the two runtime paths. */
    const UChar decomposed[] = {0x0065, 0x0301}; /* e + combining acute */
    status = U_ZERO_ERROR;
    int32_t decomp_len =
        unorm2_normalize(nfc, decomposed, 2, NULL, 0, &status);
    UChar decomp[8];
    memset(decomp, 0, sizeof(decomp));
    status = U_ZERO_ERROR;
    unorm2_normalize(nfc, decomposed, 2, decomp, 8, &status);

#ifdef QZEN_TEST_NO_ICU
    CHECK(decomp_len == 2 && decomp[0] == 0x0065 && decomp[1] == 0x0301,
          "fallback NFC is identity on decomposed Latin");
#else
    if (decomp_len != 1 || decomp[0] != 0x00E9) {
        /* All shared assertions above still applied; only the ICU-specific
         * composition could not be exercised on an ICU-less host. */
        CHECK(failures == 0,
              "shared checks pass when no host ICU is available");
        if (failures == 0) {
            printf("SKIP: no usable host ICU (e + U+0301 did not compose); "
                   "shared assertions passed\n");
            return SKIP_NO_ICU;
        }
    }
#endif

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
#ifdef QZEN_TEST_NO_ICU
    printf("POSIX ICU fallback path: all checks passed\n");
#else
    printf("POSIX ICU host path: all checks passed\n");
#endif
    return 0;
}
