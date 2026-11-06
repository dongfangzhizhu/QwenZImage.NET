/* Minimal PNG encoder for the deterministic reference backend.
 *
 * A PNG is: signature, IHDR, IDAT(zlib: stored deflate blocks of filtered
 * scanlines), IEND. Stored blocks are ~5 bytes larger per 64 KiB than a real
 * compressor and the file is roughly width*height*3 bytes, but the point of
 * this backend is deterministic, dependency-free interop testing -- not
 * compression. No zlib, no stdio, no platform APIs.
 */

#include "qzi_png.h"

#include <stdlib.h>
#include <string.h>

static uint32_t crc_table[256];
static int crc_table_ready = 0;

static void crc_init(void)
{
    uint32_t i, k;
    for (i = 0; i < 256; i++) {
        uint32_t c = i;
        for (k = 0; k < 8; k++)
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc_table[i] = c;
    }
    crc_table_ready = 1;
}

/* Standard CRC-32 (IEEE), finalised. */
static uint32_t crc32_calc(const uint8_t *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    size_t i;
    if (!crc_table_ready) crc_init();
    for (i = 0; i < len; i++)
        crc = crc_table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

static uint32_t adler32_calc(const uint8_t *data, size_t len)
{
    uint32_t a = 1, b = 0;
    size_t i;
    for (i = 0; i < len; i++) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

static void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

/* Writes one PNG chunk at p and returns the new write position, or NULL on
 * allocation failure. CRC covers the type and the data, not the length. */
static uint8_t *emit_chunk(uint8_t *p, const char type[4],
                           const uint8_t *data, uint32_t len)
{
    uint8_t *buf = (uint8_t *)malloc((size_t)len + 4u);
    uint32_t crc;
    if (!buf) return NULL;
    memcpy(buf, type, 4);
    if (len) memcpy(buf + 4, data, len);
    crc = crc32_calc(buf, (size_t)len + 4u);
    free(buf);

    put_be32(p, len); p += 4;
    memcpy(p, type, 4); p += 4;
    if (len) { memcpy(p, data, len); p += len; }
    put_be32(p, crc); p += 4;
    return p;
}

static void set_error(char *error, size_t error_cap, const char *message)
{
    size_t n;
    if (!error || error_cap == 0) return;
    n = strlen(message);
    if (n >= error_cap) n = error_cap - 1;
    memcpy(error, message, n);
    error[n] = '\0';
}

uint8_t *qzi_png_encode_rgb8(const uint8_t *rgb, uint32_t width, uint32_t height,
                             size_t *out_size, char *error, size_t error_cap)
{
    static const uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    size_t row_bytes, raw_size, blocks, idat_size, total, off, remaining;
    uint8_t *out = NULL, *raw = NULL, *idat = NULL, *p, *q;
    uint8_t ihdr[13];
    uint32_t y;

    if (out_size) *out_size = 0;
    if (!rgb || width == 0 || height == 0 || !out_size) {
        set_error(error, error_cap, "png: invalid input");
        return NULL;
    }
    if ((size_t)width > SIZE_MAX / 3u / (size_t)height ||
        (size_t)height > SIZE_MAX / ((size_t)width * 3u + 1u)) {
        set_error(error, error_cap, "png: dimensions overflow");
        return NULL;
    }
    row_bytes = (size_t)width * 3u;
    raw_size = (row_bytes + 1u) * (size_t)height;
    blocks = (raw_size + 65534u) / 65535u;
    if (blocks == 0) blocks = 1;
    if (raw_size > SIZE_MAX - (2u + 5u * blocks + 4u) - 4096u) {
        set_error(error, error_cap, "png: raw buffer overflow");
        return NULL;
    }
    idat_size = 2u + 5u * blocks + raw_size + 4u;
    total = 8u + 25u + (12u + idat_size) + 12u;
    if (idat_size > 0xFFFFFFFFu) {
        set_error(error, error_cap, "png: IDAT too large");
        return NULL;
    }

    raw = (uint8_t *)malloc(raw_size);
    out = (uint8_t *)malloc(total);
    idat = (uint8_t *)malloc(idat_size);
    if (!raw || !out || !idat) {
        free(raw); free(out); free(idat);
        set_error(error, error_cap, "png: allocation failed");
        return NULL;
    }

    for (y = 0; y < height; y++) {
        uint8_t *dst = raw + (size_t)y * (row_bytes + 1u);
        dst[0] = 0; /* filter: none */
        memcpy(dst + 1, rgb + (size_t)y * row_bytes, row_bytes);
    }

    q = idat;
    q[0] = 0x78; q[1] = 0x01; /* zlib: deflate, 32 KiB window, no preset dict */
    q += 2;
    off = 0;
    remaining = raw_size;
    while (remaining > 0) {
        size_t chunk = remaining < 65535u ? remaining : 65535u;
        size_t complement = ~chunk;
        int final = (chunk == remaining);
        *q++ = (uint8_t)(final ? 0x01 : 0x00); /* BFINAL + BTYPE=00 (stored) */
        *q++ = (uint8_t)(chunk & 0xFFu);
        *q++ = (uint8_t)((chunk >> 8) & 0xFFu);
        *q++ = (uint8_t)(complement & 0xFFu);
        *q++ = (uint8_t)((complement >> 8) & 0xFFu);
        memcpy(q, raw + off, chunk);
        q += chunk;
        off += chunk;
        remaining -= chunk;
    }
    put_be32(q, adler32_calc(raw, raw_size));

    p = out;
    memcpy(p, signature, 8); p += 8;
    memset(ihdr, 0, sizeof(ihdr));
    put_be32(ihdr, width);
    put_be32(ihdr + 4, height);
    ihdr[8] = 8; /* 8-bit */
    ihdr[9] = 2; /* truecolor RGB */
    p = emit_chunk(p, "IHDR", ihdr, 13);
    if (p) p = emit_chunk(p, "IDAT", idat, (uint32_t)idat_size);
    if (p) p = emit_chunk(p, "IEND", NULL, 0);

    free(idat);
    free(raw);
    if (!p) {
        free(out);
        set_error(error, error_cap, "png: chunk allocation failed");
        return NULL;
    }
    *out_size = (size_t)(p - out);
    return out;
}
