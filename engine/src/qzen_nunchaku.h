#ifndef QZEN_NUNCHAKU_H
#define QZEN_NUNCHAKU_H

#include "qzen_safetensors.h"
#include "qzen_weights.h"

#include <stddef.h>
#include <stdint.h>

/* Nunchaku SVDQuant int4 checkpoints (nunchaku-tech/nunchaku-z-image-turbo).
 *
 * A quantised Linear is five tensors instead of the quanto pair:
 *
 *   "<base>.qweight"       I8   [out, in/2]   two 4-bit codes per byte
 *   "<base>.wscales"       BF16 [in/64, out]  one scale per 64-wide group
 *   "<base>.smooth_factor" BF16 [in]          per input channel
 *   "<base>.proj_down"     BF16 [in, rank]    low-rank branch, 16x16 packed
 *   "<base>.proj_up"       BF16 [out, rank]   low-rank branch, 16x16 packed
 *
 * and the effective matrix the vendor kernel multiplies by is
 *
 *   W[o, k] = code[o, k] * wscale[o, k / 64] / smooth[k]
 *             + (proj_up @ proj_down)[o, k]
 *
 * The low-rank branch is *not* smoothed -- (R + L) / smooth is off by a factor
 * of 24 against the kernel, R / smooth + L agrees to bf16 rounding.
 *
 * None of the five tensors is in natural order.  Every permutation below was
 * read off the vendor kernel itself (tools/nunchaku_layout_oracle.py,
 * tools/nunchaku_scale_oracle.py) and the whole thing is verified against
 * `layer(eye(in_features))` on three layers, including a non-square FFN, to a
 * cosine of 0.99999 -- see PLAN_RTX4090.md 6.1.2 and
 * tools/nunchaku_dequant_verify.py, which is the reference this must match.
 *
 * Note that Nunchaku fuses q/k/v into one `to_qkv`, so callers that want the
 * three projections separately read three row ranges of the same base name;
 * qzen_nunchaku_load_rows keeps one dequantised matrix alive so that costs one
 * pass, not three.
 */

/* Does this checkpoint carry a Nunchaku quantised Linear under `base`? */
int qzen_nunchaku_present(const qzen_weight_store *store, const char *base);

/* Dequantise rows [row_offset, row_offset + rows) of the [out, in] matrix
 * stored under `base` into a fresh BF16 buffer owned by the caller (free()).
 * `out` and `in` describe the full matrix, not the slice. */
uint16_t *qzen_nunchaku_load_rows(const qzen_weight_store *store, const char *base,
                                uint32_t out, uint32_t in, uint32_t row_offset,
                                uint32_t rows, char *error, size_t error_size);

/* Whole-matrix convenience. */
uint16_t *qzen_nunchaku_load_bf16(const qzen_weight_store *store, const char *base,
                                uint32_t out, uint32_t in,
                                char *error, size_t error_size);

/* Index helpers, exposed for tests: they are the whole layout, and getting one
 * bit wrong produces a matrix that still looks like a matrix. */
void qzen_nunchaku_code_index(uint32_t row, uint32_t column, uint32_t k_tiles,
                            uint64_t *word, uint32_t *nibble);
void qzen_nunchaku_scale_index(uint32_t row, uint32_t group, uint32_t groups,
                             uint32_t *index);
void qzen_nunchaku_smooth_index(uint32_t column, uint32_t *index);

/* --- packed residency ------------------------------------------------------
 *
 * Everything above dequantises to BF16, which is why the 4-bit pack currently
 * saves disk and nothing else: the resident set is the expanded matrix either
 * way (PLAN §6.4.1 measured both packs at 13 093.5 MiB).  The same layout, kept
 * packed, is a quarter of that -- and the dequantisation can happen per layer
 * per forward instead of once per load, which is where the 136 s of extra load
 * time goes (PLAN §19.1).
 *
 * So this is the same un-permutation, done once, into buffers a kernel can read
 * without any index arithmetic:
 *
 *   codes      U8   [out][in/2]  row-major, low nibble = even column
 *   scales     BF16 [out][groups] row-major (groups = in / 64)
 *   smooth     F32  [in]          natural order, not inverted: the reference
 *                                 divides, and dividing is what the gate needs
 *   proj_down  BF16 [rank][in]    natural row-major
 *   proj_up    BF16 [out][rank]   natural row-major
 *
 * The weight those five describe is still R / smooth + L, with R = code*scale
 * and L = proj_up @ proj_down -- only the storage changed. */
typedef struct qzen_nunchaku_packed {
    uint32_t out;
    uint32_t in;
    uint32_t rank;
    uint32_t groups;
    uint8_t *codes;
    uint16_t *scales;
    float *smooth;
    uint16_t *proj_down;
    uint16_t *proj_up;
} qzen_nunchaku_packed;

/* Rows [row_offset, row_offset + rows) of the [full_out, in] matrix, packed.
 * `full_out` is the row count of the whole tensor: a fused to_qkv is 3 * dim
 * rows and each of q/k/v is one third, and the reorder is cached, so the three
 * slices cost one pass.  The result owns its buffers; free with
 * qzen_nunchaku_packed_free. */
int qzen_nunchaku_load_packed_rows(const qzen_weight_store *store, const char *base,
                                 uint32_t full_out, uint32_t in,
                                 uint32_t row_offset, uint32_t rows,
                                 qzen_nunchaku_packed *packed,
                                 char *error, size_t error_size);

void qzen_nunchaku_packed_free(qzen_nunchaku_packed *packed);

/* The BF16 matrix the packed form describes, computed the same way the
 * dequantiser computes it.  Exists so the reorder can be gated bit-exactly
 * against qzen_nunchaku_load_bf16, and as the CPU fallback. */
uint16_t *qzen_nunchaku_packed_to_bf16(const qzen_nunchaku_packed *packed,
                                     char *error, size_t error_size);

/* The quantised branch alone, R / smooth, without the low-rank term.  The
 * device kernel computes the low-rank branch separately (two GEMMs, the way the
 * vendor does), so the bit-exact comparison has to be against the same thing. */
uint16_t *qzen_nunchaku_packed_to_bf16_quant(const qzen_nunchaku_packed *packed,
                                           char *error, size_t error_size);

/* Same numbers as the above, written into a caller-owned dst[out][in] instead
 * of a fresh allocation.  The CPU fallback reuses one scratch across layers
 * (PLAN §21): a fresh 157 MB block per layer costs more in page faults than the
 * expansion itself, and keeping one per layer is the 10.6 GB this exists to
 * avoid. */
int qzen_nunchaku_packed_quant_into(const qzen_nunchaku_packed *packed,
                                  uint16_t *dst, char *error,
                                  size_t error_size);

#endif
