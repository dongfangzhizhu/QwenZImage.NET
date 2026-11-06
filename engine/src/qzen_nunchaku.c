#include "qzen_nunchaku.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Nunchaku SVDQuant int4 dequantisation.  See qzen_nunchaku.h for the formula;
 * this file is the four un-permutations and nothing else.
 *
 * Geometry, from pack_weight with bits = 4 and warp_n = 128:
 *
 *   mem_n = 128, mem_k = 64        num_n_packs = 8,  num_k_packs = 1
 *   n_pack_size = 2, k_pack_size = 2
 *   num_n_lanes = 8, num_k_lanes = 4
 *   reg_n = 1,       reg_k = 8     (32 bits / 4 bits)
 *
 * The flat word stream is tile-contiguous: one (128 row x 64 column) tile is
 * 1024 int32 words, and the words inside it are ordered
 * (n_pack, n_lane, k_lane, n_size, k_size).  Every index below was read off
 * the vendor kernel rather than derived, because deriving it from the Python
 * packer gives an order that round-trips perfectly and is still wrong.
 */

#define GROUP 64u
#define MEM_N 128u
#define MEM_K 64u
#define WORDS_PER_TILE (MEM_N * MEM_K / 8u)     /* 1024 */
#define LOWRANK_TILE 16u
#define WARP_N 128u
#define SCALES_PER_GROUP_BLOCK (GROUP * 2u)     /* 128 rows share a block */

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* --- the four index maps --------------------------------------------------- */

/* Row r of a 128-row tile and column c of a 64-column tile live in
 * word_in_tile = n_pack*128 + n_lane*16 + k_lane*4 + n_size*2 + k_size,
 * nibble = c % 8.  One word is eight columns, so the k lane only spans four
 * word-columns and k_size picks which 32-column half it is in.  The kernel's
 * own reading of the word index -- bit 0 moves k+32, bit 1 n+8, bits 2-3 k+8
 * and k+16, bits 4-6 n+1, n+2 and n+4, bit 7 n+16, bits 8-9 n+32 and n+64 --
 * is this same map, and it is the map that matters: the Python packer's own
 * axis order round-trips perfectly and is still not what the kernel reads. */
void qzen_nunchaku_code_index(uint32_t row, uint32_t column, uint32_t k_tiles,
                            uint64_t *word, uint32_t *nibble) {
    uint32_t n_tile = row / MEM_N;
    uint32_t n_local = row % MEM_N;
    uint32_t k_tile = column / MEM_K;
    uint32_t k_local = column % MEM_K;
    uint32_t n_pack = n_local / 16u;
    uint32_t n_size = (n_local / 8u) % 2u;
    uint32_t n_lane = n_local % 8u;
    uint32_t k_size = k_local / 32u;
    uint32_t k_lane = (k_local % 32u) / 8u;
    uint32_t in_tile = n_pack * 128u + n_lane * 16u + k_lane * 4u
                     + n_size * 2u + k_size;
    *word = (uint64_t)(n_tile * k_tiles + k_tile) * WORDS_PER_TILE + in_tile;
    *nibble = k_local % 8u;
}

/* wscales is flattened from (block, group, s_pack, b, d, c, e), so the buffer
 * index is block*7680 + group*128 + rem with rem = b*16 + d*4 + c*2 + e while
 * the row is block*128 + b*16 + c*8 + d*2 + e: c and d swap places. */
static uint32_t scale_rem(uint32_t row_local) {
    uint32_t b = row_local / 16u;
    uint32_t c = (row_local / 8u) % 2u;
    uint32_t d = (row_local % 8u) / 2u;
    uint32_t e = row_local % 2u;
    return b * 16u + d * 4u + c * 2u + e;
}

void qzen_nunchaku_scale_index(uint32_t row, uint32_t group, uint32_t groups,
                             uint32_t *index) {
    uint32_t block = row / WARP_N;
    *index = block * (groups * WARP_N) + group * WARP_N + scale_rem(row % WARP_N);
}

void qzen_nunchaku_smooth_index(uint32_t column, uint32_t *index) {
    uint32_t block = column / WARP_N;
    *index = block * WARP_N + scale_rem(column % WARP_N);
}

/* pack_lowrank_weight, 16x16 tiles.  `rank_packs` is rank / 16.
 *
 * The two factors share one formula but not one orientation.  Call (c, r) the
 * stored shape: for `proj_down` that is [in, rank] and for `proj_up`
 * [out, rank].  Inside a tile the *first* 16-axis is the tile row and the
 * second is the tile column, and which of c/r plays which swaps with `down`:
 * down puts the rank on the row axis, up puts the output on it.  The tile
 * itself is always indexed (c_pack, r_pack), so the base swaps with them. */
static uint64_t lowrank_index(uint32_t c, uint32_t r, uint32_t rank_packs,
                              int down) {
    uint32_t p = down ? r : c;                  /* tile row axis */
    uint32_t q = down ? c : r;                  /* tile column axis */
    uint32_t c_pack = (down ? q : p) / LOWRANK_TILE;
    uint32_t r_pack = (down ? p : q) / LOWRANK_TILE;
    uint64_t index = (uint64_t)(c_pack * rank_packs + r_pack)
                   * (LOWRANK_TILE * LOWRANK_TILE);
    index += (uint64_t)(p % 8u) * 32u;
    index += (uint64_t)((q / 2u) % 4u) * 8u;
    index += (uint64_t)((p / 8u) % 2u) * 4u;
    index += (uint64_t)((q / 8u) % 2u) * 2u;
    index += q % 2u;
    return index;
}

/* --- tensor access --------------------------------------------------------- */

/* Look up "<base>.<suffix>" and check its dtype, rank and shape.  A dimension
 * of 0 means "do not care", which is how the low-rank tensors are probed for
 * a rank the caller has not been told. */
static const qzen_st_tensor *find2(const qzen_weight_store *store, const char *base,
                                 const char *suffix, qzen_dtype dtype,
                                 const qzen_st_header **header,
                                 uint64_t rows, uint64_t cols, int ndim,
                                 char *error, size_t error_size) {
    char name[224];
    int written = snprintf(name, sizeof(name), "%s.%s", base, suffix);
    if (written <= 0 || (size_t)written >= sizeof(name)) {
        fail(error, error_size, "nunchaku tensor name is too long: %s", base);
        return NULL;
    }
    const qzen_st_header *found_header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &found_header);
    if (!tensor) {
        fail(error, error_size, "nunchaku tensor is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size,
             "%s has dtype %d ndim %d, expected dtype %d ndim %d", name,
             (int)tensor->dtype, tensor->ndim, (int)dtype, ndim);
        return NULL;
    }
    if (ndim == 2 && ((rows && tensor->shape[0] != rows) ||
                      (cols && tensor->shape[1] != cols))) {
        fail(error, error_size, "%s is [%llu, %llu], expected [%llu, %llu]",
             name, (unsigned long long)tensor->shape[0],
             (unsigned long long)tensor->shape[1],
             (unsigned long long)rows, (unsigned long long)cols);
        return NULL;
    }
    if (ndim == 1 && rows && tensor->shape[0] != rows) {
        fail(error, error_size, "%s has %llu elements, expected %llu", name,
             (unsigned long long)tensor->shape[0], (unsigned long long)rows);
        return NULL;
    }
    if (header) *header = found_header;
    return tensor;
}

int qzen_nunchaku_present(const qzen_weight_store *store, const char *base) {
    if (!store || !base) return 0;
    char name[224];
    int written = snprintf(name, sizeof(name), "%s.qweight", base);
    if (written <= 0 || (size_t)written >= sizeof(name)) return 0;
    return qzen_weight_find(store, name, NULL) != NULL;
}

/* --- one dequantised matrix, cached so q/k/v slices cost one pass ---------- */

static struct {
    const qzen_weight_store *store;
    char base[224];
    uint16_t *weight;
    uint32_t out;
    uint32_t in;
} cache;

static void cache_drop(void) {
    free(cache.weight);
    cache.weight = NULL;
    cache.store = NULL;
    cache.base[0] = '\0';
    cache.out = 0;
    cache.in = 0;
}

static uint16_t *dequantise(const qzen_weight_store *store, const char *base,
                            uint32_t out, uint32_t in,
                            char *error, size_t error_size) {
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = NULL;
    uint32_t k_tiles = in / MEM_K;
    uint32_t groups = in / GROUP;

    tensor = find2(store, base, "qweight", QZEN_DTYPE_I8, &header, out, in / 2, 2,
                   error, error_size);
    if (!tensor) return NULL;
    size_t code_bytes = (size_t)out * (in / 2);
    uint8_t *codes = malloc(code_bytes);
    if (!codes) {
        fail(error, error_size, "cannot allocate %zu bytes of nunchaku codes",
             code_bytes);
        return NULL;
    }
    if (!qzen_st_read_data(header, tensor, codes, code_bytes, error, error_size)) {
        free(codes);
        return NULL;
    }

    /* wscales is [groups, out] as stored; read it as a flat buffer and index
     * it, since the stored row is a 128-row block, not a group. */
    tensor = find2(store, base, "wscales", QZEN_DTYPE_BF16, &header, groups, out,
                   2, error, error_size);
    if (!tensor) {
        free(codes);
        return NULL;
    }
    size_t scale_count = (size_t)groups * out;
    uint16_t *scale_bf16 = malloc(scale_count * 2);
    float *scale = malloc(scale_count * sizeof(float));
    if (!scale_bf16 || !scale ||
        !qzen_st_read_data(header, tensor, scale_bf16, scale_count * 2, error,
                         error_size)) {
        free(codes);
        free(scale_bf16);
        free(scale);
        return NULL;
    }
    for (size_t i = 0; i < scale_count; i++) scale[i] = bf16_to_f32(scale_bf16[i]);
    free(scale_bf16);

    tensor = find2(store, base, "smooth_factor", QZEN_DTYPE_BF16, &header, in, 0,
                   1, error, error_size);
    if (!tensor) {
        free(codes);
        free(scale);
        return NULL;
    }
    uint16_t *smooth_bf16 = malloc((size_t)in * 2);
    float *smooth = malloc((size_t)in * sizeof(float));
    if (!smooth_bf16 || !smooth ||
        !qzen_st_read_data(header, tensor, smooth_bf16, (size_t)in * 2, error,
                         error_size)) {
        free(codes);
        free(scale);
        free(smooth_bf16);
        free(smooth);
        return NULL;
    }
    for (uint32_t k = 0; k < in; k++) {
        uint32_t index = 0;
        qzen_nunchaku_smooth_index(k, &index);
        smooth[k] = bf16_to_f32(smooth_bf16[index]);
    }
    free(smooth_bf16);

    /* Low-rank branch: unpack both factors into F32 and multiply. */
    uint32_t rank = 0;
    {
        const qzen_st_header *up_header = NULL;
        const qzen_st_tensor *up = find2(store, base, "proj_up", QZEN_DTYPE_BF16,
                                       &up_header, out, 0, 2, error,
                                       error_size);
        if (!up) {
            free(codes);
            free(scale);
            free(smooth);
            return NULL;
        }
        rank = (uint32_t)up->shape[1];
    }
    uint32_t rank_packs = rank / LOWRANK_TILE;
    if (!rank_packs || rank % LOWRANK_TILE) {
        fail(error, error_size, "nunchaku rank %u is not a multiple of %u",
             rank, LOWRANK_TILE);
        free(codes);
        free(scale);
        free(smooth);
        return NULL;
    }

    uint16_t *up_bf16 = malloc((size_t)out * rank * 2);
    float *up = malloc((size_t)out * rank * sizeof(float));
    uint16_t *down_bf16 = malloc((size_t)in * rank * 2);
    float *down = malloc((size_t)rank * in * sizeof(float));
    uint16_t *weight = malloc((size_t)out * in * 2);
    float *row = malloc((size_t)in * sizeof(float));
    if (!up_bf16 || !up || !down_bf16 || !down || !weight || !row) {
        fail(error, error_size, "cannot allocate the nunchaku dequant buffers");
        free(codes);
        free(scale);
        free(smooth);
        free(up_bf16);
        free(up);
        free(down_bf16);
        free(down);
        free(weight);
        free(row);
        return NULL;
    }

    {
        const qzen_st_header *header_up = NULL;
        const qzen_st_tensor *tensor_up = find2(store, base, "proj_up",
                                              QZEN_DTYPE_BF16, &header_up, out,
                                              rank, 2, error, error_size);
        const qzen_st_header *header_down = NULL;
        const qzen_st_tensor *tensor_down = find2(store, base, "proj_down",
                                                QZEN_DTYPE_BF16, &header_down,
                                                in, rank, 2, error,
                                                error_size);
        if (!tensor_up || !tensor_down ||
            !qzen_st_read_data(header_up, tensor_up, up_bf16,
                             (size_t)out * rank * 2, error, error_size) ||
            !qzen_st_read_data(header_down, tensor_down, down_bf16,
                             (size_t)in * rank * 2, error, error_size)) {
            free(codes);
            free(scale);
            free(smooth);
            free(up_bf16);
            free(up);
            free(down_bf16);
            free(down);
            free(weight);
            free(row);
            return NULL;
        }
    }

    for (uint32_t o = 0; o < out; o++) {
        for (uint32_t r = 0; r < rank; r++) {
            up[(size_t)o * rank + r] =
                bf16_to_f32(up_bf16[lowrank_index(o, r, rank_packs, 0)]);
        }
    }
    for (uint32_t r = 0; r < rank; r++) {
        for (uint32_t k = 0; k < in; k++) {
            down[(size_t)r * in + k] =
                bf16_to_f32(down_bf16[lowrank_index(k, r, rank_packs, 1)]);
        }
    }
    free(up_bf16);
    free(down_bf16);

    for (uint32_t o = 0; o < out; o++) {
        for (uint32_t k = 0; k < in; k++) {
            uint64_t word = 0;
            uint32_t nibble = 0;
            qzen_nunchaku_code_index(o, k, k_tiles, &word, &nibble);
            uint8_t byte = codes[word * 4u + nibble / 2u];
            int code = (nibble & 1u) ? (byte >> 4) & 0xF : byte & 0xF;
            if (code >= 8) code -= 16;
            uint32_t group = k / GROUP;
            uint32_t index = 0;
            qzen_nunchaku_scale_index(o, group, groups, &index);
            row[k] = (float)code * scale[index] / smooth[k];
        }
        for (uint32_t r = 0; r < rank; r++) {
            float u = up[(size_t)o * rank + r];
            const float *d = down + (size_t)r * in;
            for (uint32_t k = 0; k < in; k++) row[k] += u * d[k];
        }
        for (uint32_t k = 0; k < in; k++) {
            weight[(size_t)o * in + k] = f32_to_bf16(row[k]);
        }
    }

    free(codes);
    free(scale);
    free(smooth);
    free(up);
    free(down);
    free(row);
    return weight;
}

uint16_t *qzen_nunchaku_load_rows(const qzen_weight_store *store, const char *base,
                                uint32_t out, uint32_t in, uint32_t row_offset,
                                uint32_t rows, char *error, size_t error_size) {
    if (!store || !base || !out || !in || !rows || row_offset >= out ||
        rows > out - row_offset) {
        fail(error, error_size, "nunchaku row request is out of range");
        return NULL;
    }
    if (!cache.weight || cache.store != store || cache.out != out ||
        cache.in != in || strcmp(cache.base, base) != 0) {
        cache_drop();
        uint16_t *weight = dequantise(store, base, out, in, error, error_size);
        if (!weight) return NULL;
        cache.store = store;
        snprintf(cache.base, sizeof(cache.base), "%s", base);
        cache.weight = weight;
        cache.out = out;
        cache.in = in;
    }
    size_t bytes = (size_t)rows * in * 2;
    uint16_t *slice = malloc(bytes);
    if (!slice) {
        fail(error, error_size, "cannot allocate %zu bytes for %s rows %u..%u",
             bytes, base, row_offset, row_offset + rows);
        return NULL;
    }
    memcpy(slice, cache.weight + (size_t)row_offset * in, bytes);
    return slice;
}

uint16_t *qzen_nunchaku_load_bf16(const qzen_weight_store *store, const char *base,
                                uint32_t out, uint32_t in,
                                char *error, size_t error_size) {
    return qzen_nunchaku_load_rows(store, base, out, in, 0, out, error,
                                 error_size);
}

/* --- packed residency ------------------------------------------------------ */

/* The reorder below is a byte permutation, not an arithmetic one: one int32
 * word of the vendor stream is eight consecutive columns of one row, so a word
 * moves as four bytes once its destination is known.  WORDS_PER_TILE
 * destinations are the same for every tile -- only the base offset changes with
 * (n_tile, k_tile) -- so they are computed once per matrix. */
static void code_destination_table(uint32_t in, uint32_t *table) {
    const size_t row_bytes = in / 2u;
    for (uint32_t word = 0; word < WORDS_PER_TILE; word++) {
        const uint32_t n_pack = word / 128u;
        const uint32_t rest = word % 128u;
        const uint32_t n_lane = rest / 16u;
        const uint32_t k_lane = (rest % 16u) / 4u;
        const uint32_t n_size = (rest % 4u) / 2u;
        const uint32_t k_size = rest % 2u;
        const uint32_t row_local = n_pack * 16u + n_size * 8u + n_lane;
        const uint32_t k_local = k_size * 32u + k_lane * 8u;
        table[word] = (uint32_t)((size_t)row_local * row_bytes + k_local / 2u);
    }
}

void qzen_nunchaku_packed_free(qzen_nunchaku_packed *packed) {
    if (!packed) return;
    free(packed->codes);
    free(packed->scales);
    free(packed->smooth);
    free(packed->proj_down);
    free(packed->proj_up);
    memset(packed, 0, sizeof(*packed));
}

static int pack_full(const qzen_weight_store *store, const char *base,
                     uint32_t out, uint32_t in, qzen_nunchaku_packed *packed,
                     char *error, size_t error_size) {
    const uint32_t groups = in / GROUP;
    const uint32_t k_tiles = in / MEM_K;
    const uint32_t n_tiles = out / MEM_N;
    if (!groups || in % MEM_K || out % MEM_N) {
        fail(error, error_size,
             "nunchaku geometry %ux%u is not a multiple of %ux%u", out, in,
             MEM_N, MEM_K);
        return 0;
    }

    memset(packed, 0, sizeof(*packed));
    packed->out = out;
    packed->in = in;
    packed->groups = groups;

    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = NULL;
    size_t code_bytes = (size_t)out * (in / 2u);
    uint8_t *codes_in = malloc(code_bytes);
    uint8_t *codes_out = malloc(code_bytes);
    if (!codes_in || !codes_out) {
        fail(error, error_size, "cannot allocate %zu bytes of nunchaku codes",
             code_bytes);
        free(codes_in);
        free(codes_out);
        return 0;
    }
    tensor = find2(store, base, "qweight", QZEN_DTYPE_I8, &header, out, in / 2, 2,
                   error, error_size);
    if (!tensor ||
        !qzen_st_read_data(header, tensor, codes_in, code_bytes, error,
                         error_size)) {
        free(codes_in);
        free(codes_out);
        return 0;
    }

    uint32_t table[WORDS_PER_TILE];
    code_destination_table(in, table);
    const size_t row_bytes = in / 2u;
    for (uint32_t n_tile = 0; n_tile < n_tiles; n_tile++) {
        for (uint32_t k_tile = 0; k_tile < k_tiles; k_tile++) {
            const uint8_t *src =
                codes_in + (size_t)(n_tile * k_tiles + k_tile) *
                               (WORDS_PER_TILE * 4u);
            uint8_t *dst = codes_out + (size_t)n_tile * MEM_N * row_bytes +
                           (size_t)k_tile * 32u;
            for (uint32_t word = 0; word < WORDS_PER_TILE; word++) {
                memcpy(dst + table[word], src + word * 4u, 4);
            }
        }
    }
    free(codes_in);

    /* wscales: [groups, out] as stored, flattened as blocks of 128 rows. */
    size_t scale_count = (size_t)groups * out;
    uint16_t *scale_in = malloc(scale_count * 2);
    uint16_t *scale_out = malloc(scale_count * 2);
    if (!scale_in || !scale_out) {
        fail(error, error_size, "cannot allocate the nunchaku scales");
        free(codes_out);
        free(scale_in);
        free(scale_out);
        return 0;
    }
    tensor = find2(store, base, "wscales", QZEN_DTYPE_BF16, &header, groups, out,
                   2, error, error_size);
    if (!tensor ||
        !qzen_st_read_data(header, tensor, scale_in, scale_count * 2, error,
                         error_size)) {
        free(codes_out);
        free(scale_in);
        free(scale_out);
        return 0;
    }
    for (uint32_t o = 0; o < out; o++) {
        for (uint32_t group = 0; group < groups; group++) {
            uint32_t index = 0;
            qzen_nunchaku_scale_index(o, group, groups, &index);
            scale_out[(size_t)o * groups + group] = scale_in[index];
        }
    }
    free(scale_in);

    tensor = find2(store, base, "smooth_factor", QZEN_DTYPE_BF16, &header, in, 0,
                   1, error, error_size);
    float *smooth = malloc((size_t)in * sizeof(float));
    uint16_t *smooth_in = malloc((size_t)in * 2);
    if (!tensor || !smooth || !smooth_in ||
        !qzen_st_read_data(header, tensor, smooth_in, (size_t)in * 2, error,
                         error_size)) {
        free(codes_out);
        free(scale_out);
        free(smooth);
        free(smooth_in);
        return 0;
    }
    for (uint32_t k = 0; k < in; k++) {
        uint32_t index = 0;
        qzen_nunchaku_smooth_index(k, &index);
        smooth[k] = bf16_to_f32(smooth_in[index]);
    }
    free(smooth_in);

    uint32_t rank = 0;
    {
        const qzen_st_tensor *up = find2(store, base, "proj_up", QZEN_DTYPE_BF16,
                                       &header, out, 0, 2, error,
                                       error_size);
        if (!up) {
            free(codes_out);
            free(scale_out);
            free(smooth);
            return 0;
        }
        rank = (uint32_t)up->shape[1];
    }
    uint32_t rank_packs = rank / LOWRANK_TILE;
    if (!rank_packs || rank % LOWRANK_TILE) {
        fail(error, error_size, "nunchaku rank %u is not a multiple of %u",
             rank, LOWRANK_TILE);
        free(codes_out);
        free(scale_out);
        free(smooth);
        return 0;
    }
    uint16_t *up_in = malloc((size_t)out * rank * 2);
    uint16_t *up = malloc((size_t)out * rank * 2);
    uint16_t *down_in = malloc((size_t)in * rank * 2);
    uint16_t *down = malloc((size_t)rank * in * 2);
    if (!up_in || !up || !down_in || !down) {
        fail(error, error_size, "cannot allocate the nunchaku low-rank buffers");
        free(codes_out);
        free(scale_out);
        free(smooth);
        free(up_in);
        free(up);
        free(down_in);
        free(down);
        return 0;
    }
    {
        const qzen_st_header *up_header = NULL;
        const qzen_st_header *down_header = NULL;
        const qzen_st_tensor *up_tensor =
            find2(store, base, "proj_up", QZEN_DTYPE_BF16, &up_header, out, rank,
                  2, error, error_size);
        const qzen_st_tensor *down_tensor =
            find2(store, base, "proj_down", QZEN_DTYPE_BF16, &down_header, in,
                  rank, 2, error, error_size);
        if (!up_tensor || !down_tensor ||
            !qzen_st_read_data(up_header, up_tensor, up_in, (size_t)out * rank * 2,
                             error, error_size) ||
            !qzen_st_read_data(down_header, down_tensor, down_in,
                             (size_t)in * rank * 2, error, error_size)) {
            free(codes_out);
            free(scale_out);
            free(smooth);
            free(up_in);
            free(up);
            free(down_in);
            free(down);
            return 0;
        }
    }
    for (uint32_t o = 0; o < out; o++) {
        for (uint32_t r = 0; r < rank; r++) {
            up[(size_t)o * rank + r] = up_in[lowrank_index(o, r, rank_packs, 0)];
        }
    }
    for (uint32_t r = 0; r < rank; r++) {
        for (uint32_t k = 0; k < in; k++) {
            down[(size_t)r * in + k] = down_in[lowrank_index(k, r, rank_packs, 1)];
        }
    }
    free(up_in);
    free(down_in);

    packed->rank = rank;
    packed->codes = codes_out;
    packed->scales = scale_out;
    packed->smooth = smooth;
    packed->proj_up = up;
    packed->proj_down = down;
    return 1;
}

static struct {
    const qzen_weight_store *store;
    char base[224];
    qzen_nunchaku_packed packed;
    uint32_t out;
    uint32_t in;
    int live;
} pack_cache;

static void pack_cache_drop(void) {
    qzen_nunchaku_packed_free(&pack_cache.packed);
    pack_cache.store = NULL;
    pack_cache.base[0] = '\0';
    pack_cache.out = 0;
    pack_cache.in = 0;
    pack_cache.live = 0;
}

int qzen_nunchaku_load_packed_rows(const qzen_weight_store *store, const char *base,
                                 uint32_t full_out, uint32_t in,
                                 uint32_t row_offset, uint32_t rows,
                                 qzen_nunchaku_packed *packed,
                                 char *error, size_t error_size) {
    if (!store || !base || !packed || !full_out || !in || !rows ||
        row_offset >= full_out || rows > full_out - row_offset) {
        fail(error, error_size, "nunchaku packed row request is out of range");
        return 0;
    }
    memset(packed, 0, sizeof(*packed));
    if (!pack_cache.live || pack_cache.store != store ||
        pack_cache.out != full_out || pack_cache.in != in ||
        strcmp(pack_cache.base, base) != 0) {
        pack_cache_drop();
        if (!pack_full(store, base, full_out, in, &pack_cache.packed, error,
                       error_size))
            return 0;
        pack_cache.store = store;
        snprintf(pack_cache.base, sizeof(pack_cache.base), "%s", base);
        pack_cache.out = full_out;
        pack_cache.in = in;
        pack_cache.live = 1;
    }

    const qzen_nunchaku_packed *full = &pack_cache.packed;
    const size_t row_bytes = in / 2u;
    packed->out = rows;
    packed->in = in;
    packed->rank = full->rank;
    packed->groups = full->groups;
    packed->codes = malloc((size_t)rows * row_bytes);
    packed->scales = malloc((size_t)rows * full->groups * 2);
    packed->smooth = malloc((size_t)in * sizeof(float));
    packed->proj_up = malloc((size_t)rows * full->rank * 2);
    packed->proj_down = malloc((size_t)full->rank * in * 2);
    if (!packed->codes || !packed->scales || !packed->smooth ||
        !packed->proj_up || !packed->proj_down) {
        fail(error, error_size, "cannot allocate the packed slice of %s", base);
        qzen_nunchaku_packed_free(packed);
        return 0;
    }
    memcpy(packed->codes, full->codes + (size_t)row_offset * row_bytes,
           (size_t)rows * row_bytes);
    memcpy(packed->scales,
           full->scales + (size_t)row_offset * full->groups,
           (size_t)rows * full->groups * 2);
    memcpy(packed->smooth, full->smooth, (size_t)in * sizeof(float));
    memcpy(packed->proj_up, full->proj_up + (size_t)row_offset * full->rank,
           (size_t)rows * full->rank * 2);
    memcpy(packed->proj_down, full->proj_down, (size_t)full->rank * in * 2);
    return 1;
}

/* Expand into dst[out][in], which the caller owns.  `with_lowrank` adds the
 * SVDQuant branch per element; the CPU fallback passes 0 and adds that branch
 * as two GEMMs instead (see qzen_zimage_linear_forward, PLAN §21), because the
 * per-element form costs rank multiply-adds for every element of every layer. */
static int packed_expand_into(const qzen_nunchaku_packed *packed,
                              int with_lowrank, uint16_t *dst, char *error,
                              size_t error_size) {
    if (!packed || !packed->codes || !packed->scales || !packed->smooth ||
        !packed->out || !packed->in || !packed->groups ||
        (with_lowrank && (!packed->proj_up || !packed->proj_down ||
                          !packed->rank))) {
        fail(error, error_size, "incomplete packed nunchaku matrix");
        return 0;
    }
    const uint32_t out = packed->out;
    const uint32_t in = packed->in;
    const uint32_t rank = packed->rank;
    const size_t row_bytes = in / 2u;
    float *row = malloc((size_t)in * sizeof(float));
    if (!row) {
        fail(error, error_size, "cannot allocate %zu bytes for a packed row",
             (size_t)in * sizeof(float));
        return 0;
    }
    for (uint32_t o = 0; o < out; o++) {
        const uint8_t *codes = packed->codes + (size_t)o * row_bytes;
        const uint16_t *scales = packed->scales + (size_t)o * packed->groups;
        const uint16_t *up = with_lowrank ? packed->proj_up + (size_t)o * rank
                                          : NULL;
        for (uint32_t k = 0; k < in; k++) {
            uint8_t byte = codes[k / 2u];
            int code = (k & 1u) ? (byte >> 4) & 0xF : byte & 0xF;
            if (code >= 8) code -= 16;
            row[k] = (float)code * bf16_to_f32(scales[k / GROUP])
                   / packed->smooth[k];
        }
        for (uint32_t r = 0; r < (with_lowrank ? rank : 0u); r++) {
            float u = bf16_to_f32(up[r]);
            const uint16_t *down = packed->proj_down + (size_t)r * in;
            for (uint32_t k = 0; k < in; k++) {
                row[k] += u * bf16_to_f32(down[k]);
            }
        }
        for (uint32_t k = 0; k < in; k++) {
            dst[(size_t)o * in + k] = f32_to_bf16(row[k]);
        }
    }
    free(row);
    return 1;
}

static uint16_t *packed_to_bf16(const qzen_nunchaku_packed *packed,
                                int with_lowrank, char *error,
                                size_t error_size) {
    if (!packed || !packed->out || !packed->in) {
        fail(error, error_size, "incomplete packed nunchaku matrix");
        return NULL;
    }
    uint16_t *weight = malloc((size_t)packed->out * packed->in * 2);
    if (!weight) {
        fail(error, error_size, "cannot allocate %zu bytes for the packed matrix",
             (size_t)packed->out * packed->in * 2);
        return NULL;
    }
    if (!packed_expand_into(packed, with_lowrank, weight, error, error_size)) {
        free(weight);
        return NULL;
    }
    return weight;
}

uint16_t *qzen_nunchaku_packed_to_bf16(const qzen_nunchaku_packed *packed,
                                     char *error, size_t error_size) {
    return packed_to_bf16(packed, 1, error, error_size);
}

/* The quantised branch alone, R / smooth.  Exists because the device kernel
 * computes the low-rank branch separately (as two GEMMs, the way the vendor
 * does), so the bit-exact comparison has to be against the same thing. */
uint16_t *qzen_nunchaku_packed_to_bf16_quant(const qzen_nunchaku_packed *packed,
                                           char *error, size_t error_size) {
    return packed_to_bf16(packed, 0, error, error_size);
}

int qzen_nunchaku_packed_quant_into(const qzen_nunchaku_packed *packed,
                                  uint16_t *dst, char *error,
                                  size_t error_size) {
    return packed_expand_into(packed, 0, dst, error, error_size);
}

