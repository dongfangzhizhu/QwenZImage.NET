#include "qzi_gguf.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

/* ------------------------------------------------------------- primitives */

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float half_to_float(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    uint32_t value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            /* Subnormal: renormalise into a float exponent. */
            int shift = 0;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x3ffu;
            value = sign | (uint32_t)(127 - 15 - shift + 1) << 23 |
                    mantissa << 13;
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | mantissa << 13;
    } else {
        value = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
    }
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((bits >> 16) | 0x0040u);
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    bits += rounding;
    return (uint16_t)(bits >> 16);
}

static float bf16_to_float(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

/* ------------------------------------------------------------ block types */

typedef struct {
    unsigned block;   /* elements per block */
    unsigned bytes;   /* bytes per block */
} qzi_gguf_layout;

static qzi_gguf_layout layout_of(unsigned type) {
    switch (type) {
        case QZI_GGUF_F32: return (qzi_gguf_layout){1, 4};
        case QZI_GGUF_F16: return (qzi_gguf_layout){1, 2};
        case QZI_GGUF_BF16: return (qzi_gguf_layout){1, 2};
        case QZI_GGUF_Q4_0: return (qzi_gguf_layout){32, 18};
        case QZI_GGUF_Q4_1: return (qzi_gguf_layout){32, 20};
        case QZI_GGUF_Q5_0: return (qzi_gguf_layout){32, 22};
        case QZI_GGUF_Q5_1: return (qzi_gguf_layout){32, 24};
        case QZI_GGUF_Q8_0: return (qzi_gguf_layout){32, 34};
        case QZI_GGUF_Q2_K: return (qzi_gguf_layout){256, 84};
        case QZI_GGUF_Q3_K: return (qzi_gguf_layout){256, 110};
        case QZI_GGUF_Q4_K: return (qzi_gguf_layout){256, 144};
        case QZI_GGUF_Q5_K: return (qzi_gguf_layout){256, 176};
        case QZI_GGUF_Q6_K: return (qzi_gguf_layout){256, 210};
        default: return (qzi_gguf_layout){0, 0};
    }
}

const char *qzi_gguf_type_name(qzi_gguf_type type) {
    switch (type) {
        case QZI_GGUF_F32: return "F32";
        case QZI_GGUF_F16: return "F16";
        case QZI_GGUF_BF16: return "BF16";
        case QZI_GGUF_Q4_0: return "Q4_0";
        case QZI_GGUF_Q4_1: return "Q4_1";
        case QZI_GGUF_Q5_0: return "Q5_0";
        case QZI_GGUF_Q5_1: return "Q5_1";
        case QZI_GGUF_Q8_0: return "Q8_0";
        case QZI_GGUF_Q2_K: return "Q2_K";
        case QZI_GGUF_Q3_K: return "Q3_K";
        case QZI_GGUF_Q4_K: return "Q4_K";
        case QZI_GGUF_Q5_K: return "Q5_K";
        case QZI_GGUF_Q6_K: return "Q6_K";
        default: return "UNKNOWN";
    }
}

/* ------------------------------------------------------------- container */

typedef struct {
    char *name;             /* canonical */
    char *stored;           /* as written in the file */
    uint32_t type;
    uint32_t ndim;
    uint64_t shape[8];      /* torch order */
    uint64_t elements;
    uint64_t offset;        /* relative to the data section */
    uint64_t bytes;
} qzi_gguf_tensor;

struct qzi_gguf {
    int fd;
    char *path;
    uint64_t data_start;
    qzi_gguf_tensor *tensors;
    size_t count;
    unsigned char *scratch;
    size_t scratch_bytes;
};

typedef struct {
    qzi_gguf *gguf;
    uint64_t position;
    unsigned char *buffer;
    size_t length;
    size_t consumed;
    size_t capacity;
    int failed;
} gguf_cursor;

static int cursor_fill(gguf_cursor *cursor, size_t want) {
    if (cursor->failed) return 0;
    if (cursor->length - cursor->consumed >= want) return 1;
    /* Slide the unread tail to the front, then read a block that is at least
     * as large as the request. */
    size_t keep = cursor->length - cursor->consumed;
    if (keep && cursor->consumed)
        memmove(cursor->buffer, cursor->buffer + cursor->consumed, keep);
    cursor->position += cursor->consumed;
    cursor->length = keep;
    cursor->consumed = 0;
    size_t capacity = cursor->capacity;
    if (capacity < want) {
        size_t next = capacity ? capacity : 65536;
        while (next < want) next *= 2;
        unsigned char *grown = realloc(cursor->buffer, next);
        if (!grown) {
            cursor->failed = 1;
            return 0;
        }
        cursor->buffer = grown;
        cursor->capacity = next;
        capacity = next;
    }
    while (cursor->length < want) {
        size_t room = capacity - cursor->length;
        ssize_t got = pread(cursor->gguf->fd, cursor->buffer + cursor->length,
                            room, (off_t)(cursor->position + cursor->length));
        if (got <= 0) {
            cursor->failed = 1;
            return 0;
        }
        cursor->length += (size_t)got;
    }
    return 1;
}

static const unsigned char *cursor_take(gguf_cursor *cursor, size_t bytes) {
    if (!cursor_fill(cursor, bytes)) return NULL;
    const unsigned char *at = cursor->buffer + cursor->consumed;
    cursor->consumed += bytes;
    return at;
}

static int cursor_skip(gguf_cursor *cursor, uint64_t bytes) {
    /* Skipping past the buffer must not allocate the skipped bytes. */
    while (bytes) {
        if (cursor->consumed < cursor->length) {
            size_t available = cursor->length - cursor->consumed;
            size_t step = bytes < available ? (size_t)bytes : available;
            cursor->consumed += step;
            bytes -= step;
            continue;
        }
        if (!cursor_fill(cursor, 1)) return 0;
    }
    return 1;
}

static uint64_t read_u64(const unsigned char *at) {
    uint64_t value = 0;
    for (int index = 0; index < 8; index++)
        value |= (uint64_t)at[index] << (8 * index);
    return value;
}

static uint32_t read_u32(const unsigned char *at) {
    return (uint32_t)at[0] | (uint32_t)at[1] << 8 | (uint32_t)at[2] << 16 |
           (uint32_t)at[3] << 24;
}

static uint32_t read_u16(const unsigned char *at) {
    return (uint32_t)at[0] | (uint32_t)at[1] << 8;
}

/* GGUF value types, used only to skip metadata we do not consume. */
enum {
    GGUF_UINT8 = 0, GGUF_INT8 = 1, GGUF_UINT16 = 2, GGUF_INT16 = 3,
    GGUF_UINT32 = 4, GGUF_INT32 = 5, GGUF_FLOAT32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_UINT64 = 10, GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12
};

static int scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: return 1;
        case GGUF_UINT16: case GGUF_INT16: return 2;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
        default: return 0;
    }
}

static int cursor_string(gguf_cursor *cursor, char **out) {
    const unsigned char *length_bytes = cursor_take(cursor, 8);
    if (!length_bytes) return 0;
    uint64_t length = read_u64(length_bytes);
    if (length > (1u << 20)) return 0;
    const unsigned char *text = cursor_take(cursor, (size_t)length);
    if (!text) return 0;
    char *copy = malloc((size_t)length + 1);
    if (!copy) return 0;
    memcpy(copy, text, (size_t)length);
    copy[length] = '\0';
    *out = copy;
    return 1;
}

static int cursor_skip_value(gguf_cursor *cursor, uint32_t type) {
    if (type == GGUF_STRING) {
        const unsigned char *length_bytes = cursor_take(cursor, 8);
        if (!length_bytes) return 0;
        return cursor_skip(cursor, read_u64(length_bytes));
    }
    if (type == GGUF_ARRAY) {
        const unsigned char *head = cursor_take(cursor, 12);
        if (!head) return 0;
        uint32_t item = read_u32(head);
        uint64_t count = read_u64(head + 4);
        if (item == GGUF_STRING || item == GGUF_ARRAY) {
            for (uint64_t index = 0; index < count; index++)
                if (!cursor_skip_value(cursor, item)) return 0;
            return 1;
        }
        int size = scalar_size(item);
        if (!size) return 0;
        return cursor_skip(cursor, count * (uint64_t)size);
    }
    int size = scalar_size(type);
    if (!size) return 0;
    return cursor_skip(cursor, (uint64_t)size);
}

/* ------------------------------------------------------------ name mapping */

static int starts_with(const char *text, const char *prefix) {
    return strncmp(text, prefix, strlen(prefix)) == 0;
}

/* llama.cpp block suffixes -> HF module names. */
static const struct {
    const char *llama;
    const char *hf;
} llama_suffixes[] = {
    {"attn_q.weight", "self_attn.q_proj.weight"},
    {"attn_k.weight", "self_attn.k_proj.weight"},
    {"attn_v.weight", "self_attn.v_proj.weight"},
    {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_q_norm.weight", "self_attn.q_norm.weight"},
    {"attn_k_norm.weight", "self_attn.k_norm.weight"},
    {"attn_norm.weight", "input_layernorm.weight"},
    {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"},
    {"ffn_up.weight", "mlp.up_proj.weight"},
    {"ffn_down.weight", "mlp.down_proj.weight"}
};

/* Canonical spelling used inside this reader:
 *   model.layers.N.<module>        language decoder layers
 *   model.embed_tokens.weight      input embedding
 *   visual.<module>                vision tower
 * Both the HF spelling the engine asks for ("model.language_model.layers.N")
 * and the llama.cpp spelling ("blk.N.attn_q") collapse onto it. */
void qzi_gguf_canonical_name(const char *name, char *out, size_t size) {
    if (!name || !*name) {
        snprintf(out, size, "%s", "");
        return;
    }
    if (starts_with(name, "model.language_model.")) {
        snprintf(out, size, "model.%s", name + strlen("model.language_model."));
        return;
    }
    if (starts_with(name, "model.visual.")) {
        snprintf(out, size, "%s", name + strlen("model."));
        return;
    }
    if (starts_with(name, "language_model.")) {
        snprintf(out, size, "model.%s", name + strlen("language_model."));
        return;
    }
    if (starts_with(name, "blk.")) {
        const char *rest = name + 4;
        char *end = NULL;
        long layer = strtol(rest, &end, 10);
        if (end && *end == '.') {
            const char *suffix = end + 1;
            for (size_t index = 0;
                 index < sizeof(llama_suffixes) / sizeof(llama_suffixes[0]);
                 index++) {
                if (strcmp(suffix, llama_suffixes[index].llama) == 0) {
                    snprintf(out, size, "model.layers.%ld.%s", layer,
                             llama_suffixes[index].hf);
                    return;
                }
            }
        }
        snprintf(out, size, "%s", name);
        return;
    }
    if (strcmp(name, "token_embd.weight") == 0) {
        snprintf(out, size, "model.embed_tokens.weight");
        return;
    }
    if (strcmp(name, "output_norm.weight") == 0) {
        snprintf(out, size, "model.norm.weight");
        return;
    }
    snprintf(out, size, "%s", name);
}

/* --------------------------------------------------------------- open */

qzi_gguf *qzi_gguf_open(const char *path, char *error, size_t error_size) {
    if (!path || !*path) {
        fail(error, error_size, "gguf path is required");
        return NULL;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fail(error, error_size, "cannot open gguf: %s", path);
        return NULL;
    }
    qzi_gguf *gguf = calloc(1, sizeof(*gguf));
    if (!gguf) {
        close(fd);
        fail(error, error_size, "out of memory opening %s", path);
        return NULL;
    }
    gguf->fd = -1;
    gguf->fd = fd;
    gguf->path = strdup(path);
    uint64_t alignment = 32;

    gguf_cursor cursor = {0};
    cursor.gguf = gguf;
    cursor.position = 0;

    const unsigned char *head = cursor_take(&cursor, 24);
    if (!head || memcmp(head, "GGUF", 4) != 0) {
        fail(error, error_size, "not a gguf file: %s", path);
        qzi_gguf_free(gguf);
        return NULL;
    }
    uint32_t version = read_u32(head + 4);
    if (version < 2 || version > 3) {
        fail(error, error_size, "unsupported gguf version %u in %s", version,
             path);
        qzi_gguf_free(gguf);
        return NULL;
    }
    uint64_t tensor_count = read_u64(head + 8);
    uint64_t kv_count = read_u64(head + 16);
    if (tensor_count == 0 || tensor_count > (1u << 22)) {
        fail(error, error_size, "gguf %s reports %llu tensors", path,
             (unsigned long long)tensor_count);
        qzi_gguf_free(gguf);
        return NULL;
    }

    /* orig_shape overrides, collected while walking the metadata. */
    typedef struct { char *name; uint64_t shape[8]; uint32_t ndim; } override;
    override *overrides = NULL;
    size_t override_count = 0, override_capacity = 0;

    for (uint64_t index = 0; index < kv_count; index++) {
        char *key = NULL;
        if (!cursor_string(&cursor, &key)) {
            fail(error, error_size, "truncated gguf metadata in %s", path);
            goto failed;
        }
        const unsigned char *type_bytes = cursor_take(&cursor, 4);
        if (!type_bytes) {
            free(key);
            fail(error, error_size, "truncated gguf metadata in %s", path);
            goto failed;
        }
        uint32_t type = read_u32(type_bytes);
        if (starts_with(key, "comfy.gguf.orig_shape.") && type == GGUF_ARRAY) {
            const unsigned char *array_head = cursor_take(&cursor, 12);
            if (!array_head) {
                free(key);
                fail(error, error_size, "truncated gguf metadata in %s", path);
                goto failed;
            }
            uint32_t item = read_u32(array_head);
            uint64_t count = read_u64(array_head + 4);
            /* ComfyUI writes int32 arrays; a plain GGUF writer may use uint32. */
            if ((item != GGUF_UINT32 && item != GGUF_INT32) || count == 0 ||
                count > 8) {
                free(key);
                fail(error, error_size, "bad orig_shape entry in %s", path);
                goto failed;
            }
            const unsigned char *values = cursor_take(&cursor, 4 * (size_t)count);
            if (!values) {
                free(key);
                fail(error, error_size, "truncated gguf metadata in %s", path);
                goto failed;
            }
            if (override_count == override_capacity) {
                size_t next = override_capacity ? override_capacity * 2 : 16;
                override *grown = realloc(overrides, next * sizeof(*grown));
                if (!grown) {
                    free(key);
                    fail(error, error_size, "out of memory reading %s", path);
                    goto failed;
                }
                overrides = grown;
                override_capacity = next;
            }
            override *entry = &overrides[override_count++];
            entry->name = strdup(key + strlen("comfy.gguf.orig_shape."));
            entry->ndim = (uint32_t)count;
            for (uint64_t dim = 0; dim < count; dim++)
                entry->shape[dim] = read_u32(values + 4 * dim);
            free(key);
            continue;
        }
        if (type == GGUF_UINT32 && strcmp(key, "general.alignment") == 0) {
            const unsigned char *value = cursor_take(&cursor, 4);
            if (!value) {
                free(key);
                fail(error, error_size, "truncated gguf metadata in %s", path);
                goto failed;
            }
            uint64_t parsed = read_u32(value);
            if (parsed >= 1 && parsed <= 4096) alignment = parsed;
            free(key);
            continue;
        }
        free(key);
        if (!cursor_skip_value(&cursor, type)) {
            fail(error, error_size, "truncated gguf metadata in %s", path);
            goto failed;
        }
    }

    gguf->tensors = calloc((size_t)tensor_count, sizeof(*gguf->tensors));
    if (!gguf->tensors) {
        fail(error, error_size, "out of memory reading %s", path);
        goto failed;
    }
    gguf->count = (size_t)tensor_count;
    for (size_t index = 0; index < gguf->count; index++) {
        qzi_gguf_tensor *tensor = &gguf->tensors[index];
        if (!cursor_string(&cursor, &tensor->stored)) {
            fail(error, error_size, "truncated tensor table in %s", path);
            goto failed;
        }
        const unsigned char *rank_bytes = cursor_take(&cursor, 4);
        if (!rank_bytes) {
            fail(error, error_size, "truncated tensor table in %s", path);
            goto failed;
        }
        uint32_t ndim = read_u32(rank_bytes);
        if (ndim == 0 || ndim > 8) {
            fail(error, error_size, "tensor %s has %u dimensions",
                 tensor->stored, ndim);
            goto failed;
        }
        tensor->ndim = ndim;
        const unsigned char *dims = cursor_take(&cursor, 8 * (size_t)ndim);
        if (!dims) {
            fail(error, error_size, "truncated tensor table in %s", path);
            goto failed;
        }
        /* Copy before the next take(): refilling the cursor can move the
         * buffer out from under this pointer. */
        uint64_t raw[8];
        for (uint32_t dim = 0; dim < ndim; dim++)
            raw[dim] = read_u64(dims + 8 * dim);
        const unsigned char *tail = cursor_take(&cursor, 12);
        if (!tail) {
            fail(error, error_size, "truncated tensor table in %s", path);
            goto failed;
        }
        tensor->type = read_u32(tail);
        tensor->offset = read_u64(tail + 4);
        /* GGUF stores ne[0] fastest; torch row-major order is the reverse. */
        for (uint32_t dim = 0; dim < ndim; dim++)
            tensor->shape[dim] = raw[ndim - 1 - dim];
        uint64_t elements = 1;
        for (uint32_t dim = 0; dim < ndim; dim++) {
            if (tensor->shape[dim] && elements > UINT64_MAX / tensor->shape[dim]) {
                fail(error, error_size, "tensor %s overflows", tensor->stored);
                goto failed;
            }
            elements *= tensor->shape[dim];
        }
        tensor->elements = elements;
        qzi_gguf_layout layout = layout_of(tensor->type);
        if (!layout.block) {
            fail(error, error_size,
                 "tensor %s uses gguf type %u (%s), which this build cannot "
                 "dequantise", tensor->stored, tensor->type,
                 qzi_gguf_type_name((qzi_gguf_type)tensor->type));
            goto failed;
        }
        if (elements % layout.block) {
            fail(error, error_size,
                 "tensor %s has %llu elements, not a multiple of %u", 
                 tensor->stored, (unsigned long long)elements, layout.block);
            goto failed;
        }
        tensor->bytes = elements / layout.block * layout.bytes;
        char canonical[512];
        qzi_gguf_canonical_name(tensor->stored, canonical, sizeof(canonical));
        tensor->name = strdup(canonical);
        for (size_t over = 0; over < override_count; over++) {
            if (strcmp(overrides[over].name, canonical) != 0 &&
                strcmp(overrides[over].name, tensor->stored) != 0)
                continue;
            tensor->ndim = overrides[over].ndim;
            for (uint32_t dim = 0; dim < tensor->ndim; dim++)
                tensor->shape[dim] = overrides[over].shape[dim];
        }
    }

    /* Tensor payloads start at the next multiple of the alignment after the
     * header, which is where the cursor now sits. */
    uint64_t header_end = cursor.position + cursor.consumed;
    gguf->data_start = (header_end + alignment - 1) / alignment * alignment;
    free(cursor.buffer);
    for (size_t index = 0; index < override_count; index++) free(overrides[index].name);
    free(overrides);
    return gguf;

failed:
    free(cursor.buffer);
    for (size_t index = 0; index < override_count; index++) free(overrides[index].name);
    free(overrides);
    qzi_gguf_free(gguf);
    return NULL;
}

void qzi_gguf_free(qzi_gguf *gguf) {
    if (!gguf) return;
    if (gguf->tensors) {
        for (size_t index = 0; index < gguf->count; index++) {
            free(gguf->tensors[index].name);
            free(gguf->tensors[index].stored);
        }
        free(gguf->tensors);
    }
    free(gguf->scratch);
    free(gguf->path);
    if (gguf->fd >= 0) close(gguf->fd);
    free(gguf);
}

size_t qzi_gguf_tensor_count(const qzi_gguf *gguf) {
    return gguf ? gguf->count : 0;
}

int qzi_gguf_find(const qzi_gguf *gguf, const char *name) {
    if (!gguf || !name || !*name) return -1;
    char canonical[512];
    qzi_gguf_canonical_name(name, canonical, sizeof(canonical));
    for (size_t index = 0; index < gguf->count; index++)
        if (strcmp(gguf->tensors[index].name, canonical) == 0)
            return (int)index;
    return -1;
}

const char *qzi_gguf_tensor_name(const qzi_gguf *gguf, size_t index) {
    return gguf && index < gguf->count ? gguf->tensors[index].stored : NULL;
}

qzi_gguf_type qzi_gguf_tensor_type(const qzi_gguf *gguf, size_t index) {
    return gguf && index < gguf->count
        ? (qzi_gguf_type)gguf->tensors[index].type : QZI_GGUF_UNKNOWN;
}

int qzi_gguf_tensor_rank(const qzi_gguf *gguf, size_t index) {
    return gguf && index < gguf->count ? (int)gguf->tensors[index].ndim : 0;
}

const uint64_t *qzi_gguf_tensor_shape(const qzi_gguf *gguf, size_t index) {
    return gguf && index < gguf->count ? gguf->tensors[index].shape : NULL;
}

uint64_t qzi_gguf_tensor_elements(const qzi_gguf *gguf, size_t index) {
    return gguf && index < gguf->count ? gguf->tensors[index].elements : 0;
}

/* ---------------------------------------------------------- dequantisers */

/* Q4_K/Q5_K share the packed 6-bit scale/min layout: 12 bytes hold 8 pairs of
 * (scale, min), 6 bits each, split across two halves. */
static void k_scale_min(int group, const unsigned char *packed,
                        unsigned char *scale, unsigned char *min) {
    if (group < 4) {
        *scale = packed[group] & 63u;
        *min = packed[group + 4] & 63u;
    } else {
        *scale = (unsigned char)((packed[group + 4] & 0x0fu) |
                                 ((packed[group - 4] >> 6) << 4));
        *min = (unsigned char)((packed[group + 4] >> 4) |
                               ((packed[group] >> 6) << 4));
    }
}

#define DEQUANT_EMIT(value)                                                    \
    do {                                                                       \
        if (to_f32) destination[index] = (value);                              \
        else bf16_destination[index] = float_to_bf16(value);                   \
    } while (0)

static void dequant_q4_k(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    float min = half_to_float((uint16_t)read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *codes = block + 16;
    size_t index = 0;
    for (int chunk = 0; chunk < 4; chunk++) {
        for (int half = 0; half < 2; half++) {
            unsigned char scale, minimum;
            k_scale_min(chunk * 2 + half, scales, &scale, &minimum);
            float factor = d * (float)scale;
            float offset = min * (float)minimum;
            const unsigned char *codes_at = codes + chunk * 32;
            for (int step = 0; step < 32; step++) {
                int nibble = half == 0 ? (codes_at[step] & 0x0f)
                                       : (codes_at[step] >> 4);
                DEQUANT_EMIT(factor * (float)nibble - offset);
                index++;
            }
        }
    }
}

static void dequant_q5_k(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    float min = half_to_float((uint16_t)read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *high = block + 16;
    const unsigned char *codes = block + 48;
    size_t index = 0;
    unsigned bit1 = 1, bit2 = 2;
    for (int chunk = 0; chunk < 4; chunk++) {
        for (int half = 0; half < 2; half++) {
            unsigned char scale, minimum;
            k_scale_min(chunk * 2 + half, scales, &scale, &minimum);
            float factor = d * (float)scale;
            float offset = min * (float)minimum;
            const unsigned char *codes_at = codes + chunk * 32;
            unsigned mask = half == 0 ? bit1 : bit2;
            for (int step = 0; step < 32; step++) {
                int nibble = half == 0 ? (codes_at[step] & 0x0f)
                                       : (codes_at[step] >> 4);
                if (high[step] & mask) nibble += 16;
                DEQUANT_EMIT(factor * (float)nibble - offset);
                index++;
            }
        }
        bit1 <<= 2;
        bit2 <<= 2;
    }
}

static void dequant_q6_k(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    const unsigned char *low = block;
    const unsigned char *high = block + 128;
    const signed char *scales = (const signed char *)(block + 192);
    float d = half_to_float((uint16_t)read_u16(block + 208));
    /* Within each 128-element group the four sub-rows are interleaved: element
     * l, l+32, l+64, l+96 come from the same byte of low/high. */
    for (int chunk = 0; chunk < 2; chunk++) {
        size_t base = (size_t)chunk * 128;
        for (int step = 0; step < 32; step++) {
            int is = step / 16;
            int q1 = (int)((low[step] & 0x0f) | (((high[step] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((low[step + 32] & 0x0f) | (((high[step] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((low[step] >> 4) | (((high[step] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((low[step + 32] >> 4) | (((high[step] >> 6) & 3) << 4)) - 32;
            float v1 = d * (float)scales[is + 0] * (float)q1;
            float v2 = d * (float)scales[is + 2] * (float)q2;
            float v3 = d * (float)scales[is + 4] * (float)q3;
            float v4 = d * (float)scales[is + 6] * (float)q4;
            size_t at = base + (size_t)step;
            if (to_f32) {
                destination[at] = v1;
                destination[at + 32] = v2;
                destination[at + 64] = v3;
                destination[at + 96] = v4;
            } else {
                bf16_destination[at] = float_to_bf16(v1);
                bf16_destination[at + 32] = float_to_bf16(v2);
                bf16_destination[at + 64] = float_to_bf16(v3);
                bf16_destination[at + 96] = float_to_bf16(v4);
            }
        }
        low += 64;
        high += 32;
        scales += 8;
    }
}

static void dequant_q8_0(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    const signed char *codes = (const signed char *)(block + 2);
    for (int index = 0; index < 32; index++)
        DEQUANT_EMIT(d * (float)codes[index]);
}

static void dequant_q4_0(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    const unsigned char *codes = block + 2;
    for (int index = 0; index < 16; index++) {
        DEQUANT_EMIT(d * (float)((codes[index] & 0x0f) - 8));
    }
    for (int index = 0; index < 16; index++) {
        DEQUANT_EMIT(d * (float)((codes[index] >> 4) - 8));
    }
}

static void dequant_q4_1(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    float m = half_to_float((uint16_t)read_u16(block + 2));
    const unsigned char *codes = block + 4;
    for (int index = 0; index < 16; index++)
        DEQUANT_EMIT(d * (float)(codes[index] & 0x0f) + m);
    for (int index = 0; index < 16; index++)
        DEQUANT_EMIT(d * (float)(codes[index] >> 4) + m);
}

static void dequant_q5_0(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    uint32_t high = read_u32(block + 2);
    const unsigned char *codes = block + 6;
    for (int index = 0; index < 16; index++) {
        int value = (codes[index] & 0x0f) + (((high >> index) & 1u) ? 16 : 0) - 16;
        DEQUANT_EMIT(d * (float)value);
    }
    for (int index = 0; index < 16; index++) {
        int value = (codes[index] >> 4) + (((high >> (index + 16)) & 1u) ? 16 : 0) - 16;
        DEQUANT_EMIT(d * (float)value);
    }
}

static void dequant_q5_1(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float d = half_to_float((uint16_t)read_u16(block));
    float m = half_to_float((uint16_t)read_u16(block + 2));
    uint32_t high = read_u32(block + 4);
    const unsigned char *codes = block + 8;
    for (int index = 0; index < 16; index++) {
        int value = (codes[index] & 0x0f) + (((high >> index) & 1u) ? 16 : 0);
        DEQUANT_EMIT(d * (float)value + m);
    }
    for (int index = 0; index < 16; index++) {
        int value = (codes[index] >> 4) + (((high >> (index + 16)) & 1u) ? 16 : 0);
        DEQUANT_EMIT(d * (float)value + m);
    }
}

static void dequant_f16(const unsigned char *block, unsigned char *out,
                        int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    /* One "block" is one element for F16. */
    float value = half_to_float((uint16_t)read_u16(block));
    if (to_f32) destination[0] = value;
    else bf16_destination[0] = float_to_bf16(value);
}

static void dequant_bf16(const unsigned char *block, unsigned char *out,
                         int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float value = bf16_to_float((uint16_t)read_u16(block));
    if (to_f32) destination[0] = value;
    else bf16_destination[0] = float_to_bf16(value);
}

static void dequant_f32(const unsigned char *block, unsigned char *out,
                        int to_f32) {
    float *destination = (float *)out;
    uint16_t *bf16_destination = (uint16_t *)out;
    float value;
    uint32_t bits = read_u32(block);
    memcpy(&value, &bits, sizeof(value));
    if (to_f32) destination[0] = value;
    else bf16_destination[0] = float_to_bf16(value);
}

static void dequant_block(unsigned type, const unsigned char *block,
                          unsigned char *out, int to_f32) {
    switch (type) {
        case QZI_GGUF_Q4_K: dequant_q4_k(block, out, to_f32); break;
        case QZI_GGUF_Q5_K: dequant_q5_k(block, out, to_f32); break;
        case QZI_GGUF_Q6_K: dequant_q6_k(block, out, to_f32); break;
        case QZI_GGUF_Q8_0: dequant_q8_0(block, out, to_f32); break;
        case QZI_GGUF_Q4_0: dequant_q4_0(block, out, to_f32); break;
        case QZI_GGUF_Q4_1: dequant_q4_1(block, out, to_f32); break;
        case QZI_GGUF_Q5_0: dequant_q5_0(block, out, to_f32); break;
        case QZI_GGUF_Q5_1: dequant_q5_1(block, out, to_f32); break;
        case QZI_GGUF_F16: dequant_f16(block, out, to_f32); break;
        case QZI_GGUF_BF16: dequant_bf16(block, out, to_f32); break;
        case QZI_GGUF_F32: dequant_f32(block, out, to_f32); break;
        default: break;
    }
}

/* Reads `bytes` at an absolute offset, tolerating short reads. */
static int read_exact(int fd, void *buffer, size_t bytes, uint64_t offset) {
    unsigned char *cursor = buffer;
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, cursor + done, bytes - done,
                            (off_t)(offset + done));
        if (got <= 0) return 0;
        done += (size_t)got;
    }
    return 1;
}

/* Dequantises one tensor in bounded chunks: a 151936x5120 Q4_K embedding is a
 * 437 MiB payload and a 1.5 GiB BF16 destination, so neither the whole payload
 * nor a float staging buffer can be assumed to fit. */
static int read_tensor(const qzi_gguf *gguf, size_t index, void *out,
                       int to_f32, char *error, size_t error_size) {
    if (!gguf || index >= gguf->count) {
        fail(error, error_size, "gguf tensor index out of range");
        return 0;
    }
    const qzi_gguf_tensor *tensor = &gguf->tensors[index];
    qzi_gguf_layout layout = layout_of(tensor->type);
    if (!layout.block) {
        fail(error, error_size, "cannot dequantise %s (type %s)",
             tensor->stored, qzi_gguf_type_name((qzi_gguf_type)tensor->type));
        return 0;
    }
    const size_t chunk_target = 32u << 20;
    uint64_t blocks = tensor->elements / layout.block;
    uint64_t blocks_per_chunk = chunk_target / layout.bytes;
    if (blocks_per_chunk < 1) blocks_per_chunk = 1;
    if (blocks_per_chunk > blocks) blocks_per_chunk = blocks;
    size_t element_bytes = to_f32 ? sizeof(float) : sizeof(uint16_t);
    unsigned char *payload = malloc((size_t)blocks_per_chunk * layout.bytes);
    if (!payload) {
        fail(error, error_size, "out of memory dequantising %s", tensor->stored);
        return 0;
    }
    uint64_t block = 0;
    while (block < blocks) {
        uint64_t take = blocks - block < blocks_per_chunk
            ? blocks - block : blocks_per_chunk;
        uint64_t offset = gguf->data_start + tensor->offset +
                          block * layout.bytes;
        if (!read_exact(gguf->fd, payload, (size_t)take * layout.bytes,
                        offset)) {
            fail(error, error_size, "cannot read %s from %s", tensor->stored,
                 gguf->path);
            free(payload);
            return 0;
        }
        unsigned char *destination =
            (unsigned char *)out + block * layout.block * element_bytes;
        /* MSVC's OpenMP parser wants the index declared outside the loop. */
        long local;
        long block_count = (long)take;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (local = 0; local < block_count; local++) {
            dequant_block(tensor->type, payload + (size_t)local * layout.bytes,
                          destination + (size_t)local * layout.block *
                                            element_bytes,
                          to_f32);
        }
        block += take;
    }
    free(payload);
    return 1;
}

int qzi_gguf_read_bf16(const qzi_gguf *gguf, size_t index, uint16_t *out,
                      char *error, size_t error_size) {
    return read_tensor(gguf, index, out, 0, error, error_size);
}

int qzi_gguf_read_f32(const qzi_gguf *gguf, size_t index, float *out,
                     char *error, size_t error_size) {
    return read_tensor(gguf, index, out, 1, error, error_size);
}
