#include "qzen_safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#define QZEN_ST_MAX_HEADER (256u * 1024u * 1024u)

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
} qzen_json_cursor;

static int qzen_json_fail(qzen_json_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size) {
        snprintf(cursor->error, cursor->error_size, "%s", message);
    }
    return 0;
}

static void qzen_json_ws(qzen_json_cursor *cursor) {
    while (cursor->at < cursor->end &&
           (*cursor->at == ' ' || *cursor->at == '\n' ||
            *cursor->at == '\r' || *cursor->at == '\t')) cursor->at++;
}

static int qzen_json_take(qzen_json_cursor *cursor, char expected) {
    qzen_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != expected) {
        return qzen_json_fail(cursor, "malformed safetensors JSON header");
    }
    cursor->at++;
    return 1;
}

static int qzen_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static char *qzen_json_string(qzen_json_cursor *cursor) {
    qzen_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        qzen_json_fail(cursor, "expected JSON string");
        return NULL;
    }
    cursor->at++;
    size_t maximum = (size_t)(cursor->end - cursor->at);
    char *result = malloc(maximum + 1);
    if (!result) {
        qzen_json_fail(cursor, "out of memory parsing safetensors header");
        return NULL;
    }
    size_t length = 0;
    while (cursor->at < cursor->end && *cursor->at != '"') {
        unsigned char value = (unsigned char)*cursor->at++;
        if (value == '\\') {
            if (cursor->at >= cursor->end) goto malformed;
            value = (unsigned char)*cursor->at++;
            switch (value) {
                case '"': result[length++] = '"'; break;
                case '\\': result[length++] = '\\'; break;
                case '/': result[length++] = '/'; break;
                case 'b': result[length++] = '\b'; break;
                case 'f': result[length++] = '\f'; break;
                case 'n': result[length++] = '\n'; break;
                case 'r': result[length++] = '\r'; break;
                case 't': result[length++] = '\t'; break;
                case 'u': {
                    if (cursor->end - cursor->at < 4) goto malformed;
                    int codepoint = 0;
                    for (int index = 0; index < 4; index++) {
                        int digit = qzen_hex(cursor->at[index]);
                        if (digit < 0) goto malformed;
                        codepoint = codepoint * 16 + digit;
                    }
                    cursor->at += 4;
                    if (codepoint < 0x80) {
                        result[length++] = (char)codepoint;
                    } else if (codepoint < 0x800) {
                        result[length++] = (char)(0xc0 | (codepoint >> 6));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    } else {
                        result[length++] = (char)(0xe0 | (codepoint >> 12));
                        result[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    }
                    break;
                }
                default: goto malformed;
            }
        } else {
            if (value < 0x20) goto malformed;
            result[length++] = (char)value;
        }
    }
    if (cursor->at >= cursor->end) goto malformed;
    cursor->at++;
    result[length] = '\0';
    return result;

malformed:
    free(result);
    qzen_json_fail(cursor, "malformed JSON string escape");
    return NULL;
}

static int qzen_json_uint(qzen_json_cursor *cursor, uint64_t *result) {
    qzen_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at < '0' || *cursor->at > '9') {
        return qzen_json_fail(cursor, "expected unsigned JSON integer");
    }
    uint64_t value = 0;
    while (cursor->at < cursor->end && *cursor->at >= '0' && *cursor->at <= '9') {
        unsigned digit = (unsigned)(*cursor->at - '0');
        if (value > (UINT64_MAX - digit) / 10) {
            return qzen_json_fail(cursor, "integer overflow in safetensors header");
        }
        value = value * 10 + digit;
        cursor->at++;
    }
    *result = value;
    return 1;
}

static int qzen_json_skip(qzen_json_cursor *cursor);

static int qzen_json_skip_compound(qzen_json_cursor *cursor, char open, char close) {
    if (!qzen_json_take(cursor, open)) return 0;
    qzen_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == close) {
        cursor->at++;
        return 1;
    }
    for (;;) {
        if (open == '{') {
            char *key = qzen_json_string(cursor);
            if (!key) return 0;
            free(key);
            if (!qzen_json_take(cursor, ':')) return 0;
        }
        if (!qzen_json_skip(cursor)) return 0;
        qzen_json_ws(cursor);
        if (cursor->at >= cursor->end) return qzen_json_fail(cursor, "unterminated JSON value");
        if (*cursor->at == close) {
            cursor->at++;
            return 1;
        }
        if (*cursor->at++ != ',') return qzen_json_fail(cursor, "expected JSON comma");
    }
}

static int qzen_json_skip(qzen_json_cursor *cursor) {
    qzen_json_ws(cursor);
    if (cursor->at >= cursor->end) return qzen_json_fail(cursor, "missing JSON value");
    if (*cursor->at == '"') {
        char *value = qzen_json_string(cursor);
        if (!value) return 0;
        free(value);
        return 1;
    }
    if (*cursor->at == '{') return qzen_json_skip_compound(cursor, '{', '}');
    if (*cursor->at == '[') return qzen_json_skip_compound(cursor, '[', ']');
    const char *start = cursor->at;
    while (cursor->at < cursor->end && *cursor->at != ',' &&
           *cursor->at != '}' && *cursor->at != ']' &&
           *cursor->at != ' ' && *cursor->at != '\n' &&
           *cursor->at != '\r' && *cursor->at != '\t') cursor->at++;
    if (cursor->at == start) return qzen_json_fail(cursor, "invalid JSON scalar");
    return 1;
}

static qzen_dtype qzen_parse_dtype(const char *name) {
    if (!strcmp(name, "BOOL")) return QZEN_DTYPE_BOOL;
    if (!strcmp(name, "I8")) return QZEN_DTYPE_I8;
    if (!strcmp(name, "U8")) return QZEN_DTYPE_U8;
    if (!strcmp(name, "I16")) return QZEN_DTYPE_I16;
    if (!strcmp(name, "U16")) return QZEN_DTYPE_U16;
    if (!strcmp(name, "F16")) return QZEN_DTYPE_F16;
    if (!strcmp(name, "BF16")) return QZEN_DTYPE_BF16;
    if (!strcmp(name, "I32")) return QZEN_DTYPE_I32;
    if (!strcmp(name, "U32")) return QZEN_DTYPE_U32;
    if (!strcmp(name, "F32")) return QZEN_DTYPE_F32;
    if (!strcmp(name, "I64")) return QZEN_DTYPE_I64;
    if (!strcmp(name, "U64")) return QZEN_DTYPE_U64;
    if (!strcmp(name, "F64")) return QZEN_DTYPE_F64;
    if (!strcmp(name, "F8_E4M3")) return QZEN_DTYPE_F8_E4M3;
    if (!strcmp(name, "F8_E5M2")) return QZEN_DTYPE_F8_E5M2;
    return QZEN_DTYPE_UNKNOWN;
}

size_t qzen_dtype_size(qzen_dtype dtype) {
    switch (dtype) {
        case QZEN_DTYPE_BOOL:
        case QZEN_DTYPE_I8:
        case QZEN_DTYPE_U8: return 1;
        case QZEN_DTYPE_I16:
        case QZEN_DTYPE_U16:
        case QZEN_DTYPE_F16:
        case QZEN_DTYPE_BF16: return 2;
        case QZEN_DTYPE_I32:
        case QZEN_DTYPE_U32:
        case QZEN_DTYPE_F32: return 4;
        case QZEN_DTYPE_I64:
        case QZEN_DTYPE_U64:
        case QZEN_DTYPE_F64: return 8;
        case QZEN_DTYPE_F8_E4M3:
        case QZEN_DTYPE_F8_E5M2: return 1;
        case QZEN_DTYPE_UNKNOWN: return 0;
    }
    return 0;
}

const char *qzen_dtype_name(qzen_dtype dtype) {
    switch (dtype) {
        case QZEN_DTYPE_BOOL: return "BOOL";
        case QZEN_DTYPE_I8: return "I8";
        case QZEN_DTYPE_U8: return "U8";
        case QZEN_DTYPE_I16: return "I16";
        case QZEN_DTYPE_U16: return "U16";
        case QZEN_DTYPE_F16: return "F16";
        case QZEN_DTYPE_BF16: return "BF16";
        case QZEN_DTYPE_I32: return "I32";
        case QZEN_DTYPE_U32: return "U32";
        case QZEN_DTYPE_F32: return "F32";
        case QZEN_DTYPE_I64: return "I64";
        case QZEN_DTYPE_U64: return "U64";
        case QZEN_DTYPE_F64: return "F64";
        case QZEN_DTYPE_F8_E4M3: return "F8_E4M3";
        case QZEN_DTYPE_F8_E5M2: return "F8_E5M2";
        case QZEN_DTYPE_UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

static int qzen_parse_shape(qzen_json_cursor *cursor, qzen_st_tensor *tensor) {
    if (!qzen_json_take(cursor, '[')) return 0;
    qzen_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == ']') {
        cursor->at++;
        tensor->ndim = 0;
        return 1;
    }
    int dimensions = 0;
    for (;;) {
        uint64_t value;
        if (dimensions >= 8) return qzen_json_fail(cursor, "tensor rank exceeds 8");
        if (!qzen_json_uint(cursor, &value)) return 0;
        tensor->shape[dimensions++] = value;
        qzen_json_ws(cursor);
        if (cursor->at >= cursor->end) return qzen_json_fail(cursor, "unterminated shape");
        if (*cursor->at == ']') {
            cursor->at++;
            tensor->ndim = dimensions;
            return 1;
        }
        if (*cursor->at++ != ',') return qzen_json_fail(cursor, "expected shape comma");
    }
}

static int qzen_parse_offsets(qzen_json_cursor *cursor, qzen_st_tensor *tensor) {
    if (!qzen_json_take(cursor, '[') ||
        !qzen_json_uint(cursor, &tensor->data_begin) ||
        !qzen_json_take(cursor, ',') ||
        !qzen_json_uint(cursor, &tensor->data_end) ||
        !qzen_json_take(cursor, ']')) return 0;
    return 1;
}

static int qzen_parse_tensor(qzen_json_cursor *cursor, qzen_st_tensor *tensor) {
    if (!qzen_json_take(cursor, '{')) return 0;
    int have_dtype = 0;
    int have_shape = 0;
    int have_offsets = 0;
    for (;;) {
        qzen_json_ws(cursor);
        if (cursor->at < cursor->end && *cursor->at == '}') {
            cursor->at++;
            break;
        }
        char *key = qzen_json_string(cursor);
        if (!key || !qzen_json_take(cursor, ':')) {
            free(key);
            return 0;
        }
        int ok = 1;
        if (!strcmp(key, "dtype")) {
            char *value = qzen_json_string(cursor);
            if (!value) ok = 0;
            else {
                tensor->dtype = qzen_parse_dtype(value);
                have_dtype = tensor->dtype != QZEN_DTYPE_UNKNOWN;
            }
            free(value);
        } else if (!strcmp(key, "shape")) {
            ok = qzen_parse_shape(cursor, tensor);
            have_shape = ok;
        } else if (!strcmp(key, "data_offsets")) {
            ok = qzen_parse_offsets(cursor, tensor);
            have_offsets = ok;
        } else {
            ok = qzen_json_skip(cursor);
        }
        free(key);
        if (!ok) return 0;
        qzen_json_ws(cursor);
        if (cursor->at >= cursor->end) return qzen_json_fail(cursor, "unterminated tensor entry");
        if (*cursor->at == '}') continue;
        if (*cursor->at++ != ',') return qzen_json_fail(cursor, "expected tensor comma");
    }
    if (!have_dtype || !have_shape || !have_offsets) {
        return qzen_json_fail(cursor, "incomplete tensor descriptor");
    }
    return 1;
}

static int qzen_validate_tensor(qzen_json_cursor *cursor, qzen_st_tensor *tensor,
                              uint64_t data_start, uint64_t file_size) {
    if (tensor->data_end < tensor->data_begin ||
        tensor->data_end > file_size - data_start) {
        return qzen_json_fail(cursor, "tensor data offsets exceed file");
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] && elements > UINT64_MAX / tensor->shape[dimension]) {
            return qzen_json_fail(cursor, "tensor element count overflow");
        }
        elements *= tensor->shape[dimension];
    }
    size_t item_size = qzen_dtype_size(tensor->dtype);
    if (item_size && elements > UINT64_MAX / item_size) {
        return qzen_json_fail(cursor, "tensor byte count overflow");
    }
    uint64_t expected = elements * item_size;
    if (expected != tensor->data_end - tensor->data_begin) {
        return qzen_json_fail(cursor, "tensor shape does not match byte offsets");
    }
    tensor->file_offset = data_start + tensor->data_begin;
    return 1;
}

static int qzen_append_tensor(qzen_st_header *header, qzen_st_tensor tensor,
                            size_t *capacity, qzen_json_cursor *cursor) {
    if (header->tensor_count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 64;
        qzen_st_tensor *values = realloc(header->tensors, next * sizeof(*values));
        if (!values) return qzen_json_fail(cursor, "out of memory indexing tensors");
        header->tensors = values;
        *capacity = next;
    }
    header->tensors[header->tensor_count++] = tensor;
    return 1;
}

static uint64_t qzen_u64_le(const unsigned char bytes[8]) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++) {
        value |= (uint64_t)bytes[index] << (index * 8);
    }
    return value;
}

int qzen_st_read_header(const char *path, qzen_st_header *header,
                      char *error, size_t error_size) {
    if (!path || !header) return 0;
    memset(header, 0, sizeof(*header));
    if (error && error_size) error[0] = '\0';
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size) snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return 0;
    }
    struct stat status;
    unsigned char prefix[8];
    if (fstat(descriptor, &status) != 0 || status.st_size < 8 ||
        pread(descriptor, prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix)) {
        if (error && error_size) snprintf(error, error_size, "%s: invalid file", path);
        close(descriptor);
        return 0;
    }
    uint64_t file_size = (uint64_t)status.st_size;
    uint64_t header_size = qzen_u64_le(prefix);
    if (header_size > QZEN_ST_MAX_HEADER || header_size > file_size - 8) {
        if (error && error_size) snprintf(error, error_size, "%s: invalid header size", path);
        close(descriptor);
        return 0;
    }
    char *json = malloc((size_t)header_size + 1);
    if (!json || pread(descriptor, json, (size_t)header_size, 8) != (ssize_t)header_size) {
        if (error && error_size) snprintf(error, error_size, "%s: cannot read header", path);
        free(json);
        close(descriptor);
        return 0;
    }
    close(descriptor);
    json[header_size] = '\0';
    header->path = strdup(path);
    header->file_size = file_size;
    header->header_size = header_size;
    if (!header->path) {
        free(json);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    qzen_json_cursor cursor = {json, json + header_size, error, error_size};
    size_t capacity = 0;
    if (!qzen_json_take(&cursor, '{')) goto fail;
    qzen_json_ws(&cursor);
    if (cursor.at < cursor.end && *cursor.at == '}') cursor.at++;
    else for (;;) {
        char *name = qzen_json_string(&cursor);
        if (!name || !qzen_json_take(&cursor, ':')) {
            free(name);
            goto fail;
        }
        if (!strcmp(name, "__metadata__")) {
            free(name);
            if (!qzen_json_skip(&cursor)) goto fail;
        } else {
            qzen_st_tensor tensor;
            memset(&tensor, 0, sizeof(tensor));
            tensor.name = name;
            if (!qzen_parse_tensor(&cursor, &tensor) ||
                !qzen_validate_tensor(&cursor, &tensor, 8 + header_size, file_size) ||
                !qzen_append_tensor(header, tensor, &capacity, &cursor)) {
                free(tensor.name);
                goto fail;
            }
        }
        qzen_json_ws(&cursor);
        if (cursor.at >= cursor.end) {
            qzen_json_fail(&cursor, "unterminated safetensors object");
            goto fail;
        }
        if (*cursor.at == '}') {
            cursor.at++;
            break;
        }
        if (*cursor.at++ != ',') {
            qzen_json_fail(&cursor, "expected top-level comma");
            goto fail;
        }
    }
    qzen_json_ws(&cursor);
    if (cursor.at != cursor.end) {
        qzen_json_fail(&cursor, "trailing data in safetensors header");
        goto fail;
    }
    free(json);
    return 1;

fail:
    free(json);
    qzen_st_free_header(header);
    return 0;
}

void qzen_st_free_header(qzen_st_header *header) {
    if (!header) return;
    for (size_t index = 0; index < header->tensor_count; index++) {
        free(header->tensors[index].name);
    }
    free(header->tensors);
    free(header->path);
    memset(header, 0, sizeof(*header));
}

const qzen_st_tensor *qzen_st_find(const qzen_st_header *header, const char *name) {
    if (!header || !name) return NULL;
    for (size_t index = 0; index < header->tensor_count; index++) {
        if (!strcmp(header->tensors[index].name, name)) return &header->tensors[index];
    }
    return NULL;
}

uint64_t qzen_st_tensor_elements(const qzen_st_tensor *tensor) {
    if (!tensor) return 0;
    uint64_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] != 0 &&
            elements > UINT64_MAX / tensor->shape[dimension]) return 0;
        elements *= tensor->shape[dimension];
    }
    return elements;
}

int qzen_st_read_data(const qzen_st_header *header, const qzen_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!header || !header->path || !tensor || (!data && bytes != 0)) {
        if (error && error_size) snprintf(error, error_size, "invalid tensor read request");
        return 0;
    }
    uint64_t tensor_bytes = tensor->data_end - tensor->data_begin;
    if (tensor_bytes != bytes) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: expected %llu bytes, got %zu",
                     tensor->name ? tensor->name : "tensor",
                     (unsigned long long)tensor_bytes, bytes);
        }
        return 0;
    }
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: %s", header->path, strerror(errno));
        }
        return 0;
    }
    size_t done = 0;
    while (done < bytes) {
        size_t remaining = bytes - done;
        size_t chunk = remaining < (size_t)(1u << 30) ? remaining :
                                                            (size_t)(1u << 30);
        ssize_t count = pread(descriptor, (unsigned char *)data + done, chunk,
                              (off_t)(tensor->file_offset + done));
        if (count <= 0) {
            if (error && error_size) {
                snprintf(error, error_size, "%s: cannot read %s: %s", header->path,
                         tensor->name ? tensor->name : "tensor",
                         count < 0 ? strerror(errno) : "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        done += (size_t)count;
    }
    close(descriptor);
    return 1;
}

static int qzen_has_suffix(const char *value, const char *suffix) {
    size_t length = strlen(value);
    size_t suffix_length = strlen(suffix);
    return length >= suffix_length && !strcmp(value + length - suffix_length, suffix);
}

int qzen_st_inventory_dir(const char *directory, qzen_component_info *info,
                        char *error, size_t error_size) {
    if (!directory || !info) return 0;
    memset(info, 0, sizeof(*info));
    DIR *stream = opendir(directory);
    if (!stream) {
        if (error && error_size) snprintf(error, error_size, "%s: %s", directory, strerror(errno));
        return 0;
    }
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (entry->d_name[0] == '.' || !qzen_has_suffix(entry->d_name, ".safetensors")) continue;
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        char *path = malloc(length);
        if (!path) {
            if (error && error_size) snprintf(error, error_size, "out of memory");
            closedir(stream);
            return 0;
        }
        snprintf(path, length, "%s/%s", directory, entry->d_name);
        qzen_st_header header;
        int ok = qzen_st_read_header(path, &header, error, error_size);
        free(path);
        if (!ok) {
            closedir(stream);
            return 0;
        }
        info->files++;
        info->bytes += header.file_size;
        info->tensors += header.tensor_count;
        for (size_t index = 0; index < header.tensor_count; index++) {
            info->tensor_bytes += header.tensors[index].data_end -
                                  header.tensors[index].data_begin;
        }
        qzen_st_free_header(&header);
    }
    closedir(stream);
    if (!info->files) {
        if (error && error_size) snprintf(error, error_size, "%s: no safetensors files", directory);
        return 0;
    }
    return 1;
}

static float qzen_half_to_float(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* In-place 256-point regular Hadamard (R(256), not Sylvester): four butterfly
 * passes over strides 1, 4, 16, 64 with the 4x4 kernel [[1,1,1,-1] rows].
 * The 256 block is viewed as (256/(4*stride), 4, stride) and the four points of
 * a group are 4*stride apart inside one lane. The 1/16 normalisation is folded
 * into the INT8 scale by the caller so it is applied exactly once. */
static void qzen_convrot_hadamard_256(float *block) {
    static const int strides[] = {1, 4, 16, 64};
    for (size_t pass = 0; pass < sizeof(strides) / sizeof(strides[0]); pass++) {
        int stride = strides[pass];
        int groups = 256 / (4 * stride);
        for (int group = 0; group < groups; group++) {
            float *base = block + (size_t)group * 4 * stride;
            for (int lane = 0; lane < stride; lane++) {
                float x0 = base[lane];
                float x1 = base[stride + lane];
                float x2 = base[2 * stride + lane];
                float x3 = base[3 * stride + lane];
                float sum = (x0 + x1) + (x2 + x3);
                base[lane] = sum - 2.0f * x3;
                base[stride + lane] = sum - 2.0f * x2;
                base[2 * stride + lane] = sum - 2.0f * x1;
                base[3 * stride + lane] = sum - 2.0f * x0;
            }
        }
    }
}

/* Telemetry for the host-side ConvRot dequantiser: the checkpoint read itself
 * is I/O bound, so the report separates arithmetic from disk time. */
static double qzen_convrot_dequant_seconds;
static uint64_t qzen_convrot_dequant_elements;

void qzen_st_reset_dequant_stats(void) {
    qzen_convrot_dequant_seconds = 0.0;
    qzen_convrot_dequant_elements = 0;
}

void qzen_st_dequant_stats(double *seconds, uint64_t *elements) {
    if (seconds) *seconds = qzen_convrot_dequant_seconds;
    if (elements) *elements = qzen_convrot_dequant_elements;
}

static double qzen_monotonic_seconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

typedef struct {
    const int8_t *quantized;
    float *values;
    const float *scales;
    uint64_t rows;
    uint64_t columns;
    uint64_t row_start;
    uint64_t row_end;
} qzen_convrot_dequant_job;

static void qzen_convrot_dequant_rows(qzen_convrot_dequant_job *job) {
    const uint64_t columns = job->columns;
    for (uint64_t row = job->row_start; row < job->row_end; row++) {
        /* 1/16 folded into the row scale: scaling by 2^-4 is exact, so this is
         * bit-identical to rotating first and dividing the block afterwards. */
        const float factor = job->scales[row] * 0.0625f;
        const int8_t *source = job->quantized + row * columns;
        float *target = job->values + row * columns;
        for (uint64_t index = 0; index < columns; index++) {
            target[index] = (float)source[index] * factor;
        }
        for (uint64_t group = 0; group < columns; group += 256) {
            qzen_convrot_hadamard_256(target + group);
        }
    }
}

static void *qzen_convrot_dequant_thread(void *opaque) {
    qzen_convrot_dequant_rows(opaque);
    return NULL;
}

static int qzen_convrot_dequantize(const int8_t *quantized, const float *scales,
                                 uint64_t rows, uint64_t columns,
                                 float *values) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    if ((uint64_t)cores > rows) cores = (long)rows;
    if (cores > 32) cores = 32;
    if (cores <= 1) {
        qzen_convrot_dequant_job job = {quantized, values, scales, rows, columns,
                                      0, rows};
        qzen_convrot_dequant_rows(&job);
        return 1;
    }
    pthread_t threads[32];
    qzen_convrot_dequant_job jobs[32];
    long started = 0;
    size_t created = 0;
    for (long index = 0; index < cores; index++) {
        uint64_t end = rows * (uint64_t)(index + 1) / (uint64_t)cores;
        jobs[index] = (qzen_convrot_dequant_job){quantized, values, scales, rows,
                                               columns, (uint64_t)started, end};
        started = (long)end;
        if (end == jobs[index].row_start) continue;
        if (pthread_create(&threads[created], NULL, qzen_convrot_dequant_thread,
                           &jobs[index]) != 0) break;
        created++;
    }
    int ok = created > 0;
    for (size_t index = 0; index < created; index++) {
        pthread_join(threads[index], NULL);
    }
    return ok;
}

int qzen_st_read_f32(const qzen_st_header *header, const qzen_st_tensor *tensor,
                   const qzen_st_header *scale_header,
                   const qzen_st_tensor *scale, float **values,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (values) *values = NULL;
    if (!header || !tensor || !values) {
        if (error && error_size) snprintf(error, error_size, "invalid F32 read request");
        return 0;
    }
    uint64_t elements = qzen_st_tensor_elements(tensor);
    if (!elements || elements > SIZE_MAX / sizeof(float)) {
        if (error && error_size) snprintf(error, error_size, "%s: unsupported element count", tensor->name ? tensor->name : "tensor");
        return 0;
    }
    float *buffer = malloc((size_t)elements * sizeof(*buffer));
    if (!buffer) {
        if (error && error_size) snprintf(error, error_size, "%s: out of memory converting %llu elements", tensor->name ? tensor->name : "tensor", (unsigned long long)elements);
        return 0;
    }
    if (tensor->dtype == QZEN_DTYPE_F32) {
        if (!qzen_st_read_data(header, tensor, buffer, (size_t)elements * sizeof(*buffer), error, error_size)) {
            free(buffer);
            return 0;
        }
        *values = buffer;
        return 1;
    }
    if (tensor->dtype == QZEN_DTYPE_F16) {
        uint16_t *halves = malloc((size_t)elements * sizeof(*halves));
        if (!halves) {
            if (error && error_size) snprintf(error, error_size, "%s: out of memory reading F16 payload", tensor->name ? tensor->name : "tensor");
            free(buffer);
            return 0;
        }
        int ok = qzen_st_read_data(header, tensor, halves, (size_t)elements * sizeof(*halves), error, error_size);
        for (uint64_t index = 0; ok && index < elements; index++) {
            buffer[index] = qzen_half_to_float(halves[index]);
        }
        free(halves);
        if (!ok) {
            free(buffer);
            return 0;
        }
        *values = buffer;
        return 1;
    }
    if (tensor->dtype == QZEN_DTYPE_I8) {
        const char *name = tensor->name ? tensor->name : "tensor";
        if (tensor->ndim != 2) {
            if (error && error_size) snprintf(error, error_size, "%s: INT8 ConvRot weights must be 2-D, got rank %d", name, tensor->ndim);
            free(buffer);
            return 0;
        }
        uint64_t rows = tensor->shape[0], columns = tensor->shape[1];
        if (!rows || !columns || columns % 256) {
            if (error && error_size) snprintf(error, error_size, "%s: INT8 ConvRot input size %llu is not a multiple of 256", name, (unsigned long long)columns);
            free(buffer);
            return 0;
        }
        if (!scale || !scale_header || scale->dtype != QZEN_DTYPE_F32) {
            if (error && error_size) snprintf(error, error_size, "%s: INT8 ConvRot weight needs an F32 weight_scale sidecar", name);
            free(buffer);
            return 0;
        }
        if (qzen_st_tensor_elements(scale) != rows) {
            if (error && error_size) snprintf(error, error_size, "%s: weight_scale has %llu values, expected %llu rows", name, (unsigned long long)qzen_st_tensor_elements(scale), (unsigned long long)rows);
            free(buffer);
            return 0;
        }
        int8_t *quantized = malloc((size_t)elements);
        float *scales = malloc((size_t)rows * sizeof(*scales));
        if (!quantized || !scales) {
            if (error && error_size) snprintf(error, error_size, "%s: out of memory reading INT8 ConvRot payload", name);
            free(buffer);
            free(quantized);
            free(scales);
            return 0;
        }
        int ok = qzen_st_read_data(header, tensor, quantized, (size_t)elements, error, error_size) &&
                 qzen_st_read_data(scale_header, scale, scales, (size_t)rows * sizeof(*scales), error, error_size);
        if (ok) {
            double started = qzen_monotonic_seconds();
            ok = qzen_convrot_dequantize(quantized, scales, rows, columns, buffer);
            qzen_convrot_dequant_seconds += qzen_monotonic_seconds() - started;
            qzen_convrot_dequant_elements += elements;
        }
        if (!ok && error && error_size && !error[0]) {
            snprintf(error, error_size, "%s: cannot dequantise INT8 ConvRot weight", name);
        }
        free(quantized);
        free(scales);
        if (!ok) {
            free(buffer);
            return 0;
        }
        *values = buffer;
        return 1;
    }
    if (error && error_size) snprintf(error, error_size, "%s: dtype %s cannot be normalised to F32", tensor->name ? tensor->name : "tensor", qzen_dtype_name(tensor->dtype));
    free(buffer);
    return 0;
}
