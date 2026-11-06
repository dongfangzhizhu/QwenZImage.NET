#include "qzen_quanto.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* --- BF16 <-> F32 ----------------------------------------------------------
 * BF16 is the top half of an F32; widening is a shift, narrowing rounds to
 * nearest even on the low 16 bits.  Only widening is needed here: the scale
 * multiply runs in F32 and the product is narrowed once at the end. */

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

/* "<name>" -> "<name>._data" / "<name>._scale"; quanto appends the component
 * after the Linear tensor name instead of replacing a suffix. */
static int quanto_component_name(const char *name, const char *component,
                                 char *out, size_t size) {
    int written = snprintf(out, size, "%s.%s", name, component);
    return written > 0 && (size_t)written < size;
}

static int tensor_is(const qzen_st_tensor *tensor, qzen_dtype dtype, int ndim,
                     uint64_t rows, uint64_t cols) {
    if (!tensor || tensor->dtype != dtype || tensor->ndim != ndim) return 0;
    if (ndim == 2 && (tensor->shape[0] != rows || tensor->shape[1] != cols)) {
        return 0;
    }
    if (ndim == 1 && tensor->shape[0] != rows) return 0;
    return 1;
}

int qzen_quanto_find(const qzen_weight_store *store, const char *name,
                   uint64_t rows, uint64_t cols,
                   const qzen_st_tensor **data, const qzen_st_header **data_header,
                   const qzen_st_tensor **scale, const qzen_st_header **scale_header,
                   char *error, size_t error_size) {
    if (data) *data = NULL;
    if (data_header) *data_header = NULL;
    if (scale) *scale = NULL;
    if (scale_header) *scale_header = NULL;
    if (!store || !name || !rows || !cols) {
        fail(error, error_size, "quanto lookup needs a store, a name and a shape");
        return 0;
    }
    char data_name[192];
    char scale_name[192];
    if (!quanto_component_name(name, "_data", data_name, sizeof(data_name)) ||
        !quanto_component_name(name, "_scale", scale_name, sizeof(scale_name))) {
        fail(error, error_size, "quanto name is too long: %s", name);
        return 0;
    }
    const qzen_st_header *found_data_header = NULL;
    const qzen_st_tensor *found_data = qzen_weight_find(store, data_name,
                                                    &found_data_header);
    if (!found_data) {
        fail(error, error_size, "required quanto weight is absent: %s", data_name);
        return 0;
    }
    if (!tensor_is(found_data, QZEN_DTYPE_I8, 2, rows, cols)) {
        fail(error, error_size,
             "quanto data %s is %s[%d] with shape [%llu, %llu], expected I8[2] [%llu, %llu]",
             data_name, qzen_dtype_name(found_data->dtype), found_data->ndim,
             (unsigned long long)found_data->shape[0],
             (unsigned long long)found_data->shape[1],
             (unsigned long long)rows, (unsigned long long)cols);
        return 0;
    }
    const qzen_st_header *found_scale_header = NULL;
    const qzen_st_tensor *found_scale = qzen_weight_find(store, scale_name,
                                                     &found_scale_header);
    if (!found_scale) {
        fail(error, error_size, "quanto weight %s is missing its scale %s",
             data_name, scale_name);
        return 0;
    }
    if (!tensor_is(found_scale, QZEN_DTYPE_BF16, 2, rows, 1)) {
        fail(error, error_size,
             "quanto scale %s is %s[%d] with shape [%llu, %llu], expected BF16[2] [%llu, 1]",
             scale_name, qzen_dtype_name(found_scale->dtype), found_scale->ndim,
             (unsigned long long)found_scale->shape[0],
             (unsigned long long)found_scale->shape[1],
             (unsigned long long)rows);
        return 0;
    }
    if (data) *data = found_data;
    if (data_header) *data_header = found_data_header;
    if (scale) *scale = found_scale;
    if (scale_header) *scale_header = found_scale_header;
    return 1;
}

int qzen_quanto_dequant_bf16(const qzen_weight_store *store, const char *name,
                           uint64_t rows, uint64_t cols, uint16_t *out,
                           char *error, size_t error_size) {
    if (!store || !name || !out) {
        fail(error, error_size, "quanto dequant needs a store, a name and an output");
        return 0;
    }
    const qzen_st_header *data_header = NULL;
    const qzen_st_tensor *data = NULL;
    const qzen_st_header *scale_header = NULL;
    const qzen_st_tensor *scale = NULL;
    if (!qzen_quanto_find(store, name, rows, cols, &data, &data_header,
                        &scale, &scale_header, error, error_size)) {
        return 0;
    }
    uint64_t elements = rows * cols;
    if (elements > SIZE_MAX / sizeof(int8_t) ||
        rows > SIZE_MAX / sizeof(uint16_t)) {
        fail(error, error_size, "quanto weight %s is too large for this process", name);
        return 0;
    }
    int8_t *payload = malloc((size_t)elements);
    uint16_t *scales = malloc((size_t)rows * sizeof(uint16_t));
    if (!payload || !scales) {
        fail(error, error_size, "cannot stage quanto weight %s for dequant", name);
        free(payload);
        free(scales);
        return 0;
    }
    if (!qzen_st_read_data(data_header, data, payload, (size_t)elements,
                         error, error_size)) {
        free(payload);
        free(scales);
        return 0;
    }
    if (!qzen_st_read_data(scale_header, scale, scales,
                         (size_t)rows * sizeof(uint16_t), error, error_size)) {
        free(payload);
        free(scales);
        return 0;
    }
    /* Rows are independent (each has its own scale and disjoint output),
     * so the dequant parallelizes over them.  Every element keeps the exact
     * same math as the serial loop; only the schedule changes. */
    {
        int64_t row;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row = 0; row < (int64_t)rows; row++) {
            float row_scale = bf16_to_f32(scales[row]);
            const int8_t *source = payload + (uint64_t)row * cols;
            uint16_t *destination = out + (uint64_t)row * cols;
            for (uint64_t column = 0; column < cols; column++) {
                destination[column] = f32_to_bf16((float)source[column] * row_scale);
            }
        }
    }
    free(payload);
    free(scales);
    return 1;
}

uint16_t *qzen_quanto_load_bf16(const qzen_weight_store *store, const char *name,
                              uint64_t rows, uint64_t cols,
                              char *error, size_t error_size) {
    if (rows > SIZE_MAX / cols / sizeof(uint16_t)) {
        fail(error, error_size, "quanto weight %s is too large for this process", name);
        return NULL;
    }
    uint16_t *out = malloc((size_t)(rows * cols) * sizeof(uint16_t));
    if (!out) {
        fail(error, error_size, "cannot allocate quanto weight %s", name);
        return NULL;
    }
    if (!qzen_quanto_dequant_bf16(store, name, rows, cols, out, error, error_size)) {
        free(out);
        return NULL;
    }
    return out;
}
