#include "qzen_weights.h"
#include "qzi_gguf.h"

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct qzen_weight_store {
    qzen_st_header *headers;
    size_t count;
    /* Non-NULL when the component is a GGUF container. Tensors are presented
     * through a synthetic header so every component loader can stay agnostic
     * of the container format. */
    qzi_gguf *gguf;
    qzen_st_tensor gguf_slots[4]; /* ring: weight + scale lookups coexist */
    qzen_st_header gguf_header;
    unsigned gguf_next;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* ---------------------------------------------------------------- FP8 -> BF16
 * Hugging Face FP8 checkpoints scale the whole tensor by a scalar stored either
 * in __metadata__ ("weight_scale"/"scale") or in a sibling "<name>_scale" /
 * "<name minus .weight>.weight_scale" tensor. We decode E4M3/E5M2 to F32, apply
 * the scale, then round once to BF16 (the same rounding the BF16 loader uses). */

static uint16_t f32_to_bf16_rn(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* round-to-nearest-even, matching qzen_qwen21_dit.c's q21_f32_to_bf16 */
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float fp8_e4m3_to_f32(uint8_t byte) {
    int sign = (byte & 0x80) ? -1 : 1;
    int exp = (byte >> 3) & 0x0F;
    int mant = byte & 0x07;
    if (exp == 0) return sign * ldexpf((float)mant / 8.0f, -6); /* subnormal */
    if (exp == 15) return mant == 6 ? sign * NAN : sign * INFINITY; /* NaN/Inf */
    return sign * ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
}

static float fp8_e5m2_to_f32(uint8_t byte) {
    int sign = (byte & 0x80) ? -1 : 1;
    int exp = (byte >> 2) & 0x1F;
    int mant = byte & 0x03;
    if (exp == 0) return sign * ldexpf((float)mant / 4.0f, -14); /* subnormal */
    if (exp == 31) return mant ? sign * NAN : sign * INFINITY;
    return sign * ldexpf(1.0f + (float)mant / 4.0f, exp - 15);
}

/* Locate an optional FP8 scale for `name`. Returns 1 and sets *scale when a
 * sibling tensor is found; otherwise 0 and *scale = 1 (no scale). */
static int fp8_scale_for(const qzen_weight_store *store, const char *name,
                         float *scale, char *error, size_t error_size) {
    *scale = 1.0f;
    /* Convention A: "<base>.weight_scale" or "<base>_scale" */
    char base[512];
    size_t length = strlen(name);
    size_t keep = length;
    const char suffix[] = ".weight";
    if (length >= sizeof(suffix) - 1 &&
        strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0)
        keep -= sizeof(suffix) - 1;
    if (keep >= sizeof(base)) return 1;
    memcpy(base, name, keep);
    base[keep] = '\0';
    const char *candidates[4];
    char a[512], b[512];
    snprintf(a, sizeof(a), "%s.weight_scale", base);
    snprintf(b, sizeof(b), "%s_scale", name);
    candidates[0] = a; candidates[1] = b; candidates[2] = name; candidates[3] = NULL;
    for (int i = 0; i < 3; i++) {
        const qzen_st_header *header = NULL;
        const qzen_st_tensor *tensor = qzen_weight_find(store, candidates[i], &header);
        if (!tensor) continue;
        if (tensor->dtype != QZEN_DTYPE_F32 &&
            tensor->dtype != QZEN_DTYPE_BF16 &&
            tensor->dtype != QZEN_DTYPE_F16)
            continue;
        float *values = NULL;
        if (qzen_st_read_f32(header, tensor, NULL, NULL, &values,
                             error, error_size) && values) {
            *scale = values[0]; /* per-tensor scale: first element */
            free(values);
        }
        return 1;
    }
    return 1;
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int gguf_name(const char *name) {
    static const char suffix[] = ".gguf";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

/* Callback installed in the synthetic GGUF header: dequantise one tensor's
 * codes directly into the caller's host buffer. `index` was stashed in the
 * virtual tensor's file_offset by qzen_weight_find(). The byte count decides
 * BF16 (2 bytes/elem) vs F32 (4 bytes/elem), matching the virtual dtype. */
static int gguf_read_callback(void *opaque, uint64_t index, void *data,
                              size_t bytes, char *error, size_t error_size) {
    qzi_gguf *gguf = (qzi_gguf *)opaque;
    uint64_t elements = qzi_gguf_tensor_elements(gguf, (size_t)index);
    if (bytes == elements * 2)
        return qzi_gguf_read_bf16(gguf, (size_t)index, (uint16_t *)data,
                                  error, error_size);
    if (bytes == elements * 4)
        return qzi_gguf_read_f32(gguf, (size_t)index, (float *)data,
                                 error, error_size);
    if (error && error_size)
        snprintf(error, error_size, "GGUF tensor read size %zu mismatch", bytes);
    return 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

/* Read the headers of `count` shards into a store.  Borrows the path strings. */
static qzen_weight_store *open_paths(char **paths, size_t count,
                                   char *error, size_t error_size) {
    qzen_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!qzen_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            qzen_weight_store_free(store);
            return NULL;
        }
    }
    return store;
}

/* Open a single GGUF container as a synthetic-header store. The reader
 * presents torch-layout tensors under HF names, so every component loader
 * (which only knows qzen_weight_find + qzen_st_read_data) works unchanged. */
static qzen_weight_store *open_gguf(const char *path, char *owned_path,
                                    char *error, size_t error_size) {
    qzi_gguf *gguf = qzi_gguf_open(path, error, error_size);
    if (!gguf) return NULL;
    qzen_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        qzi_gguf_free(gguf);
        fail(error, error_size, "out of memory creating GGUF weight store");
        return NULL;
    }
    store->gguf = gguf;
    store->gguf_header.path = owned_path; /* borrowed, freed by caller */
    store->gguf_header.gguf = gguf;
    store->gguf_header.gguf_read = gguf_read_callback;
    return store;
}

qzen_weight_store *qzen_weight_store_open(const char *path,
                                      char *error, size_t error_size) {
    if (!path || !*path) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    /* A pack may be one file rather than a directory of shards: the SVDQuant
     * repacks are single 3.6-4.6 GB safetensors, and the caller that selects
     * between several of them has already resolved which one it wants.  Read
     * the stat, not the name, so a directory called "*.safetensors" cannot be
     * mistaken for one. */
    struct stat info;
    if (stat(path, &info) != 0) {
        fail(error, error_size, "cannot open weight path: %s", path);
        return NULL;
    }
    if (S_ISREG(info.st_mode)) {
        const char *name = strrchr(path, '/');
        name = name ? name + 1 : path;
        char *single = malloc(strlen(path) + 1);
        if (!single) {
            fail(error, error_size, "out of memory resolving the weight shard");
            return NULL;
        }
        strcpy(single, path);
        if (gguf_name(name)) {
            /* open_gguf takes ownership of `single` via the synthetic header. */
            return open_gguf(path, single, error, error_size);
        }
        if (!safetensors_name(name)) {
            free(single);
            fail(error, error_size, "not a safetensors or gguf file: %s", path);
            return NULL;
        }
        qzen_weight_store *store = open_paths(&single, 1, error, error_size);
        free(single);
        return store;
    }
    DIR *stream = opendir(path);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", path);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    char *gguf_path = NULL;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (gguf_name(entry->d_name)) {
            /* GGUF is a self-contained container: prefer it and ignore shards. */
            size_t length = strlen(path) + strlen(entry->d_name) + 2;
            free(gguf_path);
            gguf_path = malloc(length);
            if (!gguf_path) {
                fail(error, error_size, "out of memory resolving GGUF file");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            snprintf(gguf_path, length, "%s/%s", path, entry->d_name);
            continue;
        }
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(path) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", path, entry->d_name);
        count++;
    }
    closedir(stream);
    if (gguf_path) {
        free_paths(paths, count);
        qzen_weight_store *store = open_gguf(gguf_path, gguf_path, error, error_size);
        if (!store) free(gguf_path);
        return store;
    }
    if (!count) {
        fail(error, error_size, "no safetensors shards or gguf file in %s", path);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    qzen_weight_store *store = open_paths(paths, count, error, error_size);
    free_paths(paths, count);
    return store;
}

void qzen_weight_store_free(qzen_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        qzen_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    if (store->gguf) {
        qzi_gguf_free(store->gguf);
        free(store->gguf_header.path); /* the owned path string */
    }
    free(store);
}

size_t qzen_weight_store_shards(const qzen_weight_store *store) {
    return store ? (store->gguf ? 1 : store->count) : 0;
}

const qzen_st_tensor *qzen_weight_find(const qzen_weight_store *store,
                                   const char *name,
                                   const qzen_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    if (store->gguf) {
        int index = qzi_gguf_find(store->gguf, name);
        if (index < 0) return NULL;
        qzen_weight_store *mutable = (qzen_weight_store *)store;
        qzen_st_tensor *slot = &mutable->gguf_slots[mutable->gguf_next++ % 4];
        memset(slot, 0, sizeof(*slot));
        slot->name = (char *)qzi_gguf_tensor_name(store->gguf, (size_t)index);
        qzi_gguf_type type = qzi_gguf_tensor_type(store->gguf, (size_t)index);
        slot->dtype = type == QZI_GGUF_F32 ? QZEN_DTYPE_F32
                    : type == QZI_GGUF_F16 ? QZEN_DTYPE_F16
                    : QZEN_DTYPE_BF16;
        slot->ndim = qzi_gguf_tensor_rank(store->gguf, (size_t)index);
        const uint64_t *shape = qzi_gguf_tensor_shape(store->gguf, (size_t)index);
        for (int dimension = 0; dimension < slot->ndim && dimension < 8; dimension++)
            slot->shape[dimension] = shape[dimension];
        uint64_t elements = qzi_gguf_tensor_elements(store->gguf, (size_t)index);
        slot->data_begin = 0;
        slot->data_end = elements * (slot->dtype == QZEN_DTYPE_F32 ? 4 : 2);
        slot->file_offset = (uint64_t)index; /* index; read callback dequantises */
        if (header) *header = &mutable->gguf_header;
        return slot;
    }
    for (size_t index = 0; index < store->count; index++) {
        const qzen_st_tensor *tensor = qzen_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static qzen_gpu_tensor *load_tensor(const qzen_weight_store *store, qzen_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, qzen_dtype dtype,
                                  char *error, size_t error_size) {
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    /* FP8 checkpoints are decoded into the requested dtype (BF16 or F32).
     * Everything else must match exactly, as before. */
    int fp8 = (tensor->dtype == QZEN_DTYPE_F8_E4M3 ||
               tensor->dtype == QZEN_DTYPE_F8_E5M2) &&
              (dtype == QZEN_DTYPE_BF16 || dtype == QZEN_DTYPE_F32);
    if (!fp8 && (tensor->dtype != dtype || tensor->ndim != ndim)) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, qzen_dtype_name(tensor->dtype), tensor->ndim,
             qzen_dtype_name(dtype), ndim);
        return NULL;
    }
    if (fp8 && tensor->ndim != ndim) {
        fail(error, error_size, "weight %s rank %d, expected %d",
             name, tensor->ndim, ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    qzen_gpu_tensor *result = NULL;
    if (fp8) {
        /* Read the 1-byte-per-element payload, decode to F32, apply the
         * optional per-tensor scale, then round into the requested format. */
        uint8_t *raw = malloc((size_t)elements);
        float *floats = malloc((size_t)elements * sizeof(float));
        uint16_t *halves = dtype == QZEN_DTYPE_BF16
                               ? malloc((size_t)elements * sizeof(uint16_t))
                               : NULL;
        if (!raw || !floats || (dtype == QZEN_DTYPE_BF16 && !halves)) {
            fail(error, error_size, "out of memory decoding FP8 weight %s", name);
            free(raw); free(floats); free(halves);
            return NULL;
        }
        int ok = qzen_st_read_data(header, tensor, raw, (size_t)elements,
                                   error, error_size);
        float scale = 1.0f;
        if (ok) fp8_scale_for(store, name, &scale, error, error_size);
        if (ok) {
            int e5m2 = tensor->dtype == QZEN_DTYPE_F8_E5M2;
            for (size_t i = 0; i < (size_t)elements; i++) {
                floats[i] = (e5m2 ? fp8_e5m2_to_f32(raw[i])
                                  : fp8_e4m3_to_f32(raw[i])) * scale;
            }
            if (dtype == QZEN_DTYPE_BF16) {
                for (size_t i = 0; i < (size_t)elements; i++)
                    halves[i] = f32_to_bf16_rn(floats[i]);
                result = qzen_gpu_tensor_from_bf16(gpu, halves, (size_t)elements);
            } else {
                result = qzen_gpu_tensor_from_f32(gpu, floats, (size_t)elements);
            }
        }
        free(raw); free(floats); free(halves);
        if (!result && !(error && *error))
            fail(error, error_size, "cannot upload decoded FP8 weight %s: %s",
                 name, qzen_gpu_error(gpu));
        return result;
    }
    result = dtype == QZEN_DTYPE_BF16 ?
        qzen_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        qzen_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, qzen_gpu_error(gpu));
    }
    return result;
}

qzen_gpu_tensor *qzen_weight_load_bf16(const qzen_weight_store *store, qzen_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, QZEN_DTYPE_BF16,
                       error, error_size);
}

qzen_gpu_tensor *qzen_weight_load_f32(const qzen_weight_store *store, qzen_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, QZEN_DTYPE_F32,
                       error, error_size);
}

/* Common host reader. `want_bf16` selects the output format; F32 inputs are
 * always accepted, FP8 is decoded/scaled and BF16-rounded when requested. */
static int read_auto(const qzen_weight_store *store, const char *name,
                     int want_bf16, void **out, uint64_t *count,
                     char *error, size_t error_size) {
    *out = NULL;
    *count = 0;
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    uint64_t elements = qzen_st_tensor_elements(tensor);
    if (elements > SIZE_MAX / (want_bf16 ? 2 : 4)) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return 0;
    }
    int is_fp8 = tensor->dtype == QZEN_DTYPE_F8_E4M3 ||
                 tensor->dtype == QZEN_DTYPE_F8_E5M2;
    if (!is_fp8 && tensor->dtype != QZEN_DTYPE_BF16 &&
        tensor->dtype != QZEN_DTYPE_F16 && tensor->dtype != QZEN_DTYPE_F32) {
        fail(error, error_size, "weight %s has unsupported dtype %s", name,
             qzen_dtype_name(tensor->dtype));
        return 0;
    }
    float *floats = malloc((size_t)elements * sizeof(float));
    if (!floats) {
        fail(error, error_size, "out of memory reading %s", name);
        return 0;
    }
    int ok = 1;
    if (is_fp8) {
        uint8_t *raw = malloc((size_t)elements);
        if (!raw) { ok = 0; }
        else {
            ok = qzen_st_read_data(header, tensor, raw, (size_t)elements,
                                   error, error_size);
            float scale = 1.0f;
            if (ok) fp8_scale_for(store, name, &scale, error, error_size);
            if (ok) {
                int e5m2 = tensor->dtype == QZEN_DTYPE_F8_E5M2;
                for (size_t i = 0; i < (size_t)elements; i++)
                    floats[i] = (e5m2 ? fp8_e5m2_to_f32(raw[i])
                                      : fp8_e4m3_to_f32(raw[i])) * scale;
            }
            free(raw);
        }
    } else {
        /* qzen_st_read_f32 normalises F32/F16 (BF16 included) to host F32. */
        float *values = NULL;
        ok = qzen_st_read_f32(header, tensor, NULL, NULL, &values,
                              error, error_size) && values;
        if (ok) { memcpy(floats, values, (size_t)elements * sizeof(float)); free(values); }
    }
    if (!ok) { free(floats); return 0; }

    if (!want_bf16) {
        *out = floats;
    } else {
        uint16_t *halves = malloc((size_t)elements * sizeof(uint16_t));
        if (!halves) { free(floats); fail(error, error_size, "out of memory reading %s", name); return 0; }
        for (size_t i = 0; i < (size_t)elements; i++)
            halves[i] = f32_to_bf16_rn(floats[i]);
        free(floats);
        *out = halves;
    }
    *count = elements;
    return 1;
}

int qzen_weight_read_bf16_auto(const qzen_weight_store *store, const char *name,
                               uint16_t **out, uint64_t *count,
                               char *error, size_t error_size) {
    return read_auto(store, name, 1, (void **)out, count, error, error_size);
}

int qzen_weight_read_f32_auto(const qzen_weight_store *store, const char *name,
                              float **out, uint64_t *count,
                              char *error, size_t error_size) {
    return read_auto(store, name, 0, (void **)out, count, error, error_size);
}

int qzen_weight_is_int8(const qzen_weight_store *store, const char *name) {
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, NULL);
    return tensor && tensor->dtype == QZEN_DTYPE_I8;
}

qzen_svdq_kind qzen_weight_svdq_identify(const qzen_weight_store *store,
                                     uint32_t *rank) {
    static const char suffix[] = ".proj_up";
    const size_t suffix_length = sizeof(suffix) - 1;
    if (rank) *rank = 0;
    if (!store) return QZEN_SVDQ_NONE;
    for (size_t index = 0; index < store->count; index++) {
        const qzen_st_header *header = &store->headers[index];
        for (size_t tensor_index = 0; tensor_index < header->tensor_count;
             tensor_index++) {
            const qzen_st_tensor *up = &header->tensors[tensor_index];
            if (!up->name || up->ndim != 2) continue;
            size_t length = strlen(up->name);
            if (length <= suffix_length) continue;
            if (strcmp(up->name + length - suffix_length, suffix) != 0) continue;
            char scales_name[256];
            size_t base_length = length - suffix_length;
            if (base_length + sizeof(".wscales") > sizeof(scales_name))
                continue;
            memcpy(scales_name, up->name, base_length);
            snprintf(scales_name + base_length,
                     sizeof(scales_name) - base_length, ".wscales");
            if (rank) *rank = (uint32_t)up->shape[1];
            const qzen_st_tensor *scales = qzen_weight_find(store, scales_name,
                                                        NULL);
            /* No scale tensor at all is not an fp4 pack, it is a pack this
             * engine has no business claiming: report int4 and let the
             * dequantiser fail on the missing tensor. */
            if (!scales) return QZEN_SVDQ_INT4;
            return scales->dtype == QZEN_DTYPE_BF16 ? QZEN_SVDQ_INT4
                                                  : QZEN_SVDQ_FP4;
        }
    }
    return QZEN_SVDQ_NONE;
}

/* "<base>.weight" -> "<base>.weight_scale" is "." + "weight_scale"; the sidecar
 * replaces the last component, so strip ".weight" and append ".weight_scale". */
static int scale_sidecar_name(const char *name, char *out, size_t size) {
    static const char suffix[] = ".weight";
    size_t length = strlen(name);
    size_t keep = length;
    if (length >= sizeof(suffix) - 1 &&
        strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0) {
        keep = length - (sizeof(suffix) - 1);
    }
    int written = snprintf(out, size, "%.*s.weight_scale", (int)keep, name);
    return written > 0 && (size_t)written < size;
}

qzen_gpu_tensor *qzen_weight_load_i8_convrot(const qzen_weight_store *store, qzen_gpu *gpu,
                                         const char *name, uint64_t rows,
                                         uint64_t cols, qzen_gpu_tensor **scales,
                                         char *error, size_t error_size) {
    if (scales) *scales = NULL;
    if (!store || !gpu || !name || !scales) {
        fail(error, error_size, "INT8 weight request needs a store, GPU, name and scale slot");
        return NULL;
    }
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required INT8 weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != QZEN_DTYPE_I8 || tensor->ndim != 2) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected I8/2",
             name, qzen_dtype_name(tensor->dtype), tensor->ndim);
        return NULL;
    }
    if (tensor->shape[0] != rows || tensor->shape[1] != cols) {
        fail(error, error_size,
             "weight %s shape [%llu, %llu] does not match [%llu, %llu]", name,
             (unsigned long long)tensor->shape[0],
             (unsigned long long)tensor->shape[1],
             (unsigned long long)rows, (unsigned long long)cols);
        return NULL;
    }
    if (rows && cols > UINT64_MAX / rows) {
        fail(error, error_size, "weight %s shape overflows", name);
        return NULL;
    }
    uint64_t elements = rows * cols;
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    char sidecar[192];
    if (!scale_sidecar_name(name, sidecar, sizeof(sidecar))) {
        fail(error, error_size, "weight name is too long for a scale sidecar: %s",
             name);
        return NULL;
    }
    const qzen_st_header *scale_header = NULL;
    const qzen_st_tensor *scale_tensor = qzen_weight_find(store, sidecar, &scale_header);
    if (!scale_tensor) {
        fail(error, error_size, "INT8 weight %s is missing its scale sidecar %s",
             name, sidecar);
        return NULL;
    }
    if (scale_tensor->dtype != QZEN_DTYPE_F32) {
        fail(error, error_size, "scale %s has dtype %s, expected F32", sidecar,
             qzen_dtype_name(scale_tensor->dtype));
        return NULL;
    }
    uint64_t scale_count = 1;
    for (int dimension = 0; dimension < scale_tensor->ndim; dimension++) {
        scale_count *= scale_tensor->shape[dimension];
    }
    if (scale_count != rows) {
        fail(error, error_size,
             "scale %s has %llu values, expected %llu (one per output row)",
             sidecar, (unsigned long long)scale_count,
             (unsigned long long)rows);
        return NULL;
    }
    qzen_gpu_tensor *weight = qzen_gpu_tensor_new_i8(gpu, (size_t)elements);
    if (!weight) {
        fail(error, error_size, "cannot allocate %s: %s", name, qzen_gpu_error(gpu));
        return NULL;
    }
    if (!qzen_gpu_tensor_read_file_i8(weight, header->path, tensor->file_offset,
                                    (size_t)elements, error, error_size)) {
        qzen_gpu_tensor_free(weight);
        return NULL;
    }
    qzen_gpu_tensor *scale = qzen_gpu_tensor_load_f32(
        gpu, scale_header->path, scale_tensor->file_offset, (size_t)rows);
    if (!scale) {
        fail(error, error_size, "cannot load scale %s: %s", sidecar,
             qzen_gpu_error(gpu));
        qzen_gpu_tensor_free(weight);
        return NULL;
    }
    *scales = scale;
    return weight;
}

/* --- ConvRot qkv row order -------------------------------------------------
 * The checkpoint stores the fused qkv rows as three contiguous thirds
 * [q(56*128)][k][v]; QwenZImageNative consumes per-head triples [h0q,h0k,h0v,h1q,...]. The
 * permutation only reorders 128-row blocks, and the projection is rotated along
 * K, so the per-row scales move with their rows and nothing else changes. */

static int qkv_head_count(uint64_t rows, uint32_t head_dim, uint32_t *heads) {
    if (!head_dim || !rows) return 0;
    uint64_t block = (uint64_t)3u * (uint64_t)head_dim;
    if ((rows % block) != 0) return 0;
    uint64_t count = rows / block;
    if (!count || count > UINT32_MAX / 3u) return 0;
    *heads = (uint32_t)count;
    return 1;
}

/* Destination block i (= head * 3 + kind) comes from source block kind*heads + head. */
static uint32_t qkv_source_block(uint32_t destination, uint32_t heads) {
    return (destination % 3u) * heads + destination / 3u;
}

static int read_bytes_at(const char *path, uint64_t offset, void *data,
                         size_t bytes) {
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) return 0;
    unsigned char *cursor = data;
    size_t done = 0;
    int ok = 1;
    while (done < bytes) {
        ssize_t got = pread(descriptor, cursor + done, bytes - done,
                            (off_t)(offset + done));
        if (got <= 0) {
            ok = 0;
            break;
        }
        done += (size_t)got;
    }
    close(descriptor);
    return ok;
}

static const qzen_st_tensor *qkv_scale_sidecar(const qzen_weight_store *store,
                                             const char *name, uint64_t rows,
                                             const qzen_st_header **header,
                                             char *error, size_t error_size) {
    char sidecar[192];
    if (!scale_sidecar_name(name, sidecar, sizeof(sidecar))) {
        fail(error, error_size, "weight name is too long for a scale sidecar: %s",
             name);
        return NULL;
    }
    const qzen_st_tensor *tensor = qzen_weight_find(store, sidecar, header);
    if (!tensor) {
        fail(error, error_size, "INT8 weight %s is missing its scale sidecar %s",
             name, sidecar);
        return NULL;
    }
    if (tensor->dtype != QZEN_DTYPE_F32) {
        fail(error, error_size, "scale %s has dtype %s, expected F32", sidecar,
             qzen_dtype_name(tensor->dtype));
        return NULL;
    }
    uint64_t count = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++)
        count *= tensor->shape[dimension];
    if (count != rows) {
        fail(error, error_size,
             "scale %s has %llu values, expected %llu (one per output row)",
             sidecar, (unsigned long long)count, (unsigned long long)rows);
        return NULL;
    }
    return tensor;
}

static qzen_gpu_tensor *load_qkv_perhead(const qzen_weight_store *store, qzen_gpu *gpu,
                                       const char *name, uint64_t rows,
                                       uint64_t cols, uint32_t head_dim,
                                       qzen_dtype dtype, qzen_gpu_tensor **scales,
                                       char *error, size_t error_size) {
    if (scales) *scales = NULL;
    uint32_t heads = 0;
    if (!qkv_head_count(rows, head_dim, &heads)) {
        fail(error, error_size,
             "qkv weight %s has %llu rows, not 3 * heads * %u", name,
             (unsigned long long)rows, head_dim);
        return NULL;
    }
    const qzen_st_header *header = NULL;
    const qzen_st_tensor *tensor = qzen_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != 2) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/2",
             name, qzen_dtype_name(tensor->dtype), tensor->ndim,
             qzen_dtype_name(dtype));
        return NULL;
    }
    if (tensor->shape[0] != rows || tensor->shape[1] != cols) {
        fail(error, error_size,
             "weight %s shape [%llu, %llu] does not match [%llu, %llu]", name,
             (unsigned long long)tensor->shape[0],
             (unsigned long long)tensor->shape[1],
             (unsigned long long)rows, (unsigned long long)cols);
        return NULL;
    }
    size_t element_size = qzen_dtype_size(dtype);
    if (rows > SIZE_MAX / cols || rows * cols > SIZE_MAX / (element_size ? element_size : 1)) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    size_t elements = (size_t)(rows * cols);
    size_t block_bytes = (size_t)head_dim * (size_t)cols * element_size;
    /* One contiguous read, then a host-side block shuffle. Reading the blocks in
     * destination order instead turned one 115 MiB stream into 168 seeks plus
     * 168 open/close pairs, which cost ~25% of the load on this disk. */
    unsigned char *staging = malloc(elements * element_size);
    unsigned char *ordered = malloc(elements * element_size);
    if (!staging || !ordered) {
        fail(error, error_size, "out of memory staging %s", name);
        free(staging);
        free(ordered);
        return NULL;
    }
    if (!read_bytes_at(header->path, tensor->file_offset, staging,
                       elements * element_size)) {
        fail(error, error_size, "cannot read %s", name);
        free(staging);
        free(ordered);
        return NULL;
    }
    for (uint32_t destination = 0; destination < heads * 3u; destination++) {
        uint32_t source = qkv_source_block(destination, heads);
        memcpy(ordered + (size_t)destination * block_bytes,
               staging + (size_t)source * block_bytes, block_bytes);
    }
    free(staging);
    staging = ordered;
    qzen_gpu_tensor *weight = dtype == QZEN_DTYPE_I8
        ? qzen_gpu_tensor_new_i8(gpu, elements)
        : qzen_gpu_tensor_new_bf16(gpu, elements);
    int uploaded = weight && (dtype == QZEN_DTYPE_I8
        ? qzen_gpu_tensor_write_i8(weight, (const int8_t *)staging, elements)
        : qzen_gpu_tensor_write_bf16(weight, (const uint16_t *)staging, elements));
    free(staging);
    if (!uploaded) {
        fail(error, error_size, "cannot upload %s: %s", name, qzen_gpu_error(gpu));
        qzen_gpu_tensor_free(weight);
        return NULL;
    }
    if (dtype != QZEN_DTYPE_I8) return weight;

    const qzen_st_header *scale_header = NULL;
    const qzen_st_tensor *scale_tensor = qkv_scale_sidecar(store, name, rows,
                                                         &scale_header, error,
                                                         error_size);
    if (!scale_tensor) {
        qzen_gpu_tensor_free(weight);
        return NULL;
    }
    float *row_scales = malloc((size_t)rows * sizeof(float));
    float *scaled_rows = malloc((size_t)rows * sizeof(float));
    int scale_ok = row_scales && scaled_rows &&
        read_bytes_at(scale_header->path, scale_tensor->file_offset, row_scales,
                      (size_t)rows * sizeof(float));
    if (scale_ok) {
        for (uint32_t destination = 0; destination < heads * 3u; destination++) {
            uint32_t source = qkv_source_block(destination, heads);
            memcpy(scaled_rows + (size_t)destination * head_dim,
                   row_scales + (size_t)source * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
        qzen_gpu_tensor *scale = qzen_gpu_tensor_new_f32(gpu, (size_t)rows);
        if (!scale || !qzen_gpu_tensor_write_f32(scale, scaled_rows, (size_t)rows)) {
            fail(error, error_size, "cannot upload the scales of %s: %s", name,
                 qzen_gpu_error(gpu));
            qzen_gpu_tensor_free(scale);
            scale_ok = 0;
        } else {
            *scales = scale;
        }
    } else {
        fail(error, error_size, "cannot read the scales of %s", name);
    }
    free(row_scales);
    free(scaled_rows);
    if (!scale_ok) {
        qzen_gpu_tensor_free(weight);
        return NULL;
    }
    return weight;
}

qzen_gpu_tensor *qzen_weight_load_qkv_perhead(const qzen_weight_store *store, qzen_gpu *gpu,
                                          const char *name, uint64_t rows,
                                          uint64_t cols, uint32_t head_dim,
                                          qzen_gpu_tensor **scales, char *error,
                                          size_t error_size) {
    return load_qkv_perhead(store, gpu, name, rows, cols, head_dim,
                            QZEN_DTYPE_I8, scales, error, error_size);
}

qzen_gpu_tensor *qzen_weight_load_qkv_perhead_bf16(
                                          const qzen_weight_store *store, qzen_gpu *gpu,
                                          const char *name, uint64_t rows,
                                          uint64_t cols, uint32_t head_dim,
                                          char *error, size_t error_size) {
    return load_qkv_perhead(store, gpu, name, rows, cols, head_dim,
                            QZEN_DTYPE_BF16, NULL, error, error_size);
}
