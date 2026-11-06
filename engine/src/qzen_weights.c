#include "qzen_weights.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct qzen_weight_store {
    qzen_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
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
        if (!safetensors_name(name)) {
            fail(error, error_size, "not a safetensors file: %s", path);
            return NULL;
        }
        char *single = malloc(strlen(path) + 1);
        if (!single) {
            fail(error, error_size, "out of memory resolving the weight shard");
            return NULL;
        }
        strcpy(single, path);
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
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
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
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", path);
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
    free(store);
}

size_t qzen_weight_store_shards(const qzen_weight_store *store) {
    return store ? store->count : 0;
}

const qzen_st_tensor *qzen_weight_find(const qzen_weight_store *store,
                                   const char *name,
                                   const qzen_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
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
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, qzen_dtype_name(tensor->dtype), tensor->ndim,
             qzen_dtype_name(dtype), ndim);
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
    qzen_gpu_tensor *result = dtype == QZEN_DTYPE_BF16 ?
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
