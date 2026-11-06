#ifndef QZEN_ZIMAGE_DIT_H
#define QZEN_ZIMAGE_DIT_H

#include "qzen_quanto.h"
#include "qzen_gpu.h"
#include "qzen_nunchaku.h"

#include <stddef.h>
#include <stdint.h>

/* Z-Image Turbo DiT forward skeleton (CPU reference path).
 *
 * Architecture (Wan2GP models/z_image/z_image_transformer2d.py +
 * configs/z_image.json): 30 main blocks + 2 noise refiners (modulated) and
 * 2 context refiners (unmodulated); dim 3840, 30 heads x 128, SwiGLU FFN
 * 10240 without bias, adaLN Linear(256 -> 4*3840) in chunks
 * scale_msa / gate_msa / scale_mlp / gate_mlp with scale += 1 and
 * gate = tanh(gate); RMSNorm eps 1e-5 (also on q/k over head_dim);
 * full-sequence bidirectional self-attention over img + cap tokens with
 * 3-axis RoPE (32, 48, 48), theta 256; token sequences padded to a multiple
 * of 32 with learnable pad tokens.
 *
 * Phase 0 keeps every weight in host memory (quanto I8 dequantized to BF16 at
 * load) and computes in F32 with BF16 weights widened per element.  This
 * mirrors the runtime plan (I8 resident, dequantize, BF16 GEMM) and gives a
 * numerics reference for the CUDA port in later phases.
 */

typedef struct {
    uint32_t dim;             /* 3840 */
    uint32_t heads;           /* 30 */
    uint32_t head_dim;        /* 128 */
    uint32_t ffn_dim;         /* 10240 */
    uint32_t adaln_dim;       /* 256 (ADALN_EMBED_DIM) */
    float norm_eps;           /* 1e-5 */
    float final_norm_eps;     /* 1e-6, affine-free LayerNorm */
    uint32_t rope_axes[3];    /* 32, 48, 48 (sums to head_dim) */
    uint32_t rope_lens[3];    /* 1536, 512, 512 */
    float rope_theta;         /* 256 */
    float t_scale;            /* 1000 */
    uint32_t seq_multiple;    /* SEQ_MULTI_OF: pad token sequences to 32 */
} qzen_zimage_config;

qzen_zimage_config qzen_zimage_default_config(void);

/* A plain BF16 Linear (bias optional).  Weight is [out, in] row-major. */
typedef struct {
    uint32_t out;
    uint32_t in;
    uint16_t *weight;
    float *bias;
    /* Optional device-resident BF16 copies, uploaded by
     * qzen_zimage_linear_upload for the GPU forward path.  The host copy above
     * is kept so the CPU reference path (and fallback) keeps working. */
    qzen_gpu_tensor *gpu_weight;
    qzen_gpu_tensor *gpu_bias;
    /* Lazily widened F32 copy of the weight, built once for the host-side
     * single-row adaLN GEMMs of the GPU path (the CPU path keeps widening
     * per element).  NULL until first needed. */
    float *weight_f32;
    /* A Nunchaku pack stays 4-bit in both memories (PLAN §19).  `weight` above
     * is then NULL until a host path asks for it, and the device keeps these
     * five instead of an expanded matrix a quarter of which is zeroes in the
     * scale direction: the largest layer is 39 MB packed against 157 MB
     * expanded, and the whole DiT is 2.7 GB against 10.6 GB. */
    qzen_nunchaku_packed packed;
    int has_packed;
    qzen_gpu_tensor *gpu_codes;
    qzen_gpu_tensor *gpu_scales;
    qzen_gpu_tensor *gpu_smooth;
    qzen_gpu_tensor *gpu_up;
    qzen_gpu_tensor *gpu_down;
} qzen_zimage_linear;

/* A BF16 RMSNorm weight [width]. */
typedef struct {
    uint32_t width;
    float eps;
    uint16_t *weight;
    qzen_gpu_tensor *gpu_weight;
} qzen_zimage_rms;

int qzen_zimage_linear_load(const qzen_weight_store *store, const char *name,
                          uint32_t out, uint32_t in, int has_bias,
                          qzen_zimage_linear *linear,
                          char *error, size_t error_size);
/* Same, but the weight comes from a quanto qint8 pair (dequantized to BF16). */
int qzen_zimage_linear_load_quanto(const qzen_weight_store *store, const char *name,
                                 uint32_t out, uint32_t in, int has_bias,
                                 qzen_zimage_linear *linear,
                                 char *error, size_t error_size);
void qzen_zimage_linear_free(qzen_zimage_linear *linear);
int qzen_zimage_linear_forward(const qzen_zimage_linear *linear,
                             const float *input, uint32_t rows, float *output);

/* Gives back the CPU fallback's shared expansion scratch (at most one layer,
 * 157 MB).  qzen_zpipe_free calls it; nothing else needs to. */
void qzen_zimage_cpu_scratch_release(void);

/* The host BF16 matrix, materialising it from the packed form on the first
 * call.  NULL only when the Linear has neither.  Every host path goes through
 * this instead of touching `weight`, which is what lets a Nunchaku block be
 * loaded without expanding 10.6 GB of weights. */
const uint16_t *qzen_zimage_linear_host_weight(qzen_zimage_linear *linear);

int qzen_zimage_rms_load(const qzen_weight_store *store, const char *name,
                       uint32_t width, float eps, qzen_zimage_rms *rms,
                       char *error, size_t error_size);
void qzen_zimage_rms_free(qzen_zimage_rms *rms);
/* out = in * rsqrt(mean(in^2) + eps) * weight, row-major [rows, width]. */
int qzen_zimage_rms_forward(const qzen_zimage_rms *rms, const float *input,
                          uint32_t rows, float *output);

/* One transformer block; `section` is "layers", "noise_refiner" (both
 * adaLN-modulated) or "context_refiner" (unmodulated). */
typedef struct qzen_zimage_block qzen_zimage_block;

qzen_zimage_block *qzen_zimage_block_load(const qzen_weight_store *store,
                                      const char *section, uint32_t index,
                                      const qzen_zimage_config *config,
                                      char *error, size_t error_size);
void qzen_zimage_block_free(qzen_zimage_block *block);
int qzen_zimage_block_modulated(const qzen_zimage_block *block);

/* One of the block's loaded Linears by name -- "to_q", "to_k", "to_v",
 * "to_out", "w1", "w2", "w3" or "adaln" -- or NULL for an unknown name.
 *
 * This exists for the loader gate.  A Nunchaku checkpoint fuses q/k/v into one
 * to_qkv and w3/w1 into one net.0.proj, and reading the wrong third or the
 * wrong half yields a matrix that dequantises perfectly and is simply the
 * wrong matrix, so the split has to be asserted directly. */
const qzen_zimage_linear *qzen_zimage_block_linear(const qzen_zimage_block *block,
                                               const char *name);

/* Forward over x [seq, dim] in place (row-major F32).  `adaln` is the 256-wide
 * timestep embedding (required for modulated blocks, ignored otherwise).
 * rope_cos / rope_sin are [seq, head_dim / 2] pair tables from
 * qzen_zimage_rope_tables. */
int qzen_zimage_block_forward(const qzen_zimage_block *block, float *x,
                            uint32_t seq, const float *adaln,
                            const float *rope_cos, const float *rope_sin,
                            char *error, size_t error_size);

/* 3-axis RoPE pair tables from position ids [seq, 3] (axis 0 = caption/time
 * coordinate, axes 1/2 = image grid row/column).  cos/sin are [seq,
 * head_dim / 2]; pair i rotates the (x[2i], x[2i+1]) components of every
 * head. */
int qzen_zimage_rope_tables(const qzen_zimage_config *config, const int32_t *ids,
                          uint32_t seq, float *cos, float *sin);

/* TimestepEmbedder sinusoidal embedding: [cos(args) | sin(args)], dim 256,
 * max_period 10000. */
void qzen_zimage_timestep_embedding(float t, uint32_t dim, float *out);

/* TimestepEmbedder MLP: Linear(256 -> 1024) -> SiLU -> Linear(1024 -> 256). */
typedef struct {
    qzen_zimage_linear fc1;
    qzen_zimage_linear fc2;
} qzen_zimage_t_embedder;

int qzen_zimage_t_embedder_load(const qzen_weight_store *store,
                              const qzen_zimage_config *config,
                              qzen_zimage_t_embedder *embedder,
                              char *error, size_t error_size);
void qzen_zimage_t_embedder_free(qzen_zimage_t_embedder *embedder);
/* input: sinusoidal embedding [rows, 256]; output [rows, 256]. */
int qzen_zimage_t_embedder_forward(const qzen_zimage_t_embedder *embedder,
                                 const float *input, uint32_t rows,
                                 float *output);

/* FinalLayer: affine-free LayerNorm(eps 1e-6) * (SiLU -> Linear(256 -> dim)
 * + 1), then Linear(dim -> out_channels * patch volume). */
typedef struct {
    uint32_t dim;
    uint32_t out;
    float norm_eps;
    qzen_zimage_linear adaLN;
    qzen_zimage_linear out_proj;
} qzen_zimage_final_layer;

int qzen_zimage_final_layer_load(const qzen_weight_store *store,
                               const qzen_zimage_config *config,
                               uint32_t out, qzen_zimage_final_layer *layer,
                               char *error, size_t error_size);
void qzen_zimage_final_layer_free(qzen_zimage_final_layer *layer);
int qzen_zimage_final_layer_forward(const qzen_zimage_final_layer *layer,
                                  const float *x, uint32_t rows,
                                  const float *adaln, float *output);

/* --- GPU forward path -------------------------------------------------------- *
 *
 * The GPU path keeps every weight BF16-resident (~12.3 GiB) and runs the same
 * math as the CPU reference through the qzen_gpu.cu primitives (cuBLASLt BF16
 * GEMM for the Linears, the existing RMSNorm / adaLN / gate / QK-norm / RoPE /
 * SDPA / SiLU kernels for the rest).  Two layout adaptations are baked in at
 * upload time and are invisible to callers:
 *
 *  1. The adaLN modulation is packed into a [QZEN_ZIMAGE_MOD_SLOTS, dim] tensor:
 *     slot 0 = scale_msa (the adaLN kernel adds 1 itself), slot 1 =
 *     tanh(gate_msa), slot 2 = scale_mlp, slot 3 = tanh(gate_mlp), slot 4 =
 *     zeros (the unused shift the kernel still adds).
 *  2. qzen_gpu's RoPE kernels pair (d, d + head_dim/2) while Z-Image pairs
 *     (2i, 2i+1).  The Q/K projection rows and the Q/K norm weights are
 *     therefore permuted per head at upload time (even dimensions first), so
 *     the stock kernels reproduce the interleaved pairing.  SDPA dot products
 *     are permutation invariant, so no unpermute is needed afterwards.
 */

/* Slots of the packed modulation tensor described above. */
#define QZEN_ZIMAGE_MOD_SLOTS 5
#define QZEN_ZIMAGE_MOD_SHIFT_SLOT 4

typedef struct qzen_zimage_gpu qzen_zimage_gpu;

/* Create the GPU context (device weights are uploaded separately, per module).
 * Returns NULL when no CUDA device is available; callers fall back to CPU. */
qzen_zimage_gpu *qzen_zimage_gpu_create(const qzen_zimage_config *config,
                                    char *error, size_t error_size);
void qzen_zimage_gpu_free(qzen_zimage_gpu *zgu);
qzen_gpu *qzen_zimage_gpu_handle(qzen_zimage_gpu *zgu);

/* Upload the device copies (idempotent; frees any previous copy).  For the
 * block's to_q / to_k / norm_q / norm_k the RoPE permutation above is applied
 * while staging, so the host weights stay untouched. */
int qzen_zimage_linear_upload(qzen_zimage_linear *linear, qzen_zimage_gpu *zgu,
                            char *error, size_t error_size);
int qzen_zimage_rms_upload(qzen_zimage_rms *rms, qzen_zimage_gpu *zgu,
                         char *error, size_t error_size);
int qzen_zimage_block_upload(qzen_zimage_block *block, qzen_zimage_gpu *zgu,
                           char *error, size_t error_size);
int qzen_zimage_t_embedder_upload(qzen_zimage_t_embedder *embedder,
                                qzen_zimage_gpu *zgu, char *error,
                                size_t error_size);
/* Only out_proj (the adaLN modulation stays on the host). */
int qzen_zimage_final_layer_upload(qzen_zimage_final_layer *layer,
                                 qzen_zimage_gpu *zgu, char *error,
                                 size_t error_size);

/* Scratch-buffer management: grow-only, so pointers fetched after a reserve
 * stay valid for the rest of the forward. */
int qzen_zimage_gpu_reserve(qzen_zimage_gpu *zgu, uint32_t seq,
                          char *error, size_t error_size);
qzen_gpu_tensor *qzen_zimage_gpu_img(qzen_zimage_gpu *zgu);   /* [seq, dim] */
qzen_gpu_tensor *qzen_zimage_gpu_cap(qzen_zimage_gpu *zgu);   /* [seq, dim] */
qzen_gpu_tensor *qzen_zimage_gpu_full(qzen_zimage_gpu *zgu);  /* [seq, dim] */
/* RoPE tables for a trailing sub-sequence (the caption section runs on its
 * own tensor, so its table must start at the caption's first row). */
qzen_gpu_tensor *qzen_zimage_gpu_rope_cos_cap(qzen_zimage_gpu *zgu);
qzen_gpu_tensor *qzen_zimage_gpu_rope_sin_cap(qzen_zimage_gpu *zgu);
/* Final-layer output staging: [seq, 64] BF16 patch features. */
qzen_gpu_tensor *qzen_zimage_gpu_out64(qzen_zimage_gpu *zgu);

/* Upload the [seq, head_dim/2] RoPE pair tables (F32). */
int qzen_zimage_gpu_set_rope(qzen_zimage_gpu *zgu, const float *cos,
                           const float *sin, uint32_t seq, uint32_t half,
                           char *error, size_t error_size);
qzen_gpu_tensor *qzen_zimage_gpu_rope_cos(qzen_zimage_gpu *zgu);
qzen_gpu_tensor *qzen_zimage_gpu_rope_sin(qzen_zimage_gpu *zgu);
/* Pack the raw 4*dim adaLN output into the persistent modulation tensor. */
int qzen_zimage_gpu_set_modulation(qzen_zimage_gpu *zgu, const float *mod_raw,
                                 uint32_t dim, char *error, size_t error_size);
qzen_gpu_tensor *qzen_zimage_gpu_modulation(qzen_zimage_gpu *zgu);
/* Run a modulated block's adaLN linear on the host and pack the result into
 * the context's modulation tensor. */
int qzen_zimage_block_modulation_gpu(const qzen_zimage_block *block,
                                   qzen_zimage_gpu *zgu, const float *adaln,
                                   char *error, size_t error_size);

/* One block over x [seq, dim] BF16 in place.  rope_cos / rope_sin are the
 * [seq, head_dim/2] F32 tables (qzen_zimage_gpu_set_rope fills the context's
 * default pair; a caller running a sub-sequence, e.g. the caption section,
 * passes its own pair).  modulation is the packed tensor from
 * qzen_zimage_gpu_set_modulation, or NULL for unmodulated blocks. */
int qzen_zimage_block_forward_gpu(const qzen_zimage_block *block,
                                qzen_zimage_gpu *zgu, qzen_gpu_tensor *x,
                                uint32_t seq, qzen_gpu_tensor *modulation,
                                qzen_gpu_tensor *rope_cos,
                                qzen_gpu_tensor *rope_sin, char *error,
                                size_t error_size);

/* Host-in / host-out convenience wrapper (uploads x and the rope tables, runs
 * the block, downloads x).  adaln is the 256-wide timestep embedding (required
 * for modulated blocks, ignored otherwise); its adaLN linear stays a host
 * GEMM, so the packed modulation matches the CPU reference exactly. */
int qzen_zimage_block_forward_gpu_host(const qzen_zimage_block *block,
                                     qzen_zimage_gpu *zgu, float *x,
                                     uint32_t seq, const float *adaln,
                                     const float *rope_cos,
                                     const float *rope_sin, char *error,
                                     size_t error_size);

/* Timestep embedder MLP on the GPU (host-in / host-out). */
int qzen_zimage_t_embedder_forward_gpu(const qzen_zimage_t_embedder *embedder,
                                     qzen_zimage_gpu *zgu, const float *input,
                                     uint32_t rows, float *output,
                                     char *error, size_t error_size);

/* Final layer on the GPU: affine-free LayerNorm * (1 + adaLN scale) then the
 * output projection.  adaln is the 256-wide timestep embedding; output is the
 * [rows, out] BF16 tensor (downloaded by the caller). */
int qzen_zimage_final_layer_forward_gpu(const qzen_zimage_final_layer *layer,
                                      qzen_zimage_gpu *zgu,
                                      qzen_gpu_tensor *x, uint32_t rows,
                                      const float *adaln,
                                      qzen_gpu_tensor *output, char *error,
                                      size_t error_size);

#endif
