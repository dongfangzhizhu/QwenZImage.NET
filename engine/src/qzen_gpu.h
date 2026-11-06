#ifndef QZEN_GPU_H
#define QZEN_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct qzen_gpu qzen_gpu;
typedef struct qzen_gpu_tensor qzen_gpu_tensor;

typedef enum {
    QZEN_GPU_F32 = 0,
    QZEN_GPU_BF16,
    QZEN_GPU_I8,
    QZEN_GPU_U32,
    /* One byte like I8, but the Blackwell tensor cores read it as E4M3. */
    QZEN_GPU_F8E4M3
} qzen_gpu_dtype;

typedef struct {
    uint64_t allocated_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    uint64_t tensor_allocations;
    uint64_t direct_dispatches;
    uint64_t mps_linear_dispatches;
    uint64_t mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches;
    uint64_t blit_copies;
    uint64_t submissions;
    double command_encode_seconds;
    double command_wait_seconds;
    /* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
     * so command_wait_seconds is the complete turnaround measurement. */
    double gpu_seconds;
    /* CUDA: exclusive GPU time from event pairs around linear / SDPA / conv. */
    double gpu_linear_seconds;
    double gpu_sdpa_seconds;
    double gpu_conv_seconds;
    /* CUDA weight staging. Loader lanes run concurrently, so these seconds are
     * summed across readers and can exceed the phase's wall time. */
    uint64_t stage_bytes;
    double stage_read_seconds;
    double stage_copy_seconds;
    double stage_pin_seconds;
} qzen_gpu_stats;

qzen_gpu *qzen_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size);
void qzen_gpu_free(qzen_gpu *gpu);
int qzen_gpu_is_m5(const qzen_gpu *gpu);
int qzen_gpu_has_nax_mlp(const qzen_gpu *gpu);
int qzen_gpu_has_int8_mlp(const qzen_gpu *gpu);

qzen_gpu_tensor *qzen_gpu_tensor_new_f32(qzen_gpu *gpu, size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_new_bf16(qzen_gpu *gpu, size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_new_i8(qzen_gpu *gpu, size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_new_f8(qzen_gpu *gpu, size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_from_f32(qzen_gpu *gpu, const float *values,
                                      size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_from_bf16(qzen_gpu *gpu, const uint16_t *values,
                                       size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_from_u32(qzen_gpu *gpu, const uint32_t *values,
                                      size_t elements);
/* Allocate shared Metal storage and pread BF16 payload directly into it. */
qzen_gpu_tensor *qzen_gpu_tensor_load_bf16(qzen_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements);
qzen_gpu_tensor *qzen_gpu_tensor_load_f32(qzen_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Fill an existing shared BF16 buffer from a file. The tensor and its
 * accounting are unchanged, so this may run on an I/O thread while another
 * tensor is in flight on the GPU. */
int qzen_gpu_tensor_read_file_bf16(qzen_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* Same staged parallel read for the quantized weight cache, whose payload is
 * already in the layout the GEMM wants. */
int qzen_gpu_tensor_read_file_i8(qzen_gpu_tensor *tensor, const char *path,
                               uint64_t file_offset, size_t elements,
                               char *error, size_t error_size);
int qzen_gpu_tensor_read_file_f32(qzen_gpu_tensor *tensor, const char *path,
                                uint64_t file_offset, size_t elements,
                                char *error, size_t error_size);
int qzen_gpu_tensor_read_i8(const qzen_gpu_tensor *tensor, int8_t *values,
                          size_t elements);
/* As above, but ask Darwin to avoid retaining a second copy in the file cache.
 * Intended for large sequential weight streams whose destination is the only
 * useful resident copy. */
int qzen_gpu_tensor_stream_file_bf16(qzen_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size);
void qzen_gpu_tensor_free(qzen_gpu_tensor *tensor);
size_t qzen_gpu_tensor_elements(const qzen_gpu_tensor *tensor);
qzen_gpu_dtype qzen_gpu_tensor_dtype(const qzen_gpu_tensor *tensor);
int qzen_gpu_tensor_read_f32(const qzen_gpu_tensor *tensor, float *values,
                           size_t elements);
int qzen_gpu_tensor_read_f32_range(const qzen_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements);
int qzen_gpu_tensor_read_bf16(const qzen_gpu_tensor *tensor, uint16_t *values,
                            size_t elements);
int qzen_gpu_tensor_write_f32(qzen_gpu_tensor *tensor, const float *values,
                            size_t elements);
int qzen_gpu_tensor_write_f32_range(qzen_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements);
int qzen_gpu_tensor_write_bf16(qzen_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements);
/* Stream-ordered variant: ordered w.r.t. the GPU's compute stream. */
int qzen_gpu_tensor_write_bf16_stream(qzen_gpu *gpu, qzen_gpu_tensor *tensor,
                                    const uint16_t *values,
                                    size_t elements);
int qzen_gpu_tensor_write_i8(qzen_gpu_tensor *tensor, const int8_t *values,
                           size_t elements);
int qzen_gpu_tensor_write_bf16_range(qzen_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements);

int qzen_gpu_begin(qzen_gpu *gpu);
/* Commit the current command buffer without waiting, then continue encoding on
 * the same ordered queue. qzen_gpu_submit() waits and validates the whole chain. */
int qzen_gpu_continue(qzen_gpu *gpu);
int qzen_gpu_submit(qzen_gpu *gpu);
const char *qzen_gpu_error(const qzen_gpu *gpu);
int qzen_gpu_get_stats(const qzen_gpu *gpu, qzen_gpu_stats *stats);
/* Device-wide free and total memory, i.e. the number nvidia-smi reports, not
 * this context's tensor accounting: blocks another process (or the allocator's
 * pool) still holds count as used. Needs no context handle so a preflight can
 * run before the first weight is mapped. Returns 0 when the platform cannot
 * answer, which callers must treat as "unknown", not "plenty". */
int qzen_gpu_device_memory(uint64_t *free_bytes, uint64_t *total_bytes);
/* Bytes held by live allocations, from the stream-ordered pool's own counter.
 *
 * The difference matters after a tier switch: this engine sets the pool's
 * release threshold to 24 GiB, so freed blocks are kept for the next loader and
 * never go back to the driver.  total-free therefore reports the process
 * high-water forever -- load the 13 GiB bf16 pack, switch to the 3.5 GiB packed
 * one, and it still says 13 GiB.  Returns 0 when the pool is not in use (then
 * total-free is the live number). */
int qzen_gpu_device_memory_live(uint64_t *used_bytes);
/* Stream-ordered pool bookkeeping, for diagnosing allocator behaviour under a
 * near-full card: used/reserved are the pool's own view, high-water marks are
 * per-pool maxima since process start, and mallocs/malloc_fails count
 * cudaMallocAsync calls made by this context (failures are also fatal at the
 * call site, so a non-zero count means the pool could not grow). */
typedef struct {
    uint64_t used_current;
    uint64_t used_high;
    uint64_t reserved_current;
    uint64_t reserved_high;
    uint64_t device_free;
    uint64_t device_total;
    uint64_t mallocs;
    uint64_t malloc_fails;
} qzen_gpu_mem_stats;
int qzen_gpu_mem_query(qzen_gpu *gpu, qzen_gpu_mem_stats *stats);
/* Optional benchmark labels. With QZEN_PROFILE set, marks and context teardown
 * print wall time alongside command-buffer GPU time and allocation counters. */
void qzen_gpu_profile_set_label(qzen_gpu *gpu, const char *label);
void qzen_gpu_profile_mark(qzen_gpu *gpu, const char *phase);

int qzen_gpu_linear_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input, const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim);
int qzen_gpu_patch_linear_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                             const qzen_gpu_tensor *input,
                             const qzen_gpu_tensor *weight,
                             const qzen_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int qzen_gpu_patch_linear_bf16_offset(
                             qzen_gpu *gpu, qzen_gpu_tensor *output,
                             size_t output_offset,
                             const qzen_gpu_tensor *input, size_t input_offset,
                             const qzen_gpu_tensor *weight,
                             const qzen_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int qzen_gpu_patch_linear_bf16_map(
                             qzen_gpu *gpu, qzen_gpu_tensor *output,
                             const qzen_gpu_tensor *input,
                             const qzen_gpu_tensor *weight,
                             const qzen_gpu_tensor *bias,
                             const qzen_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int qzen_gpu_silu_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *input, uint32_t elements);
int qzen_gpu_cast_f32_to_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                            const qzen_gpu_tensor *input, uint32_t elements);
int qzen_gpu_cast_bf16_to_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                            const qzen_gpu_tensor *input, uint32_t elements);
int qzen_gpu_copy_bf16(qzen_gpu *gpu, qzen_gpu_tensor *destination,
                     size_t destination_offset,
                     const qzen_gpu_tensor *source, size_t source_offset,
                     size_t elements);
int qzen_gpu_copy_f32(qzen_gpu *gpu, qzen_gpu_tensor *destination,
                    size_t destination_offset,
                    const qzen_gpu_tensor *source, size_t source_offset,
                    size_t elements);
int qzen_gpu_rms_norm_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                        const qzen_gpu_tensor *input,
                        const qzen_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);
int qzen_gpu_adaln_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *input,
                     const qzen_gpu_tensor *norm_weight,
                     const qzen_gpu_tensor *modulation,
                     const qzen_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int qzen_gpu_gate_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *residual,
                    const qzen_gpu_tensor *branch,
                    const qzen_gpu_tensor *modulation,
                    const qzen_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot);
int qzen_gpu_qkv_rope_f32(qzen_gpu *gpu, qzen_gpu_tensor *query,
                        qzen_gpu_tensor *key, qzen_gpu_tensor *value,
                        const qzen_gpu_tensor *qkv,
                        const qzen_gpu_tensor *q_norm,
                        const qzen_gpu_tensor *k_norm,
                        const qzen_gpu_tensor *rope_cos,
                        const qzen_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon);
int qzen_gpu_sdpa_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *query, const qzen_gpu_tensor *key,
                    const qzen_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale);
int qzen_gpu_swiglu_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *fused, uint32_t rows,
                      uint32_t width);
int qzen_gpu_scale_add_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                         const qzen_gpu_tensor *residual,
                         const qzen_gpu_tensor *branch,
                         const qzen_gpu_tensor *scale, uint32_t rows,
                         uint32_t width);
/* residual + branch * scale[col], write hidden, then RMSNorm into norm.
 * Same fmaf association as the two kernels in sequence. Falls back if
 * QZEN_DISABLE_FUSED_SCALE_ADD_RMS=1 or width is too wide for the register tile. */
int qzen_gpu_scale_add_rms_norm_f32(qzen_gpu *gpu, qzen_gpu_tensor *hidden,
                                  qzen_gpu_tensor *norm,
                                  const qzen_gpu_tensor *residual,
                                  const qzen_gpu_tensor *branch,
                                  const qzen_gpu_tensor *scale,
                                  const qzen_gpu_tensor *norm_weight,
                                  uint32_t rows, uint32_t width,
                                  float epsilon);
int qzen_gpu_layer_norm_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                          const qzen_gpu_tensor *input,
                          const qzen_gpu_tensor *weight,
                          const qzen_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);
int qzen_gpu_video_qkv_rope_f32(qzen_gpu *gpu, qzen_gpu_tensor *query,
                              qzen_gpu_tensor *key, qzen_gpu_tensor *value,
                              const qzen_gpu_tensor *qkv,
                              const qzen_gpu_tensor *rope_cos,
                              const qzen_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);

/* H3 AudioVAE uses time-major [batch,length,channels] activations and stores
 * Conv1d/ConvTranspose1d weights in PyTorch OIK/IOK order respectively. */
int qzen_gpu_conv1d_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input,
                      const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation);
int qzen_gpu_conv1d_stride_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input,
                      const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation);
int qzen_gpu_conv_transpose1d_f32(
                      qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input,
                      const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding);
int qzen_gpu_weight_norm_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *vector,
                           const qzen_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner);
int qzen_gpu_add_scaled_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                          const qzen_gpu_tensor *left,
                          const qzen_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements);
int qzen_gpu_alias_free_snake_f32(
                          qzen_gpu *gpu, qzen_gpu_tensor *output,
                          const qzen_gpu_tensor *input,
                          const qzen_gpu_tensor *alpha_log,
                          const qzen_gpu_tensor *beta_log,
                          const qzen_gpu_tensor *upsample_filter,
                          const qzen_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels);
int qzen_gpu_snake1d_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *input,
                       const qzen_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels);
int qzen_gpu_audio_qkv_split_f32(qzen_gpu *gpu,
                       qzen_gpu_tensor *query, qzen_gpu_tensor *key,
                       qzen_gpu_tensor *value, const qzen_gpu_tensor *qkv,
                       const qzen_gpu_tensor *q_bias,
                       const qzen_gpu_tensor *k_bias,
                       const qzen_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim);
int qzen_gpu_sdpa_causal_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *query,
                       const qzen_gpu_tensor *key,
                       const qzen_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale);
int qzen_gpu_audio_attention_pool_f32(qzen_gpu *gpu,
                       qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim);
int qzen_gpu_geglu_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *gate,
                     const qzen_gpu_tensor *linear, uint32_t elements);
int qzen_gpu_clip_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum);

/* Visual-VAE encoder tensors use channels-last [B,T,H,W,C] storage. Spatial
 * padding reflects pixels while temporal front padding is zero-filled. */
int qzen_gpu_vae_encoder_pad_f32(
                    qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after);
int qzen_gpu_conv3d_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input,
                      const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width);
int qzen_gpu_vae_encoder_group_norm_silu_f32(
                      qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input,
                      const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);

/* Z-Image conv AutoencoderKL decoder. All tensors are NCHW F32; conv2d takes
 * kernel 1 or 3 with padding kernel/2 (weights [out, in, k, k]); the VAE SDPA
 * is a single-head spatial attention over [sequence, dim] rows whose scores
 * scratch must hold sequence*sequence elements. */
int qzen_gpu_conv2d_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input, const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t out_channels,
                      uint32_t in_channels, uint32_t height, uint32_t width,
                      uint32_t kernel, uint32_t padding);
int qzen_gpu_group_norm_chw_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                              const qzen_gpu_tensor *input,
                              const qzen_gpu_tensor *weight,
                              const qzen_gpu_tensor *bias, uint32_t channels,
                              uint32_t height, uint32_t width, uint32_t groups,
                              float epsilon, int silu);
int qzen_gpu_upsample_nearest2x_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                                  const qzen_gpu_tensor *input,
                                  uint32_t channels, uint32_t height,
                                  uint32_t width);
int qzen_gpu_transpose_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                         const qzen_gpu_tensor *input, uint32_t rows,
                         uint32_t columns);
int qzen_gpu_add_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                   const qzen_gpu_tensor *left, const qzen_gpu_tensor *right,
                   uint32_t elements);
int qzen_gpu_vae_sdpa_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                        const qzen_gpu_tensor *query, const qzen_gpu_tensor *key,
                        const qzen_gpu_tensor *value, qzen_gpu_tensor *scores,
                        uint32_t sequence, uint32_t dim, float scale);
/* Qwen-Image VAE helpers. The decoder normalises per spatial position over
 * the channel axis (y = x * sqrt(C) / ||x||_2 * gamma, channels first), which
 * is neither GroupNorm nor a row-wise RMSNorm, and its ResidualUpBlock adds a
 * DupUp3D channel shuffle of the block input to the upsampled main path. */
int qzen_gpu_vae_rms_chw_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input,
                           const qzen_gpu_tensor *gamma, uint32_t channels,
                           uint32_t height, uint32_t width);
int qzen_gpu_vae_dupup_chw_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                             const qzen_gpu_tensor *input, uint32_t in_channels,
                             uint32_t out_channels, uint32_t temporal,
                             uint32_t height, uint32_t width);

/* Portable BF16 storage path. Arithmetic accumulates in F32 and rounds at
 * operation boundaries, matching the released checkpoint's compute dtype. */
int qzen_gpu_linear_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *input,
                       const qzen_gpu_tensor *weight,
                       const qzen_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim);
/* output += input @ weight^T, no bias: the SVDQuant low-rank branch, which
 * lands in the result the quantised branch already wrote. */
int qzen_gpu_linear_bf16_accum(qzen_gpu *gpu, qzen_gpu_tensor *output,
                             const qzen_gpu_tensor *input,
                             const qzen_gpu_tensor *weight, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);

/* Expand one packed Nunchaku int4 matrix (qzen_nunchaku_packed) into the BF16
 * weight buffer a linear consumes.  This is what lets a 4-bit pack cost a
 * quarter of the VRAM instead of the same as BF16: the codes stay packed on the
 * device and are expanded per forward into one reusable scratch matrix, whose
 * largest layer is 20480 x 3840 x 2 B = 157 MB rather than the 10.6 GB of
 * resident BF16 weights (PLAN §19).
 *
 * codes   I8   [rows][columns / 2], low nibble = even column
 * scales  BF16 [rows][columns / 64], row-major
 * smooth  F32  [columns]
 * weight  BF16 [rows][columns], written
 *
 * The quantised branch only.  Bit-exact with qzen_nunchaku_packed_to_bf16_quant,
 * which is a memcmp and not a tolerance; the low-rank branch is two GEMMs the
 * caller runs (qzen_gpu_linear_bf16_accum), because at rank 256 doing it per
 * element cost 18x the whole DiT. */
int qzen_gpu_nunchaku_dequant(qzen_gpu *gpu, qzen_gpu_tensor *weight,
                            const qzen_gpu_tensor *codes,
                            const qzen_gpu_tensor *scales,
                            const qzen_gpu_tensor *smooth, uint32_t rows,
                            uint32_t columns);
/* Wan 2.2 umt5 text-encoder GEMM: F32 activations, BF16 weight rows (the host
 * quanto-dequantized copy uploaded verbatim), F32 output, exact FP32 FMAs
 * (CUBLAS_COMPUTE_32F, never TF32).  cublasGemmEx rejects the mixed
 * A=16BF/B=32F pair on this toolkit, so the activation is split into two
 * BF16 terms (hi = bf16(x), lo = bf16(x - f32(hi)); x = hi + lo to within one
 * BF16 rounding of lo) and summed through two A=B=16BF GEMMs into the F32
 * output.  Matches the CPU reference path, which widens the same BF16 weights
 * per element and accumulates in F32. No bias: T5 has none. */
int qzen_gpu_linear_bf16w_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                            const qzen_gpu_tensor *input,
                            const qzen_gpu_tensor *weight, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim);
/* T5 bidirectional self-attention with the per-layer relative position bias:
 * scores[i][j] = dot(q_i, k_j) + pos_bias[bucket[i*seq+j]*heads + head] (no
 * 1/sqrt(d) scaling), padded keys add -inf, F32 softmax over j, out = P @ V.
 * q/k/v/output are token-major F32 [seq, heads*head_dim]; buckets is U32
 * [seq*seq]; pos_bias is F32 [num_buckets*heads] (bucket-major, exactly the
 * layout qzen_wan22_text.c dequantizes into); mask is U32 [seq] (0 = padded)
 * or NULL.  Mirrors text_attention() in qzen_wan22_text.c. */
int qzen_gpu_t5_attn_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *query, const qzen_gpu_tensor *key,
                       const qzen_gpu_tensor *value,
                       const qzen_gpu_tensor *buckets,
                       const qzen_gpu_tensor *pos_bias,
                       const qzen_gpu_tensor *mask, uint32_t sequence,
                       uint32_t heads, uint32_t head_dim, uint32_t num_buckets);
/* Upload a host BF16 buffer through pinned staging memory.  Pageable
 * cudaMemcpy runs at ~1.8 GB/s on this box (the DiT resume measured it);
 * the pinned path moves the same bytes at PCIe speed.  Chunked through a
 * process-wide pinned buffer; the stream is synchronized before return, so
 * the destination is complete when this returns 1. */
int qzen_gpu_upload_i8_pinned(qzen_gpu *gpu, qzen_gpu_tensor *destination,
                            const int8_t *values, size_t elements);
int qzen_gpu_upload_bf16_pinned(qzen_gpu *gpu, qzen_gpu_tensor *destination,
                              const uint16_t *values, size_t elements);
int qzen_gpu_mlp_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *input,
                    const qzen_gpu_tensor *fc1_weight,
                    const qzen_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim);
/* Experimental M5 Metal 4 paired FC1/SwiGLU plus direct FC2 path. Available
 * only when the context was created with QZEN_NAX=mlp. */
int qzen_gpu_mlp_nax_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                        qzen_gpu_tensor *activated,
                        const qzen_gpu_tensor *input,
                        const qzen_gpu_tensor *fc1_weight,
                        const qzen_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* Experimental M5 Metal 4 int8 MLP. Weights use one F32 scale per output
 * channel; activations are quantized dynamically with one F32 scale per row. */
int qzen_gpu_quantize_weight_int8(qzen_gpu *gpu, qzen_gpu_tensor *output,
                                qzen_gpu_tensor *scales,
                                const qzen_gpu_tensor *input, uint32_t rows,
                                uint32_t columns);
/* ConvRot activation front end: rotate each 256-column group of a BF16 row by
 * the orthonormal Hadamard H = 2^-4 * kron(h4, h4, h4, h4), then quantize the
 * rotated row symmetrically to INT8 with one F32 scale per row. Weights of the
 * same projection are stored pre-rotated (W_rot = blockwise_256(W) @ H^T) and
 * quantized per output channel, so the pair feeds qzen_gpu_linear_int8_bf16
 * unchanged. `columns` must be a multiple of 256. */
int qzen_gpu_convrot_rotate_int8(qzen_gpu *gpu, qzen_gpu_tensor *quant,
                               qzen_gpu_tensor *scales,
                               const qzen_gpu_tensor *input_bf16, uint32_t rows,
                               uint32_t columns);
/* Same rotation as a BF16 -> BF16 transform, for projections whose INT8 entry
 * point quantizes its own input (every public INT8 linear does). columns must
 * be a multiple of 256; in-place is allowed. */
int qzen_gpu_convrot_rotate_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                               const qzen_gpu_tensor *input, uint32_t rows,
                               uint32_t columns);
/* F32 weights, one scale per output row. Used by opt-in QZEN_INT8_VAE. */
int qzen_gpu_quantize_weight_f32_int8(qzen_gpu *gpu, qzen_gpu_tensor *output,
                                    qzen_gpu_tensor *scales,
                                    const qzen_gpu_tensor *input, uint32_t rows,
                                    uint32_t columns);
int qzen_gpu_linear_f32_int8(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input,
                           const qzen_gpu_tensor *weight,
                           const qzen_gpu_tensor *weight_scales,
                           const qzen_gpu_tensor *bias, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim);
/* FP8-E4M3 linear. Weights carry one scale for the whole tensor, which the
 * GEMM folds in; activations keep a scale per token. Faster than the INT8 path
 * and less accurate — see the comment on the implementation. */
int qzen_gpu_quantize_weight_fp8(qzen_gpu *gpu, qzen_gpu_tensor *output,
                               qzen_gpu_tensor *scale,
                               const qzen_gpu_tensor *input, uint32_t rows,
                               uint32_t columns);
int qzen_gpu_linear_fp8_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           qzen_gpu_tensor *quantized_input,
                           qzen_gpu_tensor *input_scales,
                           const qzen_gpu_tensor *input,
                           const qzen_gpu_tensor *weight,
                           const qzen_gpu_tensor *weight_scale, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim);
int qzen_gpu_mlp_fp8_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                        qzen_gpu_tensor *fc1_output,
                        qzen_gpu_tensor *quantized_activation,
                        qzen_gpu_tensor *activation_scales,
                        const qzen_gpu_tensor *input,
                        const qzen_gpu_tensor *fc1_weight,
                        const qzen_gpu_tensor *fc1_scale,
                        const qzen_gpu_tensor *fc2_weight,
                        const qzen_gpu_tensor *fc2_scale, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* defer_scales leaves the INT8 GEMM's int32 accumulator unscaled for a
 * following consumer to fold the rescale into its own first read, skipping a
 * full BF16 round trip through memory. Consumers are
 * qzen_gpu_gate_adaln_quantize_int8, qzen_gpu_gate_adaln_bf16, and
 * qzen_gpu_gate_bf16 when called immediately afterwards: the next INT8 GEMM
 * reuses the same accumulator. Output is bit-identical either way. */
int qzen_gpu_linear_int8_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                            qzen_gpu_tensor *quantized_input,
                            qzen_gpu_tensor *input_scales,
                            const qzen_gpu_tensor *input,
                            const qzen_gpu_tensor *weight,
                            const qzen_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales,
                            int defer_scales);
/* Consume SDPA's native [head,row,dimension] BF16 layout without a full
 * BF16 transpose, gathering directly into the projection's row-major int8. */
int qzen_gpu_linear_int8_head_major_bf16(
                            qzen_gpu *gpu, qzen_gpu_tensor *output,
                            qzen_gpu_tensor *quantized_input,
                            qzen_gpu_tensor *input_scales,
                            const qzen_gpu_tensor *input,
                            const qzen_gpu_tensor *weight,
                            const qzen_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim,
                            int defer_scales);
int qzen_gpu_mlp_int8_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                         qzen_gpu_tensor *activated,
                         qzen_gpu_tensor *quantized_activation,
                         qzen_gpu_tensor *activation_scales,
                         const qzen_gpu_tensor *input,
                         const qzen_gpu_tensor *fc1_weight,
                         const qzen_gpu_tensor *fc1_scales,
                         const qzen_gpu_tensor *fc2_weight,
                         const qzen_gpu_tensor *fc2_scales,
                         const qzen_gpu_tensor *fc1_bf16,
                         const qzen_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized,
                         int defer_output_scales);
int qzen_gpu_silu_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *input, uint32_t elements);
int qzen_gpu_rms_norm_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                         const qzen_gpu_tensor *input,
                         const qzen_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);
int qzen_gpu_layer_norm_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input,
                           const qzen_gpu_tensor *weight,
                           const qzen_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon);
/* LayerNorm (mean/variance) fused with the Wan 2.2 adaLN:
 * out = LN(x) * (1 + scale) + shift over modulation slots. */
int qzen_gpu_layer_norm_adaln_bf16(
    qzen_gpu *gpu, qzen_gpu_tensor *output, const qzen_gpu_tensor *input,
    const qzen_gpu_tensor *norm_weight, const qzen_gpu_tensor *modulation,
    const qzen_gpu_tensor *residual_mod, const qzen_gpu_tensor *row_map,
    uint32_t rows, uint32_t width,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon);
/* LayerNorm over an F32 input (residual carrier) with BF16 affine. */
int qzen_gpu_layer_norm_f32in_bf16(
    qzen_gpu *gpu, qzen_gpu_tensor *output, const qzen_gpu_tensor *input,
    const qzen_gpu_tensor *weight, const qzen_gpu_tensor *bias, uint32_t rows,
    uint32_t width, float epsilon);
/* LayerNorm (mean/variance) over an F32 input fused with the Wan 2.2
 * adaLN (BF16 block table + shared e0, summed in F32). */
int qzen_gpu_layer_norm_adaln_f32in_bf16(
    qzen_gpu *gpu, qzen_gpu_tensor *output, const qzen_gpu_tensor *input,
    const qzen_gpu_tensor *norm_weight, const qzen_gpu_tensor *modulation,
    const qzen_gpu_tensor *residual_mod, const qzen_gpu_tensor *row_map,
    uint32_t rows, uint32_t width,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon);
/* Elementwise F32 -> BF16 conversion (residual staging). */
int qzen_gpu_cvt_f32_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                        const qzen_gpu_tensor *input, uint32_t elements);
/* Wan 2.2 residual update on an F32 carrier:
 * x_f32 += branch_bf16 * gate(slot). */
int qzen_gpu_gate_f32res_bf16(qzen_gpu *gpu, qzen_gpu_tensor *residual_f32,
                            const qzen_gpu_tensor *branch,
                            const qzen_gpu_tensor *modulation,
                            const qzen_gpu_tensor *residual_mod,
                            uint32_t residual_slots,
                            const qzen_gpu_tensor *row_map, uint32_t rows,
                            uint32_t width, uint32_t slots,
                            uint32_t gate_slot);
int qzen_gpu_gelu_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *input, uint32_t elements,
                     int approximate);
int qzen_gpu_vision_qkv_rope_bf16(
                     qzen_gpu *gpu, qzen_gpu_tensor *query,
                     qzen_gpu_tensor *key, qzen_gpu_tensor *value,
                     const qzen_gpu_tensor *qkv,
                     const qzen_gpu_tensor *rope_cos,
                     const qzen_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half);
int qzen_gpu_adaln_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input,
                      const qzen_gpu_tensor *norm_weight,
                      const qzen_gpu_tensor *modulation,
                      const qzen_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int qzen_gpu_adaln_bf16_offset(qzen_gpu *gpu, qzen_gpu_tensor *output,
                      const qzen_gpu_tensor *input, size_t input_offset,
                      const qzen_gpu_tensor *norm_weight,
                      const qzen_gpu_tensor *modulation,
                      const qzen_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int qzen_gpu_adaln_linear_bf16(
                      qzen_gpu *gpu, qzen_gpu_tensor *output,
                      qzen_gpu_tensor *inverse,
                      const qzen_gpu_tensor *input, size_t input_offset,
                      const qzen_gpu_tensor *norm_weight,
                      const qzen_gpu_tensor *modulation,
                      const qzen_gpu_tensor *row_map,
                      const qzen_gpu_tensor *weight,
                      const qzen_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon);
int qzen_gpu_gate_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *residual,
                     const qzen_gpu_tensor *branch,
                     const qzen_gpu_tensor *modulation,
                     const qzen_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot);
int qzen_gpu_gate_adaln_bf16(
                     qzen_gpu *gpu, qzen_gpu_tensor *gated_residual,
                     qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *residual,
                     const qzen_gpu_tensor *branch,
                     const qzen_gpu_tensor *norm_weight,
                     const qzen_gpu_tensor *gate_modulation,
                     const qzen_gpu_tensor *norm_modulation,
                     const qzen_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon);
int qzen_gpu_gate_adaln_quantize_int8(
                     qzen_gpu *gpu, qzen_gpu_tensor *gated_residual,
                     qzen_gpu_tensor *quantized_output,
                     qzen_gpu_tensor *quantized_scales,
                     const qzen_gpu_tensor *residual,
                     const qzen_gpu_tensor *branch,
                     const qzen_gpu_tensor *norm_weight,
                     const qzen_gpu_tensor *gate_modulation,
                     const qzen_gpu_tensor *norm_modulation,
                     const qzen_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon,
                     int fuse_branch_rescale);
int qzen_gpu_qkv_rope_bf16(qzen_gpu *gpu, qzen_gpu_tensor *query,
                         qzen_gpu_tensor *key, qzen_gpu_tensor *value,
                         const qzen_gpu_tensor *qkv,
                         const qzen_gpu_tensor *q_norm,
                         const qzen_gpu_tensor *k_norm,
                         const qzen_gpu_tensor *rope_cos,
                         const qzen_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon);
/* H3 checkpoint QKV rows are [head, q/k/v, dimension], unlike the
 * conventional [q/k/v, head, dimension] layout accepted above. */
int qzen_gpu_grouped_qkv_rope_bf16(qzen_gpu *gpu, qzen_gpu_tensor *query,
                                 qzen_gpu_tensor *key, qzen_gpu_tensor *value,
                                 const qzen_gpu_tensor *qkv,
                                 const qzen_gpu_tensor *q_norm,
                                 const qzen_gpu_tensor *k_norm,
                                 const qzen_gpu_tensor *rope_cos,
                                 const qzen_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
/* Project grouped H3 QKV and apply its exact Q/K norm/RoPE boundary. Metal 4
 * may route projections directly into the attention layout; other devices
 * retain the ordinary two calls. */
int qzen_gpu_grouped_qkv_linear_rope_bf16(
                                 qzen_gpu *gpu,
                                 qzen_gpu_tensor *query,
                                 qzen_gpu_tensor *key,
                                 qzen_gpu_tensor *value,
                                 qzen_gpu_tensor *qkv,
                                 const qzen_gpu_tensor *input,
                                 const qzen_gpu_tensor *weight,
                                 const qzen_gpu_tensor *q_norm,
                                 const qzen_gpu_tensor *k_norm,
                                 const qzen_gpu_tensor *rope_cos,
                                 const qzen_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon);
int qzen_gpu_grouped_qkv_linear_rope_int8(
                                 qzen_gpu *gpu,
                                 qzen_gpu_tensor *query,
                                 qzen_gpu_tensor *key,
                                 qzen_gpu_tensor *value,
                                 qzen_gpu_tensor *quantized_input,
                                 qzen_gpu_tensor *input_scales,
                                 const qzen_gpu_tensor *input,
                                 const qzen_gpu_tensor *weight,
                                 const qzen_gpu_tensor *weight_scales,
                                 const qzen_gpu_tensor *q_norm,
                                 const qzen_gpu_tensor *k_norm,
                                 const qzen_gpu_tensor *rope_cos,
                                 const qzen_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
int qzen_gpu_sdpa_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *query, const qzen_gpu_tensor *key,
                     const qzen_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Preserve SDPA's native [head,row,dimension] output for an immediately
 * following layout-aware projection. */
int qzen_gpu_sdpa_bf16_head_major_output(
                     qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *query, const qzen_gpu_tensor *key,
                     const qzen_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Cross-attention SDPA with independent query and key/value lengths
 * (Wan 2.2 cross_attn: q from the latent tokens, k/v from the text
 * context). */
int qzen_gpu_sdpa_cross_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *query,
                           const qzen_gpu_tensor *key,
                           const qzen_gpu_tensor *value,
                           uint32_t q_sequence, uint32_t kv_sequence,
                           uint32_t heads, uint32_t head_dim,
                           float scale, qzen_gpu_tensor *scores_scratch);
/* Flash-style SDPA: tensor-core tiles with an online softmax, no
 * materialized [heads, q, kv] scores, and independent query / key-value
 * lengths (Wan 2.2 self- and cross-attention). head_dim must be 128;
 * output is token-major [q_sequence, heads * head_dim] BF16. */
int qzen_gpu_sdpa_flash_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *query,
                           const qzen_gpu_tensor *key,
                           const qzen_gpu_tensor *value,
                           uint32_t q_sequence, uint32_t kv_sequence,
                           uint32_t heads, uint32_t head_dim, float scale);
/* Free/total device memory in bytes (cudaMemGetInfo), for smoke tests. */
int qzen_gpu_free_memory(qzen_gpu *gpu, size_t *free_bytes, size_t *total_bytes);
int qzen_gpu_swiglu_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *fused, uint32_t rows,
                       uint32_t width);
int qzen_gpu_embedding_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                          const qzen_gpu_tensor *weight,
                          const qzen_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width);
int qzen_gpu_text_qk_rope_bf16(qzen_gpu *gpu,
                             qzen_gpu_tensor *query_output,
                             qzen_gpu_tensor *key_output,
                             const qzen_gpu_tensor *query_input,
                             const qzen_gpu_tensor *key_input,
                             const qzen_gpu_tensor *q_norm,
                             const qzen_gpu_tensor *k_norm,
                             const qzen_gpu_tensor *rope_cos,
                             const qzen_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon);
int qzen_gpu_head_rms_norm_bf16(qzen_gpu *gpu, qzen_gpu_tensor *tensor,
                              const qzen_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);
int qzen_gpu_rope_text_bf16(qzen_gpu *gpu, qzen_gpu_tensor *query,
                          qzen_gpu_tensor *key,
                          const qzen_gpu_tensor *rope_cos_f32,
                          const qzen_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim);
int qzen_gpu_gqa_causal_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *query,
                           const qzen_gpu_tensor *key,
                           const qzen_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale);
int qzen_gpu_add_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *left, const qzen_gpu_tensor *right,
                    uint32_t elements);
int qzen_gpu_sub_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                    const qzen_gpu_tensor *left, const qzen_gpu_tensor *right,
                    uint32_t elements);
/* Relative L2 ||current - previous|| / ||previous||. Synchronizes stream. */
int qzen_gpu_rel_l2_bf16(qzen_gpu *gpu, const qzen_gpu_tensor *current,
                       const qzen_gpu_tensor *previous, uint32_t elements,
                       float *relative);
int qzen_gpu_token_pool_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input,
                           size_t input_offset,
                           qzen_gpu_tensor *original,
                           size_t original_offset,
                           qzen_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const qzen_gpu_tensor *baseline_indices,
                           const qzen_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width);
int qzen_gpu_token_pool_adaln_bf16(
                           qzen_gpu *gpu, qzen_gpu_tensor *residual,
                           qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input, size_t input_offset,
                           qzen_gpu_tensor *original, size_t original_offset,
                           qzen_gpu_tensor *baseline, size_t baseline_offset,
                           const qzen_gpu_tensor *baseline_indices,
                           const qzen_gpu_tensor *pairs,
                           const qzen_gpu_tensor *norm_weight,
                           const qzen_gpu_tensor *modulation,
                           const qzen_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
int qzen_gpu_token_expand_delta_bf16(
                           qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *original,
                           size_t original_offset,
                           const qzen_gpu_tensor *reduced,
                           const qzen_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const qzen_gpu_tensor *baseline_indices,
                           const qzen_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale);
int qzen_gpu_token_expand_adaln_bf16(
                           qzen_gpu *gpu, qzen_gpu_tensor *residual,
                           qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *original,
                           size_t original_offset,
                           const qzen_gpu_tensor *reduced,
                           const qzen_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const qzen_gpu_tensor *baseline_indices,
                           const qzen_gpu_tensor *parents,
                           const qzen_gpu_tensor *norm_weight,
                           const qzen_gpu_tensor *modulation,
                           const qzen_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
/* Apply one Euler step to an F32 sample range from BF16 velocity caches:
 * sample += delta * (last + ratio * (last - previous)). */
int qzen_gpu_euler_bf16(qzen_gpu *gpu, qzen_gpu_tensor *sample,
                      size_t sample_offset, const qzen_gpu_tensor *last,
                      const qzen_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio);
int qzen_gpu_silu_mul_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                         const qzen_gpu_tensor *gate,
                         const qzen_gpu_tensor *up, uint32_t elements);
/* Sol-Attn: layer_on 0 skips sparse SDPA for this block. prefix_blocks < 0
 * keeps the QZEN_SOL_ATTN_PREFIX default. */
void qzen_gpu_sol_attn_configure(qzen_gpu *gpu, int layer_on, int prefix_blocks);

/* Load-time PDD LoRA merge. weight = weight * scales + delta over [rows,
 * columns], then each row is requantized against a fresh scale in place.
 * delta may be NULL, which measures the unmerged rows (requantize only). */
int qzen_gpu_lora_merge_i8(qzen_gpu *gpu, qzen_gpu_tensor *weight,
                         qzen_gpu_tensor *scales, const qzen_gpu_tensor *delta,
                         uint32_t weight_row, uint32_t delta_row,
                         uint32_t rows, uint32_t columns);
/* weight += delta for the BF16 token refiner. The row offsets let the caller
 * scatter a contiguous adaptor into per-head qkv triples. */
int qzen_gpu_lora_add_bf16(qzen_gpu *gpu, qzen_gpu_tensor *weight,
                         const qzen_gpu_tensor *delta, uint32_t weight_row,
                         uint32_t delta_row, uint32_t rows, uint32_t columns);

/* --- Wan 2.2 T2V VAE decoder ops (NCTHW fp32, for WAN22_VAE_GPU=1) ---
 * Tensors are [C][T][H][W]. Causal convolutions zero-pad (kt-1) frames at
 * the temporal front and ks/2 pixels on every spatial side, matching the
 * CPU decoder in qzen_wan22_vae.c. */
int qzen_gpu_cconv3d_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                       const qzen_gpu_tensor *input, const qzen_gpu_tensor *weight,
                       const qzen_gpu_tensor *bias, uint32_t out_channels,
                       uint32_t in_channels, uint32_t t, uint32_t height,
                       uint32_t width, uint32_t kernel_t, uint32_t kernel_s);
/* Per-position RMS norm across channels (channel-first): out = in /
 * max(rms, eps) * sqrt(c) * gamma[ch], optional fused SiLU. */
int qzen_gpu_rms_norm_silu_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                             const qzen_gpu_tensor *input,
                             const qzen_gpu_tensor *gamma, uint32_t channels,
                             uint32_t t, uint32_t height, uint32_t width,
                             int silu);
int qzen_gpu_upsample_nearest2x_ncthw_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                                        const qzen_gpu_tensor *input,
                                        uint32_t channels, uint32_t t,
                                        uint32_t height, uint32_t width);
/* DupUp3d: channel repeat_interleave + temporal/spatial duplication. */
int qzen_gpu_dupup_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                     const qzen_gpu_tensor *input, uint32_t in_channels,
                     uint32_t out_channels, uint32_t ft, uint32_t fs,
                     int drop_first, uint32_t t, uint32_t height,
                     uint32_t width);
/* unpatchify(2x2): out[3][t][2h][2w] from in[12][t][h][w]. */
int qzen_gpu_unpatchify_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                          const qzen_gpu_tensor *input, uint32_t t,
                          uint32_t height, uint32_t width);
/* Per-frame single-head spatial SDPA: q/k/v/output are
 * [frames][sequence][dim], scores scratch [frames][sequence][sequence]. */
int qzen_gpu_vae_sdpa3d_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                          const qzen_gpu_tensor *query,
                          const qzen_gpu_tensor *key,
                          const qzen_gpu_tensor *value, qzen_gpu_tensor *scores,
                          uint32_t frames, uint32_t sequence, uint32_t dim,
                          float scale);
/* qkv [3c][t][plane] -> q/k/v [t*plane][c]; and back. */
int qzen_gpu_vae_qkv_pack_f32(qzen_gpu *gpu, qzen_gpu_tensor *q, qzen_gpu_tensor *k,
                            qzen_gpu_tensor *v, const qzen_gpu_tensor *qkv,
                            uint32_t channels, uint32_t t, uint32_t plane);
int qzen_gpu_vae_attn_unpack_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                               const qzen_gpu_tensor *attn, uint32_t channels,
                               uint32_t t, uint32_t plane);
/* Zero the first frame and drop input frame 0: output[c][t][plane] frame 0
 * = 0, frame i = input frame i for i >= 1 (the conv input reproducing the
 * CPU decoder's R[1:] slice conv). */
int qzen_gpu_frame_shift_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                           const qzen_gpu_tensor *input, uint32_t channels,
                           uint32_t t, uint32_t plane);
/* up3d Resample merge: out[c][2t-1][plane] frame 0 = first[c][t][plane]
 * frame 0, frames 1.. = rest[2c][t][plane] interleaved (output frame j
 * reads rest channel ((j+1)%2)*c + ch at frame (j+1)/2; rest is the
 * time_conv over the frame-shifted input). */
int qzen_gpu_vae_temporal_merge_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                                  const qzen_gpu_tensor *first,
                                  const qzen_gpu_tensor *rest,
                                  uint32_t channels, uint32_t t,
                                  uint32_t plane);
/* up3d continuation merge for streaming groups: out[c][2t][plane], frame
 * 2m = rest[ch][m][plane], frame 2m+1 = rest[c+ch][m][plane]. */
int qzen_gpu_vae_temporal_merge_cont_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                                       const qzen_gpu_tensor *rest,
                                       uint32_t channels, uint32_t t,
                                       size_t plane);
/* Causal Conv3d with a carried input state: the cache_frames frames before
 * the input group come from cache ([in_channels][cache_frames][height][width])
 * instead of the zero pad, so streaming groups match the whole-sequence conv
 * bit-exact. */
int qzen_gpu_cconv3d_cached_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                              const qzen_gpu_tensor *input,
                              const qzen_gpu_tensor *cache,
                              uint32_t cache_frames,
                              const qzen_gpu_tensor *weight,
                              const qzen_gpu_tensor *bias, uint32_t out_channels,
                              uint32_t in_channels, uint32_t t, uint32_t height,
                              uint32_t width, uint32_t kernel_t,
                              uint32_t kernel_s);
/* --- Tiled causal Conv3d fast path (WAN22_CONV_FAST, default on) ---
 * The reference kernels above are memory-latency bound (one global read per
 * FMA).  The tiled path packs the weight once into the k-major [k][O] layout
 * (k = ((c*kt + step)*ks + ky)*ks + kx) and then runs an implicit GEMM with
 * the input staged through shared memory.  Accumulation order is unchanged,
 * so the outputs are bit-identical to qzen_gpu_cconv3d_f32 / _cached_f32. */
int qzen_gpu_cconv3d_packed_f32(qzen_gpu *gpu, qzen_gpu_tensor *output,
                              const qzen_gpu_tensor *input,
                              const qzen_gpu_tensor *packed_weight,
                              const qzen_gpu_tensor *bias, uint32_t out_channels,
                              uint32_t in_channels, uint32_t t, uint32_t height,
                              uint32_t width, uint32_t kernel_t,
                              uint32_t kernel_s);
int qzen_gpu_cconv3d_packed_cached_f32(
    qzen_gpu *gpu, qzen_gpu_tensor *output, const qzen_gpu_tensor *input,
    const qzen_gpu_tensor *cache, uint32_t cache_frames,
    const qzen_gpu_tensor *packed_weight, const qzen_gpu_tensor *bias,
    uint32_t out_channels, uint32_t in_channels, uint32_t t, uint32_t height,
    uint32_t width, uint32_t kernel_t, uint32_t kernel_s);
/* Stream-ordered device-to-device fp32 copy of an element range. */
int qzen_gpu_copy_d2d_f32(qzen_gpu *gpu, qzen_gpu_tensor *dst, size_t dst_offset,
                        const qzen_gpu_tensor *src, size_t src_offset,
                        size_t elements);
/* Stream-ordered zero fill of an element range. */
int qzen_gpu_clear_f32(qzen_gpu *gpu, qzen_gpu_tensor *dst, size_t offset,
                     size_t elements);
/* Rolling two-frame tail update for the streaming decoder (per channel). */
int qzen_gpu_tail_update_f32(qzen_gpu *gpu, qzen_gpu_tensor *tail_new,
                           const qzen_gpu_tensor *tail_old,
                           const qzen_gpu_tensor *input, uint32_t channels,
                           uint32_t t_g, size_t plane, uint32_t frames_seen);
/* Qwen-Image 2.1 DiT additions. In-place interleaved-complex RoPE on
 * token-major [seq][heads * head_dim] BF16 q/k (pair (2p, 2p + 1), tables
 * BF16 [seq][head_dim / 2]), and the joint-attention causal prefix: the
 * flash kernel runs the whole sequence non-causally, this recomputes query
 * rows < prefix_rows causally (text prefix attends over itself with the
 * lower-triangle mask only). */
int qzen_gpu_qwen_rope_bf16(qzen_gpu *gpu, qzen_gpu_tensor *query,
                          qzen_gpu_tensor *key, const qzen_gpu_tensor *rope_cos,
                          const qzen_gpu_tensor *rope_sin, uint32_t sequence,
                          uint32_t heads, uint32_t head_dim);
int qzen_gpu_prefix_causal_bf16(qzen_gpu *gpu, qzen_gpu_tensor *output,
                              const qzen_gpu_tensor *query,
                              const qzen_gpu_tensor *key,
                              const qzen_gpu_tensor *value,
                              uint32_t prefix_rows, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim, float scale);
#endif
