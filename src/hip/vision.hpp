// Vision-tower kernels (mmproj). F32 weights, since the loader dequantizes the
// mmproj's BF16 tensors once.
#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

namespace gufo::ext::ling3_air::hip {

/// y[o] = sum_i x[i] * w[i][o] + bias[o], with `w` laid out [in][out] as the
/// mmproj stores it (the transpose of the text GEMV layout).
void LaunchMatvecInMajor(const float* x, const float* w, const float* bias, float* y,
                         std::size_t in, std::size_t out, hipStream_t stream);

/// The same for `tokens` rows of `x` ([tokens][in] -> [tokens][out]), so one
/// weight read serves the whole patch sequence.
void LaunchMatvecInMajorBatch(const float* x, const float* w, const float* bias,
                              float* y, std::size_t tokens, std::size_t in,
                              std::size_t out, hipStream_t stream);

/// LayerNorm with weight and bias over `cols`, one block per row.
void LaunchLayerNorm(const float* x, const float* weight, const float* bias, float* y,
                     std::size_t rows, std::size_t cols, float eps,
                     hipStream_t stream);

/// GELU, tanh approximation (`clip.use_gelu`).
void LaunchGelu(const float* x, float* y, std::size_t count, hipStream_t stream);

/// y = a + b
void LaunchAddInto(float* y, const float* a, const float* b, std::size_t count,
                   hipStream_t stream);

/// x[t] += pos[t] for `tokens` rows (the caller decides the position mapping).
void LaunchAddPosEmb(float* x, const float* pos, std::size_t tokens, std::size_t cols,
                     std::size_t grid, std::size_t dim, hipStream_t stream);

/// Gathers 2x2 patch neighbourhoods into merged rows of `4*dim`.
void LaunchMerge2x2(const float* x, float* out, std::size_t rows, std::size_t cols,
                    std::size_t dim, hipStream_t stream);

/// Bidirectional self-attention over `tokens` rows of a packed qkv buffer.
void LaunchVisionAttention(const float* qkv, float* out, std::size_t tokens,
                           std::size_t heads, std::size_t head_dim,
                           hipStream_t stream);

}  // namespace gufo::ext::ling3_air::hip
