// Small elementwise and layout kernels the Ling 3.2 blocks need.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace gufo::ext::ling3_air::hip {

/// KDA decay gate: `out[ch] = sigmoid((pre[ch] + bias[ch]) * a[ch / head_dim])
/// * lower_bound`, with `a` one scale per head.
void LaunchKdaGate(const float* pre, const float* bias, const float* a,
                   float* out, std::size_t channels, std::size_t head_dim,
                   float lower_bound, hipStream_t stream);

/// `out[i] = 1 / (1 + exp(-x[i]))`
void LaunchSigmoid(const float* x, float* out, std::size_t count,
                   hipStream_t stream);

/// `out[i] *= sigmoid(gate[i])`
void LaunchMulSigmoid(float* out, const float* gate, std::size_t count,
                      hipStream_t stream);

/// Splits MLA queries `[heads][nope + rope]` into two contiguous buffers.
void LaunchSplitQ(const float* q, float* q_nope, float* q_pe, std::size_t heads,
                  std::size_t nope, std::size_t rope, hipStream_t stream);

/// Copies one cache row: `[latent (512), k_pe (64)]` at `position`.
void LaunchMlaAppend(const float* latent, const float* k_pe, float* cache,
                     std::size_t position, std::size_t latent_count,
                     std::size_t rope, hipStream_t stream);

/// Dequantizes one vocabulary row of the token embedding into `out`
/// (`cols` floats): the model's first step.
void LaunchEmbeddingRow(const void* weights, int type, std::size_t vocab,
                        std::size_t cols, std::int32_t token, float* out,
                        hipStream_t stream);

/// Dequantizes `count` weights of `type` (0 = F32, 1 = Q8_0) into floats.
void LaunchDequantize(const void* src, int type, std::size_t count, float* dst,
                      hipStream_t stream);

/// `y = x * scale`
void LaunchScaleCopy(const float* x, float* y, float scale, std::size_t count,
                     hipStream_t stream);

}  // namespace gufo::ext::ling3_air::hip
