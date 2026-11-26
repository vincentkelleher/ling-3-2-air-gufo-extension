// HIP kernels for the Ling 3.2 Air text engine.
//
// The host side stays in plain C++ (this header), so the operator tests and
// the runner share one entry point and no HIP type leaks into host code.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace gufo::ext::ling3_air::hip {

/// Storage layouts the GEMV kernel implements. These match the GGUF types the
/// pinned quantization uses; `weights.cpp` refuses anything else.
enum StorageType : int {
  kF32 = 0,
  kQ8_0 = 1,
  kQ4K = 2,
  kQ6K = 3,
  kQ5K = 4,
};

/// Bytes occupied by one row of `cols` weights in `type`.
[[nodiscard]] std::size_t RowBytes(StorageType type, std::size_t cols);

/// y[row] = dot(weights[row], x) for `rows` rows of `cols` weights.
/// One warp per row, one lane per quantized block (stride 32).
void LaunchGemv(const void* weights, StorageType type, std::size_t rows,
                std::size_t cols, const float* x, float* y,
                hipStream_t stream);

/// y[row] = weights[row] * x + bias[row] (bias may be null).
void LaunchGemvAdd(const void* weights, StorageType type, std::size_t rows,
                   std::size_t cols, const float* x, const float* bias, float* y,
                   hipStream_t stream);

/// y[slot][row] = dot(weights[ids[slot]][row], x + slot * batch_stride) for every
/// selected expert, in one launch (batch_stride 0 shares one input).
void LaunchGemvExperts(const void* weights, StorageType type,
                       const std::int32_t* ids, std::size_t experts,
                       std::size_t rows_per_expert, std::size_t cols,
                       const float* x, std::size_t batch_stride, float* y,
                       std::size_t out_stride, hipStream_t stream);

/// Batched dot: `y[m][row] = dot(weights[row], x[m])` for up to 8 tokens.
/// Supports F32, Q8_0 and Q4_K; the batch is the prefill dimension.
void LaunchGemm(const void* weights, StorageType type, std::size_t rows,
                std::size_t cols, const float* x, std::size_t batch,
                std::size_t x_stride, float* y, std::size_t y_stride,
                hipStream_t stream);

/// y = x * rsqrt(mean(x^2) + eps) * weight, one block per row.
void LaunchRmsNorm(const float* x, const float* weight, float* y,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream);

/// y = x * sigmoid(x)
void LaunchSilu(const float* x, float* y, std::size_t count, hipStream_t stream);

/// y = x * sigmoid(gate)
void LaunchSiluMul(const float* x, const float* gate, float* y, std::size_t count,
                   hipStream_t stream);

}  // namespace gufo::ext::ling3_air::hip
