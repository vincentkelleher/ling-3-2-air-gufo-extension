// Ling 3.2 Air HIP kernels.
//
// Layouts are GGUF/llama.cpp's: Q8_0 in 32-weight blocks (fp16 scale + int8),
// Q4_K and Q6_K in 256-weight super-blocks. The unpacking matches Gufo's host
// reference (src/core/quant/ggml_dequant.cpp), which tests/dot_test.cpp checks
// against `Dot()`; tests/operator_gpu_test.cpp checks these kernels against a
// host dot on the same bytes.
#include "src/hip/kernels.hpp"

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::ling3_air::hip {
namespace {

#pragma pack(push, 1)
struct BlockQ8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
struct BlockQ6K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block layout");
static_assert(sizeof(BlockQ4K) == 144, "Q4_K block layout");
static_assert(sizeof(BlockQ6K) == 210, "Q6_K block layout");

__device__ __forceinline__ float HalfToFloat(std::uint16_t bits) {
  const __half value = *reinterpret_cast<const __half*>(&bits);
  return __half2float(value);
}

/// Q4_K sub-block scale and minimum (Gufo host: GetQ4ScaleMin).
__device__ __forceinline__ void Q4ScaleMin(int index, const std::uint8_t* packed,
                                           float& scale, float& minimum) {
  if (index < 4) {
    scale = static_cast<float>(packed[index] & 0x3FU);
    minimum = static_cast<float>(packed[index + 4] & 0x3FU);
    return;
  }
  scale = static_cast<float>((packed[index + 4] & 0x0FU) |
                             ((packed[index - 4] >> 6U) << 4U));
  minimum = static_cast<float>((packed[index + 4] >> 4U) |
                               ((packed[index] >> 6U) << 4U));
}

__device__ __forceinline__ float Q4Value(const BlockQ4K& block, int index) {
  const int group = index / 32;
  const int pair = group / 2;
  const bool high = (group & 1) != 0;
  const int lane = index % 32;
  const std::uint8_t packed = block.qs[pair * 32 + lane];
  const float quant = static_cast<float>(high ? (packed >> 4U) : (packed & 0x0FU));
  float scale = 0.0f;
  float minimum = 0.0f;
  Q4ScaleMin(group, block.scales, scale, minimum);
  return HalfToFloat(block.d) * scale * quant - HalfToFloat(block.dmin) * minimum;
}

__device__ __forceinline__ float Q6Value(const BlockQ6K& block, int index) {
  const int half = index / 128;
  const int within = index % 128;
  const int segment = within / 32;
  const int lane = within % 32;
  const int ql_base = half * 64;
  const std::uint8_t qh = block.qh[half * 32 + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }
  const int scale_index = half * 8 + lane / 16 + segment * 2;
  const float quant = static_cast<float>(static_cast<int>((high << 4U) | low) - 32);
  return HalfToFloat(block.d) * static_cast<float>(block.scales[scale_index]) * quant;
}

__device__ __forceinline__ float WarpSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

template <int TYPE>
__device__ __forceinline__ float DotRow(const std::uint8_t* row, const float* x,
                                        int cols) {
  if constexpr (TYPE == kF32) {
    const float* weights = reinterpret_cast<const float*>(row);
    float sum = 0.0f;
    for (int i = threadIdx.x % 32; i < cols; i += 32) {
      sum += weights[i] * x[i];
    }
    return WarpSum(sum);
  } else if constexpr (TYPE == kQ8_0) {
    const auto* blocks = reinterpret_cast<const BlockQ8_0*>(row);
    const int count = cols / 32;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float inner = 0.0f;
      for (int j = 0; j < 32; ++j) {
        inner += static_cast<float>(blocks[b].qs[j]) * x[b * 32 + j];
      }
      sum += HalfToFloat(blocks[b].d) * inner;
    }
    return WarpSum(sum);
  } else if constexpr (TYPE == kQ4K) {
    const auto* blocks = reinterpret_cast<const BlockQ4K*>(row);
    const int count = cols / 256;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float inner = 0.0f;
      for (int j = 0; j < 256; ++j) {
        inner += Q4Value(blocks[b], j) * x[b * 256 + j];
      }
      sum += inner;
    }
    return WarpSum(sum);
  } else {
    const auto* blocks = reinterpret_cast<const BlockQ6K*>(row);
    const int count = cols / 256;
    float sum = 0.0f;
    for (int b = threadIdx.x % 32; b < count; b += 32) {
      float inner = 0.0f;
      for (int j = 0; j < 256; ++j) {
        inner += Q6Value(blocks[b], j) * x[b * 256 + j];
      }
      sum += inner;
    }
    return WarpSum(sum);
  }
}

template <int TYPE>
__global__ void GemvKernel(const std::uint8_t* weights, const float* x,
                           float* y, int rows, int cols, int row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= rows) {
    return;
  }
  const std::uint8_t* row = weights + static_cast<std::size_t>(warp) * row_bytes;
  const float sum = DotRow<TYPE>(row, x, cols);
  if (lane == 0) {
    y[warp] = sum;
  }
}

template <int TYPE>
__global__ void GemvAddKernel(const std::uint8_t* weights, const float* x,
                              const float* bias, float* y, int rows, int cols,
                              int row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= rows) {
    return;
  }
  const std::uint8_t* row = weights + static_cast<std::size_t>(warp) * row_bytes;
  const float sum = DotRow<TYPE>(row, x, cols);
  if (lane == 0) {
    y[warp] = sum + (bias == nullptr ? 0.0f : bias[warp]);
  }
}

__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              int rows, int cols, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* source = x + static_cast<std::size_t>(row) * cols;
  float* target = y + static_cast<std::size_t>(row) * cols;
  __shared__ float shared;
  if (threadIdx.x == 0) {
    shared = 0.0f;
  }
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += source[i] * source[i];
  }
  sum = WarpSum(sum);
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&shared, sum);
  }
  __syncthreads();
  const float scale = rsqrtf(shared / static_cast<float>(cols) + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    target[i] = source[i] * scale * weight[i];
  }
}

__global__ void SiluKernel(const float* x, float* y, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

__global__ void SiluMulKernel(const float* x, const float* gate, float* y,
                              std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    y[i] = (v / (1.0f + expf(-v))) * (1.0f / (1.0f + expf(-gate[i])));
  }
}

constexpr int kThreads = 256;

}  // namespace

std::size_t RowBytes(StorageType type, std::size_t cols) {
  switch (type) {
    case kF32:
      return cols * sizeof(float);
    case kQ8_0:
      return cols / 32 * sizeof(BlockQ8_0);
    case kQ4K:
      return cols / 256 * sizeof(BlockQ4K);
    case kQ6K:
      return cols / 256 * sizeof(BlockQ6K);
  }
  return 0;
}

void LaunchGemv(const void* weights, StorageType type, std::size_t rows,
                std::size_t cols, const float* x, float* y,
                hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int warps = static_cast<int>(rows);
  const int blocks = (warps * 32 + kThreads - 1) / kThreads;
  const auto* data = static_cast<const std::uint8_t*>(weights);
  switch (type) {
    case kF32:
      GemvKernel<kF32><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ8_0:
      GemvKernel<kQ8_0><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ4K:
      GemvKernel<kQ4K><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ6K:
      GemvKernel<kQ6K><<<blocks, kThreads, 0, stream>>>(
          data, x, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
  }
}

void LaunchGemvAdd(const void* weights, StorageType type, std::size_t rows,
                   std::size_t cols, const float* x, const float* bias, float* y,
                   hipStream_t stream) {
  const int row_bytes = static_cast<int>(RowBytes(type, cols));
  const int warps = static_cast<int>(rows);
  const int blocks = (warps * 32 + kThreads - 1) / kThreads;
  const auto* data = static_cast<const std::uint8_t*>(weights);
  switch (type) {
    case kF32:
      GemvAddKernel<kF32><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ8_0:
      GemvAddKernel<kQ8_0><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ4K:
      GemvAddKernel<kQ4K><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
    case kQ6K:
      GemvAddKernel<kQ6K><<<blocks, kThreads, 0, stream>>>(
          data, x, bias, y, static_cast<int>(rows), static_cast<int>(cols), row_bytes);
      break;
  }
}

void LaunchRmsNorm(const float* x, const float* weight, float* y,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream) {
  RmsNormKernel<<<static_cast<int>(rows), kThreads, 0, stream>>>(
      x, weight, y, static_cast<int>(rows), static_cast<int>(cols), eps);
}

void LaunchSilu(const float* x, float* y, std::size_t count, hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  SiluKernel<<<blocks, kThreads, 0, stream>>>(x, y, count);
}

void LaunchSiluMul(const float* x, const float* gate, float* y, std::size_t count,
                   hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  SiluMulKernel<<<blocks, kThreads, 0, stream>>>(x, gate, y, count);
}

}  // namespace gufo::ext::ling3_air::hip
