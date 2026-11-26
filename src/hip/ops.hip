#include "src/hip/ops.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_fp16.h>

namespace gufo::ext::ling3_air::hip {
namespace {

constexpr int kThreads = 256;

__global__ void KdaGateKernel(const float* pre, const float* bias, const float* a,
                              float* out, int channels, int head_dim,
                              float lower_bound) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= channels) {
    return;
  }
  const float scale = a[i / head_dim];
  out[i] = (1.0f / (1.0f + expf(-(pre[i] + bias[i]) * scale))) * lower_bound;
}

__global__ void SigmoidKernel(const float* x, float* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

__global__ void MulSigmoidKernel(float* out, const float* gate, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] *= 1.0f / (1.0f + expf(-gate[i]));
  }
}

__global__ void SplitQKernel(const float* q, float* q_nope, float* q_pe,
                             int heads, int nope, int rope) {
  const int head = blockIdx.x;
  const int d = threadIdx.x;
  if (head >= heads) {
    return;
  }
  const float* row = q + static_cast<std::size_t>(head) * (nope + rope);
  if (d < nope) {
    q_nope[static_cast<std::size_t>(head) * nope + d] = row[d];
  }
  if (d < rope) {
    q_pe[static_cast<std::size_t>(head) * rope + d] = row[nope + d];
  }
}

__global__ void MlaAppendKernel(const float* latent, const float* k_pe,
                                float* cache, std::size_t position,
                                int latent_count, int rope) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float* row = cache + position * static_cast<std::size_t>(latent_count + rope);
  // llama.cpp keeps the KV cache in f16 by default; rounding on store matches
  // that precision instead of keeping fp32 (notes/engine-plan.md).
  if (i < latent_count) {
    row[i] = __half2float(__float2half(latent[i]));
  }
  if (i < rope) {
    row[latent_count + i] = __half2float(__float2half(k_pe[i]));
  }
}

__global__ void EmbeddingRowKernel(const void* weights, int type, int vocab,
                                   int cols, int token, float* out) {
  if (token < 0 || token >= vocab) {
    return;
  }
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) {
    return;
  }
  if (type == 0) {  // F32
    const auto* row = static_cast<const float*>(weights) +
                      static_cast<std::size_t>(token) * cols;
    out[i] = row[i];
    return;
  }
  // Q8_0: 32-value blocks with an fp16 scale.
  const int blocks_per_row = cols / 32;
  const auto* base = static_cast<const std::uint8_t*>(weights) +
                     static_cast<std::size_t>(token) * blocks_per_row * 34;
  const auto* block = reinterpret_cast<const std::uint8_t*>(base) + (i / 32) * 34;
  std::uint16_t scale_bits;
  std::memcpy(&scale_bits, block, sizeof(scale_bits));
  const __half scale = *reinterpret_cast<const __half*>(&scale_bits);
  const auto quant = static_cast<std::int8_t>(block[2 + (i % 32)]);
  out[i] = __half2float(scale) * static_cast<float>(quant);
}

__global__ void DequantizeQ8_0Kernel(const std::uint8_t* src, int blocks,
                                    float* dst) {
  const int block = blockIdx.x;
  const int i = threadIdx.x;
  if (block >= blocks) {
    return;
  }
  const std::uint8_t* base = src + static_cast<std::size_t>(block) * 34;
  std::uint16_t scale_bits;
  std::memcpy(&scale_bits, base, sizeof(scale_bits));
  const __half scale = *reinterpret_cast<const __half*>(&scale_bits);
  const auto quant = static_cast<std::int8_t>(base[2 + i]);
  dst[static_cast<std::size_t>(block) * 32 + i] =
      __half2float(scale) * static_cast<float>(quant);
}

__global__ void ScaleCopyKernel(const float* x, float* y, float scale, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    y[i] = x[i] * scale;
  }
}

}  // namespace

void LaunchKdaGate(const float* pre, const float* bias, const float* a,
                   float* out, std::size_t channels, std::size_t head_dim,
                   float lower_bound, hipStream_t stream) {
  const int blocks = static_cast<int>((channels + kThreads - 1) / kThreads);
  KdaGateKernel<<<blocks, kThreads, 0, stream>>>(pre, bias, a, out,
                                                static_cast<int>(channels),
                                                static_cast<int>(head_dim),
                                                lower_bound);
}

void LaunchSigmoid(const float* x, float* out, std::size_t count,
                   hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  SigmoidKernel<<<blocks, kThreads, 0, stream>>>(x, out, static_cast<int>(count));
}

void LaunchMulSigmoid(float* out, const float* gate, std::size_t count,
                      hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  MulSigmoidKernel<<<blocks, kThreads, 0, stream>>>(out, gate,
                                                   static_cast<int>(count));
}

void LaunchSplitQ(const float* q, float* q_nope, float* q_pe, std::size_t heads,
                  std::size_t nope, std::size_t rope, hipStream_t stream) {
  SplitQKernel<<<static_cast<int>(heads), 192, 0, stream>>>(
      q, q_nope, q_pe, static_cast<int>(heads), static_cast<int>(nope),
      static_cast<int>(rope));
}

void LaunchMlaAppend(const float* latent, const float* k_pe, float* cache,
                     std::size_t position, std::size_t latent_count,
                     std::size_t rope, hipStream_t stream) {
  const int blocks = static_cast<int>((latent_count + kThreads - 1) / kThreads);
  MlaAppendKernel<<<blocks, kThreads, 0, stream>>>(
      latent, k_pe, cache, position, static_cast<int>(latent_count),
      static_cast<int>(rope));
}

void LaunchEmbeddingRow(const void* weights, int type, std::size_t vocab,
                        std::size_t cols, std::int32_t token, float* out,
                        hipStream_t stream) {
  const int blocks = static_cast<int>((cols + kThreads - 1) / kThreads);
  EmbeddingRowKernel<<<blocks, kThreads, 0, stream>>>(
      weights, type, static_cast<int>(vocab), static_cast<int>(cols),
      static_cast<int>(token), out);
}

void LaunchDequantize(const void* src, int type, std::size_t count, float* dst,
                      hipStream_t stream) {
  if (type == 0) {
    hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, stream);
    return;
  }
  const int blocks = static_cast<int>(count / 32);
  DequantizeQ8_0Kernel<<<blocks, 32, 0, stream>>>(
      static_cast<const std::uint8_t*>(src), blocks, dst);
}

void LaunchScaleCopy(const float* x, float* y, float scale, std::size_t count,
                     hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  ScaleCopyKernel<<<blocks, kThreads, 0, stream>>>(x, y, scale,
                                                  static_cast<int>(count));
}

}  // namespace gufo::ext::ling3_air::hip
