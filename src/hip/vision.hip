// Vision-tower kernels for the mmproj (`clip.vision`, ling3vl projector).
//
// The mmproj stores BF16 weights, which the text kernels do not implement, so the
// loader dequantizes them to F32 and these kernels take F32 weights. Matrices in
// the mmproj are laid out [in][out] (the output dimension is contiguous), which
// is the transpose of the text GEMV's [out][in]; `MatvecInMajor` is the
// corresponding kernel.
#include "src/hip/vision.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::ext::ling3_air::hip {
namespace {

constexpr int kThreads = 256;

/// y[out] = sum_i x[i] * w[i][out] + bias[out]
__global__ void MatvecInMajorKernel(const float* x, const float* w, const float* bias,
                                    float* y, int in, int out) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) {
    return;
  }
  float sum = bias == nullptr ? 0.0f : bias[o];
  for (int i = 0; i < in; ++i) {
    sum += x[i] * w[static_cast<std::size_t>(i) * out + o];
  }
  y[o] = sum;
}

/// Batched form: rows tokens share the weight read.
__global__ void MatvecInMajorBatchKernel(const float* x, const float* w,
                                         const float* bias, float* y, int tokens,
                                         int in, int out) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) {
    return;
  }
  for (int t = 0; t < tokens; ++t) {
    float sum = bias == nullptr ? 0.0f : bias[o];
    const float* row = x + static_cast<std::size_t>(t) * in;
    for (int i = 0; i < in; ++i) {
      sum += row[i] * w[static_cast<std::size_t>(i) * out + o];
    }
    y[static_cast<std::size_t>(t) * out + o] = sum;
  }
}

/// LayerNorm with weight and bias, one block per row.
__global__ void LayerNormKernel(const float* x, const float* weight, const float* bias,
                                float* y, int rows, int cols, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* source = x + static_cast<std::size_t>(row) * cols;
  float* target = y + static_cast<std::size_t>(row) * cols;
  __shared__ float reduce[32];
  const int warps = static_cast<int>((blockDim.x + 31) / 32);
  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += source[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    reduce[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  float mean = 0.0f;
  for (int w = 0; w < warps; ++w) {
    mean += reduce[w];
  }
  mean /= static_cast<float>(cols);
  __syncthreads();
  float squares = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float value = source[i] - mean;
    squares += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    squares += __shfl_down(squares, offset);
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    reduce[threadIdx.x >> 5] = squares;
  }
  __syncthreads();
  float variance = 0.0f;
  for (int w = 0; w < warps; ++w) {
    variance += reduce[w];
  }
  variance /= static_cast<float>(cols);
  const float scale = rsqrtf(variance + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    target[i] = (source[i] - mean) * scale * weight[i] + bias[i];
  }
}

/// act[i] = gelu_tanh(x[i]) (clip.use_gelu records the tanh approximation)
__global__ void GeluKernel(const float* x, float* y, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
  }
}

__global__ void AddKernel(float* y, const float* a, const float* b, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    y[i] = a[i] + b[i];
  }
}

/// Bidirectional self-attention over `tokens` patch rows (the vision tower is not
/// causal, unlike the text decoder), one block per (head, token).
__global__ void AttentionKernel(const float* qkv, float* out, int tokens, int heads,
                                int head_dim) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  if (head >= heads || token >= tokens) {
    return;
  }
  const int dim = heads * head_dim;
  const float* q = qkv + static_cast<std::size_t>(token) * 3 * dim + head * head_dim;
  __shared__ float scores[8192];
  // Scores for this (head, token) against every key: bidirectional.
  for (int other = threadIdx.x; other < tokens; other += blockDim.x) {
    const float* k =
        qkv + static_cast<std::size_t>(other) * 3 * dim + dim + head * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += q[d] * k[d];
    }
    scores[other] = dot / sqrtf(static_cast<float>(head_dim));
  }
  __syncthreads();
  // Softmax over the whole row.
  float local_max = -INFINITY;
  for (int other = threadIdx.x; other < tokens; other += blockDim.x) {
    local_max = fmaxf(local_max, scores[other]);
  }
  __shared__ float reduce[32];
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    reduce[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  float maximum = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) {
    maximum = fmaxf(maximum, reduce[w]);
  }
  __syncthreads();
  float total = 0.0f;
  for (int other = threadIdx.x; other < tokens; other += blockDim.x) {
    scores[other] = expf(scores[other] - maximum);
    total += scores[other];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    total += __shfl_down(total, offset);
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    reduce[threadIdx.x >> 5] = total;
  }
  __syncthreads();
  float denominator = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) {
    denominator += reduce[w];
  }
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float sum = 0.0f;
    for (int other = 0; other < tokens; ++other) {
      const float* v =
          qkv + static_cast<std::size_t>(other) * 3 * dim + 2 * dim + head * head_dim;
      sum += scores[other] / denominator * v[d];
    }
    out[static_cast<std::size_t>(token) * dim + head * head_dim + d] = sum;
  }
}

}  // namespace

void LaunchMatvecInMajor(const float* x, const float* w, const float* bias, float* y,
                         std::size_t in, std::size_t out, hipStream_t stream) {
  const int blocks = static_cast<int>((out + kThreads - 1) / kThreads);
  MatvecInMajorKernel<<<blocks, kThreads, 0, stream>>>(x, w, bias, y,
                                                      static_cast<int>(in),
                                                      static_cast<int>(out));
}

void LaunchMatvecInMajorBatch(const float* x, const float* w, const float* bias,
                              float* y, std::size_t tokens, std::size_t in,
                              std::size_t out, hipStream_t stream) {
  const int blocks = static_cast<int>((out + kThreads - 1) / kThreads);
  MatvecInMajorBatchKernel<<<blocks, kThreads, 0, stream>>>(
      x, w, bias, y, static_cast<int>(tokens), static_cast<int>(in),
      static_cast<int>(out));
}

void LaunchLayerNorm(const float* x, const float* weight, const float* bias, float* y,
                     std::size_t rows, std::size_t cols, float eps,
                     hipStream_t stream) {
  LayerNormKernel<<<static_cast<int>(rows), kThreads, 0, stream>>>(
      x, weight, bias, y, static_cast<int>(rows), static_cast<int>(cols), eps);
}

void LaunchGelu(const float* x, float* y, std::size_t count, hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  GeluKernel<<<blocks, kThreads, 0, stream>>>(x, y, static_cast<int>(count));
}

void LaunchAddInto(float* y, const float* a, const float* b, std::size_t count,
                   hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  AddKernel<<<blocks, kThreads, 0, stream>>>(y, a, b, static_cast<int>(count));
}

void LaunchVisionAttention(const float* qkv, float* out, std::size_t tokens,
                           std::size_t heads, std::size_t head_dim,
                           hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(tokens));
  AttentionKernel<<<grid, 128, 0, stream>>>(qkv, out, static_cast<int>(tokens),
                                           static_cast<int>(heads),
                                           static_cast<int>(head_dim));
}

}  // namespace gufo::ext::ling3_air::hip
