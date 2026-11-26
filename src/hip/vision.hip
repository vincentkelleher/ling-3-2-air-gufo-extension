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
  const float* row = w + static_cast<std::size_t>(o) * in;
  for (int i = 0; i < in; ++i) {
    sum += x[i] * row[i];
  }
  y[o] = sum;
}

/// Batched form: rows tokens share the weight read.
__global__ void MatvecOutMajorBatchKernel(const float* x, const float* w,
                                          const float* bias, float* y, int tokens,
                                          int in, int out) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) {
    return;
  }
  const float* row = w + static_cast<std::size_t>(o) * in;
  for (int t = 0; t < tokens; ++t) {
    float sum = bias == nullptr ? 0.0f : bias[o];
    const float* x_row = x + static_cast<std::size_t>(t) * in;
    for (int i = 0; i < in; ++i) {
      sum += x_row[i] * row[i];
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

/// Vision M-RoPE, as llama.cpp's ling3vl graph applies it: for a head of `head_dim`
/// the first `head_dim/2` dims pair with the last `head_dim/2`, the first half of
/// those pairs rotate by the patch's row position and the second half by its column,
/// with frequency 10000^(-2 i / (head_dim/2)) reset at each section. Applied to q and
/// k in place; qkv is laid out [token][3 * dim] with v after k.
__global__ void VisionRopeKernel(float* qkv, int tokens, int heads, int head_dim,
                                 int cols) {
  const int t = blockIdx.x;   // token
  const int head = blockIdx.y;
  if (t >= tokens || head >= heads) {
    return;
  }
  const int dim = heads * head_dim;
  const int half = head_dim / 2;      // 36 for this tower
  const int section = half / 2;       // 18
  const int row = cols > 0 ? t / cols : 0;
  const int col = cols > 0 ? t % cols : t;
  float* q = qkv + static_cast<std::size_t>(t) * 3 * dim + head * head_dim;
  float* k = q + dim;
  // Threads split the rotated dims; without this every thread would re-apply the
  // whole rotation and the result would depend on scheduling.
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const int position = i < section ? row : col;
    const int index = i < section ? i : i - section;
    const float angle =
        static_cast<float>(position) * powf(10000.0f, -2.0f * index / half);
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float q0 = q[i];
    const float q1 = q[i + half];
    q[i] = q0 * c - q1 * s;
    q[i + half] = q0 * s + q1 * c;
    const float k0 = k[i];
    const float k1 = k[i + half];
    k[i] = k0 * c - k1 * s;
    k[i + half] = k0 * s + k1 * c;
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

/// x[t][d] += pos[(t/grid)*grid + (t%grid)][d], i.e. the top-left rows x cols block of
/// the square position grid. Production images are 768x768, whose 48x48 patch grid
/// matches the stored grid exactly; smaller grids are a test-only crop.
__global__ void AddPosEmbKernel(float* x, const float* pos, int tokens, int cols,
                                int grid, int dim) {
  const int t = blockIdx.x;
  if (t >= tokens) {
    return;
  }
  // Row-major crop: row t/cols, column t%cols of the stored square grid.
  const int pos_index = (t / cols) * grid + (t % cols);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    x[static_cast<std::size_t>(t) * dim + d] +=
        pos[static_cast<std::size_t>(pos_index) * dim + d];
  }
}

/// Gathers 2x2 patch neighbourhoods into merged rows: out[(mh*gw/2 + mw)][4*dim] with
/// the four patches in (row, row+1) x (col, col+1) order.
__global__ void Merge2x2Kernel(const float* x, float* out, int rows, int cols, int dim) {
  const int merged_cols = cols / 2;
  const int merged_rows = rows / 2;
  const int total = merged_rows * merged_cols;
  const int m = blockIdx.x;
  if (m >= total) {
    return;
  }
  const int mh = m / merged_cols;
  const int mw = m % merged_cols;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        const int source_row = mh * 2 + dy;
        const int source_col = mw * 2 + dx;
        const std::size_t source =
            (static_cast<std::size_t>(source_row) * cols + source_col) * dim + d;
        out[static_cast<std::size_t>(m) * 4 * dim + (dy * 2 + dx) * dim + d] = x[source];
      }
    }
  }
}

}  // namespace

void LaunchAddPosEmb(float* x, const float* pos, std::size_t tokens, std::size_t cols,
                     std::size_t grid, std::size_t dim, hipStream_t stream) {
  AddPosEmbKernel<<<static_cast<int>(tokens), 128, 0, stream>>>(
      x, pos, static_cast<int>(tokens), static_cast<int>(cols),
      static_cast<int>(grid), static_cast<int>(dim));
}

void LaunchMerge2x2(const float* x, float* out, std::size_t rows, std::size_t cols,
                    std::size_t dim, hipStream_t stream) {
  const int blocks = static_cast<int>((rows / 2) * (cols / 2));
  Merge2x2Kernel<<<blocks, 128, 0, stream>>>(x, out, static_cast<int>(rows),
                                             static_cast<int>(cols),
                                             static_cast<int>(dim));
}

void LaunchMatvecOutMajor(const float* x, const float* w, const float* bias, float* y,
                         std::size_t in, std::size_t out, hipStream_t stream) {
  const int blocks = static_cast<int>((out + kThreads - 1) / kThreads);
  MatvecInMajorKernel<<<blocks, kThreads, 0, stream>>>(x, w, bias, y,
                                                      static_cast<int>(in),
                                                      static_cast<int>(out));
}

void LaunchMatvecOutMajorBatch(const float* x, const float* w, const float* bias,
                               float* y, std::size_t tokens, std::size_t in,
                               std::size_t out, hipStream_t stream) {
  const int blocks = static_cast<int>((out + kThreads - 1) / kThreads);
  MatvecOutMajorBatchKernel<<<blocks, kThreads, 0, stream>>>(
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

void LaunchVisionRope(float* qkv, std::size_t tokens, std::size_t heads,
                      std::size_t head_dim, std::size_t cols, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));
  VisionRopeKernel<<<grid, 64, 0, stream>>>(qkv, static_cast<int>(tokens),
                                            static_cast<int>(heads),
                                            static_cast<int>(head_dim),
                                            static_cast<int>(cols));
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
