#include "src/hip/layers.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::ling3_air::hip {
namespace {

constexpr int kThreads = 256;

__global__ void Conv1dStepKernel(const float* x, const float* weight,
                                float* state, float* out, int channels, int slot, float* debug) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  float* slice = state + static_cast<std::size_t>(slot) * 3 * channels;
  const float window[4] = {slice[c], slice[channels + c], slice[2 * channels + c],
                           x[c]};
  // The GGUF stores ssm_conv1d_*.weight as [kernel, 1, channels]: the kernel
  // index is the fastest-varying one, so channel c owns the four consecutive
  // weights weight[c * 4 + k].
  float sum = 0.0f;
  for (int k = 0; k < 4; ++k) {
    sum += weight[c * 4 + k] * window[k];
  }
  if (debug != nullptr && c < 4) {
    for (int k = 0; k < 4; ++k) debug[c * 8 + k] = window[k];
    for (int k = 0; k < 4; ++k) debug[c * 8 + 4 + k] = weight[c * 4 + k];
    debug[c * 8 + 8 - 1] = static_cast<float>(slot);
  }
  out[c] = sum;
  slice[c] = window[1];
  slice[channels + c] = window[2];
  slice[2 * channels + c] = window[3];
}

__global__ void L2NormKernel(const float* x, float* y, int heads, int head_dim,
                             float eps) {
  const int head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  const float* source = x + static_cast<std::size_t>(head) * head_dim;
  float* target = y + static_cast<std::size_t>(head) * head_dim;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sum += source[i] * source[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  __shared__ float warp_sums[32];
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>((blockDim.x + 31) / 32); ++w) {
    total += warp_sums[w];
  }
  const float norm = sqrtf(total);
  const float scale = 1.0f / (norm < eps ? eps : norm);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    target[i] = source[i] * scale;
  }
}

/// Rotates one pair of dimensions of one head.
__device__ __forceinline__ void RotatePair(float* base, int i, int j, float angle) {
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float a = base[i];
  const float b = base[j];
  base[i] = a * cosine - b * sine;
  base[j] = a * sine + b * cosine;
}

__global__ void RopeKernel(float* x, int heads, int head_dim, int n_rot,
                           int position, float freq_base, int neox) {
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int pairs = n_rot / 2;
  if (head >= heads || pair >= pairs) {
    return;
  }
  float* base = x + static_cast<std::size_t>(head) * head_dim;
  // Same frequency layout as llama.cpp: inverse frequency over the rotary
  // dimensions, then multiplied by the position.
  const float inv_freq = powf(freq_base, -2.0f * static_cast<float>(pair) /
                                               static_cast<float>(n_rot));
  const float angle = static_cast<float>(position) * inv_freq;
  if (neox != 0) {
    RotatePair(base, pair, pair + pairs, angle);
  } else {
    RotatePair(base, 2 * pair, 2 * pair + 1, angle);
  }
}

/// One block per head; thread `j` owns row `j` of the transposed state, so no
/// two threads touch the same floats.
__global__ void DeltaRuleKernel(const float* q, const float* k, const float* v,
                                const float* g, const float* beta, float* state,
                                float* out, int heads, int head_dim,
                                float* debug) {
  const int head = blockIdx.x;
  if (head >= heads || threadIdx.x >= head_dim) {
    return;
  }
  const int j = threadIdx.x;
  extern __shared__ float shared[];  // exp(g) and k for the block
  float* decay = shared;
  float* k_shared = decay + head_dim;

  if (j < head_dim) {
    decay[j] = expf(g[head * head_dim + j]);
    k_shared[j] = k[head * head_dim + j];
  }
  __syncthreads();

  float* row = state + (static_cast<std::size_t>(head) * head_dim + j) * head_dim;
  if (debug != nullptr && head == 0 && j == 0) {
    for (int i = 0; i < head_dim; ++i) debug[i] = row[i];                 // pre
    for (int i = 0; i < head_dim; ++i) debug[head_dim + i] = decay[i];    // decay
  }
  for (int i = 0; i < head_dim; ++i) {
    row[i] *= decay[i];
  }
  if (debug != nullptr && head == 0 && j == 0) {
    for (int i = 0; i < head_dim; ++i) debug[2 * head_dim + i] = row[i];  // post
  }
  float dot = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    dot += row[i] * k_shared[i];
  }
  if (debug != nullptr && head == 0 && j == 0) {
    for (int i = 0; i < head_dim; ++i) debug[3 * head_dim + i] = v[i];
    debug[4 * head_dim] = dot;
    debug[4 * head_dim + 1] = beta[head];
  }
  const float delta = (v[head * head_dim + j] - dot) * beta[head];
  for (int i = 0; i < head_dim; ++i) {
    row[i] += k_shared[i] * delta;
  }
  const float* q_head = q + static_cast<std::size_t>(head) * head_dim;
  float sum = 0.0f;
  for (int i = 0; i < head_dim; ++i) {
    sum += row[i] * q_head[i];
  }
  const float result = sum / sqrtf(static_cast<float>(head_dim));
  out[head * head_dim + j] = result;
  if (debug != nullptr && head == 0) {
    debug[7 * head_dim + j] = result;  // what this launch intends to write
  }
}

__global__ void ArgmaxKernel(const float* x, int count, std::int32_t* result) {
  // Phase 1: each thread's best over its stride, reduced within the warp and
  // then across warps. Ties resolve to the lower index (deterministic).
  __shared__ float best_value[8];
  __shared__ int best_index[8];
  float local_value = -INFINITY;
  int local_index = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float value = x[i];
    if (value > local_value) {
      local_value = value;
      local_index = i;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down(local_value, offset);
    const int other_index = __shfl_down(local_index, offset);
    if (other > local_value) {
      local_value = other;
      local_index = other_index;
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    best_value[warp] = local_value;
    best_index[warp] = local_index;
  }
  __syncthreads();
  const int warps = static_cast<int>(blockDim.x) / 32;
  if (threadIdx.x == 0) {
    int winner = 0;
    for (int w = 1; w < warps; ++w) {
      if (best_value[w] > best_value[winner]) {
        winner = w;
      }
    }
    *result = best_index[winner];
  }
}

__global__ void MulIntoKernel(const float* a, const float* b, float* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = a[i] * b[i];
  }
}

__global__ void AxpyKernel(float* y, const float* z, float scale, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    y[i] += scale * z[i];
  }
}

}  // namespace

void LaunchConv1dStep(const float* x, const float* weight, float* state,
                      std::size_t channels, std::size_t slot, float* out,
                      hipStream_t stream, float* debug) {
  const int blocks = static_cast<int>((channels + kThreads - 1) / kThreads);
  Conv1dStepKernel<<<blocks, kThreads, 0, stream>>>(
      x, weight, state, out, static_cast<int>(channels), static_cast<int>(slot),
      debug);
}

void LaunchL2Norm(const float* x, float* y, std::size_t heads,
                  std::size_t head_dim, float eps, hipStream_t stream) {
  L2NormKernel<<<static_cast<int>(heads), 128, 0, stream>>>(
      x, y, static_cast<int>(heads), static_cast<int>(head_dim), eps);
}

void LaunchRope(float* x, std::size_t heads, std::size_t head_dim,
                std::size_t n_rot, std::size_t position, float freq_base,
                bool neox, hipStream_t stream) {
  const int pairs = static_cast<int>(n_rot / 2);
  RopeKernel<<<static_cast<int>(heads), pairs, 0, stream>>>(
      x, static_cast<int>(heads), static_cast<int>(head_dim),
      static_cast<int>(n_rot), static_cast<int>(position), freq_base,
      neox ? 1 : 0);
}

void LaunchDeltaRuleStep(const float* q, const float* k, const float* v,
                         const float* g, const float* beta, float* state,
                         float* out, std::size_t heads, std::size_t head_dim,
                         hipStream_t stream, float* debug) {
  const std::size_t shared = 2 * head_dim * sizeof(float);
  DeltaRuleKernel<<<static_cast<int>(heads), static_cast<int>(head_dim),
                    shared, stream>>>(q, k, v, g, beta, state, out,
                                      static_cast<int>(heads),
                                      static_cast<int>(head_dim), debug);
}

void LaunchArgmax(const float* x, std::size_t count, std::int32_t* result,
                  hipStream_t stream) {
  ArgmaxKernel<<<1, kThreads, 0, stream>>>(x, static_cast<int>(count), result);
}

void LaunchMulInto(const float* a, const float* b, float* out, std::size_t count,
                   hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  MulIntoKernel<<<blocks, kThreads, 0, stream>>>(a, b, out, static_cast<int>(count));
}

void LaunchAxpy(float* y, const float* z, float scale, std::size_t count,
                hipStream_t stream) {
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  AxpyKernel<<<blocks, kThreads, 0, stream>>>(y, z, scale, count);
}

}  // namespace gufo::ext::ling3_air::hip
