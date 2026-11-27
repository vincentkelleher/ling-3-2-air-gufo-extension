#include "src/hip/attention.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::ext::ling3_air::hip {
namespace {

constexpr int kThreads = 128;

__global__ void AbsorbQKernel(const float* q_nope, const float* wk_b, float* q_abs,
                              int heads) {
  const int head = blockIdx.x;
  const int i = threadIdx.x;
  if (head >= heads || i >= static_cast<int>(kMlaLatent)) {
    return;
  }
  const float* q = q_nope + static_cast<std::size_t>(head) * kMlaNope;
  const float* row =
      wk_b + (static_cast<std::size_t>(head) * kMlaLatent + i) * kMlaNope;
  float sum = 0.0f;
  for (std::size_t d = 0; d < kMlaNope; ++d) {
    sum += q[d] * row[d];
  }
  q_abs[static_cast<std::size_t>(head) * kMlaLatent + i] = sum;
}

__global__ void ScoresKernel(const float* q_abs, const float* q_pe,
                             const float* cache, int heads, int cache_len,
                             float* scores) {
  const int head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  const float* qa = q_abs + static_cast<std::size_t>(head) * kMlaLatent;
  const float* qr = q_pe + static_cast<std::size_t>(head) * kMlaRope;
  for (int p = threadIdx.x; p < cache_len; p += blockDim.x) {
    const float* token = cache + static_cast<std::size_t>(p) * kMlaCacheStride;
    float sum = 0.0f;
    for (std::size_t i = 0; i < kMlaLatent; ++i) {
      sum += qa[i] * token[i];
    }
    for (std::size_t j = 0; j < kMlaRope; ++j) {
      sum += qr[j] * token[kMlaLatent + j];
    }
    scores[static_cast<std::size_t>(head) * cache_len + p] = sum * kMlaScoreScale;
  }
}

__global__ void SoftmaxKernel(float* scores, int heads, int cache_len) {
  const int head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  float* row = scores + static_cast<std::size_t>(head) * cache_len;
  __shared__ float warp_buf[32];
  __shared__ float shared_max;
  __shared__ float shared_sum;
  const int warps = static_cast<int>((blockDim.x + 31) / 32);

  // Reduce to one scalar per warp, then to one scalar for the block, and *broadcast* it:
  // after the shuffles only lane 0 holds the warp result, so every lane must read the
  // block-wide value from shared memory. Not doing so made each lane subtract its own
  // score as the max, which collapses a two-position softmax to exactly 0.5/0.5.
  float max_value = -INFINITY;
  for (int p = threadIdx.x; p < cache_len; p += blockDim.x) {
    max_value = fmaxf(max_value, row[p]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_value = fmaxf(max_value, __shfl_down(max_value, offset));
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    warp_buf[threadIdx.x >> 5] = max_value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float best = warp_buf[0];
    for (int w = 1; w < warps; ++w) {
      best = fmaxf(best, warp_buf[w]);
    }
    shared_max = best;
  }
  __syncthreads();
  max_value = shared_max;

  float sum = 0.0f;
  for (int p = threadIdx.x; p < cache_len; p += blockDim.x) {
    sum += expf(row[p] - max_value);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    warp_buf[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) {
      total += warp_buf[w];
    }
    shared_sum = total;
  }
  __syncthreads();

  for (int p = threadIdx.x; p < cache_len; p += blockDim.x) {
    row[p] = expf(row[p] - max_value) / shared_sum;
  }
}

__global__ void LatentSumKernel(const float* probs, const float* cache,
                                int heads, int cache_len, float* latent) {
  const int head = blockIdx.x;
  const int i = threadIdx.x;
  if (head >= heads || i >= static_cast<int>(kMlaLatent)) {
    return;
  }
  const float* row = probs + static_cast<std::size_t>(head) * cache_len;
  float sum = 0.0f;
  for (int p = 0; p < cache_len; ++p) {
    sum += row[p] * cache[static_cast<std::size_t>(p) * kMlaCacheStride + i];
  }
  latent[static_cast<std::size_t>(head) * kMlaLatent + i] = sum;
}

__global__ void ValueAbsorbKernel(const float* latent, const float* wv_b,
                                  const float* gate, float* out, int heads) {
  const int head = blockIdx.x;
  const int o = threadIdx.x;
  if (head >= heads || o >= static_cast<int>(kMlaV)) {
    return;
  }
  const float* src = latent + static_cast<std::size_t>(head) * kMlaLatent;
  const float* row = wv_b + (static_cast<std::size_t>(head) * kMlaV + o) * kMlaLatent;
  float sum = 0.0f;
  for (std::size_t i = 0; i < kMlaLatent; ++i) {
    sum += src[i] * row[i];
  }
  out[static_cast<std::size_t>(head) * kMlaV + o] =
      sum * (1.0f / (1.0f + expf(-gate[head])));
}

}  // namespace

void LaunchMlaAbsorbQ(const float* q_nope, const float* wk_b, float* q_abs,
                      std::size_t heads, hipStream_t stream) {
  AbsorbQKernel<<<static_cast<int>(heads), static_cast<int>(kMlaLatent), 0, stream>>>(
      q_nope, wk_b, q_abs, static_cast<int>(heads));
}

void LaunchMlaScores(const float* q_abs, const float* q_pe, const float* cache,
                     std::size_t heads, std::size_t cache_len, float* scores,
                     hipStream_t stream) {
  ScoresKernel<<<static_cast<int>(heads), kThreads, 0, stream>>>(
      q_abs, q_pe, cache, static_cast<int>(heads), static_cast<int>(cache_len),
      scores);
}

void LaunchMlaSoftmax(float* scores, std::size_t heads, std::size_t cache_len,
                      hipStream_t stream) {
  SoftmaxKernel<<<static_cast<int>(heads), kThreads, 0, stream>>>(
      scores, static_cast<int>(heads), static_cast<int>(cache_len));
}

void LaunchMlaLatentSum(const float* probs, const float* cache,
                        std::size_t heads, std::size_t cache_len,
                        float* latent, hipStream_t stream) {
  LatentSumKernel<<<static_cast<int>(heads), static_cast<int>(kMlaLatent), 0,
                    stream>>>(probs, cache, static_cast<int>(heads),
                              static_cast<int>(cache_len), latent);
}

void LaunchMlaValueAbsorb(const float* latent, const float* wv_b,
                          const float* gate, float* out, std::size_t heads,
                          hipStream_t stream) {
  ValueAbsorbKernel<<<static_cast<int>(heads), static_cast<int>(kMlaV), 0, stream>>>(
      latent, wv_b, gate, out, static_cast<int>(heads));
}

}  // namespace gufo::ext::ling3_air::hip
