#include "src/hip/moe.hpp"

#include "src/hip/layers.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::ling3_air::hip {
namespace {

constexpr int kMaxExperts = 512;
constexpr int kThreads = 256;

__global__ void MulKernel(float* out, const float* a, const float* b, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = a[i] * b[i];
  }
}

/// Routing runs on one thread: 512 scores, a handful of groups. Making the
/// order explicit keeps tie-breaking identical to the reference.
__global__ void RouterKernel(const float* logits, const float* bias, int n_expert,
                            int n_group, int group_used, int top_k, float scale,
                            int* ids, float* weights) {
  __shared__ float scores[kMaxExperts];
  __shared__ unsigned char used[kMaxExperts];
  if (blockIdx.x != 0) {
    return;
  }
  for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
    scores[e] = 1.0f / (1.0f + expf(-(logits[e] + (bias == nullptr ? 0.0f : bias[e]))));
  }
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }

  const int per_group = n_expert / n_group;
  float group_score[16];
  for (int g = 0; g < n_group; ++g) {
    float first = -1.0f;
    float second = -1.0f;
    for (int k = 0; k < per_group; ++k) {
      const float value = scores[g * per_group + k];
      if (value > first) {
        second = first;
        first = value;
      } else if (value > second) {
        second = value;
      }
    }
    group_score[g] = first + second;
  }

  bool group_kept[16];
  for (int g = 0; g < n_group; ++g) {
    group_kept[g] = false;
  }
  for (int picked = 0; picked < group_used; ++picked) {
    int best = -1;
    float best_value = -1.0f;
    for (int g = 0; g < n_group; ++g) {
      if (!group_kept[g] && group_score[g] > best_value) {
        best_value = group_score[g];
        best = g;
      }
    }
    if (best < 0) {
      break;
    }
    group_kept[best] = true;
  }

  // Single pass with a small ordered insert: 512 comparisons instead of
  // top_k passes over 512 experts (the old 8-pass scan cost 11% of a token).
  int best_ids[16];
  float best_scores[16];
  for (int i = 0; i < top_k; ++i) {
    best_ids[i] = -1;
    best_scores[i] = -1.0f;
  }
  for (int e = 0; e < n_expert; ++e) {
    if (!group_kept[e / per_group]) {
      continue;
    }
    const float value = scores[e];
    if (value <= best_scores[top_k - 1]) {
      continue;
    }
    int position = top_k - 1;
    while (position > 0 && best_scores[position - 1] < value) {
      best_scores[position] = best_scores[position - 1];
      best_ids[position] = best_ids[position - 1];
      --position;
    }
    best_scores[position] = value;
    best_ids[position] = e;
  }
  for (int i = 0; i < top_k; ++i) {
    ids[i] = best_ids[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < top_k; ++i) {
    sum += scores[ids[i]];
  }
  const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (int i = 0; i < top_k; ++i) {
    weights[i] = scores[ids[i]] * inverse * scale;
  }
}

}  // namespace

void LaunchRouterSelect(const float* logits, const float* bias,
                        std::size_t n_expert, std::size_t n_group,
                        std::size_t group_used, std::size_t top_k, float scale,
                        std::int32_t* ids, float* weights, hipStream_t stream) {
  RouterKernel<<<1, kThreads, 0, stream>>>(
      logits, bias, static_cast<int>(n_expert), static_cast<int>(n_group),
      static_cast<int>(group_used), static_cast<int>(top_k), scale,
      reinterpret_cast<int*>(ids), weights);
}

void LaunchFfn(const void* gate, StorageType gate_type, const void* up,
               StorageType up_type, const void* down, StorageType down_type,
               std::size_t n_embd, std::size_t n_ff, const float* x,
               float weight, float* out, float* expert_gate, float* expert_up,
               float* expert_down, hipStream_t stream) {
  LaunchGemv(gate, gate_type, n_ff, n_embd, x, expert_gate, stream);
  LaunchGemv(up, up_type, n_ff, n_embd, x, expert_up, stream);
  LaunchSilu(expert_gate, expert_gate, n_ff, stream);
  const int blocks = static_cast<int>((n_ff + kThreads - 1) / kThreads);
  MulKernel<<<blocks, kThreads, 0, stream>>>(expert_gate, expert_gate, expert_up,
                                            static_cast<int>(n_ff));
  LaunchGemv(down, down_type, n_embd, n_ff, expert_gate, expert_down, stream);
  LaunchAxpy(out, expert_down, weight, n_embd, stream);
}

void LaunchMoeGather(const void* gate_exps, StorageType gate_type,
                     const void* up_exps, StorageType up_type,
                     const void* down_exps, StorageType down_type,
                     std::size_t n_embd, std::size_t n_ff, const float* x,
                     const std::int32_t* ids, const float* weights,
                     std::size_t n_selected, float* out, float* expert_gate,
                     float* expert_up, float* expert_down, hipStream_t stream) {
  const std::size_t gate_row = RowBytes(gate_type, n_embd);
  const std::size_t up_row = RowBytes(up_type, n_embd);
  const std::size_t down_row = RowBytes(down_type, n_ff);
  // Expert ids live on the device; copy them once per step.
  std::int32_t host_ids[64];
  float host_weights[64];
  const std::size_t count = n_selected < 64 ? n_selected : 64;
  hipMemcpyAsync(host_ids, ids, count * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, stream);
  hipMemcpyAsync(host_weights, weights, count * sizeof(float),
                 hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t expert = static_cast<std::size_t>(host_ids[i]);
    LaunchFfn(static_cast<const char*>(gate_exps) + expert * n_ff * gate_row,
              gate_type, static_cast<const char*>(up_exps) + expert * n_ff * up_row,
              up_type, static_cast<const char*>(down_exps) + expert * n_embd * down_row,
              down_type, n_embd, n_ff, x, host_weights[i], out, expert_gate,
              expert_up, expert_down, stream);
  }
}

}  // namespace gufo::ext::ling3_air::hip
