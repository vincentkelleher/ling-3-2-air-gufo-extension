#include "src/hip/moe.hpp"

#include "src/hip/layers.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::ling3_air::hip {
namespace {

constexpr int kMaxExperts = 512;
constexpr int kThreads = 256;

/// Counts and scatters (token slot) pairs into per-expert buckets.
__global__ void BuildExpertListsKernel(const std::int32_t* ids, int pairs,
                                       int top_k, int n_expert,
                                       std::int32_t* counts, std::int32_t* cursor,
                                       const std::int32_t* offsets,
                                       std::int32_t* token_slots, int scatter) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pairs) {
    return;
  }
  const int token = p / top_k;
  const int expert = ids[p];
  if (expert < 0 || expert >= n_expert) {
    return;
  }
  if (scatter == 0) {
    atomicAdd(&counts[expert], 1);
  } else {
    const int slot = atomicAdd(&cursor[expert], 1);
    token_slots[offsets[expert] + slot] = p;  // pair index: token*top_k + k
  }
}

__global__ void GatherRowsKernel(const float* src, const std::int32_t* indices,
                                int count, int width, float* dst, int divisor) {
  const int row = blockIdx.y;
  if (row >= count) {
    return;
  }
  const int from = divisor > 1 ? indices[row] / divisor : indices[row];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.x) {
    dst[static_cast<std::size_t>(row) * width + i] =
        src[static_cast<std::size_t>(from) * width + i];
  }
}

__global__ void ScatterAddKernel(float* dst, const std::int32_t* indices,
                                 const float* weight_table, int count, int width,
                                 const float* src, int divisor) {
  const int row = blockIdx.y;
  if (row >= count) {
    return;
  }
  const int to = divisor > 1 ? indices[row] / divisor : indices[row];
  const float weight = divisor > 1 ? weight_table[indices[row]] : weight_table[row];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += blockDim.x * gridDim.x) {
    atomicAdd(&dst[static_cast<std::size_t>(to) * width + i],
              weight * src[static_cast<std::size_t>(row) * width + i]);
  }
}

__global__ void MoeActivateKernel(const float* gate, const float* up, float* act,
                                  int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = gate[i];
    act[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}

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

void LaunchBuildExpertLists(const std::int32_t* ids, std::size_t tokens,
                            std::size_t top_k, std::size_t n_expert,
                            std::int32_t* counts, std::int32_t* cursor,
                            std::int32_t* offsets, std::int32_t* token_slots,
                            hipStream_t stream,
                            std::vector<std::int32_t>* host_counts,
                            std::vector<std::int32_t>* host_offsets) {
  const int pairs = static_cast<int>(tokens * top_k);
  const int blocks = (pairs + kThreads - 1) / kThreads;

  hipMemsetAsync(counts, 0, n_expert * sizeof(std::int32_t), stream);
  BuildExpertListsKernel<<<blocks, kThreads, 0, stream>>>(
      ids, pairs, static_cast<int>(top_k), static_cast<int>(n_expert), counts,
      cursor, offsets, token_slots, /*scatter=*/0);

  // Prefix sum on the host: 512 entries, and it keeps the order deterministic.
  std::vector<std::int32_t> host_counts_out(n_expert);
  hipMemcpyAsync(host_counts_out.data(), counts, n_expert * sizeof(std::int32_t),
                 hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
  std::vector<std::int32_t> offsets_out(n_expert);
  std::int32_t running = 0;
  for (std::size_t e = 0; e < n_expert; ++e) {
    offsets_out[e] = running;
    running += host_counts_out[e];
  }
  if (host_counts != nullptr) *host_counts = host_counts_out;
  if (host_offsets != nullptr) *host_offsets = offsets_out;
  hipMemcpyAsync(offsets, offsets_out.data(), n_expert * sizeof(std::int32_t),
                 hipMemcpyHostToDevice, stream);
  hipMemsetAsync(cursor, 0, n_expert * sizeof(std::int32_t), stream);
  BuildExpertListsKernel<<<blocks, kThreads, 0, stream>>>(
      ids, pairs, static_cast<int>(top_k), static_cast<int>(n_expert), counts,
      cursor, offsets, token_slots, /*scatter=*/1);
}

void LaunchGatherRows(const float* src, const std::int32_t* indices,
                      std::size_t count, std::size_t width, float* dst,
                      hipStream_t stream, std::size_t divisor) {
  if (count == 0) return;
  const dim3 grid((static_cast<unsigned>(width) + kThreads - 1) / kThreads,
                  static_cast<unsigned>(count));
  GatherRowsKernel<<<grid, kThreads, 0, stream>>>(
      src, indices, static_cast<int>(count), static_cast<int>(width), dst,
      static_cast<int>(divisor));
}

void LaunchScatterAdd(float* dst, const std::int32_t* indices,
                      const float* weight_table, std::size_t count,
                      std::size_t width, const float* src, hipStream_t stream,
                      std::size_t divisor) {
  if (count == 0) return;
  const dim3 grid((static_cast<unsigned>(width) + kThreads - 1) / kThreads,
                  static_cast<unsigned>(count));
  ScatterAddKernel<<<grid, kThreads, 0, stream>>>(
      dst, indices, weight_table, static_cast<int>(count), static_cast<int>(width),
      src, static_cast<int>(divisor));
}

/// out[i] += sum over slots of weights[slot] * down[slot][i]: the combine step of
/// the routed MoE, in the same slot order the host loop used.
__global__ void MoeCombineKernel(const float* down, const float* weights, int experts,
                                 int width, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) {
    return;
  }
  float sum = out[i];
  for (int slot = 0; slot < experts; ++slot) {
    sum += weights[slot] * down[static_cast<std::size_t>(slot) * width + i];
  }
  out[i] = sum;
}

void LaunchMoeGather(const void* gate_exps, StorageType gate_type,
                     const void* up_exps, StorageType up_type,
                     const void* down_exps, StorageType down_type,
                     std::size_t n_embd, std::size_t n_ff, const float* x,
                     const std::int32_t* ids, const float* weights,
                     std::size_t n_selected, float* out, float* expert_gate,
                     float* expert_up, float* expert_down, hipStream_t stream) {
  // Every selected expert is covered by one launch per matrix, with the ids left in
  // device memory. The previous version copied ids and weights to the host and
  // synchronized the stream on every token, then launched the expert FFNs from a host
  // loop -- 35 synchronizations per token, which is why decode and prefill ran at the
  // same speed.
  LaunchGemvExperts(gate_exps, gate_type, ids, n_selected, n_ff, n_embd, x,
                    /*batch_stride=*/0, expert_gate, /*out_stride=*/n_ff, stream);
  LaunchGemvExperts(up_exps, up_type, ids, n_selected, n_ff, n_embd, x,
                    /*batch_stride=*/0, expert_up, /*out_stride=*/n_ff, stream);
  LaunchSilu(expert_gate, expert_gate, n_selected * n_ff, stream);
  LaunchMulInto(expert_gate, expert_up, expert_gate, n_selected * n_ff, stream);
  LaunchGemvExperts(down_exps, down_type, ids, n_selected, n_embd, n_ff, expert_gate,
                    /*batch_stride=*/n_ff, expert_down, /*out_stride=*/n_embd, stream);
  const int blocks = static_cast<int>((n_embd + kThreads - 1) / kThreads);
  MoeCombineKernel<<<blocks, kThreads, 0, stream>>>(
      expert_down, weights, static_cast<int>(n_selected), static_cast<int>(n_embd), out);
}

}  // namespace gufo::ext::ling3_air::hip
