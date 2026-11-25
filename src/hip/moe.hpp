// MoE router and expert FFN kernels for the Ling 3.2 Air blocks.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "src/hip/kernels.hpp"

namespace gufo::ext::ling3_air::hip {

/// Router for one token: `scores = sigmoid(logits + bias)`, group-limited
/// top-k (each group's score is its two best experts, the best `group_used`
/// groups are kept), then the best `top_k` experts inside them. The selected
/// weights are normalised to sum 1 and multiplied by `scale`. Ties resolve to
/// the lower expert index, as the reference does.
/// `ids` and `weights` hold `top_k` entries.
void LaunchRouterSelect(const float* logits, const float* bias,
                        std::size_t n_expert, std::size_t n_group,
                        std::size_t group_used, std::size_t top_k, float scale,
                        std::int32_t* ids, float* weights, hipStream_t stream);

/// One SwiGLU FFN over `n_embd -> n_ff -> n_embd`, added into `out` with
/// `weight`: `out += weight * down(silu(gate(x)) * up(x))`.
/// `expert_gate`/`expert_up`/`expert_act` are `n_ff` floats of scratch,
/// `expert_down` is `n_embd`.
void LaunchFfn(const void* gate, StorageType gate_type, const void* up,
               StorageType up_type, const void* down, StorageType down_type,
               std::size_t n_embd, std::size_t n_ff, const float* x,
               float weight, float* out, float* expert_gate, float* expert_up,
               float* expert_down, hipStream_t stream);

/// Sums the selected experts: `out += sum_i weights[i] * ffn_i(x)`.
/// Expert `i`'s matrices start at `i * n_ff` rows (gate/up) and `i * n_embd`
/// rows (down), which is the GGUF layout `[in, out, n_expert]`.
void LaunchMoeGather(const void* gate_exps, StorageType gate_type,
                     const void* up_exps, StorageType up_type,
                     const void* down_exps, StorageType down_type,
                     std::size_t n_embd, std::size_t n_ff, const float* x,
                     const std::int32_t* ids, const float* weights,
                     std::size_t n_selected, float* out, float* expert_gate,
                     float* expert_up, float* expert_down, hipStream_t stream);

}  // namespace gufo::ext::ling3_air::hip
