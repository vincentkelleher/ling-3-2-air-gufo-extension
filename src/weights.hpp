// The Ling 3.2 Air weight map: every tensor the engine reads, validated
// against the GGUF before anything is uploaded to the GPU.
//
// Validation is strict on purpose. A name or shape this map does not expect is
// an error, not a skipped tensor: a model that silently drops a weight decodes
// plausible-looking garbage.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/config.hpp"

namespace gufo::ext::ling3_air {

/// One mapped tensor: where it lives, how big it is and how it is stored.
struct Tensor {
  const void* data{nullptr};
  std::size_t bytes{0};
  core::GgmlType type{core::GgmlType::kF32};
  std::array<std::uint64_t, 4> ne{1, 1, 1, 1};
  std::uint32_t ndims{0};

  [[nodiscard]] bool Valid() const { return data != nullptr; }
  [[nodiscard]] std::uint64_t Rows() const { return ndims > 1 ? ne[1] : 1; }
  [[nodiscard]] std::uint64_t Cols() const { return ndims > 0 ? ne[0] : 0; }
};

/// Per-block weights. A block fills either the KDA or the MLA fields, then the
/// dense or the MoE fields; the MTP block additionally fills `nextn`.
struct LayerWeights {
  LayerKind kind{LayerKind::kKda};
  bool dense{false};

  Tensor attn_norm;
  Tensor ffn_norm;

  // KDA
  Tensor kda_q, kda_k, kda_v;
  Tensor kda_conv_q, kda_conv_k, kda_conv_v;
  Tensor kda_f_a, kda_beta, kda_a, kda_dt, kda_g_a, kda_o_norm, kda_out;

  // MLA
  Tensor mla_q, mla_kv_a, mla_kv_a_norm, mla_k_b, mla_v_b, mla_gate, mla_out;

  // FFN
  Tensor ffn_gate, ffn_up, ffn_down;              // dense blocks
  Tensor gate_inp, exp_probs_b;                   // router
  Tensor gate_exps, up_exps, down_exps;           // routed experts
  Tensor gate_shexp, up_shexp, down_shexp;        // shared expert

  // MTP block only
  Tensor nextn_eh_proj, nextn_enorm, nextn_hnorm, nextn_shared_head_norm;
};

struct Ling3Weights {
  Tensor token_embd;
  Tensor output_norm;
  Tensor output;
  std::vector<LayerWeights> layers;
  std::size_t total_bytes{0};
  std::size_t tensor_count{0};

  /// Bytes per storage type, for the loader's diagnostics.
  std::vector<std::pair<core::GgmlType, std::size_t>> bytes_by_type;
};

/// Maps and validates every tensor. Returns false and sets `*error` on the
/// first missing tensor, unexpected shape, unsupported storage type, or
/// unmapped `blk.*` tensor.
[[nodiscard]] bool LoadWeights(const core::GgufReader& reader,
                               const Ling3Config& config, Ling3Weights* weights,
                               std::string* error);

/// One-line-per-type byte summary, for logs and the config test.
[[nodiscard]] std::string DescribeWeights(const Ling3Weights& weights);

}  // namespace gufo::ext::ling3_air
