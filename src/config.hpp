// Ling 3.2 Air configuration, read from the GGUF metadata.
//
// Every value here is checked against the key llama.cpp writes for the
// `bailingmoe3` architecture; nothing is hard-coded except the invariants this
// extension refuses to guess (see `ParseConfig`).
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::ext::ling3_air {

/// Attention flavour of one trunk block. Ling 3.2 alternates seven MLA blocks
/// (one every six) with KDA linear-attention blocks; the GGUF's per-layer
/// `attention.head_count_kv` array says which is which (0 = KDA).
enum class LayerKind : std::uint8_t { kKda = 0, kMla = 1 };

struct Ling3Config {
  // Trunk
  std::uint32_t n_layer{0};       // transformer blocks (block_count - nextn)
  std::uint32_t n_layer_total{0}; // includes the MTP block
  std::uint32_t n_embd{0};
  std::uint32_t n_ff{0};           // dense FFN width (blocks 0..dense_lead-1)
  std::uint32_t n_ff_exp{0};       // routed expert width
  std::uint32_t n_ff_shexp{0};     // shared expert width
  std::uint32_t n_vocab{0};
  std::uint32_t n_ctx_train{0};
  std::uint32_t n_dense_lead{0};
  float rms_eps{0.0f};

  // Attention, shared by both flavours
  std::uint32_t n_head{0};
  std::uint32_t n_layer_nextn{0};
  std::vector<std::uint32_t> n_head_kv;  // per layer, 0 = KDA
  std::vector<LayerKind> kind;

  // MLA blocks
  std::uint32_t n_head_dim_mla{0};     // qk per head (nope + rope)
  std::uint32_t n_head_dim_v_mla{0};   // v per head
  std::uint32_t n_lora_kv{0};          // KV latent width
  std::uint32_t n_rot{0};              // RoPE dimensions
  std::vector<std::uint32_t> rope_sections;
  float rope_freq_base{0.0f};

  // KDA blocks
  std::uint32_t n_kda_head_dim{0};
  std::uint32_t ssm_conv_kernel{0};
  bool kda_safe_gate{true};
  float kda_gate_lower_bound{0.0f};

  // MoE
  std::uint32_t n_expert{0};
  std::uint32_t n_expert_used{0};
  std::uint32_t n_expert_shared{0};
  std::uint32_t n_group{0};
  std::uint32_t n_group_used{0};
  std::uint32_t expert_gating_func{0};  // 1 = softmax, 2 = sigmoid
  bool expert_weights_norm{false};
  float expert_weights_scale{1.0f};

  // SwiGLU clamps, per layer (0 disables). The GGUF stores one entry per block,
  // including the MTP block.
  std::vector<float> swiglu_clamp_exp;
  std::vector<float> swiglu_clamp_shexp;

  [[nodiscard]] bool IsKda(std::uint32_t layer) const {
    return kind[layer] == LayerKind::kKda;
  }
  [[nodiscard]] bool IsDense(std::uint32_t layer) const {
    return layer < n_dense_lead;
  }
  [[nodiscard]] std::uint32_t NopeDim() const {
    return n_head_dim_mla - n_rot;
  }
  [[nodiscard]] std::uint32_t KdaInner() const {
    return n_head * n_kda_head_dim;
  }
  [[nodiscard]] std::size_t KdaCount() const;
  [[nodiscard]] std::size_t MlaCount() const;
};

/// Parses and validates the metadata. Returns false and sets `*error` when a
/// key is missing, out of range, or describes a checkpoint this extension does
/// not implement (for example a different Ling size).
[[nodiscard]] bool ParseConfig(const core::GgufReader& reader,
                               Ling3Config* config, std::string* error);

/// Short human-readable summary, one line per section (for `--verbose` and
/// tests).
[[nodiscard]] std::string DescribeConfig(const Ling3Config& config);

}  // namespace gufo::ext::ling3_air
