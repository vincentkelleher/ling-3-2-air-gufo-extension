#include "src/config.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <variant>

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::ext::ling3_air {
namespace {

class Reader {
 public:
  Reader(const core::GgufReader& gguf, std::string architecture)
      : gguf_(gguf), prefix_(std::move(architecture)) {}

  void set_error(std::string* error) { error_ = error; }

  [[nodiscard]] std::string Key(std::string_view suffix) const {
    std::string key = prefix_;
    key.push_back('.');
    key.append(suffix);
    return key;
  }

  [[nodiscard]] bool Fail(std::string message) {
    if (error_ != nullptr) {
      *error_ = std::move(message);
    }
    return false;
  }

  [[nodiscard]] bool U32(std::string_view suffix, std::uint32_t* out,
                         bool required = true) {
    const auto key = Key(suffix);
    if (const auto value = gguf_.GetMetadataUint32(key)) {
      *out = *value;
      return true;
    }
    if (const auto value = gguf_.GetMetadataUint64(key)) {
      if (*value > 0xFFFFFFFFull) {
        return Fail("config: " + key + " does not fit in 32 bits");
      }
      *out = static_cast<std::uint32_t>(*value);
      return true;
    }
    if (!required) {
      return true;
    }
    return Fail("config: missing or non-integer " + key);
  }

  /// A key llama.cpp writes either as a scalar or as a per-layer array; the
  /// extension supports only the uniform case and says so.
  [[nodiscard]] bool U32OrArray(std::string_view suffix, std::uint32_t* out) {
    if (U32(suffix, out, /*required=*/false)) {
      const auto key = Key(suffix);
      if (gguf_.GetMetadataUint32(key) || gguf_.GetMetadataUint64(key)) {
        return true;
      }
    }
    std::vector<std::int64_t> values;
    if (!IntArray(suffix, &values)) {
      return false;
    }
    if (values.empty()) {
      return Fail("config: " + Key(suffix) + " is empty");
    }
    for (const auto value : values) {
      if (value != values.front()) {
        return Fail("config: " + Key(suffix) +
                    " varies per layer, which this extension does not support");
      }
    }
    *out = static_cast<std::uint32_t>(values.front());
    return true;
  }

  [[nodiscard]] bool F32(std::string_view suffix, float* out,
                         bool required = true) {
    const auto key = Key(suffix);
    if (const auto value = gguf_.GetMetadataFloat32(key)) {
      *out = *value;
      return true;
    }
    if (!required) {
      return true;
    }
    return Fail("config: missing or non-float " + key);
  }

  [[nodiscard]] bool Bool(std::string_view suffix, bool* out,
                          bool required = true) {
    const auto key = Key(suffix);
    if (const auto value = gguf_.GetMetadataBool(key)) {
      *out = *value;
      return true;
    }
    if (!required) {
      return true;
    }
    return Fail("config: missing or non-boolean " + key);
  }

  [[nodiscard]] bool IntArray(std::string_view suffix,
                              std::vector<std::int64_t>* out,
                              bool required = true) {
    const auto key = Key(suffix);
    const auto* value = gguf_.FindMetadata(key);
    if (value == nullptr) {
      return required ? Fail("config: missing " + key) : true;
    }
    if (const auto* ints =
            std::get_if<std::vector<std::int64_t>>(&value->value)) {
      *out = *ints;
      return true;
    }
    if (const auto* uints =
            std::get_if<std::vector<std::uint64_t>>(&value->value)) {
      out->assign(uints->begin(), uints->end());
      return true;
    }
    return Fail("config: " + key + " is not an integer array");
  }

  [[nodiscard]] bool FloatArray(std::string_view suffix,
                                std::vector<float>* out,
                                bool required = true) {
    const auto key = Key(suffix);
    const auto* value = gguf_.FindMetadata(key);
    if (value == nullptr) {
      return required ? Fail("config: missing " + key) : true;
    }
    if (const auto* doubles =
            std::get_if<std::vector<double>>(&value->value)) {
      out->assign(doubles->begin(), doubles->end());
      return true;
    }
    return Fail("config: " + key + " is not a float array");
  }

 private:
  const core::GgufReader& gguf_;
  std::string prefix_;
  std::string* error_{nullptr};
};

}  // namespace

std::size_t Ling3Config::KdaCount() const {
  return static_cast<std::size_t>(
      std::count(kind.begin(), kind.end(), LayerKind::kKda));
}

std::size_t Ling3Config::MlaCount() const {
  return static_cast<std::size_t>(
      std::count(kind.begin(), kind.end(), LayerKind::kMla));
}

bool ParseConfig(const core::GgufReader& reader, Ling3Config* config,
                 std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "bailingmoe3") {
    if (error != nullptr) {
      *error = "config: general.architecture is '" + std::string(architecture) +
               "', expected 'bailingmoe3'";
    }
    return false;
  }

  Reader in(reader, std::string(architecture));
  in.set_error(error);
  Ling3Config out;

  bool ok = in.U32("block_count", &out.n_layer_total) &&
            in.U32("nextn_predict_layers", &out.n_layer_nextn, false) &&
            in.U32OrArray("embedding_length", &out.n_embd) &&
            in.U32OrArray("feed_forward_length", &out.n_ff) &&
            in.U32OrArray("expert_feed_forward_length", &out.n_ff_exp) &&
            in.U32OrArray("expert_shared_feed_forward_length", &out.n_ff_shexp) &&
            in.U32("vocab_size", &out.n_vocab) &&
            in.U32("context_length", &out.n_ctx_train) &&
            in.U32("leading_dense_block_count", &out.n_dense_lead) &&
            in.F32("attention.layer_norm_rms_epsilon", &out.rms_eps) &&
            in.U32("attention.head_count", &out.n_head) &&
            in.U32("attention.key_length_mla", &out.n_head_dim_mla) &&
            in.U32("attention.value_length_mla", &out.n_head_dim_v_mla) &&
            in.U32("attention.kv_lora_rank", &out.n_lora_kv) &&
            in.U32("rope.dimension_count", &out.n_rot) &&
            in.F32("rope.freq_base", &out.rope_freq_base) &&
            in.U32("kda.head_dim", &out.n_kda_head_dim) &&
            in.U32("ssm.conv_kernel", &out.ssm_conv_kernel) &&
            in.F32("kda.gate_lower_bound", &out.kda_gate_lower_bound) &&
            in.Bool("kda.safe_gate", &out.kda_safe_gate, false) &&
            in.U32("expert_count", &out.n_expert) &&
            in.U32("expert_used_count", &out.n_expert_used) &&
            in.U32("expert_shared_count", &out.n_expert_shared) &&
            in.U32("expert_group_count", &out.n_group) &&
            in.U32("expert_group_used_count", &out.n_group_used) &&
            in.U32("expert_gating_func", &out.expert_gating_func) &&
            in.Bool("expert_weights_norm", &out.expert_weights_norm, false) &&
            in.F32("expert_weights_scale", &out.expert_weights_scale, false);
  if (!ok) {
    return false;
  }

  // Per-layer attention flavour: 0 KV heads marks a KDA block.
  std::vector<std::int64_t> head_kv;
  if (!in.IntArray("attention.head_count_kv", &head_kv)) {
    return false;
  }
  if (head_kv.size() != out.n_layer_total) {
    return in.Fail("config: attention.head_count_kv has " +
                   std::to_string(head_kv.size()) + " entries, expected " +
                   std::to_string(out.n_layer_total));
  }
  if (out.n_layer_nextn > 1) {
    return in.Fail("config: nextn_predict_layers is " +
                   std::to_string(out.n_layer_nextn) +
                   "; this extension implements one MTP layer");
  }
  out.n_layer = out.n_layer_total - out.n_layer_nextn;
  out.n_head_kv.resize(out.n_layer_total);
  out.kind.resize(out.n_layer_total);
  for (std::uint32_t il = 0; il < out.n_layer_total; ++il) {
    if (head_kv[il] < 0 || head_kv[il] > 1) {
      return in.Fail("config: attention.head_count_kv[" + std::to_string(il) +
                     "] is " + std::to_string(head_kv[il]) +
                     "; only 0 (KDA) and 1 (MLA) are supported");
    }
    out.n_head_kv[il] = static_cast<std::uint32_t>(head_kv[il]);
    out.kind[il] = head_kv[il] == 0 ? LayerKind::kKda : LayerKind::kMla;
  }

  std::vector<std::int64_t> sections;
  if (in.IntArray("rope.dimension_sections", &sections, false) &&
      !sections.empty()) {
    std::uint32_t sum = 0;
    for (const auto section : sections) {
      if (section < 0) {
        return in.Fail("config: rope.dimension_sections is negative");
      }
      sum += static_cast<std::uint32_t>(section);
      out.rope_sections.push_back(static_cast<std::uint32_t>(section));
    }
    // The sections count rotary *pairs* (llama.cpp's convention for mrope):
    // [8, 12, 12] over 64 rotary dimensions is 32 pairs.
    if (sum * 2 != out.n_rot) {
      return in.Fail("config: rope.dimension_sections sum " +
                     std::to_string(sum) + " pairs != rope.dimension_count/2 " +
                     std::to_string(out.n_rot / 2));
    }
  }

  // Sampling defaults live under general.sampling.*, not the architecture prefix.
  if (const auto value = reader.GetMetadataFloat32("general.sampling.temp")) {
    out.sampling_temperature = *value;
  }
  if (const auto value = reader.GetMetadataUint32("general.sampling.top_k")) {
    out.sampling_top_k = *value;
  }
  if (const auto value = reader.GetMetadataFloat32("general.sampling.top_p")) {
    out.sampling_top_p = *value;
  }

  if (!in.FloatArray("swiglu_clamp_exp", &out.swiglu_clamp_exp, false) ||
      !in.FloatArray("swiglu_clamp_shexp", &out.swiglu_clamp_shexp, false)) {
    return false;
  }
  if (!out.swiglu_clamp_exp.empty() &&
      out.swiglu_clamp_exp.size() != out.n_layer_total) {
    return in.Fail("config: swiglu_clamp_exp has " +
                   std::to_string(out.swiglu_clamp_exp.size()) +
                   " entries, expected " + std::to_string(out.n_layer_total));
  }
  if (!out.swiglu_clamp_shexp.empty() &&
      out.swiglu_clamp_shexp.size() != out.n_layer_total) {
    return in.Fail("config: swiglu_clamp_shexp has " +
                   std::to_string(out.swiglu_clamp_shexp.size()) +
                   " entries, expected " + std::to_string(out.n_layer_total));
  }

  // Invariants of the checkpoint this extension implements (Ling 3.2 Air).
  const auto expect = [&](std::string_view name, std::uint64_t value,
                          std::uint64_t expected) {
    if (value == expected) {
      return true;
    }
    return in.Fail("config: " + std::string(name) + " is " +
                   std::to_string(value) + ", this extension implements " +
                   std::to_string(expected));
  };
  if (!expect("block_count", out.n_layer_total, 43) ||
      !expect("nextn_predict_layers", out.n_layer_nextn, 1) ||
      !expect("embedding_length", out.n_embd, 2560) ||
      !expect("feed_forward_length", out.n_ff, 6144) ||
      !expect("expert_feed_forward_length", out.n_ff_exp, 768) ||
      !expect("expert_shared_feed_forward_length", out.n_ff_shexp, 768) ||
      !expect("leading_dense_block_count", out.n_dense_lead, 2) ||
      !expect("attention.head_count", out.n_head, 32) ||
      !expect("attention.key_length_mla", out.n_head_dim_mla, 192) ||
      !expect("attention.value_length_mla", out.n_head_dim_v_mla, 128) ||
      !expect("attention.kv_lora_rank", out.n_lora_kv, 512) ||
      !expect("rope.dimension_count", out.n_rot, 64) ||
      !expect("kda.head_dim", out.n_kda_head_dim, 128) ||
      !expect("ssm.conv_kernel", out.ssm_conv_kernel, 4) ||
      !expect("expert_count", out.n_expert, 512) ||
      !expect("expert_used_count", out.n_expert_used, 8) ||
      !expect("expert_shared_count", out.n_expert_shared, 1) ||
      !expect("expert_group_count", out.n_group, 8) ||
      !expect("expert_group_used_count", out.n_group_used, 4)) {
    return false;
  }
  if (out.expert_gating_func != 2) {
    return in.Fail("config: expert_gating_func is " +
                   std::to_string(out.expert_gating_func) +
                   ", expected 2 (sigmoid)");
  }

  *config = std::move(out);
  return true;
}

std::string DescribeConfig(const Ling3Config& config) {
  std::ostringstream out;
  out << "blocks: " << config.n_layer << " trunk + " << config.n_layer_nextn
      << " MTP (" << config.KdaCount() << " KDA / " << config.MlaCount()
      << " MLA)\n"
      << "hidden: " << config.n_embd << ", dense ffn " << config.n_ff
      << ", experts " << config.n_expert << " of " << config.n_ff_exp
      << ", shared " << config.n_ff_shexp << "\n"
      << "attention: " << config.n_head << " heads, MLA qk "
      << config.n_head_dim_mla << " (rope " << config.n_rot << "), v "
      << config.n_head_dim_v_mla << ", kv lora " << config.n_lora_kv
      << "; KDA head " << config.n_kda_head_dim << ", conv "
      << config.ssm_conv_kernel << "\n"
      << "moe: top-" << config.n_expert_used << " of " << config.n_group
      << "x" << config.n_group / config.n_group_used << " groups ("
      << config.n_group_used << " used), scale " << config.expert_weights_scale
      << ", norm " << (config.expert_weights_norm ? "on" : "off") << "\n"
      << "vocab " << config.n_vocab << ", context " << config.n_ctx_train
      << ", rms eps " << config.rms_eps << ", rope base "
      << config.rope_freq_base;
  return out.str();
}

}  // namespace gufo::ext::ling3_air
