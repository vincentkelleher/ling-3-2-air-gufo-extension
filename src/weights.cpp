#include "src/weights.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

namespace gufo::ext::ling3_air {
namespace {

/// Storage types the kernels implement. Anything else (an unusual quant) is
/// refused up front instead of producing wrong numbers later.
bool SupportedType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kF32:
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
      return true;
    default:
      return false;
  }
}

class Mapper {
 public:
  Mapper(const core::GgufReader& reader, const Ling3Config& config,
         Ling3Weights* weights)
      : reader_(reader), config_(config), weights_(weights) {}

  void set_error(std::string* error) { error_ = error; }

  [[nodiscard]] bool Fail(std::string message) {
    if (error_ != nullptr) {
      *error_ = std::move(message);
    }
    return false;
  }

  /// Maps `name`, requiring it to be `ndims`-dimensional with the expected
  /// extents. `ndims == 0` checks a 1-D tensor of `dims[0]` elements.
  [[nodiscard]] bool Map(std::string_view name, Tensor* out,
                         std::initializer_list<std::uint64_t> dims,
                         bool required = true) {
    const core::GgufTensorInfo* info = reader_.FindTensor(name);
    if (info == nullptr) {
      if (!required) {
        return true;
      }
      return Fail("weights: missing tensor " + std::string(name));
    }
    if (!info->sized) {
      return Fail("weights: tensor " + std::string(name) + " has an unsupported storage type");
    }
    if (!SupportedType(info->type)) {
      return Fail("weights: tensor " + std::string(name) + " is " +
                  std::string(core::ToString(info->type)) +
                  ", which the kernels do not implement");
    }
    const std::vector<std::uint64_t> expected(dims.begin(), dims.end());
    if (info->dimensions != expected) {
      std::ostringstream message;
      message << "weights: tensor " << name << " has shape [";
      for (std::size_t i = 0; i < info->dimensions.size(); ++i) {
        message << (i == 0 ? "" : ", ") << info->dimensions[i];
      }
      message << "], expected [";
      for (std::size_t i = 0; i < expected.size(); ++i) {
        message << (i == 0 ? "" : ", ") << expected[i];
      }
      message << "]";
      return Fail(message.str());
    }
    out->data = info->data;
    out->bytes = info->size_bytes;
    out->type = info->type;
    out->ndims = static_cast<std::uint32_t>(info->dimensions.size());
    for (std::size_t i = 0; i < info->dimensions.size() && i < 4; ++i) {
      out->ne[i] = info->dimensions[i];
    }
    return true;
  }

  [[nodiscard]] bool MapName(std::string_view name, Tensor* out,
                             std::initializer_list<std::uint64_t> dims) {
    return Map(std::string(name), out, dims);
  }

 private:
  const core::GgufReader& reader_;
  const Ling3Config& config_;
  Ling3Weights* weights_;
  std::string* error_{nullptr};
};

std::string LayerName(std::uint32_t layer, std::string_view suffix) {
  std::string name = "blk.";
  name += std::to_string(layer);
  name.push_back('.');
  name.append(suffix);
  return name;
}

}  // namespace

bool LoadWeights(const core::GgufReader& reader, const Ling3Config& config,
                 Ling3Weights* weights, std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  Ling3Weights out;
  Mapper map(reader, config, &out);
  map.set_error(error);

  const std::uint64_t n_embd = config.n_embd;
  const std::uint64_t n_vocab = config.n_vocab;
  const std::uint64_t n_head = config.n_head;
  const std::uint64_t kda_inner = config.KdaInner();
  const std::uint64_t head_dim = config.n_kda_head_dim;
  const std::uint64_t qk_dim = config.n_head_dim_mla;
  const std::uint64_t nope_dim = config.NopeDim();
  const std::uint64_t v_dim = config.n_head_dim_v_mla;
  const std::uint64_t kv_lora = config.n_lora_kv;
  const std::uint64_t n_expert = config.n_expert;
  const std::uint64_t ff_exp = config.n_ff_exp;
  const std::uint64_t ff_shexp = config.n_ff_shexp;

  if (!map.Map("token_embd.weight", &out.token_embd, {n_embd, n_vocab}) ||
      !map.Map("output_norm.weight", &out.output_norm, {n_embd}) ||
      !map.Map("output.weight", &out.output, {n_embd, n_vocab}, false)) {
    return false;
  }

  out.layers.resize(config.n_layer_total);

  for (std::uint32_t il = 0; il < config.n_layer_total; ++il) {
    LayerWeights& layer = out.layers[il];
    layer.kind = config.kind[il];
    layer.dense = il < config.n_dense_lead;
    const bool is_mtp = il >= config.n_layer;

    if (!map.Map(LayerName(il, "attn_norm.weight"), &layer.attn_norm, {n_embd}) ||
        !map.Map(LayerName(il, "ffn_norm.weight"), &layer.ffn_norm, {n_embd})) {
      return false;
    }

    if (layer.kind == LayerKind::kKda) {
      if (!map.Map(LayerName(il, "attn_q.weight"), &layer.kda_q, {n_embd, kda_inner}) ||
          !map.Map(LayerName(il, "attn_k.weight"), &layer.kda_k, {n_embd, kda_inner}) ||
          !map.Map(LayerName(il, "attn_v.weight"), &layer.kda_v, {n_embd, kda_inner}) ||
          !map.Map(LayerName(il, "ssm_conv1d_q.weight"), &layer.kda_conv_q,
                   {config.ssm_conv_kernel, 1, kda_inner}) ||
          !map.Map(LayerName(il, "ssm_conv1d_k.weight"), &layer.kda_conv_k,
                   {config.ssm_conv_kernel, 1, kda_inner}) ||
          !map.Map(LayerName(il, "ssm_conv1d_v.weight"), &layer.kda_conv_v,
                   {config.ssm_conv_kernel, 1, kda_inner}) ||
          !map.Map(LayerName(il, "ssm_f_a.weight"), &layer.kda_f_a, {n_embd, kda_inner}) ||
          !map.Map(LayerName(il, "ssm_beta.weight"), &layer.kda_beta, {n_embd, n_head}) ||
          !map.Map(LayerName(il, "ssm_a"), &layer.kda_a, {1, n_head}) ||
          !map.Map(LayerName(il, "ssm_dt.bias"), &layer.kda_dt, {kda_inner}) ||
          !map.Map(LayerName(il, "ssm_g_a.weight"), &layer.kda_g_a, {n_embd, kda_inner}) ||
          !map.Map(LayerName(il, "ssm_norm.weight"), &layer.kda_o_norm, {head_dim}) ||
          !map.Map(LayerName(il, "attn_output.weight"), &layer.kda_out, {kda_inner, n_embd})) {
        return false;
      }
    } else {
      if (!map.Map(LayerName(il, "attn_q.weight"), &layer.mla_q, {n_embd, n_head * qk_dim}) ||
          !map.Map(LayerName(il, "attn_kv_a_mqa.weight"), &layer.mla_kv_a,
                   {n_embd, kv_lora + config.n_rot}) ||
          !map.Map(LayerName(il, "attn_kv_a_norm.weight"), &layer.mla_kv_a_norm, {kv_lora}) ||
          !map.Map(LayerName(il, "attn_k_b.weight"), &layer.mla_k_b,
                   {nope_dim, kv_lora, n_head}) ||
          !map.Map(LayerName(il, "attn_v_b.weight"), &layer.mla_v_b,
                   {kv_lora, v_dim, n_head}) ||
          !map.Map(LayerName(il, "attn_gate.weight"), &layer.mla_gate, {n_embd, n_head}) ||
          !map.Map(LayerName(il, "attn_output.weight"), &layer.mla_out,
                   {n_head * v_dim, n_embd})) {
        return false;
      }
    }

    if (layer.dense) {
      if (!map.Map(LayerName(il, "ffn_gate.weight"), &layer.ffn_gate, {n_embd, config.n_ff}) ||
          !map.Map(LayerName(il, "ffn_up.weight"), &layer.ffn_up, {n_embd, config.n_ff}) ||
          !map.Map(LayerName(il, "ffn_down.weight"), &layer.ffn_down, {config.n_ff, n_embd})) {
        return false;
      }
    } else {
      if (!map.Map(LayerName(il, "ffn_gate_inp.weight"), &layer.gate_inp, {n_embd, n_expert}) ||
          !map.Map(LayerName(il, "exp_probs_b.bias"), &layer.exp_probs_b, {n_expert}) ||
          !map.Map(LayerName(il, "ffn_gate_exps.weight"), &layer.gate_exps,
                   {n_embd, ff_exp, n_expert}) ||
          !map.Map(LayerName(il, "ffn_up_exps.weight"), &layer.up_exps,
                   {n_embd, ff_exp, n_expert}) ||
          !map.Map(LayerName(il, "ffn_down_exps.weight"), &layer.down_exps,
                   {ff_exp, n_embd, n_expert}) ||
          !map.Map(LayerName(il, "ffn_gate_shexp.weight"), &layer.gate_shexp, {n_embd, ff_shexp}) ||
          !map.Map(LayerName(il, "ffn_up_shexp.weight"), &layer.up_shexp, {n_embd, ff_shexp}) ||
          !map.Map(LayerName(il, "ffn_down_shexp.weight"), &layer.down_shexp, {ff_shexp, n_embd})) {
        return false;
      }
    }

    if (is_mtp) {
      if (!map.Map(LayerName(il, "nextn.eh_proj.weight"), &layer.nextn_eh_proj,
                   {2 * n_embd, n_embd}) ||
          !map.Map(LayerName(il, "nextn.enorm.weight"), &layer.nextn_enorm, {n_embd}) ||
          !map.Map(LayerName(il, "nextn.hnorm.weight"), &layer.nextn_hnorm, {n_embd}) ||
          !map.Map(LayerName(il, "layer_output_norm.weight"),
                   &layer.nextn_shared_head_norm, {n_embd})) {
        return false;
      }
    }
  }

  // Every non-tokenizer tensor must be one this map claimed: an unexpected
  // `blk.*` tensor means the checkpoint differs from what the engine
  // implements.
  // Owns the names: LayerName() returns temporaries.
  std::set<std::string> mapped;
  const auto add = [&mapped](const Tensor& tensor, std::string_view name) {
    if (tensor.Valid()) {
      mapped.insert(std::string(name));
    }
  };
  add(out.token_embd, "token_embd.weight");
  add(out.output_norm, "output_norm.weight");
  add(out.output, "output.weight");
  static constexpr std::string_view kSuffixes[] = {
      "attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
      "attn_v.weight", "attn_output.weight", "ssm_conv1d_q.weight",
      "ssm_conv1d_k.weight", "ssm_conv1d_v.weight", "ssm_f_a.weight",
      "ssm_beta.weight", "ssm_a", "ssm_dt.bias", "ssm_g_a.weight",
      "ssm_norm.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
      "attn_k_b.weight", "attn_v_b.weight", "attn_gate.weight",
      "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
      "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
      "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
      "ffn_up_shexp.weight", "ffn_down_shexp.weight", "nextn.eh_proj.weight",
      "nextn.enorm.weight", "nextn.hnorm.weight", "layer_output_norm.weight"};
  for (std::uint32_t il = 0; il < config.n_layer_total; ++il) {
    for (const auto suffix : kSuffixes) {
      mapped.insert(LayerName(il, suffix));
    }
  }

  std::size_t count = 0;
  std::size_t bytes = 0;
  std::map<core::GgmlType, std::size_t> by_type;
  for (const auto& info : reader.GetTensors()) {
    const bool owned = info.name.rfind("blk.", 0) == 0 ||
                       info.name == "token_embd.weight" ||
                       info.name == "output_norm.weight" ||
                       info.name == "output.weight";
    if (!owned) {
      continue;  // tokenizer / other metadata tensors
    }
    if (mapped.count(std::string(info.name)) == 0) {
      return map.Fail("weights: unexpected tensor " + std::string(info.name));
    }
    ++count;
    bytes += info.size_bytes;
    by_type[info.type] += info.size_bytes;
  }
  out.tensor_count = count;
  out.total_bytes = bytes;
  out.bytes_by_type.assign(by_type.begin(), by_type.end());

  *weights = std::move(out);
  return true;
}

std::string DescribeWeights(const Ling3Weights& weights) {
  std::ostringstream out;
  out << "tensors: " << weights.tensor_count << ", "
      << (weights.total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB";
  for (const auto& [type, bytes] : weights.bytes_by_type) {
    out << "\n  " << core::ToString(type) << ": "
        << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB";
  }
  return out.str();
}

}  // namespace gufo::ext::ling3_air
