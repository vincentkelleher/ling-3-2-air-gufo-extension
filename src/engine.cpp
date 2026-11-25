#include "src/engine.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include <sys/mman.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/hip/attention.hpp"
#include "src/hip/kernels.hpp"
#include "src/hip/layers.hpp"
#include "src/hip/moe.hpp"
#include "src/hip/ops.hpp"

namespace gufo::ext::ling3_air {
namespace {

using hip::StorageType;

StorageType Storage(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ8_0:
      return hip::kQ8_0;
    case core::GgmlType::kQ4_K:
      return hip::kQ4K;
    case core::GgmlType::kQ6_K:
      return hip::kQ6K;
    case core::GgmlType::kQ5_K:
      return hip::kQ5K;
    default:
      return hip::kF32;
  }
}

int RawType(core::GgmlType type) {
  return type == core::GgmlType::kF32 ? 0 : 1;
}

}  // namespace

struct Engine::Impl {
  Ling3Config config;
  Ling3Weights weights;
  EngineOptions options;

  std::vector<const std::uint8_t*> host_base;
  std::vector<std::size_t> host_size;
  std::vector<std::uint8_t*> device_base;
  std::size_t device_bytes{0};

  // Activations and small scratch.
  float* x{nullptr};
  float* h{nullptr};
  float* attn_out{nullptr};
  float* ffn_inp{nullptr};
  float* ffn_h{nullptr};
  float* q{nullptr};
  float* k{nullptr};
  float* v{nullptr};
  float* cq{nullptr};
  float* ck{nullptr};
  float* cv{nullptr};
  float* gate_pre{nullptr};
  float* kda_gate{nullptr};
  float* beta_pre{nullptr};
  float* beta{nullptr};
  float* qn{nullptr};
  float* kn{nullptr};
  float* q_abs{nullptr};
  float* q_pe{nullptr};
  float* scores{nullptr};
  float* latent{nullptr};
  float* mla_out{nullptr};
  float* kvq{nullptr};
  float* moe_logits{nullptr};
  float* expert_gate{nullptr};
  float* expert_up{nullptr};
  float* expert_down{nullptr};
  float* logits{nullptr};
  std::int32_t* moe_ids{nullptr};
  float* moe_weights{nullptr};
  std::int32_t* argmax{nullptr};

  std::vector<float*> conv_state;
  std::vector<float*> delta_state;
  std::vector<float*> mla_cache;
  std::vector<float*> k_absorb;
  std::vector<float*> v_absorb;
  bool rope_neox{true};

  [[nodiscard]] const void* Dev(const Tensor& tensor) const {
    const auto* pointer = static_cast<const std::uint8_t*>(tensor.data);
    for (std::size_t i = 0; i < host_base.size(); ++i) {
      if (pointer >= host_base[i] && pointer < host_base[i] + host_size[i]) {
        return device_base[i] + (pointer - host_base[i]);
      }
    }
    return nullptr;
  }
};

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::Create(const core::GgufReader& reader,
                                       const Ling3Config& config,
                                       const Ling3Weights& weights,
                                       const EngineOptions& options,
                                       std::string* error) {
  auto fail = [&](std::string message) -> std::unique_ptr<Engine> {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return nullptr;
  };

  std::unique_ptr<Engine> engine(new Engine());
  engine->impl_ = std::make_unique<Impl>();
  Impl& m = *engine->impl_;
  m.config = config;
  m.options = options;
  if (const char* rope = std::getenv("LING3_ROPE")) {
    m.rope_neox = std::string(rope) != "interleaved";
  }
  m.weights = weights;

  // Zero copy: Strix Halo host memory is device-accessible, and hipHostRegister
  // accepts the reader's read-only mapping. Registering it keeps one copy of
  // the weights (the page cache) instead of duplicating 80 GiB into GTT, which
  // is what made an uploading engine thrash swap. tests/../tools/zerocopy_probe
  // measured 238 GB/s warm through the registered mapping versus 223 GB/s for a
  // device copy, so nothing is given up.
  for (const auto& region : reader.GetMappedRegions()) {
    void* host = const_cast<void*>(region.data);
    const hipError_t registered =
        hipHostRegister(host, region.size, hipHostRegisterMapped);
    if (registered != hipSuccess) {
      return fail(std::string("engine: cannot register a weight shard: ") +
                  hipGetErrorString(registered));
    }
    void* device = nullptr;
    const hipError_t mapped = hipHostGetDevicePointer(&device, host, 0);
    if (mapped != hipSuccess || device == nullptr) {
      return fail(std::string("engine: cannot map a weight shard: ") +
                  hipGetErrorString(mapped));
    }
    // Prove the registered range is readable end to end: sample four spots on
    // the GPU and compare with the mapped bytes. A partially pinned range would
    // otherwise show up much later as NaN activations.
    for (const double fraction : {0.0, 0.33, 0.66, 0.999}) {
      const std::size_t offset = static_cast<std::size_t>(
          fraction * static_cast<double>(region.size - 256));
      std::uint8_t probe[256];
      if (hipMemcpy(probe, static_cast<const std::uint8_t*>(device) + offset,
                    sizeof(probe), hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("engine: cannot read back a weight shard");
      }
      if (std::memcmp(probe, static_cast<const std::uint8_t*>(region.data) + offset,
                      sizeof(probe)) != 0) {
        return fail("engine: registered shard reads back differently at offset " +
                    std::to_string(offset));
      }
    }
    m.host_base.push_back(static_cast<const std::uint8_t*>(region.data));
    m.host_size.push_back(region.size);
    m.device_base.push_back(static_cast<std::uint8_t*>(device));
    m.device_bytes += region.size;
  }

  // With LING3_STATS, sample every MiB of each tensor in blocks 30-32 (the
  // first block that goes NaN) and compare the device bytes with the mapping.
  if (std::getenv("LING3_STATS") != nullptr) {
    for (const auto& info : reader.GetTensors()) {
      const std::string name(info.name);
      const bool interesting = name.rfind("blk.30.", 0) == 0 ||
                               name.rfind("blk.31.", 0) == 0 ||
                               name.rfind("blk.32.", 0) == 0;
      if (!interesting) {
        continue;
      }
      Tensor probe;
      probe.data = info.data;
      const void* device = m.Dev(probe);
      if (device == nullptr) {
        return fail("engine: tensor outside the registered shards: " + name);
      }
      // Compare every byte in 1 MiB chunks for block 31 (2.2 GiB of experts)
      // and sample the others.
      const bool thorough = name.rfind("blk.31.", 0) == 0;
      const std::size_t chunk = 1024 * 1024;
      std::vector<std::uint8_t> buffer(chunk);
      for (std::size_t offset = 0; offset < info.size_bytes; offset += chunk) {
        const std::size_t limit =
            offset + chunk <= info.size_bytes ? chunk : info.size_bytes - offset;
        if (!thorough && offset % (64 * chunk) != 0) {
          continue;
        }
        hipMemcpy(buffer.data(), static_cast<const std::uint8_t*>(device) + offset,
                  limit, hipMemcpyDeviceToHost);
        if (std::memcmp(buffer.data(),
                        static_cast<const std::uint8_t*>(info.data) + offset,
                        limit) != 0) {
          return fail("engine: " + name + " differs at offset " +
                      std::to_string(offset) +
                      (thorough ? " (thorough)" : " (sampled)"));
        }
      }
    }
    std::fprintf(stderr, "engine: block 30-32 tensors read back correctly\n");
  }

  // Refuse storage types without a kernel: reading an unimplemented
  // quantization as floats produced the first NaN hunt (block 31's down_exps is
  // Q5_K in this per-tensor layout).
  for (const auto& info : reader.GetTensors()) {
    const auto type = info.type;
    const bool known = type == core::GgmlType::kF32 || type == core::GgmlType::kF16 ||
                       type == core::GgmlType::kBF16 ||
                       type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_0 ||
                       type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
                       type == core::GgmlType::kQ6_K;
    if (!known) {
      return fail(std::string("engine: unsupported storage type ") +
                  std::string(core::ToString(type)) + " in " +
                  std::string(info.name));
    }
  }

  // Every tensor must resolve to a device pointer that reads back the same
  // bytes; a mis-mapped tensor shows up later as NaN activations.
  for (const auto& info : reader.GetTensors()) {
    Tensor probe;
    probe.data = info.data;
    probe.bytes = info.size_bytes;
    const void* device = m.Dev(probe);
    if (device == nullptr) {
      return fail(std::string("engine: tensor outside the registered shards: ") +
                  std::string(info.name));
    }
    std::uint8_t a[64];
    std::uint8_t b[64];
    hipMemcpy(a, device, sizeof(a), hipMemcpyDeviceToHost);
    std::memcpy(b, info.data, sizeof(b));
    if (std::memcmp(a, b, sizeof(a)) != 0) {
      return fail(std::string("engine: tensor maps to the wrong bytes: ") +
                  std::string(info.name));
    }
  }

  const std::size_t n_embd = config.n_embd;
  const std::size_t heads = config.n_head;
  const std::size_t kda_dim = config.n_kda_head_dim;
  const std::size_t kda_inner = config.KdaInner();
  const std::size_t q_dim = heads * config.n_head_dim_mla;
  const std::size_t v_dim = heads * config.n_head_dim_v_mla;
  const std::size_t latent = config.n_lora_kv;
  const std::size_t kv_dim = latent + config.n_rot;
  const std::size_t context = options.max_context;

  auto alloc = [](float** pointer, std::size_t count) {
    return hipMalloc(reinterpret_cast<void**>(pointer), count * sizeof(float)) ==
           hipSuccess;
  };
  auto alloc_int = [](std::int32_t** pointer, std::size_t count) {
    return hipMalloc(reinterpret_cast<void**>(pointer),
                     count * sizeof(std::int32_t)) == hipSuccess;
  };

  const bool ok =
      alloc(&m.x, n_embd) && alloc(&m.h, n_embd) && alloc(&m.attn_out, n_embd) &&
      alloc(&m.ffn_inp, n_embd) && alloc(&m.ffn_h, n_embd) && alloc(&m.q, q_dim) &&
      alloc(&m.k, kda_inner) && alloc(&m.v, kda_inner) && alloc(&m.cq, kda_inner) &&
      alloc(&m.ck, kda_inner) && alloc(&m.cv, kda_inner) &&
      alloc(&m.gate_pre, kda_inner > q_dim ? kda_inner : q_dim) &&
      alloc(&m.kda_gate, kda_inner) && alloc(&m.beta_pre, heads) &&
      alloc(&m.beta, heads) && alloc(&m.qn, kda_inner) &&
      alloc(&m.kn, kda_inner > v_dim ? kda_inner : v_dim) &&
      alloc(&m.q_abs, heads * latent) && alloc(&m.q_pe, heads * config.n_rot) &&
      alloc(&m.scores, heads * context) && alloc(&m.latent, heads * latent) &&
      alloc(&m.mla_out, v_dim) && alloc(&m.kvq, kv_dim) &&
      alloc(&m.moe_logits, config.n_expert) &&
      alloc(&m.expert_gate, config.n_ff > config.n_ff_exp ? config.n_ff : config.n_ff_exp) &&
      alloc(&m.expert_up, config.n_ff > config.n_ff_exp ? config.n_ff : config.n_ff_exp) &&
      alloc(&m.expert_down, n_embd) && alloc(&m.logits, config.n_vocab) &&
      alloc_int(&m.moe_ids, config.n_expert_used) &&
      alloc(&m.moe_weights, config.n_expert_used) && alloc_int(&m.argmax, 1);
  if (!ok) {
    return fail("engine: device scratch allocation failed");
  }

  m.conv_state.assign(config.n_layer_total, nullptr);
  m.delta_state.assign(config.n_layer_total, nullptr);
  m.mla_cache.assign(config.n_layer_total, nullptr);
  m.k_absorb.assign(config.n_layer_total, nullptr);
  m.v_absorb.assign(config.n_layer_total, nullptr);
  for (std::uint32_t il = 0; il < config.n_layer_total; ++il) {
    if (config.IsKda(il)) {
      if (!alloc(&m.conv_state[il], 3 * kda_inner) ||
          !alloc(&m.delta_state[il], heads * kda_dim * kda_dim)) {
        return fail("engine: KDA state allocation failed");
      }
      hipMemset(m.conv_state[il], 0, 3 * kda_inner * sizeof(float));
      hipMemset(m.delta_state[il], 0, heads * kda_dim * kda_dim * sizeof(float));
    } else {
      if (!alloc(&m.mla_cache[il], context * kv_dim)) {
        return fail("engine: MLA cache allocation failed");
      }
      hipMemset(m.mla_cache[il], 0, context * kv_dim * sizeof(float));
    }
  }

  // attn_k_b and attn_v_b are quantized per-head matrices the absorb kernels
  // read as floats; dequantize them once (8 blocks x 2 x 8 MiB).
  for (std::uint32_t il = 0; il < config.n_layer_total; ++il) {
    if (config.IsKda(il)) {
      continue;
    }
    const LayerWeights& layer = weights.layers[il];
    if (!alloc(&m.k_absorb[il], heads * latent * config.NopeDim()) ||
        !alloc(&m.v_absorb[il], heads * config.n_head_dim_v_mla * latent)) {
      return fail("engine: absorb matrix allocation failed");
    }
    hip::LaunchDequantize(m.Dev(layer.mla_k_b), RawType(layer.mla_k_b.type),
                          heads * latent * config.NopeDim(), m.k_absorb[il], 0);
    hip::LaunchDequantize(m.Dev(layer.mla_v_b), RawType(layer.mla_v_b.type),
                          heads * config.n_head_dim_v_mla * latent, m.v_absorb[il], 0);
  }
  if (error != nullptr) {
    error->clear();
  }
  return engine;
}

bool Engine::ForwardToken(std::int32_t token, std::uint32_t position,
                          std::string* error) {
  Impl& m = *impl_;
  const Ling3Config& c = m.config;
  const auto& weights = m.weights;
  const float eps = c.rms_eps;
  const std::size_t n_embd = c.n_embd;
  const std::size_t heads = c.n_head;
  const std::size_t kda_dim = c.n_kda_head_dim;
  const std::size_t kda_inner = c.KdaInner();
  const std::size_t q_dim = heads * c.n_head_dim_mla;
  const std::size_t v_dim = heads * c.n_head_dim_v_mla;
  const std::size_t latent = c.n_lora_kv;
  hipStream_t stream = nullptr;

  auto dev = [&m](const Tensor& tensor) { return m.Dev(tensor); };
  const bool stats = std::getenv("LING3_STATS") != nullptr;
  std::vector<float> host_row;
  auto report_buf = [&](const char* label, const float* device, std::size_t count) {
    if (!stats) {
      return;
    }
    std::vector<float> host(count);
    hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost);
    double sum = 0.0;
    double maximum = 0.0;
    int nan = 0;
    for (const float value : host) {
      if (value != value) {
        ++nan;
        continue;
      }
      sum += static_cast<double>(value) * value;
      maximum = std::fabs(value) > maximum ? std::fabs(value) : maximum;
    }
    std::fprintf(stderr, "  %s: rms %.5f max %.5f nan %d\n", label,
                 std::sqrt(sum / (count == 0 ? 1 : count)), maximum, nan);
  };
  auto report = [&](const char* label, std::uint32_t layer) {
    if (!stats) {
      return;
    }
    host_row.resize(n_embd);
    hipMemcpy(host_row.data(), m.x, n_embd * sizeof(float), hipMemcpyDeviceToHost);
    double sum = 0.0;
    double maximum = 0.0;
    int nan = 0;
    for (const float value : host_row) {
      if (value != value) {
        ++nan;
        continue;
      }
      sum += static_cast<double>(value) * value;
      maximum = std::fabs(value) > maximum ? std::fabs(value) : maximum;
    }
    std::fprintf(stderr, "%s %u: rms %.4f max %.4f nan %d\n", label, layer,
                 std::sqrt(sum / n_embd), maximum, nan);
  };

  hip::LaunchEmbeddingRow(dev(weights.token_embd), RawType(weights.token_embd.type),
                          c.n_vocab, n_embd, token, m.x, stream);

  for (std::uint32_t il = 0; il < c.n_layer; ++il) {
    const LayerWeights& layer = weights.layers[il];
    if (il >= 30) {
      report_buf("pre-norm x", m.x, n_embd);
    }
    hip::LaunchRmsNorm(m.x, static_cast<const float*>(dev(layer.attn_norm)), m.h, 1,
                       n_embd, eps, stream);
    if (il >= 30) {
      report_buf("post-norm h", m.h, n_embd);
    }

    if (c.IsKda(il)) {
      hip::LaunchGemv(dev(layer.kda_q), Storage(layer.kda_q.type), kda_inner, n_embd,
                      m.h, m.q, stream);
      hip::LaunchGemv(dev(layer.kda_k), Storage(layer.kda_k.type), kda_inner, n_embd,
                      m.h, m.k, stream);
      hip::LaunchGemv(dev(layer.kda_v), Storage(layer.kda_v.type), kda_inner, n_embd,
                      m.h, m.v, stream);
      hip::LaunchConv1dStep(m.q, static_cast<const float*>(dev(layer.kda_conv_q)),
                            m.conv_state[il], m.cq, kda_inner, stream);
      hip::LaunchSilu(m.cq, m.cq, kda_inner, stream);
      hip::LaunchConv1dStep(m.k, static_cast<const float*>(dev(layer.kda_conv_k)),
                            m.conv_state[il] + kda_inner, m.ck, kda_inner, stream);
      hip::LaunchSilu(m.ck, m.ck, kda_inner, stream);
      hip::LaunchConv1dStep(m.v, static_cast<const float*>(dev(layer.kda_conv_v)),
                            m.conv_state[il] + 2 * kda_inner, m.cv, kda_inner, stream);
      hip::LaunchSilu(m.cv, m.cv, kda_inner, stream);

      hip::LaunchGemv(dev(layer.kda_f_a), Storage(layer.kda_f_a.type), kda_inner,
                      n_embd, m.h, m.gate_pre, stream);
      hip::LaunchKdaGate(m.gate_pre, static_cast<const float*>(dev(layer.kda_dt)),
                         static_cast<const float*>(dev(layer.kda_a)), m.kda_gate,
                         kda_inner, kda_dim, c.kda_gate_lower_bound, stream);
      hip::LaunchGemv(dev(layer.kda_beta), Storage(layer.kda_beta.type), heads,
                      n_embd, m.h, m.beta_pre, stream);
      hip::LaunchSigmoid(m.beta_pre, m.beta, heads, stream);

      hip::LaunchL2Norm(m.cq, m.qn, heads, kda_dim, eps, stream);
      hip::LaunchL2Norm(m.ck, m.kn, heads, kda_dim, eps, stream);
      // `cq` is free after the L2 normalisation; the delta step reads `kn`
      // while writing, so it needs a different buffer.
      hip::LaunchDeltaRuleStep(m.qn, m.kn, m.cv, m.kda_gate, m.beta,
                               m.delta_state[il], m.cq, heads, kda_dim, stream);
      hip::LaunchRmsNorm(m.cq, static_cast<const float*>(dev(layer.kda_o_norm)), m.qn,
                         heads, kda_dim, eps, stream);
      hip::LaunchGemv(dev(layer.kda_g_a), Storage(layer.kda_g_a.type), kda_inner,
                      n_embd, m.h, m.gate_pre, stream);
      hip::LaunchMulSigmoid(m.qn, m.gate_pre, kda_inner, stream);
      if (il >= 28) {
        report_buf("kda gate", m.kda_gate, kda_inner);
        report_buf("kda conv q", m.cq, kda_inner);
        report_buf("kda dn out", m.kn, kda_inner);
      }
      hip::LaunchGemv(dev(layer.kda_out), Storage(layer.kda_out.type), n_embd,
                      kda_inner, m.qn, m.attn_out, stream);
      if (il >= 28) {
        report_buf("kda attn", m.attn_out, n_embd);
      }
    } else {
      hip::LaunchGemv(dev(layer.mla_q), Storage(layer.mla_q.type), q_dim, n_embd,
                      m.h, m.q, stream);
      hip::LaunchSplitQ(m.q, m.qn, m.q_pe, heads, c.NopeDim(), c.n_rot, stream);
      hip::LaunchGemv(dev(layer.mla_kv_a), Storage(layer.mla_kv_a.type),
                      latent + c.n_rot, n_embd, m.h, m.kvq, stream);
      hip::LaunchRmsNorm(m.kvq, static_cast<const float*>(dev(layer.mla_kv_a_norm)),
                         m.latent, 1, latent, eps, stream);
      hip::LaunchRope(m.q_pe, heads, c.n_rot, c.n_rot, position, c.rope_freq_base,
                      m.rope_neox, stream);
      hip::LaunchRope(m.kvq + latent, 1, c.n_rot, c.n_rot, position,
                      c.rope_freq_base, m.rope_neox, stream);
      hip::LaunchMlaAppend(m.latent, m.kvq + latent, m.mla_cache[il], position,
                           latent, c.n_rot, stream);
      hip::LaunchGemv(dev(layer.mla_gate), Storage(layer.mla_gate.type), heads,
                      n_embd, m.h, m.beta_pre, stream);
      hip::LaunchMlaAbsorbQ(m.qn, m.k_absorb[il], m.q_abs, heads, stream);
      hip::LaunchMlaScores(m.q_abs, m.q_pe, m.mla_cache[il], heads, position + 1,
                           m.scores, stream);
      hip::LaunchMlaSoftmax(m.scores, heads, position + 1, stream);
      hip::LaunchMlaLatentSum(m.scores, m.mla_cache[il], heads, position + 1,
                              m.latent, stream);
      if (il == 5) {
        report_buf("mla q_abs", m.q_abs, heads * latent);
        report_buf("mla probs", m.scores, heads * (position + 1));
        report_buf("mla latent", m.latent, heads * latent);
      }
      hip::LaunchMlaValueAbsorb(m.latent, m.v_absorb[il], m.beta_pre, m.mla_out,
                                heads, stream);
      if (il == 5) {
        report_buf("mla out", m.mla_out, v_dim);
      }
      hip::LaunchGemv(dev(layer.mla_out), Storage(layer.mla_out.type), n_embd, v_dim,
                      m.mla_out, m.attn_out, stream);
    }

    hip::LaunchAxpy(m.x, m.attn_out, 1.0f, n_embd, stream);
    hipMemcpyAsync(m.ffn_inp, m.x, n_embd * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    hip::LaunchRmsNorm(m.ffn_inp, static_cast<const float*>(dev(layer.ffn_norm)),
                       m.ffn_h, 1, n_embd, eps, stream);

    if (layer.dense) {
      hip::LaunchFfn(dev(layer.ffn_gate), Storage(layer.ffn_gate.type),
                     dev(layer.ffn_up), Storage(layer.ffn_up.type),
                     dev(layer.ffn_down), Storage(layer.ffn_down.type), n_embd,
                     c.n_ff, m.ffn_h, 1.0f, m.x, m.expert_gate, m.expert_up,
                     m.expert_down, stream);
    } else {
      hip::LaunchGemv(dev(layer.gate_inp), Storage(layer.gate_inp.type),
                      c.n_expert, n_embd, m.ffn_h, m.moe_logits, stream);
      hip::LaunchRouterSelect(m.moe_logits,
                              static_cast<const float*>(dev(layer.exp_probs_b)),
                              c.n_expert, c.n_group, c.n_group_used,
                              c.n_expert_used, c.expert_weights_scale, m.moe_ids,
                              m.moe_weights, stream);
      if (il == 31) {
        std::int32_t ids[8] = {0};
        float w[8] = {0};
        hipMemcpy(ids, m.moe_ids, sizeof(ids), hipMemcpyDeviceToHost);
        hipMemcpy(w, m.moe_weights, sizeof(w), hipMemcpyDeviceToHost);
        std::fprintf(stderr, "  router 31: ids %d %d %d %d %d %d %d %d weights %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                     ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
                     w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        report_buf("ffn_h", m.ffn_h, n_embd);
        report_buf("expert_gate", m.expert_gate, c.n_ff_exp);
        report_buf("expert_down", m.expert_down, n_embd);
      }
      hip::LaunchMoeGather(dev(layer.gate_exps), Storage(layer.gate_exps.type),
                           dev(layer.up_exps), Storage(layer.up_exps.type),
                           dev(layer.down_exps), Storage(layer.down_exps.type),
                           n_embd, c.n_ff_exp, m.ffn_h, m.moe_ids, m.moe_weights,
                           c.n_expert_used, m.x, m.expert_gate, m.expert_up,
                           m.expert_down, stream);
      hip::LaunchFfn(dev(layer.gate_shexp), Storage(layer.gate_shexp.type),
                     dev(layer.up_shexp), Storage(layer.up_shexp.type),
                     dev(layer.down_shexp), Storage(layer.down_shexp.type), n_embd,
                     c.n_ff_shexp, m.ffn_h, 1.0f, m.x, m.expert_gate, m.expert_up,
                     m.expert_down, stream);
      if (il == 31) {
        report_buf("x after shared", m.x, n_embd);
      }
    }
    report("block", il);
  }

  report("final", c.n_layer);
  hip::LaunchRmsNorm(m.x, static_cast<const float*>(dev(weights.output_norm)), m.h, 1,
                     n_embd, eps, stream);
  hip::LaunchGemv(dev(weights.output), Storage(weights.output.type), c.n_vocab,
                  n_embd, m.h, m.logits, stream);
  hip::LaunchArgmax(m.logits, c.n_vocab, m.argmax, stream);

  const hipError_t status = hipStreamSynchronize(stream);
  if (status != hipSuccess) {
    if (error != nullptr) {
      *error = std::string("engine: forward failed: ") + hipGetErrorString(status);
    }
    return false;
  }
  return true;
}

void Engine::ReadLogits(float* host_logits) {
  hipMemcpy(host_logits, impl_->logits, impl_->config.n_vocab * sizeof(float),
            hipMemcpyDeviceToHost);
}

std::int32_t Engine::Argmax() {
  std::int32_t value = -1;
  hipMemcpy(&value, impl_->argmax, sizeof(value), hipMemcpyDeviceToHost);
  return value;
}

std::uint32_t Engine::VocabSize() const { return impl_->config.n_vocab; }

std::size_t Engine::DeviceBytes() const { return impl_->device_bytes; }

}  // namespace gufo::ext::ling3_air
