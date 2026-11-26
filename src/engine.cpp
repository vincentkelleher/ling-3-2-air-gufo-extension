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
#include <map>
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
    case core::GgmlType::kQ4_0:
      return hip::kQ4_0;
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
  /// Pre-quantised activation for the "prepared" GEMV path: a head with 156k rows
  /// otherwise re-quantises the same 2560 values once per row.
  signed char* act_values{nullptr};
  float* act_scales{nullptr};
  int* act_sums{nullptr};
  float* expert_gate{nullptr};
  float* expert_up{nullptr};
  float* expert_down{nullptr};
  float* logits{nullptr};
  std::int32_t* moe_ids{nullptr};
  float* moe_weights{nullptr};
  std::int32_t* argmax{nullptr};
  float* base_hidden{nullptr};
  float* bx{nullptr};
  float* bh{nullptr};
  float* bq{nullptr};
  float* bk{nullptr};
  float* bv{nullptr};
  float* bgated{nullptr};
  float* battn{nullptr};
  float* bffnh{nullptr};
  float* bffa{nullptr};
  float* bffb{nullptr};
  float* bffout{nullptr};
  static constexpr int kMaxChunk = 256;
  float* brooter{nullptr};
  std::int32_t* brouter_ids{nullptr};
  float* brouter_w{nullptr};
  std::int32_t* bslots{nullptr};
  std::int32_t* bcounts{nullptr};
  std::int32_t* boffsets{nullptr};
  std::int32_t* bcursor{nullptr};
  float* bgather{nullptr};
  float* bgate{nullptr};
  float* bup{nullptr};
  float* bout{nullptr};
  float* blogits{nullptr};
  float* snapshot{nullptr};
  std::size_t snapshot_floats{0};
  hipStream_t queue{nullptr};
  float* kernel_debug{nullptr};

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
  // A reader may hand back regions that start anywhere inside one mapping (Gufo's
  // engine, for instance, registers each shard from its first tensor). hipHostRegister
  // needs a page-aligned start and rejects the range otherwise, so align down and
  // register each mapping once, then offset the device pointer for each region.
  std::map<std::uintptr_t, void*> pinned;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto* data = static_cast<const std::uint8_t*>(region.data);
    if (data == nullptr || region.size == 0) {
      continue;
    }
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(data);
    const std::uintptr_t aligned_address =
        address & ~static_cast<std::uintptr_t>(4095);
    const std::size_t delta = address - aligned_address;
    void* device = nullptr;
    const auto existing = pinned.find(aligned_address);
    if (existing != pinned.end()) {
      device = existing->second;
    } else {
      void* host = reinterpret_cast<void*>(aligned_address);
      const std::size_t aligned_size = delta + region.size;
      const hipError_t registered =
          hipHostRegister(host, aligned_size, hipHostRegisterMapped);
      if (registered != hipSuccess) {
        // The usual cause on this box is another process holding the GPU's GTT:
        // a 100 GiB registration fails with EINVAL when the device is full.
        return fail(std::string("engine: cannot register a weight shard: ") +
                    hipGetErrorString(registered) + " (region " +
                    std::to_string(region.size / 1048576) + " MiB at " +
                    std::to_string(delta) + " bytes into a page)");
      }
      const hipError_t mapped = hipHostGetDevicePointer(&device, host, 0);
      if (mapped != hipSuccess || device == nullptr) {
        return fail(std::string("engine: cannot map a weight shard: ") +
                    hipGetErrorString(mapped));
      }
      pinned.emplace(aligned_address, device);
    }
    // Prove the registered range is readable end to end: sample four spots on the
    // GPU and compare with the mapped bytes. A partially pinned range would show up
    // much later as NaN activations.
    for (const double fraction : {0.0, 0.33, 0.66, 0.999}) {
      const std::size_t offset = static_cast<std::size_t>(
          fraction * static_cast<double>(region.size - 256));
      std::uint8_t probe[256];
      if (hipMemcpy(probe, static_cast<const std::uint8_t*>(device) + delta + offset,
                    sizeof(probe), hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("engine: cannot read back a weight shard");
      }
      if (std::memcmp(probe, data + offset, sizeof(probe)) != 0) {
        return fail("engine: registered shard reads back differently at offset " +
                    std::to_string(offset));
      }
    }
    m.host_base.push_back(data);
    m.host_size.push_back(region.size);
    m.device_base.push_back(static_cast<std::uint8_t*>(device) + delta);
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

  // One explicit stream for every kernel and copy: work issued on the implicit
  // stream mixed with blocking copies was not always ordered, which made
  // decoding non-deterministic (see notes/engine-plan.md).
  if (hipStreamCreateWithFlags(&m.queue, hipStreamNonBlocking) != hipSuccess) {
    return fail("engine: cannot create the compute stream");
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
  // The widest activation the prepared/wide path quantises is the KDA inner width (the
  // output projection reads it); the rest are n_embd.
  const std::size_t act_cols = std::max<std::size_t>(config.n_embd, config.KdaInner());

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
      hipMalloc(reinterpret_cast<void**>(&m.act_values), act_cols * sizeof(signed char)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&m.act_scales), (act_cols / 256) * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&m.act_sums), (act_cols / 256) * 8 * sizeof(int)) == hipSuccess &&
      alloc(&m.expert_gate, 8 * (config.n_ff > config.n_ff_exp ? config.n_ff : config.n_ff_exp)) &&
      alloc(&m.expert_up, 8 * (config.n_ff > config.n_ff_exp ? config.n_ff : config.n_ff_exp)) &&
      alloc(&m.expert_down, 8 * n_embd) && alloc(&m.logits, config.n_vocab) && alloc(&m.base_hidden, n_embd) &&
      alloc_int(&m.moe_ids, config.n_expert_used) &&
      alloc(&m.moe_weights, config.n_expert_used) && alloc_int(&m.argmax, 1) &&
      alloc(&m.kernel_debug, 8 * kda_dim) &&
      alloc(&m.bx, Impl::kMaxChunk * n_embd) && alloc(&m.bh, Impl::kMaxChunk * n_embd) &&
      alloc(&m.bq, Impl::kMaxChunk * q_dim) &&
      alloc(&m.bk, Impl::kMaxChunk * (kda_inner > kv_dim ? kda_inner : kv_dim)) &&
      alloc(&m.bv, Impl::kMaxChunk * kda_inner) && alloc(&m.bgated, Impl::kMaxChunk * q_dim) &&
      alloc(&m.battn, Impl::kMaxChunk * n_embd) && alloc(&m.bffnh, Impl::kMaxChunk * n_embd) &&
      alloc(&m.bffa, Impl::kMaxChunk * config.n_ff) && alloc(&m.bffb, Impl::kMaxChunk * config.n_ff) &&
      alloc(&m.bffout, Impl::kMaxChunk * n_embd) &&
      alloc(&m.brooter, Impl::kMaxChunk * config.n_expert) &&
      alloc_int(&m.brouter_ids, Impl::kMaxChunk * config.n_expert_used) &&
      alloc(&m.brouter_w, Impl::kMaxChunk * config.n_expert_used) &&
      alloc_int(&m.bslots, Impl::kMaxChunk * config.n_expert_used) &&
      alloc_int(&m.bcounts, config.n_expert) &&
      alloc_int(&m.boffsets, config.n_expert) &&
      alloc_int(&m.bcursor, config.n_expert) &&
      alloc(&m.bgather, Impl::kMaxChunk * n_embd) &&
      alloc(&m.bgate, Impl::kMaxChunk * config.n_ff_exp) &&
      alloc(&m.bup, Impl::kMaxChunk * config.n_ff_exp) &&
      alloc(&m.bout, Impl::kMaxChunk * n_embd) &&
      alloc(&m.blogits, 8 * static_cast<std::size_t>(config.n_vocab)) &&
      alloc(&m.snapshot, config.n_layer_total *
                             (9 * kda_inner +
                              static_cast<std::size_t>(heads) * kda_dim * kda_dim +
                              options.max_context * kv_dim));
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
      if (!alloc(&m.conv_state[il], 9 * kda_inner) ||
          !alloc(&m.delta_state[il], heads * kda_dim * kda_dim)) {
        return fail("engine: KDA state allocation failed");
      }
      hipMemset(m.conv_state[il], 0, 9 * kda_inner * sizeof(float));
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
  hip::LaunchEmbeddingRow(m.Dev(m.weights.token_embd),
                          RawType(m.weights.token_embd.type), m.config.n_vocab,
                          m.config.n_embd, token, m.x, m.queue);
  return ForwardFromEmbedded(position, error);
}

bool Engine::ForwardEmbedding(const float* embedding, std::uint32_t position,
                              std::string* error) {
  Impl& m = *impl_;
  if (embedding == nullptr) {
    if (error != nullptr) *error = "ForwardEmbedding: null embedding";
    return false;
  }
  if (hipMemcpy(m.x, embedding, m.config.n_embd * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    if (error != nullptr) *error = "ForwardEmbedding: cannot upload the embedding";
    return false;
  }
  return ForwardFromEmbedded(position, error);
}

bool Engine::ForwardFromEmbedded(std::uint32_t position, std::string* error) {
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
  hipStream_t stream = m.queue;

  auto dev = [&m](const Tensor& tensor) { return m.Dev(tensor); };
  const bool stats = std::getenv("LING3_STATS") != nullptr;
  std::vector<float> host_row;
  auto dump_raw = [&](const char* path, const float* device, std::size_t count) {
    if (std::getenv("LING3_DUMP") == nullptr) {
      return;
    }
    std::vector<float> host(count);
    hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost);
    FILE* file = std::fopen(path, "wb");
    if (file != nullptr) {
      std::fwrite(host.data(), sizeof(float), count, file);
      std::fclose(file);
    }
  };
  auto dump_at = [&](const char* format, const float* device, std::size_t count,
                     std::uint32_t position) {
    if (std::getenv("LING3_DUMP") == nullptr) {
      return;
    }
    char path[128];
    std::snprintf(path, sizeof(path), format, position);
    std::vector<float> host(count);
    hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost);
    FILE* file = std::fopen(path, "wb");
    if (file != nullptr) {
      std::fwrite(host.data(), sizeof(float), count, file);
      std::fclose(file);
    }
  };
  auto dump_lp = [&](const char* format, const float* device, std::size_t count,
                     std::uint32_t layer, std::uint32_t position) {
    if (std::getenv("LING3_DUMP_BLOCKS") == nullptr) {
      return;
    }
    char path[128];
    std::snprintf(path, sizeof(path), format, layer, position);
    std::vector<float> host(count);
    hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost);
    FILE* file = std::fopen(path, "wb");
    if (file != nullptr) {
      std::fwrite(host.data(), sizeof(float), count, file);
      std::fclose(file);
    }
  };
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
  // `LING3_PROFILE=1` times the per-token groups with events (recorded
  // asynchronously, resolved after the single stream sync) so optimisation work is
  // driven by where the time actually goes.
  const bool profile = std::getenv("LING3_PROFILE") != nullptr;
  std::vector<hipEvent_t> profile_events;
  if (profile) {
    profile_events.resize(static_cast<std::size_t>(c.n_layer) * 8, nullptr);
    for (auto& event : profile_events) {
      hipEventCreateWithFlags(&event, hipEventDefault);
    }
  }
  auto mark = [&](std::uint32_t layer, int slot) {
    if (profile) {
      hipEventRecord(profile_events[static_cast<std::size_t>(layer) * 8 + slot], stream);
    }
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

    mark(il, 0);
    if (c.IsKda(il)) {
      hip::LaunchGemv(dev(layer.kda_q), Storage(layer.kda_q.type), kda_inner, n_embd,
                      m.h, m.q, stream);
      hip::LaunchGemv(dev(layer.kda_k), Storage(layer.kda_k.type), kda_inner, n_embd,
                      m.h, m.k, stream);
      hip::LaunchGemv(dev(layer.kda_v), Storage(layer.kda_v.type), kda_inner, n_embd,
                      m.h, m.v, stream);
      if (il == 0 && std::getenv("LING3_DUMP") != nullptr) {
        auto dump = [&](const char* path, const float* device, std::size_t count) {
          std::vector<float> host(count);
          hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost);
          FILE* file = std::fopen(path, "wb");
          if (file != nullptr) {
            std::fwrite(host.data(), sizeof(float), count, file);
            std::fclose(file);
          }
        };
        dump("/work/ling3_q0.bin", m.q, kda_inner);
        dump("/work/ling3_h0.bin", m.h, n_embd);
      }
      if (il == 0) {
        dump_at("/work/p_qproj_%u.bin", m.q, kda_inner, position);
        dump_at("/work/p_kproj_%u.bin", m.k, kda_inner, position);
        dump_at("/work/p_vproj_%u.bin", m.v, kda_inner, position);
      }
      mark(il, 3);
      hip::LaunchConv1dStep(m.q, static_cast<const float*>(dev(layer.kda_conv_q)),
                            m.conv_state[il], kda_inner, 0, m.cq, stream,
                            (il == 0 && std::getenv("LING3_DUMP") != nullptr) ? m.kernel_debug : nullptr);
      if (il == 0) {
        dump_at("/work/p_convq_%u.bin", m.kernel_debug, 32, position);
      }
      hip::LaunchSilu(m.cq, m.cq, kda_inner, stream);
      if (il == 0 && std::getenv("LING3_DUMP") != nullptr) {
        std::vector<float> host(kda_inner);
        hipMemcpy(host.data(), m.cq, kda_inner * sizeof(float), hipMemcpyDeviceToHost);
        FILE* file = std::fopen("/work/ling3_cq0.bin", "wb");
        if (file != nullptr) {
          std::fwrite(host.data(), sizeof(float), kda_inner, file);
          std::fclose(file);
        }
      }
      hip::LaunchConv1dStep(m.k, static_cast<const float*>(dev(layer.kda_conv_k)),
                            m.conv_state[il], kda_inner, 1, m.ck, stream,
                            (il == 0 && std::getenv("LING3_DUMP") != nullptr) ? m.kernel_debug : nullptr);
      if (il == 0) {
        dump_at("/work/p_convk_%u.bin", m.kernel_debug, 32, position);
      }
      hip::LaunchSilu(m.ck, m.ck, kda_inner, stream);
      hip::LaunchConv1dStep(m.v, static_cast<const float*>(dev(layer.kda_conv_v)),
                            m.conv_state[il], kda_inner, 2, m.cv, stream,
                            (il == 0 && std::getenv("LING3_DUMP") != nullptr) ? m.kernel_debug : nullptr);
      if (il == 0) {
        dump_at("/work/p_convv_%u.bin", m.kernel_debug, 32, position);
      }
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
      if (il == 0) {
        dump_at("/work/p_ck_%u.bin", m.ck, kda_inner, position);
        dump_at("/work/p_cv_%u.bin", m.cv, kda_inner, position);
        dump_at("/work/p_qn_%u.bin", m.qn, kda_inner, position);
        dump_at("/work/p_kn_%u.bin", m.kn, kda_inner, position);
        dump_at("/work/p_gate_%u.bin", m.kda_gate, kda_inner, position);
        dump_at("/work/p_beta_%u.bin", m.beta, heads, position);
      }
      // `cq` is free after the L2 normalisation; the delta step reads `kn`
      // while writing, so it needs a different buffer.
      mark(il, 4);
      hip::LaunchDeltaRuleStep(m.qn, m.kn, m.cv, m.kda_gate, m.beta,
                               m.delta_state[il], m.cq, heads, kda_dim, stream,
                               (il == 0 && std::getenv("LING3_DUMP") != nullptr)
                                   ? m.kernel_debug
                                   : nullptr);
      if (il == 0) {
        dump_at("/work/p_kernel_%u.bin", m.kernel_debug, 8 * kda_dim, position);
      }
      if (il == 0) {
        dump_at("/work/p_delta_%u.bin", m.cq, kda_inner, position);
        dump_at("/work/p_state_%u.bin", m.delta_state[il],
                heads * kda_dim * kda_dim, position);
      }
      mark(il, 5);
      hip::LaunchRmsNorm(m.cq, static_cast<const float*>(dev(layer.kda_o_norm)), m.qn,
                         heads, kda_dim, eps, stream);
      if (il == 0) {
        dump_raw("/work/d_onorm.bin", m.qn, kda_inner);
      }
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
      if (il == 0) {
        dump_raw("/work/d0_onorm_gated.bin", m.qn, kda_inner);
        dump_raw("/work/d0_attnout.bin", m.attn_out, n_embd);
      }
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
      if (il == 5) {
        dump_raw("/work/d5_h.bin", m.h, n_embd);
        dump_raw("/work/d5_q.bin", m.q, q_dim);
        dump_raw("/work/d5_kvq.bin", m.kvq, latent + c.n_rot);
        dump_raw("/work/d5_latnorm.bin", m.latent, latent);
        dump_raw("/work/d5_gate.bin", m.beta_pre, heads);
      }
      hip::LaunchMlaAbsorbQ(m.qn, m.k_absorb[il], m.q_abs, heads, stream);
      if (il == 5) {
        dump_raw("/work/d5_qabs.bin", m.q_abs, heads * latent);
        dump_raw("/work/d5_cache.bin", m.mla_cache[il], latent + c.n_rot);
      }
      hip::LaunchMlaScores(m.q_abs, m.q_pe, m.mla_cache[il], heads, position + 1,
                           m.scores, stream);
      if (il == 5) {
        dump_raw("/work/d5_scores.bin", m.scores, heads * (position + 1));
      }
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
        dump_raw("/work/d5_qnope.bin", m.qn, heads * c.NopeDim());
        dump_raw("/work/d5_qpe.bin", m.q_pe, heads * c.n_rot);
        dump_raw("/work/d5_mlaout.bin", m.mla_out, v_dim);
      }
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

    mark(il, 1);
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
      if (il == 2) {
        dump_raw("/work/d2_ffnh.bin", m.ffn_h, n_embd);
        dump_raw("/work/d2_logits.bin", m.moe_logits, c.n_expert);
        dump_raw("/work/d2_ids.bin", reinterpret_cast<const float*>(m.moe_ids),
                 c.n_expert_used * 0 + 8);
      }
      hip::LaunchMoeGather(dev(layer.gate_exps), Storage(layer.gate_exps.type),
                           dev(layer.up_exps), Storage(layer.up_exps.type),
                           dev(layer.down_exps), Storage(layer.down_exps.type),
                           n_embd, c.n_ff_exp, m.ffn_h, m.moe_ids, m.moe_weights,
                           c.n_expert_used, m.x, m.expert_gate, m.expert_up,
                           m.expert_down, stream);
      if (il == 2) {
        dump_raw("/work/d2_after_gather.bin", m.x, n_embd);
      }
      hip::LaunchFfn(dev(layer.gate_shexp), Storage(layer.gate_shexp.type),
                     dev(layer.up_shexp), Storage(layer.up_shexp.type),
                     dev(layer.down_shexp), Storage(layer.down_shexp.type), n_embd,
                     c.n_ff_shexp, m.ffn_h, 1.0f, m.x, m.expert_gate, m.expert_up,
                     m.expert_down, stream);
      if (il == 2) {
        dump_raw("/work/d2_after_shared.bin", m.x, n_embd);
      }
      if (il == 31) {
        report_buf("x after shared", m.x, n_embd);
      }
    }
    dump_lp("/work/x_%u_%u.bin", m.x, n_embd, il, position);
    if (il == 0) {
      dump_raw("/work/d0_xout.bin", m.x, n_embd);
      dump_raw("/work/d0_ffnh.bin", m.ffn_h, n_embd);
    }
    report("block", il);
    mark(il, 2);
  }

  report("final", c.n_layer);
  hip::LaunchRmsNorm(m.x, static_cast<const float*>(dev(weights.output_norm)), m.h, 1,
                     n_embd, eps, stream);
  if (Storage(weights.output.type) == hip::kQ4K ||
      Storage(weights.output.type) == hip::kQ6K) {
    hip::LaunchQuantizeActivations(m.h, n_embd, m.act_values, m.act_scales, stream);
    hip::LaunchGemvPrepared(dev(weights.output), Storage(weights.output.type), c.n_vocab,
                            n_embd, m.act_values, m.act_scales, m.logits, stream);
  } else {
    hip::LaunchGemv(dev(weights.output), Storage(weights.output.type), c.n_vocab,
                    n_embd, m.h, m.logits, stream);
  }
  hip::LaunchArgmax(m.logits, c.n_vocab, m.argmax, stream);
  dump_at("/work/logits_%u.bin", m.logits, c.n_vocab, position);

  const hipError_t status = hipStreamSynchronize(stream);
  if (profile) {
    // Attention spans marks 0->1; everything else in the layer (norm, residual, FFN
    // or MoE, next layer's prologue) spans 1 -> the next layer's mark 0.
    double attention_ms = 0.0;
    double rest_ms = 0.0;
    float elapsed = 0.0f;
    double ffn_ms = 0.0;
    double proj_ms = 0.0;
    double conv_ms = 0.0;
    double delta_ms = 0.0;
    double out_ms = 0.0;
    for (std::uint32_t il = 0; il < c.n_layer; ++il) {
      hipEventElapsedTime(&elapsed, profile_events[il * 8], profile_events[il * 8 + 1]);
      attention_ms += elapsed;
      hipEventElapsedTime(&elapsed, profile_events[il * 8 + 1], profile_events[il * 8 + 2]);
      ffn_ms += elapsed;
      const hipEvent_t next = il + 1 < c.n_layer ? profile_events[(il + 1) * 8]
                                                 : profile_events[il * 8 + 2];
      hipEventElapsedTime(&elapsed, profile_events[il * 8 + 2], next);
      rest_ms += elapsed;
      if (c.IsKda(il)) {
        hipEventElapsedTime(&elapsed, profile_events[il * 8], profile_events[il * 8 + 3]);
        proj_ms += elapsed;
        hipEventElapsedTime(&elapsed, profile_events[il * 8 + 3], profile_events[il * 8 + 4]);
        conv_ms += elapsed;
        hipEventElapsedTime(&elapsed, profile_events[il * 8 + 4], profile_events[il * 8 + 5]);
        delta_ms += elapsed;
        hipEventElapsedTime(&elapsed, profile_events[il * 8 + 5], profile_events[il * 8 + 1]);
        out_ms += elapsed;
      }
    }
    static long tokens_profiled = 0;
    static double attention_total = 0.0;
    static double rest_total = 0.0;
    static double ffn_total = 0.0;
    static double proj_total = 0.0;
    static double conv_total = 0.0;
    static double delta_total = 0.0;
    static double out_total = 0.0;
    ++tokens_profiled;
    attention_total += attention_ms;
    rest_total += rest_ms;
    ffn_total += ffn_ms;
    proj_total += proj_ms;
    conv_total += conv_ms;
    delta_total += delta_ms;
    out_total += out_ms;
    if (tokens_profiled % 8 == 0) {
      std::fprintf(stderr,
                   "profile: %ld tokens, attention %.2f (kda proj %.2f, conv/gate %.2f, "
                   "delta %.2f, norm/out %.2f), ffn/moe %.2f, other %.2f, total %.2f "
                   "ms/token\n",
                   tokens_profiled, attention_total / tokens_profiled,
                   proj_total / tokens_profiled, conv_total / tokens_profiled,
                   delta_total / tokens_profiled, out_total / tokens_profiled,
                   ffn_total / tokens_profiled, rest_total / tokens_profiled,
                   (attention_total + ffn_total + rest_total) / tokens_profiled);
    }
    for (auto& event : profile_events) {
      hipEventDestroy(event);
    }
  }
  if (status != hipSuccess) {
    if (error != nullptr) {
      *error = std::string("engine: forward failed: ") + hipGetErrorString(status);
    }
    return false;
  }
  return true;
}

void Engine::SnapshotState() {
  Impl& m = *impl_;
  if (m.snapshot == nullptr) {
    return;
  }
  std::size_t offset = 0;
  const auto copy_in = [&](const float* source, std::size_t count) {
    hipMemcpyAsync(m.snapshot + offset, source, count * sizeof(float),
                   hipMemcpyDeviceToDevice, m.queue);
    offset += count;
  };
  for (std::uint32_t il = 0; il < m.config.n_layer_total; ++il) {
    if (m.config.IsKda(il)) {
      const std::size_t conv = 9 * m.config.KdaInner();
      const std::size_t delta = static_cast<std::size_t>(m.config.n_head) *
                                m.config.n_kda_head_dim * m.config.n_kda_head_dim;
      copy_in(m.conv_state[il], conv);
      copy_in(m.delta_state[il], delta);
    } else {
      copy_in(m.mla_cache[il], m.options.max_context * (m.config.n_lora_kv + m.config.n_rot));
    }
  }
  m.snapshot_floats = offset;
  hipStreamSynchronize(m.queue);
}

void Engine::RestoreState() {
  Impl& m = *impl_;
  if (m.snapshot == nullptr || m.snapshot_floats == 0) {
    return;
  }
  std::size_t offset = 0;
  const auto copy_out = [&](float* target, std::size_t count) {
    hipMemcpyAsync(target, m.snapshot + offset, count * sizeof(float),
                   hipMemcpyDeviceToDevice, m.queue);
    offset += count;
  };
  for (std::uint32_t il = 0; il < m.config.n_layer_total; ++il) {
    if (m.config.IsKda(il)) {
      const std::size_t conv = 9 * m.config.KdaInner();
      const std::size_t delta = static_cast<std::size_t>(m.config.n_head) *
                                m.config.n_kda_head_dim * m.config.n_kda_head_dim;
      copy_out(m.conv_state[il], conv);
      copy_out(m.delta_state[il], delta);
    } else {
      copy_out(m.mla_cache[il], m.options.max_context * (m.config.n_lora_kv + m.config.n_rot));
    }
  }
  hipStreamSynchronize(m.queue);
}

bool Engine::VerifyDrafts(const std::int32_t* tokens, std::size_t count,
                          std::uint32_t start_position,
                          std::int32_t* argmax_per_row, std::string* error) {
  Impl& m = *impl_;
  const std::size_t n_embd = m.config.n_embd;
  if (count == 0 || count > 8) {
    if (error != nullptr) *error = "engine: verification batch must be 1..8";
    return false;
  }
  SnapshotState();
  // Stop after the last trunk block: the chunk buffer then holds each row's final
  // hidden state, which the head turns into per-row logits.
  if (!PrefillChunk(tokens, count, start_position,
                    static_cast<std::int32_t>(m.config.n_layer) - 1, error)) {
    return false;
  }
  hip::LaunchGemm(m.Dev(m.weights.output), Storage(m.weights.output.type),
                  m.config.n_vocab, n_embd, m.bx, count, n_embd, m.blogits,
                  m.config.n_vocab, m.queue);
  for (std::size_t row = 0; row < count; ++row) {
    hip::LaunchRmsNorm(m.bx + row * n_embd,
                       static_cast<const float*>(m.Dev(m.weights.output_norm)), m.h, 1,
                       n_embd, m.config.rms_eps, m.queue);
    hip::LaunchGemv(m.Dev(m.weights.output), Storage(m.weights.output.type),
                    m.config.n_vocab, n_embd, m.h, m.logits, m.queue);
    hip::LaunchArgmax(m.logits, m.config.n_vocab, m.argmax, m.queue);
    hipStreamSynchronize(m.queue);
    std::int32_t value = -1;
    hipMemcpy(&value, m.argmax, sizeof(value), hipMemcpyDeviceToHost);
    argmax_per_row[row] = value;
  }
  return true;
}

bool Engine::PrefillChunk(const std::int32_t* tokens, std::size_t count,
                          std::uint32_t start_position,
                          std::int32_t stop_after_block, std::string* error) {
  Impl& m = *impl_;
  const Ling3Config& c = m.config;
  const auto& weights = m.weights;
  hipStream_t stream = m.queue;
  const std::size_t n_embd = c.n_embd;
  const std::size_t heads = c.n_head;
  const std::size_t latent = c.n_lora_kv;
  const std::size_t kda_dim = c.n_kda_head_dim;
  const std::size_t kda_inner = c.KdaInner();
  const std::size_t q_dim = heads * c.n_head_dim_mla;
  const std::size_t v_dim = heads * c.n_head_dim_v_mla;
  const std::size_t kv_row = latent + c.n_rot;
  const float eps = c.rms_eps;
  auto dev = [&m](const Tensor& tensor) { return m.Dev(tensor); };
  const int M = static_cast<int>(count);
  if (count == 0 || count > static_cast<std::size_t>(Impl::kMaxChunk)) {
    if (error != nullptr) *error = "engine: chunk size out of range";
    return false;
  }

  for (int i = 0; i < M; ++i) {
    hip::LaunchEmbeddingRow(dev(weights.token_embd),
                            RawType(weights.token_embd.type), c.n_vocab, n_embd,
                            tokens[i], m.bx + static_cast<std::size_t>(i) * n_embd,
                            stream);
  }

  for (std::uint32_t il = 0; il < c.n_layer; ++il) {
    const LayerWeights& layer = weights.layers[il];
    const bool kda = c.IsKda(il);
    hip::LaunchRmsNorm(m.bx, static_cast<const float*>(dev(layer.attn_norm)), m.bh, M,
                       n_embd, eps, stream);
    if (kda) {
      hip::LaunchGemm(dev(layer.kda_q), Storage(layer.kda_q.type), kda_inner, n_embd,
                      m.bh, M, n_embd, m.bq, kda_inner, stream);
      hip::LaunchGemm(dev(layer.kda_k), Storage(layer.kda_k.type), kda_inner, n_embd,
                      m.bh, M, n_embd, m.bk, kda_inner, stream);
      hip::LaunchGemm(dev(layer.kda_v), Storage(layer.kda_v.type), kda_inner, n_embd,
                      m.bh, M, n_embd, m.bv, kda_inner, stream);
    } else {
      hip::LaunchGemm(dev(layer.mla_q), Storage(layer.mla_q.type), q_dim, n_embd, m.bh,
                      M, n_embd, m.bq, q_dim, stream);
      hip::LaunchGemm(dev(layer.mla_kv_a), Storage(layer.mla_kv_a.type), kv_row, n_embd,
                      m.bh, M, n_embd, m.bk, kv_row, stream);
    }

    for (int i = 0; i < M; ++i) {
      const std::size_t row = static_cast<std::size_t>(i) * n_embd;
      const std::uint32_t position = start_position + static_cast<std::uint32_t>(i);
      if (kda) {
        const float* q_row = m.bq + static_cast<std::size_t>(i) * kda_inner;
        const float* k_row = m.bk + static_cast<std::size_t>(i) * kda_inner;
        const float* v_row = m.bv + static_cast<std::size_t>(i) * kda_inner;
        hip::LaunchConv1dStep(q_row, static_cast<const float*>(dev(layer.kda_conv_q)),
                              m.conv_state[il], kda_inner, 0, m.cq, stream);
        hip::LaunchSilu(m.cq, m.cq, kda_inner, stream);
        hip::LaunchConv1dStep(k_row, static_cast<const float*>(dev(layer.kda_conv_k)),
                              m.conv_state[il], kda_inner, 1, m.ck, stream);
        hip::LaunchSilu(m.ck, m.ck, kda_inner, stream);
        hip::LaunchConv1dStep(v_row, static_cast<const float*>(dev(layer.kda_conv_v)),
                              m.conv_state[il], kda_inner, 2, m.cv, stream);
        hip::LaunchSilu(m.cv, m.cv, kda_inner, stream);
        hip::LaunchGemv(dev(layer.kda_f_a), Storage(layer.kda_f_a.type), kda_inner,
                        n_embd, m.bh + row, m.gate_pre, stream);
        hip::LaunchKdaGate(m.gate_pre, static_cast<const float*>(dev(layer.kda_dt)),
                           static_cast<const float*>(dev(layer.kda_a)), m.kda_gate,
                           kda_inner, kda_dim, c.kda_gate_lower_bound, stream);
        hip::LaunchGemv(dev(layer.kda_beta), Storage(layer.kda_beta.type), heads, n_embd,
                        m.bh + row, m.beta_pre, stream);
        hip::LaunchSigmoid(m.beta_pre, m.beta, heads, stream);
        hip::LaunchL2Norm(m.cq, m.qn, heads, kda_dim, eps, stream);
        hip::LaunchL2Norm(m.ck, m.kn, heads, kda_dim, eps, stream);
      hip::LaunchDeltaRuleStep(m.qn, m.kn, m.cv, m.kda_gate, m.beta,
                                 m.delta_state[il], m.cq, heads, kda_dim, stream);
      hip::LaunchRmsNorm(m.cq, static_cast<const float*>(dev(layer.kda_o_norm)), m.qn,
                           heads, kda_dim, eps, stream);
        hip::LaunchGemv(dev(layer.kda_g_a), Storage(layer.kda_g_a.type), kda_inner,
                        n_embd, m.bh + row, m.gate_pre, stream);
        hip::LaunchMulSigmoid(m.qn, m.gate_pre, kda_inner, stream);
        hipMemcpyAsync(m.bgated + static_cast<std::size_t>(i) * kda_inner, m.qn,
                       kda_inner * sizeof(float), hipMemcpyDeviceToDevice, stream);
      } else {
        const float* q_row = m.bq + static_cast<std::size_t>(i) * q_dim;
        float* kv_ptr = m.bk + static_cast<std::size_t>(i) * kv_row;
        hip::LaunchSplitQ(q_row, m.qn, m.q_pe, heads, c.NopeDim(), c.n_rot, stream);
        hip::LaunchRmsNorm(kv_ptr, static_cast<const float*>(dev(layer.mla_kv_a_norm)),
                           m.latent, 1, latent, eps, stream);
        hip::LaunchRope(m.q_pe, heads, c.n_rot, c.n_rot, position, c.rope_freq_base, true,
                        stream);
        hip::LaunchRope(kv_ptr + latent, 1, c.n_rot, c.n_rot, position,
                        c.rope_freq_base, true, stream);
        hip::LaunchMlaAppend(m.latent, kv_ptr + latent, m.mla_cache[il], position, latent,
                             c.n_rot, stream);
        hip::LaunchGemv(dev(layer.mla_gate), Storage(layer.mla_gate.type), heads, n_embd,
                        m.bh + row, m.beta_pre, stream);
        hip::LaunchMlaAbsorbQ(m.qn, m.k_absorb[il], m.q_abs, heads, stream);
        hip::LaunchMlaScores(m.q_abs, m.q_pe, m.mla_cache[il], heads, position + 1,
                             m.scores, stream);
        hip::LaunchMlaSoftmax(m.scores, heads, position + 1, stream);
        hip::LaunchMlaLatentSum(m.scores, m.mla_cache[il], heads, position + 1, m.latent,
                                stream);
        hip::LaunchMlaValueAbsorb(m.latent, m.v_absorb[il], m.beta_pre, m.mla_out, heads,
                                  stream);
        hipMemcpyAsync(m.bgated + static_cast<std::size_t>(i) * v_dim, m.mla_out,
                       v_dim * sizeof(float), hipMemcpyDeviceToDevice, stream);
      }
    }

    const Tensor& out_weight = kda ? layer.kda_out : layer.mla_out;
    const std::size_t inner = kda ? kda_inner : v_dim;
    hip::LaunchGemm(dev(out_weight), Storage(out_weight.type), n_embd, inner, m.bgated,
                    M, inner, m.battn, n_embd, stream);
    for (int i = 0; i < M; ++i) {
      const std::size_t row = static_cast<std::size_t>(i) * n_embd;
      hip::LaunchAxpy(m.bx + row, m.battn + row, 1.0f, n_embd, stream);
    }
    if (stop_after_block >= 0 && static_cast<std::int32_t>(il) == stop_after_block) {
      hipStreamSynchronize(stream);
      return true;
    }
    hip::LaunchRmsNorm(m.bx, static_cast<const float*>(dev(layer.ffn_norm)), m.bffnh, M,
                       n_embd, eps, stream);
    if (layer.dense) {
      hip::LaunchGemm(dev(layer.ffn_gate), Storage(layer.ffn_gate.type), c.n_ff, n_embd,
                      m.bffnh, M, n_embd, m.bffa, c.n_ff, stream);
      hip::LaunchGemm(dev(layer.ffn_up), Storage(layer.ffn_up.type), c.n_ff, n_embd,
                      m.bffnh, M, n_embd, m.bffb, c.n_ff, stream);
      hip::LaunchSilu(m.bffa, m.bffa, M * c.n_ff, stream);
      hip::LaunchMulInto(m.bffa, m.bffb, m.bffa, M * c.n_ff, stream);
      hip::LaunchGemm(dev(layer.ffn_down), Storage(layer.ffn_down.type), n_embd, c.n_ff,
                      m.bffa, M, c.n_ff, m.bffout, n_embd, stream);
      for (int i = 0; i < M; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * n_embd;
        hip::LaunchAxpy(m.bx + row, m.bffout + row, 1.0f, n_embd, stream);
      }
    } else {
      for (int i = 0; i < M; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * n_embd;
        hipMemcpyAsync(m.ffn_h, m.bffnh + row, n_embd * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
        hip::LaunchGemv(dev(layer.gate_inp), Storage(layer.gate_inp.type), c.n_expert,
                        n_embd, m.ffn_h, m.moe_logits, stream);
        hip::LaunchRouterSelect(m.moe_logits,
                                static_cast<const float*>(dev(layer.exp_probs_b)),
                                c.n_expert, c.n_group, c.n_group_used, c.n_expert_used,
                                c.expert_weights_scale, m.moe_ids, m.moe_weights, stream);
        hip::LaunchMoeGather(dev(layer.gate_exps), Storage(layer.gate_exps.type),
                             dev(layer.up_exps), Storage(layer.up_exps.type),
                             dev(layer.down_exps), Storage(layer.down_exps.type), n_embd,
                             c.n_ff_exp, m.ffn_h, m.moe_ids, m.moe_weights,
                             c.n_expert_used, m.bx + row, m.expert_gate, m.expert_up,
                             m.expert_down, stream);
      }
      hip::LaunchGemm(dev(layer.gate_shexp), Storage(layer.gate_shexp.type),
                      c.n_ff_shexp, n_embd, m.bffnh, M, n_embd, m.bffa, c.n_ff_shexp,
                      stream);
      hip::LaunchGemm(dev(layer.up_shexp), Storage(layer.up_shexp.type), c.n_ff_shexp,
                      n_embd, m.bffnh, M, n_embd, m.bffb, c.n_ff_shexp, stream);
      hip::LaunchSilu(m.bffa, m.bffa, M * c.n_ff_shexp, stream);
      hip::LaunchMulInto(m.bffa, m.bffb, m.bffa, M * c.n_ff_shexp, stream);
      hip::LaunchGemm(dev(layer.down_shexp), Storage(layer.down_shexp.type), n_embd,
                      c.n_ff_shexp, m.bffa, M, c.n_ff_shexp, m.bffout, n_embd, stream);
      for (int i = 0; i < M; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * n_embd;
        hip::LaunchAxpy(m.bx + row, m.bffout + row, 1.0f, n_embd, stream);
      }
    }
  }

  const std::size_t last = static_cast<std::size_t>(M - 1) * n_embd;
  hip::LaunchRmsNorm(m.bx + last, static_cast<const float*>(dev(weights.output_norm)),
                     m.h, 1, n_embd, eps, stream);
  if (Storage(weights.output.type) == hip::kQ4K ||
      Storage(weights.output.type) == hip::kQ6K) {
    hip::LaunchQuantizeActivations(m.h, n_embd, m.act_values, m.act_scales, stream);
    hip::LaunchGemvPrepared(dev(weights.output), Storage(weights.output.type), c.n_vocab,
                            n_embd, m.act_values, m.act_scales, m.logits, stream);
  } else {
    hip::LaunchGemv(dev(weights.output), Storage(weights.output.type), c.n_vocab, n_embd,
                    m.h, m.logits, stream);
  }
  hip::LaunchArgmax(m.logits, c.n_vocab, m.argmax, stream);
  const hipError_t status = hipStreamSynchronize(stream);

  if (status != hipSuccess) {
    if (error != nullptr) {
      *error = std::string("engine: chunked prefill failed: ") + hipGetErrorString(status);
    }
    return false;
  }
  return true;
}

void Engine::ReadChunkActivations(std::size_t rows, float* host_rows) {
  hipMemcpy(host_rows, impl_->bx, rows * impl_->config.n_embd * sizeof(float),
            hipMemcpyDeviceToHost);
}

bool Engine::ForwardMtp(std::int32_t token, std::uint32_t position,
                        std::string* error) {
  Impl& m = *impl_;
  const Ling3Config& c = m.config;
  const auto& weights = m.weights;
  const LayerWeights& layer = weights.layers[c.n_layer];  // block 42
  hipStream_t stream = m.queue;
  const std::size_t n_embd = c.n_embd;
  const std::size_t heads = c.n_head;
  const std::size_t latent = c.n_lora_kv;
  const std::size_t v_dim = heads * c.n_head_dim_v_mla;
  const float eps = c.rms_eps;
  auto dev = [&m](const Tensor& tensor) { return m.Dev(tensor); };

  // m.h holds the base model's normalised final hidden state; keep it, the
  // block's scratch reuses m.h below.
  hipMemcpyAsync(m.base_hidden, m.h, n_embd * sizeof(float),
                 hipMemcpyDeviceToDevice, stream);
  hip::LaunchRmsNorm(m.base_hidden, static_cast<const float*>(dev(layer.nextn_hnorm)),
                     m.qn, 1, n_embd, eps, stream);
  hip::LaunchEmbeddingRow(dev(weights.token_embd), RawType(weights.token_embd.type),
                          c.n_vocab, n_embd, token, m.kn, stream);
  hip::LaunchRmsNorm(m.kn, static_cast<const float*>(dev(layer.nextn_enorm)), m.kn, 1,
                     n_embd, eps, stream);
  hipMemcpyAsync(m.q, m.kn, n_embd * sizeof(float), hipMemcpyDeviceToDevice, stream);
  hipMemcpyAsync(m.q + n_embd, m.qn, n_embd * sizeof(float),
                 hipMemcpyDeviceToDevice, stream);
  hip::LaunchGemv(dev(layer.nextn_eh_proj), Storage(layer.nextn_eh_proj.type), n_embd,
                  2 * n_embd, m.q, m.x, stream);

  hip::LaunchRmsNorm(m.x, static_cast<const float*>(dev(layer.attn_norm)), m.h, 1,
                     n_embd, eps, stream);
  // Same MLA block as the trunk, on block 42's own weights and cache.
  hip::LaunchGemv(dev(layer.mla_q), Storage(layer.mla_q.type),
                  heads * c.n_head_dim_mla, n_embd, m.h, m.q, stream);
  hip::LaunchSplitQ(m.q, m.qn, m.q_pe, heads, c.NopeDim(), c.n_rot, stream);
  hip::LaunchGemv(dev(layer.mla_kv_a), Storage(layer.mla_kv_a.type),
                  latent + c.n_rot, n_embd, m.h, m.kvq, stream);
  hip::LaunchRmsNorm(m.kvq, static_cast<const float*>(dev(layer.mla_kv_a_norm)),
                     m.latent, 1, latent, eps, stream);
  hip::LaunchRope(m.q_pe, heads, c.n_rot, c.n_rot, position, c.rope_freq_base, true,
                  stream);
  hip::LaunchRope(m.kvq + latent, 1, c.n_rot, c.n_rot, position, c.rope_freq_base,
                  true, stream);
  hip::LaunchMlaAppend(m.latent, m.kvq + latent, m.mla_cache[c.n_layer], position,
                       latent, c.n_rot, stream);
  hip::LaunchGemv(dev(layer.mla_gate), Storage(layer.mla_gate.type), heads, n_embd,
                  m.h, m.beta_pre, stream);
  hip::LaunchMlaAbsorbQ(m.qn, m.k_absorb[c.n_layer], m.q_abs, heads, stream);
  hip::LaunchMlaScores(m.q_abs, m.q_pe, m.mla_cache[c.n_layer], heads, position + 1,
                       m.scores, stream);
  hip::LaunchMlaSoftmax(m.scores, heads, position + 1, stream);
  hip::LaunchMlaLatentSum(m.scores, m.mla_cache[c.n_layer], heads, position + 1,
                          m.latent, stream);
  hip::LaunchMlaValueAbsorb(m.latent, m.v_absorb[c.n_layer], m.beta_pre, m.mla_out,
                            heads, stream);
  hip::LaunchGemv(dev(layer.mla_out), Storage(layer.mla_out.type), n_embd, v_dim,
                  m.mla_out, m.attn_out, stream);

  hip::LaunchAxpy(m.x, m.attn_out, 1.0f, n_embd, stream);
  hipMemcpyAsync(m.ffn_inp, m.x, n_embd * sizeof(float), hipMemcpyDeviceToDevice,
                 stream);
  hip::LaunchRmsNorm(m.ffn_inp, static_cast<const float*>(dev(layer.ffn_norm)),
                     m.ffn_h, 1, n_embd, eps, stream);
  hip::LaunchGemv(dev(layer.gate_inp), Storage(layer.gate_inp.type), c.n_expert,
                  n_embd, m.ffn_h, m.moe_logits, stream);
  hip::LaunchRouterSelect(m.moe_logits,
                          static_cast<const float*>(dev(layer.exp_probs_b)),
                          c.n_expert, c.n_group, c.n_group_used, c.n_expert_used,
                          c.expert_weights_scale, m.moe_ids, m.moe_weights, stream);
  hip::LaunchMoeGather(dev(layer.gate_exps), Storage(layer.gate_exps.type),
                       dev(layer.up_exps), Storage(layer.up_exps.type),
                       dev(layer.down_exps), Storage(layer.down_exps.type), n_embd,
                       c.n_ff_exp, m.ffn_h, m.moe_ids, m.moe_weights,
                       c.n_expert_used, m.x, m.expert_gate, m.expert_up,
                       m.expert_down, stream);
  hip::LaunchFfn(dev(layer.gate_shexp), Storage(layer.gate_shexp.type),
                 dev(layer.up_shexp), Storage(layer.up_shexp.type),
                 dev(layer.down_shexp), Storage(layer.down_shexp.type), n_embd,
                 c.n_ff_shexp, m.ffn_h, 1.0f, m.x, m.expert_gate, m.expert_up,
                 m.expert_down, stream);

  hip::LaunchRmsNorm(m.x, static_cast<const float*>(dev(layer.nextn_shared_head_norm)),
                     m.h, 1, n_embd, eps, stream);
  hip::LaunchGemv(dev(weights.output), Storage(weights.output.type), c.n_vocab,
                  n_embd, m.h, m.logits, stream);
  hip::LaunchArgmax(m.logits, c.n_vocab, m.argmax, stream);
  const hipError_t status = hipStreamSynchronize(stream);

  if (status != hipSuccess) {
    if (error != nullptr) {
      *error = std::string("engine: MTP forward failed: ") + hipGetErrorString(status);
    }
    return false;
  }
  return true;
}

void Engine::DebugChunkStage1(const std::int32_t* tokens, std::size_t count,
                              float* host_x, float* host_h) {
  Impl& m = *impl_;
  const std::size_t n_embd = m.config.n_embd;
  const std::size_t rows = count < 32 ? count : 32;
  for (std::size_t i = 0; i < rows; ++i) {
    hip::LaunchEmbeddingRow(m.Dev(m.weights.token_embd),
                            RawType(m.weights.token_embd.type), m.config.n_vocab,
                            n_embd, tokens[i],
                            m.bx + i * n_embd, m.queue);
  }
  const auto& layer = m.weights.layers[0];
  hip::LaunchRmsNorm(m.bx, static_cast<const float*>(m.Dev(layer.attn_norm)), m.bh,
                     rows, n_embd, m.config.rms_eps, m.queue);
  hipStreamSynchronize(m.queue);
  hipMemcpy(host_x, m.bx, rows * n_embd * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(host_h, m.bh, rows * n_embd * sizeof(float), hipMemcpyDeviceToHost);
}

float Engine::DebugChunkProjectionDiff(std::size_t rows) {
  Impl& m = *impl_;
  const Ling3Config& c = m.config;
  const std::size_t n_embd = c.n_embd;
  const std::size_t inner = c.KdaInner();
  const auto& layer = m.weights.layers[0];
  // Reuse whatever the chunk buffer already holds as the input rows; the caller
  // fills it through DebugChunkStage1.
  hip::LaunchGemm(m.Dev(layer.kda_q), Storage(layer.kda_q.type), inner, n_embd, m.bx,
                  rows, n_embd, m.bq, inner, m.queue);
  std::vector<float> batched(rows * inner), single(inner);
  hipStreamSynchronize(m.queue);
  hipMemcpy(batched.data(), m.bq, rows * inner * sizeof(float), hipMemcpyDeviceToHost);
  float worst = 0.0f;
  for (std::size_t i = 0; i < rows; ++i) {
    hip::LaunchGemv(m.Dev(layer.kda_q), Storage(layer.kda_q.type), inner, n_embd,
                    m.bx + i * n_embd, m.k, m.queue);
    hipStreamSynchronize(m.queue);
    hipMemcpy(single.data(), m.k, inner * sizeof(float), hipMemcpyDeviceToHost);
    for (std::size_t j = 0; j < inner; ++j) {
      const float diff = std::fabs(batched[i * inner + j] - single[j]);
      worst = diff > worst ? diff : worst;
    }
  }
  return worst;
}

void Engine::DebugEmbedding(std::int32_t token, float* host_row) {
  Impl& m = *impl_;
  const std::size_t n_embd = m.config.n_embd;
  hip::LaunchEmbeddingRow(m.Dev(m.weights.token_embd),
                          RawType(m.weights.token_embd.type), m.config.n_vocab,
                          n_embd, token, m.x, m.queue);
  hipStreamSynchronize(m.queue);
  hipMemcpy(host_row, m.x, n_embd * sizeof(float), hipMemcpyDeviceToHost);
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
