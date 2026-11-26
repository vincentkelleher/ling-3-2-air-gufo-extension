#include "src/vision.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "src/hip/vision.hpp"

namespace gufo::ext::ling3_air {
namespace {

using hip::LaunchAddInto;
using hip::LaunchGelu;
using hip::LaunchLayerNorm;
using hip::LaunchMatvecInMajor;
using hip::LaunchMatvecInMajorBatch;
using hip::LaunchVisionAttention;

struct TensorRef {
  float* device{nullptr};
  std::size_t count{0};
};

/// Copies one mmproj tensor to the device as F32, converting BF16 and F32.
bool Upload(const core::GgufReader& reader, const std::string& name,
            TensorRef* out, std::string* error) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr) {
    if (error != nullptr) *error = "vision: missing tensor " + name;
    return false;
  }
  const std::size_t count = info->ElementCount();
  std::vector<float> host(count);
  if (info->type == core::GgmlType::kF32) {
    std::memcpy(host.data(), info->data, count * sizeof(float));
  } else if (info->type == core::GgmlType::kBF16) {
    const auto* raw = static_cast<const std::uint16_t*>(info->data);
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16;
      std::memcpy(&host[i], &bits, sizeof(bits));
    }
  } else {
    if (error != nullptr) {
      *error = "vision: tensor " + name + " is " +
               std::string(core::ToString(info->type)) + ", expected BF16 or F32";
    }
    return false;
  }
  const hipError_t allocated =
      hipMalloc(reinterpret_cast<void**>(&out->device), count * sizeof(float));
  if (allocated != hipSuccess) {
    if (error != nullptr) {
      *error = "vision: cannot allocate " + name + " (" +
               std::to_string(count * sizeof(float) / 1048576) + " MiB: " +
               hipGetErrorString(allocated) + ")";
    }
    return false;
  }
  if (hipMemcpy(out->device, host.data(), count * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    if (error != nullptr) *error = "vision: cannot upload " + name;
    return false;
  }
  out->count = count;
  return true;
}

struct BlockWeights {
  TensorRef ln1_w, ln1_b, ln2_w, ln2_b;
  TensorRef qkv_w, qkv_b, out_w, out_b;
  TensorRef up_w, up_b, down_w, down_b;
};

}  // namespace

struct VisionEncoder::Impl {
  VisionConfig config;
  TensorRef patch0, patch1, patch_bias;
  std::vector<BlockWeights> blocks;
  // Scratch, sized for the largest patch sequence we run.
  float* x{nullptr};
  float* h{nullptr};
  float* qkv{nullptr};
  float* attn{nullptr};
  float* ffn{nullptr};
  std::size_t scratch_tokens{0};
  std::size_t device_bytes{0};

  bool EnsureScratch(std::size_t tokens, std::string* error) {
    if (tokens <= scratch_tokens) {
      return true;
    }
    for (float** pointer : {&x, &h, &qkv, &attn, &ffn}) {
      if (*pointer != nullptr) {
        hipFree(*pointer);
        *pointer = nullptr;
      }
    }
    const std::size_t dim = config.dim;
    const std::size_t qkv_dim = 3 * dim;
    const bool ok =
        hipMalloc(reinterpret_cast<void**>(&x), tokens * dim * sizeof(float)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&h), tokens * dim * sizeof(float)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&qkv), tokens * qkv_dim * sizeof(float)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&attn), tokens * dim * sizeof(float)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&ffn), tokens * config.ffn * sizeof(float)) == hipSuccess;
    if (!ok) {
      if (error != nullptr) *error = "vision: scratch allocation failed";
      return false;
    }
    scratch_tokens = tokens;
    device_bytes += tokens * (3 * dim + config.ffn) * sizeof(float);
    return true;
  }
};

VisionEncoder::~VisionEncoder() {
  if (!impl_) {
    return;
  }
  Impl& m = *impl_;
  for (float** pointer : {&m.x, &m.h, &m.qkv, &m.attn, &m.ffn}) {
    if (*pointer != nullptr) hipFree(*pointer);
  }
  for (const auto& block : m.blocks) {
    for (const TensorRef* ref : {&block.ln1_w, &block.ln1_b, &block.ln2_w, &block.ln2_b,
                                 &block.qkv_w, &block.qkv_b, &block.out_w, &block.out_b,
                                 &block.up_w, &block.up_b, &block.down_w, &block.down_b}) {
      if (ref->device != nullptr) hipFree(ref->device);
    }
  }
  for (const TensorRef* ref : {&m.patch0, &m.patch1, &m.patch_bias}) {
    if (ref->device != nullptr) hipFree(ref->device);
  }
}

std::unique_ptr<VisionEncoder> VisionEncoder::Create(const core::GgufReader& mmproj,
                                                     std::string* error) {
  auto fail = [&](const std::string& message) -> std::unique_ptr<VisionEncoder> {
    if (error != nullptr) *error = message;
    return nullptr;
  };
  const auto architecture = mmproj.GetMetadataString("general.architecture").value_or("");
  if (architecture != "clip") {
    return fail("vision: not an mmproj (architecture '" + std::string(architecture) + "')");
  }
  std::unique_ptr<VisionEncoder> encoder(new VisionEncoder());
  encoder->impl_ = std::make_unique<Impl>();
  Impl& m = *encoder->impl_;
  m.config.dim = mmproj.GetMetadataUint32("clip.vision.embedding_length").value_or(0);
  m.config.heads = mmproj.GetMetadataUint32("clip.vision.attention.head_count").value_or(0);
  m.config.ffn = mmproj.GetMetadataUint32("clip.vision.feed_forward_length").value_or(0);
  m.config.blocks = mmproj.GetMetadataUint32("clip.vision.block_count").value_or(0);
  m.config.patch = mmproj.GetMetadataUint32("clip.vision.patch_size").value_or(16);
  m.config.merge = mmproj.GetMetadataUint32("clip.vision.spatial_merge_size").value_or(2);
  if (const auto eps = mmproj.GetMetadataFloat32("clip.vision.attention.layer_norm_epsilon")) {
    m.config.eps = *eps;
  }
  if (m.config.dim == 0 || m.config.heads == 0 || m.config.ffn == 0 || m.config.blocks == 0) {
    return fail("vision: mmproj has no vision hparams");
  }

  std::string local;
  if (!Upload(mmproj, "v.patch_embd.weight", &m.patch0, &local) ||
      !Upload(mmproj, "v.patch_embd.weight.1", &m.patch1, &local) ||
      !Upload(mmproj, "v.patch_embd.bias", &m.patch_bias, &local)) {
    return fail(local);
  }
  m.device_bytes += m.patch0.count * sizeof(float) + m.patch1.count * sizeof(float) +
                    m.patch_bias.count * sizeof(float);

  m.blocks.resize(m.config.blocks);
  for (std::uint32_t b = 0; b < m.config.blocks; ++b) {
    BlockWeights& w = m.blocks[b];
    const std::string prefix = "v.blk." + std::to_string(b) + ".";
    const bool ok =
        Upload(mmproj, prefix + "ln1.weight", &w.ln1_w, &local) &&
        Upload(mmproj, prefix + "ln1.bias", &w.ln1_b, &local) &&
        Upload(mmproj, prefix + "ln2.weight", &w.ln2_w, &local) &&
        Upload(mmproj, prefix + "ln2.bias", &w.ln2_b, &local) &&
        Upload(mmproj, prefix + "attn_qkv.weight", &w.qkv_w, &local) &&
        Upload(mmproj, prefix + "attn_qkv.bias", &w.qkv_b, &local) &&
        Upload(mmproj, prefix + "attn_out.weight", &w.out_w, &local) &&
        Upload(mmproj, prefix + "attn_out.bias", &w.out_b, &local) &&
        Upload(mmproj, prefix + "ffn_up.weight", &w.up_w, &local) &&
        Upload(mmproj, prefix + "ffn_up.bias", &w.up_b, &local) &&
        Upload(mmproj, prefix + "ffn_down.weight", &w.down_w, &local) &&
        Upload(mmproj, prefix + "ffn_down.bias", &w.down_b, &local);
    if (!ok) {
      return fail(local);
    }
    for (const TensorRef* ref : {&w.ln1_w, &w.ln1_b, &w.ln2_w, &w.ln2_b, &w.qkv_w,
                                 &w.qkv_b, &w.out_w, &w.out_b, &w.up_w, &w.up_b,
                                 &w.down_w, &w.down_b}) {
      m.device_bytes += ref->count * sizeof(float);
    }
  }
  if (error != nullptr) error->clear();
  return encoder;
}

const VisionConfig& VisionEncoder::Config() const { return impl_->config; }

bool VisionEncoder::PatchEmbed(const float* patch0, const float* patch1, float* out,
                               std::string* error) {
  Impl& m = *impl_;
  const std::size_t in = static_cast<std::size_t>(m.config.patch) * m.config.patch * 3;
  const std::size_t dim = m.config.dim;
  float* device_p0 = nullptr;
  float* device_p1 = nullptr;
  float* device_a = nullptr;
  float* device_b = nullptr;
  const bool ok =
      hipMalloc(reinterpret_cast<void**>(&device_p0), in * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&device_p1), in * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&device_a), dim * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&device_b), dim * sizeof(float)) == hipSuccess;
  if (!ok) {
    if (error != nullptr) *error = "vision: patch-embed allocation failed";
    return false;
  }
  hipMemcpy(device_p0, patch0, in * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(device_p1, patch1, in * sizeof(float), hipMemcpyHostToDevice);
  LaunchMatvecInMajor(device_p0, m.patch0.device, nullptr, device_a, in, dim, nullptr);
  LaunchMatvecInMajor(device_p1, m.patch1.device, m.patch_bias.device, device_b, in, dim,
                      nullptr);
  LaunchAddInto(device_a, device_a, device_b, dim, nullptr);
  hipStreamSynchronize(nullptr);
  hipMemcpy(out, device_a, dim * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(device_p0);
  hipFree(device_p1);
  hipFree(device_a);
  hipFree(device_b);
  return true;
}

bool VisionEncoder::RunBlock(std::uint32_t block, float* x, std::size_t tokens,
                             std::string* error) {
  Impl& m = *impl_;
  if (block >= m.blocks.size()) {
    if (error != nullptr) *error = "vision: block out of range";
    return false;
  }
  const std::size_t dim = m.config.dim;
  const std::size_t rows = tokens * dim;
  // Keep the caller's rows and the scratch separate: `x` belongs to the caller.
  if (!m.EnsureScratch(tokens, error)) {
    return false;
  }
  hipMemcpy(m.x, x, rows * sizeof(float), hipMemcpyDeviceToDevice);
  const BlockWeights& w = m.blocks[block];

  LaunchLayerNorm(m.x, w.ln1_w.device, w.ln1_b.device, m.h, tokens, dim, m.config.eps,
                  nullptr);
  LaunchMatvecInMajorBatch(m.h, w.qkv_w.device, w.qkv_b.device, m.qkv, tokens, dim,
                           3 * dim, nullptr);
  LaunchVisionAttention(m.qkv, m.attn, tokens, m.config.heads, m.config.HeadDim(),
                        nullptr);
  LaunchMatvecInMajorBatch(m.attn, w.out_w.device, w.out_b.device, m.h, tokens, dim, dim,
                           nullptr);
  LaunchAddInto(m.x, m.x, m.h, rows, nullptr);

  LaunchLayerNorm(m.x, w.ln2_w.device, w.ln2_b.device, m.h, tokens, dim, m.config.eps,
                  nullptr);
  LaunchMatvecInMajorBatch(m.h, w.up_w.device, w.up_b.device, m.ffn, tokens, dim,
                           m.config.ffn, nullptr);
  LaunchGelu(m.ffn, m.ffn, tokens * m.config.ffn, nullptr);
  LaunchMatvecInMajorBatch(m.ffn, w.down_w.device, w.down_b.device, m.h, tokens,
                           m.config.ffn, dim, nullptr);
  LaunchAddInto(m.x, m.x, m.h, rows, nullptr);
  if (hipStreamSynchronize(nullptr) != hipSuccess) {
    if (error != nullptr) *error = "vision: block failed";
    return false;
  }
  hipMemcpy(x, m.x, rows * sizeof(float), hipMemcpyDeviceToDevice);
  return true;
}

bool VisionEncoder::RunAll(std::size_t tokens, float* x, std::string* error) {
  for (std::uint32_t b = 0; b < impl_->config.blocks; ++b) {
    if (!RunBlock(b, x, tokens, error)) {
      return false;
    }
  }
  return true;
}

std::size_t VisionEncoder::DeviceBytes() const { return impl_->device_bytes; }

}  // namespace gufo::ext::ling3_air
