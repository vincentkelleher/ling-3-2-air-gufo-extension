#include "src/vision.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "src/hip/vision.hpp"

namespace gufo::ext::ling3_air {
namespace {

using hip::LaunchAddInto;
using hip::LaunchAddPosEmb;
using hip::LaunchMerge2x2;
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
  TensorRef pos_embd;
  TensorRef input_norm_w, input_norm_b;
  TensorRef mm0_w, mm0_b, mm2_w, mm2_b;
  std::uint32_t text_dim{0};
  std::uint32_t pos_grid{0};

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
  for (const TensorRef* ref : {&m.patch0, &m.patch1, &m.patch_bias, &m.pos_embd,
                               &m.input_norm_w, &m.input_norm_b, &m.mm0_w, &m.mm0_b,
                               &m.mm2_w, &m.mm2_b}) {
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
  const bool projector_ok =
      Upload(mmproj, "v.position_embd.weight", &m.pos_embd, &local) &&
      Upload(mmproj, "mm.input_norm.weight", &m.input_norm_w, &local) &&
      Upload(mmproj, "mm.input_norm.bias", &m.input_norm_b, &local) &&
      Upload(mmproj, "mm.0.weight", &m.mm0_w, &local) &&
      Upload(mmproj, "mm.0.bias", &m.mm0_b, &local) &&
      Upload(mmproj, "mm.2.weight", &m.mm2_w, &local) &&
      Upload(mmproj, "mm.2.bias", &m.mm2_b, &local);
  if (!projector_ok) {
    return fail(local);
  }
  for (const TensorRef* ref : {&m.pos_embd, &m.input_norm_w, &m.input_norm_b, &m.mm0_w,
                               &m.mm0_b, &m.mm2_w, &m.mm2_b}) {
    m.device_bytes += ref->count * sizeof(float);
  }
  // Position embeddings are a square grid: 1152 x (grid*grid).
  m.pos_grid = static_cast<std::uint32_t>(
      std::lround(std::sqrt(static_cast<double>(m.pos_embd.count) / m.config.dim)));
  m.text_dim = static_cast<std::uint32_t>(m.mm2_b.count);
  if (m.pos_grid * m.pos_grid * m.config.dim != m.pos_embd.count ||
      m.mm0_b.count != m.text_dim) {
    return fail("vision: projector shapes are inconsistent");
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

bool VisionEncoder::RunAll(std::size_t tokens, float* x, std::string* error,
                           int max_blocks) {
  const std::uint32_t limit =
      max_blocks < 0 || static_cast<std::uint32_t>(max_blocks) > impl_->config.blocks
          ? impl_->config.blocks
          : static_cast<std::uint32_t>(max_blocks);
  for (std::uint32_t b = 0; b < limit; ++b) {
    if (!RunBlock(b, x, tokens, error)) {
      return false;
    }
  }
  return true;
}

std::uint32_t VisionEncoder::text_dim() const { return impl_->text_dim; }

bool VisionEncoder::EncodeImage(const float* chw, std::uint32_t grid_h,
                                std::uint32_t grid_w, std::vector<float>* out,
                                std::string* error, std::vector<float>* tower_out,
                                int max_blocks) {
  Impl& m = *impl_;
  const std::uint32_t patch = m.config.patch;
  const std::uint32_t dim = m.config.dim;
  const std::size_t tokens = static_cast<std::size_t>(grid_h) * grid_w;
  if (grid_h % 2 != 0 || grid_w % 2 != 0) {
    if (error != nullptr) *error = "vision: patch grid must be even for the 2x2 merge";
    return false;
  }
  const std::size_t pixels_h = static_cast<std::size_t>(grid_h) * patch;
  const std::size_t pixels_w = static_cast<std::size_t>(grid_w) * patch;
  const std::size_t patch_values = static_cast<std::size_t>(patch) * patch * 3;

  // Patchify on the host. A still image feeds the same pixels to both temporal
  // halves of the kernel (`v.patch_embd.weight` and `.weight.1`).
  std::vector<float> patches(tokens * 2 * patch_values);
  for (std::uint32_t gh = 0; gh < grid_h; ++gh) {
    for (std::uint32_t gw = 0; gw < grid_w; ++gw) {
      const std::size_t row = static_cast<std::size_t>(gh) * grid_w + gw;
      for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t y = 0; y < patch; ++y) {
          for (std::uint32_t x = 0; x < patch; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(gh) * patch + y) * pixels_w +
                                      static_cast<std::size_t>(gw) * patch + x;
            const float value = chw[static_cast<std::size_t>(c) * pixels_h * pixels_w + pixel];
            // Both temporal halves must be contiguous blocks (`[rows][patch]` then
            // `[rows][patch]` again), because they are fed to two batched matvecs.
            const std::size_t target = row * patch_values + (c * patch + y) * patch + x;
            patches[target] = value;
            patches[target + tokens * patch_values] = value;
          }
        }
      }
    }
  }

  const std::size_t merged_rows = tokens / 4;
  const std::size_t merged_dim = 4 * static_cast<std::size_t>(dim);
  float* device_patches = nullptr;
  float* hidden = nullptr;
  float* second = nullptr;
  float* merged = nullptr;
  float* normed = nullptr;
  float* projected = nullptr;
  auto release = [&] {
    for (float** pointer : {&device_patches, &hidden, &second, &merged, &normed, &projected}) {
      if (*pointer != nullptr) {
        hipFree(*pointer);
        *pointer = nullptr;
      }
    }
  };
  const bool allocated =
      hipMalloc(reinterpret_cast<void**>(&device_patches), patches.size() * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&hidden), tokens * dim * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&second), tokens * dim * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&merged), merged_rows * merged_dim * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&normed), merged_rows * merged_dim * sizeof(float)) == hipSuccess &&
      hipMalloc(reinterpret_cast<void**>(&projected), merged_rows * m.text_dim * sizeof(float)) == hipSuccess;
  if (!allocated) {
    release();
    if (error != nullptr) *error = "vision: image buffers failed";
    return false;
  }
  hipMemcpy(device_patches, patches.data(), patches.size() * sizeof(float),
            hipMemcpyHostToDevice);

  // Patch embed: two halves of the temporal kernel, summed, plus bias.
  LaunchMatvecInMajorBatch(device_patches, m.patch0.device, m.patch_bias.device, hidden,
                           tokens, patch_values, dim, nullptr);
  LaunchMatvecInMajorBatch(device_patches + tokens * patch_values, m.patch1.device,
                           nullptr, second, tokens, patch_values, dim, nullptr);
  LaunchAddInto(hidden, hidden, second, tokens * dim, nullptr);
  LaunchAddPosEmb(hidden, m.pos_embd.device, tokens, grid_w, m.pos_grid, dim, nullptr);
  if (!RunAll(tokens, hidden, error, max_blocks)) {
    release();
    return false;
  }
  if (tower_out != nullptr) {
    tower_out->resize(tokens * dim);
    hipMemcpy(tower_out->data(), hidden, tower_out->size() * sizeof(float),
              hipMemcpyDeviceToHost);
  }
  // `mm.input_norm` is `dim`-wide: normalise every patch, then merge 2x2.
  LaunchLayerNorm(hidden, m.input_norm_w.device, m.input_norm_b.device, hidden, tokens,
                  dim, m.config.eps, nullptr);
  LaunchMerge2x2(hidden, merged, grid_h, grid_w, dim, nullptr);
  hipMemcpy(normed, merged, merged_rows * merged_dim * sizeof(float),
            hipMemcpyDeviceToDevice);
  LaunchMatvecInMajorBatch(normed, m.mm0_w.device, m.mm0_b.device, projected,
                           merged_rows, merged_dim, m.text_dim, nullptr);
  LaunchGelu(projected, projected, merged_rows * m.text_dim, nullptr);
  LaunchMatvecInMajorBatch(projected, m.mm2_w.device, m.mm2_b.device, normed,
                           merged_rows, m.text_dim, m.text_dim, nullptr);
  if (hipStreamSynchronize(nullptr) != hipSuccess) {
    release();
    if (error != nullptr) *error = "vision: projector failed";
    return false;
  }
  out->resize(merged_rows * m.text_dim);
  hipMemcpy(out->data(), normed, out->size() * sizeof(float), hipMemcpyDeviceToHost);
  release();
  if (error != nullptr) error->clear();
  return true;
}

std::uint32_t VisionEncoder::image_side() const {
  return impl_->config.patch * impl_->pos_grid;
}

std::uint32_t VisionEncoder::tokens_per_image() const {
  return impl_->pos_grid * impl_->pos_grid / 4;
}

std::size_t VisionEncoder::DeviceBytes() const { return impl_->device_bytes; }

}  // namespace gufo::ext::ling3_air
