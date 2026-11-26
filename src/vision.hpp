// Vision tower from the mmproj (`clip.vision`, ling3vl projector).
//
// The mmproj stores BF16 weights; this loader dequantizes each tensor it needs to
// F32 on the host and uploads it, so the vision kernels stay simple and the text
// kernels do not need a BF16 path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::ext::ling3_air {

struct VisionConfig {
  std::uint32_t dim{1152};
  std::uint32_t heads{16};
  std::uint32_t ffn{4304};
  std::uint32_t blocks{27};
  std::uint32_t patch{16};
  std::uint32_t merge{2};
  float eps{1e-6f};

  [[nodiscard]] std::uint32_t HeadDim() const { return dim / heads; }
};

class VisionEncoder {
 public:
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  [[nodiscard]] static std::unique_ptr<VisionEncoder> Create(
      const core::GgufReader& mmproj, std::string* error);

  [[nodiscard]] const VisionConfig& Config() const;

  /// Patch embed for one temporal pair: `out[o] = p0·w0[o] + p1·w1[o] + bias[o]`,
  /// with `p0`/`p1` each `patch*patch*3` floats. Copies the result to the host.
  [[nodiscard]] bool PatchEmbed(const float* patch0, const float* patch1, float* out,
                                std::string* error);

  /// Runs one encoder block over `tokens` rows of `x` (device, `[tokens][dim]`),
  /// in place.
  [[nodiscard]] bool RunBlock(std::uint32_t block, float* x, std::size_t tokens,
                              std::string* error);

  /// Runs every block in sequence (or the first `max_blocks` when non-negative), in
  /// place.
  [[nodiscard]] bool RunAll(std::size_t tokens, float* x, std::string* error,
                            int max_blocks = -1);

  /// Full image path: `chw` is `grid_h*patch x grid_w*patch x 3` floats, planar RGB
  /// and already normalised with the mmproj's mean/std. Writes one embedding row of
  /// `text_dim` per 2x2 merged patch. Productions images are the mmproj's own
  /// 768x768 (48x48 patches); smaller grids are a test-only crop.
  [[nodiscard]] bool EncodeImage(const float* chw, std::uint32_t grid_h,
                                 std::uint32_t grid_w, std::vector<float>* out,
                                 std::string* error,
                                 std::vector<float>* tower_out = nullptr,
                                 int max_blocks = -1);

  /// Output width of the projector (the text model's hidden size).
  [[nodiscard]] std::uint32_t text_dim() const;

  [[nodiscard]] std::size_t DeviceBytes() const;

 private:
  VisionEncoder() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::ext::ling3_air
