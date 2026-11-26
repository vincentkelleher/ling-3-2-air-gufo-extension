// Image preprocessing for the vision tower: resize to the mmproj's input size and
// turn packed RGB8 into normalised planar CHW floats, as CLIP-style towers expect.
#pragma once

#include <cstdint>
#include <vector>

namespace gufo::ext::ling3_air {

/// Per-channel normalisation constants (`clip.vision.image_mean/image_std`).
struct ImageNorm {
  float mean[3]{0.48145467f, 0.45782751f, 0.40821072f};
  float std[3]{0.26862955f, 0.26130259f, 0.27577710f};
};

/// Bilinear resize of packed RGB8 to `target` x `target`, then `(value/255 - mean) /
/// std` into planar CHW floats. The sampling is centre-aligned
/// (`(i + 0.5) * scale - 0.5` with clamped edges), which is what PIL's BILINEAR and
/// the HF image processors use, so results agree with the reference within one 8-bit
/// step.
[[nodiscard]] std::vector<float> ResizeToChw(const std::uint8_t* rgb, int width,
                                             int height, int target,
                                             const ImageNorm& norm);

}  // namespace gufo::ext::ling3_air
