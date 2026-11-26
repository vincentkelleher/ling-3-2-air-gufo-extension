#include "src/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::ext::ling3_air {

std::vector<float> ResizeToChw(const std::uint8_t* rgb, int width, int height,
                               int target, const ImageNorm& norm) {
  std::vector<float> out(static_cast<std::size_t>(3) * target * target);
  if (rgb == nullptr || width <= 0 || height <= 0 || target <= 0) {
    return out;
  }
  const double scale_x = static_cast<double>(width) / target;
  const double scale_y = static_cast<double>(height) / target;
  for (int oy = 0; oy < target; ++oy) {
    const double centre_y = (oy + 0.5) * scale_y - 0.5;
    int y0 = static_cast<int>(std::floor(centre_y));
    double fy = centre_y - y0;
    if (y0 < 0) {
      y0 = 0;
      fy = 0.0;
    } else if (y0 >= height - 1) {
      y0 = height - 1;
      fy = 0.0;
    }
    const int y1 = std::min(y0 + 1, height - 1);
    for (int ox = 0; ox < target; ++ox) {
      const double centre_x = (ox + 0.5) * scale_x - 0.5;
      int x0 = static_cast<int>(std::floor(centre_x));
      double fx = centre_x - x0;
      if (x0 < 0) {
        x0 = 0;
        fx = 0.0;
      } else if (x0 >= width - 1) {
        x0 = width - 1;
        fx = 0.0;
      }
      const int x1 = std::min(x0 + 1, width - 1);
      const std::size_t top = static_cast<std::size_t>(y0) * width;
      const std::size_t bottom = static_cast<std::size_t>(y1) * width;
      for (int c = 0; c < 3; ++c) {
        const auto sample = [&](std::size_t row, int x) {
          return static_cast<double>(rgb[(row + x) * 3 + c]);
        };
        const double top_value = sample(top, x0) * (1.0 - fx) + sample(top, x1) * fx;
        const double bottom_value =
            sample(bottom, x0) * (1.0 - fx) + sample(bottom, x1) * fx;
        const double value = top_value * (1.0 - fy) + bottom_value * fy;
        out[(static_cast<std::size_t>(c) * target + oy) * target + ox] =
            static_cast<float>((value / 255.0 - norm.mean[c]) / norm.std[c]);
      }
    }
  }
  return out;
}

}  // namespace gufo::ext::ling3_air
