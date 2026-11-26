// Vision encoder against the numpy reference from tools/make_vision_fixture.py.
//
// Usage: vision_test <mmproj.gguf> [vision_fixture.bin]
//   (else $LING3_MMPROJ / $LING3_VISION_FIXTURE; without them it skips)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include "src/vision.hpp"

namespace {

std::string FromEnv(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

int Failures = 0;

void Check(bool ok, const char* what, double measured, double limit) {
  std::printf("%s %-34s %.3g (limit %.1g)\n", ok ? "ok  " : "FAIL", what, measured, limit);
  if (!ok) {
    ++Failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  namespace ling = gufo::ext::ling3_air;
  const std::string mmproj_path = argc > 1 ? argv[1] : FromEnv("LING3_MMPROJ");
  const std::string fixture_path =
      argc > 2 ? argv[2]
               : (FromEnv("LING3_VISION_FIXTURE").empty()
                      ? "/work/vision_fixture.bin"
                      : FromEnv("LING3_VISION_FIXTURE"));
  if (mmproj_path.empty()) {
    std::printf("SKIP: set LING3_MMPROJ or pass the mmproj path\n");
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(mmproj_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", mmproj_path.c_str(), error.c_str());
    return 2;
  }
  const auto encoder = ling::VisionEncoder::Create(*reader, &error);
  if (!encoder) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const auto& config = encoder->Config();
  std::printf("vision: dim %u, %u heads x %u, ffn %u, %u blocks, %.2f GiB of weights\n",
              config.dim, config.heads, config.HeadDim(), config.ffn, config.blocks,
              encoder->DeviceBytes() / 1073741824.0);

  FILE* file = std::fopen(fixture_path.c_str(), "rb");
  if (file == nullptr) {
    std::printf("SKIP: no fixture at %s\n", fixture_path.c_str());
    return 77;
  }
  int header[6] = {0, 0, 0, 0, 0, 0};
  if (std::fread(header, sizeof(int), 6, file) != 6) {
    std::fprintf(stderr, "fixture header is short\n");
    return 2;
  }
  const int tokens = header[0];
  const int dim = header[1];
  if (dim != static_cast<int>(config.dim)) {
    std::fprintf(stderr, "fixture dim %d does not match the mmproj %u\n", dim, config.dim);
    return 2;
  }
  const std::size_t patch_in = 16 * 16 * 3;
  std::vector<float> patch0(patch_in), patch1(patch_in), embed(dim);
  std::vector<float> x(static_cast<std::size_t>(tokens) * dim);
  std::vector<float> expected(static_cast<std::size_t>(tokens) * dim);
  auto read = [&](std::vector<float>* buffer) {
    return std::fread(buffer->data(), sizeof(float), buffer->size(), file) ==
           buffer->size();
  };
  if (!read(&patch0) || !read(&patch1) || !read(&embed) || !read(&x) || !read(&expected)) {
    std::fprintf(stderr, "fixture body is short\n");
    return 2;
  }
  std::vector<float> actual_embed(dim);
  if (!encoder->PatchEmbed(patch0.data(), patch1.data(), actual_embed.data(), &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  double worst_embed = 0.0;
  for (int i = 0; i < dim; ++i) {
    worst_embed = std::max(worst_embed, std::fabs(static_cast<double>(actual_embed[i]) - embed[i]));
  }
  Check(worst_embed <= 1e-3, "patch embed max abs error", worst_embed, 1e-3);

  float* device_x = nullptr;
  const std::size_t rows = x.size();
  if (hipMalloc(reinterpret_cast<void**>(&device_x), rows * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "vision: cannot allocate the block input\n");
    return 1;
  }
  hipMemcpy(device_x, x.data(), rows * sizeof(float), hipMemcpyHostToDevice);
  if (!encoder->RunBlock(0, device_x, tokens, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<float> actual(rows);
  hipMemcpy(actual.data(), device_x, rows * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(device_x);

  double worst = 0.0;
  double magnitude = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    magnitude = std::max(magnitude, std::fabs(static_cast<double>(expected[i])));
  }
  Check(worst <= 2e-3, "encoder block 0 max abs error", worst, 2e-3);
  std::printf("     (reference magnitude up to %.3f)\n", magnitude);

  // Full image path section (optional: older fixtures stop at the block).
  int grid[2] = {0, 0};
  if (std::fread(grid, sizeof(int), 2, file) == 2) {
    const int gh = grid[0];
    const int gw = grid[1];
    const std::size_t side_h = static_cast<std::size_t>(gh) * 16;
    const std::size_t side_w = static_cast<std::size_t>(gw) * 16;
    std::vector<float> image(3 * side_h * side_w);
    std::vector<float> tower_expected(static_cast<std::size_t>(gh * gw) * dim);
    std::vector<float> pre_expected(static_cast<std::size_t>(gh * gw) * dim);
    std::vector<float> embed_img0_expected(dim);
    std::vector<float> expected_image(static_cast<std::size_t>(gh * gw / 4) * encoder->text_dim());
    if (std::fread(image.data(), sizeof(float), image.size(), file) != image.size() ||
        std::fread(embed_img0_expected.data(), sizeof(float),
                   embed_img0_expected.size(), file) != embed_img0_expected.size() ||
        std::fread(pre_expected.data(), sizeof(float), pre_expected.size(), file) !=
            pre_expected.size() ||
        std::fread(tower_expected.data(), sizeof(float), tower_expected.size(), file) !=
            tower_expected.size() ||
        std::fread(expected_image.data(), sizeof(float), expected_image.size(), file) !=
            expected_image.size()) {
      std::fprintf(stderr, "fixture image section is short\n");
      return 2;
    }

    std::vector<float> actual_image;
    std::vector<float> tower_actual;
    if (!encoder->EncodeImage(image.data(), static_cast<std::uint32_t>(gh),
                              static_cast<std::uint32_t>(gw), &actual_image, &error,
                              &tower_actual)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    double worst_image = 0.0;
    double magnitude_image = 0.0;
    const std::size_t count = std::min(actual_image.size(), expected_image.size());
    for (std::size_t i = 0; i < count; ++i) {
      worst_image = std::max(worst_image,
                             std::fabs(static_cast<double>(actual_image[i]) - expected_image[i]));
      magnitude_image = std::max(magnitude_image, std::fabs(static_cast<double>(expected_image[i])));
    }
    // Patch embed + position embeddings, before any block: `max_blocks = 0` stops at
    // exactly that state and returns it as the tower capture.
    {
      std::vector<float> pre_image;
      std::vector<float> pre;
      if (!encoder->EncodeImage(image.data(), static_cast<std::uint32_t>(gh),
                                static_cast<std::uint32_t>(gw), &pre_image, &error, &pre,
                                0)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
      }
      double worst_pre = 0.0;
      const std::size_t before_count = std::min(pre.size(), pre_expected.size());
      for (std::size_t i = 0; i < before_count; ++i) {
        worst_pre = std::max(worst_pre,
                             std::fabs(static_cast<double>(pre[i]) - pre_expected[i]));
      }
      Check(pre.size() == pre_expected.size(), "patch-embed + pos shape",
            static_cast<double>(pre.size()), static_cast<double>(pre_expected.size()));
      Check(worst_pre <= 2e-3, "patch-embed + pos max abs error", worst_pre, 2e-3);

    }

    // The same patch vector through `PatchEmbed` (no position embeddings), to tell a
    // patchify mistake apart from a weight or normalisation mistake.
    {
      std::vector<float> p0(768), p1(768);
      const std::size_t pw = static_cast<std::size_t>(gw) * 16;
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 16; ++y) {
          for (int x = 0; x < 16; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * pw + x;
            p0[(c * 16 + y) * 16 + x] = image[static_cast<std::size_t>(c) * side_h * side_w + pixel];
            p1[(c * 16 + y) * 16 + x] = p0[(c * 16 + y) * 16 + x];
          }
        }
      }
      std::vector<float> directly(dim);
      if (!encoder->PatchEmbed(p0.data(), p1.data(), directly.data(), &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
      }
      double worst_direct = 0.0;
      for (std::size_t i = 0; i < directly.size(); ++i) {
        worst_direct = std::max(worst_direct,
                                std::fabs(static_cast<double>(directly[i]) - embed_img0_expected[i]));
      }
      Check(worst_direct <= 2e-3, "patch embed of image patch 0", worst_direct, 2e-3);

    }

    Check(actual_image.size() == expected_image.size(), "image embedding shape",
          static_cast<double>(actual_image.size()), static_cast<double>(expected_image.size()));
    Check(worst_image <= 5e-3, "image embedding max abs error", worst_image, 5e-3);
    double worst_tower = 0.0;
    const std::size_t tower_count = std::min(tower_actual.size(), tower_expected.size());
    for (std::size_t i = 0; i < tower_count; ++i) {
      worst_tower = std::max(worst_tower,
                             std::fabs(static_cast<double>(tower_actual[i]) - tower_expected[i]));
    }
    Check(tower_actual.size() == tower_expected.size(), "tower output shape",
          static_cast<double>(tower_actual.size()), static_cast<double>(tower_expected.size()));
    Check(worst_tower <= 5e-3, "tower output max abs error (27 blocks)", worst_tower, 5e-3);

    int trace_count = 0;
    if (std::fread(&trace_count, sizeof(int), 1, file) == 1) {
      for (int t = 0; t < trace_count; ++t) {
        int block_index = 0;
        std::vector<float> reference(static_cast<std::size_t>(gh * gw) * config.dim);
        if (std::fread(&block_index, sizeof(int), 1, file) != 1 ||
            std::fread(reference.data(), sizeof(float), reference.size(), file) !=
                reference.size()) {
          std::fprintf(stderr, "fixture trace is short\n");
          return 2;
        }
        std::vector<float> sampled;
        std::vector<float> trace;
        if (!encoder->EncodeImage(image.data(), static_cast<std::uint32_t>(gh),
                                  static_cast<std::uint32_t>(gw), &sampled, &error,
                                  &trace, block_index + 1)) {
          std::fprintf(stderr, "%s\n", error.c_str());
          return 1;
        }
        double worst_trace = 0.0;
        const std::size_t trace_count_values = std::min(trace.size(), reference.size());
        for (std::size_t i = 0; i < trace_count_values; ++i) {
          worst_trace = std::max(worst_trace,
                                 std::fabs(static_cast<double>(trace[i]) - reference[i]));
        }
        char label[48];
        std::snprintf(label, sizeof(label), "trace after block %d", block_index);
        Check(worst_trace <= 2e-3, label, worst_trace, 2e-3);
      }
    }
    std::printf("     (image reference magnitude up to %.3f)\n", magnitude_image);
  }
  std::fclose(file);

  std::printf("vision: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
