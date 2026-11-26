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
  std::fclose(file);

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

  std::printf("vision: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
