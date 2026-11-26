// Image preprocessing against the numpy (arithmetic) and PIL (convention) references
// from tools/make_preprocess_fixture.py.
//
// Usage: preprocess_test [preprocess_fixture.bin]
//   (else $LING3_PREPROCESS_FIXTURE; without it the default path is tried)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/preprocess.hpp"

namespace {

int Failures = 0;

void Check(bool ok, const char* what, double measured, double limit) {
  std::printf("%s %-34s %.3g (limit %.1g)\n", ok ? "ok  " : "FAIL", what, measured, limit);
  if (!ok) {
    ++Failures;
  }
}

std::string FromEnv(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

}  // namespace

int main(int argc, char** argv) {
  namespace ling = gufo::ext::ling3_air;
  const std::string path =
      argc > 1 ? argv[1]
               : (FromEnv("LING3_PREPROCESS_FIXTURE").empty()
                      ? "/work/preprocess_fixture.bin"
                      : FromEnv("LING3_PREPROCESS_FIXTURE"));
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::printf("SKIP: no fixture at %s\n", path.c_str());
    return 77;
  }
  int header[3] = {0, 0, 0};
  if (std::fread(header, sizeof(int), 3, file) != 3) {
    std::fprintf(stderr, "fixture header is short\n");
    return 2;
  }
  const int width = header[0];
  const int height = header[1];
  const int target = header[2];
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  std::vector<float> reference(static_cast<std::size_t>(3) * target * target);
  std::vector<float> pil(reference.size());
  if (std::fread(rgb.data(), 1, rgb.size(), file) != rgb.size() ||
      std::fread(reference.data(), sizeof(float), reference.size(), file) !=
          reference.size() ||
      std::fread(pil.data(), sizeof(float), pil.size(), file) != pil.size()) {
    std::fprintf(stderr, "fixture body is short\n");
    return 2;
  }
  std::fclose(file);

  const std::vector<float> actual =
      ling::ResizeToChw(rgb.data(), width, height, target, ling::ImageNorm{});
  Check(actual.size() == reference.size(), "preprocess output size",
        static_cast<double>(actual.size()), static_cast<double>(reference.size()));

  double worst_numpy = 0.0;
  double worst_pil = 0.0;
  for (std::size_t i = 0; i < actual.size() && i < reference.size(); ++i) {
    worst_numpy = std::max(worst_numpy,
                           std::fabs(static_cast<double>(actual[i]) - reference[i]));
    worst_pil = std::max(worst_pil, std::fabs(static_cast<double>(actual[i]) - pil[i]));
  }
  std::printf("preprocess: %dx%d -> %dx%d\n", width, height, target, target);
  Check(worst_numpy <= 2e-3, "vs numpy reference (arithmetic)", worst_numpy, 2e-3);
  Check(worst_pil <= 3e-2, "vs PIL reference (convention)", worst_pil, 3e-2);

  std::printf("preprocess: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
