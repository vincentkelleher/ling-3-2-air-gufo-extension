// Config and weight-map validation against a real GGUF.
//
// Usage: config_test [model.gguf]   (else $LING3_GGUF; without one it skips)

#include <cstdio>
#include <cstdlib>
#include <string>

#include <gufo/ext/gguf_reader.hpp>

#include "src/config.hpp"
#include "src/weights.hpp"

namespace {

std::string ModelPath(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }
  const char* env = std::getenv("LING3_GGUF");
  return env == nullptr ? std::string() : std::string(env);
}

}  // namespace

int main(int argc, char** argv) {
  namespace ling = gufo::ext::ling3_air;
  const std::string model_path = ModelPath(argc, argv);
  if (model_path.empty()) {
    std::printf("SKIP: set LING3_GGUF or pass the model .gguf path\n");
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(),
                 error.c_str());
    return 2;
  }

  ling::Ling3Config config;
  if (!ling::ParseConfig(*reader, &config, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::printf("%s\n", ling::DescribeConfig(config).c_str());

  std::string kinds;
  for (std::uint32_t il = 0; il < config.n_layer_total; ++il) {
    kinds += config.IsKda(il) ? 'K' : 'M';
  }
  std::printf("layers: %s\n", kinds.c_str());

  ling::Ling3Weights weights;
  if (!ling::LoadWeights(*reader, config, &weights, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::printf("%s\n", ling::DescribeWeights(weights).c_str());

  int failures = 0;
  const auto check = [&failures](bool condition, const char* what) {
    if (!condition) {
      std::printf("FAIL: %s\n", what);
      ++failures;
    }
  };
  check(config.KdaCount() == 35, "35 KDA blocks");
  check(config.MlaCount() == 8, "7 MLA blocks plus the MLA MTP block");
  check(config.n_layer_total == weights.layers.size(), "one weight set per block");
  check(config.kind[config.n_layer] == ling::LayerKind::kMla,
        "the MTP block is MLA");
  check(weights.layers[config.n_layer].nextn_eh_proj.Valid(),
        "the MTP block has its nextn tensors");
  check(!weights.layers[config.n_layer].dense, "the MTP block is MoE");
  check(weights.total_bytes > 70ull * 1024 * 1024 * 1024,
        "the mapped weights are the full model (>70 GiB)");

  std::printf("config: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
