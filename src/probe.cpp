#include "src/probe.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::ext::ling3_air {
namespace {

constexpr std::string_view kArchitectureSuffix = ".block_count";
constexpr std::string_view kExpertCountSuffix = ".expert_count";

// The 128B Air checkpoint: 42 trunk blocks plus one MTP block (43 in the GGUF)
// with 512 routed experts. The other shapes that share the architecture name
// are not what this extension implements, so both keys must match.
constexpr std::uint32_t kBlockCount = 43;
constexpr std::uint32_t kExpertCount = 512;

std::string Key(std::string_view architecture, std::string_view suffix) {
  std::string key(architecture);
  key.append(suffix);
  return key;
}

}  // namespace

bool ProbeLing3Gguf(const core::GgufReader& reader) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture || *architecture != kGgufArchitecture) {
    return false;
  }
  const auto blocks = reader.GetMetadataUint32(
      Key(*architecture, kArchitectureSuffix));
  if (!blocks || *blocks != kBlockCount) {
    return false;
  }
  const auto experts = reader.GetMetadataUint32(
      Key(*architecture, kExpertCountSuffix));
  return experts && *experts == kExpertCount;
}

bool ProbeLing3(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::is_regular_file(path)) {
      return false;
    }
    std::string error;
    const auto reader = core::GgufReader::OpenFile(path, &error);
    if (!reader) {
      return false;
    }
    return ProbeLing3Gguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::ext::ling3_air
