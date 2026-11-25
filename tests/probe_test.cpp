// Probe behaviour: claims the pinned Ling 3.2 Air GGUF, refuses anything else,
// and never opens the GPU.
//
// Usage: probe_test [model.gguf]   (else $LING3_GGUF; without one it skips)

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gufo/ext/gguf_reader.hpp>

#include "src/probe.hpp"

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

  int failures = 0;
  const auto check = [&failures](bool condition, const char* what) {
    std::printf("%s: %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition) {
      ++failures;
    }
  };

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(),
                 error.c_str());
    return 2;
  }

  check(ling::ProbeLing3Gguf(*reader), "probe accepts the pinned GGUF");
  check(reader->GetMetadataString("general.architecture").value_or("") ==
            ling::kGgufArchitecture,
        "the file's architecture is bailingmoe3");
  check(ling::ProbeLing3(model_path), "the path form accepts the pinned GGUF");

  check(!ling::ProbeLing3("/nonexistent/ling3.gguf"),
        "the path form refuses a missing file");
  check(!ling::ProbeLing3("/tmp"), "the path form refuses a directory");

  // A file that is not a GGUF must never be claimed.
  const std::filesystem::path junk =
      std::filesystem::temp_directory_path() / "ling3_probe_junk.gguf";
  {
    std::ofstream out(junk, std::ios::binary);
    out << "not a gguf";
  }
  check(!ling::ProbeLing3(junk.string()), "the path form refuses a non-GGUF");
  std::error_code ignored;
  std::filesystem::remove(junk, ignored);

  std::printf("probe: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
