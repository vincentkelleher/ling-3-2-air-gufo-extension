// Tokenizer conformance: every case in fixtures/tokenizer_cases.json must
// encode to the ids the reference tokenizer produced.
//
// Usage: tokenizer_test <model.gguf> [cases.json]
// Exit code 0 on success; prints the first mismatches otherwise.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>

#include "src/text/tokenizer.hpp"

namespace {

namespace text = gufo::ext::ling3_air::text;

std::string ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string Escape(const std::string& value) {
  std::string out;
  for (const unsigned char byte : value) {
    if (byte == '\n') {
      out += "\\n";
    } else if (byte == '\r') {
      out += "\\r";
    } else if (byte == '\t') {
      out += "\\t";
    } else if (byte < 0x20) {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", byte);
      out += buffer;
    } else {
      out.push_back(static_cast<char>(byte));
    }
  }
  return out;
}

}  // namespace

namespace {

/// The model file comes from the command line, else LING3_GGUF. Without one
/// the test skips: the weights are not part of this repository.
std::string ModelPath(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }
  const char* env = std::getenv("LING3_GGUF");
  return env == nullptr ? std::string() : std::string(env);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_path = ModelPath(argc, argv);
  if (model_path.empty()) {
    std::printf("SKIP: set LING3_GGUF or pass the model .gguf path\n");
    return 77;
  }
  const std::string cases_path = argc > 2 ? argv[2]
                                           : LING3_FIXTURES_DIR "/tokenizer_cases.json";

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(),
                 error.c_str());
    return 2;
  }
  const auto tokenizer = text::Tokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
    return 2;
  }
  std::printf("vocab %zu, bos %u, eos %u, pad %u, template %zu bytes, ICU %s\n",
              tokenizer->VocabSize(), tokenizer->Bos(), tokenizer->Eos(),
              tokenizer->Pad(), tokenizer->ChatTemplate().size(),
              text::HasIcuNormalization() ? "on" : "off");

  const std::string fixture_text = ReadFile(cases_path);
  const auto fixture = gufo::json::parse(fixture_text);
  const auto* cases = fixture.find("cases");
  if (cases == nullptr || !cases->is_array()) {
    std::fprintf(stderr, "%s: no cases array\n", cases_path.c_str());
    return 2;
  }

  std::size_t checked = 0;
  std::size_t tokens = 0;
  std::size_t failures = 0;
  for (const auto& entry : cases->items()) {
    const auto input = entry.member_str("text");
    const auto* expected_ids = entry.find("ids");
    if (expected_ids == nullptr || !expected_ids->is_array()) {
      continue;
    }
    std::vector<text::TokenId> expected;
    expected.reserve(expected_ids->items().size());
    for (const auto& id : expected_ids->items()) {
      expected.push_back(static_cast<text::TokenId>(id.as_size(0)));
    }
    const auto actual = tokenizer->Encode(input, /*parse_special=*/true);
    ++checked;
    tokens += expected.size();
    if (actual == expected) {
      continue;
    }
    ++failures;
    if (failures <= 10) {
      std::size_t first = 0;
      while (first < actual.size() && first < expected.size() &&
             actual[first] == expected[first]) {
        ++first;
      }
      std::printf("MISMATCH case %zu at token %zu: %s\n", checked, first,
                  Escape(input).c_str());
      auto dump = [&](const std::vector<text::TokenId>& ids) {
        std::string line;
        for (std::size_t i = 0; i < ids.size() && i < first + 8; ++i) {
          char buffer[16];
          std::snprintf(buffer, sizeof(buffer), "%s%u", i == 0 ? "" : ",",
                        ids[i]);
          line += buffer;
        }
        return line;
      };
      std::printf("  expected: %s\n  actual:   %s\n", dump(expected).c_str(),
                  dump(actual).c_str());
    }
  }

  std::printf("tokenizer: %zu cases, %zu tokens, %zu failures\n", checked,
              tokens, failures);
  return failures == 0 ? 0 : 1;
}
