// Chat-template conformance against fixtures rendered by the reference
// template (tools/ref_chat.py).
//
// Usage: chat_template_test [cases.json]

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

#include "src/text/chat_template.hpp"

namespace {

namespace text = gufo::ext::ling3_air::text;
using gufo::json::Value;

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

text::ChatTurn ParseTurn(const Value& entry) {
  text::ChatTurn turn;
  turn.role = entry.member_str("role");
  if (const auto* parts = entry.find("parts"); parts != nullptr) {
    for (const auto& part : parts->items()) {
      text::ChatPart out;
      out.image = part.contains("image");
      out.text = part.member_str("text");
      turn.parts.push_back(std::move(out));
    }
  }
  turn.thought = entry.member_str("thought");
  if (const auto* calls = entry.find("tool_calls"); calls != nullptr) {
    for (const auto& call : calls->items()) {
      text::ChatToolCall parsed;
      parsed.name = call.member_str("name");
      if (const auto* args = call.find("arguments"); args != nullptr) {
        for (const auto& arg : args->items()) {
          const auto& triple = arg.items();
          if (triple.size() < 2) {
            continue;
          }
          text::ChatToolArgument out;
          out.name = triple[0].str();
          out.value = triple[1].str();
          out.is_string = triple.size() < 3 || triple[2].as_bool(true);
          parsed.arguments.push_back(std::move(out));
        }
      }
      turn.tool_calls.push_back(std::move(parsed));
    }
  }
  return turn;
}

std::string Show(const std::string& value) {
  std::string out;
  for (const char ch : value) {
    if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else if (static_cast<unsigned char>(ch) < 0x20 ||
               static_cast<unsigned char>(ch) >= 0x7F) {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x",
                    static_cast<unsigned char>(ch));
      out += buffer;
    } else {
      out.push_back(ch);
    }
    if (out.size() > 400) {
      out += "...";
      break;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cases_path =
      argc > 1 ? argv[1] : LING3_FIXTURES_DIR "/chat_cases.json";
  const auto fixture = gufo::json::parse(ReadFile(cases_path));
  const auto* cases = fixture.find("cases");
  if (cases == nullptr || !cases->is_array()) {
    std::fprintf(stderr, "%s: no cases array\n", cases_path.c_str());
    return 2;
  }

  std::size_t failures = 0;
  std::size_t checked = 0;
  for (const auto& entry : cases->items()) {
    const auto* turns_value = entry.find("turns");
    if (turns_value == nullptr) {
      continue;
    }
    std::vector<text::ChatTurn> turns;
    for (const auto& turn : turns_value->items()) {
      turns.push_back(ParseTurn(turn));
    }
    text::ChatTemplateOptions options;
    options.enable_thinking = entry.find("enable_thinking") == nullptr ||
                              entry.find("enable_thinking")->as_bool(true);
    options.preserve_thinking = entry.find("preserve_thinking") == nullptr ||
                                entry.find("preserve_thinking")->as_bool(true);
    options.add_vision_id = entry.find("add_vision_id") != nullptr &&
                            entry.find("add_vision_id")->as_bool(false);
    if (const auto* tools = entry.find("tools"); tools != nullptr) {
      for (const auto& tool : tools->items()) {
        options.tool_definitions.push_back(tool.str());
      }
    }
    ++checked;
    const std::string actual = text::RenderChat(turns, options);
    const std::string expected = entry.member_str("expected");
    if (actual == expected) {
      continue;
    }
    ++failures;
    if (failures <= 6) {
      std::size_t first = 0;
      while (first < expected.size() && first < actual.size() &&
             expected[first] == actual[first]) {
        ++first;
      }
      auto hex = [](const std::string& text) {
        std::string out;
        const std::size_t start = text.size() > 14 ? text.size() - 14 : 0;
        for (std::size_t i = start; i < text.size(); ++i) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "%02x ", static_cast<unsigned char>(text[i]));
          out += buffer;
        }
        return out;
      };
      std::printf("  tail expected: %s\n  tail actual:   %s\n", hex(expected).c_str(),
                  hex(actual).c_str());
      std::printf("MISMATCH %s at byte %zu (len %zu vs %zu)\n  expected: %s\n  actual:   %s\n",
                  entry.member_str("name").c_str(), first, expected.size(),
                  actual.size(),
                  Show(expected.substr(first > 40 ? first - 40 : 0, 120)).c_str(),
                  Show(actual.substr(first > 40 ? first - 40 : 0, 120)).c_str());
    }
  }

  std::printf("chat template: %zu cases, %zu failures\n", checked, failures);
  return failures == 0 ? 0 : 1;
}
