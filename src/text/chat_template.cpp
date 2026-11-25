#include "src/text/chat_template.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext::ling3_air::text {
namespace {

constexpr std::string_view kRoleEnd = "<|role_end|>";
constexpr std::string_view kThinkingOn = "detailed thinking on";
constexpr std::string_view kThinkingOff = "detailed thinking off";
constexpr std::string_view kImageMarker =
    "<|vision_start|><|image_pad|><|vision_end|>";
// The model's markers are "<think" + ">" and "</think" + ">"
// (bytes 3c 74 68 69 6e 6b 3e / 3c 2f 74 68 69 6e 6b 3e), not the
// " thinking" / " response" spelling used by other Qwen-family models.
constexpr std::string_view kThinkOpen = "<think" ">";
constexpr std::string_view kThinkClose = "</think" ">";

bool Contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void StripLeadingNewlines(std::string* text) {
  std::size_t i = 0;
  while (i < text->size() && (*text)[i] == '\n') {
    ++i;
  }
  text->erase(0, i);
}

void StripTrailingNewlines(std::string* text) {
  std::size_t i = text->size();
  while (i > 0 && (*text)[i - 1] == '\n') {
    --i;
  }
  text->resize(i);
}

/// The `<tools>` block, byte for byte as the template writes it.
std::string ToolsBlock(const std::vector<std::string>& definitions) {
  std::string out =
      "# Tools\n\nYou may call one or more functions to assist with the user "
      "query.\n\nYou are provided with function signatures within "
      "<tools></tools> XML tags:\n<tools>";
  for (const auto& definition : definitions) {
    out += "\n";
    out += definition;
  }
  out +=
      "\n</tools>\n\nIf none of the functions can be used, point it out. If "
      "the given question lacks the parameters required by the function, also "
      "point it out.\nIf you need to use a function, for each function call, "
      "output the function name and arguments within the following XML "
      "format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n"
      "<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n"
      "<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n";
  return out;
}

/// The content of one user message for the multi-step-tool scan, which does
/// not advance the image counter.
std::string ScanContent(const ChatTurn& turn, const ChatTemplateOptions& options) {
  int counter = 0;
  return RenderContent(turn, /*count_vision=*/false, &counter,
                       options.add_vision_id);
}

}  // namespace

std::string RenderContent(const ChatTurn& turn, bool count_vision,
                          int* image_counter, bool add_vision_id) {
  std::string out;
  for (const auto& part : turn.parts) {
    if (!part.image) {
      out += part.text;
      continue;
    }
    if (count_vision && image_counter != nullptr) {
      ++*image_counter;
    }
    if (add_vision_id) {
      out += "Picture ";
      out += std::to_string(image_counter == nullptr ? 0 : *image_counter);
      out += ": ";
    }
    out.append(kImageMarker);
  }
  return out;
}

std::string RenderChatWithoutGeneration(const std::vector<ChatTurn>& turns,
                                       const ChatTemplateOptions& options) {
  std::string out;
  const bool has_tools = !options.tool_definitions.empty();

  // Message contents, rendered in order so the image counter follows the
  // conversation (the template counts inside the message loop only).
  std::vector<std::string> content;
  content.reserve(turns.size());
  int image_counter = 0;
  for (const auto& turn : turns) {
    content.push_back(RenderContent(turn, /*count_vision=*/true, &image_counter,
                                    options.add_vision_id));
  }

  out.append("<role>SYSTEM</role>");

  bool has_thinking_instruction = false;
  if (!turns.empty() && turns.front().role == "system") {
    const std::string system_content = ScanContent(turns.front(), options);
    has_thinking_instruction =
        Contains(system_content, kThinkingOn) || Contains(system_content, kThinkingOff);
    out += system_content;
    if ((has_tools || !has_thinking_instruction) && !system_content.empty() &&
        system_content.back() != '\n') {
      out += '\n';
    }
  }
  if (has_tools) {
    out += ToolsBlock(options.tool_definitions);
  }
  if (!has_thinking_instruction) {
    out.append(options.enable_thinking ? kThinkingOn : kThinkingOff);
  }
  out.append(kRoleEnd);

  // Index of the last user message that is a real query rather than a tool
  // echo: everything after it is "the current turn", where preserved thinking
  // is always kept.
  std::size_t last_query_index = turns.empty() ? 0 : turns.size() - 1;
  for (std::size_t i = turns.size(); i-- > 0;) {
    if (turns[i].role != "user") {
      continue;
    }
    const std::string scanned = ScanContent(turns[i], options);
    if (StartsWith(scanned, "<tool_response>") &&
        EndsWith(scanned, "</tool_response>")) {
      continue;
    }
    last_query_index = i;
    break;
  }

  for (std::size_t i = 0; i < turns.size(); ++i) {
    const ChatTurn& turn = turns[i];
    if (turn.role == "user") {
      out.append("<role>HUMAN</role>");
      out += content[i];
      out.append(kRoleEnd);
    } else if (turn.role == "system") {
      if (i == 0) {
        continue;  // already written above
      }
      out.append("<role>SYSTEM</role>");
      out += content[i];
      out.append(kRoleEnd);
    } else if (turn.role == "assistant") {
      std::string body = content[i];
      std::string reasoning = turn.thought;
      if (reasoning.empty() && Contains(body, kThinkClose)) {
        const auto close_first = body.find(kThinkClose);
        std::string head = body.substr(0, close_first);
        StripTrailingNewlines(&head);
        const auto open = head.rfind(kThinkOpen);
        reasoning = open == std::string::npos ? head : head.substr(open + kThinkOpen.size());
        StripLeadingNewlines(&reasoning);
        std::string tail = body.substr(body.rfind(kThinkClose) + kThinkClose.size());
        StripLeadingNewlines(&tail);
        body = tail;
      }

      out.append("<role>ASSISTANT</role>");
      const bool keep_reasoning = options.preserve_thinking || i > last_query_index;
      if (keep_reasoning && !reasoning.empty()) {
        StripLeadingNewlines(&reasoning);
        StripTrailingNewlines(&reasoning);
        out += "\n";
        out.append(kThinkOpen);
        out += reasoning;
        out.append(kThinkClose);
        StripLeadingNewlines(&body);
        out += body;
      } else {
        out += "\n";
        out.append(kThinkOpen);
        out.append(kThinkClose);
        out += body;
      }

      for (std::size_t c = 0; c < turn.tool_calls.size(); ++c) {
        const auto& call = turn.tool_calls[c];
        if ((c == 0 && !body.empty()) || c > 0) {
          out += '\n';
        }
        out.append("<tool_call>");
        out += call.name;
        for (const auto& argument : call.arguments) {
          out.append("\n<arg_key>");
          out += argument.name;
          out.append("</arg_key>");
          out.append("\n<arg_value>");
          out += argument.value;
          out.append("</arg_value>");
        }
        out.append("\n</tool_call>");
      }
      out.append(kRoleEnd);
    } else if (turn.role == "tool") {
      if (i == 0 || turns[i - 1].role != "tool") {
        out.append("<role>OBSERVATION</role>");
      }
      out.append("\n<tool_response>\n");
      out += content[i];
      out.append("\n</tool_response>");
      if (i + 1 == turns.size() || turns[i + 1].role != "tool") {
        out.append(kRoleEnd);
      }
    } else {
      out.append("<role>");
      out += turn.role;
      out.append("</role>");
      out += content[i];
      out.append(kRoleEnd);
    }
  }
  return out;
}

std::string RenderChat(const std::vector<ChatTurn>& turns,
                       const ChatTemplateOptions& options) {
  std::string out = RenderChatWithoutGeneration(turns, options);
  out.append("<role>ASSISTANT</role>");
  out += '\n';
  out.append(kThinkOpen);
  if (!options.enable_thinking) {
    out.append(kThinkClose);
  }
  return out;
}

}  // namespace gufo::ext::ling3_air::text
