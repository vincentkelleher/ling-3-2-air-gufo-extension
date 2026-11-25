// Ling 3.2 chat framing: a C++ rendering of the model's own
// `tokenizer.chat_template` (the "Bailing V3 unified chat template").
//
// The renderer is byte-compatible with the Jinja template as transformers runs
// it; tests/chat_template_test.cpp checks it against fixtures rendered by
// jinja2 from the template file itself (tools/ref_chat.py).
#pragma once

#include <string>
#include <vector>

namespace gufo::ext::ling3_air::text {

/// One content part. An image part expands to the vision marker the template
/// writes (`<|vision_start|><|image_pad|><|vision_end|>`).
struct ChatPart {
  std::string text;
  bool image{false};
};

struct ChatToolArgument {
  std::string name;
  std::string value;      // raw string when `is_string`, else JSON text
  bool is_string{true};
};

struct ChatToolCall {
  std::string name;
  std::vector<ChatToolArgument> arguments;
};

/// One conversation turn, already split into parts. `role` is the wire name:
/// "system", "user", "assistant", "tool", or anything else (emitted verbatim).
struct ChatTurn {
  std::string role;
  std::vector<ChatPart> parts;
  /// Assistant reasoning, when the caller has it as a separate field.
  std::string thought;
  std::vector<ChatToolCall> tool_calls;
};

struct ChatTemplateOptions {
  bool enable_thinking{true};
  bool preserve_thinking{true};
  bool add_vision_id{false};
  /// Tool definitions as JSON text, emitted verbatim inside `<tools>`.
  std::vector<std::string> tool_definitions;
};

/// Renders the prompt through the assistant generation marker.
[[nodiscard]] std::string RenderChat(const std::vector<ChatTurn>& turns,
                                    const ChatTemplateOptions& options);

/// The same rendering without the trailing generation prompt, for tests and
/// for callers that append their own continuation.
[[nodiscard]] std::string RenderChatWithoutGeneration(
    const std::vector<ChatTurn>& turns, const ChatTemplateOptions& options);

/// Text of one turn's parts with images expanded, as the template sees it.
[[nodiscard]] std::string RenderContent(const ChatTurn& turn,
                                        bool count_vision, int* image_counter,
                                        bool add_vision_id);

}  // namespace gufo::ext::ling3_air::text
