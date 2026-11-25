// Ling 3.2 byte-level BPE tokenizer.
//
// Vocabulary and merges come from the GGUF (llama.cpp layout: tokens in the
// GPT-2 byte-to-unicode spelling, `tokenizer.ggml.merges`, `tokenizer.ggml.
// token_type`). Pre-tokenization is the scanner in pretokenizer.hpp, which is
// the HF `Split` pattern of this model's tokenizer.json. Text is NFC-normalized
// before splitting when the build provides ICU, matching the reference
// tokenizer's normalizer.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::core {
class GgufReader;
}  // namespace gufo::core

namespace gufo::ext::ling3_air::text {

using TokenId = std::uint32_t;
inline constexpr TokenId kInvalidToken = 0xFFFFFFFFu;

/// Token roles, matching GGUF `tokenizer.ggml.token_type`.
enum class TokenType : std::int32_t {
  kUndefined = 0,
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

/// Everything the encoder needs, independent of where it came from.
struct TokenizerData {
  std::vector<std::string> tokens;       // id -> token text (byte-level spelling)
  std::vector<TokenType> types;          // id -> token type (may be empty)
  std::vector<std::string> merges;       // "left right", in rank order
  TokenId bos{0};
  TokenId eos{0};
  TokenId pad{0};
  std::string chat_template;             // tokenizer.chat_template, if present
};

class Tokenizer {
 public:
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  /// Returns null and sets `*error` when the data cannot be used.
  [[nodiscard]] static std::unique_ptr<Tokenizer> Create(
      TokenizerData data, std::string* error = nullptr);

  /// Reads the tokenizer section of a GGUF file (single file or first shard).
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error = nullptr);

  /// Text to ids. With `parse_special`, literal spellings of control and
  /// user-defined tokens in `text` become those tokens (leftmost, longest);
  /// otherwise they are ordinary text.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            bool parse_special = true) const;

  /// BPE of one pre-tokenized piece, no special-token parsing.
  void EncodeOrdinary(std::string_view text, std::vector<TokenId>* out) const;

  /// Ids to text. Control tokens are omitted unless `keep_control`.
  [[nodiscard]] std::string Decode(const std::vector<TokenId>& ids,
                                   bool keep_control = false) const;

  [[nodiscard]] std::size_t VocabSize() const;
  [[nodiscard]] TokenId Bos() const;
  [[nodiscard]] TokenId Eos() const;
  [[nodiscard]] TokenId Pad() const;
  [[nodiscard]] const std::string& ChatTemplate() const;
  [[nodiscard]] std::optional<TokenId> TokenFor(std::string_view literal) const;

 private:
  Tokenizer() = default;

  struct Special {
    std::string literal;
    TokenId id;
  };

  std::vector<std::string> tokens_;
  std::vector<TokenType> types_;
  std::unordered_map<std::string, TokenId> by_token_;
  std::unordered_map<std::string, std::uint32_t> merges_;
  std::vector<Special> specials_;       // sorted longest-first for matching
  std::vector<bool> special_codepoint_; // per token: literal is a special token
  std::array<std::string, 256> byte_to_char_{};
  std::array<std::int32_t, 512> char_to_byte_{};
  TokenId bos_{0};
  TokenId eos_{0};
  TokenId pad_{0};
  std::string chat_template_;
};

/// NFC-normalizes `text` (identity when the build has no ICU).
std::string NormalizeNfc(std::string_view text);

/// Whether this build normalizes with ICU.
bool HasIcuNormalization();

}  // namespace gufo::ext::ling3_air::text
