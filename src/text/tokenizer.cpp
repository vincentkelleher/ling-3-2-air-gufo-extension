#include "src/text/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/text/pretokenizer.hpp"
#include "src/text/unicode.hpp"

#if defined(LING3_HAVE_ICU)
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#endif

namespace gufo::ext::ling3_air::text {
namespace {

// GPT-2 byte-to-unicode: printable ASCII plus both Latin-1 halves map to
// themselves, every other byte to U+0100 + n in byte order.
struct ByteAlphabet {
  std::array<std::string, 256> byte_to_char;
  std::array<std::int32_t, 512> char_to_byte;

  ByteAlphabet() {
    char_to_byte.fill(-1);
    std::vector<int> bytes;
    std::vector<int> codepoints;
    for (int b = 0x21; b <= 0x7E; ++b) {
      bytes.push_back(b);
      codepoints.push_back(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
      bytes.push_back(b);
      codepoints.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
      bytes.push_back(b);
      codepoints.push_back(b);
    }
    std::array<bool, 256> used{};
    for (const int b : bytes) {
      used[static_cast<std::size_t>(b)] = true;
    }
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      if (!used[static_cast<std::size_t>(b)]) {
        bytes.push_back(b);
        codepoints.push_back(256 + extra);
        ++extra;
      }
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      const auto cp = static_cast<std::uint32_t>(codepoints[i]);
      std::string utf8;
      if (cp < 0x80) {
        utf8.push_back(static_cast<char>(cp));
      } else if (cp < 0x800) {
        utf8.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        utf8.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        utf8.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      byte_to_char[static_cast<std::size_t>(bytes[i])] = utf8;
      char_to_byte[static_cast<std::size_t>(codepoints[i])] = bytes[i];
    }
  }
};

const ByteAlphabet& Alphabet() {
  static const ByteAlphabet alphabet;
  return alphabet;
}

bool IsSpecialType(TokenType type) {
  return type == TokenType::kControl || type == TokenType::kUserDefined ||
         type == TokenType::kUnused;
}

}  // namespace

Tokenizer::~Tokenizer() = default;

std::unique_ptr<Tokenizer> Tokenizer::Create(TokenizerData data,
                                             std::string* error) {
  auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return nullptr;
  };

  if (data.tokens.empty()) {
    return fail("tokenizer: empty vocabulary");
  }
  if (data.merges.empty()) {
    return fail("tokenizer: no BPE merges");
  }
  std::unique_ptr<Tokenizer> tokenizer(new Tokenizer());
  tokenizer->tokens_ = std::move(data.tokens);
  tokenizer->bos_ = data.bos;
  tokenizer->eos_ = data.eos;
  tokenizer->pad_ = data.pad;
  tokenizer->chat_template_ = std::move(data.chat_template);
  tokenizer->types_.assign(tokenizer->tokens_.size(), TokenType::kNormal);
  if (!data.types.empty()) {
    if (data.types.size() != tokenizer->tokens_.size()) {
      return fail("tokenizer: token_type count does not match the vocabulary");
    }
    tokenizer->types_ = std::move(data.types);
  }

  tokenizer->by_token_.reserve(tokenizer->tokens_.size() * 2);
  tokenizer->special_codepoint_.assign(tokenizer->tokens_.size(), false);
  for (std::size_t id = 0; id < tokenizer->tokens_.size(); ++id) {
    const auto& token = tokenizer->tokens_[id];
    if (token.empty()) {
      continue;
    }
    tokenizer->by_token_.emplace(token, static_cast<TokenId>(id));
    if (IsSpecialType(tokenizer->types_[id])) {
      tokenizer->special_codepoint_[id] = true;
      tokenizer->specials_.push_back({token, static_cast<TokenId>(id)});
    }
  }
  std::sort(tokenizer->specials_.begin(), tokenizer->specials_.end(),
            [](const Special& a, const Special& b) {
              if (a.literal.size() != b.literal.size()) {
                return a.literal.size() > b.literal.size();
              }
              return a.id < b.id;
            });

  tokenizer->merges_.reserve(data.merges.size() * 2);
  for (std::size_t rank = 0; rank < data.merges.size(); ++rank) {
    tokenizer->merges_.emplace(data.merges[rank],
                               static_cast<std::uint32_t>(rank));
  }

  const auto& alphabet = Alphabet();
  tokenizer->byte_to_char_ = alphabet.byte_to_char;
  tokenizer->char_to_byte_ = alphabet.char_to_byte;
  if (error != nullptr) {
    error->clear();
  }
  return tokenizer;
}

std::unique_ptr<Tokenizer> Tokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error) {
  auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return nullptr;
  };

  const auto* tokens_value = reader.FindMetadata("tokenizer.ggml.tokens");
  if (tokens_value == nullptr) {
    return fail("tokenizer: GGUF has no tokenizer.ggml.tokens");
  }
  const auto* token_list =
      std::get_if<std::vector<std::string_view>>(&tokens_value->value);
  if (token_list == nullptr || token_list->empty()) {
    return fail("tokenizer: tokenizer.ggml.tokens is not a string array");
  }

  TokenizerData data;
  data.tokens.reserve(token_list->size());
  for (const auto token : *token_list) {
    data.tokens.emplace_back(token);
  }

  if (const auto* types = reader.FindMetadata("tokenizer.ggml.token_type")) {
    if (const auto* ints =
            std::get_if<std::vector<std::int64_t>>(&types->value)) {
      data.types.reserve(ints->size());
      for (const auto value : *ints) {
        data.types.push_back(
            static_cast<TokenType>(static_cast<std::int32_t>(value)));
      }
    } else if (const auto* uints =
                   std::get_if<std::vector<std::uint64_t>>(&types->value)) {
      data.types.reserve(uints->size());
      for (const auto value : *uints) {
        data.types.push_back(
            static_cast<TokenType>(static_cast<std::int32_t>(value)));
      }
    } else {
      return fail("tokenizer: tokenizer.ggml.token_type is not an integer array");
    }
  }

  if (const auto* merges = reader.FindMetadata("tokenizer.ggml.merges")) {
    if (const auto* list =
            std::get_if<std::vector<std::string_view>>(&merges->value)) {
      data.merges.reserve(list->size());
      for (const auto merge : *list) {
        data.merges.emplace_back(merge);
      }
    } else {
      return fail("tokenizer: tokenizer.ggml.merges is not a string array");
    }
  }

  const auto id_of = [&](std::string_view key, std::uint32_t fallback) {
    if (const auto value = reader.GetMetadataUint32(key)) {
      return static_cast<TokenId>(*value);
    }
    return static_cast<TokenId>(fallback);
  };
  data.bos = id_of("tokenizer.ggml.bos_token_id", 0);
  data.eos = id_of("tokenizer.ggml.eos_token_id", 0);
  data.pad = id_of("tokenizer.ggml.padding_token_id", 0);
  if (const auto chat = reader.GetMetadataString("tokenizer.chat_template")) {
    data.chat_template = std::string(*chat);
  }
  return Create(std::move(data), error);
}

void Tokenizer::EncodeOrdinary(std::string_view text,
                               std::vector<TokenId>* out) const {
  std::vector<Piece> pieces;
  pieces.reserve(text.size() / 4 + 1);
  SplitPieces(text, &pieces);

  for (const auto& piece : pieces) {
    // The piece's bytes, each as its byte-level unicode spelling.
    std::vector<std::string> parts;
    parts.reserve(piece.second - piece.first);
    for (std::uint32_t i = piece.first; i < piece.second; ++i) {
      parts.push_back(byte_to_char_[static_cast<unsigned char>(text[i])]);
    }

    // Repeatedly merge the adjacent pair with the lowest rank (the reference
    // BPE order: lowest rank first, ties leftmost).
    std::string key;
    while (parts.size() > 1) {
      std::size_t best = parts.size();
      std::uint32_t best_rank = 0xFFFFFFFFu;
      for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        key.clear();
        key.append(parts[i]);
        key.push_back(' ');
        key.append(parts[i + 1]);
        const auto it = merges_.find(key);
        if (it != merges_.end() && it->second < best_rank) {
          best_rank = it->second;
          best = i;
        }
      }
      if (best == parts.size()) {
        break;
      }
      const auto left = parts[best];
      const auto right = parts[best + 1];
      std::vector<std::string> merged;
      merged.reserve(parts.size());
      for (std::size_t i = 0; i < parts.size();) {
        if (i + 1 < parts.size() && parts[i] == left && parts[i + 1] == right) {
          merged.push_back(left + right);
          i += 2;
        } else {
          merged.push_back(std::move(parts[i]));
          ++i;
        }
      }
      parts = std::move(merged);
    }

    for (const auto& symbol : parts) {
      const auto it = by_token_.find(symbol);
      if (it != by_token_.end()) {
        out->push_back(it->second);
        continue;
      }
      // A merged symbol with no vocabulary entry is a corrupt tokenizer; emit
      // its bytes one at a time so the output stays decodable and the error is
      // visible as text rather than a silent drop.
      for (const auto ch : symbol) {
        const auto single = by_token_.find(std::string(1, ch));
        if (single != by_token_.end()) {
          out->push_back(single->second);
        }
      }
    }
  }
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text,
                                       bool parse_special) const {
  std::vector<TokenId> out;
  const std::string normalized = NormalizeNfc(text);
  const std::string_view input(normalized);

  if (!parse_special || specials_.empty()) {
    EncodeOrdinary(input, &out);
    return out;
  }

  std::size_t position = 0;
  while (position < input.size()) {
    std::size_t best_position = std::string_view::npos;
    const Special* best = nullptr;
    for (const auto& special : specials_) {
      const auto found = input.find(special.literal, position);
      if (found == std::string_view::npos) {
        continue;
      }
      if (best_position == std::string_view::npos || found < best_position ||
          (found == best_position && best != nullptr &&
           special.literal.size() > best->literal.size())) {
        best_position = found;
        best = &special;
      }
    }
    if (best == nullptr) {
      EncodeOrdinary(input.substr(position), &out);
      break;
    }
    if (best_position > position) {
      EncodeOrdinary(input.substr(position, best_position - position), &out);
    }
    out.push_back(best->id);
    position = best_position + best->literal.size();
  }
  return out;
}

std::string Tokenizer::Decode(const std::vector<TokenId>& ids,
                              bool keep_control) const {
  std::string out;
  for (const auto id : ids) {
    if (id >= tokens_.size()) {
      continue;
    }
    if (special_codepoint_[id] && !keep_control) {
      continue;
    }
    const auto& token = tokens_[id];
    for (std::size_t i = 0; i < token.size();) {
      const auto decoded = unicode::Decode(token, i);
      const auto byte = decoded.cp < std::size(char_to_byte_)
                            ? char_to_byte_[decoded.cp]
                            : -1;
      if (byte < 0) {
        out.append(token, i, decoded.len);
      } else {
        out.push_back(static_cast<char>(byte));
      }
      i += decoded.len;
    }
  }
  return out;
}

std::size_t Tokenizer::VocabSize() const { return tokens_.size(); }
TokenId Tokenizer::Bos() const { return bos_; }
TokenId Tokenizer::Eos() const { return eos_; }
TokenId Tokenizer::Pad() const { return pad_; }
const std::string& Tokenizer::ChatTemplate() const { return chat_template_; }

std::optional<TokenId> Tokenizer::TokenFor(std::string_view literal) const {
  const auto it = by_token_.find(std::string(literal));
  if (it == by_token_.end()) {
    return std::nullopt;
  }
  return it->second;
}

#if defined(LING3_HAVE_ICU)
namespace {

/// UTF-8 to UTF-16; false when the input is not valid UTF-8.
bool ToUtf16(std::string_view utf8, std::u16string* out) {
  UErrorCode status = U_ZERO_ERROR;
  int32_t length = 0;
  u_strFromUTF8(nullptr, 0, &length, utf8.data(),
                static_cast<int32_t>(utf8.size()), &status);
  if (status != U_BUFFER_OVERFLOW_ERROR || length < 0) {
    return false;
  }
  status = U_ZERO_ERROR;
  out->assign(static_cast<std::size_t>(length), u'\0');
  u_strFromUTF8(out->data(), length, &length, utf8.data(),
                static_cast<int32_t>(utf8.size()), &status);
  if (U_FAILURE(status)) {
    return false;
  }
  out->resize(static_cast<std::size_t>(length));
  return true;
}

bool ToUtf8(const std::u16string& utf16, std::string* out) {
  UErrorCode status = U_ZERO_ERROR;
  int32_t length = 0;
  u_strToUTF8(nullptr, 0, &length, utf16.data(),
              static_cast<int32_t>(utf16.size()), &status);
  if (status != U_BUFFER_OVERFLOW_ERROR || length < 0) {
    return false;
  }
  status = U_ZERO_ERROR;
  out->assign(static_cast<std::size_t>(length), '\0');
  u_strToUTF8(out->data(), length, &length, utf16.data(),
              static_cast<int32_t>(utf16.size()), &status);
  if (U_FAILURE(status)) {
    return false;
  }
  out->resize(static_cast<std::size_t>(length));
  return true;
}

}  // namespace

std::string NormalizeNfc(std::string_view text) {
  std::u16string utf16;
  if (!ToUtf16(text, &utf16)) {
    return std::string(text);
  }
  UErrorCode status = U_ZERO_ERROR;
  const UNormalizer2* normalizer = unorm2_getNFCInstance(&status);
  if (U_FAILURE(status) || normalizer == nullptr) {
    return std::string(text);
  }
  status = U_ZERO_ERROR;
  const int32_t needed =
      unorm2_normalize(normalizer, utf16.data(),
                       static_cast<int32_t>(utf16.size()), nullptr, 0, &status);
  if (status != U_BUFFER_OVERFLOW_ERROR || needed < 0) {
    return std::string(text);
  }
  status = U_ZERO_ERROR;
  std::u16string normalized(static_cast<std::size_t>(needed), u'\0');
  unorm2_normalize(normalizer, utf16.data(), static_cast<int32_t>(utf16.size()),
                   normalized.data(), needed, &status);
  if (U_FAILURE(status)) {
    return std::string(text);
  }
  normalized.resize(static_cast<std::size_t>(needed));
  std::string utf8;
  if (!ToUtf8(normalized, &utf8)) {
    return std::string(text);
  }
  return utf8;
}

bool HasIcuNormalization() { return true; }
#else
std::string NormalizeNfc(std::string_view text) { return std::string(text); }
bool HasIcuNormalization() { return false; }
#endif

}  // namespace gufo::ext::ling3_air::text
