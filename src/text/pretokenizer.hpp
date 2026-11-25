// Ling 3.2 pre-tokenizer: the HF `Split` pattern of tokenizer.json.
//
//   '(?i:[sdmt]|ll|ve|re)
//   |[^\r\n\p{L}\p{N}]?+\p{L}+
//   |\p{N}
//   | ?[^\s\p{L}\p{N}]++[\r\n]*
//   |\s*[\r\n]
//   |\s+(?!\S)
//   |\s+
//
// Alternatives are tried left to right; the two possessive quantifiers never
// give a character back, and the two whitespace alternatives keep the
// backtracking behaviour the reference implementation has (see SplitPieces).
#pragma once

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::ext::ling3_air::text {

/// One pre-tokenized piece: [begin, end) byte offsets into the input.
using Piece = std::pair<std::uint32_t, std::uint32_t>;

/// Splits `text` into the pieces the byte-level BPE encoder consumes.
/// Pieces are contiguous, non-empty and cover the whole input.
void SplitPieces(std::string_view text, std::vector<Piece>* out);

}  // namespace gufo::ext::ling3_air::text
