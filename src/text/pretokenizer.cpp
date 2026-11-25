#include "src/text/pretokenizer.hpp"

#include <cstddef>

#include "src/text/unicode.hpp"

namespace gufo::ext::ling3_air::text {
namespace {

namespace uc = unicode;

bool IsNewline(char32_t cp) noexcept { return cp == '\r' || cp == '\n'; }

struct Scanner {
  std::string_view s;

  [[nodiscard]] bool More(std::size_t p) const noexcept { return p < s.size(); }

  // `'(?i:[sdmt]|ll|ve|re)` at i; returns the end, or i when absent.
  [[nodiscard]] std::size_t Contraction(std::size_t i) const noexcept {
    if (i + 1 >= s.size() || s[i] != '\'') {
      return i;
    }
    const auto fold = [&](std::size_t p) -> int {
      if (p >= s.size()) {
        return -1;
      }
      const auto c = static_cast<unsigned char>(s[p]);
      return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };
    const int a = fold(i + 1);
    const int b = fold(i + 2);
    if (a == 's' || a == 'd' || a == 'm' || a == 't') {
      return i + 2;
    }
    if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e')) {
      return i + 3;
    }
    return i;
  }

  // `\s` run boundaries and the end of its last CR/LF.
  struct Run {
    std::size_t end;
    std::size_t last_newline_end;
    std::size_t last_codepoint_start;
  };

  [[nodiscard]] Run WhitespaceRun(std::size_t i) const noexcept {
    Run run{i, i, i};
    std::size_t p = i;
    while (More(p)) {
      const auto d = uc::Decode(s, p);
      if (!uc::IsSpace(d.cp)) {
        break;
      }
      run.last_codepoint_start = p;
      p += d.len;
      run.end = p;
      if (IsNewline(d.cp)) {
        run.last_newline_end = p;
      }
    }
    return run;
  }

  // Alternatives 5 to 7. `i` is the first byte of a white-space code point.
  //
  // `\s*[\r\n]` matches through the last CR/LF of the run, because the greedy
  // `\s*` runs to the end and then gives characters back only until the final
  // alternative can match, i.e. one. `\s+(?!\S)` matches the whole run when it
  // ends the text, and otherwise all but its last code point, which the next
  // piece takes as the optional prefix of a word or a punctuation run. `\s+`
  // is the single-code-point fallback.
  [[nodiscard]] std::size_t Whitespace(std::size_t i) const noexcept {
    const Run run = WhitespaceRun(i);
    if (run.last_newline_end > i) {
      return run.last_newline_end;
    }
    if (!More(run.end)) {
      return run.end;
    }
    if (run.last_codepoint_start > i) {
      return run.last_codepoint_start;
    }
    return run.end;
  }

  // Returns the end of the piece starting at `i` (always > i).
  [[nodiscard]] std::size_t Next(std::size_t i) const noexcept {
    // 1: contraction.
    if (const auto e = Contraction(i); e > i) {
      return e;
    }

    const auto first = uc::Decode(s, i);
    const bool first_is_space = uc::IsSpace(first.cp);
    const bool first_is_letter = uc::IsLetter(first.cp);
    const bool first_is_number = uc::IsNumber(first.cp);
    const bool first_is_newline = IsNewline(first.cp);

    // 2: `[^\r\n\p{L}\p{N}]?+\p{L}+`. The optional prefix is possessive: when
    // it is consumed and no letter follows, this alternative fails and the
    // characters fall through to 4.
    if (!first_is_newline && !first_is_letter && !first_is_number) {
      std::size_t p = NextLetters(i + first.len);
      if (p > i + first.len) {
        return p;
      }
    }
    if (first_is_letter) {
      return NextLetters(i);
    }

    // 3: one number.
    if (first_is_number) {
      return i + first.len;
    }

    // 4: ` ?[^\s\p{L}\p{N}]++[\r\n]*`.
    {
      std::size_t p = i + (first.cp == ' ' ? 1 : 0);
      const std::size_t start = p;
      while (More(p)) {
        const auto d = uc::Decode(s, p);
        if (uc::IsSpace(d.cp) || uc::IsLetter(d.cp) || uc::IsNumber(d.cp)) {
          break;
        }
        p += d.len;
      }
      if (p > start) {
        while (More(p) && (s[p] == '\r' || s[p] == '\n')) {
          ++p;
        }
        return p;
      }
    }

    // 5 to 7. Every remaining case starts with white space.
    if (first_is_space) {
      return Whitespace(i);
    }

    // Not reachable for well-formed input: a byte that is not white space,
    // letter, number or newline is matched by 4. Consume one code point so the
    // scanner always advances.
    return i + first.len;
  }

  [[nodiscard]] std::size_t NextLetters(std::size_t p) const noexcept {
    while (More(p)) {
      const auto d = uc::Decode(s, p);
      if (!uc::IsLetter(d.cp)) {
        break;
      }
      p += d.len;
    }
    return p;
  }
};

}  // namespace

void SplitPieces(std::string_view text, std::vector<Piece>* out) {
  Scanner scanner{text};
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t end = scanner.Next(i);
    out->emplace_back(static_cast<std::uint32_t>(i),
                      static_cast<std::uint32_t>(end));
    i = end;
  }
}

}  // namespace gufo::ext::ling3_air::text
