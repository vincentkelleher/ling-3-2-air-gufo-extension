#include "src/text/unicode.hpp"

namespace gufo::ext::ling3_air::text::unicode {
namespace {

bool InRange(char32_t cp, const CodepointRange& r) noexcept {
  return cp >= r.first && cp <= r.last;
}

}  // namespace

std::uint8_t Lookup(char32_t cp) noexcept {
  if (cp > 0x10FFFF) {
    return 0;
  }
  // kRanges is sorted and disjoint; binary search the last range whose first
  // is <= cp.
  std::size_t lo = 0;
  std::size_t hi = kRangeCount;
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (kRanges[mid].first <= cp) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo == 0) {
    return 0;
  }
  const auto& range = kRanges[lo - 1];
  return InRange(cp, range) ? range.code : 0;
}

Decoded Decode(std::string_view s, std::size_t pos) noexcept {
  const auto lead = static_cast<unsigned char>(s[pos]);
  if (lead < 0x80) {
    return {lead, 1};
  }
  std::size_t need = 0;
  char32_t cp = 0;
  if ((lead & 0xE0) == 0xC0) {
    need = 1;
    cp = lead & 0x1F;
  } else if ((lead & 0xF0) == 0xE0) {
    need = 2;
    cp = lead & 0x0F;
  } else if ((lead & 0xF8) == 0xF0) {
    need = 3;
    cp = lead & 0x07;
  } else {
    return {0x110000 + lead, 1};
  }
  if (pos + need >= s.size()) {
    return {0x110000 + lead, 1};
  }
  for (std::size_t k = 1; k <= need; ++k) {
    const auto byte = static_cast<unsigned char>(s[pos + k]);
    if ((byte & 0xC0) != 0x80) {
      return {0x110000 + lead, 1};
    }
    cp = (cp << 6) | (byte & 0x3F);
  }
  return {cp, static_cast<std::uint8_t>(need + 1)};
}

std::size_t Codepoints(std::string_view s) noexcept {
  std::size_t count = 0;
  for (std::size_t i = 0; i < s.size();) {
    i += Decode(s, i).len;
    ++count;
  }
  return count;
}

}  // namespace gufo::ext::ling3_air::text::unicode
