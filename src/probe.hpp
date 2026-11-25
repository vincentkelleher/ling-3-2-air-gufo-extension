#pragma once

#include <string_view>

namespace gufo::core {
class GgufReader;
}  // namespace gufo::core

namespace gufo::ext::ling3_air {

/// The GGUF architecture these weights declare. Ling 3.2 Air is a
/// `bailing_moe_v3_vl` checkpoint with a hybrid KDA/MLA trunk; llama.cpp
/// writes and reads it under the `bailingmoe3` architecture name.
inline constexpr std::string_view kGgufArchitecture = "bailingmoe3";

/// True when `reader` describes a Ling 3.2 Air trunk.
///
/// Host-only and GPU-free: reads the architecture key and the one shape that
/// distinguishes the 128B Air checkpoint (42 blocks, 512 experts) from the
/// smaller sibling it shares an architecture name with. A file this probe
/// cannot identify is not claimed.
[[nodiscard]] bool ProbeLing3Gguf(const core::GgufReader& reader);

/// Directory or file form used by `TextProvider::probe`.
[[nodiscard]] bool ProbeLing3(std::string_view model_root);

}  // namespace gufo::ext::ling3_air
