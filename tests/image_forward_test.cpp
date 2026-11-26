// Image conditioning end to end: preprocess -> vision tower -> embeddings -> trunk.
//
// Usage: image_forward_test [model.gguf] [mmproj.gguf] [image] [steps]
//   (else $LING3_GGUF, $LING3_MMPROJ, $LING3_IMAGE; defaults: 12 steps)
//
// Two things are checked, then a generation is printed:
//   1. `ForwardEmbedding(token embedding)` reproduces `ForwardToken` exactly, so the
//      entry point that carries vision rows does not change the trunk.
//   2. The number of `<|image_pad|>` markers the chat template produces equals the
//      tower's `tokens_per_image()`, so the rows land on the markers one for one.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

// Gufo's PNG/JPEG/WebP decoder lives in the core tree, which extensions already
// include for internal headers (cache, ext runtime) as Gufo's own models do.
#include "src/core/image.hpp"

#include "src/config.hpp"
#include "src/engine.hpp"
#include "src/preprocess.hpp"
#include "src/text/tokenizer.hpp"
#include "src/vision.hpp"
#include "src/weights.hpp"

namespace {

int Failures = 0;

void Check(bool ok, const char* what, double measured, double limit) {
  std::printf("%s %-38s %.3g (limit %.1g)\n", ok ? "ok  " : "FAIL", what, measured, limit);
  if (!ok) {
    ++Failures;
  }
}

std::string EnvOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string(fallback == nullptr ? "" : fallback)
                          : std::string(value);
}

/// A deterministic image with structure (vertical colour bands), used when no image
/// path is given: enough for the tower to encode something that is not noise.
std::vector<std::uint8_t> SyntheticImage(int width, int height) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3;
      const int band = x * 4 / width;
      rgb[i + 0] = static_cast<std::uint8_t>(60 + 45 * band);
      rgb[i + 1] = static_cast<std::uint8_t>(220 - 40 * band);
      rgb[i + 2] = static_cast<std::uint8_t>(x * 255 / width);
    }
  }
  return rgb;
}

}  // namespace

int main(int argc, char** argv) {
  namespace ling = gufo::ext::ling3_air;
  const std::string model_path = argc > 1 ? argv[1] : EnvOr("LING3_GGUF", "");
  const std::string mmproj_path = argc > 2 ? argv[2] : EnvOr("LING3_MMPROJ", "");
  const std::string image_path = argc > 3 ? argv[3] : EnvOr("LING3_IMAGE", "");
  int steps = argc > 4 ? std::atoi(argv[4]) : 12;
  if (const char* env_steps = std::getenv("LING3_STEPS")) {
    steps = std::atoi(env_steps);
  }
  if (model_path.empty() || mmproj_path.empty()) {
    std::printf("SKIP: set LING3_GGUF and LING3_MMPROJ (or pass both paths)\n");
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(), error.c_str());
    return 2;
  }
  ling::Ling3Config config;
  if (!ling::ParseConfig(*reader, &config, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  ling::Ling3Weights weights;
  if (!ling::LoadWeights(*reader, config, &weights, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  ling::EngineOptions options;
  options.max_context = 4096;
  auto engine = ling::Engine::Create(*reader, config, weights, options, &error);
  if (!engine) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  const auto mmproj = gufo::core::GgufReader::OpenFile(mmproj_path, &error);
  if (!mmproj) {
    std::fprintf(stderr, "cannot open %s: %s\n", mmproj_path.c_str(), error.c_str());
    return 2;
  }
  const auto vision = ling::VisionEncoder::Create(*mmproj, &error);
  if (!vision) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const int side = static_cast<int>(vision->image_side());
  std::printf("tower: input %dx%d, %u rows per image, text dim %u\n", side, side,
              vision->tokens_per_image(), vision->text_dim());

  // --- the image, resized and encoded ---------------------------------------
  int width = 96;
  int height = 96;
  std::vector<std::uint8_t> rgb;
  if (!image_path.empty()) {
    try {
      const auto bytes = gufo::core::ReadImageFile(image_path);
      const auto decoded = gufo::core::DecodeImage(bytes);
      width = static_cast<int>(decoded.width);
      height = static_cast<int>(decoded.height);
      rgb = decoded.pixels;
      std::printf("image: %s (%dx%d)\n", image_path.c_str(), width, height);
    } catch (const std::exception& failure) {
      std::fprintf(stderr, "cannot read %s: %s\n", image_path.c_str(), failure.what());
      return 2;
    }
  } else {
    rgb = SyntheticImage(width, height);
    std::printf("image: synthetic %dx%d colour bands\n", width, height);
  }
  // `LING3_GRAY` reproduces llama-mtmd-debug's "gray" input, which fills the already
  // normalised CHW tensor with 0.5 rather than a grey photograph. It makes our tower
  // directly comparable to that tool's per-node sums.
  const bool gray = std::getenv("LING3_GRAY") != nullptr;
  int gray_side = side;
  if (const char* override_side = std::getenv("LING3_GRAY_SIDE")) {
    gray_side = std::atoi(override_side);
  }
  std::vector<float> chw;
  if (gray) {
    chw.assign(static_cast<std::size_t>(3) * gray_side * gray_side, 0.5f);
  } else {
    chw = ling::ResizeToChw(rgb.data(), width, height, side, ling::ImageNorm{});
  }
  std::vector<float> image_rows;
  const auto encode_start = std::chrono::steady_clock::now();
  const int grid = (gray ? gray_side : side) / 16;
  if (!vision->EncodeImage(chw.data(), static_cast<std::uint32_t>(grid),
                           static_cast<std::uint32_t>(grid), &image_rows, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const double encode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start)
          .count();
  std::printf("encoded %zu rows (%u per row) in %.2f s\n",
              image_rows.size() / vision->text_dim(), vision->text_dim(),
              encode_seconds);
  if (gray) {
    double total = 0.0;
    double magnitude = 0.0;
    for (const float value : image_rows) {
      total += value;
      magnitude = std::max(magnitude, std::fabs(static_cast<double>(value)));
    }
    // llama-mtmd-debug on the same input: node_819 (mm.2 + bias) sums to
    // -1310.917358. The sum is order-invariant, so it compares merged rows directly.
    std::printf("gray embedding sum %.6f, max |v| %.4f\n", total, magnitude);
    std::printf("gray first4:");
    for (int i = 0; i < 4 && i < static_cast<int>(image_rows.size()); ++i) {
      std::printf(" %.4f", image_rows[i]);
    }
    std::printf("\n");
    std::printf("gray last4:");
    for (int i = 0; i < 4; ++i) {
      std::printf(" %.4f",
                  image_rows[image_rows.size() - 4 + static_cast<std::size_t>(i)]);
    }
    std::printf("\n");
    if (gray_side != side) {
      // A cropped grid exists to compare the tower against llama-mtmd-debug's small
      // inputs; the prompt and language-model stages need a full 768x768 image.
      std::printf("image_forward: %d failures (tower only)\n", Failures);
      return Failures == 0 ? 0 : 1;
    }
  }

  // --- 1. the embedding entry point must equal the token path ----------------
  const std::int32_t probe = 1000;
  std::vector<float> probe_embedding(config.n_embd);
  engine->DebugEmbedding(probe, probe_embedding.data());
  std::vector<float> logits_token(engine->VocabSize());
  std::vector<float> logits_embedding(engine->VocabSize());
  engine->SnapshotState();
  if (!engine->ForwardToken(probe, 0, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const std::int32_t argmax_token = engine->Argmax();
  engine->ReadLogits(logits_token.data());
  engine->RestoreState();
  engine->SnapshotState();
  if (!engine->ForwardEmbedding(probe_embedding.data(), 0, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const std::int32_t argmax_embedding = engine->Argmax();
  engine->ReadLogits(logits_embedding.data());
  engine->RestoreState();
  Check(argmax_token == argmax_embedding, "ForwardEmbedding argmax equals ForwardToken",
        static_cast<double>(argmax_embedding), static_cast<double>(argmax_token));
  double worst_logits = 0.0;
  for (std::size_t i = 0; i < logits_token.size(); ++i) {
    worst_logits = std::max(worst_logits,
                            std::fabs(static_cast<double>(logits_embedding[i]) -
                                      logits_token[i]));
  }
  Check(worst_logits == 0.0, "ForwardEmbedding logits are bit-identical",
        worst_logits, 0.0);

  // --- 2. the prompt's image markers must match the tower's row count --------
  auto tokenizer = ling::text::Tokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
    return 1;
  }
  const std::string prompt =
      EnvOr("LING3_IMAGE_PROMPT", "Describe this image in one sentence.");
  // The chat template spells the image as one marker per tower row inside the vision
  // delimiters, so the prompt is built with the real markers rather than a placeholder.
  std::string marker_count_probe =
      "<role>SYSTEM</role>detailed thinking off<|role_end|>"
      "<role>HUMAN</role><|vision_start|>";
  for (std::uint32_t i = 0; i < vision->tokens_per_image(); ++i) {
    marker_count_probe += "<|image_pad|>";
  }
  marker_count_probe += "<|vision_end|>" + prompt + "<|role_end|><role>ASSISTANT</role>\n<think" ">";
  const auto tokens = tokenizer->Encode(marker_count_probe, /*parse_special=*/true);
  std::size_t markers = 0;
  std::size_t first_marker = 0;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (tokenizer->Decode({tokens[i]}, true) == "<|image_pad|>") {
      if (markers == 0) {
        first_marker = i;
      }
      ++markers;
    }
  }
  std::printf("prompt: %zu tokens, %zu image markers (first at %zu)\n", tokens.size(),
              markers, first_marker);
  Check(markers == vision->tokens_per_image(), "image markers equal tower rows",
        static_cast<double>(markers), static_cast<double>(vision->tokens_per_image()));

  // --- prefill: vision rows where the markers are, tokens elsewhere ----------
  const std::size_t image_dim = vision->text_dim();
  const auto prefill_start = std::chrono::steady_clock::now();
  std::size_t row = 0;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const bool is_marker =
        i >= first_marker && i < first_marker + markers;
    if (is_marker) {
      if (!engine->ForwardEmbedding(
              image_rows.data() + row * image_dim, static_cast<std::uint32_t>(i),
              &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
      }
      ++row;
    } else if (!engine->ForwardToken(static_cast<std::int32_t>(tokens[i]),
                                     static_cast<std::uint32_t>(i), &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
  }
  const double prefill_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start)
          .count();
  std::printf("prefill: %zu steps in %.2f s (%.2f tok/s)\n", tokens.size(),
              prefill_seconds, tokens.size() / prefill_seconds);

  // --- greedy continuation ---------------------------------------------------
  std::vector<ling::text::TokenId> produced;
  std::int32_t next = engine->Argmax();
  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < steps; ++step) {
    produced.push_back(next);
    if (!engine->ForwardToken(next, static_cast<std::uint32_t>(tokens.size() + step),
                              &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    next = engine->Argmax();
  }
  const double decode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start)
          .count();
  std::printf("decode: %d steps in %.2f s (%.2f tok/s)\n", steps, decode_seconds,
              steps / decode_seconds);
  std::printf("text: %s\n", tokenizer->Decode(produced, false).c_str());

  std::printf("image_forward: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
