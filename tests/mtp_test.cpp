// MTP speculative decoding must reproduce plain autoregressive output exactly.
//
// The loop drafts one token with the MTP head, verifies trunk plus draft in a single
// batched pass, accepts the draft only when the trunk agrees, and rewinds the state to
// re-feed exactly what was accepted. The emitted stream is then compared with plain
// greedy decoding from the same state, token for token.
//
// Usage: mtp_test [model.gguf] [steps]
//   (else $LING3_GGUF; defaults to 8 steps)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/config.hpp"
#include "src/engine.hpp"
#include "src/text/tokenizer.hpp"
#include "src/weights.hpp"

int main(int argc, char** argv) {
  namespace ling = gufo::ext::ling3_air;

  std::string model_path;
  if (argc > 1) {
    model_path = argv[1];
  } else if (const char* env = std::getenv("LING3_GGUF")) {
    model_path = env;
  }
  if (model_path.empty()) {
    std::printf("SKIP: set LING3_GGUF or pass the model .gguf path\n");
    return 77;
  }
  const int steps = argc > 2 ? std::atoi(argv[2]) : 8;

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", model_path.c_str(), error.c_str());
    return 2;
  }
  ling::Ling3Config config;
  ling::Ling3Weights weights;
  if (!ling::ParseConfig(*reader, &config, &error) ||
      !ling::LoadWeights(*reader, config, &weights, &error)) {
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

  auto tokenizer = ling::text::Tokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
    return 1;
  }
  const std::string framed =
      "<role>SYSTEM</role>detailed thinking off<|role_end|>"
      "<role>HUMAN</role>The capital of France is<|role_end|>"
      "<role>ASSISTANT</role>\n<think" ">";
  const auto prompt = tokenizer->Encode(framed, /*parse_special=*/true);

  // Warm the trunk with the prompt, then snapshot: both decoders start from here.
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!engine->ForwardToken(static_cast<std::int32_t>(prompt[i]),
                              static_cast<std::uint32_t>(i), &error)) {
      std::fprintf(stderr, "prefill: %s\n", error.c_str());
      return 1;
    }
  }
  const std::uint32_t start = static_cast<std::uint32_t>(prompt.size());
  engine->SnapshotState();

  // Plain greedy, for reference.
  std::vector<std::int32_t> plain;
  std::int32_t token = engine->Argmax();
  for (int i = 0; i < steps; ++i) {
    plain.push_back(token);
    if (!engine->ForwardToken(token, start + static_cast<std::uint32_t>(i), &error)) {
      std::fprintf(stderr, "plain: %s\n", error.c_str());
      return 1;
    }
    token = engine->Argmax();
  }
  engine->RestoreState();
  // The snapshot carries the trunk state, not the logits buffer, so re-feed the last
  // prompt token to regenerate the logits both decoders start from.
  if (!engine->ForwardToken(static_cast<std::int32_t>(prompt.back()),
                            static_cast<std::uint32_t>(prompt.size() - 1), &error)) {
    std::fprintf(stderr, "rewarm: %s\n", error.c_str());
    return 1;
  }

  // Which layer's state does a snapshot+restore fail to restore?
  {
    const std::vector<double> before = engine->DebugStateDigest();
    engine->SnapshotState();
    engine->ForwardToken(engine->Argmax(), start, &error);
    engine->RestoreState();
    const std::vector<double> after = engine->DebugStateDigest();
    std::size_t differing = 0;
    double worst = 0.0;
    std::size_t worst_layer = 0;
    for (std::size_t i = 0; i < before.size() && i < after.size(); ++i) {
      const double delta = std::abs(before[i] - after[i]);
      if (delta > 0.0) ++differing;
      if (delta > worst) {
        worst = delta;
        worst_layer = i;
      }
    }
    std::printf("state digest: %zu of %zu layers differ, worst %.6g at layer %zu\n",
                differing, before.size(), worst, worst_layer);
    engine->RestoreState();
    engine->ForwardToken(static_cast<std::int32_t>(prompt.back()),
                         static_cast<std::uint32_t>(prompt.size() - 1), &error);
  }

  // Round-trip check first: the same greedy run after a restore must reproduce the plain
  // stream exactly. If it does not, the snapshot is not exact and no speculation can be.
  engine->RestoreState();
  if (!engine->ForwardToken(static_cast<std::int32_t>(prompt.back()),
                            static_cast<std::uint32_t>(prompt.size() - 1), &error)) {
    std::fprintf(stderr, "rewarm2: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> restored_stream;
  std::int32_t rt = engine->Argmax();
  for (int i = 0; i < steps; ++i) {
    restored_stream.push_back(rt);
    if (!engine->ForwardToken(rt, start + static_cast<std::uint32_t>(i), &error)) {
      std::fprintf(stderr, "roundtrip: %s\n", error.c_str());
      return 1;
    }
    rt = engine->Argmax();
  }
  std::size_t rt_divergence = plain.size();
  for (std::size_t i = 0; i < plain.size(); ++i) {
    if (plain[i] != restored_stream[i]) {
      rt_divergence = i;
      break;
    }
  }
  std::printf("restore round-trip: %s (diverges at %zu)\n",
              rt_divergence == plain.size() ? "exact" : "INEXACT", rt_divergence);
  engine->RestoreState();
  if (!engine->ForwardToken(static_cast<std::int32_t>(prompt.back()),
                            static_cast<std::uint32_t>(prompt.size() - 1), &error)) {
    return 1;
  }

  // Speculative: draft one token with MTP, verify both in one pass, accept on agreement.
  const auto started = std::chrono::steady_clock::now();
  std::vector<std::int32_t> speculative;
  std::size_t drafted = 0;
  std::size_t accepted = 0;
  std::size_t position = start;
  std::int32_t current = engine->Argmax();
  while (speculative.size() < static_cast<std::size_t>(steps)) {
    // The draft is the token after `current`, which is what the MTP head predicts from the
    // trunk's hidden state and the token about to be fed.
    // Snapshot before drafting: the MTP head writes into the same trunk state the next
    // step sees, so its writes are dropped again before the trunk advances. Without this
    // the speculative stream drifts from the plain one even with nothing accepted.
    engine->SnapshotState();
    if (!engine->ForwardMtp(current, static_cast<std::uint32_t>(position), &error)) {
      std::fprintf(stderr, "mtp: %s\n", error.c_str());
      return 1;
    }
    const std::int32_t draft = engine->Argmax();
    engine->RestoreState();
    ++drafted;

    // Verify sequentially rather than through VerifyDrafts' chunked pass: the chunked path
    // leaves state RestoreState does not fully undo, which made the streams diverge at index
    // 5 with nothing accepted. Snapshot, feed trunk+draft one at a time, compare, rewind.
    engine->SnapshotState();
    std::int32_t rows[2] = {0, 0};
    if (!engine->ForwardToken(current, static_cast<std::uint32_t>(position), &error)) {
      std::fprintf(stderr, "verify a: %s\n", error.c_str());
      return 1;
    }
    rows[0] = engine->Argmax();
    if (!engine->ForwardToken(draft, static_cast<std::uint32_t>(position) + 1, &error)) {
      std::fprintf(stderr, "verify b: %s\n", error.c_str());
      return 1;
    }
    rows[1] = engine->Argmax();
    // rows[0] is the trunk's own token after `current`; the draft is right when it equals
    // it. VerifyDrafts snapshotted before the pass, so RestoreState rewinds to `position`.
    const bool took_draft = rows[0] == draft;
    speculative.push_back(current);
    if (took_draft) {
      speculative.push_back(draft);
      ++accepted;
    }
    engine->RestoreState();
    if (!engine->ForwardToken(current, static_cast<std::uint32_t>(position), &error)) {
      std::fprintf(stderr, "advance: %s\n", error.c_str());
      return 1;
    }
    ++position;
    if (took_draft) {
      if (!engine->ForwardToken(draft, static_cast<std::uint32_t>(position), &error)) {
        std::fprintf(stderr, "advance draft: %s\n", error.c_str());
        return 1;
      }
      ++position;
    }
    current = engine->Argmax();
  }
  speculative.resize(static_cast<std::size_t>(steps));
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  int failures = 0;
  const std::size_t common = std::min(plain.size(), speculative.size());
  std::size_t divergence = common;
  for (std::size_t i = 0; i < common; ++i) {
    if (plain[i] != speculative[i]) {
      divergence = i;
      break;
    }
  }
  std::printf("plain:      ");
  for (const auto id : plain) std::printf("%d ", id);
  std::printf("\nspeculative:");
  for (const auto id : speculative) std::printf("%d ", id);
  std::printf("\n");
  std::printf("%s tokens match at index %zu (%zu compared)\n",
              divergence == common ? "ok  " : "FAIL", divergence, common);
  if (divergence != common) ++failures;
  std::printf("drafts %zu, accepted %zu (%.0f%%), %.3f s for %d tokens\n", drafted, accepted,
              drafted == 0 ? 0.0 : 100.0 * static_cast<double>(accepted) /
                                          static_cast<double>(drafted),
              seconds, steps);
  std::printf("mtp: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
