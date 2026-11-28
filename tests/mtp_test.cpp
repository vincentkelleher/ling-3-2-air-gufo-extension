// Restore round-trip, state exactness and MTP speculative equivalence.
//
// One rule throughout: the engine owns a SINGLE snapshot buffer, and `VerifyDrafts` takes
// its own. So every probe here takes its own checkpoint immediately before use, and every
// restore is followed by re-feeding the last prompt token, because the snapshot carries the
// trunk state but not the logits buffer.
//
// Usage: mtp_test [model.gguf] [steps]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/config.hpp"
#include "src/engine.hpp"
#include "src/text/tokenizer.hpp"
#include "src/weights.hpp"

namespace {

using Model = gufo::ext::ling3_air::Engine;

int Failures = 0;

void Check(bool ok, const char* what, double measured, double limit) {
  std::printf("%s %-36s %.3g (limit %.1g)\n", ok ? "ok  " : "FAIL", what, measured, limit);
  if (!ok) {
    ++Failures;
  }
}

/// Greedy `steps` tokens from the current state, recording logits per step.
bool Greedy(Model& engine, std::uint32_t start, int steps, std::vector<std::int32_t>* tokens,
            std::vector<std::vector<float>>* logits, std::string* error) {
  std::int32_t token = engine.Argmax();
  std::vector<float> row(engine.VocabSize());
  for (int i = 0; i < steps; ++i) {
    tokens->push_back(token);
    if (!engine.ForwardToken(token, start + static_cast<std::uint32_t>(i), error)) {
      return false;
    }
    if (logits != nullptr) {
      engine.ReadLogits(row.data());
      logits->push_back(row);
    }
    token = engine.Argmax();
  }
  return true;
}

/// One speculative step in the serving shape: draft with the MTP head, verify
/// {current, draft} in one chunked pass, accept only an exact greedy match, undo both
/// passes and re-feed exactly what was accepted. The engine owns one snapshot buffer,
/// so the draft's checkpoint is dropped before VerifyDrafts takes its own; no
/// checkpoint is left live across a call.
bool SpeculativeStep(Model& engine, std::uint32_t position, std::int32_t current,
                     std::int32_t* draft, bool* accepted, std::string* error) {
  engine.SnapshotState();
  if (!engine.ForwardMtp(current, position, error)) {
    engine.RestoreState();
    return false;
  }
  *draft = engine.Argmax();
  engine.RestoreState();
  const std::int32_t pair[2] = {current, *draft};
  std::int32_t rows[2] = {0, 0};
  if (!engine.VerifyDrafts(pair, 2, position, rows, error)) {
    engine.RestoreState();
    return false;
  }
  *accepted = rows[0] == *draft;
  engine.RestoreState();
  if (!engine.ForwardToken(current, position, error)) return false;
  if (*accepted && !engine.ForwardToken(*draft, position + 1, error)) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
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
  gufo::ext::ling3_air::Ling3Config config;
  gufo::ext::ling3_air::Ling3Weights weights;
  if (!gufo::ext::ling3_air::ParseConfig(*reader, &config, &error) ||
      !gufo::ext::ling3_air::LoadWeights(*reader, config, &weights, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  gufo::ext::ling3_air::EngineOptions options;
  options.max_context = 4096;
  auto engine = Model::Create(*reader, config, weights, options, &error);
  if (!engine) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  auto tokenizer = gufo::ext::ling3_air::text::Tokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
    return 1;
  }

  // Chunked prefill must be a function of the state and the tokens: the router gives each
  // row the weights of its own expert selection and writes them to that row's slots, so a
  // pass may not depend on where its rows fall in a chunk or on which block ran last. Three
  // identical prefills must agree bit for bit.
  //
  // LING3_PREFILL_REPEAT=1 runs it (the goldens and the serving suite cover the rest).
  if (std::getenv("LING3_PREFILL_REPEAT") != nullptr) {
    std::string text;
    for (int i = 0; i < 8; ++i) text += "Here is one fact about the red fox. ";
    text += "What does it eat?";
    const auto encoded = tokenizer->Encode(text, /*parse_special=*/true);
    std::vector<std::int32_t> prompt(encoded.begin(), encoded.end());
    std::vector<float> logits(engine->VocabSize());
    std::vector<float> reference;
    std::uint64_t reference_hash = 0;
    std::size_t differing = 0;
    for (int run = 0; run < 3; ++run) {
      if (!engine->ClearServingState() ||
          !engine->PrefillChunk(prompt.data(), 32, 0, -1, &error) ||
          !engine->PrefillChunk(prompt.data() + 32, 32, 32, -1, &error) ||
          !engine->PrefillChunk(prompt.data() + 64,
                                static_cast<std::size_t>(prompt.size()) - 64, 64, -1,
                                &error)) {
        std::fprintf(stderr, "repeat prefill: %s\n", error.c_str());
        return 1;
      }
      engine->ReadLogits(logits.data());
      std::uint64_t hash = 1469598103934665603ull;
      const auto* bytes = reinterpret_cast<const std::byte*>(logits.data());
      for (std::size_t i = 0; i < logits.size() * sizeof(float); ++i) {
        hash = (hash ^ static_cast<std::uint64_t>(bytes[i])) * 1099511628211ull;
      }
      float worst = 0.0f;
      for (std::size_t i = 0; i < logits.size() && !reference.empty(); ++i) {
        worst = std::max(worst, std::fabs(logits[i] - reference[i]));
      }
      if (run == 0) {
        reference = logits;
        reference_hash = hash;
      } else if (hash != reference_hash) {
        ++differing;
      }
      std::printf("     chunked prefill repeat %d: %zu tokens, logits %016llx, worst %.3g\n",
                  run, prompt.size(), static_cast<unsigned long long>(hash), worst);
    }
    Check(differing == 0, "prefills whose logits differ", static_cast<double>(differing), 0.0);
  }

  const std::string framed =
      "<role>SYSTEM</role>detailed thinking off<|role_end|>"
      "<role>HUMAN</role>The capital of France is<|role_end|>"
      "<role>ASSISTANT</role>\n<think" ">";
  const auto prompt = tokenizer->Encode(framed, /*parse_special=*/true);
  // Feed everything except the last prompt token, checkpoint, then feed that token: the
  // checkpoint therefore sits *before* it, so a rewind feeds it exactly once. Taking the
  // checkpoint after it made every rewind shift the KDA conv window a second time, which is
  // what made the round trip diverge.
  for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
    if (!engine->ForwardToken(static_cast<std::int32_t>(prompt[i]),
                              static_cast<std::uint32_t>(i), &error)) {
      std::fprintf(stderr, "prefill: %s\n", error.c_str());
      return 1;
    }
  }
  const std::uint32_t start = static_cast<std::uint32_t>(prompt.size());
  const std::int32_t last_prompt = static_cast<std::int32_t>(prompt.back());
  const std::uint32_t last_position = static_cast<std::uint32_t>(prompt.size() - 1);
  // The checkpoint every probe returns to, and the re-warm that makes it usable: the
  // snapshot does not carry the logits, so the last prompt token is fed again after it.
  engine->SnapshotState();
  // The same state in a serving checkpoint, which the equivalence checks below
  // restore instead of the shared snapshot buffer. Restoring it and feeding the last
  // prompt token warms `m.h` without a checkpoint outliving a step.
  std::vector<std::byte> pre_last(Model::ServingBytes(config, options.max_context));
  if (!engine->SaveServingState(pre_last)) {
    std::fprintf(stderr, "frontier checkpoint failed\n");
    return 1;
  }
  if (!engine->ForwardToken(static_cast<std::int32_t>(prompt.back()),
                            static_cast<std::uint32_t>(prompt.size() - 1), &error)) {
    std::fprintf(stderr, "first token: %s\n", error.c_str());
    return 1;
  }
  const auto rewind = [&]() {
    engine->RestoreState();
    return engine->ForwardToken(last_prompt, last_position, &error);
  };
  // The state the reference decode starts from, rebuilt through a serving checkpoint so
  // it does not depend on the snapshot buffer, plus the re-feed that makes m.h valid.
  const auto to_frontier = [&]() {
    return engine->LoadServingState(pre_last) &&
           engine->ForwardToken(last_prompt, last_position, &error);
  };

  std::vector<std::int32_t> reference;
  std::vector<std::vector<float>> reference_logits;
  if (!Greedy(*engine, start, steps, &reference, &reference_logits, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  if (!rewind()) {
    std::fprintf(stderr, "rewind: %s\n", error.c_str());
    return 1;
  }

  // Serving restores without replaying the last token: even the head output
  // must survive a cold reset. Compare every byte of the published payload.
  {
    std::vector<std::byte> saved(Model::ServingBytes(config, options.max_context)), copy(saved.size());
    if (!engine->SaveServingState(saved) || !engine->ClearServingState() ||
        engine->LoadServingState(std::span(saved).first(saved.size() - 1)) ||
        !engine->LoadServingState(saved) || !engine->SaveServingState(copy) || saved != copy) {
      std::fprintf(stderr, "serving checkpoint byte round-trip failed\n");
      return 1;
    }
    std::vector<std::int32_t> after;
    std::vector<std::vector<float>> rows;
    if (!Greedy(*engine, start, steps, &after, &rows, &error) || after != reference ||
        rows != reference_logits || !rewind()) return 1;
    std::printf("ok   serving checkpoint %zu bytes, %d decode rows\n", saved.size(), steps);
  }

  // 1. Restore round-trip: the same decode after a rewind must reproduce tokens and logits.
  {
    std::vector<std::int32_t> again;
    std::vector<std::vector<float>> logits;
    if (!Greedy(*engine, start, steps, &again, &logits, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::size_t divergence = reference.size();
    for (std::size_t i = 0; i < std::min(again.size(), reference.size()); ++i) {
      if (again[i] != reference[i]) {
        divergence = i;
        break;
      }
    }
    Check(divergence == reference.size(), "round-trip tokens (divergence index)",
          static_cast<double>(divergence), static_cast<double>(reference.size()));
    double worst = 0.0;
    for (std::size_t i = 0; i < std::min(logits.size(), reference_logits.size()); ++i) {
      for (std::size_t j = 0; j < logits[i].size(); ++j) {
        worst = std::max(worst, std::fabs(static_cast<double>(logits[i][j]) -
                                          reference_logits[i][j]));
      }
    }
    Check(worst == 0.0, "round-trip logits (max delta)", worst, 0.0);
    if (!rewind()) {
      return 1;
    }
  }

  // 2. Every layer's state survives the round trip.
  {
    const std::vector<double> before = engine->DebugStateDigest();
    engine->SnapshotState();
    engine->ForwardToken(engine->Argmax(), start, &error);
    engine->RestoreState();
    const std::vector<double> after = engine->DebugStateDigest();
    std::size_t differing = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (before[i] != after[i]) {
        ++differing;
      }
    }
    Check(differing == 0, "layers whose state differs", static_cast<double>(differing),
          static_cast<double>(before.size()));
    if (!rewind()) {
      return 1;
    }
  }

  // 2b. Does the MTP head write state the snapshot covers? Digest, draft, restore, digest.
  {
    const std::vector<double> before = engine->DebugStateDigest();
    engine->SnapshotState();
    engine->ForwardMtp(engine->Argmax(), start, &error);
    engine->RestoreState();
    const std::vector<double> after = engine->DebugStateDigest();
    std::size_t differing = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (before[i] != after[i]) ++differing;
    }
    std::size_t ps = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (std::abs(before[i] - after[i]) > 1e-9) ++ps;
    }
    std::printf("     mtp writes: %zu of %zu layers differ after restore (near: %zu)\n",
                differing, before.size(), ps);
    if (!rewind()) return 1;
  }

  // 2c. Restore after TWO consecutive feeds, which is what a verify pass does.
  {
    const std::vector<double> before = engine->DebugStateDigest();
    engine->SnapshotState();
    engine->ForwardToken(engine->Argmax(), start, &error);
    engine->ForwardToken(engine->Argmax(), start + 1, &error);
    engine->RestoreState();
    const std::vector<double> after = engine->DebugStateDigest();
    std::size_t differing = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (before[i] != after[i]) ++differing;
    }
    std::printf("     two-feed restore: %zu of %zu layers differ\n", differing, before.size());
    if (!rewind()) return 1;
  }

  // 2d. The composition the loop actually performs: MTP, two feeds, one restore.
  {
    const std::vector<double> before = engine->DebugStateDigest();
    engine->SnapshotState();
    engine->ForwardMtp(engine->Argmax(), start, &error);
    engine->ForwardToken(engine->Argmax(), start, &error);
    engine->ForwardToken(engine->Argmax(), start + 1, &error);
    engine->RestoreState();
    const std::vector<double> after = engine->DebugStateDigest();
    std::size_t differing = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (before[i] != after[i]) ++differing;
    }
    std::printf("     composed (mtp+2 feeds) restore: %zu of %zu layers differ\n", differing,
                before.size());
    if (!rewind()) return 1;
  }

  // 3. Speculation: draft one token with the MTP head, verify trunk + draft in one pass,
  // accept the draft only when the trunk's own token matches, rewind and re-feed exactly what
  // was accepted. VerifyDrafts checkpoints internally, so this loop takes none of its own.
  {
    // Own checkpoint and own reference: the probes above each called SnapshotState, and there
    // is one snapshot buffer, so the base checkpoint they overwrote must be re-established here
    // (rewind() + a fresh reference decode from exactly this state).
    if (!rewind()) {
      return 1;
    }
    engine->SnapshotState();
    if (!engine->ForwardToken(last_prompt, last_position, &error)) {
      return 1;
    }
    std::vector<std::int32_t> local_reference;
    std::vector<std::vector<float>> local_logits;
    if (!Greedy(*engine, start, steps, &local_reference, &local_logits, &error)) {
      return 1;
    }
    if (!rewind()) {
      return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::int32_t> speculative;
    std::size_t drafted = 0;
    std::size_t accepted = 0;
    std::uint32_t position = start;
    std::int32_t current = engine->Argmax();
    while (speculative.size() < static_cast<std::size_t>(steps)) {
      // Draft, then drop the draft's writes: the MTP head advances state the trunk's next
      // step reads, so a draft that is later rejected must leave no trace. The engine has one
      // snapshot buffer, so the checkpoint is retaken from the same state before verifying.
      engine->SnapshotState();
      if (!engine->ForwardMtp(current, position, &error)) {
        std::fprintf(stderr, "mtp: %s\n", error.c_str());
        return 1;
      }
      const std::int32_t draft = engine->Argmax();
      engine->RestoreState();
      ++drafted;
      // Verify sequentially rather than through VerifyDrafts' chunked pass: the chunked
      // path is a different code path from the one the round-trip proved exact, so it would
      // mix concerns. Checkpoint, feed trunk then draft, compare, rewind.
      engine->SnapshotState();
      std::int32_t rows[2] = {0, 0};
      if (!engine->ForwardToken(current, position, &error)) {
        return 1;
      }
      rows[0] = engine->Argmax();
      if (!engine->ForwardToken(draft, position + 1, &error)) {
        return 1;
      }
      rows[1] = engine->Argmax();
      const bool took_draft = rows[0] == draft;

      speculative.push_back(current);
      if (took_draft) {
        speculative.push_back(draft);
        ++accepted;
      }
      engine->RestoreState();
      if (!engine->ForwardToken(current, position, &error)) {
        return 1;
      }
      ++position;
      if (took_draft) {
        if (!engine->ForwardToken(draft, position, &error)) {
          return 1;
        }
        ++position;
      }
      current = engine->Argmax();
    }
    speculative.resize(static_cast<std::size_t>(steps));
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::size_t divergence = local_reference.size();
    for (std::size_t i = 0; i < std::min(speculative.size(), local_reference.size()); ++i) {
      if (speculative[i] != local_reference[i]) {
        divergence = i;
        break;
      }
    }
    // Reported, not asserted: the loop still diverges (see notes/prefill-design.md), and this
    // test's contract is the restore exactness above. Printing it keeps the evidence in every
    // run without turning an unproven feature into a failing suite.
    Check(divergence == local_reference.size(), "speculative stream (divergence index)",
          static_cast<double>(divergence), static_cast<double>(local_reference.size()));
    std::printf("     drafts %zu, accepted %zu (%.0f %%), %.2f s for %d tokens\n", drafted,
                accepted, drafted == 0 ? 0.0 : 100.0 * static_cast<double>(accepted) / drafted,
                seconds, steps);
  }

  // 4. VerifyDrafts equivalence (design §7 row 1). One chunked pass over
  // {current, draft} must return the same argmaxes as two ForwardToken calls; undoing
  // the pass and re-feeding exactly the accepted prefix must reproduce the next four
  // greedy tokens with logits delta 0 and an identical state digest.
  {
    if (!to_frontier()) {
      std::fprintf(stderr, "frontier: %s\n", error.c_str());
      return 1;
    }
    std::vector<std::int32_t> plain;
    std::vector<std::vector<float>> plain_logits;
    if (!Greedy(*engine, start, 5, &plain, &plain_logits, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const std::vector<double> digest5 = engine->DebugStateDigest();
    if (!Greedy(*engine, start + 5, 1, &plain, &plain_logits, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const std::vector<double> digest6 = engine->DebugStateDigest();
    if (!to_frontier()) {
      std::fprintf(stderr, "frontier: %s\n", error.c_str());
      return 1;
    }
    const std::int32_t first = engine->Argmax();
    Check(first == plain[0], "VerifyDrafts frontier token (delta)",
          static_cast<double>(first == plain[0] ? 0 : 1), 0.0);
    // Draft, then drop the draft's writes: the MTP head advances block 42's cache,
    // and VerifyDrafts takes its own checkpoint at entry.
    engine->SnapshotState();
    if (!engine->ForwardMtp(first, start, &error)) {
      std::fprintf(stderr, "mtp: %s\n", error.c_str());
      return 1;
    }
    const std::int32_t draft = engine->Argmax();
    engine->RestoreState();
    const std::int32_t pair[2] = {first, draft};
    std::int32_t rows[2] = {0, 0};
    if (!engine->VerifyDrafts(pair, 2, start, rows, &error)) {
      std::fprintf(stderr, "verify: %s\n", error.c_str());
      return 1;
    }
    engine->RestoreState();
    // The same two tokens through the token path, from the restored frontier: both
    // verify rows must equal the token path's argmaxes, including the row after a
    // draft that is rejected.
    if (!engine->ForwardToken(first, start, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const std::int32_t sequential0 = engine->Argmax();
    if (!engine->ForwardToken(draft, start + 1, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const std::int32_t sequential1 = engine->Argmax();
    const std::size_t row_mismatches = static_cast<std::size_t>(rows[0] != sequential0) +
                                       static_cast<std::size_t>(rows[1] != sequential1);
    Check(row_mismatches == 0, "VerifyDrafts rows vs ForwardToken",
          static_cast<double>(row_mismatches), 0.0);
    Check(sequential0 == plain[1], "sequential row0 vs greedy reference (delta)",
          static_cast<double>(sequential0 == plain[1] ? 0 : 1), 0.0);
    // Undo the pass and the sequential feed, then re-feed exactly what was accepted:
    // the serving step's re-commit discipline.
    engine->RestoreState();
    const bool accepted = rows[0] == draft;
    if (!engine->ForwardToken(first, start, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::uint32_t position = start + 1;
    if (accepted) {
      if (!engine->ForwardToken(draft, position, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
      }
      ++position;
    }
    std::vector<std::int32_t> refed;
    std::vector<std::vector<float>> refed_logits;
    if (!Greedy(*engine, position, 4, &refed, &refed_logits, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const std::size_t base = 1 + static_cast<std::size_t>(accepted);
    std::size_t token_mismatches = 0;
    for (std::size_t i = 0; i < refed.size(); ++i) {
      if (refed[i] != plain[base + i]) ++token_mismatches;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < refed_logits.size(); ++i) {
      for (std::size_t j = 0; j < refed_logits[i].size(); ++j) {
        worst = std::max(worst, std::fabs(static_cast<double>(refed_logits[i][j]) -
                                          plain_logits[base + i][j]));
      }
    }
    Check(token_mismatches == 0, "re-feed tokens vs greedy reference",
          static_cast<double>(token_mismatches), 0.0);
    Check(worst == 0.0, "re-feed logits vs greedy (max delta)", worst, 0.0);
    const std::vector<double>& expected_digest = accepted ? digest6 : digest5;
    const std::vector<double> refed_digest = engine->DebugStateDigest();
    std::size_t digest_mismatches = 0;
    for (std::size_t i = 0; i < refed_digest.size(); ++i) {
      if (refed_digest[i] != expected_digest[i]) ++digest_mismatches;
    }
    Check(digest_mismatches == 0, "re-feed state digest differs",
          static_cast<double>(digest_mismatches), 0.0);
    std::printf("     verify: draft %d, accepted %d; %zu layers digest\n", draft,
                accepted ? 1 : 0, refed_digest.size());
  }

  // 5. Loop equivalence (design §7 row 2): the serving step shape over 32 tokens from
  // the same frontier must equal plain greedy token for token, with logits delta 0.
  {
    if (!to_frontier()) {
      std::fprintf(stderr, "frontier: %s\n", error.c_str());
      return 1;
    }
    std::vector<std::int32_t> plain;
    std::vector<std::vector<float>> plain_logits;
    if (!Greedy(*engine, start, 32, &plain, &plain_logits, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    if (!to_frontier()) {
      std::fprintf(stderr, "frontier: %s\n", error.c_str());
      return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    std::int32_t current = engine->Argmax();
    std::uint32_t position = start;
    std::vector<std::int32_t> got;
    std::size_t drafts = 0;
    std::size_t accepted = 0;
    std::size_t logit_mismatches = 0;
    double worst = 0.0;
    std::vector<float> row(engine->VocabSize());
    while (got.size() < plain.size()) {
      if (plain.size() - got.size() >= 2) {
        std::int32_t draft = 0;
        bool took = false;
        if (!SpeculativeStep(*engine, position, current, &draft, &took, &error)) {
          std::fprintf(stderr, "%s\n", error.c_str());
          return 1;
        }
        ++drafts;
        got.push_back(current);
        ++position;
        if (took) {
          got.push_back(draft);
          ++position;
          ++accepted;
        }
      } else {
        got.push_back(current);
        if (!engine->ForwardToken(current, position, &error)) {
          std::fprintf(stderr, "%s\n", error.c_str());
          return 1;
        }
        ++position;
      }
      engine->ReadLogits(row.data());
      const auto& expected = plain_logits[got.size() - 1];
      for (std::size_t j = 0; j < row.size(); ++j) {
        const double delta = std::fabs(static_cast<double>(row[j]) - expected[j]);
        if (delta != 0.0) ++logit_mismatches;
        worst = std::max(worst, delta);
      }
      current = engine->Argmax();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::size_t token_mismatches = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != plain[i]) ++token_mismatches;
    }
    Check(token_mismatches == 0, "speculative loop tokens",
          static_cast<double>(token_mismatches), 0.0);
    Check(logit_mismatches == 0, "speculative loop logits (mismatches)",
          static_cast<double>(logit_mismatches), 0.0);
    Check(worst == 0.0, "speculative loop logits (max delta)", worst, 0.0);
    std::printf("     loop: %zu drafts, %zu accepted (%.0f %%), %.2f s for 32 tokens\n",
                drafts, accepted,
                drafts == 0 ? 0.0 : 100.0 * static_cast<double>(accepted) / drafts, seconds);
  }

  std::printf("mtp: %d failures\n", Failures);
  return Failures == 0 ? 0 : 1;
}
