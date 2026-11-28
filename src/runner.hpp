// Text model runner: the contract `gufo serve llm` drives.
//
// First milestone: a complete runner whose cache adapter reports **no components**, so
// there is no prefix reuse yet and every request prefills from scratch. That keeps the
// transfer methods (row copies, private-state capture, borrow views) unreachable while
// still satisfying the interface, and it is enough for the server to serve.
//
// Sampling is greedy on purpose: `SelectNext` ignores the sampler and takes the argmax,
// which is exact at temperature 0. Honouring temperature/top-k/top-p means threading
// `sampling::SamplerState` through, and is the next step; the limitation is stated in
// docs/quality.md rather than hidden.
//
// `DecodeStep` can return two tokens per step when MTP speculation is enabled with
// `LING3_SPEC_DRAFTS=1`: it drafts one token with block 42, verifies it with one
// chunked trunk pass, and re-feeds exactly the accepted prefix. The gate is a greedy
// request (`can_use_unmodified_argmax`), a budget of at least two tokens, and a valid
// trunk hidden for the draft to read. The default is off because the verifier runs
// the trunk over the pair, which costs about one extra trunk pass per accepted token;
// docs/quality.md carries the measured acceptance and rate.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gufo/ext/text.hpp>

#include "src/config.hpp"
#include "src/text/tokenizer.hpp"

namespace gufo::ext::ling3_air {

class Engine;
struct Ling3RunnerOptions;

/// Owns the engine and the position the runner has prefilled to.
class Ling3RunnerState final : public TextRunnerState {
 public:
  /// The engine is owned by the cache adapter, which outlives every state bound to a
  /// slot; the state only borrows it, so a destroyed state cannot leave a dangling
  /// pointer behind in the adapter.
  Ling3RunnerState(Engine& engine, Ling3Config config);
  ~Ling3RunnerState() override;

  [[nodiscard]] std::unique_lock<std::mutex> LockEngine();
  [[nodiscard]] Engine& engine();
  [[nodiscard]] std::size_t position() const;
  void set_position(std::size_t position);
  /// True while the engine's `m.h` holds the trunk's final hidden for the last step,
  /// which is what the MTP head drafts from. A restore replaces the trunk state but
  /// not that scratch, so an acquired state starts false and the first plain step
  /// (or a prefill) sets it.
  [[nodiscard]] bool hidden_valid() const;
  void set_hidden_valid(bool value);
  /// Serving trace: count one speculated step and report the per-request totals when
  /// the next request acquires this slot (the host already logs its own summary).
  void NoteSpeculation(bool accepted);
  void ReportSpeculation();
  /// Image rows travel with the request as a prompt context (text.hpp), so the state
  /// keeps whatever the host last attached and the prefill step reads it from there.
  void set_context(std::shared_ptr<const TextPromptContext> context);
  [[nodiscard]] std::shared_ptr<const TextPromptContext> context() const;

  /// The cache asks the adapter for this slot's frontier, so the state reports where it
  /// has prefilled to.
  struct PositionSink {
    virtual ~PositionSink() = default;
    virtual std::unique_lock<std::mutex> LockSlot(const void* slot) = 0;
    virtual void OnPosition(void* slot, std::size_t position) = 0;
  };
  void Attach(PositionSink* sink, void* slot);
  void Invalidate() noexcept override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The runner itself: tokenization and chat rendering delegate to the extension's
/// tokenizer, prompt ingestion to the engine's chunked prefill, decoding to the
/// single-token step.
class Ling3Runner final : public TextModelRunner {
 public:
  Ling3Runner(Ling3Config config, std::unique_ptr<text::Tokenizer> tokenizer,
              std::string model_id, std::uint32_t max_context,
              std::string vision_model_path = {});
  ~Ling3Runner() override;

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override;
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override;
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override;
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override;
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override;
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override;
  /// Image input: renders `<|image_pad|>` per projected row and carries the encoded rows
  /// in a prompt context. Without images this is the host's default behaviour.
  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override;
  void SetPromptContext(TextRunnerState&,
                        std::shared_ptr<const TextPromptContext>) const override;
  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger&, std::size_t) const override;
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter&, cache::Slot&) const override;
  void ReconcileComponentState(TextRunnerState&, std::size_t) const override;
  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState& state,
                                        std::span<const TextRunnerToken> prompt,
                                        std::size_t offset,
                                        std::size_t max_input_tokens) const override;
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;
  void Advance(TextRunnerState& state, TextRunnerToken token) const override;
  /// Two selections when the speculation gate is open, otherwise the inherited
  /// single-token step. The step holds the engine lock for the draft, the verify pass
  /// and the re-feed, so the serving bank is selected once and no other operation can
  /// interleave with the engine's one snapshot buffer.
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override;
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override;
  /// The host asks this before reusing a session's prefix. This runner can only continue a prompt
  /// whose length equals where its state actually stands; otherwise the host must prefill again,
  /// because the engine cannot rewind past a generated token.
  [[nodiscard]] bool CanContinue(const TextRunnerState& state,
                                 std::size_t prompt_tokens) const override;

  /// One factory invocation per cache adapter; all its slots share the engine.
  using EngineFactory = std::unique_ptr<Engine> (*)(std::string* error);
  static void SetEngineFactory(EngineFactory factory);

 private:
  /// `Advance` without the engine lock; the decode step already holds it.
  void AdvanceLocked(Ling3RunnerState& state, TextRunnerToken token) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::ext::ling3_air
