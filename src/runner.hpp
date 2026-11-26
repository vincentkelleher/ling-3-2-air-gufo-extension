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
#pragma once

#include <cstdint>
#include <memory>
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

  [[nodiscard]] Engine& engine();
  [[nodiscard]] std::size_t position() const;
  void set_position(std::size_t position);

  /// The cache asks the adapter for this slot's frontier, so the state reports where it
  /// has prefilled to.
  struct PositionSink {
    virtual ~PositionSink() = default;
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
              std::string model_id, std::uint32_t max_context);
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
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override;

  /// Engine factory: the runner creates one per serving slot.
  using EngineFactory = std::unique_ptr<Engine> (*)(std::string* error);
  static void SetEngineFactory(EngineFactory factory);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::ext::ling3_air
