#include "src/runner.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/adapter.hpp>
#include <gufo/ext/registry.hpp>

#include "src/cache/component_resources.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
#include <utility>

#include "src/engine.hpp"

namespace gufo::ext::ling3_air {
namespace {

/// Set by the extension at registration: creates one engine per serving slot.
Ling3Runner::EngineFactory g_engine_factory = nullptr;

/// Completion signal that is ready as soon as it is constructed: with no cache
/// components there is never a transfer to wait for.
class ReadySignal final : public cache::CompletionSignal {
 public:
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override { return {}; }
};

/// The chat template spells roles in upper case.
[[nodiscard]] const char* RoleName(ChatRole role) {
  switch (role) {
    case ChatRole::kSystem:
      return "SYSTEM";
    case ChatRole::kDeveloper:
      return "SYSTEM";
    case ChatRole::kUser:
      return "HUMAN";
    case ChatRole::kAssistant:
      return "ASSISTANT";
    case ChatRole::kTool:
      return "TOOL";
  }
  return "HUMAN";
}

bool Trace() {
  static const bool enabled = std::getenv("LING3_RUNNER_TRACE") != nullptr;
  return enabled;
}

[[nodiscard]] cache::Completion Ready() {
  return cache::Completion(std::make_unique<ReadySignal>());
}

/// The cache owns streams but only requires `Synchronize`: this adapter copies with
/// hipMemcpy on the default stream, so a lease that settles the device is all it needs.
class DeviceStream final : public cache::Stream {
 public:
  /// Nothing to wait for: this adapter's copies are synchronous `hipMemcpy` calls, so
  /// they are already complete when their `Completion` is returned. Touching the device
  /// here also crashed the server on shutdown, when the HIP context may already be gone.
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override { return {}; }
};

/// A slot with no components: it only has to be valid and empty.
class NullSlot final : public cache::Slot {
 public:
  [[nodiscard]] bool IsValid() const noexcept override { return true; }
};

/// Adapter with one private-state component: the trunk's snapshot buffer (KDA conv and
/// delta state, the MLA latent caches and their positions). Its bytes are exactly what
/// `Engine::SnapshotState` writes and `RestoreState` reads, so a restore reproduces an
/// uninterrupted run rather than approximating it.
///
/// Copies are synchronous on the default stream. That is correct but serialised; the
/// asynchronous form needs the device stream type behind `cache::Stream`, which is the
/// follow-up rather than the first milestone.
class Ling3ComponentAdapter final : public cache::Adapter,
                                   public Ling3RunnerState::PositionSink {
 public:
  /// Rows one prefill pass covers: the engine's chunk width.
  static constexpr cache::Rows kPassTokens = 32;
  Ling3ComponentAdapter(cache::Identity identity, std::size_t state_bytes,
                        std::uint32_t context)
      : identity_(std::move(identity)), context_(context) {
    component_.id = cache::ComponentId{0};
    component_.layout_version = 1;
    component_.kind = cache::ComponentKind::kPrivateState;
    component_.state_bytes = state_bytes;
  }

  /// The engine already serving this slot, if any. Binding a state twice must not build a
  /// second engine: creating one re-registers the same 80 GiB mapping, `hipHostRegister`
  /// refuses an already-registered range, and the pool is then left with no state (which
  /// is the null reservation that crashed Acquire).
  [[nodiscard]] Engine* Existing(cache::Slot* slot) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto found = engines_.find(slot);
    return found == engines_.end() ? nullptr : found->second;
  }

  Engine& Adopt(cache::Slot* slot, std::unique_ptr<Engine> engine) {
    const std::lock_guard<std::mutex> lock(mutex_);
    Engine* raw = engine.release();
    engines_[slot] = raw;
    return *raw;
  }

  /// `continuation` tells the host this adapter can restore a padded prefix, which is what
  /// the private-state component makes true: the snapshot holds the whole trunk state.
  [[nodiscard]] cache::Capabilities GetCapabilities() const override {
    cache::Capabilities capabilities;
    capabilities.continuation = true;
    return capabilities;
  }
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override {
    return {&component_, 1};
  }
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override {
    return identity_;
  }
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override {
    return std::make_unique<NullSlot>();
  }
  /// A private component reports its logical execution boundary: the position this
  /// slot's state has prefilled to. An empty vector is "incomplete capture positions".
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot& slot) const override {
    cache::ComponentPosition position;
    position.id = component_.id;
    const auto found = positions_.find(const_cast<cache::Slot*>(&slot));
    position.valid_rows = found == positions_.end() ? 0 : found->second;
    return {position};
  }

  void OnPosition(void* slot, std::size_t position) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    positions_[static_cast<cache::Slot*>(slot)] = position;
  }
  /// One pass covers at most `kPassTokens` rows, so the plan is the ascending
  /// boundaries those passes end on, finishing exactly at `end`.
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(const cache::Slot&,
                                                     cache::Rows first,
                                                     cache::Rows end) const override {
    std::vector<cache::Rows> rows;
    if (end <= first) {
      return rows;
    }
    for (cache::Rows at = first + kPassTokens; at < end; at += kPassTokens) {
      rows.push_back(at);
    }
    rows.push_back(end);
    return rows;
  }
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>) override {}

  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot& slot,
                                                 cache::ComponentId,
                                                 std::span<std::byte> payload,
                                                 cache::Stream&) override {
    if (Trace()) {
      std::fprintf(stderr, "[runner] capture slot %p size %zu\n", static_cast<const void*>(&slot),
                   payload.size());
    }
    Engine* engine = EngineFor(slot);
    if (engine == nullptr || payload.size() < component_.state_bytes) {
      return Ready();
    }
    hipMemcpy(payload.data(), engine->SnapshotData(), component_.state_bytes,
              hipMemcpyDeviceToHost);
    return Ready();
  }
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot& slot, cache::ComponentId,
                                              std::span<const std::byte> payload,
                                              cache::Stream&) override {
    Engine* engine = EngineFor(slot);
    if (engine == nullptr || payload.size() < component_.state_bytes) {
      return Ready();
    }
    hipMemcpy(const_cast<float*>(engine->SnapshotData()), payload.data(),
              component_.state_bytes, hipMemcpyHostToDevice);
    return Ready();
  }
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId,
                                              cache::Rows, cache::Rows,
                                              std::span<std::byte>,
                                              cache::Stream&) override {
    return Ready();
  }
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows, cache::Rows,
                                             std::span<const std::byte>,
                                             cache::Stream&) override {
    return Ready();
  }
  [[nodiscard]] bool Validate(cache::Slot&,
                              std::span<const cache::ComponentPosition>) override {
    return true;
  }
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override { return true; }

 private:
  [[nodiscard]] Engine* EngineFor(const cache::Slot& slot) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto found = engines_.find(const_cast<cache::Slot*>(&slot));
    return found == engines_.end() ? nullptr : found->second;
  }

  mutable std::mutex mutex_;
  cache::Identity identity_;
  std::uint32_t context_{0};
  cache::ComponentDescriptor component_{};
  /// Engines live for the process. The host may tear down sessions (and the HIP context)
  /// in any order at shutdown, and freeing an engine after its context has gone is what
  /// crashed the server on SIGTERM, so ownership is deliberately not returned here.
  std::map<cache::Slot*, Engine*> engines_;
  std::map<cache::Slot*, cache::Rows> positions_;
};

}  // namespace

struct Ling3RunnerState::Impl {
  Engine* engine{nullptr};
  Ling3Config config;
  std::size_t position{0};
  Ling3RunnerState::PositionSink* sink{nullptr};
  void* slot{nullptr};
};

Ling3RunnerState::Ling3RunnerState(Engine& engine, Ling3Config config)
    : impl_(std::make_unique<Impl>()) {
  impl_->engine = &engine;
  impl_->config = config;
}

Ling3RunnerState::~Ling3RunnerState() = default;

void Ling3RunnerState::Invalidate() noexcept {}

Engine& Ling3RunnerState::engine() { return *impl_->engine; }

std::size_t Ling3RunnerState::position() const { return impl_->position; }

void Ling3RunnerState::Attach(PositionSink* sink, void* slot) {
  impl_->sink = sink;
  impl_->slot = slot;
}

void Ling3RunnerState::set_position(std::size_t position) {
  impl_->position = position;
  if (impl_->sink != nullptr) {
    impl_->sink->OnPosition(impl_->slot, position);
  }
}

void Ling3Runner::SetEngineFactory(EngineFactory factory) {
  g_engine_factory = factory;
}

/// Floats the engine allocates for its snapshot: the private-state component's exact size
/// and therefore what the RAM cache has to grant per session.
[[nodiscard]] std::size_t SnapshotBytes(const Ling3Config& c, std::uint32_t context) {
  const std::size_t heads = c.n_head;
  const std::size_t kda_inner = c.KdaInner();
  const std::size_t kda_dim = c.n_kda_head_dim;
  const std::size_t kv_dim = c.n_lora_kv + c.n_rot;
  const std::size_t floats_per_layer =
      9 * kda_inner + heads * kda_dim * kda_dim + static_cast<std::size_t>(context) * kv_dim;
  return static_cast<std::size_t>(c.n_layer_total) * floats_per_layer * sizeof(float);
}

struct Ling3Runner::Impl {
  Ling3Config config;
  std::unique_ptr<text::Tokenizer> tokenizer;
  std::string model_id;
  std::uint32_t max_context{0};
};

Ling3Runner::Ling3Runner(Ling3Config config, std::unique_ptr<text::Tokenizer> tokenizer,
                         std::string model_id, std::uint32_t max_context)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  impl_->tokenizer = std::move(tokenizer);
  impl_->model_id = std::move(model_id);
  impl_->max_context = max_context;
}

Ling3Runner::~Ling3Runner() = default;

TextRunnerDescriptor Ling3Runner::Descriptor() const {
  TextRunnerDescriptor descriptor;
  descriptor.model_id = impl_->model_id;
  descriptor.state_abi = "ling3_air/1";
  descriptor.max_context = impl_->max_context;
  descriptor.capabilities.supports_vision = false;
  descriptor.capabilities.incremental_prefill = false;
  // One pass covers the chunk the engine's prefill actually batches (PrefillChunk's
  // width); it is not split across layer groups, and one decoder is stepped at a time.
  descriptor.scheduling.pass_tokens = 32;
  descriptor.scheduling.max_layer_groups = 1;
  descriptor.scheduling.max_yield_decode_width = 1;
  return descriptor;
}

TextRunnerResourceClaim Ling3Runner::ResourceClaim() const {
  TextRunnerResourceClaim claim;
  // The cache grants this much host memory for one session's state; without it the
  // prepared-session pass refuses with budget 0.
  const std::size_t state_bytes = SnapshotBytes(impl_->config, impl_->max_context);
  claim.state_capacity_bytes = state_bytes;
  claim.per_request_state_bytes = state_bytes;
  // A prompt token writes one row into each MLA layer's latent cache.
  std::size_t mla_layers = 0;
  for (std::uint32_t il = 0; il < impl_->config.n_layer; ++il) {
    if (!impl_->config.IsKda(il)) {
      ++mla_layers;
    }
  }
  // Retained checkpoints are what the RAM cache budget is sized from; without them the
  // host reports budget_bytes=0 and refuses to prepare sessions.
  // One capture is a whole snapshot, and the cache may hold a few before evicting, so the
  // capacity is several snapshots rather than one (a single-snapshot budget made
  // Reserve() throw bad_alloc on the first capture).
  claim.retained_snapshot_capacity_bytes = 4 * state_bytes;
  claim.retained_snapshot_ceiling_bytes = 16 * state_bytes;
  claim.prompt_state_bytes_per_token =
      mla_layers * static_cast<std::size_t>(impl_->config.n_lora_kv) * sizeof(float);
  return claim;
}

std::vector<TextExecutionPlan> Ling3Runner::SupportedPlans() const {
  return {TextExecutionPlan{}};
}

std::vector<TextRunnerToken> Ling3Runner::Tokenize(std::string_view text) const {
  if (impl_->tokenizer == nullptr) {
    return {};
  }
  return impl_->tokenizer->Encode(std::string(text), /*parse_special=*/true);
}

std::optional<std::vector<TextRunnerToken>> Ling3Runner::RenderAndTokenize(
    const ChatRequest& request) const {
  if (impl_->tokenizer == nullptr) {
    return std::nullopt;
  }
  // The chat template renders one system instruction plus the message turns; tool
  // definitions and images are refused rather than silently dropped.
  if (!request.tools.empty()) {
    return std::nullopt;
  }
  std::string text;
  for (const auto& message : request.messages) {
    if (!message.images.empty()) {
      return std::nullopt;
    }
    text += "<role>";
    text += RoleName(message.role);
    text += "</role>";
    text += message.content;
    text += "<|role_end|>";
  }
  text += "<role>ASSISTANT</role>\n<think" ">";
  return impl_->tokenizer->Encode(text, /*parse_special=*/true);
}

std::string Ling3Runner::Decode(std::span<const TextRunnerToken> tokens) const {
  if (impl_->tokenizer == nullptr) {
    return {};
  }
  std::vector<text::TokenId> ids(tokens.begin(), tokens.end());
  return impl_->tokenizer->Decode(ids);
}

std::unique_ptr<ComponentCacheResources> Ling3Runner::CreateComponentCacheResources(
    cache::ResourceLedger& ledger, std::size_t) const {
  auto resources = std::make_unique<ComponentCacheResources>();
  // Identity is an opaque byte blob: the model id distinguishes two checkpoints.
  cache::Identity identity(impl_->model_id.begin(), impl_->model_id.end());
  // The snapshot is n_layer_total x (9 * kda_inner + heads * kda_dim^2 + context * kv_dim)
  // floats; the engine allocates exactly that (see Engine::Create).
  const std::size_t state_bytes = SnapshotBytes(impl_->config, impl_->max_context);
  resources->adapter =
      std::make_unique<Ling3ComponentAdapter>(identity, state_bytes, impl_->max_context);
  // The host asks this for the buffer it captures a component into, then takes ownership
  // through `finish`. Host memory is right here: the snapshot is copied device-to-host
  // by the adapter itself.
  // The capture buffer is charged to the ledger the host hands us: a
  // default-constructed ResourceCharge carries no ledger handle, and the host crashed on
  // exactly that ("frontier_capture_refused reason=empty ledger handle").
  cache::ResourceLedger* ledger_ref = &ledger;
  resources->allocate = [ledger_ref](const cache::PayloadRequest& request) {
    auto storage = std::make_shared<std::vector<std::byte>>(request.bytes);
    cache::ResourceReservation reservation =
        ledger_ref->Reserve(request.category, request.bytes);
    cache::ResourceCharge charge = reservation.Convert();
    cache::ComponentCaptureBuffer buffer;
    buffer.bytes = std::span<std::byte>(storage->data(), storage->size());
    buffer.allocation = storage->data();
    buffer.finish = [storage, charge = std::move(charge)]() mutable {
      return cache::Payload::Committed(std::move(charge),
                                       std::static_pointer_cast<const void>(storage));
    };
    return buffer;
  };
  // The payload lives in host memory, so it is never device-only, and this adapter
  // never shares append rows (it has none).
  resources->stream = [] { return std::unique_ptr<cache::Stream>(new DeviceStream()); };
  resources->device_only = [](const void*) { return false; };
  resources->share = [](const cache::PayloadRequest&,
                        std::span<const std::byte>) -> std::optional<cache::Payload> {
    return std::nullopt;
  };
  return resources;
}

std::unique_ptr<TextRunnerState> Ling3Runner::BindComponentState(
    cache::Adapter& adapter, cache::Slot& slot) const {
  if (Trace()) {
    std::fprintf(stderr, "[runner] bind slot %p\n", static_cast<const void*>(&slot));
  }
  auto* components = dynamic_cast<Ling3ComponentAdapter*>(&adapter);
  if (components == nullptr) {
    return nullptr;
  }
  Engine* engine = components->Existing(&slot);
  if (engine == nullptr) {
    std::string error;
    auto created = g_engine_factory == nullptr ? nullptr : g_engine_factory(&error);
    if (created == nullptr) {
      if (Trace()) {
        std::fprintf(stderr, "[runner] engine creation failed: %s\n", error.c_str());
      }
      return nullptr;
    }
    engine = &components->Adopt(&slot, std::move(created));
  }
  auto state = std::make_unique<Ling3RunnerState>(*engine, impl_->config);
  state->Attach(components, &slot);
  return state;
}

void Ling3Runner::ReconcileComponentState(TextRunnerState& state,
                                         std::size_t position) const {
  // After the cache restores rows into a slot, the runner must believe it stands where the
  // checkpoint was taken; otherwise it prefills from the wrong place.
  static_cast<Ling3RunnerState&>(state).set_position(position);
}

TextPrefillStep Ling3Runner::Prefill(TextRunnerState& state,
                                     std::span<const TextRunnerToken> prompt,
                                     std::size_t offset,
                                     std::size_t max_input_tokens) const {
  TextPrefillStep step;
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  if (Trace()) {
    std::fprintf(stderr, "[runner] prefill offset %zu max %zu prompt %zu pos %zu\n", offset,
                 max_input_tokens, prompt.size(), runner_state.position());
  }
  // The engine's chunked pass has a fixed width (Engine::PrefillChunk's buffer); a
  // larger request would run past it. Clamping here also gives the host the frontier it
  // expects: it simply asks for the next pass.
  const std::size_t limit = std::min<std::size_t>(max_input_tokens, 32);
  const std::size_t end = std::min(prompt.size(), offset + limit);
  std::vector<std::int32_t> ids(prompt.begin() + offset, prompt.begin() + end);
  std::string error;
  if (!ids.empty() &&
      !runner_state.engine().PrefillChunk(ids.data(), ids.size(),
                                          static_cast<std::uint32_t>(offset), -1,
                                          &error)) {
    step.interrupted = true;
    return step;
  }
  runner_state.set_position(end);
  step.consumed_tokens = ids.size();
  step.decode_ready = end == prompt.size();
  return step;
}

TextDecodeSelection Ling3Runner::SelectNext(TextRunnerState& state,
                                            sampling::SamplerState&) const {
  TextDecodeSelection selection;
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  selection.token = runner_state.engine().Argmax();
  const TextRunnerToken single = selection.token;
  selection.piece = Decode(std::span<const TextRunnerToken>(&single, 1));
  return selection;
}

void Ling3Runner::Advance(TextRunnerState& state, TextRunnerToken token) const {
  if (Trace()) {
    std::fprintf(stderr, "[runner] advance token %u\n", static_cast<unsigned>(token));
  }
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  std::string error;
  runner_state.engine().ForwardToken(static_cast<std::int32_t>(token),
                                     static_cast<std::uint32_t>(runner_state.position()),
                                     &error);
  runner_state.set_position(runner_state.position() + 1);
}

std::size_t Ling3Runner::CheckpointPosition(const TextRunnerState& state) const {
  return static_cast<const Ling3RunnerState&>(state).position();
}

}  // namespace gufo::ext::ling3_air
