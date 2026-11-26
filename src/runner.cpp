#include "src/runner.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/adapter.hpp>
#include <gufo/ext/registry.hpp>

#include "src/cache/component_resources.hpp"

#include <algorithm>
#include <cstring>
#include <map>
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

[[nodiscard]] cache::Completion Ready() {
  return cache::Completion(std::make_unique<ReadySignal>());
}

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
class Ling3ComponentAdapter final : public cache::Adapter {
 public:
  Ling3ComponentAdapter(cache::Identity identity, std::size_t state_bytes,
                        std::uint32_t context)
      : identity_(std::move(identity)), context_(context) {
    component_.id = cache::ComponentId{0};
    component_.layout_version = 1;
    component_.kind = cache::ComponentKind::kPrivateState;
    component_.state_bytes = state_bytes;
  }

  void Register(cache::Slot* slot, Engine* engine) { engines_[slot] = engine; }

  [[nodiscard]] cache::Capabilities GetCapabilities() const override { return {}; }
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
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override {
    return {};
  }
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(const cache::Slot&,
                                                     cache::Rows first,
                                                     cache::Rows end) const override {
    return {first, end};
  }
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>) override {}

  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot& slot,
                                                 cache::ComponentId,
                                                 std::span<std::byte> payload,
                                                 cache::Stream&) override {
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
    const auto found = engines_.find(const_cast<cache::Slot*>(&slot));
    return found == engines_.end() ? nullptr : found->second;
  }

  cache::Identity identity_;
  std::uint32_t context_{0};
  cache::ComponentDescriptor component_{};
  std::map<cache::Slot*, Engine*> engines_;
};

}  // namespace

struct Ling3RunnerState::Impl {
  std::unique_ptr<Engine> engine;
  Ling3Config config;
  std::size_t position{0};
};

Ling3RunnerState::Ling3RunnerState(std::unique_ptr<Engine> engine, Ling3Config config)
    : impl_(std::make_unique<Impl>()) {
  impl_->engine = std::move(engine);
  impl_->config = config;
}

Ling3RunnerState::~Ling3RunnerState() = default;

void Ling3RunnerState::Invalidate() noexcept {}

Engine& Ling3RunnerState::engine() { return *impl_->engine; }

std::size_t Ling3RunnerState::position() const { return impl_->position; }

void Ling3RunnerState::set_position(std::size_t position) {
  impl_->position = position;
}

void Ling3Runner::SetEngineFactory(EngineFactory factory) {
  g_engine_factory = factory;
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
  return descriptor;
}

TextRunnerResourceClaim Ling3Runner::ResourceClaim() const {
  TextRunnerResourceClaim claim;
  claim.state_capacity_bytes = 0;
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
    cache::ResourceLedger&, std::size_t) const {
  auto resources = std::make_unique<ComponentCacheResources>();
  // Identity is an opaque byte blob: the model id distinguishes two checkpoints.
  cache::Identity identity(impl_->model_id.begin(), impl_->model_id.end());
  // The snapshot is n_layer_total x (9 * kda_inner + heads * kda_dim^2 + context * kv_dim)
  // floats; the engine allocates exactly that (see Engine::Create).
  const Ling3Config& c = impl_->config;
  const std::size_t heads = c.n_head;
  const std::size_t kda_inner = c.KdaInner();
  const std::size_t kda_dim = c.n_kda_head_dim;
  const std::size_t kv_dim = c.n_lora_kv + c.n_rot;
  const std::size_t floats_per_layer =
      9 * kda_inner + heads * kda_dim * kda_dim + impl_->max_context * kv_dim;
  const std::size_t state_bytes =
      static_cast<std::size_t>(c.n_layer_total) * floats_per_layer * sizeof(float);
  resources->adapter =
      std::make_unique<Ling3ComponentAdapter>(identity, state_bytes, impl_->max_context);
  // The host asks this for the buffer it captures a component into, then takes ownership
  // through `finish`. Host memory is right here: the snapshot is copied device-to-host
  // by the adapter itself.
  resources->allocate = [](const cache::PayloadRequest& request) {
    auto storage = std::make_shared<std::vector<std::byte>>(request.bytes);
    cache::ComponentCaptureBuffer buffer;
    buffer.bytes = std::span<std::byte>(storage->data(), storage->size());
    buffer.allocation = storage->data();
    buffer.finish = [storage]() {
      return cache::Payload::Committed(
          cache::ResourceCharge{},
          std::static_pointer_cast<const void>(storage));
    };
    return buffer;
  };
  // The payload lives in host memory, so it is never device-only, and this adapter
  // never shares append rows (it has none).
  resources->device_only = [](const void*) { return false; };
  resources->share = [](const cache::PayloadRequest&,
                        std::span<const std::byte>) -> std::optional<cache::Payload> {
    return std::nullopt;
  };
  return resources;
}

std::unique_ptr<TextRunnerState> Ling3Runner::BindComponentState(
    cache::Adapter& adapter, cache::Slot& slot) const {
  std::string error;
  auto engine = g_engine_factory == nullptr ? nullptr : g_engine_factory(&error);
  if (engine == nullptr) {
    return nullptr;
  }
  auto state = std::make_unique<Ling3RunnerState>(std::move(engine), impl_->config);
  // The adapter copies this slot's snapshot, so it needs the engine that owns it.
  auto* components = dynamic_cast<Ling3ComponentAdapter*>(&adapter);
  if (components != nullptr) {
    components->Register(&slot, &state->engine());
  }
  return state;
}

void Ling3Runner::ReconcileComponentState(TextRunnerState&, std::size_t) const {}

TextPrefillStep Ling3Runner::Prefill(TextRunnerState& state,
                                     std::span<const TextRunnerToken> prompt,
                                     std::size_t offset,
                                     std::size_t max_input_tokens) const {
  TextPrefillStep step;
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  const std::size_t end = std::min(prompt.size(), offset + max_input_tokens);
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
