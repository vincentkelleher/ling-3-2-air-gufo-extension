#include "src/runner.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/adapter.hpp>
#include <gufo/ext/registry.hpp>

#include "src/cache/component_resources.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <map>
#include <mutex>
#include <vector>
#include <utility>

#include "src/engine.hpp"
#include "src/preprocess.hpp"
#include "src/vision.hpp"

#include <gufo/ext/image.hpp>

namespace gufo::ext::ling3_air {
namespace {

/// Set by the extension at registration: creates one engine per serving slot.
Ling3Runner::EngineFactory g_engine_factory = nullptr;

/// Completion signal that is ready as soon as it is constructed: with no cache
/// components there is never a transfer to wait for.
class ReadySignal final : public cache::CompletionSignal {
 public:
  explicit ReadySignal(bool ok) : ok_(ok) {}
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override {
    return ok_ ? cache::TransferResult::kSucceeded : cache::TransferResult::kFailed;
  }
 private:
  bool ok_;
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

/// Speculation is opt-in: a server started without LING3_SPEC_DRAFTS=1 decodes one
/// token per step exactly as before. It is not on by default because the verifier
/// costs a second trunk pass over the pair, which measured slower than plain decode.
bool SpecDraftsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("LING3_SPEC_DRAFTS");
    return value != nullptr && std::atoi(value) != 0;
  }();
  return enabled;
}

[[nodiscard]] cache::Completion Ready(bool success = true) {
  return cache::Completion(std::make_unique<ReadySignal>(success));
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
  bool loaded{false};
  std::optional<std::size_t> bank;
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
    component_.layout_version = 2;
    component_.kind = cache::ComponentKind::kPrivateState;
    component_.state_bytes = state_bytes;
  }

  Engine& Bind(cache::Slot& slot) {
    const std::lock_guard lock(execution_);
    if (!engine_) {
      std::string error;
      if (g_engine_factory) engine_ = g_engine_factory(&error);
      if (!engine_) throw std::runtime_error("Ling engine creation failed: " + error);
    }
    auto& bank = static_cast<NullSlot&>(slot).bank;
    if (!bank) {
      engine_->SelectServingSlot(next_bank_);
      bank = next_bank_++;
      if (Trace()) std::fprintf(stderr, "[runner] bank %zu engine %p slot %p\n", *bank,
                                static_cast<void*>(engine_.get()), static_cast<void*>(&slot));
    }
    return *engine_;
  }

  std::unique_lock<std::mutex> LockSlot(const void* slot) override {
    std::unique_lock lock(execution_);
    const auto& bank = static_cast<const NullSlot*>(static_cast<const cache::Slot*>(slot))->bank;
    if (!engine_ || !bank) throw std::logic_error("unbound Ling slot");
    engine_->SelectServingSlot(*bank);
    return lock;
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
    const std::lock_guard<std::mutex> lock(mutex_);
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
  /// A pass ending at `boundary` leaves the whole private state there, so a
  /// checkpoint can be retained at any pass end. The host asks before reserving an
  /// aligned prompt target; refusing it would cost every continuation of a prompt
  /// whose length is not a pass multiple. Only sizing uses the answer.
  [[nodiscard]] std::vector<cache::ComponentPosition> PrefillCheckpointPositions(
      const cache::Slot&, cache::Rows boundary) const override {
    if (boundary == 0 || boundary > context_) {
      throw std::invalid_argument("checkpoint boundary outside the context");
    }
    cache::ComponentPosition position;
    position.id = component_.id;
    position.valid_rows = boundary;
    return {position};
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
  void BeginRestore(cache::Slot& slot, std::span<const cache::ComponentPosition>) override {
    static_cast<NullSlot&>(slot).loaded = false;
  }

  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot& slot,
                                                 cache::ComponentId,
                                                 std::span<std::byte> payload,
                                                 cache::Stream&) override {
    if (Trace()) {
      std::fprintf(stderr, "[runner] capture slot %p size %zu\n", static_cast<const void*>(&slot),
                   payload.size());
    }
    auto lock = LockSlot(&slot);
    return Ready(engine_->SaveServingState(payload));
  }
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot& slot, cache::ComponentId,
                                              std::span<const std::byte> payload,
                                              cache::Stream&) override {
    auto lock = LockSlot(&slot);
    bool ok = engine_->LoadServingState(payload);
    static_cast<NullSlot&>(slot).loaded = ok;
    if (Trace()) std::fprintf(stderr, "[runner] restore %zu bytes ok=%d\n", payload.size(), ok);
    return Ready(ok);
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
  [[nodiscard]] bool Validate(cache::Slot& slot,
                              std::span<const cache::ComponentPosition> positions) override {
    if (!static_cast<NullSlot&>(slot).loaded || positions.size() != 1 ||
        positions[0].id != component_.id || positions[0].valid_rows > context_) return false;
    OnPosition(&slot, positions[0].valid_rows);
    return true;
  }
  [[nodiscard]] bool Invalidate(cache::Slot& slot) noexcept override {
    static_cast<NullSlot&>(slot).loaded = false;
    try {
      auto lock = LockSlot(&slot);
      return engine_->ClearServingState();
    } catch (...) { return false; }
  }

 private:
  std::mutex execution_;
  std::size_t next_bank_{0};
  std::unique_ptr<Engine> engine_;

  mutable std::mutex mutex_;
  cache::Identity identity_;
  std::uint32_t context_{0};
  cache::ComponentDescriptor component_{};
  std::map<cache::Slot*, cache::Rows> positions_;
};

/// Encoded image rows and the token spans they replace. The spans keep prefill from
/// ending inside them: rows are injected as embeddings, one per marker token.
class Ling3ImageContext final : public TextPromptContext {
 public:
  std::vector<float> rows;
  std::uint32_t row_width{0};
  std::vector<std::pair<std::size_t, std::size_t>> spans;

  [[nodiscard]] std::size_t PrefillStop(std::size_t position, bool up) const override {
    for (const auto& span : spans) {
      const std::size_t end = span.first + span.second;
      if (position > span.first && position < end) {
        return up ? end : span.first;
      }
    }
    return position;
  }
};

}  // namespace

struct Ling3RunnerState::Impl {
  Engine* engine{nullptr};
  Ling3Config config;
  std::size_t position{0};
  Ling3RunnerState::PositionSink* sink{nullptr};
  void* slot{nullptr};
  std::shared_ptr<const TextPromptContext> context;
  bool hidden_valid{false};
  std::size_t spec_drafts{0};
  std::size_t spec_accepted{0};
};

Ling3RunnerState::Ling3RunnerState(Engine& engine, Ling3Config config)
    : impl_(std::make_unique<Impl>()) {
  impl_->engine = &engine;
  impl_->config = config;
}

Ling3RunnerState::~Ling3RunnerState() = default;

void Ling3RunnerState::Invalidate() noexcept {}

std::unique_lock<std::mutex> Ling3RunnerState::LockEngine() {
  return impl_->sink->LockSlot(impl_->slot);
}

Engine& Ling3RunnerState::engine() { return *impl_->engine; }

std::size_t Ling3RunnerState::position() const { return impl_->position; }

bool Ling3RunnerState::hidden_valid() const { return impl_->hidden_valid; }

void Ling3RunnerState::set_hidden_valid(bool value) { impl_->hidden_valid = value; }

void Ling3RunnerState::NoteSpeculation(bool accepted) {
  ++impl_->spec_drafts;
  if (accepted) ++impl_->spec_accepted;
}

void Ling3RunnerState::ReportSpeculation() {
  if (impl_->spec_drafts == 0) return;
  if (Trace()) {
    std::fprintf(stderr, "[runner] spec request drafts=%zu accepted=%zu (%zu%%)\n",
                 impl_->spec_drafts, impl_->spec_accepted,
                 100 * impl_->spec_accepted / impl_->spec_drafts);
  }
  impl_->spec_drafts = 0;
  impl_->spec_accepted = 0;
}

void Ling3RunnerState::set_context(std::shared_ptr<const TextPromptContext> context) {
  impl_->context = std::move(context);
}

std::shared_ptr<const TextPromptContext> Ling3RunnerState::context() const {
  return impl_->context;
}

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


struct Ling3Runner::Impl {
  Ling3Config config;
  std::unique_ptr<text::Tokenizer> tokenizer;
  std::string model_id;
  std::uint32_t max_context{0};
  /// Image input: the mmproj path, the encoder built from it, and the reader that owns
  /// the mapping the encoder points into.
  std::string vision_model_path;
  mutable std::mutex vision_mutex;
  mutable std::shared_ptr<const core::GgufReader> mmproj;
  mutable std::unique_ptr<VisionEncoder> vision;
};

Ling3Runner::Ling3Runner(Ling3Config config, std::unique_ptr<text::Tokenizer> tokenizer,
                         std::string model_id, std::uint32_t max_context,
                         std::string vision_model_path)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  impl_->tokenizer = std::move(tokenizer);
  impl_->model_id = std::move(model_id);
  impl_->max_context = max_context;
  impl_->vision_model_path = std::move(vision_model_path);
}

Ling3Runner::~Ling3Runner() = default;

TextRunnerDescriptor Ling3Runner::Descriptor() const {
  TextRunnerDescriptor descriptor;
  descriptor.model_id = impl_->model_id;
  descriptor.state_abi = "ling3_air/2";
  descriptor.max_context = impl_->max_context;
  descriptor.capabilities.supports_vision = !impl_->vision_model_path.empty();
  descriptor.capabilities.incremental_prefill = false;
  // Decode steps may publish two tokens: one MTP draft verified by the trunk when
  // the gate in DecodeStep is open. The default is one token per step.
  descriptor.capabilities.multi_token_decode = true;


  // One pass covers the chunk the engine's prefill actually batches (PrefillChunk's
  // width); it is not split across layer groups, and one decoder is stepped at a time.
  descriptor.scheduling.pass_tokens = 32;
  // A restored frontier continues the prompt exactly only from a pass boundary: the
  // chunked kernels accumulate per pass, so a checkpoint taken inside a pass would be
  // continued with different chunk widths than a cold prefill uses. The cache may
  // still continue an unchanged retry at the complete prompt.
  descriptor.capabilities.prefill_alignment = 32;
  descriptor.scheduling.max_layer_groups = 1;
  descriptor.scheduling.max_yield_decode_width = 1;
  return descriptor;
}

TextRunnerResourceClaim Ling3Runner::ResourceClaim() const {
  TextRunnerResourceClaim claim;
  // The cache grants this much host memory for one session's state; without it the
  // prepared-session pass refuses with budget 0.
  const std::size_t state_bytes = Engine::ServingBytes(impl_->config, impl_->max_context);
  // Budget for the two independently resident banks tested by the HTTP suite.
  claim.state_capacity_bytes = 2 * state_bytes;
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
  const std::size_t state_bytes = Engine::ServingBytes(impl_->config, impl_->max_context);
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
      // PinRows uses owner.get() as the first payload byte. Keep the vector
      // alive, but expose its data, not the address of the vector object.
      return cache::Payload::Committed(std::move(charge),
                                       std::shared_ptr<const void>(storage, storage->data()));
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
  auto state = std::make_unique<Ling3RunnerState>(components->Bind(slot), impl_->config);
  state->Attach(components, &slot);
  return state;
}

void Ling3Runner::ReconcileComponentState(TextRunnerState& state,
                                         std::size_t position) const {
  // After the cache restores rows into a slot, the runner must believe it stands where the
  // checkpoint was taken; otherwise it prefills from the wrong place.
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  runner_state.ReportSpeculation();
  // A restore carries the trunk state but not the scratch holding the last hidden, so
  // the first step after an acquire is plain and re-establishes it.
  runner_state.set_hidden_valid(false);
  runner_state.set_position(position);
}

namespace {
constexpr const char* kImageMarker = "<|image_pad|>";
}  // namespace

std::optional<TextPreparedPrompt> Ling3Runner::PreparePrompt(
    const ChatRequest& request) const {
  std::size_t image_count = 0;
  for (const auto& message : request.messages) {
    image_count += message.images.size();
  }
  if (image_count == 0) {
    auto tokens = RenderAndTokenize(request);
    if (!tokens) {
      return std::nullopt;
    }
    return TextPreparedPrompt{std::move(*tokens), nullptr, 0};
  }
  if (impl_->vision_model_path.empty() || impl_->tokenizer == nullptr) {
    return std::nullopt;
  }
  std::string error;
  std::unique_lock<std::mutex> lock(impl_->vision_mutex);
  if (impl_->vision == nullptr) {
    auto reader = core::GgufReader::OpenFile(impl_->vision_model_path, &error);
    auto encoder = reader ? VisionEncoder::Create(*reader, &error) : nullptr;
    if (encoder == nullptr) {
      if (Trace()) {
        std::fprintf(stderr, "[runner] mmproj: %s\n", error.c_str());
      }
      return std::nullopt;
    }
    impl_->mmproj = std::move(reader);
    impl_->vision = std::move(encoder);
  }
  VisionEncoder& vision = *impl_->vision;
  const std::uint32_t rows_per_image = vision.tokens_per_image();
  const int side = static_cast<int>(vision.image_side());
  const int grid = side / static_cast<int>(vision.Config().patch);

  auto context = std::make_shared<Ling3ImageContext>();
  context->row_width = impl_->config.n_embd;
  // The image markers are CONTROL tokens: they cannot be produced by encoding the literal text,
  // so their id is spliced in and each span is recorded where it is placed. That also removes the
  // fragile "rescan the token list and count markers" step, which is what failed before.
  const auto marker_ids = impl_->tokenizer->Encode(kImageMarker, /*parse_special=*/true);
  if (marker_ids.size() != 1) {
    if (Trace()) {
      std::fprintf(stderr, "[runner] %s resolves to %zu tokens, not 1\n", kImageMarker,
                   marker_ids.size());
    }
    return std::nullopt;
  }
  const auto marker = static_cast<TextRunnerToken>(marker_ids.front());
  std::vector<TextRunnerToken> prepared;
  const auto flush = [&](std::string& pending) {
    if (pending.empty()) {
      return;
    }
    const auto encoded = impl_->tokenizer->Encode(pending, /*parse_special=*/true);
    prepared.insert(prepared.end(), encoded.begin(), encoded.end());
    pending.clear();
  };
  std::string text;
  for (const auto& message : request.messages) {
    text += "<role>";
    text += RoleName(message.role);
    text += "</role>";
    for (const auto& part : message.images) {
      if (!part.bytes || part.bytes->empty()) {
        continue;
      }
      core::Image image;
      try {
        image = core::DecodeImage(std::span<const std::uint8_t>(*part.bytes));
      } catch (const std::exception& failure) {
        if (Trace()) {
          std::fprintf(stderr, "[runner] image: %s\n", failure.what());
        }
        return std::nullopt;
      }
      const std::vector<float> chw =
          ResizeToChw(image.pixels.data(), static_cast<int>(image.width),
                      static_cast<int>(image.height), side, ImageNorm{});
      std::vector<float> encoded;
      if (!vision.EncodeImage(chw.data(), static_cast<std::uint32_t>(grid),
                              static_cast<std::uint32_t>(grid), &encoded, &error)) {
        return std::nullopt;
      }
      context->rows.insert(context->rows.end(), encoded.begin(), encoded.end());
      flush(text);
      const std::size_t start = prepared.size();
      for (std::uint32_t row = 0; row < rows_per_image; ++row) {
        prepared.push_back(marker);
      }
      context->spans.emplace_back(start, rows_per_image);
    }
    text += message.content;
    text += "<|role_end|>";
  }
  text += "<role>ASSISTANT</role>\n<think" ">";
  flush(text);
  if (prepared.empty() || context->row_width == 0) {
    return std::nullopt;
  }
  std::size_t placed = 0;
  for (const auto& span : context->spans) {
    placed += span.second;
  }
  if (placed * context->row_width != context->rows.size()) {
    if (Trace()) {
      std::fprintf(stderr, "[runner] %zu markers placed vs %zu row values\n", placed,
                   context->rows.size());
    }
    return std::nullopt;
  }
  // The rendered prompt does not identify the rows, so they join the cache identity.
  std::uint64_t hash = 1469598103934665603ull;
  for (const float value : context->rows) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ull;
  }
  context->cache_identity.resize(sizeof(hash));
  for (std::size_t byte = 0; byte < sizeof(hash); ++byte) {
    context->cache_identity[byte] = static_cast<std::uint8_t>((hash >> (8 * byte)) & 0xFF);
  }
  return TextPreparedPrompt{std::move(prepared), std::move(context), 0};
}

void Ling3Runner::SetPromptContext(TextRunnerState& state,
                                  std::shared_ptr<const TextPromptContext> context) const {
  static_cast<Ling3RunnerState&>(state).set_context(std::move(context));
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
  std::size_t end = std::min(prompt.size(), offset + limit);
  // A pass must not end inside an image span: the host rejects that ("stopped prefill inside an
  // indivisible input"), and the rows are only meaningful together, so the span is consumed whole
  // even beyond the token budget.
  if (const auto context = runner_state.context()) {
    if (const auto* images = dynamic_cast<const Ling3ImageContext*>(context.get())) {
      for (const auto& span : images->spans) {
        const std::size_t span_end = span.first + span.second;
        if (span.first < end && end < span_end) {
          end = std::min(prompt.size(), span_end);
        }
      }
    }
  }
  std::vector<std::int32_t> ids(prompt.begin() + offset, prompt.begin() + end);
  // Image rows enter as embeddings, one marker at a time, because the chunked path takes
  // token ids. Without an image context nothing below changes.
  const auto image_context = runner_state.context();
  const auto* images = dynamic_cast<const Ling3ImageContext*>(image_context.get());
  if (images != nullptr && !images->spans.empty() && images->row_width > 0) {
    std::string error;
    std::size_t consumed = 0;
    for (std::size_t position = offset; position < end; ++position) {
      std::size_t row = std::numeric_limits<std::size_t>::max();
      std::size_t before = 0;
      for (const auto& span : images->spans) {
        if (position >= span.first && position < span.first + span.second) {
          row = before + (position - span.first);
          break;
        }
        before += span.second;
      }
      const bool ok =
          row == std::numeric_limits<std::size_t>::max()
              ? runner_state.engine().ForwardToken(ids[position - offset], position, &error)
              : runner_state.engine().ForwardEmbedding(
                    images->rows.data() + row * images->row_width, position, &error);
      if (!ok) {
        if (Trace()) {
          std::fprintf(stderr, "[runner] image prefill: %s\n", error.c_str());
        }
        step.interrupted = true;
        return step;
      }
      ++consumed;
    }
    runner_state.set_position(end);
    step.consumed_tokens = consumed;
    step.decode_ready = end == prompt.size();
    if (consumed > 0) runner_state.set_hidden_valid(true);
    return step;
  }
  std::string error;
  auto lock = runner_state.LockEngine();
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
  // The chunked pass leaves the last row's normalized hidden in the head scratch.
  if (!ids.empty()) runner_state.set_hidden_valid(true);
  return step;
}

TextDecodeSelection Ling3Runner::SelectNext(TextRunnerState& state,
                                            sampling::SamplerState&) const {
  TextDecodeSelection selection;
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  auto lock = runner_state.LockEngine();
  selection.token = runner_state.engine().Argmax();
  const TextRunnerToken single = selection.token;
  selection.piece = Decode(std::span<const TextRunnerToken>(&single, 1));
  return selection;
}

void Ling3Runner::AdvanceLocked(Ling3RunnerState& state, TextRunnerToken token) const {
  std::string error;
  if (!state.engine().ForwardToken(static_cast<std::int32_t>(token),
                                   static_cast<std::uint32_t>(state.position()), &error)) {
    throw std::runtime_error(error);
  }
  state.set_position(state.position() + 1);
  state.set_hidden_valid(true);
}

void Ling3Runner::Advance(TextRunnerState& state, TextRunnerToken token) const {
  if (Trace()) {
    std::fprintf(stderr, "[runner] advance token %u\n", static_cast<unsigned>(token));
  }
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  auto lock = runner_state.LockEngine();
  AdvanceLocked(runner_state, token);
}

TextDecodeStep Ling3Runner::DecodeStep(TextRunnerState& state, std::size_t max_tokens,
                                       sampling::SamplerState& sampler) const {
  auto& runner_state = static_cast<Ling3RunnerState&>(state);
  // Exact only where the output is the unmodified argmax. The plain step already
  // ignores the sampler, so the gate cannot change served output; it only decides
  // whether the second selection is attempted.
  const bool speculate =
      SpecDraftsEnabled() && max_tokens >= 2 && runner_state.hidden_valid() &&
      sampler.config().can_use_unmodified_argmax();
  if (!speculate) {
    return TextModelRunner::DecodeStep(state, max_tokens, sampler);
  }

  auto lock = runner_state.LockEngine();
  Engine& engine = runner_state.engine();
  const auto position = static_cast<std::uint32_t>(runner_state.position());
  const std::int32_t current = engine.Argmax();
  std::string error;
  // The engine owns one snapshot buffer: take the draft's checkpoint, drop it, then
  // let VerifyDrafts take its own. Neither outlives this step.
  engine.SnapshotState();
  if (!engine.ForwardMtp(current, position, &error)) {
    engine.RestoreState();
    throw std::runtime_error(error);
  }
  const std::int32_t draft = engine.Argmax();
  engine.RestoreState();
  const std::int32_t pair[2] = {current, draft};
  std::int32_t rows[2] = {0, 0};
  if (!engine.VerifyDrafts(pair, 2, position, rows, &error)) {
    engine.RestoreState();
    throw std::runtime_error(error);
  }
  const bool accepted = rows[0] == draft;
  engine.RestoreState();

  TextDecodeStep step;
  step.draft_rounds = 1;
  step.draft_tokens = 1;
  step.draft_accepted_tokens = accepted ? 1 : 0;
  runner_state.NoteSpeculation(accepted);
  if (Trace()) {
    std::fprintf(stderr, "[runner] spec step draft=1 accepted=%d pos=%zu\n",
                 accepted ? 1 : 0, runner_state.position());
  }
  const auto emit = [&](std::int32_t token) {
    TextDecodeSelection selection;
    selection.token = token;
    const TextRunnerToken single = token;
    selection.piece = Decode(std::span<const TextRunnerToken>(&single, 1));
    step.selections.push_back(std::move(selection));
  };
  // Re-feed exactly the accepted prefix. The retained state then comes only from
  // ForwardToken at consecutive positions, which is what the equivalence tests prove.
  emit(current);
  AdvanceLocked(runner_state, current);
  if (accepted) {
    emit(draft);
    AdvanceLocked(runner_state, draft);
  }
  return step;
}

bool Ling3Runner::CanContinue(const TextRunnerState&, std::size_t prompt_tokens) const {
  // Gufo checks prefix identity and boundary first. This query runs before it
  // reconciles the newly bound runner position. All rows and the head output
  // are present, including when the entire prompt was restored.
  return prompt_tokens <= impl_->max_context;
}

std::size_t Ling3Runner::CheckpointPosition(const TextRunnerState& state) const {
  return static_cast<const Ling3RunnerState&>(state).position();
}

}  // namespace gufo::ext::ling3_air
