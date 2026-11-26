// Ling 3.2 Air text engine: weights on the GPU, one token per forward.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "src/config.hpp"
#include "src/weights.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::ext::ling3_air {

struct EngineOptions {
  /// Context the KV latent and position budget is sized for.
  std::uint32_t max_context{4096};
};

/// A loaded model. Create() uploads the mapped GGUF regions as they are (one
/// copy per shard), so tensor pointers are resolved by offset.
class Engine {
 public:
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  [[nodiscard]] static std::unique_ptr<Engine> Create(
      const core::GgufReader& reader, const Ling3Config& config,
      const Ling3Weights& weights, const EngineOptions& options,
      std::string* error);

  /// Runs one token at `position` and writes the vocab logits into `logits`
  /// (device memory, `VocabSize()` floats).
  [[nodiscard]] bool ForwardToken(std::int32_t token, std::uint32_t position,
                                  std::string* error);

  /// MTP: predicts the token after `token` given the base model's final hidden
  /// state (the one produced by the last ForwardToken). Mirrors llama.cpp's
  /// graph_mtp: eh_proj(concat(token embedding, hidden)) -> block 42 -> head.
  [[nodiscard]] bool ForwardMtp(std::int32_t token, std::uint32_t position,
                                std::string* error);

  /// Copies the embedding row the first step would compute (debug/test hook).
  void DebugEmbedding(std::int32_t token, float* host_row);

  /// Copies the last step's logits to the host (large: `VocabSize()` floats).
  void ReadLogits(float* host_logits);
  /// Greedy token of the last step, computed on the device.
  std::int32_t Argmax();

  [[nodiscard]] std::uint32_t VocabSize() const;
  [[nodiscard]] std::size_t DeviceBytes() const;

 private:
  Engine() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::ext::ling3_air
