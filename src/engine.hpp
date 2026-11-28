// Ling 3.2 Air text engine: weights on the GPU, one token per forward.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
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

  /// Runs one step from a caller-supplied embedding (`n_embd` floats, host) instead
  /// of a token id. This is how vision rows enter the trunk: everything after the
  /// embedding lookup is shared with ForwardToken, so positions, caches and logits
  /// behave identically.
  [[nodiscard]] bool ForwardEmbedding(const float* embedding, std::uint32_t position,
                                      std::string* error);

  /// Chunked-path stage 1 (debug): embed `count` tokens into the chunk buffer,
  /// apply block 0's attention RMSNorm over all rows, and copy both out.
  void DebugChunkStage1(const std::int32_t* tokens, std::size_t count, float* host_x,
                        float* host_h);

  /// Chunked-path stage 2 (debug): projects block 0's KDA query both ways, batched
  /// through the GEMM and per row through the GEMV, and returns the largest
  /// absolute difference between the two.
  float DebugChunkProjectionDiff(std::size_t rows);

  /// Runs `count` tokens (up to 32) in one pass per block: projections, dense FFN
  /// and shared expert are GEMMs over the chunk; KDA conv/delta, MLA attention and
  /// routed MoE stay per token. With `stop_after_block` set it stops after that
  /// block and leaves the chunk activations in the chunk buffer (for bisecting).
  [[nodiscard]] bool PrefillChunk(const std::int32_t* tokens, std::size_t count,
                                  std::uint32_t start_position,
                                  std::int32_t stop_after_block,
                                  std::string* error);

  /// Copies `rows` rows of the chunk activation buffer out (debug/bisect).
  void ReadChunkActivations(std::size_t rows, float* host_rows);

  /// Speculative verification: runs `count` tokens from `start_position` in one
  /// chunked pass and reports the greedy argmax at every position. The trunk
  /// state is snapshotted first, so the caller can accept a prefix and rewind with
  /// RestoreState if it rejects any draft. The pass is the full trunk: the head
  /// reads the last block's residual, so stopping inside that block would verify
  /// against a hidden state the token path never produces.
  [[nodiscard]] bool VerifyDrafts(const std::int32_t* tokens, std::size_t count,
                                  std::uint32_t start_position,
                                  std::int32_t* argmax_per_row, std::string* error);

  /// Snapshots / restores the trunk state (KDA conv+delta, MLA caches).
  void SnapshotState();

  /// Fixed-width host checkpoint of the live recurrent buffers and decode outputs.
  static std::size_t ServingBytes(const Ling3Config& config, std::uint32_t context);
  bool SaveServingState(std::span<std::byte> destination);
  bool LoadServingState(std::span<const std::byte> source);
  bool ClearServingState();
  /// Slot zero uses the original allocations. Additional slots have separate
  /// persistent buffers but share the queue, weights and transient activations.
  /// Caller serializes selection and execution as one operation.
  void SelectServingSlot(std::size_t index);
  /// One number per layer of the state the snapshot is meant to cover, for localising an
  /// inexact restore: equal before and after a snapshot+restore round-trip means that
  /// layer's state survived.
  [[nodiscard]] std::vector<double> DebugStateDigest();

  /// Clears the per-token scratch. Used to test whether a restore's residual difference is
  /// stale scratch rather than state.
  void DebugZeroScratch();

  [[nodiscard]] const float* SnapshotData() const;
  [[nodiscard]] std::size_t SnapshotFloats() const;
  void RestoreState();

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
  /// The shared trunk step: assumes `m.x` already holds this step's embedding.
  [[nodiscard]] bool ForwardFromEmbedded(std::uint32_t position, std::string* error);

  /// Runs the output head on the normalized hidden already in `m.h`: writes
  /// `m.logits` and `m.argmax`. Shared by the token, chunked and verification
  /// paths so their logits are computed by the same kernels in the same order
  /// (the prepared head quantises the activation; the token path uses it).
  void RunOutputHead();

  Engine() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::ext::ling3_air
