// MLA (multi-head latent attention) kernels for the Ling 3.2 Air blocks.
//
// The block keeps a compressed cache: per token, 512 latent values plus the 64
// rope dimensions of the key (`kLatent + kRope = 576` floats). Queries are
// absorbed into the latent space with `attn_k_b` before scoring, and the
// attention output is projected back to 128 values per head with `attn_v_b`.
#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

namespace gufo::ext::ling3_air::hip {

inline constexpr std::size_t kMlaLatent = 512;
inline constexpr std::size_t kMlaRope = 64;
inline constexpr std::size_t kMlaNope = 128;
inline constexpr std::size_t kMlaV = 128;
inline constexpr std::size_t kMlaCacheStride = kMlaLatent + kMlaRope;
/// `1 / sqrt(128 + 64)`, the score scale llama.cpp uses.
inline constexpr float kMlaScoreScale = 1.0f / 13.856406460551018f;

/// `q_abs[h][i] = sum_d q_nope[h][d] * wk_b[h][i][d]`
/// (`wk_b` is `[heads][kMlaLatent][kMlaNope]`).
void LaunchMlaAbsorbQ(const float* q_nope, const float* wk_b, float* q_abs,
                      std::size_t heads, hipStream_t stream);

/// Scores of one query against `cache_len` cached tokens, scaled by
/// `kMlaScoreScale`; `scores` holds `heads * cache_len` floats.
void LaunchMlaScores(const float* q_abs, const float* q_pe, const float* cache,
                     std::size_t heads, std::size_t cache_len, float* scores,
                     hipStream_t stream);

/// In-place softmax over each head's `cache_len` scores.
void LaunchMlaSoftmax(float* scores, std::size_t heads, std::size_t cache_len,
                      hipStream_t stream);

/// `latent[h][i] = sum_p probs[h][p] * cache[p][i]` over the 512 latent values.
void LaunchMlaLatentSum(const float* probs, const float* cache,
                        std::size_t heads, std::size_t cache_len,
                        float* latent, hipStream_t stream);

/// `out[h][o] = sigmoid(gate[h]) * sum_i latent[h][i] * wv_b[h][o][i]`
/// (`wv_b` is `[heads][kMlaV][kMlaLatent]`).
void LaunchMlaValueAbsorb(const float* latent, const float* wv_b,
                          const float* gate, float* out, std::size_t heads,
                          hipStream_t stream);

}  // namespace gufo::ext::ling3_air::hip
