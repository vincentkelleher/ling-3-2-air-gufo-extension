// Block-level HIP kernels for the Ling 3.2 Air text engine.
//
// Same contract as kernels.hpp: plain C++ signatures on the host side, HIP
// only inside the .hip translation unit.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace gufo::ext::ling3_air::hip {

/// Causal depthwise conv1d, one step: `out[c] = sum_k w[k][c] * window[k]`
/// with `window = [state0, state1, state2, x]`, then the state shifts left and
/// `x` becomes the newest column. `state` holds `3 * channels` floats and is
/// updated in place. SiLU is applied by the caller.
/// `slot` selects one of three state slices inside `state` (`3 * channels`
/// floats); the pointers are not offset by the caller.
void LaunchConv1dStep(const float* x, const float* weight, float* state,
                      std::size_t channels, std::size_t slot, float* out,
                      hipStream_t stream, float* debug = nullptr);

/// L2 normalisation over each `head_dim` slice of a `[heads][head_dim]` row:
/// `y = x / max(||x||_2, eps)`.
void LaunchL2Norm(const float* x, float* y, std::size_t heads,
                  std::size_t head_dim, float eps, hipStream_t stream);

/// Rotary embedding on the first `n_rot` dimensions of each head.
/// `neox` (default for this family) rotates the pair `(i, i + n_rot/2)`;
/// otherwise the pair `(2i, 2i + 1)` is rotated instead.
void LaunchRope(float* x, std::size_t heads, std::size_t head_dim,
                std::size_t n_rot, std::size_t position, float freq_base,
                bool neox, hipStream_t stream);

/// One KDA (gated delta rule) step for every head, exactly as ggml's
/// GGML_OP_GATED_DELTA_NET does it: the state is `[heads][head_dim][head_dim]`
/// stored transposed (`state[h][j * head_dim + i] == S[i][j]`), and per token
///   1. `M[j][i] *= exp(g[h][i])`
///   2. `delta[j] = (v[h][j] - dot(M[j], k[h])) * beta[h]`
///   3. `M[j][i] += k[h][i] * delta[j]`
///   4. `out[h][j] = dot(M[j], q[h]) / sqrt(head_dim)`
/// `g` is `[heads][head_dim]`, `beta` is `[heads]`; the state updates in place.
/// The delta rule for a whole chunk in one launch, so the recurrent state stays resident
/// across the chunk's rows. Per-row maths is identical to LaunchDeltaRuleStep.
void LaunchDeltaRuleChunk(const float* q, const float* k, const float* v,
                          const float* g, const float* beta, float* state, float* out,
                          std::size_t heads, std::size_t head_dim, std::size_t rows,
                          std::size_t qkv_stride, std::size_t g_stride,
                          std::size_t out_stride, hipStream_t stream);

void LaunchDeltaRuleStep(const float* q, const float* k, const float* v,
                         const float* g, const float* beta, float* state,
                         float* out, std::size_t heads, std::size_t head_dim,
                         hipStream_t stream, float* debug = nullptr);

/// Index of the largest value; ties resolve to the lower index.
void LaunchArgmax(const float* x, std::size_t count, std::int32_t* result,
                  hipStream_t stream);

/// `out[i] = a[i] * b[i]` (may alias neither input).
void LaunchMulInto(const float* a, const float* b, float* out, std::size_t count,
                   hipStream_t stream);

/// `y = x + scale * z`
void LaunchAxpy(float* y, const float* z, float scale, std::size_t count,
                hipStream_t stream);

}  // namespace gufo::ext::ling3_air::hip
