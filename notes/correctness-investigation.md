
# Single-token / short-context mismatch (open, 2026-11-27)

Measured with explicit ids on both sides (no tokenizer, no chat framing):

| context | llama.cpp argmax | ours | relative logit error |
| --- | --- | --- | --- |
| `[156891]` | 90827 | 128206 | 5.9e-01 |
| `[156891, 156891]` | 2496 | 2496 | 1.6e-01 |
| `[156891, 156891, 156891]` | 2496 | 678 | 1.9e-01 |

Hypotheses already excluded, each by reading the reference rather than guessing:

- **sectioned rope.** The GGUF carries `rope.dimension_sections = [0, 8, 12, 12]`, but
  `bailingmoe3.cpp` only parses `hparams.rope_sections` and never passes it to `build_rope`, so the
  reference applies plain rope. Nothing to match.
- **KDA gate semantics.** The reference computes `gate = sigmoid(gate_proj * ssm_a) *
  kda_gate_lower_bound` and the CPU op then multiplies the state by `expf(gate)`
  (`ggml-cpu/ops.cpp`, the `kda` branch of `GGML_OP_GATED_DELTA_NET`). Our `LaunchKdaGate` uses the
  same sigmoid-with-lower-bound form and `LaunchDeltaRuleStep` applies `exp(g)` to the same value, so
  the decay convention matches; a wrong convention here would have been the obvious candidate for an
  error this size.

- **embedding dequantization.** `tests/forward_test.cpp`'s Q8_0 row check reports *worst absolute
  error 0* against the file's own bytes, so the embedding path is exact and cannot explain position
  zero.

- **MLA score scale.** The reference uses `1/sqrt(qk_head_dim)` = `1/sqrt(192)`; ours is the
  compile-time `kMlaScoreScale = 1/13.856406460551018` = `1/sqrt(192)`. Equal, and a mismatch here
  would have produced exactly this kind of systematic error.
- **MLA block structure.** Reference: `q_a -> attn_q_a_norm -> q_b`, split `q_nope`/`q_pe`, rope on
  both `q_pe` and `k_pe`, RMS-norm the latent, key = concat(normed latent, k_pe), value = normed
  latent, head output gated by `sigmoid(wqkv_gate * pre-attention hidden)`, then `wo`. Ours stores
  the same concatenated row in `mla_cache`, normalises the latent (`d5_latnorm`), uses the same
  per-head gate (`beta_pre` -> sigmoid), and the same latent value sum.

Still open: which block first diverges, and therefore whether the cause is the conv1d state, the MLA
path, the routed MoE, or the head. A per-block residual comparison against the reference's
`l_out-<n>` nodes is in progress.
