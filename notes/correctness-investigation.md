
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

- **the dump harness itself.** The per-block dumps were wrong: `dump_lp` copied with a bare
  `hipMemcpy`, ordered only against the default stream, while every block kernel runs on the
  engine's `stream`. Consecutive blocks therefore dumped identical (stale) contents, and
  `LING3_TOKEN_IDS`-style comparisons built on them measured nothing. `dump_raw`, `dump_at` and
  `dump_lp` now synchronise the stream first; the dumps are distinct block to block (42 files, 0
  duplicates).

## Where it diverges, measured on corrected dumps (2026-11-27)

Context `[156891, 156891]`, comparing our residual stream against the reference's `l_out-<n>` rows:

| block | rel err @ position 0 | rel err @ position 1 |
| --- | --- | --- |
| 0 | 0.970 | 0.833 |
| 1 | 0.870 | 1.444 |
| 2 | 0.883 | 1.861 |
| 4 | 0.217 | 1.323 |
| 11 | 0.104 | 0.942 |
| 20 | 0.046 | 1.321 |
| 30 | 0.122 | 0.923 |

Read it as a *shape*, not per-number: at position 0 the residuals converge as depth grows
(4.6 % by block 20), while position 1 stays wrong at every depth. The engine reproduces the first
token and loses the state that the second token needs, from the first block onward — a state-update
defect, not a per-kernel arithmetic one. The small early denominators make the position-0 figures at
blocks 0-3 noisy, so they should not be read as a second defect.

KDA pieces already verified against the reference source: conv window order `[s0,s1,s2,x]` with
`w[3]` on the newest sample, `silu` after the conv (both in the sequential path and in the chunked
path), 3-channel conv state of `d_conv-1 = 3` rows at `slot * 3 * channels`, `sigmoid(gate*a) *
lower_bound` then `exp(g)` in the delta rule, L2 norm of q/k, `o_norm` and the sigmoid output gate.

Still open: which block first diverges, and therefore whether the cause is the conv1d state, the MLA
path, the routed MoE, or the head. A per-block residual comparison against the reference's
`l_out-<n>` nodes is in progress.

## Localised to the MLA score path (2026-11-27)

Full-precision comparison against the reference's own tensors, context `[156891, 156891]`,
position 1:

| block | tensor | rel error |
| --- | --- | --- |
| 0-4 | `kda_out` (KDA attention contribution) | 0.001 - 0.008 |
| **5** | **`mla_out` (first MLA attention contribution)** | **0.420** |
| 6-8 | `kda_out` | 0.10 - 0.11 (inherited from block 5) |

The KDA attention blocks reproduce the reference to under 1 %, so the recurrent path is sound and
the routed MoE and the head are not implicated. The first MLA layer is.

Splitting block 5 by position isolates the half:

| position | rel error | what it exercises |
| --- | --- | --- |
| 0 | **0.005** | softmax over one key is 1, so only the value path: latent norm, `wv_b` absorb, gate, `wo` |
| 1 | **0.420** | the first position where the *scores* matter |

So the value path is right and the **score path** is wrong. Checked and found correct: the `wk_b`
absorb indexing (`q_abs[h][i] = sum_d q_nope[h][d] * wk_b[h][i][d]`, matching ggml's `{128,512,H}`
row-major layout), the `1/sqrt(192)` scale, and the cached row's value bytes (proven by position 0).
Not yet excluded: the `q_pe` rope at position >= 1, and what the score reads from a cached row.

`LING3_ROPE` cannot be used as a probe on the sequential path: that call hardcodes `true` (neox), so
the option has no effect there and must be fixed before the switch means anything.

### Probes that did not resolve it

`LING3_ROPE` was hardcoded to neox on the sequential MLA path (two `LaunchRope` calls passed a
literal `true`); that is fixed, so the switch now reaches the kernel. Comparing the two conventions
changed the dumped `d5_qpe.bin` but left `attn_5_1.bin` byte-identical, which cannot both be true for
a score that reads `q_pe` — so the dump and the score are not reading the same code path, and the
rope probe is inconclusive until the *sequential* path's scores are dumped (the `d5_*` dumps are
written from the chunked prefill branch, while the decode route runs `ForwardFromEmbedded`).

That is the next step, and it is well defined: dump `m.scores` inside the sequential MLA branch, then
recompute the expected probabilities in numpy from the reference's `q_b` and the cached row, and
compare probabilities rather than outputs.

Also excluded: the cache's storage type. `MlaAppendKernel` rounds the latent row to fp16
("llama.cpp keeps the KV cache in f16 by default"). Re-running the reference with
`type_k`/`type_v = GGML_TYPE_F32` produced byte-identical `mla_out-5` values, so this hybrid model's
reference cache does not follow those parameters and the fp16 store is not what separates the two
sides. It is still an approximation worth recording: with an f32 expectation it would cap the latent
row at ~5e-4 relative, which is the same order as the 0.1-0.8 % seen at the KDA blocks.
