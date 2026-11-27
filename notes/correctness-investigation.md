
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

### An unresolved contradiction worth picking up first

On the current build, with the sequential-path dumps working (a staged-source problem had been
hiding them: the CMake tree compiles the extension's staged copy, so `gufo extension install` must
run before every rebuild — that is what made several earlier probes look like no-ops):

- the score kernel matches a host recomputation from the dumped `q_abs`, `q_pe` and the two cache
  rows to 3e-06, so the kernel and those inputs are self-consistent;
- the rope term dominates the score (mean |rope| 4.60 at position 0, 1.26 at position 1, against
  scores of order 1-5), and `d5_qpe.bin` demonstrably differs between `LING3_ROPE=neox` and
  `interleaved`;
- yet `attn_5_1.bin` is **byte-identical** between those two rope conventions.

Attention output cannot be independent of a term that dominates the score, so either the output is
not a function of the probabilities the dumps show, or the two dumps come from different executions.
The next step is therefore not another file comparison: print the post-softmax probabilities from
inside the run (both rope modes) and confirm they differ; if they do, trace which probability array
`MlaLatentSum` actually reads.

## The defect (found and fixed, 2026-11-27)

`SoftmaxKernel` never broadcast the block-wide maximum. After the warp shuffles only lane 0 of
each warp holds a valid value, and the combine loop started at `w = 1`, so every other lane kept a
partial maximum. With `cache_len = 2` each lane therefore subtracted *its own* score, `exp(0)`
became 1 for both positions and every head returned **exactly 0.5/0.5** — a uniform attention that
ignores the scores entirely. That single defect explains every observation gathered above:

- position 0 looks correct (one key makes any distribution 1.0);
- position 1 is wrong by 42 % on the first MLA layer's output;
- two protocols (neox and interleaved rope) give byte-identical results, because the
  probabilities no longer depend on the scores at all;
- the error then accumulates through the remaining blocks.

Fix: reduce to one scalar per warp, reduce those in shared memory, and broadcast the result through
`__shared__` before normalising (both for the max and for the sum). Measured after the fix:

| quantity | before | after |
| --- | --- | --- |
| `mla_out-5` relative error, position 1 | 0.420 | **0.0156** |
| residual after block 5, position 1 | 0.050 | 0.050 |
| first-layer probabilities | 0.500 / 0.500 (all heads) | head-varying, e.g. 0.977 / 0.023 |

The cache was also switched from fp16-rounded stores to fp32: the reference's own `type_k`/`type_v`
made no difference to its output, and the rounding only injected error into later scores.

What remains after the fix is accumulation rather than a wrong formula: blocks 0-3 agree to
0.3-0.6 %, the first MLA layer adds ~1.5 %, and the logits still differ (relative 0.19 at two
tokens) because sub-percent residue in the residual stream flips router choices near their ties.

## Router parity (fixed) and the last two blocks (open, 2026-11-27)

The reference's MoE order, read from `llama-graph.cpp`: `probs = sigmoid(logits)`; the *bias*
(`exp_probs_b`) is added only to form `selection_probs`; weights come from the **unbiased** probs
(`ggml_get_rows(probs, selected_experts)`), are divided by their sum when `expert_weights_norm` is
true, and only then multiplied by `expert_weights_scale` (2.5 here). Our router added the bias
inside the sigmoid and normalised the biased values, so every expert weight was slightly wrong.

After fixing it (selection biased, weights unbiased -> normalise -> scale):

| context | rel logit error before | after |
| --- | --- | --- |
| `[156891, 156891]` | 0.158 | **0.095** |
| `[156891, 156891, 156891]` | 0.153 | **0.070** |

Attention contributions now agree to ~3 % at every block, including the last three.

**Still open: blocks 40 and 41.** At position 0 the residual rms jumps in the reference
(1.52 -> 3.53 between blocks 39 and 40) and jumps *further* in ours (1.53 -> 7.64). Isolating the
block's FFN contribution (`l_out_n - l_out_(n-1) - attention_n`, all dumped):

| block | ref ffn rms | our ffn rms | rel |
| --- | --- | --- | --- |
| 38 | 0.286 | 0.283 | 0.092 |
| 39 | 0.241 | 0.256 | 0.131 |
| **40** | **2.666** | **6.863** | **1.59** |
| **41** | **2.608** | **1.843** | **0.77** |

Two facts to explain: the reference's FFN contribution is ~10x larger at 40 and 41 than at 38-39,
and ours is 2.6x too large at 40 while too small at 41. `leading_dense_block_count` is 2, so 40 is a
Mixture-of-Experts block on both sides; the next thing to check is whether the last trunk blocks in
this architecture carry a different expert layout (or are the NextN/MTP block) rather than being
ordinary MoE layers, since a structural mismatch there would produce exactly this shape.

### Reading the rms trace

Per-block residual rms at position 0, reference versus ours, agrees to four decimals from block 0
through 39 (e.g. 2.2836 vs 2.2831 at block 20, 1.5209 vs 1.5358 at block 39) and then splits:

| block | ref rms | our rms |
| --- | --- | --- |
| 39 | 1.5209 | 1.5358 |
| 40 | 3.5326 | **7.6410** |
| 41 | 3.6703 | **5.9903** |

Both sides jump ~10x in contribution at block 40, so that amplification is real in the reference
too; ours overshoots. `block_count` is 43 with `nextn_predict_layers` 1, so the trunk is 42 blocks
on both sides and 40/41 are ordinary trunk layers — this is not an off-by-one in the layer count.

The precise next step, cheap and decisive: dump the router's selected expert ids at block 40 (the
existing `d2_ids` dump covers block 2 only) and compare them with the reference's `ffn_moe_topk`
tensor, which `build_moe_ffn` reports through `cb`. If the ids differ, the fault is in routing at
that layer (group config or bias); if they match, it is in the expert arithmetic itself.

### The ids dump cannot be trusted; use stderr instead

Extending the dump to block 40 gave router logits that look correct on both a known-good and the
suspect layer (block 2: 512 values from -6.28 to -2.72; block 40: -8.02 to -1.02) while the **ids
are all zeros on both** — impossible given distinct logits, so the ids file is not showing the
router's actual selection. Do not build conclusions on it.

The next probe should print the selected ids to stderr from the call site (like the probability
trace that found the softmax bug), not through a file: several of the file-based dumps in this
engine have needed a staging fix or a stream synchronise before they told the truth, and that cost
more time than the bug itself.

With routing indistinguishable at block 40 and blocks 0-39 matching to four decimals, the
remaining explanations for the block-40 overshoot are (a) a routing difference that only the
stderr trace can settle, or (b) the expert arithmetic for that layer, e.g. a per-layer expert
tensor whose quantization type our dequantiser mishandles.

### Router trace (stderr, reliable)

With `LING3_ROUTER_TRACE=1`, printed from the call site:

| layer | position | ids | weight sum |
| --- | --- | --- | --- |
| 2 | 0 | 326 468 499 498 454 330 188 270 | 2.5000 |
| 40 | 0 | 425 324 176 0 134 54 172 345 | 2.5000 |
| 2 | 1 | 486 326 169 511 69 339 471 65 | 2.5000 |
| 40 | 1 | 277 315 319 317 296 112 282 256 | 2.5000 |

Eight distinct experts each, weights summing to exactly the 2.5 scale, so the routing path behaves
sanely at the layer that diverges — including a `0` id at layer 40, which also proves the earlier
all-zero ids dump was the file, not the router. What is still missing is the reference's own top-k
for comparison: extending the reference tool's selector to `ffn_moe_topk` produced no files, so
that selector needs checking before the ids can be compared rather than just inspected.
