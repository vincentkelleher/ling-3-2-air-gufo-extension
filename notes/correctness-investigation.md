
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

## The router "fix" measured worse on the actual criterion, so it was reverted

Reading `build_moe_ffn` suggested weights must come from the unbiased sigmoid (bias only for
selection). Implemented, it improved the raw number it was tuned against (two-token logit error
0.158 -> 0.095) and looked right in every dump. The criterion the contract actually names told a
different story:

| router variant | plain | arithmetic | tool-call | thinking-on | full matches |
| --- | --- | --- | --- | --- | --- |
| biased (original) | 93 | 3 | **110** | **157** | **1 / 4** |
| unbiased weights (reference reading) | 94 | 3 | 0 | 9 | 0 / 4 |

Reverted, keeping the softmax fix and the fp32 cache. The lesson is about the evidence order: logit
RMS is a proxy I chose, token-for-token agreement is what the contract asks for, and here they
disagreed. Most likely our `exp_probs_b` is not the tensor that reference code expects, so the two
bias placements are not comparable until that tensor is verified — that is the follow-up, not
another blind swap.

## The arithmetic case: one token choice, not a broken stream

`golden_arithmetic` diverges at character 3. Token level: both sides emit `17` then a space, then
ours emits `" times"` (id 3105) while llama.cpp emits `" ×"`. Everything before that is identical,
and the prompt tokenisation is byte-identical (the framed prompt is 54 tokens, printed via
`LING3_PRINT_PROMPT_IDS`). So this is a single greedy choice at one position, not a mis-tokenised
prompt or a drifting stream.

Which of the two explanations it is — a near-tie our numerics nudged, or our logits ranking the
reference's token far down — needs the logits at *that* step, and `forward_test` only prints its
top-5 for the final step. Next step: dump logits after the prefill pass (before the first generated
token) with `LING3_FINAL_LOGITS_F32`, feed the same prompt ids to the llama reference row tool
(already supports an explicit id list), and compare the top-5 and the softmax margin for id `×`
against id 3105. A margin of a few hundredths says numerics; a wide gap says something structural
still remains.

## The golden harness was comparing different contexts (2026-11-27)

Feeding both sides the *same token ids* makes the arithmetic case match **exactly** for 8 tokens
(`16 22 3105 220 17 18 13 2954`), while `llama-completion -f` on the rendered prompt file answers
`17 × 23 =`. The two were never in the same context: the CLI applies its own BOS and whitespace
handling. So the text-based divergences (93 / 3 / 110 / 157 characters) mixed harness error with
model error.

`tools/golden_set_ids.py` replaces it: the engine tokenises the rendered prompt, then both sides are
fed that exact id list (`tools/llama_greedy_ids.cpp` for the reference, `LING3_GREEDY_FROM_IDS` for
the engine) and decoded greedily. Measured at 32 tokens:

| case | result | first divergence |
| --- | --- | --- |
| plain | no | token 19 (engine 289, llama 259) |
| arithmetic | **yes** | — |
| tool-call | no | token 0 (engine 678, llama 156896) |
| thinking-on | no | token 2 (engine 341, llama 18549) |

`tool-call` first appeared to diverge at token 0, but that was a second bug in the *new* harness: it
carried its own approximation of the tool-call framing instead of the real one. With the renderer
imported from `golden_set.py` (one renderer, both harnesses) the case diverges at token 16, and the
four indices become 19 / none / 16 / 2 — a uniform, small numerical difference rather than one
prompt behaving specially.

Trying to isolate the last few percent at the head did not converge: selecting `result_output` /
`result_norm` in the reference dumper produced no files even though the run completed, so that
observer is still not reporting what it selects. Treat the head comparison as open until the dumper
is proven to write those tensors.

## Why a few percent remains: activation quantisation schemes differ

Checked this session: the LM head is `Q8_0 [2560, 157184]` and `token_embd` is `Q8_0` too, so the
engine's "prepared" (quantise-the-activation) path never applies to the head — `LING3_HEAD_PLAIN`,
added to test exactly that, is correctly inert and measured identical. The final logits are computed
by a plain Q8_0 GEMV, so the remaining few-percent gap is not an avoidable approximation at the head.

The residual difference is upstream and is a *scheme* difference, not an obvious bug: llama.cpp's CPU
backend quantises the activation row before a Q4_K matvec too (`ggml_vec_dot_q4_K_q8_K`), but with
Q8_K blocks — 256 elements with 8-bit sub-block scales — while our kernels use a 32-element
Q8_0-style layout with fp16 scales. Both are quantised; they are not the same quantisation, so
results differ by a fraction of a percent per matvec, which accumulates to the observed
0.3-4 % residual spread and flips near ties (golden divergences at tokens 19 / 16 / 2).

Reproducing llama.cpp token-for-token therefore most likely means reproducing its Q8_K activation
quantisation for Q4_K/Q6_K matvecs, not further tuning of the f32 paths. That is a contained but
real piece of work, and it is where the next session should start.

## Consolidated: what is left between this engine and token-for-token parity

Everything below was checked against the reference source or measured, not assumed. The order is
"closest to the truth first", because each item narrows the next.

1. **Fixed and verified** — uniform softmax (every head 0.5/0.5), fp16-rounded cache stores,
   stream-synchronised dumps, `LING3_ROPE` reaching the kernel, and the text-based golden harness
   that compared two different contexts (now `tools/golden_set_ids.py`).
2. **Checked and found equivalent** — the Q4_K activation quantisation. llama.cpp uses
   `iscale = -127/max` with `d = 1/iscale`; ours uses `amax/127`. The two reconstruct identically
   because `q_ref = -q_ours` and `d_ref = -d_ours`, so the product is unchanged; only exact `.5`
   ties round differently (`nearest_int` versus `__float2int_rn`). Not the gap.
3. **Checked and found matching** — the Q4_K dequantiser: `d*sc*code - dmin*m` per 32-value
   sub-block with `Q4ScaleMin` mirroring `get_scale_min_k4`, which is the reference formula. The
   "0.25 %" figure in `docs/quality.md` refers to the Q4_0 MTP-block comparison, not Q4_K.
4. **Still different, and the likely remaining cause** — accumulation arithmetic. llama.cpp's CPU
   path dots int8-quantised activations against Q4_K weights in integer arithmetic with int16
   sub-sums and one f32 scale per 256-block, while this engine's kernels accumulate in f32 per
   sub-block. Both are quantised; the *rounding at every accumulation step* differs, which is the
   right size to explain a 0.3-4 % residual spread that grows with depth and flips near ties
   (golden divergences at tokens 19 / 16 / 2 with `arithmetic` exact).

So the next honest attempt at parity is not another hypothesis hunt: it is to reproduce
`ggml_vec_dot_q4_K_q8_K`'s accumulation exactly (int16 sub-sums per 16 values, single f32 scale
application per super-block) and measure with `tools/golden_set_ids.py`. If that still leaves ties
flipping, the conclusion is that parity needs the reference's whole arithmetic, and the documented
divergence indices are the honest deliverable.

### The reference dumper cannot be extended reliably

Three attempts to add a tensor to the reference dumper (`q_b`, `result_output`/`result_norm`,
`ffn_moe_weights`) each produced **no files**, while `l_out`, `kda_out` and `mla_out` dump fine and
the run exits 0. The selected node apparently does not survive the scheduler's node selection unless
it is already reported for another reason, so "add a selector" is not a working method here. Any
further reference-side measurement should use `llama_greedy_ids` / the row tool (explicit ids,
final logits), which have been reliable, rather than new selectors.

Consequence for the open question: the reference's own MoE weights per layer are still unmeasured,
so the 2.5 scale placement (normalise then scale, which we do, versus scale then normalise, which
the source suggests is equivalent) is not verified against data — only against reading. With the dot
paths pinned to 1e-4, a 0.26 % difference at block 0 remains unexplained and would be settled fastest
by a reference that reports `ffn_moe_weights`, which this fork's dumper does not.

## The chunked prefill path disagrees with the sequential one (found 2026-11-27)

The engine implements the same block math twice: `Engine::PrefillChunk` (batched, used by serving
prefill) and `Engine::ForwardToken` (per token, used by decode). Run on the *same* prompt and
compared at the same final position, they disagree:

| block | rel (sequential vs chunked) |
| --- | --- |
| 0 | 3.0e-03 |
| 1 | 3.5e-03 |
| 5 | 8.6e-03 |
| 20 | 5.4e-02 |
| 41 | 1.6e-01 |

Two implementations of identical arithmetic must agree to ~1e-6, so this is a defect rather than an
accumulation-order effect — and the note in `docs/quality.md` that "chunked prefill sums the same
products in a different order, so it perturbs greedy output at these margins" understates it by three
orders of magnitude.

It also reframes the reference comparison: our per-block agreement with llama.cpp (0.3-4 %) was
measured on the *sequential* path, and 3e-03 at block 0 is the same magnitude as the unexplained
0.26 % there. Since serving prefills chunked and decodes sequentially, the mismatch perturbs every
served request and is the most likely cause of the golden divergence indices.

Block 0 is dense (`leading_dense_block_count = 2`), so the suspects are narrow: the batched GEMM used
for the chunk (`LaunchGemm` on Q4_K/Q6_K) versus the per-token GEMV the sequential path uses, or the
chunked KDA step.

Measurement note, so this is not repeated: the numbers above are trustworthy because both sides were
read with the *same* instrument — each run's final decode step dumps `x_<block>_<pos>.bin` from
whatever state its prefill produced, so the delta between the two files is the prefill divergence.
The existing bisect hook (`LING3_CHUNK_BLOCK`, which writes `chunk_x.bin`) is **not** comparable: the
same comparison through it reports 30-70 %, i.e. three orders of magnitude more, so its contents are
not the post-block chunk residual I assumed. That dump now exists: `PrefillChunk` writes the chunk's **post-block** residual for its last row to
`/work/xc_<block>.bin` when `LING3_DUMP_BLOCKS` is set, the same quantity the per-token path writes to
`x_<block>_<position>.bin`. Measured like-for-like at position 22 on the small prompt:

| block | rel (chunked vs sequential) |
| --- | --- |
| 0 | 6.1e-03 |
| 1 | 7.2e-03 |
| 2 | 1.6e-02 |
| 10 | 3.9e-02 |
| 20 | 2.2e-01 |
| 33 | 2.8e-01 |
| 41 | 2.1e-01 |

So the two paths are not equivalent: they agree to 0.6 % at the first block and diverge by up to 28 %.
This is the strongest defect signal found in this repository, and it explains why served output
(chunked prefill) and harness output (sequential) behave differently on the same prompt.

### First fix: the Q8_0 batched GEMM did not quantise its activation

`GemmKernel<kQ8_0>` multiplied dequantised weights by the raw f32 activation, while its per-token
counterpart `DotRow<kQ8_0>` quantises the activation per 32-value block (`d = amax/127`,
`__float2int_rn`, clamp ±127) — which is also what llama.cpp does. The chunked and per-token paths
therefore computed different products for every Q8_0 weight.

After aligning the GEMM to the same scheme, block 0 (whose attention weights are Q8_0) goes from
5.9e-03 to **1.3e-07** on both the attention contribution and the residual: the two paths now agree
to rounding at that block. Divergence now starts at block 1 (9.2e-04) and still grows with depth, so
the same class of mismatch exists for at least one more weight type; Q4_K and Q6_K use identical
quantisers in both paths (`QuantizeGemmActivationsKernel` and `ActivationScale` both compute
`amax/127` with the same rounding and clamp), so the next suspect is a projection that reaches the
batched path through a different kernel than the per-token path uses.

That suspicion is now excluded for block 1: blocks 0 and 1 are *both* entirely Q8_0 (attn_q/k/v,
attn_output, ffn_gate/up/down), block 0 agrees to 1.3e-07 after the fix, and block 1 still differs by
9.2e-04 with inputs that match to 1.3e-07. So the residue is inside the KDA per-row computation with
identical inputs and identical matvecs — the conv step, the delta rule, the gate, or the L2 norm —
reached through different buffers (`m.bq`/`m.bk`/`m.bv` rows versus `m.q`/`m.k`/`m.v`).

**Located exactly.** With `LING3_STATE_TAG`, both paths dump their recurrent state after prefill at
the same position. Measured on block 1:

| state | chunked vs sequential |
| --- | --- |
| conv state (3 channels x 3 rows) | **1.0e-07** (identical) |
| delta state (32 heads x 128 x 128) | **4.7e-04** |

So the projections, the conv and the SiLU feeding it all agree, and the divergence is in the KDA
delta-rule state update itself — which is exactly the magnitude seen at block 1 (9.2e-04 in the
attention contribution) and the seed of everything downstream. The remaining suspects are the
per-row inputs the chunked path hands that kernel (L2-normed q/k, the gated v, the decay gate, beta)
or the buffer it accumulates into.

Onset, now exact (per-row dumps). Block 1's delta state compared after every row:

| row | delta-state rel |
| --- | --- |
| 0 | 0.0 |
| 1 | 1.2e-07 |
| 2 | 8.1e-05 |
| 8 | 9.7e-04 |
| 22 | 5.0e-04 |

and the delta-rule *inputs* at block 1, per row:

| row | qn | kn | cv | gate | beta |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.0e-07 | 1.0e-07 | 7.4e-08 | 1.6e-07 | 4.1e-08 |
| 1 | **2.1e-04** | **1.2e-04** | **4.0e-05** | **1.8e-04** | 4.6e-08 |
| 2 | 9.9e-05 | 1.8e-04 | 1.4e-04 | 1.3e-07 | 5.3e-08 |

and the *input* those derive from matches at every row:

| row | post-norm hidden (the GEMV input) |
| --- | --- |
| 0 | 9.7e-08 |
| 1 | 1.2e-07 |
| 2 | 1.1e-07 |

**Conclusion — this is quantisation amplification, not different math.** Identical inputs (to 1e-7)
produce outputs that differ by 1e-4, because the GEMV quantises the activation to int8 with a
per-block scale: a last-bit difference in the activation flips one code near a rounding boundary,
which is about 1/127 = 0.8 % on that element and 1e-4 relative on the output, and then compounds with
depth to the 5e-2 measured at block 33. `beta` matches only because its matvec is small/F32, where
nothing is quantised mid-flow.

Consequences, and they are the important part of this whole investigation:

- the two prefill implementations agree on their arithmetic; the divergence is a numerical property of
  quantised matvecs, not a coding defect;
- bit-for-bit agreement between two implementations requires *identical* pre-quantisation values, i.e.
  the reference's exact summation order per block, not merely the same formula;
- the same mechanism explains the 0.3-4 % reference gap and the near-tie token flips.

So the state after row 0 is identical, and at row 1 the conv/L2-normed q/k/v and the decay gate already
differ while `beta` does not — the divergence enters through the projection/conv/gate path at the
second row, not through the recurrent state. `gate` and `beta` come from the *same* `m.bh + row`, and
`beta` matches while `gate` does not, so the difference is in the gate projection/output rather than in
the hidden input — the next thing to check is `kda_gate_pre` (the GEMV output) versus `kda_gate` (the
activation kernel) at row 1.

Earlier position-resolved measurement (the chunk dump now writes every row of the chunk, so positions
compare directly with the per-token path's `x_<block>_<position>.bin`):

| block | positions matching | first divergence |
| --- | --- | --- |
| 0 | 0-6 (about 1e-07) | position 7 (1.4e-03) |
| 1 | 0 (1.3e-07) | position 1 (1.7e-03) |

This rules out a pure per-position arithmetic difference: block 1's input at position 1 is block 0's
output there, which matches to 1.3e-07, yet block 1's own output at position 1 differs. Whatever
differs is carried *between* positions — the recurrent state after the previous row — and it starts
earlier in block 1 than in block 0, so two mechanisms may be involved. Also measured: all five
delta-rule inputs at block 1's last row match to about 1e-07, which is consistent with the state being
the carrier rather than the inputs.

Inspected since: the delta step *is* inside the per-row loop (`for (int i = 0; i < M; ++i)`, only
mis-indented), the conv state proves `cq/ck/cv` match, and the L2 norms, the decay gate and beta are
deterministic functions of that same data — so the kernel inputs should be identical. Before
instrumenting further, rule out the mundane explanation: the two runs may not have processed the same
number of tokens (`steps=0` reaches the chunked entry point differently), which would advance the
delta state by one row while the conv window still looks consistent. Print the processed token count
in both paths first; only if they match does dumping `qn/kn/cv/kda_gate/beta` per row become the next
step.

Earlier next step, retained for the KDA inputs comparison: dump the chunked path's per-row KDA
intermediates (`cq`/`ck`/`cv` after conv + SiLU, `kda_gate`, `beta`, the delta-rule output) next to
the per-token path's `p_convq_<pos>`/`ling3_cq0` dumps, and compare them in order. The first of those
that differs names the defect.

Two discriminators have been tried and are not usable as-is: `LING3_CHUNK_BLOCK` returns before the
FFN, and `LING3_CHUNK_SIZE` did not change any result in this mode (the run with size 1 was
bit-identical to size 8), so the chunk-width lever needs verifying before it can attribute the
difference to the batched GEMM versus the per-token kernels inside the chunk.
