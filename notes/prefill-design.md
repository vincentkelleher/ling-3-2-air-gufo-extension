# Batched prefill: design

Today `ForwardToken` runs one token at a time, so "prefill" costs the same
per token as decode (12.5 tok/s). That is why prefill sits at ~0.2 % of its
6,277 tok/s roofline while decode is at 25 % of its own. This note is the plan
for making prefill a batched pass; it is written to be executed without
re-deriving anything.

## Why it pays twice

A batched pass (M tokens per launch) reads each weight matrix once per *launch*
instead of once per token: for M = 32 the 4.75 GB of weights per token becomes
~150 MB per token amortised, so prefill approaches the bandwidth roofline rather
than being launch-latency-bound. It also makes the fused expert kernel worthwhile
(`LaunchGemvExperts`), because the MoE work per launch grows with M and the extra
silu/accumulate passes stop dominating.

## Kernel inventory

| stage | today | batched form |
| --- | --- | --- |
| embeddings | one row dequant | gather M rows (same kernel, M launches or one grid) |
| RMSNorm | row per block | one block per row, already M-capable: pass rows = M |
| GEMV (all projections) | warp per row, x is one vector | **GEMM**: out[m][n] = W[n] · x[m]; tile M x 8 rows per block, reuse the Q4_K/Q5_K/Q6_K/Q8_0 integer path per (row, tile) |
| KDA conv1d | per step, state shifted | run M steps in order inside one kernel (state is sequential), or M launches of the existing kernel |
| KDA delta rule | per token, state sequential | must stay sequential in token order; batch only across heads (already) |
| KDA L2 norm / gate / out-norm | per token | batched over M rows |
| MLA scores/softmax/latent sum | per token over the cache | flash-style: for M tokens, each attends to cache + earlier tokens in the chunk; needs the causal mask inside the chunk |
| MoE router | one token | one block per token (independent) — launch M blocks |
| MoE gather | per token, 24 launches | `LaunchGemvExperts` with ids from all M tokens, batch_stride = n_ff, out per (token, expert) |
| shared expert FFN | per token | GEMM over M rows |
| LM head | per token | only the last token needs logits during prefill |

## Order of work (each step measurable)

1. **GEMM for the dense projections only** (attention projections, dense FFN,
   shared expert). Keep the rest per token; run the block loop over M rows for
   those stages. Expected: prefill ~4-8x on the projection traffic, which is
   54.7 % of the bytes.
2. **MoE over M tokens**: router per token (M blocks), gather with M*top_k ids
   via `LaunchGemvExperts`, then the accumulate kernel indexed by token.
3. **MLA attention over the chunk** with a causal mask; the KDA path stays
   sequential but cheap (23.7 ms per token today, 5.7 %).
4. **Chunked prompt ingestion** in the test and the runner: prefill in
   512-token chunks, then decode.

## Validation

- Numerical: any M > 1 pass must produce the same logits for the last token as
  M sequential single-token calls, to within the fp32 noise already characterised
  (see `notes/engine-plan.md`: 0.01-0.02 logit differences decide tokens).
- Behaviour: the same 32-token greedy output as today for the pinned prompts —
  anything else means the mask or the cache append is wrong.
- Performance: report pp2048 and the depth rows through `tools/roofline.py`
  exactly as the README does today.

## What not to do

- Do not batch the KDA delta rule across tokens: the state is a running product,
  and the deterministic-reduction work showed how easily a reordering flips
  tokens.
- Do not shrink the tolerance of the GEMV checks again: the quantized dots differ
  from an exact float reference by design (see the operator test's comment).

## Batched GEMM kernel: implemented and tested (2026-11-27)

`LaunchGemm` computes `out[m][row] = dot(W[row], X[m])` for up to 8 tokens with
one block per row, so a weight block is read once and reused across the batch --
the property that makes prefill pay. F32 and Q8_0 are implemented; K-quants are
not yet (the dispatch deliberately falls through instead of running untested
code).

GPU unit test (in `tests/operator_gpu_test.cpp`): 6 rows x 64 cols x 4 tokens,
F32 and Q8_0 against a host reference: **4.8e-07** and **1.4e-06**. The operator
suite is now 26 checks, 0 failures.

Next: wire it into the block forward for the dense projections (attention
projections, dense FFN, shared expert) and validate that an M-token pass gives
the same last-token logits as M single-token calls, per
`notes/prefill-design.md`.

## The GEMM needed element-wise parallelism (2026-11-27)

First version spread one thread per 32-weight block, so only 80 of 256 threads
worked on a 2560-wide row and per-thread work grew with the batch: batch 8 took
**8.8x** batch 1 on the same 4096x2560 Q8_0 matrix (11 MB, one attention
projection) -- no amortisation at all.

Spreading elements across the block instead (`k = threadIdx.x; k += blockDim.x`)
gives, on that matrix:

| batch | ms/call | weight bandwidth | per-token cost |
| --- | --- | --- | --- |
| 1 | 0.099 | 113 GB/s | 0.099 ms |
| 8 | 0.272 | 41 GB/s | **0.034 ms** |

So a batched pass is worth **2.9x per token** on this shape, and the unit test
still matches the host reference at 4.8e-07. The kernel is in the operator suite
(`gemm bench`, printed every run) so any regression shows up immediately.

Next: wire it into the dense projections of the block forward and validate that
an M-token pass reproduces the sequential last-token logits.

## MTP pays only once verification is batched, 2026-11-27

Drafting works (7 of 9 drafts match), but an accept loop cannot speed anything up
in this engine today: verification means running the trunk over the drafted
tokens, and the trunk is a one-token forward, so K drafts cost K trunk steps —
exactly the unassisted decode cost — plus the draft block (~4 %). Speculative
decoding needs the verifier to consume K tokens in **one** pass and produce K+1
distributions, which is the same M dimension the prefill work needs.

So the order is now unambiguous:

1. **Batched forward** (M tokens per pass) — `LaunchGemm` is tested for F32/Q8_0
   at 2.9x per token on a real 4096x2560 projection; KDA and MLA state stay
   sequential and cheap (8 % and 17 % of the token respectively).
2. **Prefill** uses it first: pp2048 becomes measurable at all, which is half of
   the roofline report that is currently "not measured".
3. **MTP accept loop** uses the same path: draft K, verify in one pass, accept the
   matching prefix. Exactness is then structural (the verifier is the trunk
   itself), and the speedup is (accepted tokens per trunk pass) / (pass cost).

This also explains why the batched GEMM's per-token win (2.9x) matters twice: it is
the prefill gain and the speculative-verify gain at the same time.

### A batched forward must include the MoE to be worth it, 2026-11-27

Measured shares of a decode/token pass (`notes/engine-plan.md`, profiled):

| stage | share of time |
| --- | --- |
| Q8_0 GEMV (attention projections, LM head) | 34 % |
| Q4_K expert gate | 25 % |
| Q6_K expert up/down | 17 % |
| delta rule, router, F32, Q5_K | ~17 % |

A batched pass that only covers the dense stages (the easy part, since they are
Q8_0 and `LaunchGemm` already handles Q8_0) would touch 34 % of the work and, at
2.9x on that share, buy ~1.3x overall. The MoE has to be batched too for the 4-8x
the design note projects, and batching it means grouping the union of selected
experts across the M tokens (a token-by-expert-tile schedule) rather than the
current per-token per-expert GEMVs.

That is the single largest remaining piece of engineering in this project, and it
is also what makes MTP's accept loop pay (see `notes/prefill-design.md`). It should
be done in one go with the logits-equality check, not incrementally.

### MoE grouping needs large M, so prefill must be chunked, 2026-11-27

Arithmetic that changes the design: with top-8 of 512 experts, a batch of M tokens
makes M*8 selections over 512 experts, so the expected number of tokens per expert
is M*8/512 = M/64. At M=8 that is 0.125 selections per expert on average — i.e.
almost every selected (expert, token) pair is alone, and an expert's weights are
still read for a single token. Grouping buys nothing.

The expert read amortises only when M is a few hundred:

| M | expected tokens per expert | expert weight reads saved |
| --- | --- | --- |
| 8 | 0.125 | ~0 |
| 64 | 1 | ~1x |
| 256 | 4 | ~4x |
| 512 | 8 | ~8x |

So the batched forward that matters is **chunked prefill with M around 256-512**,
not a small batch: prompt ingestion in 512-token chunks, one pass per chunk, with
the MoE grouped by expert across the chunk. That also sets the memory budget: the
per-token scratch (q, attn out, ffn) becomes 512x its current size, which is
affordable for a 512-token chunk in unified memory but must be allocated
deliberately rather than by multiplying the existing buffers.

This is one coherent piece of work — chunked forward, grouped MoE, batched MLA
attention with a causal mask, validated by last-token logits against the sequential
path. It is the largest remaining engineering item in the project.

## Grouped-MoE schedule kernel, 2026-11-27

`LaunchBuildExpertLists` turns `ids[tokens][top_k]` into per-expert token lists
(counts, offsets, token slots), which is what a chunked pass needs to read each
expert's weights once for all the tokens that selected it. Counts are scattered on
the device; the 512-entry prefix sum runs on the host, which keeps the order
deterministic and costs less than a device scan at this size.

Unit test in the operator suite: 4 tokens x top-3 over 6 experts, checking counts,
offsets and that every slot really selected its expert — `29 checks, 0 failures`.

This is infrastructure for a chunked forward (M around 256-512) that does not
exist yet; it was built now because it is small, self-contained and testable, in
the same order the GEMM kernel was built ahead of the prefill wiring.

## Chunked forward: first attempt hit an aperture violation, 2026-11-27

Implemented `PrefillChunk` (dense stages batched through `LaunchGemm`, KDA/MLA and
routed MoE per token) and ran it on the 25-token prompt. The probe crashed in
`GemmKernel<kQ8_0>` with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` and a grid of
**[1048576, 1, 1]** blocks. A grid that large means the launch received a row count
of 2^20 instead of the real one (`kda_inner` 4096, `q_dim` 6144, `n_ff` 6144,
`n_embd` 2560, `n_vocab` 157184 are all far from it), so one of the calls in the new
path passes its arguments out of order -- most likely where a GEMM's `rows` and
`cols` were swapped for the MLA `kv_a` (576 per row) or the shared expert (768).

Reverted rather than left in the tree: the path is unreachable by default, and a
segfaulting branch is worse than a missing one. `LaunchMulInto` (an elementwise
multiply used by the chunked FFN) was kept, but it needs a check in the operator
suite before use -- untested code is what caused the earlier regressions.

Next attempt should build the chunk path **block by block**: first embeddings and
the norm over M rows (assert `m.bx` equals M sequential `ForwardToken` embeddings),
then one KDA block's projections, and so on, with an equality check at each step.
That is slower to write and much faster to debug than a 250-line method validated
only at the end.

## Chunked forward: two faults, reverted again (2026-11-27, keep this)

Attempt 2 added `PrefillChunk` with the `kda_out`/`mla_out` fix (KDA blocks keep
their output projection in `kda_out`; using `mla_out` dereferenced a null tensor,
which was the first aperture violation). It still faulted:
`HSA_STATUS_ERROR_MEMORY_FAULT`, no output.

What *is* proven on this path, and should be kept in the next attempt:

- Chunk embeddings equal per-token embeddings exactly (a debug hook compared 8 rows
  against `DebugEmbedding`; worst absolute difference **0**).
- The batched GEMM itself is validated for F32 and Q8_0 against a host reference
  (4.8e-07, 1.4e-06) and measured at 2.9x per token at M=8 on a real projection.
- Buffer sizes are right on paper: `bx/bh/battn/bffnh/bffout` 32 x n_embd,
  `bq` 32 x q_dim, `bk` 32 x max(kda_inner, 576), `bv` 32 x kda_inner,
  `bgated` 32 x max(kda_inner, v_dim), `bffa/b` 32 x n_ff.

So the defect is an index, not a size, and it is somewhere in the per-token loop
(the only part that mixes chunk rows with single-token scratch). The next attempt
should not re-run the whole thing: build it block by block with a dump after each
stage, exactly as the nump verification was done for the single-token path --
embeddings (done), then one KDA block's three projections with the GEMM output
compared against three GEMVs on the same rows, then conv, and so on. Each step is a
five-minute check with an exact equality, instead of a crash after 250 lines.

### Chunk path stages 1-2 verified, 2026-11-27

Two stages are now checked by the engine itself and printed by
`tests/forward_test.cpp` with `LING3_CHUNK_STAGE1=1`:

| stage | check | worst diff |
| --- | --- | --- |
| 1 | M-row RMSNorm vs a host recomputation from the same embeddings and block 0's weight | 1.4e-07 |
| 2 | batched projection (`LaunchGemm` over M rows) vs per-row `LaunchGemv` on the same rows | 2.98e-08 |

So the chunk buffers, the row indexing and the GEMM wiring are right for the
dense front of a block. Also found while reviewing the buffers: `bk` must hold
`kda_inner` per row (4096) for KDA blocks even though the MLA branch needs only
576, so it is sized for the larger of the two -- an overflow there would have been
the next fault after the null `mla_out` one.

The remaining unverified part is the per-token loop (conv, delta, MLA attention)
and the MoE/shared-expert tail. Stage 3 should therefore be: block 0 only, chunked,
with `m.bx` dumped and compared against the numpy block-0 reference already built
for the single-token path (which matched at 7e-07).

## Chunked forward works, 2026-11-27

`Engine::PrefillChunk` runs M tokens per block pass: the projections, the out
projection, the dense FFN and the shared expert are `LaunchGemm` over the chunk;
KDA conv/delta, MLA attention and the routed MoE stay per token. Two bugs were
fixed to get here:

- `bk` must hold `kda_inner` (4096) per row for KDA blocks even though the MLA
  branch needs only 576 -- sizing it by `kv_row` wrote past the buffer (the
  `HSA_STATUS_ERROR_MEMORY_FAULT`).
- KDA blocks keep their output projection in `kda_out`, not `mla_out`; using the
  MLA field dereferenced a null tensor (the earlier aperture violation).

Validation and measurement on the pinned 25-token prompt:

| path | time | rate | argmax |
| --- | --- | --- | --- |
| sequential (one token per pass) | 1.918 s | 12.51 tok/s | 678 (`The`) |
| chunked (8-token chunks) | 1.622 s | **15.41 tok/s** | **678 (`The`)** |

So the chunked path is **1.23x** faster and produces the same decision, which is
what the design projected for batching only the dense stages (~1.3x).

Next, in order: raise the chunk size to 256-512 (buffers are sized for 32 today)
and group the routed MoE by expert across the chunk, which is where the 4-8x lives;
then the MTP accept loop, which needs this same verifier.

### Chunked prefill scaling, 2026-11-27

`LaunchGemm` now tiles its batch in slices of 8 (the kernel keeps one accumulator
per token in registers), so chunk sizes above 8 work; buffers are sized for 256.

104-token prompt, same argmax (678) in every row:

| path | time | rate | vs sequential |
| --- | --- | --- | --- |
| sequential | 8.330 s | 12.48 tok/s | 1.00x |
| chunk 8 | 6.886 s | 15.25 tok/s | 1.22x |
| chunk 64 | 6.478 s | 16.21 tok/s | **1.30x** |

That is the plateau the profile predicted for batching only the dense stages:
Q8_0 work is 34 % of a pass, and 2.9x on that share gives ~1.3x overall. The 4-8x
needs the routed MoE grouped by expert across the chunk (25 % + 17 % of the pass),
which is the next piece.

### Grouped MoE: implemented correctly, still slower than per-token (2026-11-27)

Grouped MoE is in the chunked path behind `LING3_GROUPED_MOE`. It is *correct* (the
same argmax at every chunk size after the K-quant GEMM existed -- before that it
was silently wrong because `LaunchGemm` dropped K-quant launches, which is exactly
the class of bug the operator suite exists to catch and did not, since no test
covered a K-quant GEMM).

What was built and kept:

- `GemmKernel` branches for Q4_K and Q6_K with the same per-group arithmetic as the
  per-row dot (`d*sc*code - dmin*m`, prepared int8 activations per 256-block).
- Prepared activations quantized per 8-token tile, immediately before its GEMM;
  quantizing the whole chunk overflowed the 8-token buffer, and quantizing inside
  the weight loop cost 5x.
- `LaunchGatherRows`/`LaunchScatterAdd` with pair-index slots, plus host counts and
  offsets from `LaunchBuildExpertLists`.

Measured, 104-token prompt, all with argmax 678:

| path | rate |
| --- | --- |
| sequential | 12.48 tok/s |
| chunked, per-token MoE (default) | **16.20 tok/s** |
| chunked, grouped MoE | 6.5-7.7 tok/s |

Grouped MoE loses because the host loops over all 512 experts per block (most hold
one or two tokens at this chunk size) and the naive K-quant GEMM is several times
slower per launch than the tuned GEMV. To make it pay it needs (a) a K-quant GEMM
with the GEMV's efficiency, and (b) chunk sizes where experts actually collect
tokens (M in the hundreds). Until then the default stays per-token.

### MTP accept loop: first attempt accepts nothing (2026-11-27)

Added `Engine::SnapshotState`/`RestoreState` (device copies of every KDA conv +
delta state and every MLA cache) and `VerifyDrafts` (chunked pass that returns the
greedy argmax at each position, preceded by a snapshot so a rejected prefix can be
rewound). Both compile; `VerifyDrafts` is **unverified** and currently unused.

The speculative loop built on them in `tests/forward_test.cpp` measured 2.54 tok/s
with **0 drafts accepted** and produced a sequence that is not the plain greedy one,
so its position bookkeeping is wrong: the trunk consumes the current token inside
the loop and then verifies drafts starting at the same position, which double-counts
one position. The hook was removed; the engine primitives were kept.

Correct shape for the next attempt:

1. State positions **explicitly**: `position` is the index the next token will
   occupy. Feeding `token` consumes `position` and advances it.
2. Draft K tokens with block 42 *at* the advancing positions, chaining each draft's
   hidden state (which `ForwardMtp` leaves in `m.h`).
3. Verify with one chunked pass starting at the position of the first draft, so
   `argmax[i]` is the trunk's prediction at draft slot `i`.
4. Accept while `argmax[i] == draft[i]`; the token to emit next is
   `argmax[accepted]`. Restore the snapshot, then commit exactly the accepted
   prefix so the state matches the emitted sequence.
5. Check the simplest invariant first: with K = 0, the loop must reproduce plain
   greedy byte for byte. Only then raise K.

## Why the MTP accept loop cannot pay yet (2026-11-27, arithmetic)

Speculative decoding spends, per iteration: one trunk step to commit the last
token, K-1 MTP block forwards to draft, and one verification pass over K tokens.
With K = 4 and the rates measured on this engine:

| piece | cost |
| --- | --- |
| trunk step (decode) | 80 ms |
| MTP block (one block of 42, per draft) | ~10 ms x 3 |
| verify 4 tokens, chunked | 4 / 16.2 tok/s = 247 ms |
| total per iteration | ~357 ms |
| tokens produced if 3 of 4 draft tokens are accepted | 3 |
| effective | **119 ms/token, worse than the 80 ms of plain decode** |

The reason is structural: verification is only cheap when a K-token pass costs
about what one token costs, because the weights (4.75 GB per token) are read once
either way. Today a 4-token chunked pass costs 4/1.3 = 3.1 token-equivalents, so
verification has no advantage and the drafts are pure overhead.

The fix is the same work the prefill gate needs: a K-quant GEMM as efficient as the
GEMV path and a grouped MoE whose per-expert loop runs on the device, so an M-token
pass approaches one token's weight traffic. At that point the table above turns
over: verify 4 tokens for ~1 token-equivalent, accept ~3, and the loop is worth
running.

So the order stands as recorded earlier, but for a sharper reason: **batching is
not just a prefill optimisation, it is what makes speculation possible at all.**

### Device-side grouped MoE: put the kernel in kernels.hip (2026-11-27)

The grouped schedule needs a kernel that, for each expert, loops over the tokens
that selected it and computes the dot for every output row -- so the expert's
weights are read once per chunk instead of once per token. That kernel must live
**in `kernels.hip`**, beside `DotRow`, which is private to that translation unit:
a `__device__` dispatch declared in one .hip file is not visible to another under
this build (each file is compiled separately, device functions are not exported).

Attempted the other way round first (dispatch in `kernels.hip`, call from
`moe.hip`) and it does not link, so the work was reverted; the tree is green at 31
operator checks.

Shape of the next attempt, worth ~2x on the grouped path by removing both the host
loop over 512 experts and the per-expert launch overhead:

1. In `kernels.hip`: `MoeExpertMatvecKernel<TYPE>` with one warp per (expert, row),
   looping over `slots[offsets[e] .. offsets[e] + counts[e])` and writing
   `out[pair][row]`; reuse `DotRow`.
2. In `kernels.hip`: `MoeActivateKernel` for `act[pair][j] = silu(gate) * up`.
3. In `moe.hip`: a launcher that takes the counts/offsets/slots and issues those
   two, then the existing down projection and `LaunchScatterAdd`.
4. Validation is already in place: the chunked path's argmax must stay 678 at every
   chunk size, and the golden set must not move.

### Grouped MoE removed from the tree, 2026-11-27

The device-side grouped schedule (one warp per expert-row, looping that expert's
tokens) was written, wired and measured: it **faulted** with
`HSA_STATUS_ERROR_MEMORY_FAULT`, so it was removed together with its two kernels
rather than left as a hazard behind a flag. The design and the four-step shape are
recorded above; nothing else in the tree depended on it.

State after the removal, measured on the 104-token prompt and re-verified:

| path | rate | argmax |
| --- | --- | --- |
| chunk 8 | 15.61 tok/s | 678 |
| chunk 104 | **16.21 tok/s** | 678 |
| sequential | 12.48 tok/s | 678 |

Operator suite 31 checks / 0 failures, `ctest` 5/5.

What remains for the prefill gate: make a K-token pass cost closer to one token's
weight traffic. The two levers are the ones already identified -- a K-quant GEMM as
fast as the GEMV path, and a grouped MoE whose per-expert loop runs on the device
(the removed attempt; the fault is most likely an expert count or offset mismatch
between `LaunchBuildExpertLists`' 512-entry tables and the kernel's `experts`
argument, which should be passed rather than hard-coded).

## MTP equivalence and restore: probes inconclusive, claims not established (2026-11-27)

Three probes in `tests/mtp_test.cpp`, and the honest reading of them:

| probe | result |
| --- | --- |
| per-layer state digest across snapshot → token → restore | **0 of 43 layers differ** — state restore is bit-exact |
| logits of two identical 6-step decodes | **max delta 0** at every step — decoding is deterministic |
| restore round-trip of an 8-token greedy run | reports INEXACT at index 5 |
| MTP speculate-and-verify vs plain | diverges at index 5, 0 drafts accepted |

The first two and the last two contradict each other, and the contradiction is in the
harness, not the engine: the engine owns **one** snapshot buffer, every probe calls
`SnapshotState`, and `VerifyDrafts` also snapshots internally, so a `RestoreState` in one
block can return to a checkpoint taken in another. The round-trip block has been made
self-contained once and still reports INEXACT, which means at least one more such ordering
error remains (the most likely: it restores without re-feeding the last prompt token, so the
logits it starts from belong to the previous run's last step).

So the correct statement today is **not established**, not "broken":

- cache restore is *state-exact* (digest) and decoding is deterministic (logits), which is
  what the byte-exact claim needs, but the end-to-end equality test does not yet pass;
- MTP equivalence is unproven, and the 0 % acceptance suggests the head is not predicting the
  next token on this prompt at all, which needs checking separately from the rewind.

Next step is to rewrite the harness around a single rule: one probe per engine state, every
restore paired with its own snapshot and a re-warm of the logits, and no probe relying on a
snapshot taken by another. `ling3_air.mtp` stays `WILL_FAIL` until then.
