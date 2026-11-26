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
