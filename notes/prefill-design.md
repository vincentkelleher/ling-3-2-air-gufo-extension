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
