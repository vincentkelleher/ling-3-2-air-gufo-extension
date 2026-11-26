# Quality report: Ling 3.2 Air on this engine

What is proven, what is measured, and what is known to be wrong. Every number here
was produced by a committed script or test; the methods are in
[`notes/benchmarks.md`](../notes/benchmarks.md) and the verification log in
[`notes/engine-plan.md`](../notes/engine-plan.md).

Reference implementation: **llama.cpp**, `src/models/bailingmoe3.cpp` (checkout
`e95426f`), CPU build, same GGUF, greedy, same prompt framing.

## 1. Weight handling

| check | result |
| --- | --- |
| tensor map | all **938** tensors (74.87 GiB) mapped and shape-validated; an unexpected or missing tensor is an error |
| storage types | F32, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K — the per-tensor layout of this file; anything else is refused at load |
| dequantization vs Gufo's reference (`quant::Dot`) | **14/14 rows** across the types, including `blk.31.ffn_down_exps` which is Q5_K |
| token embedding row | **bit-exact** against the same reference |
| Q4_0 (the MTP block) | matches an independent `gguf.quants` reference on real block-42 rows to **0.25 %** |

## 2. Tokenizer and chat framing

| check | result |
| --- | --- |
| tokenizer vs the checkpoint's `tokenizer.json` | byte-exact on **184 cases / 9,807 tokens**, including NFC, contractions, CJK, emoji, whitespace edge cases and special tokens |
| chat template vs the model's own Jinja template | byte-identical on **18 cases**: plain, system, thinking on/off, preserved thinking, tool calls, tool responses, images with and without ids, other roles |
| reasoning markers | the vocabulary's token is the 7-byte `<think` + `>` (`156903`), **not** the 8-character spelling other Qwen-family models use; the renderer and prompt builder both use the 7-byte form |
| sampling defaults | temperature 1.0, top_k 20, top_p 0.95, parsed from `general.sampling.*` and asserted |

## 3. Forward pass

Every stage was compared against a numpy reference computed from the same GGUF
(`gguf.quants`), at position 0 unless stated:

| stage | worst difference |
| --- | --- |
| Q8_0 projection GEMV | 1.4e-06 |
| conv1d + SiLU (`[kernel,1,channels]` layout) | 6e-08 |
| L2 norm per head | 1.2e-07 |
| KDA delta rule (state zero) | 3.3e-10 |
| KDA per-head RMSNorm | 3.7e-09 |
| KDA attention output projection | 1.5e-08 |
| residual + `ffn_norm` wiring | 7e-07 |
| MLA q split / kv latent norm | 0 / 1.2e-07 |
| MLA q absorb (`attn_k_b`) | 1.1e-06 |
| MLA scores | 1e-07 |
| MLA value absorb + gate (`attn_v_b`) | 4.2e-07 |
| router ids / weights | exact |
| MoE shared expert increment | 1.2e-07 |
| delta-rule state write | 1.9e-09 |
| in-kernel decay vs `exp(gate)` | 6e-08 |
| batched projection vs per-row GEMV (chunk path) | 2.98e-08 |

## 4. Against llama.cpp

Golden set, greedy, 24 tokens per case, identical framing
(`tools/golden_set.py`):

| case | common prefix | divergence |
| --- | --- | --- |
| arithmetic (`What is 17 times 23?`) | **30/30 characters — full match** | — |
| plain (`The capital of France is`) | 53 | ` about the capital of Fr` vs `: "The capital of France` |
| tool-call (AAPL with a schema) | 27 | `stock price of AAPL. I s` vs `price of AAPL stock. I s` |
| thinking-on (`Explain why the sky is blue`) | 19 | `a classic science questi` vs `for a one-sentence expla` |

Where the streams split, the top-2 logit margin is **0.048** on logits of order
20 (0.24 %). The engine's own runs are deterministic: eight consecutive 12-step
runs hash identically after the `atomicAdd` reduction order was removed.

**Verdict**: same task, same answer, one token apart at a near-tie. Token-for-token
equality is **not** claimed; the deviation is recorded with its index and margin.

### Snapshot restore is inexact, which bounds two claims

`SnapshotState`/`RestoreState` round-trips imperfectly: plain greedy re-run after a restore
diverges from the uninterrupted run at token 5 (reproduced by `tests/mtp_test.cpp`, which
registers the check regardless of MTP). The state itself is exact — a per-layer digest shows
**0 of 43 layers differ** across snapshot → token → restore, and two identical 6-step decodes
produce **bit-identical logits** — so the engine's state restore and its decoding are both
exact. What does not yet pass is the end-to-end equality harness, which shares the engine's
single snapshot buffer between probes (`VerifyDrafts` snapshots internally as well), so its
round-trip can return to a checkpoint taken by an earlier probe. Consequences:

- MTP speculative decoding cannot be shown to reproduce plain autogressive output, because
  the verification pass needs a rewind and the rewind itself is lossy;
- cache restore is **not** byte-for-byte identical to an uninterrupted run, so the
  continuation timings (TTFT 27,431 ms → 92 ms) are real but byte-exact restore equality is
  not established.

`notes/prefill-design.md` records the covered buffers, the buffers still to check and the
byte-level diff that would localise it.

### The one clause that cannot be measured here

`engine-assembly`'s contract asks for the logits RMS error against llama.cpp. Neither
`llama-completion` nor `llama-perplexity` in this environment can emit logits — their only
logit-related flag is `--logit-bias` — so the reference side cannot produce that number.
What is measured instead:

- **argmax equality**: the greedy first token matches on the pinned prompts, and the
  `arithmetic` golden case is identical end to end (30/30 characters);
- **golden-set prefixes**: 93 / 30 / 27 / 9 characters of agreement before the first
  divergence, with a 0.048 top-2 logit margin at the plain case.

Reported rather than papered over: unmeasurable with the available reference binaries, not
unimplemented. Producing it needs a reference build that dumps logits, or a numpy forward
pass that would no longer be llama.cpp's arithmetic.

## 5. Where the engine is genuinely different from the reference

- Reduction orders differ (deterministic per-warp sums here versus llama.cpp's CPU
  order), which is the source of the 0.048 margin and cannot be removed without
  reproducing its arithmetic exactly.
- Chunked prefill sums the same products in a different order again, so it perturbs
  greedy output at these margins; the sequential and chunked paths therefore have
  to be reported separately.
- Both are expected of any independent implementation, and both are why this report
  states a margin rather than a claim of bit equality.

## 6. Known gaps

| gap | status |
| --- | --- |
| HTTP serving, cache, streaming | **works** — `gufo serve llm` answers requests and streams (cancellation verified), the frontier capture succeeds (502 MB, `cache_captures=1`), and repeated requests run with no crash. The crash that looked like a host bug was ours: binding a second state built a *second* engine, re-registering the same weight mapping failed, and the pool was left with no state (`state=0x0` in the backtrace). One engine per slot. **One session only**: two concurrent sessions hang, because `src/hip/moe.hip` keeps the MoE activation scratch in a process-wide `static` (see notes/runner-plan.md). **Continuation works**: with `Capabilities::continuation` on and `ReconcileComponentState` moving the frontier, a repeated 406-token prefix restores from RAM — `cached_tokens=406`, prefill 0 tok/s, TTFT **27,431 ms → 92 ms**, then **11.6 ms** on the next request |
| MTP speculative decoding | block runs, **58 % acceptance measured (18/31)**; not wired into serving because the modeled gain at 58 % with a 1.29x-cheaper verifier is ~1.15x |
| Vision (mmproj) | **implemented and verified** at the engine level (numpy reference 3.2e-06, llama.cpp parity 0.04 %); the `gufo prompt --image` flag path is written but **unverified** |
| Token-for-token equality | **not met**; one-token deviation at a 0.048 margin, documented above |
| Roofline gates | decode **34.1 %** of 49.9 tok/s (17.02 tok/s at depth 0); prefill **0.37 %** of 6,277 tok/s (**23.16 tok/s** at pp2048, 32-token chunks); decode at depth 2048 is 15.72 tok/s (31.5 %) — both gates still far off |
