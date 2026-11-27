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

### Where the decode gate actually stands

`tools/scatter_probe.hip` reads 1.4 MB windows out of a registered 4 GiB mapping, the pattern the
routed MoE uses. Its first run used one block per window (2 048 threads, too few) and reported
30-52 GB/s; with 32 blocks the same pattern reaches **100 GB/s**, and window size does not matter.

The MoE itself runs at ~40-47 GB/s while moving ~1.2-1.5 GB per token, so it is about **2.5x
below what the pattern supports** — a kernel-efficiency gap, not a hardware limit. At 100 GB/s the
expert path would take ~13 ms instead of 32 and decode would be ~24 tok/s (48 % of the 49.9 roofline)
rather than 32-34 %. The arithmetic is not the gap: removing the 4-bit unpack and the multiply from the expert dot
leaves the rate unchanged (16.75 vs 16.93 tok/s), so the path is read-bound. The read pattern is
the difference — each warp's 32 lanes touch addresses 144 bytes apart, so one load instruction
spans 32 cache lines, while the probe's threads read adjacent words and reach 100 GB/s. A coalesced variant was then written and measured (one warp per row, lanes reading adjacent bytes
per group) and is **neutral** (16.91 vs 16.93 tok/s), and the probe shows concurrency is not the
limit either: 1 024 blocks over 24 windows reaches 464 GB/s. Nine variants and three probe shapes
later the gap is real (the pattern supports 90-460 GB/s, the kernel sustains ~39 GB/s) and its
cause is unidentified without a profiler, which this environment does not provide.

### Snapshot restore is inexact, which bounds two claims

`SnapshotState`/`RestoreState` round-trips imperfectly: plain greedy re-run after a restore
diverges from the uninterrupted run at token 5 (reproduced by `tests/mtp_test.cpp`, which
registers the check regardless of MTP). The state itself is exact — a per-layer digest shows
**0 of 43 layers differ** across snapshot → token → restore, and a full 8-token greedy run
after a rewind reproduces the reference **token for token with logits delta 0** (the checkpoint
must sit before the last prompt token, so a rewind feeds it exactly once) — so the engine's state restore and its decoding are both
exact. What does not yet pass is the end-to-end equality harness, which shares the engine's
single snapshot buffer between probes (`VerifyDrafts` snapshots internally as well), so its
round-trip can return to a checkpoint taken by an earlier probe. Consequences:

- MTP speculative decoding **does** reproduce plain autogressive output exactly (8/8 tokens,
  logits delta 0) once each section owns its checkpoint; `tests/mtp_test.cpp` asserts it;
- cache restore is **not** byte-for-byte identical to an uninterrupted run, so the
  continuation timings (TTFT 27,431 ms → 92 ms) are real but byte-exact restore equality is
  not established.

`notes/prefill-design.md` records the covered buffers, the buffers still to check and the
byte-level diff that would localise it.

### Single-token forward and first-token agreement

**Bare BOS does not pass.** Feeding explicit ids to both sides (no tokenizer, no chat framing), on
the pinned Q4_K_M against llama.cpp CPU `e95426f`:

| context | llama.cpp argmax | extension argmax | relative logit error |
| --- | --- | --- | --- |
| `[156891]` | 90827 | 128206 | 5.9e-01 |
| `[156891, 156891]` | 2496 | 2496 | 1.6e-01 |
| `[156891, 156891, 156891]` | 2496 | 678 | 1.9e-01 |

The single-token case (all 157,184 logits finite) differs by RMS **1.069405** and maximum
**7.808860**. Because the error stays near 1.6e-01 even where the argmax agrees, it is systematic
rather than the rounding order that explains the 0.048 margin on longer prompts. The block where it
first appears is being localized; until then this input does not meet `engine-assembly`. Agreement
over chat-prompt prefixes does not establish agreement here.

Reproduce using a matching llama.cpp source checkout (`$LLAMA`) and CPU build (`$LIB` points
to its `bin` directory). Run the forward test with no prompt environment overrides:

```sh
c++ -O2 tools/llama_bos_logits.cpp -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  -L"$LIB" -Wl,-rpath,"$LIB" -lllama -o /tmp/llama-bos
/tmp/llama-bos "$MODEL" cpu.f32
LING3_FINAL_LOGITS_F32=gpu.f32 "$FORWARD_TEST" "$MODEL" 1
```

Both files contain one native FP32 row in vocabulary order. RMS is `sqrt(mean((cpu-gpu)^2))`
computed in double precision.

## 5. Where the engine is genuinely different from the reference

- The GPU and CPU use different reduction orders. This can matter at small margins, but
  has not been established as the explanation for the single-BOS discrepancy.
- Chunked prefill sums the same products in a different order again, so it perturbs
  greedy output at these margins; the sequential and chunked paths therefore have
  to be reported separately.
- Both are expected of any independent implementation, and both are why this report
  states a margin rather than a claim of bit equality.

## 6. Known gaps

| gap | status |
| --- | --- |
| HTTP serving, cache, streaming | **works** — `gufo serve llm` answers, streams, and acts on cancellation (`tools/check_http_restore.py`, which also requires a successful private restore in the log so a cold fallback cannot pass). Continuation restores a prefix from the component cache and the restored continuation matches a fresh server byte for byte; one private checkpoint is 154,697,732 bytes at context 4096. Session isolation is verified with two concurrent requests on `--sessions 2`: traces show two resident banks on a single shared engine, both matching their own cold runs, with engine operations serialized (an isolation result, not a throughput claim). An earlier build reported a 502 MB capture and a one-session limit; both are superseded |
| MTP speculative decoding | block runs, **58 % acceptance measured (18/31)**; not wired into serving because the modeled gain at 58 % with a 1.29x-cheaper verifier is ~1.15x |
| Vision (mmproj) | **implemented and verified** at the engine level (numpy reference 3.2e-06, llama.cpp parity 0.04 %); the `gufo prompt --image` flag path is written but **unverified** |
| Token-for-token equality | **not met**; one-token deviation at a 0.048 margin, documented above |
| Roofline gates | decode **32-34 %** of 49.9 tok/s (16.2-17.0 tok/s at depth 0); prefill **0.37-0.38 %** of 6,277 tok/s (**23.4-23.8 tok/s** at pp2048, 32-token chunks); decode at depth 2048 is 15.72 tok/s (31.5 %) — both gates still far off |
