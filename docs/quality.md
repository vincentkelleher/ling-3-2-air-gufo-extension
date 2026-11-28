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

### Snapshot restore is exact under the per-probe checkpoint rule

`SnapshotState`/`RestoreState` round-trip the trunk state (KDA conv + delta, the MLA
caches) exactly. The engine owns one snapshot buffer, so each probe must take its
checkpoint immediately before use and, when the checkpoint sits before the last prompt
token, feed that token again after restoring because the snapshot does not carry the
logits. Under that rule a per-layer digest shows **0 of 43 layers differ** across
snapshot → token → restore, an 8-token greedy run after a rewind reproduces the
reference **token for token with logits delta 0**, and the speculative step shape
reproduces plain greedy over 32 tokens with logits delta 0 (`tests/mtp_test.cpp`). The
earlier "diverges at token 5" observation was probes sharing that one buffer; it was a
harness defect, not an inexact restore.

The 8/8 equivalence (`9196682`, 2026-11-27) predates the chunked-router fixes
(`063801c`, `7215325`) but is decode-side and unaffected by them: the proof feeds the
prompt one token at a time and verifies sequentially, so it never enters the `rows > 1`
chunk path those fixes changed. `VerifyDrafts` itself was unproven until it was checked
against two `ForwardToken`s; that check found it stopped after the last block's
attention (so its rows came from a hidden state the token path never produces) and ran
the head through a different GEMV than the token path. Both are fixed by running the
full pass and sharing one output-head implementation, and its per-row argmaxes now match
the token path exactly.

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

- The GPU and CPU use different reduction orders, and the engine quantises activations to int8 for
  its quantised matvecs exactly as llama.cpp does. This was measured the hard way: two of this
  engine's own implementations of the same block math (batched prefill and per-token decode) agreed on
  the *post-norm hidden input* to 1e-7 at every row and still disagreed on the outputs by 1e-4 from
  the second row onward — a last-bit difference in the activation flips one quantised code near a
  rounding boundary, which is about 1/127 = 0.8 % on that element, and then compounds with depth.
  Aligning the batched Q8_0 dot and the F32 router reduction to the per-token order fixed exactly
  that: the two paths are now **byte-identical** (966 of 966 residual rows and 42 final-position
  attention contributions, verified independently), so the remaining reference gap is the engine's
  own arithmetic versus llama.cpp's, not an internal inconsistency.

  The same mechanism is behind both the remaining reference gap and the near-tie token flips:
  matching the reference token-for-token would require reproducing its exact summation order per
  block, not merely the same formula. It is not a coding defect, and it is why this report states a
  margin and documented divergence indices rather than claiming bit equality.
- Chunked prefill sums the same products in a different order again, so it perturbs
  greedy output at these margins; the sequential and chunked paths therefore have
  to be reported separately.
- Both are expected of any independent implementation, and both are why this report
  states a margin rather than a claim of bit equality.

### MTP speculation is wired but opt-in (2026-11-28)

`DecodeStep` returns two selections when all of these hold: the server was started with
`LING3_SPEC_DRAFTS=1`, the request is an unmodified argmax (temperature 0, no penalty,
constraint or reasoning budget), the token budget is at least two, and the trunk's final
hidden is valid. It drafts one token with block 42, verifies `{current, draft}` in one
chunked trunk pass, restores, and re-feeds exactly the accepted prefix with
`ForwardToken`, so the retained state comes only from the token path.
`draft_tokens`, `draft_tokens_accepted`, `draft_rounds` and `acceptance_pct` come from the
host's normal generation metrics; the runner adds `[runner] spec step ...` under
`LING3_RUNNER_TRACE`. Speculation is off by default because it measured slower.

Measured (Q4_K_M, greedy, one request at a time, 2026-11-28):

| prompt (`max_tokens` 64) | drafts | accepted | acceptance | decode |
| --- | --- | --- | --- | --- |
| `The capital of France is` | 61 | 3 | 4.9 % | 7.17 tok/s |
| `Here is one fact about the red fox. Where does it live?` | 62 | 1 | 1.6 % | 7.10 tok/s |
| `Question: What is 17 times 23?` | 59 | 4 | 6.8 % | 7.33 tok/s |
| speculation off (engine path, x3) | — | — | — | 19.84-19.85 tok/s |

The engine-level `LING3_MTP=1` probe, which lets block 42's cache accumulate across steps
instead of restoring it with the trunk, measures 0-2 matches in 23-30 on the same prompts,
so the low acceptance is the MTP head at HEAD, not the per-step rollback. At K=1 the
verifier costs about one extra trunk pass per accepted token, so the design's ~1.8x
modeled loss is the floor, not the estimate. The older **58 % (18/31)** row does not
reproduce at HEAD and the 1.15x model built on it is void.

Enable with `LING3_SPEC_DRAFTS=1` in the server environment; leave it unset (or 0) for
plain decode. `tools/check_http_restore.py --dump DIR` writes every completion by prompt
hash plus each server log, which is how spec-off/spec-on byte identity is checked.

## 6. Known gaps

| gap | status |
| --- | --- |
| HTTP serving, cache, streaming | **works** — `gufo serve llm` answers, streams, and acts on cancellation (`tools/check_http_restore.py`, which also requires a successful private restore in the log so a cold fallback cannot pass). Continuation restores a prefix from the component cache and resumes on the pass grid a cold prefill uses: the runner declares `prefill_alignment` 32 and the adapter retains the aligned checkpoint below a prompt whose length is not a pass multiple, so the continued suffix runs the same chunk widths as a fresh server. The restored continuation then matches a fresh server byte for byte and `tools/check_http_restore.py` prints all three OK lines on five consecutive runs (2026-11-28). Getting there also needed two chunked-router fixes in `RouterKernel`: the weights were *read* through the batch base pointer (every row after the first was weighted with the chunk's first row's experts) and *written* through it as well (every row's weights went into the first row's slots, so rows 1..N-1 were combined with zero weights and the first slot held whichever block ran last). With both fixed the chunked prefill is bit-reproducible: three identical passes give one logits hash (`LING3_PREFILL_REPEAT=1` in tests/mtp_test.cpp) and the serving suite passed five consecutive runs (see notes/correctness-investigation.md). One private checkpoint is 154,697,732 bytes at context 4096. Session isolation is verified with two concurrent requests on `--sessions 2`: traces show two resident banks on a single shared engine, both matching their own cold runs, with engine operations serialized (an isolation result, not a throughput claim). An earlier build reported a 502 MB capture and a one-session limit; both are superseded |
| MTP speculative decoding | **wired, opt-in behind `LING3_SPEC_DRAFTS=1` (default off); slower than plain decode** — K=1, greedy only. `VerifyDrafts` matches two `ForwardToken`s row for row, and the serving step reproduces plain greedy over 32 tokens with logits delta 0 (`tests/mtp_test.cpp`). Measured at HEAD: 8/182 drafts accepted (4.4 %) over three 64-token greedy requests (3/61, 1/62, 4/59), decode **7.10-7.33 tok/s** against 19.84-19.85 with speculation off (~2.7x slower); the restore suite's shorter 12-token prompts accepted 12/77 (15.6 %, per request 0-42.9 %). Spec-off and spec-on completions are byte-identical through restore, a two-turn continuation and two concurrent banks (`tools/check_http_restore.py --dump`, empty diffs). The older **58 % (18/31)** figure does not reproduce at HEAD: `LING3_MTP=1` in `tests/forward_test.cpp` measures 0-2 of 23-30 on the same prompts, so the 1.15x model is void and the measured cost is at least the modeled ~1.8x. |
| Vision (mmproj) | **implemented and verified** at the engine level (numpy reference 3.2e-06, llama.cpp parity 0.04 %) and through `gufo prompt --image`: on the red-circle/blue-square/"ling" test image the CLI answers "A large red circle is centered on a light background with a smaller blue square inside it" (`--think off`), and names the "ling" text with thinking on. The first CLI run exposed a stride bug — `GenerateWithImages` walked the 2560-wide projected rows by the 576-row count, so the conditioning was lost — now fixed in `src/drivers.cpp` and guarded by a rows/markers/width check |
| Token-for-token equality | **not met**; one-token deviation at a 0.048 margin, documented above |
| Roofline gates | decode **32-34 %** of 49.9 tok/s (16.2-17.0 tok/s at depth 0); prefill **0.37-0.38 %** of 6,277 tok/s (**23.4-23.8 tok/s** at pp2048, 32-token chunks); decode at depth 2048 is 15.72 tok/s (31.5 %) — both gates still far off |
