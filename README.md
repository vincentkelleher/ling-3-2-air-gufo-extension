# ling32-air-gufo

A Gufo extension for **Ling 3.2 Air** — a 128B-parameter hybrid MoE
(`bailing_moe_v3_vl`) served from the `bartowski/Ling-3.2-air-GGUF` Q4_K_M
quantization on AMD Strix Halo.

Status: **experimental — the text engine runs, serves no HTTP yet, and is not
token-identical to llama.cpp.** Details and numbers below; the honest summary is
at the bottom.

## What works

- **Module build for Gufo 0.36.0** (extension API 0.3.x): `gufo extension
  install ./ling32-air-gufo --yes`, then `gufo diagnose --section extensions`
  lists the provider `ling3-air`.
- **Pinned weights**: `models.lock.json` pins the revision, sizes and SHA-256 of
  the three Q4_K_M shards and the BF16 multimodal projector; `gufo model
  install ling-3.2-air` fetches them and `gufo model verify` checks them.
- **Config and tensor map**: every bailingmoe3 hyperparameter is parsed and
  checked; all 938 tensors (74.87 GiB) are mapped and shape-validated, including
  the per-tensor quantization of this file (Q8_0 attention, Q4_K/Q5_K/Q6_K
  experts, Q4_0 MTP block).
- **Tokenizer and chat framing**: byte-level BPE from the GGUF with this model's
  pre-tokenizer regex and NFC normalization — byte-exact against the reference
  tokenizer on 184 cases (9,807 tokens) — and a chat renderer byte-identical to
  the model's own Jinja template on 18 cases.
- **Text engine on the GPU**: all 43 blocks (35 KDA + 7 MLA + MTP block parsed
  but unused), MoE routing and expert gather, LM head, greedy sampler. Weights
  are registered zero-copy from the reader's mapping, so an 80 GiB model loads in
  ~2 s warm.
- **Usable from the CLI**: `gufo prompt`, `gufo chat` and `gufo bench` run the
  model (chunked prefill 16.8 tok/s, decode 16.9 tok/s on this box). The HTTP
  server still needs the runner below.
- **Coherent output**: `The capital of France is` continues with the same
  sentence as llama.cpp; the two agree for 53 characters before a near-tie
  (0.048 logit margin) sends them apart, and 1 of 4 pinned cases is
  token-identical end to end.
- **MTP block runs and is measured**: block 42 is readable (its Q4_0 storage has a
  dot, validated against real rows), and `Engine::ForwardMtp` drafts from the base
  model's final hidden state: **18 of 31 drafts (58 %)** match what the trunk then
  produces. Speculative decoding is deliberately not wired into the serving path —
  at 58 % acceptance and a batched verifier only 1.29x cheaper per token, the modeled
  gain is ~1.15x, which does not justify the equality risk. See
  [docs/quality.md](docs/quality.md).
- **Image input works at the engine level**: the mmproj tower (27 ViT blocks, vision
  M-RoPE, 2x2 merge, projector) encodes a 768x768 image into 576 rows of 2560, and
  `Engine::ForwardEmbedding` feeds those rows into the trunk at the chat template's
  image markers. Verified two ways: against a numpy reference from the same weights
  (final embedding 3.2e-06) and against llama.cpp's own encoder (embedding sum agrees
  to 0.04 %, block-0 stages to four decimals). On a test image it answers "there's a
  large red circle, and inside it, centered, is a blue square. In the upper-left
  corner, there's small text". The `gufo prompt --image` flag path is written but
  **unverified** (see the quality report).

## Measured performance (2026-11-27)

| case | engine | reference |
| --- | --- | --- |
| decode, short context | **16.2-17.0 tok/s** (median 17.0) | llama.cpp CPU: 12.52 tok/s |
| prefill, 104-token prompt | 12.52 tok/s sequential, **16.13 tok/s chunked (1.29x)** | llama.cpp CPU prompt eval: 24.20 tok/s |
| prefill, pp2048 | **21.4-23.2 tok/s** = 0.34-0.37 % of the 6,277 tok/s roofline | llama.cpp prompt eval: 24.20 tok/s |
| weight load | 2.1 s warm, 35.9 s cold | llama.cpp mmap: 1.58 s |

Machine ceilings measured in-repo (`tools/bandwidth_probe.hip`,
`tools/roofline.py`): **237 GB/s** read bandwidth, one decode step reads
**4,753 MB**, so the roofline is **49.9 tok/s** decode and ~**6,277 tok/s**
prefill at 2,048 tokens. The engine is at **32-34 %** of the decode roofline, ahead of
llama.cpp's CPU path on the same machine (25 %). Reproduce decode with the committed
harness (medians of 3 runs):

    tools/bench_report.py --engine <forward_test> --model <first shard> \
        --depths 0 512 --steps 8 --runs 3

Depth rows: 16.2-17.0 / 16.2-17.0 tok/s at depth 0 / 512. Medians of three runs move by up to
~5 % between invocations on this box (measured: decode 16.18 and 17.02 tok/s, pp2048 21.36 and
23.16 tok/s), so the numbers are quoted as ranges rather than single samples. `IMPROVEMENTS.md` records every
change with before/after, including the ones that measured no gain.

## What does not work yet

- **No HTTP serving**: `gufo prompt/chat/bench` work through the provider's
  drivers, but `gufo serve llm --model ling-3.2-air` still cannot use it: the
  `TextModelRunner` contract (component cache, continuation, streaming) is not
  implemented. `notes/runner-plan.md` is the plan.
- **Prefill is only partly batched**: `PrefillChunk` batches the projections, the
  out projection, the dense FFN and the shared expert over a chunk, which gives
  **16.13 tok/s against 12.52 sequential (1.29x)** on a 104-token prompt. The
  routed MoE and the attention state are still per token; a correct grouped-MoE
  schedule exists but is opt-in because it needs an efficient K-quant GEMM before
  it pays (`notes/prefill-design.md`). Chunked ingestion is also opt-in: it sums the
  same products in a different order, which perturbs greedy output at the 0.01-0.05
  margins measured here, so any token-for-token claim has to name its prefill path. `notes/prefill-design.md` quantifies why: with top-8 of 512 experts a batch of M
tokens gives M/64 tokens per expert, so prefill must be **chunked** (M around
256-512) for the expert weights to amortise; batching only the dense stages would
buy about 1.3x.
- **Not token-for-token with llama.cpp except on simple arithmetic.** Measured
  with `tools/golden_set.py` (24 greedy tokens per case, identical prompts): the
  arithmetic case matches end to end (30/30 characters), the others agree for
  the opening of a sentence (53, 27 and 19 characters) and then split where the
  top-2 logit margin is 0.01-0.02 — a 0.1 % difference deciding the token. Same
  task, same meaning, different wording. The integer per-group dots already match
  llama.cpp's dot products (`tests/dot_vs_gufo_test.cpp`), so the residual is
  elsewhere.
- **MTP drafted but not accepted**: the block runs and its drafts are usually
  right (7 of 9), and a chunked verifier plus state snapshot/restore now exist
  (`VerifyDrafts`, `SnapshotState`, `RestoreState`), but the accept loop's position
  bookkeeping is unfinished — the first attempt accepted nothing, and the correct
  discipline is written down in `notes/prefill-design.md`.
- **Images only through the engine, not yet through the CLI flag**: the tower and the
  prompt substitution are verified, but `gufo prompt --image` did not reach them in
  this environment and is reported as unverified rather than claimed.
- **No sampler beyond greedy**, no reasoning markers, no tool-call dialect.

## Quality

[docs/quality.md](docs/quality.md) is the quality report: what is proven against
Gufo's reference and llama.cpp, the per-stage numerical differences, the golden-set
result (1 of 4 cases token-identical, the rest one token apart at a 0.048 logit
margin), the verified vision tower with its parity numbers, MTP acceptance, and the
known gaps — no HTTP serving, no CLI image flag, MTP not wired.

## Layout

```text
extension.json       manifest: provider, keys, pinned model files
extension.cmake      host library, HIP runner library, tests
src/config.cpp       bailingmoe3 hyperparameters, validated
src/weights.cpp      the 938-tensor map, validated
src/engine.cpp       one-token forward: KDA, MLA, MoE, LM head
src/hip/*.hip        GEMV (F32/Q8_0/Q4_K/Q5_K/Q6_K), conv1d, delta rule,
                     MLA absorb attention, router, gather, norms, RoPE
src/text/*           tokenizer, pre-tokenizer, chat template
tests/               probe, config, tokenizer, chat, dot-vs-gufo, GPU operators
tools/               bandwidth probe, roofline model, reference fixtures
docs/quality.md      quality report: what is proven, what is measured, what is broken
notes/               engine-plan (status + verification log), benchmarks, prefill and
                     runner plans, vision plan, build notes
```

## Building and testing

See [notes/build.md](notes/build.md) for the exact commands (Nix devshell in a container
with the GPU, a Gufo 0.36.0 checkout, the module build, install and `diagnose`).

`ctest -R '^ling3_air\.'` runs nine tests and all pass: probe, config/weights mapping,
tokenizer, chat template, dequantization against Gufo, dot versus Gufo, image
preprocessing, the vision encoder against a numpy reference, and end-to-end image
conditioning. The last two need `LING3_MMPROJ` plus a fixture from `tools/` and skip
themselves otherwise, as do the end-to-end forward and vision tests without `LING3_GGUF`.
The GPU operator suite (`tests/operator_gpu_test.cpp`, 31 checks) runs outside ctest.

Measurements come from committed scripts rather than pasted numbers:
`tools/bench_report.py` (bandwidth, both rooflines, % of roofline, medians of N),
`tools/golden_set.py` (the llama.cpp comparison) and `tools/roofline.py`. Two records sit
beside this file: [docs/conformance.md](docs/conformance.md) maps Gufo's conformance
checklist to evidence, and [docs/parity.md](docs/parity.md) is the side-by-side against
llama.cpp with the commands behind every row.

## Honest summary

The extension loads Ling 3.2 Air and generates coherent text on the GPU, with
every operator checked against a reference and every stage of the forward checked
against numpy. It is **not yet a serving extension**: no runner, no cache, no
HTTP, no CLI image flag, MTP not wired — and it is slower than llama.cpp's CPU path on the same
machine, at 14 % of the memory-bandwidth roofline. Those are the next pieces of
work, in that order.

## Licenses

Code: MIT ([LICENSE](LICENSE)). Weights: the publisher's terms, see
[THIRD_PARTY.md](THIRD_PARTY.md). No weights are distributed here.
