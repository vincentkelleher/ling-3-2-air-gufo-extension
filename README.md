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
- **Coherent output**: `The capital of France is` continues with the same
  sentence as llama.cpp; the two agree for 53 characters before a near-tie
  (0.048 logit margin) sends them apart, and 1 of 4 pinned cases is
  token-identical end to end.
- **MTP block runs**: block 42 is readable (its Q4_0 storage now has a dot,
  validated against real rows), and `Engine::ForwardMtp` drafts the next token
  from the base model's final hidden state — 7 of 9 drafts match what the trunk
  then produces on the pinned prompt. Drafting does not yet speed anything up:
  verification is one token per pass, so see the note below.

## Measured performance (2026-11-27)

| case | engine | reference |
| --- | --- | --- |
| decode, short context | **12.49 tok/s** | llama.cpp CPU: 12.52 tok/s |
| prefill, 104-token prompt | 12.52 tok/s sequential, **16.13 tok/s chunked (1.29x)** | llama.cpp CPU prompt eval: 24.20 tok/s |
| weight load | 2.1 s warm, 35.9 s cold | llama.cpp mmap: 1.58 s |

Machine ceilings measured in-repo (`tools/bandwidth_probe.hip`,
`tools/roofline.py`): **237 GB/s** read bandwidth, one decode step reads
**4,753 MB**, so the roofline is **49.9 tok/s** decode and ~**6,277 tok/s**
prefill at 2,048 tokens. The engine is at **25 %** of the decode roofline, level with llama.cpp's CPU
path on the same machine. Depth rows: 12.54 / 12.41 / 12.04 tok/s at depth
0 / 512 / 2048.

## What does not work yet

- **No HTTP serving**: the runner does not implement the `TextModelRunner`
  contract yet, so `gufo serve llm --model ling-3.2-air` cannot use it. The
  component cache, continuation and streaming are therefore absent.
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
- **No image input**: the projector is downloaded but no vision path exists.
- **No sampler beyond greedy**, no reasoning markers, no tool-call dialect.

## Quality

[docs/quality.md](docs/quality.md) is the quality report: what is proven against
Gufo's reference and llama.cpp, the per-stage numerical differences, the golden-set
result (1 of 4 cases token-identical, the rest one token apart at a 0.048 logit
margin) and the known gaps — no HTTP serving, no vision, MTP not accepted.

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
notes/               engine plan, verification log, benchmarks
```

## Building and testing

See [notes/build.md](notes/build.md) for the exact commands (Nix devshell in a
container with the GPU, a Gufo 0.36.0 checkout, `cmake --preset` equivalent,
module install, and ctest with `LING3_GGUF` set).

`ctest -R '^ling3_air\.'` runs: probe, config/weights, tokenizer, chat template
and the dequantization comparison against Gufo (5 tests, all passing). The GPU
operator suite (`tests/operator_gpu_test.cpp`, 23 checks) and the end-to-end
forward (`tests/forward_test.cpp`) run outside ctest because they need the GPU.

## Honest summary

The extension loads Ling 3.2 Air and generates coherent text on the GPU, with
every operator checked against a reference and every stage of the forward checked
against numpy. It is **not yet a serving extension**: no runner, no cache, no
HTTP, no MTP, no vision — and it is slower than llama.cpp's CPU path on the same
machine, at 14 % of the memory-bandwidth roofline. Those are the next pieces of
work, in that order.

## Licenses

Code: MIT ([LICENSE](LICENSE)). Weights: the publisher's terms, see
[THIRD_PARTY.md](THIRD_PARTY.md). No weights are distributed here.
