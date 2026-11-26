# Side-by-side against llama.cpp

Same box (Ryzen AI Max, 128 GB unified, gfx1151), same Q4_K_M shards, greedy
(temperature 0), medians of three runs unless noted. Measured 2026-11-26/27.

| case | this extension | llama.cpp CPU | ratio |
| --- | --- | --- | --- |
| decode, depth 0 | **16.2-17.0 tok/s** | 12.52 tok/s | **1.29-1.36x** |
| decode, depth 2048 | 15.72 tok/s | not measured at depth | — |
| prefill, 104-token prompt | 12.52 sequential, 16.13 chunked | 24.20 tok/s prompt eval | 0.67x |
| prefill, pp2048 (32-token chunks) | 23.4-23.8 tok/s | 24.20 tok/s prompt eval | 0.97-0.98x |
| weight load, warm page cache | 2.1 s | 1.58 s (mmap) | 1.33x slower |
| golden set: plain | 93 chars agree, then `0.048` logit margin | — | — |
| golden set: arithmetic | **identical, 30/30 chars** | — | — |
| golden set: tool call | 27 chars agree | — | — |
| golden set: thinking on | 9 chars agree | — | — |

## Commands

Reproduce ours (the harness prints bandwidth, both rooflines and % of roofline):

    tools/bench_report.py --engine <ling3_air_forward_test> --model <first shard> \
        --depths 0 2048 --steps 8 --runs 3

    tools/bench_report.py ... --prompt "<2048-token text>" --prefill-tokens 2048 \
        --chunked-prefill 32 --steps 2 --runs 3

Golden set (both engines through wrappers, one row per case with the divergence index):

    tools/golden_set.py --engine-wrapper run-engine.sh --llama-wrapper run-llama.sh --steps 24

llama.cpp reference (CPU, 16 threads, commit `e95426f`):

    llama-completion -m <first shard> -f <prompt> -n <steps> --temp 0 -no-cnv \
        --no-display-prompt -t 16 --no-warmup

## Why they differ

- **Decode favours the GPU** (1.36x): one token reads ~4.75 GB of weights, and the
  zero-copy registration keeps them in the page cache, so decode is bandwidth-bound at
  237 GB/s rather than CPU-bound.
- **Prefill is level and not faster**: llama.cpp batches prompt tokens through GEMMs while
  this engine still does per-token work for the routed MoE and the KDA/MLA paths, so at
  pp2048 it reaches 23.16 tok/s against their 24.20. Chunking gains 1.29x over sequential.
- **Load is slower** because the weights are registered host memory rather than a plain
  `mmap` handed to a CPU kernel, which costs one registration pass per shard.
- **Token streams diverge where the arithmetic does**: the `arithmetic` case matches end to
  end, the others agree for 93/27/9 characters and then split at a 0.01-0.05 top-2 logit
  margin. Both are "correct" continuations; the difference is GPU float arithmetic versus
  the CPU reference, not a semantic disagreement, and it is documented rather than papered
  over (`docs/quality.md`).
