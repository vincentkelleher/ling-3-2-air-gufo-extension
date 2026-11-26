# Measured performance (2026-11-27)

Method: `gufo bench`-style decode measured by `tests/forward_test.cpp`
(prompt prefilled token by token, then greedy decode), medians of repeated runs,
machine otherwise idle, all weights zero-copy in the page cache.

## Machine ceiling

`tools/bandwidth_probe.hip` (8 GiB, medians of 3): read **237.0 GB/s**,
copy 204.3 GB/s, write 198.4 GB/s.

`tools/roofline.py` (GGUF tensor types + 8-of-512 routing): one decode step reads
**4,753 MB** -> roofline **49.9 tok/s** (attention projections 54.7 %, routed
experts 26.6 %, LM head 9.0 %, shared expert 5.3 %, router 4.4 %).
Prefill roofline at 2,048 tokens (every expert touched once): ~**6,277 tok/s**.

## This engine vs the references

| case | measured | % of roofline | reference |
| --- | --- | --- | --- |
| decode, short context | **12.48 tok/s** | 25.0 % | llama.cpp CPU `llama-completion`: 12.52 tok/s |
| prefill, 25-token prompt | 7.1 tok/s (token by token) | 0.1 % | llama.cpp CPU prompt eval: 24.20 tok/s |
| weight load | 2.1 s warm / 35.9 s cold | - | llama.cpp mmap load: 1.58 s |

Depth rows (`LING3_DEPTH`, context 4096, 8 decode steps after prefilling the
cache with a repeated token; `tools/roofline.py --depth` accounts for the MLA
latent cache):

| depth | decode | vs depth 0 |
| --- | --- | --- |
| 0 | 12.54 tok/s | - |
| 512 | 12.41 tok/s | -1.0 % |
| 2048 | 12.04 tok/s | -4.0 % |

Decode is almost depth-insensitive: the MLA cache is 576 floats/token on seven
blocks (16.1 KiB/token), far smaller than the 4.75 GiB of weights read per step.

The engine has no batched prefill yet: it feeds prompt tokens one at a time, so
its "prefill" is just decode speed. That is the single biggest gap and the first
item for `prefill-opt` (a batched forward would also lift decode indirectly by
letting experts be shared across tokens).

Optimization log (each step measured, output unchanged):

| change | decode |
| --- | --- |
| baseline engine | 6.52 tok/s |
| parallel argmax + shared-memory router flags | 6.91 |
| reciprocal multiply + `__float2int_rn` for activations | 9.14 |
| router top-k in one insertion pass | 10.05 |
| segment-major Q6_K unpacking (validated on CPU first) | **12.48** |

Decode is now level with llama.cpp's CPU path (12.52 tok/s) but still 25 % of the
49.9 tok/s roofline, so the 60 % gate is far. Two attempts that did *not* help are
recorded in `notes/engine-plan.md` (block-per-row GEMV, hoisting activation
quantization) — the kernel is memory-bound, so the next gains need fewer bytes or
better access patterns, not less arithmetic:

1. one warp per row spends most of its time on dequantization; a row-parallel
   dot with vectorised loads (and Q8 activations held in registers) is the
   obvious next step;
2. the MoE gather issues 8 x 3 GEMVs per block with a host round-trip for the
   selected ids (a device-side gather removes 40 stream syncs per token);
3. the LM head is a 157k-row GEMV per token (9 % of the bytes) and runs at the
   same warp-per-row rate;
4. the router reads 5 MB of F32 weights per block (4.4 % of bytes) - keep it in
   bf16.
