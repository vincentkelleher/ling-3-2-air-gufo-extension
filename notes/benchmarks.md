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
| decode, short context | **7.08 tok/s** | 14.2 % | llama.cpp CPU `llama-completion`: 12.52 tok/s |
| prefill, 25-token prompt | 7.1 tok/s (token by token) | 0.1 % | llama.cpp CPU prompt eval: 24.20 tok/s |
| weight load | 2.1 s warm / 35.9 s cold | - | llama.cpp mmap load: 1.58 s |

The engine has no batched prefill yet: it feeds prompt tokens one at a time, so
its "prefill" is just decode speed. That is the single biggest gap and the first
item for `prefill-opt` (a batched forward would also lift decode indirectly by
letting experts be shared across tokens).

Decode at 14.2 % of the roofline is far from the 60 % gate in the goal. With
7.08 tok/s the engine is also slower than llama.cpp's CPU path on this box
(12.52 tok/s), so *every* optimization below is still ahead:

1. one warp per row spends most of its time on dequantization; a row-parallel
   dot with vectorised loads (and Q8 activations held in registers) is the
   obvious next step;
2. the MoE gather issues 8 x 3 GEMVs per block with a host round-trip for the
   selected ids (a device-side gather removes 40 stream syncs per token);
3. the LM head is a 157k-row GEMV per token (9 % of the bytes) and runs at the
   same warp-per-row rate;
4. the router reads 5 MB of F32 weights per block (4.4 % of bytes) - keep it in
   bf16.
