# Re-quantising tensors into a derived artifact

A decode token reads 4.753 GB and this box sustains 237 GB/s, so the memory ceiling is 49.9 tok/s.
Reading fewer bytes per token is the only way past it, and the Q8_0 tensors (LM head, KDA and MLA
projections) cost 8.5 bits per weight.

`tools/requantise.cpp` re-quantises one tensor using ggml's own quantiser, so sub-block scales and mins
are exactly the library's, and reports the reconstruction error before anything loads the result:

    c++ -O2 tools/requantise.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
        -L$LIB -Wl,-rpath,$LIB -lggml-base -o requantise
    ./requantise MODEL.gguf out.bin blk.0.attn_q.weight q6_K

Measured on `blk.0.attn_q.weight` (Q8_0 source, 11.1 MB):

| target | smaller by | relative error |
| --- | --- | --- |
| Q4_K | 47.1 % | 7.2 % |
| Q5_K | 35.3 % | 3.6 % |
| Q6_K | 22.8 % | 1.7 % |

So Q6_K is the quality-preserving choice: about a fifth fewer bytes per Q8_0 tensor for 1.7 % weight
error. `tools/requantise_q4_0.py` is a dependency-free alternative for Q4_0 specifically (4.5 bits, but
9.5 % error on the same tensor, which is why the K-quants are preferred).

What is *not* done yet, and is the other half of the work: loading a derived tensor. The engine resolves
weight pointers as offsets into the reader's mapped regions, so a derived buffer has to be registered
like a region and resolved by `Impl::Dev` — a change to the loader, not to this tool.

## Loading attempt: memory fault, reverted

A first loader attempt (2026-11-27) mapped the derived buffers as host regions and had the weight
mapper swap them in, declaring Q6_K. Without `LING3_DERIVED` the engine measured 19.63-19.64 tok/s as
usual; with it, the first step raised `HSA_STATUS_ERROR_MEMORY_FAULT`. Two likely causes, both
recorded before another attempt:

1. `hipHostRegister` on a `malloc`'d buffer without page alignment — the shard path aligns the region
   start down to a page for exactly this reason, and the derived path does not;
2. consumers that assume the pinned tensor's size or row stride (the token embedding's row width is
   computed from `bytes / ne[1]`, for instance), which a smaller tensor invalidates.

So the size win is real — Q8_0 is 70.7 % of a token's traffic and Q6_K is 22.8 % smaller on those
tensors, which would take 4.753 GB/token to ~3.98 GB and the memory ceiling from 49.9 to ~59.6 tok/s —
but the loader has to be done properly: page-align the derived buffers, and audit every consumer of a
substituted tensor's bytes before enabling it.

## Measured: fewer bytes makes decode slower here (2026-11-27)

With the loader wired (token embedding excluded), a Q6_K substitution of every Q8_0 projection and the
LM head was measured end to end:

| build | decode, 32 steps |
| --- | --- |
| pinned weights (Q8_0 + Q4_K/Q6_K) | **19.63 tok/s** |
| Q8_0 tensors substituted with Q6_K (-22.8 % bytes on those, -16 % on a token) | **4.93 tok/s** |

Four times slower with 16 % fewer bytes. That is consistent with the rest of this session's evidence:
at 39 % of the 49.9 tok/s roofline the engine is **not bandwidth-bound**, so removing bytes buys
nothing, while Q6_K's 6-bit unpacking and 16 sub-block scales per 256 values cost real compute in a
path that is already latency-exposed. Bytes/token is therefore *not* a lever until the quantised dots
are efficient; the 60 tok/s target cannot be reached by re-quantising, only by making the kernels
faster (or by a different machine).

The loader changes were reverted; the tooling stays, and this is the measurement that closes the
`lane-decode-bytes` question.
