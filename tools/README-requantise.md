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
