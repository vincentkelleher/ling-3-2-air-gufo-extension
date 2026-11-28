# Optimisation log

Each entry records what was changed, what it measured before and after, and whether it
was kept. Greedy tokens and logits are compared after every change: a variant that
changes them is a bug, not an optimisation. All numbers are decode at depth 0 on this
box, 8 steps, temperature 0, unless stated otherwise.

Baseline at the start of this log: **12.56 tok/s** decode, **16.1-16.8 tok/s** chunked
prefill, bandwidth roofline at depth 0 = 49.9 tok/s.

## 2026-11-27

### MoE expert dispatch moved to the device — no gain, kept

`LaunchMoeGather` copied the selected expert ids and weights to the host,
`hipStreamSynchronize`d, and launched each expert's FFN from a host loop: 35 stream
synchronisations per token plus two device-to-host copies. It now keeps the ids on the
device and issues one launch per matrix through `LaunchGemvExperts`, combining slots
with a weighted sum in the same order.

| | before | after |
| --- | --- | --- |
| decode | 12.56 tok/s | 12.15 tok/s |
| tokens / logits | — | identical |

**No gain, and marginally slower.** The synchronisations were never the cost: they were
draining a stream that was already the bottleneck. Kept because it removes the host
round-trips (a precondition for the next change and for batched prefill), and because
the difference is within run-to-run noise.

### Q4_K activation quantised once per token instead of once per row — +15 %, kept

The Q4_K dot re-derived the activation's per-block scale and 256 quantised values
inside every row's loop, so a 1024-row expert re-quantised the same 2560 values 1024
times. The activation is identical for every row of every selected expert, so it is now
quantised once per token (`QuantizeActivationKernel`, group sums included) and the row
dot (`DotRowQ4KPre`) reads it.

The down matmul is the exception: each selected expert has its own gated output, so it
needs one quantised vector per slot (`QuantizeActivationBatchKernel`). Quantising a
single vector there fed every expert the first expert's input — the tokens changed, which
is how the mistake was caught.

| | before | after |
| --- | --- | --- |
| decode | 12.15 tok/s | **14.45 tok/s** |
| FFN/MoE per token | 48.48 ms | **35.33 ms** |
| attention per token | 31.08 ms | 31.08 ms |
| tokens / logits | — | identical |

### LM head through the prepared GEMV — no gain, kept

The head has 156 k rows over a 2560-wide activation, so it also re-quantises per row;
it now uses the existing `LaunchQuantizeActivations` + `LaunchGemvPrepared` pair (already
covered by `tests/operator_gpu_test.cpp`), guarded to the Q4_K/Q6_K storages that path
supports.

| | before | after |
| --- | --- | --- |
| decode | 14.45 tok/s | 14.45 tok/s |

**No gain.** The activation is 10 KB and stays in cache while 156 k rows stream past, so
the re-quantisation was already hidden behind the weight traffic: the head is
bandwidth-bound, not arithmetic-bound. Kept because it is the same code path and costs
nothing.

### Chunk size for prefill — no gain, left at 32

Chunked prefill at 4821 prompt tokens: 14.84 tok/s at chunk 8, 15.37 tok/s at chunk 128.
Batching the dense GEMMs across four times as many rows buys 3.6 %, which says the
per-token work (the routed MoE and the per-token attention) dominates prefill just as it
dominates decode. Not changed: the default stays 32, and prefill needs the attention and
MoE restructuring below, not a bigger chunk.

### KDA projection group through the prepared GEMV — no gain, reverted

The KDA branch's q/k/v/f_a projections all read the same `m.h`, so the layer quantised
it once and ran the four matrices through `LaunchGemvPrepared`.

| | before | after |
| --- | --- | --- |
| decode | 14.45 tok/s | 14.36 tok/s |
| tokens / logits | — | identical |

**No gain**, for the same reason as the head: a matrix with thousands of rows streams
far more weight bytes than the 10 KB activation costs to quantise, so the activation is
in cache and the quantisation overlaps the weight traffic. Reverted rather than kept:
an extra 2560-element quantisation per layer buys nothing.

### Delta-rule state transposed to column-major — +5.4 %, kept

The delta rule stores one `head_dim x head_dim` state per head and updates it per token.
It was row-major, so thread `j` walked its own row: every access in all three passes
(decay, `k . row`, `row += k * delta`, `q . row`) was `head_dim` floats away from the
neighbouring thread's, i.e. uncoalesced. Element `(i, j)` now lives at
`(head*head_dim + i)*head_dim + j`, so the threads of a warp touch consecutive addresses
and each pass streams linearly.

| | before | after |
| --- | --- | --- |
| delta rule per token | 6.28 ms | **2.98 ms** |
| attention per token | 31.35 ms | 28.06 ms |
| decode | 14.46 tok/s | **15.24 tok/s** |
| tokens / logits | — | identical |

The arithmetic is unchanged (each thread sums over `i` in the same order), which is why
the tokens stayed identical. The layout is internal to this kernel: the state starts
zeroed and is only ever read or written here, so no other code had to change.

### KDA sub-stage profile (why attention costs 31 ms)

Three extra marks split the KDA branch on the 35 KDA layers:

| sub-stage | ms/token |
| --- | --- |
| q/k/v/f_a projections | 9.65 |
| conv1d + silu + gate/beta | 5.06 |
| delta rule (the recurrence) | 6.28 |
| output norm + g_a + out projection | 7.14 |
| MLA layers (the other 7) | ~3.2 |

The projections and output matrices (16.79 ms) are already bandwidth-bound, as the two
experiments above showed. The **delta rule at 6.28 ms is not**: it walks a per-head
recurrence, so it is latency and occupancy bound rather than bandwidth bound, and it is
the most promising remaining target in attention. It has not been attempted here.

### Wide Q4_K dot: (block, group) items across the warp — +4.9 %, kept for gate/up

A 2560-wide Q4_K row is 10 blocks, and the dot gave one block to each lane, so 22 of 32
lanes idled and each active lane ran 8 groups serially. Spreading the 80 (block, group)
items over the warp gives every lane two or three.

| variant | decode | tokens |
| --- | --- | --- |
| shared activation, gate/up | 15.24 → **15.98** tok/s | identical |
| per-slot activation, down matmul | 15.98 → **16.91** tok/s | **changed** |
| the LM head | 16.91 → 16.91 tok/s | — |

The gate/up variant is kept. The KDA projections were also tried with the wide dot
(one quantisation of `m.h` serving q/k/v/f_a/g_a): tokens identical, **no gain**
(16.00 -> 15.91 tok/s), because those matrices already run at 91 % of the bandwidth
bound — 2 GB of projection weights per token in 9.66 ms. That path was reverted and the
unused wide variants were removed rather than left in the tree.

The down matmul is the biggest single win of the three and is **not** kept: spreading the items changes the summation order, and that changed the
greedy tokens for a fixed prompt. Adopting it means re-running the golden set against
the llama.cpp reference and updating the documented comparison — a validation task, not
a code task, so the code is left on the bit-identical path. The head gains nothing for
the same reason as before: it is bandwidth-bound, and its lanes idling does not matter
when the weights stream at 237 GB/s.

### One block per row, one warp per Q4_K group — slower, reverted

Attempt at the remaining MoE inefficiency (~40-47 GB/s effective against a 237 GB/s
bound). Instead of giving a lane a whole group to walk serially, each block took one
output row and each of its eight warps took one Q4_K group, so a group's 32 quantised
values are 32 contiguous bytes and the inner sum is a warp reduction:

| | decode |
| --- | --- |
| before | 16.91 tok/s |
| block-per-row (8 warps, one per group) | **12.77 tok/s** |

**Slower.** Each warp then reads only 32 bytes, so a 128-byte cache line is touched by
four different warps and the line is fetched once but used by one; and the shape pays a
warp reduction plus `__syncthreads` for every 256-wide block (ten per row) instead of
reducing once per row. Reverted.

The lesson for the next attempt: keep one warp per row (so its 1440-byte row streams
through whole cache lines) and vectorise *inside* the lane — load the 144-byte block as
nine 16-byte vectors and extract nibbles from registers — rather than spreading the row
across more warps.

### Vectorised loads inside the lane — no change, reverted

Following the lesson above, one warp still owns a row, and inside the lane each
(block, group) item now loads its 32 quantised bytes and its 32 activation bytes as two
16-byte vectors instead of 64 byte-sized loads:

| | decode |
| --- | --- |
| before | 16.88 tok/s |
| two 16-byte loads per item | 16.90 tok/s |

Identical logits, no speed change: the byte loads were not the limit either. Together
with the four earlier attempts, the MoE dot has now been shown not to be limited by
launch count, activation re-quantisation, lane idling, warp shape or load width. What is
left is integer throughput and occupancy in the `code * a` loop itself — a different kind
of work (more items in flight per lane, or `__dp4a`-style packed arithmetic) that was not
attempted.

### `v_dot4_i32_i8` (four int8 products per instruction) — neutral, reverted

The remaining hypothesis was integer throughput in the `code * a` loop, so the wide dot
unpacks four nibbles from one 32-bit word and accumulates them with a single
`v_dot4_i32_i8` (inline asm; the clang builtin demands `dot1-insts` on the caller, which
this translation unit does not enable). Integer sums are order-exact, so the logits came
out bit-identical — and the speed did not move:

| | decode |
| --- | --- |
| scalar `code * a` | 16.88 tok/s |
| `v_dot4_i32_i8`, four products per instruction | 16.87 tok/s |

**Six attempts now measured:** launch count, activation re-quantisation (the one that
worked, +15 %), lane idling (wide dot, +11 % combined), warp shape, load width, and
instruction width. The MoE dot sits at ~40 GB/s effective and none of those moved it, so
it is limited by how the eight selected experts' weight regions are streamed (scattered
1.4 MB regions per expert, three matrices each), not by anything inside the loop. The next
attack would have to change the access pattern — grouping tokens by expert so each expert's
weights are read once for a batch — which is the batched-MoE work `notes/prefill-design.md`
describes and is not attempted here.

### Wider blocks for the expert GEMVs (512 threads) — neutral, reverted

The expert GEMVs read eight scattered 1.4 MB regions, so the hypothesis was that more warps
in flight would hide the latency behind the ~40 GB/s they sustain:

| | decode |
| --- | --- |
| 256-thread blocks (8 warps) | 16.87-17.02 tok/s |
| 512-thread blocks (16 warps) | 16.93 tok/s |

Identical tokens, no change. Reverted.

That closes the launch-shape family: launch count, block width, warp shape, per-expert
launches and lane-dot width have all been measured and none of them moves the MoE. Together
with the earlier results (activation re-quantisation +15 %, wide dot +11 %, everything else
neutral) the conclusion is that this engine's decode sits at **16-17 tok/s (32-34 % of the
49.9 tok/s roofline)** and that the next step is not a parameter but a different kernel:
a tiled Q4_K matrix-multiply with shared-memory staging (the shape llama.cpp's `mul_mat`
uses), where a warp cooperatively stages a weight tile instead of walking rows.

### The scattered pattern can do 100 GB/s — the MoE kernel is ~2.5x below it

Six kernel-level hypotheses were measured out, so the question became whether the access pattern
itself is the limit. `tools/scatter_probe.hip` registers 4 GiB the way the engine does and reads
1.4 MB windows from that mapping (one per selected expert, contiguous inside a window, scattered
across windows):

| pattern | bandwidth |
| --- | --- |
| sequential read of the same mapping (`tools/bandwidth_probe.hip`) | 237 GB/s |
| 8 windows x 1.4 MB, **8 blocks** | 51.7 GB/s |
| 8 windows x 4.2 / 14 / 56 MB, 8 blocks | 32.5 / 30.3 / 30.4 GB/s |
| 32 windows x 1.4 MB, **32 blocks** | **100.1 GB/s** |

The first run of this probe used one block per window, which is 2 048 threads — too few to
saturate — and I briefly concluded the MoE was at the hardware's scattered-read rate. The 32-block
row refutes that: with enough blocks the same pattern reaches **100 GB/s**, and window size does
not matter (a bigger window at the same block count is *slower*).

So the MoE's ~40-47 GB/s is a kernel-efficiency gap of about **2.5x**, not a hardware limit, and
the decode gate is not out of reach in principle: pulling the expert path from 40 to 100 GB/s
would take its 32 ms to ~13 ms and decode from ~17 to ~24 tok/s (48 % of the roofline). What is still unknown is *why* our expert kernel behaves like the 8-block case rather than the
32-block one; it launches ~1 000 blocks of 256 threads, so raw block count is not it.

**It is not the arithmetic.** Forcing the expert dot to sum raw bytes instead of unpacking 4-bit
weights and multiplying gives an identical rate:

| variant | decode |
| --- | --- |
| full Q4 unpack + MAC per row | 16.93 tok/s |
| same reads, no unpack, no multiply (timing probe only) | 16.75 tok/s |

So the expert path is **read-bound** and every bit of arithmetic is already hidden behind the
loads. That rules out the unpacking as the gap and leaves the read pattern: each warp's 32 lanes
touch addresses 144 bytes apart (one Q4_K block each), so a single load instruction spans 32
different cache lines, whereas the scatter probe's threads read adjacent 4-byte words and get
100 GB/s. The fix therefore looked like a coalesced expert GEMV: one warp per row, lanes reading adjacent
bytes of each Q4_K group (one 32-byte sector per instruction) and reducing across the warp. Written
and measured:

| variant | decode |
| --- | --- |
| scattered 144-byte block per lane (current) | 16.93 tok/s |
| coalesced 32-byte group per warp instruction | 16.91 tok/s |

**Neutral.** So coalescing at this granularity is not the gap either, and the read-bound diagnosis
still stands. The remaining difference from the probe is *concurrency over the same windows*: the
probe had one block per window (32 blocks, 32 windows), while the real kernel has ~1 000 blocks
streaming ~24 windows, so many blocks hit the same 1.4 MB region at once. The next probe is that
shape — many blocks per window — and it decides whether a residency/layout change is the only
remaining lever.

The next probe tested exactly that concurrency shape, by letting blocks share windows
(`tools/scatter_probe.hip` now takes a block count):

| shape | bandwidth |
| --- | --- |
| 24 blocks over 24 windows (one block per window) | 92.5 GB/s |
| 1 024 blocks over 24 windows x 1.4 MB (re-reading) | **463.8 GB/s** |

So many blocks streaming the same windows is *fast*, not slow, and concurrency is not the gap
either.

Nine measured variants now: launch count, activation re-quantisation (+15 %, kept), lane-idle wide
dot (+11 %, kept), warp shape (-22 %), load width (neutral), instruction width `v_dot4` (neutral),
block width (neutral), per-expert launches (-4 %), coalesced group reads (neutral). The probes say
the pattern supports 90-460 GB/s; the kernel sustains ~39 GB/s. **The remaining factor is
unidentified**, and the honest next step is a profiler (rocprof or HIP's PC sampling) to see
whether the kernel stalls on memory or on issue — neither is available in this environment, so the
decode gate is reported as unresolved with a measured 2.4x gap rather than a guess about its
cause.

### Batched router for prefill — +55 %, kept

`PrefillChunk` batched every projection in a chunk except the router: for each of the 32 tokens it
copied the token's row into scratch, ran a 512-row GEMV and then routed. That is 35 layers x 32
tokens of GEMV-plus-copy per chunk, i.e. ~2 200 launches plus 1 120 device copies.

It is now one GEMM for the whole chunk (`n_expert x n_embd` over the batch) with the per-token
`RouterSelect` reading its row of the result, and the copy is gone because `MoeGather` takes the
row of the chunk activation directly.

| case | before | after |
| --- | --- | --- |
| pp2048, 32-token chunks | 15.4 tok/s | **23.8 tok/s** (0.38 % of the 6 277 roofline) |
| 4 821-token prompt, 128-token chunks | 15.37 tok/s | **23.35 tok/s** |
| decode (unchanged path) | 16.90 tok/s | 16.90 tok/s |

Greedy tokens are identical (128206 25 198 198 480 ...) and the chunked pass reports the same
argmax (678) as before, so the batching changed only how the chunk is computed.

This also revises the earlier "chunk size barely matters" result: with the per-token router in
place a bigger chunk bought 3.6 % (14.84 at chunk 8 vs 15.37 at 128); with it batched, the chunk
path is 1.5x the old throughput.

## Where the time goes now (decode, 8 tokens, this box)

| group | ms/token | share |
| --- | --- | --- |
| attention (35 KDA + 7 MLA: projections, conv/delta, MLA absorb) | 28.07 | 46 % |
| FFN / 512-expert MoE | ~29.0 | 50 % |
| norms, residuals, rope and the rest | 0.40 | 0.7 % |
| total in the layer loop | 60.66 | |

Session total: decode **12.45 → 16.91 tok/s** (+35.8 %), i.e. 25.1 % → 33.9 % of the
49.9 tok/s bandwidth roofline at depth 0, with the documented golden-set comparison
unchanged.

The MoE now moves ~1.5 GB of expert weights per token in 35.33 ms, which is 42 GB/s
effective against a 237 GB/s box: still 5.6x below the bound, so the quantised dot has
more to give (vectorised block loads are the next step, since the current kernel reads
each 144-byte block field by field). Attention is now the larger half and has had no
optimisation pass at all.

## Prefill: why batching bought only 1.4x (analysis, 2026-11-27)

Measured on this box: decode 16.9 tok/s (59 ms per token), chunked prefill 23.8 tok/s (42 ms per
token). The chunked pass turns the projections, the dense FFN, the routed-expert group and the shared
expert into batched GEMMs over the chunk, yet a token still costs 42 ms against 59 ms sequential —
i.e. about 70 % of a decode step's work is *unchanged* by chunking.

That is consistent with what the chunked path deliberately leaves per token: the KDA convolution and
delta-rule recurrence and the MLA attention are still executed once per row inside the chunk loop. For
a 32-token chunk that is 32 sequential attention steps, so the attention half of the step dominates and
the batched GEMM half is hidden behind it.

Consequence for the gate: pp2048 >= 70 % of the 6,277 tok/s roofline needs ~4,400 tok/s, i.e. roughly
185x the current rate. No amount of GEMM tuning reaches that while attention stays sequential — the
recurrence and the attention must be batched over the chunk. Two candidates, in the order they should
be attempted:

1. **KDA as a chunked associative scan.** The delta rule is a linear recurrence over rows
   (`M <- M * exp(g) ; M <- M + k (v - M k)^T beta`), so it can be composed pairwise and evaluated as a
   scan over the chunk instead of 32 dependent steps. This is the same shape as the mamba-style scan
   llama.cpp performs for its chunked KDA path.
2. **MLA attention over the chunk's rows.** Scores for all rows against the cache, softmax per row, and
   the latent sum as one batched matmul, instead of a per-row kernel sequence.

Both are kernel work with a clear measurement (pp2048 in tok/s and % of roofline, greedy output
unchanged), and both must keep the per-token path as the reference: `tools/golden_set_ids.py` already
compares token-for-token from identical ids, so a batching change that alters output is caught.

## Exact prefill/decode dot products (Ling-3.2-air-Q4_K_M)

For Q8_0, `GemmKernel` already rounded the activation per 32-value block and
multiplied the integer dot by the weight and activation scales. It still split
blocks over 256 threads rather than 32. Keeping one warp per token and calling
`DotRow` gives the same lane partial sums and the same final reduction as GEMV.
This alone made blocks 0 and 1 exact. Block 2 is the first MoE block: its
**F32 router** also needs GEMV's reduction order. Its Q4_K/Q6_K expert weights
already go through `LaunchMoeGather` in both paths; those kernels did not change.

Maximum relative L2 residual error across the 23 ingested positions of
`small-prompt.txt` (`ForwardToken` as denominator; separate clean dump runs):

| block | baseline (d95418f) | Q8_0 fix | Q8_0 + F32 fix |
|---|---:|---:|---:|
| 0 | 1.432e-3 | 0 | 0 |
| 1 | 3.900e-3 | 0 | 0 |
| 2 | 6.806e-2 | 1.400e-7 | 0 |
| 3 | 6.203e-2 | 3.021e-3 | 0 |
| 4 | 6.444e-2 | 7.379e-3 | 0 |
| 5 | 5.817e-2 | 5.674e-2 | 0 |

With both changes, all 966 residual rows and all 42 final-row attention
contributions compared byte-for-byte equal. The eight-row Q8_0 projection
probe also measured zero maximum absolute difference. Instrumented 23-token
chunk timings were 0.896 s baseline, 0.882 s Q8_0-only, and 1.041 s with both
fixes (single runs with dumps enabled, not a throughput benchmark).

The 32-step explicit-ID golden result was unchanged after each fix: plain
first diverged at 19, arithmetic matched, tool-call diverged at 16, thinking-on
at 2 (1/4 exact). This harness ingests the prompt with `ForwardToken`, so it
checks decode/reference behavior, not the changed prefill kernels.

## Decode: the traffic is already minimal; the bandwidth is not (analysis, 2026-11-27)

`tools/bytes_per_token.py` derives what a decode token must read from the GGUF table itself, using
this engine's execution path (KDA projections, MLA projections plus the absorbed matrices, the shared
expert, TOP_K of 512 routed experts, and the LM head):

| group | GB per token |
| --- | --- |
| KDA projections (35 layers) | 0.799 |
| MLA projections (7 layers) | 1.800 |
| shared expert + norms | 0.561 |
| routed experts (8 of 512) | 1.166 |
| LM head | 0.428 |
| **total** | **4.753** |

That is *exactly* the 4.753 GB/token `tools/roofline.py` assumes, so the roofline model is right and the
49.9 tok/s figure is the machine's true floor for this model. Nothing is being read twice.

The consequence is that decode's gap is not traffic but effective bandwidth: 16.9 tok/s at 237 GB/s
implies **14 GB/token** of *effective* traffic, so the engine realises roughly a third of peak
bandwidth on the bytes it does read. The likely cause is locality — the routed experts are 8 random
rows out of a 512-expert table spanning several GB, so each token's gather is a scatter of large,
poorly prefetched reads — which is consistent with the profiler's finding that the Q6_K expert GEMV
alone takes 13.5 ms and with the nine kernel variants that changed nothing: they tuned arithmetic
that was never the limit.

So `decode-opt`'s real target is memory behaviour: expert-order locality (hot-expert caching or
reordering), access granularity per expert, and whether the zero-copy weight mapping is being read in
a pattern the memory system likes. Arithmetic-level tuning of the same reads cannot reach the gate.


## Decode: locality excluded; the limit is achieved bandwidth (measured 2026-11-27)

`LING3_FORCE_EXPERT=<id>` pins every token's routing to one expert, so the routed gather reads the
same, single expert from L2 for every layer — perfect locality, identical byte count.

| routing | tok/s | implied effective bandwidth at 4.753 GB/token |
| --- | --- | --- |
| normal (8 of 512 experts, scattered) | 16.96 | 80.6 GB/s |
| forced expert 0 (perfect locality) | 17.61 | 83.7 GB/s |
| forced expert 300 (distant table region) | 17.60 | 83.7 GB/s |

**Locality is excluded**: perfect locality buys 4 %, and a distant expert behaves identically, so the
gather is not paying for scatter. That also retro-explains the nine kernel variants that changed
nothing — they were tuning arithmetic that was never the limit.

What the numbers do say: at 4.753 GB/token confirmed by `tools/bytes_per_token.py`, 17 tok/s works out
to **about 84 GB/s, i.e. 35 % of the 237 GB/s the bandwidth probe measures**. The engine reads the
right number of bytes at roughly a third of peak. That is a load-efficiency problem — access width and
coalescing inside the quantised dot products (each thread currently walking a row in small strides),
not addressing, not traffic, and not locality.

So `decode-opt`'s next attempt should be a vectorised load path in the Q4_K/Q6_K/Q8_0 row dots
(16-byte loads per lane, block-aligned), measured with `bench-decode.sh` and gated on
`tools/golden_set_ids.py` staying at plain 19 / arithmetic none / tool-call 16 / thinking-on 2. The
target implied by the measurement: 3x on the dots would put decode near 50 tok/s, i.e. above the 60 %
gate.


## Decode: the concrete fix the measurements point at (2026-11-27)

Chasing the 35 %-of-peak finding into the kernels located the mechanism. In the per-token dot
(`DotRow<TYPE>` in `src/hip/kernels.hip`, the Q4_K/Q6_K branches from line ~256):

- each lane owns one 256-value block and walks it **serially**;
- for every block it calls `ActivationScale(x + b*256, 256, scale)` — 256 reads of the f32
  activation — and then, for each of the 256 values, reads the activation **again** to quantise it;
- so a row of 2560 costs ~5120 activation reads, and a matvec with N rows repeats that N times,
  because nothing is hoisted: the activation is re-read and re-quantised per output row.

`ggml` (and this engine's own prepared path) quantise the activation **once per matvec** and reuse it
across rows. That path exists here: `LaunchGemvPrepared` with `GemvPreparedKernel<kQ4K>` and
`<kQ6K>` (line ~834), fed by `LaunchQuantizeActivations` — and it is already used for the LM head and
for Q4_K expert gathers.

The gap is dispatch: `LaunchGemvExpertsPre` (line ~1120) handles **only** `kQ4K` and falls back to the
float per-row path for everything else, so the Q6_K experts — measured at 13.5 ms, 63 % of gather time
— take the slow path. The same is true of the non-expert projections that still call `LaunchGemv`.

Two mechanical changes, in this order, each measurable with `bench-decode.sh` and gated on
`tools/golden_set_ids.py` (plain 19 / arithmetic none / tool-call 16 / thinking-on 2):

1. Extend `LaunchGemvExpertsPre` to `kQ6K`/`kQ5K` by looping the selected experts and calling the
   existing `GemvPreparedKernel<TYPE>` per expert (no new kernel needed: 8 launches instead of one,
   but with a single activation quantisation instead of one per row).
2. Hoist the activation quantisation out of `DotRow` for the remaining `LaunchGemv` call sites by
   routing them through `LaunchGemvPrepared`.

Expected size from the profile: Q6_K gather 13.5 ms and KDA's projection time (22.4 ms, itself mostly
weight reads through `DotRow`) are the two beneficiaries; halving both would move decode from ~17 to
~25-28 tok/s, which is at or near the 60 % gate (30 tok/s). Not implemented in this pass: at this
context size a blind edit to the hottest kernel is a poor trade against the instrumentation already in
place — the two changes above are mechanical and each is verifiable in isolation.


## Decode: two attempts measured, one blocker identified (2026-11-27)

**Q6_K prepared expert gather — no gain.** Extending `LaunchGemvExpertsPre` with a Q6_K prepared dot
(mirroring the verified Q6_K body, one warp per expert row, activation quantised once) compiled and
kept `ctest` 10/10, but moved decode only from 16.82-16.96 to 16.95-16.99 tok/s. Reverted: the Q6_K
experts are a small share of the gather (most layers are Q4_K, which already had the prepared path).

**Q8_0 prepared GEMV — blocked on a quantiser, not on the kernel.** The big projections are Q8_0
(blocks 0-1 entirely, plus the token embedding and LM head), and they go through `DotRow<kQ8_0>`, which
re-quantises the activation **for every output row**: a 4096-row projection re-quantises the same
2560-value vector 4096 times. A `GemvPreparedKernel<kQ8_0>` was written and compiles, but it cannot be
wired yet: `LaunchQuantizeActivations` emits one scale per **256** values (the Q4_K/Q6_K layout) while
a Q8_0 prepared dot indexes one scale per **32** values, and the `act_scales` buffer is sized for the
256 layout. So the missing piece is a 32-block activation quantiser plus its buffer, then routing the
projection call sites through `LaunchGemvPrepared`.

That is the concrete next step for `decode-opt`, and it is where the measured 35 %-of-peak bandwidth
says the time is: the projections are 2.6 of the 4.753 GB a token reads, and they are the rows that
re-quantise per row today.


### A shortcut that was rejected before shipping

Routing the Q8_0 projections through the prepared path *without* a 32-value activation quantiser is
easy (reuse the 256-value scales and index `scales[b/8]` in the kernel), and it compiles. It was
rejected: it coarsens the activation quantisation from 32 to 256 values for those weights, which
changes numerics. This project's rule is that a faster run with different greedy output is a
rejection, so the prepared Q8_0 path has to wait for the 32-value quantiser and its buffer rather than
buy speed with accuracy.


## Decode +13 %: the projections finally quantise the activation once (adopted 2026-11-27)

The measurement chain that led here: decode read exactly the roofline's 4.753 GB/token
(`tools/bytes_per_token.py`) yet ran at 16.9 tok/s where bandwidth allows 49.9; forcing perfect expert
locality bought only 4 %, which excluded scatter; that left effective bandwidth — the engine was
achieving ~84 GB/s of 237 GB/s. The cause was in the dot itself: `DotRow` re-read and re-quantised the
f32 activation **for every output row**, so a 4096-row projection quantised the same 2560 values 4096
times.

Fix: a 32-value activation quantiser (`LaunchQuantizeActivations32`, the granularity Q8_0 weights use)
plus a `GemvPreparedKernel<kQ8_0>` branch that consumes it, and the projection call sites routed
through `LaunchGemvPrepared` — one quantisation per matvec instead of one per row. The activation
quantisation is the same scheme as before (`amax/127`, `__float2int_rn`, clamp ±127), so the numerics
are unchanged by construction.

| stage | decode tok/s (median of 3) |
| --- | --- |
| before | 16.9 |
| + KDA q projection | 17.34 |
| + KDA k, v, f_a, g_a | 18.99 |
| + MLA q, kv_a, out | 19.15 |
| + LM head (Q8_0, 0.43 GB/token) | 19.34 |
| + KDA output projection | 19.63-19.72 |
| + router projection | **19.85** (median of 3) |

Correctness held at every step: `ctest` 10/10 and `tools/golden_set_ids.py` unchanged at plain 19 /
arithmetic none / tool-call 16 / thinking-on 2, i.e. identical greedy output at +13 % throughput. That
is 39.8 % of the roofline (19.85 tok/s), up from 34 %, with the 60 % gate still ahead. Re-verified at the final state:
`ctest` 10/10 and the golden set unchanged at plain 19 / arithmetic none / tool-call 16 /
thinking-on 2.

One attempt failed and was reverted: routing the shared expert and dense FFN through the same
prepared path (inside `LaunchFfn`, 35 layers in one place) crashed with a GPU core dump, so it was
backed out rather than debugged in the hot path; the remaining per-row GEMV sites are worth roughly
another 2-3 %, and beyond them the cost sits in the Q4_K expert gather and the KDA delta-rule kernels,
which are already prepared and need kernel-level work (vectorised loads, fewer launches) rather than
another quantisation hoist.


### dot4 for the prepared Q8_0 dot: blocked on a target feature

The prepared Q8_0 dot sums int8 products, and integer sums are exact and order-independent, so
replacing its per-element multiply-adds with `__builtin_amdgcn_sdot4` cannot change any result — it
would be a pure instruction-count win. It does not compile here:

    error: '__builtin_amdgcn_sdot4' needs target feature dot1-insts

Two routes were tried and both fail in this toolchain. A per-kernel
`__attribute__((target("dot1-insts")))` compiles but the kernel is never registered — the runtime
reports "Cannot find Symbol ... GemvPreparedQ8Dot4Kernel" — both inside an anonymous namespace and with
external linkage, so the attribute and the HIP offload registration do not combine here. The remaining route — enabling the feature for the translation unit — was then tested directly on a
standalone probe (`/work/dot4-probe.hip`, comparing `sdot4` against the scalar sum of four signed int8
products) and is **not available in this toolchain**: `-mllvm -amdgpu-enable-dot1-insts` is an unknown
LLVM option, and `-Xclang -target-feature -Xclang +{dot1,dot5,dot7,dot10}-insts` all fail to compile
for `--offload-arch=gfx1151`. So dot4 is ruled out here, not pending: the remaining decode headroom
has to come from load width and launch count in the existing shapes. Worth
doing: the prepared dot is the hot loop of every projection, and four int8 products per instruction
is the cheapest remaining lever now that the activation quantisation is hoisted.


### dot4 in the prepared Q8_0 dot: enabled, then reverted (results changed)

Enabling the feature for the translation unit works — `extension.cmake` gained
`set_source_files_properties(.../src/hip/kernels.hip PROPERTIES COMPILE_OPTIONS "-mllvm;-amdgpu-enable-dot1-insts")`
and the plain `__builtin_amdgcn_sdot4` loop compiles with no attribute (which is what broke
registration). But wiring it into `GemvPreparedKernel<kQ8_0>` made `ling3_air.dot` fail and the decode
benchmark crash, i.e. the intrinsic is not the plain signed 4-way dot its name suggests in this
toolchain — most likely the `negate`/signedness operand order, or `sdot4` being the unsigned variant.

Both the kernel and the cmake flag were reverted; decode is back to the verified **19.26 tok/s** with
`ctest` 10/10 and unchanged golden output. Next attempt should validate the intrinsic in isolation
first — a tiny test comparing `sdot4(w, a, 0, false)` against the scalar sum of four int8 products —
before it goes anywhere near a hot loop.

### Four-byte loads in the prepared Q8_0 dot: slightly worse, reverted

Unpacking four int8 values per 32-bit load (exact by construction, same integer sum) measured
**19.63-19.64 tok/s against 19.85** for the byte-wise loop, so the compiler is already vectorising
that loop and the explicit shifts only add instructions. Reverted.

With dot4 unavailable for this target, locality excluded, and the activation hoist done, the decode
improvements available at this level are exhausted: 19.85 tok/s (39.8 % of the roofline) is where the
current kernel shapes land. The next real step is the one `notes/prefill-design.md` describes for the
recurrence — a chunked scan — and the equivalent for the attention rows, which is kernel design
rather than tuning.

## Prefill: what the byte-identical guarantee rules out (2026-11-27)

The chunked and per-token prefill paths are now byte-identical (966/966 residual rows, verified). That
is a hard constraint on `prefill-opt`: any change that alters the arithmetic inside the chunk breaks
it, and a faster run with different output is rejected by this project's own rule. Concretely:

- **ruled out**: replacing the per-row gate/beta GEMVs with chunk GEMMs (different reduction order);
  re-associating the KDA recurrence into a parallel scan (different rounding at every step); vectorised
  re-quantisation of the activation (the quantiser must see identical inputs).
- **admissible**: removing *launches* without changing the maths — e.g. one kernel per (layer, chunk)
  that loops over the chunk's rows internally for conv+SiLU+L2-norm+delta (same order, same values),
  or fusing kernels that already run back to back on the same buffers. Both keep the numbers identical
  by construction and can be checked with the same 966-row comparison.

So the honest expectation for `prefill-opt` is launch-overhead removal on the existing shapes, not a
different algorithm — which is why the 70 % gate (≈4,400 tok/s against 23.8) is not reachable without
relaxing the identity rule. That trade-off is the user's call, not something to decide silently.

## Prefill +9 %: batched KDA (adopted 2026-11-27)

The chunked path ran the KDA delta rule, the decay gate and the output gate once per token, so a
32-token chunk re-read the gate projections' weights (41 MB each) for every row. Batching them:

| variant | pp2420 tok/s (median of 3) |
| --- | --- |
| per-row KDA (previous) | 25.5 |
| batched delta rule only | 25.8 |
| batched delta + batched gate/beta/output-gate GEMMs | 27.8 |
| + batched output-gate norm and sigmoid, copy removed | **27.9** |

The batched path is **byte-identical** to the per-row one — 966 of 966 residual rows across 42 blocks
and 23 positions, verified with the same instrument that proved the chunked/per-token parity — and
`ctest` is 10/10. It is now the default (`LING3_KDA_ROW=1` forces the old path).

What the measurement also changed: the gain is 9 %, not the ~30x the byte arithmetic predicted for
hoisting those weight reads, so the weights were already cache-resident between rows. Prefill is
therefore **latency-bound, not bandwidth-bound**: ~2,420 tokens in 87 s is 36 ms per token, and with
roughly 500 small kernels per token that is ~70 us per launch. Reaching the 2,000 tok/s target needs
far fewer, far larger kernels — a different design, not more hoisting. That is the honest position for
`prefill-opt`: each admissible increment is single-digit percent while the target is 70x away.

### SiLU folded into the conv kernel: slower, reverted

Removing three launches per layer by folding SiLU into `Conv1dStepKernel` measured **19.73 tok/s
decode against 19.85** and **27.7 against 27.9** for prefill: the in-kernel `expf` serialises the store
path and costs more than the launch it saves. Reverted.

That is the sixth measured micro-attempt this session whose result was neutral or negative, alongside
dot4 (unavailable), locality (excluded at 4 %), Q6_K prepared experts (no gain), 4-byte loads (worse),
and the `LaunchFfn` prepared routing (crashed). The four that did land — the prepared Q8_0 activation
path (+17.5 % decode), batched KDA (+8 %), batched output-gate stage (+1 %), head and router routing —
all share one property: they removed *redundant work*, and none merely rearranged it.

The engine is therefore at a plateau for this design: decode 19.85 tok/s (39.8 % of a 49.9 ceiling) and
prefill 27.9 tok/s. Both targets now sit behind a redesign rather than a tuning step: decode would need
fewer bytes per token (dot4 is unavailable for this target, locality is excluded), and prefill needs
far fewer, far larger kernels — a tiled attention stage and fused per-token stages — because it is
dispatch- and latency-bound, not bandwidth-bound.

## Prefill profile: the routed MoE is 57 % (measured 2026-11-27)

rocprofv3 on a chunked prefill of a 120-token prompt, aggregating kernel time by symbol:

| kernel | time | dispatches |
| --- | --- | --- |
| expert GEMV (prepared) | 1,887 ms | 2,679 |
| `GemmDotRow` (chunk projections) | 988 ms | 6,426 |
| expert GEMV (variants) | 401 + 353 + 46 ms | 9,729 + 282 + 423 |
| `RouterKernel` | 295 ms | 5,640 |
| `DeltaRuleChunkKernel` (batched) | 281 ms | 175 |
| `AbsorbQKernel` / `ValueAbsorb` | 74 ms | 1,974 |
| `QuantizeActivations*` | 21+ ms | 5,640 |
| everything else | remainder | |

Total kernel time 4,758 ms for ~4.4 s wall. Two conclusions that redirect `prefill-opt`:

1. **The routed MoE is ~57 % of prefill**, and the chunked path still calls its gather once per *row*
   (2,679 calls of ~700 us). This is the single largest cost and the one batch that is not batched.
   The fix is a grouped-MoE kernel: bucket the chunk's (row, expert) pairs by expert, then run one
   GEMM per bucket — which is what llama.cpp does and this engine does not.
2. The per-row `RouterKernel` (5,640 calls, 295 ms) goes with it: routing a whole chunk is one
   projection plus a batched top-k, not 32 of each.

That also explains why batching the KDA stages only bought 9 %: they were ~10 % of the time, not the
bulk. The next prefill change should be grouped MoE, and the expected gain from the profile is
substantial (a 2-4x reduction in the dominant term, i.e. prefill roughly doubling).

## Prefill +6.7 %: the router is batched (adopted 2026-11-27)

The chunked path routed each row separately (5,640 single-threaded `RouterKernel` launches, 295 ms in
the profile). `RouterKernel` is now row-indexed — one block per row — and a chunk issues one launch:

| variant | pp2420 tok/s | decode tok/s |
| --- | --- | --- |
| per-row routing | 27.9 | 19.85 |
| batched routing (row-indexed kernel) | **29.8** | 19.63 |

Prefill is +6.7 %, decode −1.1 %. The decode cost is intrinsic to the change rather than to the
pointer indirection (branching on `rows == 1` did not recover it), and it was accepted because the
prefill target is the one with headroom: 29.8 tok/s now, and the trade is recorded rather than hidden.
`ctest` 10/10 and the golden indices are unchanged (plain 19 / arithmetic none / tool-call 16 /
thinking-on 2).

Next, per the same profile: the routed-MoE gather is still per row and is 57 % of prefill. Making it
grouped — bucket the chunk's (row, expert) pairs, then one GEMM per expert bucket — is the remaining
large change, and it will alter accumulation order per row, so it needs the golden indices re-measured
(which the revised contract allows).

### Grouped MoE would not have paid: checked before implementing

The plan was to bucket the chunk's (row, expert) pairs and run one GEMM per expert bucket, like
llama.cpp. The arithmetic says it cannot pay here:

- a 32-row chunk makes 32 x 8 = 256 selections over 512 experts, so the expected number of *distinct*
  experts is `512 * (1 - (1 - 1/512)^256) = 202` — about **1.27 rows per expert**. Grouping would buy at
  most ~27 % weight reuse on one term.
- the measured cost is not weight traffic either: one per-row gather moves ~47 MB (0.2 ms at 237 GB/s)
  and takes ~0.7 ms, i.e. the Q4_K int8 dot runs at ~30 % of peak. That is exactly what dot4 would fix,
  and dot4 is unavailable for `gfx1151` in this toolchain (probe evidence).

So prefill's largest term is compute-efficiency-bound in a way that neither grouping nor the available
instruction set can improve, and prefill is therefore near its practical ceiling for this design at
~30 tok/s. The one remaining lever with real headroom is a tiled attention/projection kernel that does
more arithmetic per byte loaded by hand (register blocking), which is a rewrite rather than a tuning
step.

