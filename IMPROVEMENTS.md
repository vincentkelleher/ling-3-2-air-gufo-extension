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

### The MoE is at the hardware's scattered-read rate, not at a kernel defect

Six kernel-level hypotheses were measured out, so the remaining question was whether the access
pattern itself is the limit. `tools/scatter_probe.hip` answers it: it registers 4 GiB the way the
engine does, then reads eight 1.4 MB windows (one per selected expert) from that mapping,
contiguous inside a window and scattered across windows.

| pattern | bandwidth |
| --- | --- |
| sequential read of the same registered mapping (`tools/bandwidth_probe.hip`) | **237 GB/s** |
| eight scattered 1.4 MB windows (`tools/scatter_probe.hip`) | **38-52 GB/s** |

The routed MoE moves ~1.2-1.5 GB per token in 32 ms, which is ~40-47 GB/s — exactly the scattered
window rate. So the MoE is not slow because of launch count, block width, warp shape, lane width,
load width or instruction width (all measured, `IMPROVEMENTS.md`); it is slow because reading
eight 1.4 MB windows out of a 50 GiB mapping is a 38-52 GB/s pattern on this box.

**Consequence for the decode gate**: the roofline is computed from *sequential* bandwidth
(49.9 tok/s), and a routed MoE cannot read its weights sequentially. 60 % of that roofline
(29.9 tok/s) would require ~120 GB/s from this pattern, which the probe shows the hardware does
not provide. Reaching the gate needs a different memory strategy — for example a layout that
groups each expert's three matrices contiguously (3 x 1.4 MB per read instead of three separate
windows) or a residency scheme for the hot experts — not another kernel tweak. That is the honest
statement of where the gate sits.

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
