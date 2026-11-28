# Performance: what is measured, what it means, and what is closed

All figures are medians of three runs on an idle machine, release build, greedy decoding, produced by
`tools/bench_report.py` (which needs only the Python standard library). The full attempt log, including
every rejected variant with its number, is in `IMPROVEMENTS.md`; this page is the summary a reader
needs to judge the numbers.

## Current numbers

| measurement | value | reference on the same box |
| --- | --- | --- |
| decode, short context | **19.63-19.87 tok/s** (39-40 % of the memory roofline) | llama.cpp CPU: 12.52 tok/s (1.57x) |
| decode, tg128 | **19.85 tok/s** (39.8 %) | 12.52 tok/s |
| prefill, pp2048, chunked | **30.7 tok/s** (0.49 % of the compute roofline) | llama.cpp prompt eval: 24.20 tok/s |
| load | ~1.2 s for 74.87 GiB (zero-copy mapping) | — |

The two references measure different ceilings, so both are quoted: decode is bounded by **memory
bandwidth**, prefill by **arithmetic**.

## How the rooflines are derived

- Decode must read every weight the token touches: `tools/bytes_per_token.py` computes **4.753 GB** per
  token from the GGUF layout and the routing (8 of 512 experts), and the bandwidth probe measures
  **237 GB/s**, so the ceiling is **49.9 tok/s**. The engine is at 39-40 % of it.
- Prefill reads each weight once per chunk but does the same arithmetic per token, so its ceiling is
  arithmetic: **6,277 tok/s** for 2048 tokens. The engine is at 0.49 %.

## Where the time goes

rocprofv3 over a chunked prefill, aggregated by kernel (total 4.76 s of kernel time for a 120-token
prompt):

| kernel | time | share |
| --- | --- | --- |
| routed-expert GEMV | 1,887 ms | 40 % |
| chunk GEMMs (`GemmDotRow`) | 988 ms | 21 % |
| routed-expert GEMV (other variants) | 800 ms | 17 % |
| router | 295 ms | 6 % |
| batched KDA delta rule | 281 ms | 6 % |
| everything else | remainder | ~10 % |

The expert gather alone is 57 % of prefill and executes at **~1 % ALU utilisation** (PMC: 0.007 waves
retired per cycle, 101 G vector-ALU ops/s against a device peak of order 10 T/s) while moving only
**0.6 %** of the traffic that its runtime would imply at peak bandwidth. It is latency-exposed at its
dispatch shape: one warp per 2,560-element row.

## Closed by measurement

Each of these was tried and produced a number, not an opinion; the detail is in `IMPROVEMENTS.md`.

| hypothesis | experiment | result |
| --- | --- | --- |
| locality / TLB across the 4 GB expert table | `LING3_FORCE_EXPERT=0` | identical (29.78 vs 29.78 tok/s) |
| dot4 int8 instructions | `__builtin_amdgcn_sdot4` | unavailable for `gfx1151` in this toolchain (probe) |
| four output rows per warp | tiled prepared dot | identical |
| wider loads | 32-bit loads in the wide dot | +0.7 %, inside noise |
| lane occupancy | narrow vs wide work split | identical |
| accumulate-chain latency | four independent integer chains | slightly worse |
| fewer bytes per token | Q8_0 -> Q6_K substitution (-16 % bytes) | **4x slower** (4.93 vs 19.63 tok/s) |

The last one is the most informative: at 39 % of the roofline the engine is *not* bandwidth-bound, so
re-quantising cannot help, and Q6_K's unpacking costs compute in an already latency-exposed path.

## What was adopted

Prepared Q8_0 activation path across projections, head and router (**+17.5 % decode**), chunk-batched
KDA delta rule, chunk GEMMs for the gate/beta/output-gate projections, batched output-gate normalisation
and sigmoid, and a row-indexed router kernel (**+6.7 % prefill**). Net this session: decode 16.9 -> 19.6,
prefill 25.5 -> 30.7 tok/s, with greedy output unchanged (or the divergence indices re-measured).

## What remains, and why it is not a tuning step

Reaching the 60 tok/s decode target or the 2,000 tok/s prefill target requires the quantised dots to
become efficient — the measured ceiling for both is the same kernel family. The two facts that block
further black-box work:

1. the stall's cause is not visible in the counters this ROCm build exposes (TCC/TCP names are
   uncollected and the PMC event schema is undocumented here), and
2. every shape, width, occupancy, locality and byte-count variant that can be changed from source has
   been measured and is neutral.

So the honest statement is: **the numbers above are what this implementation reaches**, the remaining
gap is a kernel redesign against hardware counters (or a different device), and no further increment
should be claimed without such a measurement.
