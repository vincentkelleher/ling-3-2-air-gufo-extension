# Text-engine plan

Everything the runner needs, with the exact tensor names and the reference code
to consult. Written to be resumed from cold: the operator layer is already done
and tested (see the end).

## Weights per block (verified against the GGUF, `src/weights.cpp`)

43 blocks: 0–41 trunk, 42 = MTP. Block kinds come from
`attention.head_count_kv` (0 → KDA, 1 → MLA): KDA = 0–4, 6–10, …, 36–40 (35
blocks), MLA = 5, 11, 17, 23, 29, 35, 41 (7 blocks), MTP 42 = MLA.

Common: `attn_norm.weight` [2560] F32, `ffn_norm.weight` [2560] F32.

**KDA block** (`ssm_*`, all Q8_0 unless noted; 4096 = 32 heads × 128):

| tensor | shape | notes |
| --- | --- | --- |
| `attn_q.weight`, `attn_k.weight`, `attn_v.weight` | [2560, 4096] | project to 32 heads × 128 |
| `ssm_conv1d_q/k/v.weight` | [4, 1, 4096] F32 | causal conv, kernel 4 |
| `ssm_f_a.weight` | [2560, 4096] | gate pre-activation |
| `ssm_a` | [1, 32] F32 | per-head decay scale |
| `ssm_dt.bias` | [4096] F32 | gate bias |
| `ssm_beta.weight` | [2560, 32] F32 | per-head beta |
| `ssm_g_a.weight` | [2560, 4096] | output gate |
| `ssm_norm.weight` | [128] F32 | per-head RMSNorm |
| `attn_output.weight` | [4096, 2560] | out projection |

**MLA block**:

| tensor | shape | notes |
| --- | --- | --- |
| `attn_q.weight` | [2560, 6144] | 32 heads × 192 (no q LoRA) |
| `attn_kv_a_mqa.weight` | [2560, 576] | 512 latent + 64 rope |
| `attn_kv_a_norm.weight` | [512] F32 | on the latent |
| `attn_k_b.weight` | [128, 512, 32] | key absorb (nope 128 ← 512) |
| `attn_v_b.weight` | [512, 128, 32] | value absorb |
| `attn_gate.weight` | [2560, 32] F32 | per-head output gate |
| `attn_output.weight` | [4096, 2560] | 32×128 → 2560 |

**FFN**: blocks 0–1 dense (`ffn_gate`/`ffn_up` [2560, 6144], `ffn_down`
[6144, 2560], Q8_0). Blocks 2–42 MoE: `ffn_gate_inp.weight` [2560, 512] F32,
`exp_probs_b.bias` [512] F32, `ffn_gate_exps` [2560, 768, 512] Q4_K,
`ffn_up_exps` [2560, 768, 512] Q6_K, `ffn_down_exps` [768, 2560, 512] Q6_K,
shared expert `ffn_{gate,up,down}_shexp` Q8_0 [2560, 768] / [768, 2560].

**MTP block 42** additionally: `nextn.eh_proj.weight` [5120, 2560] Q4_0,
`nextn.enorm.weight`, `nextn.hnorm.weight`, `layer_output_norm.weight` (all
[2560] F32). Everything else in it is Q4_0.

Global: `token_embd.weight` [2560, 157184] Q8_0, `output_norm.weight` [2560],
`output.weight` [2560, 157184] Q8_0 (not tied).

## Forward, one token (batch 1), decode

Reference: `~/Source/llama.cpp/src/models/bailingmoe3.cpp` (graph builder) and,
for the delta rule, `GGML_OP_GATED_DELTA_NET` in `ggml/src/ggml-cpu/ops.cpp`
(search `gated_delta_net`) — read that file when implementing the KDA step; it
is the normative math.

1. `x = token_embd[token]` (Q8_0 row dequant, 2560 floats).
2. Per block: `h = rmsnorm(x) * attn_norm`.
   - **KDA**: q,k,v = `attn_{q,k,v} h` → causal conv1d with the last 3 columns of
     state, SiLU → L2-normalise q,k → gate `g = sigmoid((f_a h + dt) * ssm_a) *
     kda_gate_lower_bound` (per head-dim), `beta = sigmoid(ssm_beta h)` (per
     head) → delta-rule update of the per-head state (head_dim × head_dim) →
     per-head RMSNorm (`ssm_norm`) → multiply by `sigmoid(g_a h)` → `attn_output`.
   - **MLA**: q = `attn_q h` → split 192 = 128 nope + 64 rope; rope on q_pe and
     on the 64 rope dims of `kv_a = wkv_a_mqa h`; `kv_lat = rmsnorm(attn_kv_a_norm)`
     over the 512 latent; append (kv_lat, k_pe) to the latent cache; q_nope
     absorbed by `attn_k_b` per head (128 → 512); scores = q_abs · [k_lat; k_pe]
     / sqrt(192); softmax; values = attention over the 512-wide latent, then
     `attn_v_b` per head (512 → 128); multiply by `sigmoid(attn_gate h)` →
     `attn_output`.
   - `x += attn_out`; `h2 = rmsnorm(x) * ffn_norm`.
   - Dense: `x += ffn_down(silu(gate) * up)`.
   - MoE: `logits = gate_inp h2` (F32), `+ exp_probs_b` bias, sigmoid, group
     limit (8 groups, top-4 groups by summed top-2), top-8 experts, normalise
     weights, ×2.5; `sum_e w_e * down_e(silu(gate_e) * up_e)`; plus the shared
     expert with no gate scaling; `x += moe + shared`.
3. `h = rmsnorm(x) * output_norm`; `logits = output h` (157k × 2560 Q8_0 GEMV).
4. Greedy: argmax. Later sampling follows the model card's defaults (temp 1.0,
   top_k 20, top_p 0.95) for chat, greedy for the conformance runs.

## State per session

- KDA block: conv state `3 × (kernel-1) × 4096` floats (q,k,v) + delta state
  `32 × 128 × 128` floats (~2 MiB per block, 35 blocks → ~72 MiB). Append-only
  in time; the cache treats it as private state.
- MLA block: latent cache `576` floats per token per block (7 blocks) →
  16.1 KiB/token; append rows.
- MTP: reuses the MLA cache of block 42's own attention.

## Kernels still to write (all HIP, same shape as `src/hip/kernels.hip`)

conv1d step + state shift; L2 normalise; rope (partial, 64 dims); delta-rule
step; per-head RMSNorm + gate; MLA absorb attention (scores, softmax, weighted
sum, v absorb); router (sigmoid, group top-k, normalise); expert gather GEMV
(one token × 8 experts × 3 matrices); fused silu-mul; LM-head GEMV; argmax.

## Tests

1. `tests/operator_gpu_test.cpp` — done: GEMV (F32/Q8_0/Q4_K/Q6_K), RMSNorm,
   SiLU; 17/17 pass, worst relative error 1.4e-6.
2. Per-operator checks against a host reference computed from the same GGUF
   rows (extend the operator test with real rows once the runner can read them).
3. Whole-model: greedy token equality against llama.cpp CPU
   (`/home/ai-box/ling32-work/llama-cpu/bin/llama-completion --model <shard 1>`)
   on the pinned golden set; then `gufo bench` numbers against the roofline.

## Build and run

`~/ling32-work/`: `dev-env.sh` (CPU), `run-gpu.sh` (GPU visible),
`build-hip.sh` (Gufo + extension compiled in), `build-tests.sh` +
`run-tests.sh` (ctest with `LING3_GGUF`). Standalone operator test:
`hipcc -O3 --offload-arch=gfx1151 -I/work/ext-build tests/operator_gpu_test.cpp
src/hip/kernels.hip -o /work/operator_gpu_test`.

## Status 2026-11-26 (engine runs, output not yet correct)

`tests/forward_test.cpp` loads the model and decodes on the GPU. Three findings
worth keeping:

1. **Zero-copy weights.** Copying 80 GiB into GTT doubles the footprint and
   thrashes (the first run never finished). `hipHostRegister` accepts the
   reader's read-only mmap and `hipHostGetDevicePointer` returns the same
   address; `tools/zerocopy_probe.hip` measures 238 GB/s warm through it versus
   223 GB/s through a device copy, so registering the three shards is both
   correct and as fast. Load time is now ~2 s warm, 36 s cold.
2. **This quantization varies per tensor and per layer.** `blk.31.ffn_down_exps`
   is Q5_K while its neighbours are Q4_K; reading an unimplemented type as
   floats produced all-NaN logits. The engine now refuses unknown storage types
   at load, and Q5_K has a kernel.
3. **Not yet correct.** Logits are finite (nan 0, range -11.9..8.6) but the text
   is gibberish, so some semantics are still wrong. Next suspects, in order:
   the RoPE convention (neox pairing vs interleaved — `LING3_ROPE=interleaved`
   switches it), the thinking marker in the prompt, the KDA conv/state layout,
   and the MLA score scale. The decisive test is the llama.cpp comparison:
   `~/ling32-work/llama-cpu/bin/llama-completion` reads the same GGUF and shares
   the page cache, so it can run beside this without another 80 GiB.

## What is proven and what is still wrong (2026-11-26 late)

Proven, with tests that pass:

- Dequantization matches Gufo's reference on real rows of every storage type,
  including `blk.31.ffn_down_exps` (Q5_K): `tests/dot_vs_gufo_test.cpp`, 14/14.
- The embedding-row kernel is bit-exact against the same reference.
- 23 GPU operator checks (GEMV F32/Q8_0/Q4_K/Q5_K/Q6_K, RMSNorm, conv1d step,
  L2 norm, RoPE, argmax, KDA delta rule, MLA pipeline, router, expert gather).

Two real bugs found and fixed on the way: the Q5_K-in-place-of-float read (all
NaN) and the conv1d weight layout (`ssm_conv1d_*.weight` is `[kernel, 1,
channels]`, so channel `c` owns `weight[c*4 + k]`; reading it transposed gives
finite garbage).

Still wrong: end-to-end text is not coherent (llama.cpp answers the same prompt
with "The user is asking"). Logits are finite and plausible in range, so the
remaining error is semantic, not numeric. Untested suspects, in the order worth
checking:

1. KDA state *ordering across tokens*: the conv window and the delta state are
   updated per step; a mismatch only shows after several tokens.
2. RoPE: `LING3_ROPE=interleaved` switches the pairing; both were garbage before
   the conv fix, so re-test both now.
3. Router semantics versus llama.cpp's `build_moe_ffn` (normalisation order,
   whether the shared expert is scaled).
4. MLA: whether `attn_v_b` applies to the latent output per head (current) or
   after concatenation.

## Stage-by-stage verification (2026-11-26, position 0)

Every stage of the forward now matches a numpy reference built from the GGUF
(with `gguf.quants.dequantize`), using dumps written by `LING3_DUMP=1`:

| stage | max abs diff |
| --- | --- |
| token embedding row | 0 (bit-exact) |
| Q8_0 projection GEMV (attn_q @ h) | 1.4e-6 |
| conv1d + SiLU, layout `[kernel,1,channels]` | 6e-8 |
| L2 norm per head | 1.2e-7 |
| KDA delta rule (state zero) | 3.3e-10 |
| KDA per-head RMSNorm | 3.7e-9 |
| KDA attn output projection | 1.5e-8 |
| residual + ffn_norm wiring | 7e-7 |
| MLA q split / kv latent norm | 0 / 1.2e-7 |
| MLA q absorb (attn_k_b, Q8_0) | 1.1e-6 |
| MLA scores (q_abs·latent + q_pe·k_pe, /sqrt192) | 1e-7 |
| MLA value absorb + gate (attn_v_b) | 4.2e-7 |
| MoE router ids / weights | exact |
| MoE shared expert increment | 1.2e-7 |

So the whole single-token forward is numerically correct stage by stage. What is
still missing is the *end-to-end* comparison against llama.cpp logits, which is
also the `engine-assembly` contract. The earlier attempt failed on setup, not on
the engine:

- `llama-tokenize -f prompt.txt` and the reference tokenizer produce the same 25
  ids, so tokenization is not the issue.
- `llama-perplexity --save-all-logits` writes a file whose rows are not plain
  `n_vocab` float rows (4,401,412 bytes for 7 rows of 628,736 = +260 bytes), so
  it needs its header decoded before comparing.
- Do not compare across positions: a 3-step run's `d5_scores` is a 3-key softmax
  input, not the position-0 value (this produced one false alarm).

Next: dump logits at every prefill position (`dump_raw` in the prefill loop) and
align them with the perplexity file's rows; report RMS and argmax agreement.

## Next step, precisely: verify position 1 (the state update)

Every position-0 stage is verified, so the remaining error must be in the
recurrent state or in the end-to-end comparison. The cheapest decisive check is
the delta rule at position 1, which is closed-form from position-0 dumps:

    M1[j][i] = k0[i] * v0[j] * beta0            (state after token 0)
    M1'[j][i] = M1[j][i] * exp(g1[i])           (decay at token 1)
    delta1[j] = (v1[j] - sum_i M1'[j][i] * k1[i]) * beta1

Run a two-token prompt with `LING3_DUMP=1`, keep position 0's `kn`/`cv`/`beta`/
`gate`, dump position 1's, and compare `d_delta.bin` against the formula above.
If it matches, the state is right and the difference is only in the end-to-end
comparison; if not, the conv shift (`LaunchConv1dStep` state rotation) or the
decayed-state multiply is wrong.

Also note for the llama.cpp comparison: `--save-all-logits` writes a **custom
file**, not raw float rows — 4,401,412 bytes with a 260-byte header that starts
with ASCII text and includes `n_vocab` and a token count, followed by 7 rows
that contain 87 NaNs. Decode that header (or use token equality instead of
logits) before comparing.

## Result of the position-1 check (2026-11-26, current lead)

Reproduction: run the engine with `LING3_DUMP=1` for 1 step (position 0) and
again for 2 steps (position 1), keep both sets of `d_kn`, `d_cv`, `d_beta`,
`d_gate`, `d_delta`, then compare position 1 against the closed form

    M[j][i]  = k0[i] * v0[j] * beta0
    M'[j][i] = M[j][i] * exp(g1[i])
    delta1[j] = (v1[j] - sum_i M'[j][i] * k1[i]) * beta1

Numbers: reference delta rms **0.01905**, engine delta rms **0.00027** (max diff
0.288). Position 0 matched, so the divergence is in carrying the state: the
engine's updated term is 70x too small relative to `v1`, which means the state it
reads back is not `M` (or the decay/beta applied to it is not the one the dumps
show). Position 0's own state write is unverified, so both halves are suspect:

1. the state written after token 0 (`row[i] += k[i]*delta[j]` in
   `DeltaRuleKernel`), and
2. the decay `row[i] *= exp(g[i])` read back at token 1.

Next concrete step: dump `delta_state[0]` after token 0 and compare with
`k0[i]*v0[j]*beta0` elementwise (2 MiB dump); that splits the two.

### State write is correct; the read path is the suspect

`d_state.bin` after token 0 equals `k0[i]*v0[j]*beta0` to **1.9e-9** (rms 0.00096
both sides), so `DeltaRuleKernel` writes the state correctly. The position-1
mismatch therefore comes from what the kernel *reads back* (decay + dot with
`k`), or from assembling the numpy reference out of separate dumps — the engine's
position-1 delta rms (0.00027) matches `(v1 - small) * beta1` with a small
`beta1`, while the reference built from the dumps gives 0.019, which is what you
get with `beta1 ≈ 0.9`. One of the two is not the pair the kernel used.

Next step (removes the ambiguity): have the kernel itself dump `dot` (the
`sum_i M'[j][i]*k[i]` term) for block 0 at position 1, and compare against
`(v1 - dot)/beta1` in numpy. That is a single number per head-dim, so it cannot
be an artefact of mixing dumps from different positions.

### The kernel's `g` is not the buffer that was dumped

Numbers from the two-position experiment:

| quantity | value |
| --- | --- |
| state after token 1 vs `M0*exp(g1) + k1 (x) delta1` | 0.0363 |
| state after token 1 vs `M0 + k1 (x) delta1` (no decay) | **0.00161** |
| `exp(g1)` rms from the dumped gate | 0.7038 |
| engine-implied `dot` rms vs `v1` rms | 0.0219 vs 0.0220 |

So the stored state is the *undecayed* update, even though `DeltaRuleKernel`
does `row[i] *= decay[i]` with `decay[j] = expf(g[head*head_dim + j])`, and the
`g` buffer dumped in the same iteration has rms `exp` of 0.70. The kernel must be
reading a different `g` than `m.kda_gate` (or reading it before it is written:
`LaunchKdaGate` and the delta rule are on the same stream, so ordering should
hold — unless the gate buffer is reused by another launch in between).

Next step: in `DeltaRuleKernel`, dump `decay[0..head_dim)` for head 0 to a file
on the first call, and compare with `expf(d_gate.bin[0..128])`. If they differ, the
pointer is wrong; if they match, the multiply is being lost (e.g. a compiler or
aliasing problem) and the fix is to write the decayed state explicitly.

### Correction: the position-1 experiment was invalid (do not chase the decay)

`d_state.bin` — the file I treated as "state after token 1" — equals position 0's
`M0 = k0[i]*v0[j]*beta0` to **1.9e-9**, while `d_kn`/`d_cv`/`d_gate` in the same
comparison *do* differ between the 1-step and 2-step runs. So the state dump was
position 0's, and every "the decay is not applied" number follows from that:
`state == M0*exp(g1) + upd` looked wrong (3.6e-2) and `state == M0 + upd` looked
right (1.6e-3) purely because the file was M0.

Before redoing the check, make the dumps **position-stamped** (`d_state_p%u.bin`,
same for the other stages). Then the two-position comparison is meaningful, and
only then is it worth asking whether `exp(g)` reaches `DeltaRuleKernel`.

What this leaves: every position-0 stage verified (table above), the state write
verified, and the end-to-end difference from llama.cpp unexplained. The next
two steps are therefore (1) position-stamped dumps + the position-1 delta check,
and (2) token-for-token comparison against llama.cpp on the pinned prompt, which
is what `engine-assembly` actually requires.

### Position 1, with stamped dumps: the kernel's state update is not the one modelled

Position-stamped dumps (`p_*_%u.bin`) now separate the steps, and the picture is:

| check | max abs diff |
| --- | --- |
| `state_0` vs `v0[j]*k0[i]*beta0` | 1.9e-9 |
| `delta_1` vs `(v1 - sum_i (M0*exp(g1))[j][i]*k1[i]) * beta1` | 0.288 (engine rms 2.7e-4, ref 1.9e-2) |
| `state_1` vs `state_0*exp(g1) + k1(x)delta_1` | 0.0786 |
| `state_1` vs `state_0 + k1(x)delta_1` | 0.0812 |
| `state_1` vs `state_0*exp(g0) + ...` | 0.0744 |

`state_1` rms (1.79e-3) is *larger* than `state_0` (0.96e-3) while the update term
is tiny (2.4e-5), and the elementwise effective decay has median 0.65 (close to
`exp(g1)`'s median 0.695) but a near-zero correlation — the magnitude looks like a
decay was applied while the per-element values do not match any of the variants
above. That pattern fits an axis/layout mismatch or a race rather than a missing
decay.

Next step (one build): have `DeltaRuleKernel` dump, for head 0 only, `row[0..127)`
before the decay, `decay[0..127)`, and `row[0..127)` after it. Compare those three
with numpy and the axis question answers itself. Everything up to position 0, and
the whole state *write*, is verified.

### In-kernel dump: the decay is right, and two dumps disagree within one launch

`DeltaRuleKernel` now records, for head 0 row 0: the row before decay, `decay`,
the row after decay, `v`, `dot` and `beta`. Results:

- `decay == expf(gate[0..128))`: **5.96e-8** (the gate buffer the kernel sees is
  the one dumped, and the multiply is applied: `post == pre*decay` exactly).
- Position 1: `pre` rms 5.96e-4 -> `post` rms 1.40e-4, i.e. the decay shrinks the
  state as it should.
- The kernel's `v` equals `m.cv` (bit-identical) and `dot` is 6.6e-4 with
  `beta` 0.8813, so its own arithmetic says `out[0] = (v[0] - dot)*beta = 0.0119`.
- But the dumped `out` (`p_delta_1.bin`) holds **2.5e-4** at index 0, which
  corresponds to `v[0] ~ 9.5e-4`, not 0.0142.

So within what should be one launch the debug record and the output disagree,
which means the two files were not written by the same launch (the debug buffer
is overwritten once per KDA block 0 call, and the output dump sits in the same
`if (il == 0)` block). Next step: add a launch counter to the debug record (index
`5*head_dim` in `m.kernel_debug`) and stamp the output dumps with it, so the
pairing is unambiguous; then redo the position-1 comparison. Every conclusion
above about the decay being correct stands — it is measured inside the kernel.

### Resolved: `m.cq` after the delta call is the attention output, not the delta

`DeltaRuleKernel` writes `out[j] = (sum_i row[i]*q[i]) / sqrt(head_dim)` — the
q read-out — while `delta[j] = (v[j] - dot)*beta` is an intermediate that is
never stored. All my position-1 comparisons treated `p_delta_*.bin` as `delta`
and so compared an attention output (rms 3.4e-4) against a delta formula
(rms 1.9e-2). That is the whole "70x too small" and "does not match any variant"
story: analysis error, not an engine bug.

The kernel-recorded row now equals the output buffer **exactly** (0 diff at
positions 0 and 1), `decay == expf(gate)` to 6e-8, `post == pre*decay` exactly,
and the state write after token 0 equals `v0[j]*k0[i]*beta0` to 1.9e-9.

So the KDA block is verified end to end, including the state, and the same holds
for the MoE (router exact, shared 1.2e-7) and MLA (scores 1e-7, value absorb
4.2e-7) blocks.

### What that leaves

Nothing measured so far is wrong, yet the composed model emits newlines where
llama.cpp answers "The user is asking". The remaining honest step is a **numpy
forward of the first blocks compared against per-block `x` dumps** — dumps exist
only for block 0 today, so:

1. dump `x` after every block (`p_x_<layer>_<position>.bin`),
2. port blocks 0-5 to numpy (KDA, dense FFN, MoE, MLA) using `gguf.quants`,
3. compare block by block and stop at the first mismatch.

Do not repeat the position/stage dump arithmetic without an explicit
"what buffer holds what" table first; three of the four anomalies so far were
mismatched buffers, not mismatched math.

### Better reference than a numpy port: llama.cpp's own per-op dumps

`llama-eval-callback` (built at `~/ling32-work/llama-cpu/bin`) writes the tensors
of every graph node for a prompt, which is the reference this extension should be
compared against block by block — no hand-written numpy forward needed, and it
removes the last chance of a reference-implementation mistake (three of the
anomalies in this file were exactly that).

Plan: run it on the pinned 25-token prompt, take llama.cpp's `l_out-<n>` (block
output) and `result_norm`/`result_output`, and compare with the engine's
`x_<layer>_<position>.bin` dumps (now written when `LING3_DUMP_BLOCKS` is set).
The first layer where they diverge is the bug.

Current evidence that the composition, not a stage, is at fault: every stage
listed in the table above matches, yet the engine's top-8 logits for the pinned
prompt (65522, 4279, 28057, 70, 12773, 14, 12, 39) do not contain `The` (678),
which is what llama.cpp generates for that prompt.

### `llama-eval-callback` prints, it does not dump

Run: `cd /work/cbout && llama-eval-callback -m <gguf> -f /work/prompt.txt -n 1
--temp 0`. It writes **no files** (the directory stayed empty) and prints every
node's tensor to stdout in a truncated array form (`[2.7559, 8.7211, ..., -2.7697]`)
plus a `sum = ...` per tensor. Truncation makes it useless for an exact
comparison, and parsing stdout for 42 blocks x dozens of ops is not worth it.

Better next step for the composition bug: a numpy forward of **blocks 0 and 1**
only (both KDA + dense, whose primitives are already verified), compared against
`x_0_0.bin` and `x_1_0.bin`. If block 0 and 1 match, extend to block 2 (MoE) and
block 5 (MLA) the same way; the first block whose output diverges is the bug, and
the numpy code is exact rather than truncated.

Cheaper still, if the numpy port is too slow: compare our `x_5_1.bin` (MLA block,
second position) against `x_5_0.bin` under the hypothesis that only the *state*
paths (KDA conv/state, MLA cache) are wrong — i.e. run a two-token prompt whose
second token is the same as the first and check that the block outputs differ in
the way the cache implies.

## Block 0 composition: k and v disagree, q is exact (2026-11-26, current lead)

A numpy forward of block 0 against `x_0_0.bin` gives correlation 0.9898 and
rms 0.0389 (numpy) vs 0.0337 (engine) — close, not exact. Stage by stage at
position 0, numpy vs the engine's own dumps:

| stage | max abs diff | verdict |
| --- | --- | --- |
| `qn` (q conv + SiLU + L2) | 1.8e-7 | exact |
| `gate` (f_a·h + dt, ·a, sigmoid, ×(−5)) | 1.4e-6 | exact |
| `beta` | 1.2e-7 | exact |
| `cv` (v conv + SiLU) | 7.1e-2 | **differs** |
| `kn` (k conv + SiLU + L2) | 7.5e-1 | **differs** |

Same kernel, same layout, same call shape — only the tensors differ, so the
suspects are the weight mapping for `kda_k`/`kda_v` and the conv weight tensors
`ssm_conv1d_k/v.weight` (q's path proves the conv formula and layout are right).

Next: dump `m.q`, `m.k`, `m.v` (projections) at position 0 and compare each with
`W@h` in numpy. That splits "projection mapped to the wrong tensor" from "conv
weight misread", and the fix is one of the two.

### Projections are exact; the k/v conv is the remaining difference

`m.q`, `m.k`, `m.v` all match `W@h` in numpy to **1e-6** (rms identical), so the
projection and the tensor mapping are right. With `qn` exact as well, the only
stage that differs in block 0 is the conv output for k and v (`cv` 7.1e-2,
`kn` 7.5e-1) — the same kernel whose q path is exact.

Inverting `silu` to recover the effective conv weights was a dead end: it divides
by `v[j]`, which is often near zero, so the recovered values blow up (up to 4.7e5)
and mean nothing. Don't repeat that.

Next step, precisely: dump `dev(layer.kda_conv_k)` and `dev(layer.kda_conv_v)`
(the first 8 floats) at load and compare with
`f32('blk.0.ssm_conv1d_k.weight')[:8]` / `..._v`. If they match, the conv weights
are right and the difference must come from the *input* or an aliasing/ordering
problem — in that case dump the conv input (`m.k`, `m.v`) and the output in the
same call and compare them against the dumped projection, which catches a buffer
being reused between the two launches.

### No tensor swap: the k/v conv uses the right weights but the wrong window

Comparing each dumped conv output against `silu(w*x)` for all three conv tensors:

| path | vs conv_q | vs conv_k | vs conv_v |
| --- | --- | --- | --- |
| k | 1.45 | **0.751** | 1.40 |
| v | 0.209 | 0.216 | **0.0713** |

Each path matches its own tensor best, so the mapping is right, but neither is
exact even though the input projection is exact to 1e-6 and the state should be
zero at position 0. The q path (whose state slice is written first) *is* exact.

That asymmetry is the clue: q runs first and matches, k and v run after it and
do not, which is what happens if the conv reads a window that is not
`[0, 0, 0, x]` — i.e. the state slice or the shift inside `LaunchConv1dStep` is
not what the code reads like at call time. Next step: dump, inside the conv
kernel for `channels` 0..3, the four window values it actually multiplies and the
four weights, for the q, k and v launches of block 0 at position 0. Four floats
each; that ends the guessing.

### Fixed: the conv state was indexed out of bounds for k and v

`LaunchConv1dStep` indexes `state[c]`, `state[channels + c]`, `state[2*channels + c]`,
but the k and v calls passed a pointer already offset by `kda_inner` and
`2*kda_inner`. With a `3 * channels` allocation that reads (and writes) up to
`4*channels` and `5*channels` past the base — out of bounds for k and v, in bounds
only for q. That is exactly the asymmetry measured (q exact, k/v wrong).

The kernel now takes a `slot` (0/1/2) and the base pointer, so all three slices
are addressed correctly. Logits changed (top-5 went from 65522/4279/... to
16/17/1018/351/64) but the model still emits newlines, so this was a real bug and
not the only one.

Next: re-run the block-0 numpy comparison (block 0 was corr 0.9898 before this
fix) — if it is now exact, move to block 1 and continue block by block; if not,
diff the conv window as described above.

### The k/v conv still differs with exact inputs, correct mapping and zero state

After the slot fix, block 0 is unchanged (corr 0.9898): `vproj`/`kproj` match
numpy to 7e-7, `cv` differs by 7.1e-2 (engine rms 1.39e-2 vs numpy 1.31e-2) and
`kn` by 7.5e-1. The mapping is right (`weights.cpp` maps each `ssm_conv1d_*` to
its own field, and the calls use the matching fields), the state is memset to zero
at load, and at position 0 the window must be `[0,0,0,x]`, i.e.
`out = silu(w[c*4+3]*x)` — exactly what numpy computes. The engine's slightly
larger rms suggests an extra nonzero window term, i.e. the state is not the slice
the kernel thinks.

Next (single build): in `Conv1dStepKernel`, for `c < 4`, write the four window
values, the four weights and the slot index into a debug buffer; dump it for the
q, k and v launches of block 0 at position 0. Four floats each decide whether the
window, the weights or the slot is wrong.

## SOLVED: conv state slot stride (2026-11-27)

Each conv slot holds three history columns, so slots are `3 * channels` apart,
not `channels`. With a `channels` stride, q's shift wrote into k's window and k's
into v's (the in-kernel dump showed k/v windows `[0, x_q, 0, x]`). Fix: allocate
`9 * channels` and address `state + slot * 3 * channels`.

Result: coherent output. `The capital of France is` now continues with
"The user is asking a straightforward factual question about the", and
llama.cpp's answer for the same prompt is "The user is asking" — so the engine is
at or near token equality. Next: the pinned golden set, then the serving path.

## Assembly check: argmax matches llama.cpp (2026-11-27)

`The capital of France is`, greedy, same 25-token prompt:

- engine: step 0 argmax = **678** = `The`, logits max 25.46, five top-5
  `678, 1921, 1672, 90833, 480`.
- llama.cpp (CPU, same file, same prompt): first generated token `The` = 678.

So the first-token contract holds. Remaining for `engine-assembly` before the
task can be called complete: report the logits RMS against llama.cpp, which needs
`--save-all-logits`'s header decoded (it is a custom file: `_logits_` header with
context, n_vocab, counts, then rows that include NaNs at some offsets).

## 32-token comparison vs llama.cpp (2026-11-27)

Same prompt, greedy, 32 tokens:

- llama.cpp: `The user is asking a straightforward factual question: "The capital
  of France is" — this is a well-known fact. The capital of France is Paris.`
- engine: `The user is asking a straightforward factual question about the
  capital of France. The answer is Paris.The capital of France is **Paris**.Are
  you`

Same answer, same first eight tokens, then divergence at the punctuation. So the
engine is numerically *close* but not bit-identical, and greedy decoding turns
that into a different sentence. Getting token equality needs the per-token
logit difference down — likely accumulation-order effects (llama.cpp dots in a
different order and quantizes activations), not a structural bug.

Next: measure the divergence position and the logit margin at it (how close the
top-2 are), then reduce error where it matters — for example, keep activations in
fp32 through the MoE gather and match llama.cpp's `Q8_K` activation quantization.

## The divergence is decided by a 0.022 logit margin (2026-11-27)

Per-step argmax and top-2 margin for the pinned prompt (logits dumped per
position with `LING3_DUMP`):

| step | token | text | margin |
| --- | --- | --- | --- |
| 0 | 678 | `The` | 8.134 |
| 1 | 2838 | ` user` | 8.961 |
| 2 | 341 | ` is` | 5.322 |
| 3 | 10519 | ` asking` | 8.567 |
| 4 | 259 | ` a` | 3.399 |
| 5 | 22365 | ` straightfo…` | 0.595 |
| 6 | 36593 | ` factual` | 4.097 |
| 7 | 3244 | ` question` | 4.291 |
| 8 | 866 | ` about` | **0.022** |
| 9 | 268 | ` the` | 4.587 |
| 10 | 7706 | ` capital` | 13.334 |
| 11 | 300 | ` of` | 9.027 |

The engine and llama.cpp agree for eight tokens and split at step 8, where the
top-2 margin is 0.022 on logits of order 20 — 0.1 %. So the divergence is decided
by numerical noise, not by a structural difference: matching llama.cpp token for
token needs the logit error below ~0.02 at such steps, which means matching its
quantized-activation dots (`Q8_K` activation quantization before the K-quant dot
products) rather than accumulating dequantized weights against fp32 activations.

This is the documented deviation the `engine-prefill` contract asks for: pinned
prompt, divergence index 8, margin 0.022, same sentence meaning and same final
answer.

## Golden-set spot check (2026-11-27)

Prompt 2 ("What is 17 times 23?"): llama.cpp answers "17 x 23 = 17 x 20 + 17 x 3 = 3", the engine answers "The user is asking for the product of 17 and 23. Let me compute this." — same task, different opening, so this prompt diverges at token 0. Prompt 1 matched for eight tokens. Agreement is therefore prompt-dependent, consistent with a logit error of order 0.02-0.5 deciding greedy picks. Both answers are coherent.

## Q8_K activation parity attempted (2026-11-27)

The K-quant GEMV branches now quantize activations the way llama.cpp does (one
scale per 256-weight block, values rounded to int8 and clipped to +-127) before
the dot. Prompt 1 still diverges at the same word ("about" vs ":"), so this alone
does not flip that token; the remaining error must come from elsewhere (the Q8_0
dots, the LM head, or accumulation order inside the attention/MoE paths).

## Single-token context: margin 0.013 (2026-11-27)

BOS-only context (token 156891, confirmed by `llama-tokenize -f bos.txt`): the
engine argmax is 23282 with top-5 `23282=4.529, 128206=4.516, 34775=4.047,
8010=4.021, 90827=4.002` — a top-2 margin of **0.013**. llama.cpp printed an
empty first token for the same input (the token is whitespace-only), so the
argmax comparison for this case is not observable through its text output.

That is the third razor-thin margin measured (0.022 at prompt 1 step 8, 0.013
here), which is consistent with a systematic logit error of ~0.02 and means
token-for-token agreement on a golden set of any size will be decided by that
error. `engine-assembly` stays open: its RMS half needs llama.cpp logits, which
`--save-all-logits` does not provide in a usable form, and its argmax half needs
a reference token for the single-token case.

## Two negative results on the parity gap (2026-11-27)

Both attempted and reverted/kept-neutral, recorded so they are not repeated:

1. Quantizing activations to Q8_K per 256-block before the K-quant dots (what
   llama.cpp does) changed nothing: prompt 1 still splits at the same word.
2. Accumulating the K-quant dots in double precision instead of float changed
   nothing either, and cost 40% of throughput (6.4 -> 3.76 tok/s).

So the residual difference is not activation rounding and not fp32 accumulation
loss. What is left is the shape of the dot itself: llama.cpp sums integer
products per sub-block and applies a single float scale per group, while these
kernels multiply dequantized floats. That rewrite (Q4_K: 8x32 groups with d*sc
and dmin*m corrections; Q6_K: 16-element groups) is the next and only change on
this critical path.

## Attempted integer Q4_K dot: wrong, reverted (2026-11-27)

Rewriting the Q4_K inner product as per-group integer sums (q*a and sum_a with
`d*sc`/`dmin*m`) produced garbage text, so the unpacking I used is wrong
(likely the 6-bit scale/min extraction per group, which llama.cpp does with its
own `get_scale_min_k4` on the 12 packed bytes). Reverted to the dequantizing
dot, which is correct and coherent. If this is retried, port
`get_scale_min_k4` exactly and unit-test the integer dot against
`quant::Dot` per block before touching the engine.

## Integer Q4_K dot in the kernel: correct, faster, same divergence (2026-11-27)

With the activation scale included (the bug the new CPU unit test caught in one
run), the Q4_K branch now sums integer products per 32-group like llama.cpp:

- `ling3_air.dot` integer-vs-dequant check: rel 0.0019 / 0.0018 / 0.0002 on real
  blocks (activation rounding only).
- Engine: coherent text unchanged, throughput 6.4 -> 7.08 tok/s (+11%).
- Prompt 1 still diverges at the same word, so the residual difference is in the
  Q6_K (up/down experts, 14.6 GiB) and Q8_0 (attention, LM head) dots, which
  still dequantize.

Next: the same integer form for Q6_K (`-32` codes, 16-element groups, one float
scale each), then Q8_0, each validated by the dot test before the engine.
