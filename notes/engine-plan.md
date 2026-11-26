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
