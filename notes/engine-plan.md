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
