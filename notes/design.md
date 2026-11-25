# Design notes: Ling 3.2 Air on Gufo

Working notes for the engine. Everything here is measured or read from the
checkpoint/GGUF, with the source named. The reference implementation is
llama.cpp's `src/models/bailingmoe3.cpp` (checkout at `e95426f`), which is what
the bartowski quants were made with.

## Model shape (`inclusionAI/Ling-3.2-air` config.json)

| Field | Value | Consequence |
| --- | --- | --- |
| `model_type` | `bailing_moe_v3_vl` (`text_config.model_type = bailing_hybrid`) | GGUF architecture key `bailingmoe3` |
| `num_hidden_layers` | 42 | trunk blocks 0–41, plus 1 MTP block (index 42 in the GGUF) |
| `hidden_size` | 2560 | residual width |
| `num_attention_heads` | 32, `kv_lora_rank` 512 | MLA blocks: 32 heads, KV compressed to a 512-wide latent |
| `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128 | | MLA q/k per head 192, v 128 |
| `num_experts` 512, `num_experts_per_tok` 8, `n_group` 8, `topk_group` 4 | | routed MoE with group-limited top-k |
| `moe_intermediate_size` 768, `moe_shared_expert_intermediate_size` 768 | | small experts; one always-on shared expert |
| `intermediate_size` 6144, `first_k_dense_replace` 2 | | blocks 0–1 dense |
| `vocab_size` 157184, `max_position_embeddings` 262144 | | 157k vocab, 256k context |
| `rope_theta` 6e6, `rotary_dim` 64, `partial_rotary_factor` 0.5 | | partial RoPE on the 64 rope dims only |
| `score_function` sigmoid, `routed_scaling_factor` 2.5, `norm_topk_prob` | | routing math |
| `moe_router_enable_expert_bias` | true | per-expert bias vector in the router |
| `linear_silu`, `kda_safe_gate`, `kda_lower_bound` −5.0, `short_conv_kernel_size` 4, `no_kda_lora` | | KDA block math |
| `layer_group_size` 6, `group_norm_size` 1 | | 1 MLA block per 6 blocks; the rest KDA |
| `num_nextn_predict_layers` 1 | | MTP draft layer, stored in the same GGUF |
| vision tower | `qwen3_moe_vit`, depth 27, hidden 1152, patch 16, merge 2, out 4608 | separate `mmproj` GGUF |
| quantization (original) | fp8 e4m3, 128×128 blocks | only relevant to the HF reference |

Effective arithmetic per decoded token is dominated by the routed experts:
8 of 512 experts × 3 matrices × 768 × 2560 per MoE block, 40 MoE blocks, plus
the shared expert and the MLA/KDA projections.

## Trunk structure (llama.cpp `bailingmoe3.cpp`, graph builder)

Per block `il`:

1. `attn_norm` (RMSNorm) → attention branch, chosen by `n_head_kv(il) == 0`:
   - **KDA block** (recurrent): `wq`/`wk`/`wv` projections → causal conv1d
     (kernel 4, state = last 3 rows) with SiLU → L2-normalize q,k → gated delta
     rule with a sigmoid gate `sigmoid(f_a(x) + dt_b) * ssm_a` scaled by
     `kda_lower_bound` and sigmoid beta from `ssm_beta` → RMSNorm per head and
     output gate `sigmoid(g_a(x))` → `wo`.
     Tensor names: `ssm_conv1d_q/k/v`, `ssm_f_a`, `ssm_beta`, `ssm_a`
     (`A_log`-style), `ssm_dt` bias, `ssm_g_a`, `ssm_norm`, `attn_output`.
   - **MLA block**: `wq` (192×32 heads) → split q_nope/q_pe, q_pe RoPE;
     `wkv_a_mqa` → 512 latent + 64 rope; `attn_kv_a_norm` on the latent; key
     absorb via `wk_b`, value absorb via `wv_b`; attention over the 512-latent
     plus shared rope key; output gate `sigmoid(wqkv_gate(x))` per head;
     `wo` (32×128 → 2560).
2. residual add → `ffn_norm`.
3. Dense FFN (`ffn_gate`/`ffn_up`/`ffn_down`, SiLU) for `il < 2`; otherwise
   routed MoE (sigmoid router with per-expert bias, group-limited top-4 groups,
   then top-8, weights normalized then ×2.5) plus the always-on shared expert,
   summed, then residual add.

Final `output_norm` → `output` (LM head; not tied).

## MTP block

One extra block at index 42 (`blk.42.*`), with `nextn.eh_proj` (concat of
token embedding norm and hidden-state norm, 2×2560 → 2560), the same MLA
attention as above, routed MoE + shared expert, and `nextn.shared_head_norm`
before the shared LM head. Drafting: feed the hidden state after `output_norm`
of the verified token plus the candidate token embedding.

## KDA state and the cache

- KDA blocks keep three conv states of `(d_conv − 1) × head_dim × n_head` and
  one recurrent state of `head_dim × head_dim × n_head` per session.
- MLA blocks keep a KV latent cache: 512 + 64 values per token per block.
- Sizes per token are far below a dense model of this width: the KV side is
  ~576 floats/token/block on MLA blocks only (7 of 42 blocks), so long context
  is cheap; the weights dominate.

## Where the time goes (measured 2026-11-25)

`tools/bandwidth_probe.hip` on this box (8 GiB, 3 runs, GPU visible in the
devshell): **read 237.0 GB/s, copy 204.3 GB/s, write 198.4 GB/s**. That read
number is the ceiling a decode step can reach.

`tools/roofline.py`, from the GGUF tensor table with 8 of 512 experts per
block, one decode step reads **4,753 MB**:

| component | MB/token | share |
| --- | --- | --- |
| attention projections (KDA+MLA) | 2,598 | 54.7 % |
| routed experts (8/512) | 1,266 | 26.6 % |
| LM head (Q8_0, 157k vocab) | 428 | 9.0 % |
| shared expert | 251 | 5.3 % |
| router (`ffn_gate_inp` F32, 5 MB × 40) | 210 | 4.4 % |
| norms | 1 | 0.0 % |

So the decode roofline is **≈ 50 tok/s** and the 60 % gate is ≈ 30 tok/s;
prefill's memory roofline at 2,048 tokens (every expert touched once) is
≈ 6,300 tok/s. Two entries are worth attacking beyond raw bandwidth use:

- the router is stored F32 (210 MB/token, 4.4 %): keeping it in bf16 at load
  removes half of that;
- the LM head is 9 % of every token: nothing to do at the same quant, but it
  means a 157k-vocab matvec is a first-order part of the design.

## Milestones

1. GGUF open + config parse + tensor map validation (host), against
   `blk.*` names, with the shapes asserted.
2. Tokenizer and chat template, byte-exact against the model card's template.
3. GPU forward, decode only: embeddings → per-block KDA/MLA → MoE → LM head,
   compared against a llama.cpp CPU reference token by token.
4. Prefill by reusing the same ops at larger row counts.
5. Component cache (append rows for MLA KV, private state for KDA conv/state),
   server path, cancellation.
6. MTP draft loop with exact-output verification.
7. Vision: `qwen3_moe_vit` encoder from the mmproj plus image preprocessing.
8. Roofline harness + optimization loop.

## Measurement plan

- `tools/bench_bandwidth`: stream/copy kernels over ~16 GiB to measure achieved
  read bandwidth on this box, three runs, median.
- `tools/roofline.py`: parse the GGUF tensor table, sum the bytes of the
  tensors a decode step reads given measured routing from the engine, print
  roofline tok/s and the measured percentage for each bench row.
- `gufo bench --model ... --n-prompt 2048 --n-gen 128 [--n-depth 32768]`, three
  runs each, medians, thinking off, greedy.
- Reference: llama.cpp CPU (`bailingmoe3`) for token equality; llama.cpp with
  the same quant on the same GPU for a speed reference once it builds here.
