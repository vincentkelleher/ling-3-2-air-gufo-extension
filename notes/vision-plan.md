# Vision plan: the mmproj encoder

Read from the file itself (`mmproj-Ling-3.2-air-bf16.gguf`, 878 MB, 334 tensors),
not from the model card.

## What the projector contains

| key | value |
| --- | --- |
| `general.architecture` | `clip` |
| `clip.projector_type` | `ling3vl` |
| `clip.vision.block_count` | 27 |
| `clip.vision.embedding_length` | 1152 |
| `clip.vision.attention.head_count` | 16 (so head_dim 72) |
| `clip.vision.feed_forward_length` | 4304 |
| `clip.vision.patch_size` | **16** |
| `clip.vision.spatial_merge_size` | **2** |
| `clip.vision.image_size` | 768 (48 patches per side → 2304 positions) |
| `clip.vision.projection_dim` | 2560 |
| `clip.vision.image_mean` / `image_std` | CLIP values (0.48145467, 0.4578275, 0.40821073 / 0.26862955, 0.26130259, 0.27577710) |
| `clip.use_gelu` | true |

Tensor families (334 tensors, all BF16 weights plus F32 biases/norms):

| tensor | count | shape |
| --- | --- | --- |
| `v.patch_embd.weight` and `v.patch_embd.weight.1` | 2 | `[16, 16, 3, 1152]` F32 |
| `v.position_embd.weight` | 1 | `[1152, 2304]` F32 |
| `v.blk.N.ln1/ln2.{weight,bias}` | 27 × 4 | `[1152]` |
| `v.blk.N.attn_qkv.{weight,bias}` | 27 | `[1152, 3456]` / `[3456]` |
| `v.blk.N.attn_out.{weight,bias}` | 27 | `[1152, 1152]` / `[1152]` |
| `v.blk.N.ffn_up` / `ffn_down` | 27 each | `[1152, 4304]` / `[4304, 1152]` |
| `mm.input_norm.{weight,bias}` | 1 | `[1152]` |
| `mm.0` / `mm.2` | 1 each | `[4608, 2560]` / `[2560, 2560]` |

**Two** patch embeddings is the detail worth noticing: `temporal_patch_size` is 2,
so the encoder always consumes a pair of frames, and a still image is duplicated.
The 4608 in `mm.0` is 1152 × 2 × 2, i.e. the `spatial_merge_size` of 2.

## Pipeline to implement

1. **Preprocess** per `BailingMM2ImageProcessor` (transformers remote code):
   resize within `min_pixels` 4096 / `max_pixels` 4194304, normalize with the CLIP
   mean/std above, and build patches of `16 × 16 × 3 × 2` (two frames). Emit the
   `<|vision_start|><|image_pad|><|vision_end|>` marker that the chat template
   already writes, with one pad token per **merged** patch (2×2 positions → one
   token), which is what the text side expects.
2. **Patch embed**: `x = patch_embd_0 · p0 + patch_embd_1 · p1 + pos_embd` on a
   48 × 48 grid; the second frame is a copy for a still image.
3. **27 ViT blocks**: `ln1 → qkv (3456 = 3 × 16 × 72) → attention → attn_out →
   residual → ln2 → ffn_up (4304, GELU) → ffn_down → residual`.
4. **Merge + projector**: 2 × 2 spatial concat (4608) → `input_norm` →
   `mm.0` (4608 → 2560) → GELU → `mm.2` (2560 → 2560). One output row per merged
   patch, which is the sequence that replaces the `<|image_pad|>` embeddings.
5. **Text side**: feed those rows as the embeddings of the image-pad tokens, and
   keep positions contiguous (the text model's `mrope_section` is
   `[8, 12, 12, 0]`, so image rows would need the multimodal position rule if the
   reference uses 2-D positions for them; verify against the HF processor rather
   than assume).

## Validation

- Compare one pinned image end to end with the HF reference: same number of pad
  tokens, then the same generated text at temperature 0.
- Compare the encoder in isolation first: patch-embed and a single ViT block against
  a numpy port on a tiny crop, as the text path was verified stage by stage.
- The `mmproj` file is already pinned in `models.lock.json` (878,174,208 bytes,
  verified by `gufo model verify`).

## Prerequisites and effort

- ~230 M parameters of BF16 vision weights, plus the projector: ~0.9 GB resident.
- Kernels needed: patch embed (a GEMM over 16×16×3×2 patches), LayerNorm (not RMS),
  GELU, attention over the patch sequence, and the projector GEMMs. The text side
  already has every other building block.
- This is the second-largest remaining piece after the runner; it depends on
  nothing else, so it can be built in parallel with the runner once the serving
  path exists (an image can only be exercised through the server).

## Vision encoder: core verified against numpy (2026-11-27)

The mmproj is BF16, which the text kernels do not implement, so `VisionEncoder`
dequantizes each tensor it needs to F32 on the host and uploads it (27 blocks =
1.54 GiB of device weights). Kernels: `MatvecInMajor*` (the mmproj stores weights
`[in][out]`, the transpose of the text GEMV), `LayerNormKernel` (weight+bias),
`GeluKernel` (tanh form, as `clip.use_gelu` records), `VisionAttention` and `Add`.

`tools/make_vision_fixture.py` writes a reference from the real weights (numpy,
F32), and `tests/vision_test.cpp` checks the encoder against it:

| stage | max abs error | limit |
| --- | --- | --- |
| patch embed (two 768-float patches, 4608→1152) | 5.0e-06 | 1e-3 |
| encoder block 0 (LN → qkv → attention → out → residual → LN → ffn → residual) | 3.4e-05 | 2e-3 |

Two things this established:

- The vision tower's attention is **bidirectional**, not causal. Attending causally
  gave a 1.13 error against reference magnitudes of 11.2; bidirectional gives
  3.4e-05. Worth recording because the text decoder's kernels are all causal and
  reusing one silently produces plausible-but-wrong image features.
- BF16→F32 at load is sufficient: no BF16 kernels are needed for the tower, at the
  cost of 1.54 GiB of device weights (the text weights are zero-copy and untouched).

Not yet done: position embeddings with dynamic-resolution interpolation, the
spatial merge (2x2 → 576 tokens for a 768x768 image), `v.post_ln` and the `mm.*`
projector, the image-preprocessing path, and wiring image embeddings into the text
prompt. `PatchEmbed` and `RunBlock` are the validated primitives those build on.
