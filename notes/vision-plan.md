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

## Full image path now verified (2026-11-27)

`VisionEncoder::EncodeImage` does the whole chain and matches the numpy reference
from the real mmproj weights:

| stage | max abs error | limit |
| --- | --- | --- |
| patch embed of image patch 0 | 1.7e-06 | 2e-3 |
| patch embed + position embeddings (2x2 grid) | 1.4e-06 | 2e-3 |
| trace after blocks 0 / 1 / 2 / 3 / 26 | 2.0e-05 … 9.2e-05 | 2e-3 |
| tower output after all 27 blocks | 9.2e-05 | 5e-3 |
| final embedding (768x768-equivalent projector) | 1.6e-06 | 5e-3 |

The shape of the pipeline, settled from the mmproj rather than guessed:

1. **Fixed 768x768 input.** `clip.vision.image_size` is 768, patch 16, so the patch
   grid is 48x48 = 2304, exactly the number of stored position embeddings. There is
   no dynamic resolution and no interpolation to implement — smaller grids (only
   used by the test) take a row-major crop of the 48x48 grid.
2. **Two temporal halves.** `v.patch_embd.weight` and `.weight.1` each map 768
   pixels to 1152; a still image feeds the same pixels to both. The two halves must
   be contiguous blocks in the patch buffer, because they go to two batched matvecs
   — interleaving them per row silently produced a half-magnitude patch embed.
3. **`mm.input_norm` is 1152-wide**, so it normalises each patch *before* the 2x2
   merge (4608 = 4x1152), not the merged vector.
4. Merge order is (row, row+1) x (col, col+1) → `dy*2+dx`, then `mm.0` (4608→2560),
   tanh GELU, `mm.2` (2560→2560). `text_dim` is 2560, the text model's embedding
   width.

Three bugs this test caught, all of which produced plausible-looking numbers:

- capturing the tower output *after* the in-place input norm, which made every
  traced stage look wrong and hid the real one;
- the strided second temporal half described above;
- comparing a pre-norm value against a post-position-embed value (an invalid
  reference in my own test).

Not done yet: resize/normalise from a decoded image (PIL-equivalent resampling to
768x768 with the mmproj's mean/std), the `<image>` token plumbing into the text
prompt, and parity against the HF `BailingMoeV3VLForConditionalGeneration` tower
(criterion 3) — that needs torch + transformers in the venv, which are not
installed; the numpy reference from the same weights is what stands behind the
numbers above.

## Two bugs that made the tower wrong while looking right (2026-11-27)

The numpy reference agreed with the kernel at every stage while both produced
garbage descriptions. Agreement between two implementations that share an assumption
proves nothing; `llama-mtmd-debug -p encode` (which prints per-node values and sums)
was the reference that broke the tie.

1. **Weight layout.** ggml stores a tensor's slowest dimension last, so a weight of
   shape `[in, out]` lies in memory as `[out][in]` — one row per output, the same
   layout as the text GEMV. The vision kernels used the transpose. The tell was that
   *sums* matched exactly while *values* did not: with a uniform input, a transposed
   matvec preserves the sum and permutes the outputs. After the fix, block 0's stages
   match llama.cpp to four decimals (qkv -161.037 vs -161.062, attention 30.925 vs
   30.905, residual 115.113 vs 115.105).
2. **A rope race.** `VisionRopeKernel` launched 64 threads but looped over the rotated
   dims serially, so every thread re-applied the same rotation: the result depended on
   scheduling. It now splits the loop across threads.

Also settled against llama.cpp's graph rather than by reasoning:

- Position embeddings are **resampled** with bilinear sampling and aligned corners
  (`ggml_upscale`), not cropped. For the model's own 48x48 grid that is the identity,
  so production images pay nothing, but any other grid needs it.
- A still image runs **both** patch kernels on the same frame and sums them
  (`build_inp_temporal_merge` with `n_batch == 1`), which is what we do.
- Vision M-RoPE pairs `(i, i + head_dim/2)` inside each head, with the patch's row
  position for the first `head_dim/4` pairs and its column for the next, frequency
  `10000^(-2 i / (head_dim/2))` reset per section.

Numerical agreement with llama.cpp on a constant-gray 32x32 input (2x2 patches):

| quantity | ours | llama.cpp |
| --- | --- | --- |
| patch embed + bias | 170.750095 | 170.750427 |
| + position embeddings | 137.946183 | 137.946426 |
| block 0 qkv (with bias) | -161.037051 | -161.062393 |
| block 0 attention | 30.925476 | 30.905478 |
| block 0 residual | 115.112945 | 115.105118 |
| final embedding | -3.618567 | -3.619933 |
| first three values | 0.0071 -0.0857 0.0462 | 0.0064 -0.0860 0.0460 |
| last three values | -0.0011 -0.0171 0.0724 | -0.0014 -0.0160 0.0720 |

The remaining difference is llama.cpp's flash attention running K/V through f16.

Behaviourally, on a red circle with a blue square and small text, this extension now
answers: "there's a large red circle, and inside it, centered, is a blue square. In
the upper-left corner, there's small text" — where llama.cpp says "a large, solid red
circle". Both are correct; ours names the square and the text too.

## CLI image flag: open question (2026-11-27)

`gufo prompt --image` does not reach the driver's image handling, while
`tests/image_forward_test.cpp` proves the same helpers work. Facts collected:

- `gufo prompt --help` lists `--image`; the CLI stores it in `options.image_paths`
  (verified in Gufo's `src/cli/prompt/prompt.cpp`), and it accepts a nonexistent path
  without complaint, so parsing happens.
- The driver entry `RunLing3Prompt` calls `RunPromptLike`, and a temporary
  `fprintf(stderr, ...)` at the top of `RunPromptLike` never printed, even though the
  freshly installed module contains that string and `gufo diagnose` reports that exact
  `.so` path as loaded.
- The plain prompt path through the same function prints its own timing line, so the
  driver is being called.

The next step for whoever picks this up: run the same probe inside `RunLing3Prompt` and
`RunLing3Chat` separately, and check whether `gufo extension install` is loading a
different module than `diagnose` reports (for example a build-cache copy inside
`/work/ext-build`). Until then the flag is documented as unverified.
