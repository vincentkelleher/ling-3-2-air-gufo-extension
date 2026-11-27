#!/usr/bin/env python3
"""Write a vision-encoder fixture from the real mmproj weights.

The mmproj is BF16, which the engine's kernels do not implement, so the plan is to
dequantize it to F32 at load and reuse the F32 paths. This script produces the
reference for that: a synthetic patch, the patch-embed output, and one full encoder
block (LayerNorm -> qkv -> attention -> out projection -> residual -> LayerNorm ->
ffn up/GELU/down -> residual) computed here in numpy from the same weights.

    tools/make_vision_fixture.py --out /work/vision_fixture.bin

Layout (little endian):
    int32 tokens, dim, heads, head_dim, ffn, blocks_validated
    float32 patch[tokens][dim]      (input, after the two patch embeddings)
    float32 embed[dim]              (patch-embed output for token 0)
    float32 x[tokens][dim]          (one encoder block's input)
    float32 block[tokens][dim]      (that block's output for all tokens)
    int32 grid, grid                (small image, grid x grid patches)
    float32 image[3][grid*16][grid*16]
    float32 embedding[grid*grid/4][text_dim]
    int32 trace_count, then trace_count x (int32 block, float32 h[grid*grid][dim])
"""

import argparse
import struct

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT
from gguf.quants import dequantize

DEFAULT = ('models/Ling-3.2-air-GGUF/'
           'mmproj-Ling-3.2-air-bf16.gguf')


def tensor(reader, name):
    found = next((t for t in reader.tensors if t.name == name), None)
    if found is None:
        raise SystemExit(f'missing tensor {name}')
    raw = np.asarray(found.data).view(np.uint8).reshape(-1)
    if found.tensor_type == QT.BF16:
        # BF16 is the upper half of an FP32; shifting keeps the reference exact.
        values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif found.tensor_type == QT.F32:
        values = raw.view(np.float32)
    else:
        values = dequantize(raw, found.tensor_type).astype(np.float32)
    shape = [int(x) for x in found.shape]
    return values.reshape(shape) if shape else values


def matrix(reader, name, out_dim, in_dim):
    """A weight as [out][in]. ggml stores a tensor's slowest dimension last, so a
    weight of shape [in, out] lies in memory as [out][in] (one row per output)."""
    values = tensor(reader, name)
    if values.size != out_dim * in_dim:
        raise SystemExit(f'{name}: {values.size} values, expected {out_dim}x{in_dim}')
    return np.asarray(values, dtype=np.float64).reshape(out_dim, in_dim)


def layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def vision_rope(q_heads_kv, rows, cols, head_dim=72):
    """llama.cpp's vision M-RoPE: pairs (i, i+half); row position for the first
    section, column for the next; frequency 10000^(-2*index/half) reset per section."""
    half = head_dim // 2
    section = half // 2
    for t in range(rows):
        row, col = divmod(t, cols)
        for i in range(half):
            position = row if i < section else col
            index = i if i < section else i - section
            angle = position * (10000.0 ** (-2.0 * index / half))
            c, s = np.cos(angle), np.sin(angle)
            for tensor in q_heads_kv:
                # Layout is (heads, tokens, head_dim): rotate one token across all
                # heads. Copies, because views would be updated in place and the
                # second output would use the rotated values instead of the originals.
                lo = tensor[:, t, i].copy()
                hi = tensor[:, t, i + half].copy()
                tensor[:, t, i] = lo * c - hi * s
                tensor[:, t, i + half] = lo * s + hi * c


def gelu(x):
    # tanh approximation, as clip.use_gelu records for the vision tower
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT)
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261127)
    parser.add_argument('--out', default='/work/vision_fixture.bin')
    args = parser.parse_args()

    reader = GGUFReader(args.model)
    dim = int(reader.fields['clip.vision.embedding_length'].contents())
    heads = int(reader.fields['clip.vision.attention.head_count'].contents())
    head_dim = dim // heads
    ffn = int(reader.fields['clip.vision.feed_forward_length'].contents())
    eps = float(reader.fields['clip.vision.attention.layer_norm_epsilon'].contents())
    tokens = args.tokens

    rng = np.random.default_rng(args.seed)
    x0 = rng.standard_normal((tokens, dim)).astype(np.float32)  # block input
    x = x0.copy()

    # Patch embed: two 16x16x3 patches (temporal pair) through one 4608->1152 GEMM.
    w0 = tensor(reader, 'v.patch_embd.weight')      # [16,16,3,1152]
    w1 = tensor(reader, 'v.patch_embd.weight.1')
    bias = tensor(reader, 'v.patch_embd.bias')      # [1152]
    patch0 = rng.standard_normal(16 * 16 * 3).astype(np.float32)
    patch1 = rng.standard_normal(16 * 16 * 3).astype(np.float32)
    flat0 = w0.reshape(dim, 768)
    flat1 = w1.reshape(dim, 768)
    embed = patch0 @ flat0.T + patch1 @ flat1.T + bias

    # One encoder block on `tokens` patch rows (position embeddings omitted here).
    ln1w, ln1b = tensor(reader, 'v.blk.0.ln1.weight'), tensor(reader, 'v.blk.0.ln1.bias')
    ln2w, ln2b = tensor(reader, 'v.blk.0.ln2.weight'), tensor(reader, 'v.blk.0.ln2.bias')
    qkv_w, qkv_b = matrix(reader, 'v.blk.0.attn_qkv.weight', 3 * dim, dim), tensor(reader, 'v.blk.0.attn_qkv.bias')
    out_w, out_b = matrix(reader, 'v.blk.0.attn_out.weight', dim, dim), tensor(reader, 'v.blk.0.attn_out.bias')
    up_w, up_b = matrix(reader, 'v.blk.0.ffn_up.weight', ffn, dim), tensor(reader, 'v.blk.0.ffn_up.bias')
    down_w, down_b = matrix(reader, 'v.blk.0.ffn_down.weight', dim, ffn), tensor(reader, 'v.blk.0.ffn_down.bias')

    h = layer_norm(x, ln1w, ln1b, eps)
    qkv = h @ qkv_w.T + qkv_b                    # [tokens, 3*dim]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    vision_rope([q, k], tokens, tokens, head_dim)
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attn = (weights @ v).transpose(1, 0, 2).reshape(tokens, dim)
    x = x + attn @ out_w.T + out_b

    h = layer_norm(x, ln2w, ln2b, eps)
    x = x + gelu(h @ up_w.T + up_b) @ down_w.T + down_b

    # Full image path on a small grid (2x2 patches = a 32x32 image): patchify with a
    # duplicated temporal frame, cropped position embeddings, every block, the 2x2
    # merge, the input norm and both projector matrices.
    grid = 2
    side = grid * 16
    image = rng.standard_normal((3, side, side)).astype(np.float32)
    patch_dim = dim
    rows = grid * grid
    pos = tensor(reader, 'v.position_embd.weight').reshape(-1, dim)   # [2304][1152]
    patches = np.zeros((rows, 2 * 768), dtype=np.float32)
    for gh in range(grid):
        for gw in range(grid):
            row = gh * grid + gw
            block_px = image[:, gh * 16:(gh + 1) * 16, gw * 16:(gw + 1) * 16].reshape(-1)
            patches[row, :768] = block_px
            patches[row, 768:] = block_px
    w0 = tensor(reader, 'v.patch_embd.weight').reshape(dim, 768)
    w1 = tensor(reader, 'v.patch_embd.weight.1').reshape(dim, 768)
    h = patches[:, :768] @ w0.T + patches[:, 768:] @ w1.T + bias
    embed_img0 = h[0].copy()

    def resized_pos(row, col):
        """Bilinear, corner-aligned resample of the 48x48 grid to the patch grid, as
        llama.cpp's UPSCALE does (identity for the model's own 48x48)."""
        grid_in = 48
        sy = row * (grid_in - 1) / (grid - 1) if grid > 1 else 0.0
        sx = col * (grid_in - 1) / (grid - 1) if grid > 1 else 0.0
        y0, x0 = int(sy), int(sx)
        y1, x1 = min(y0 + 1, grid_in - 1), min(x0 + 1, grid_in - 1)
        fy, fx = sy - y0, sx - x0
        at = lambda y, x: pos[y * grid_in + x]
        return ((at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) +
                (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy)

    for t in range(rows):
        h[t] += resized_pos(t // grid, t % grid)
    before_blocks = h.copy()
    traces = []
    nb = int(reader.fields['clip.vision.block_count'].contents())
    for b in range(nb):
        prefix = f'v.blk.{b}.'
        ln1w, ln1b = tensor(reader, prefix + 'ln1.weight'), tensor(reader, prefix + 'ln1.bias')
        ln2w, ln2b = tensor(reader, prefix + 'ln2.weight'), tensor(reader, prefix + 'ln2.bias')
        qkv_w, qkv_b = matrix(reader, prefix + 'attn_qkv.weight', 3 * dim, dim), tensor(reader, prefix + 'attn_qkv.bias')
        out_w, out_b = matrix(reader, prefix + 'attn_out.weight', dim, dim), tensor(reader, prefix + 'attn_out.bias')
        up_w, up_b = matrix(reader, prefix + 'ffn_up.weight', ffn, dim), tensor(reader, prefix + 'ffn_up.bias')
        down_w, down_b = matrix(reader, prefix + 'ffn_down.weight', dim, ffn), tensor(reader, prefix + 'ffn_down.bias')
        hh = layer_norm(h, ln1w, ln1b, eps)
        qkv = hh @ qkv_w.T + qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(rows, heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(rows, heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(rows, heads, head_dim).transpose(1, 0, 2)
        vision_rope([q, k], rows, grid, head_dim)
        sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)
        w = np.exp(sc - sc.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        attn = (w @ v).transpose(1, 0, 2).reshape(rows, dim)
        h = h + attn @ out_w.T + out_b
        hh = layer_norm(h, ln2w, ln2b, eps)
        h = h + gelu(hh @ up_w.T + up_b) @ down_w.T + down_b
        if b in (0, 1, 2, 3, nb - 1):
            traces.append((b, h.copy()))
    tower_out = h.copy()
    norm_w, norm_b = tensor(reader, 'mm.input_norm.weight'), tensor(reader, 'mm.input_norm.bias')
    # mm.input_norm is `dim`-wide: it normalises each patch before the 2x2 merge.
    h = layer_norm(h, norm_w, norm_b, eps)
    merged = np.concatenate([h[0], h[1], h[grid], h[grid + 1]]).astype(np.float32)
    text_dim = int(reader.fields['clip.vision.projection_dim'].contents()) if 'clip.vision.projection_dim' in reader.fields else 2560
    mm0w, mm0b = matrix(reader, 'mm.0.weight', text_dim, 4 * dim), tensor(reader, 'mm.0.bias')
    mm2w, mm2b = matrix(reader, 'mm.2.weight', text_dim, text_dim), tensor(reader, 'mm.2.bias')
    embedding = gelu(merged @ mm0w.T + mm0b) @ mm2w.T + mm2b

    with open(args.out, 'wb') as handle:
        handle.write(struct.pack('<iiiiii', tokens, dim, heads, head_dim, ffn, 1))
        handle.write(patch0.astype(np.float32).tobytes())
        handle.write(patch1.astype(np.float32).tobytes())
        handle.write(embed.astype(np.float32).tobytes())
        handle.write(x0.astype(np.float32).tobytes())
        handle.write(x.astype(np.float32).tobytes())
        handle.write(struct.pack('<ii', grid, grid))
        handle.write(image.astype(np.float32).tobytes())
        handle.write(embed_img0.astype(np.float32).tobytes())
        handle.write(before_blocks.astype(np.float32).tobytes())
        handle.write(tower_out.astype(np.float32).tobytes())
        handle.write(embedding.astype(np.float32).tobytes())
        handle.write(struct.pack('<i', len(traces)))
        for block_index, values in traces:
            handle.write(struct.pack('<i', block_index))
            handle.write(values.astype(np.float32).tobytes())

    for block_index, values in traces:
        print(f'  trace block {block_index}: rms {np.sqrt((values ** 2).mean()):.4f}')
    print(f'image path: {grid}x{grid} patches, embedding dim {embedding.shape[0]}, '
          f'rms {np.sqrt((embedding ** 2).mean()):.4f}')
    print(f'wrote {args.out}: {tokens} tokens, dim {dim}, {heads} heads x {head_dim}, '
          f'ffn {ffn}; block-0 output rms {np.sqrt((x ** 2).mean()):.4f}')


if __name__ == '__main__':
    main()
