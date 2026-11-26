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
"""

import argparse
import struct

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT
from gguf.quants import dequantize

DEFAULT = ('/home/ai-box/Models/bartowski/Ling-3.2-air-GGUF/'
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


def layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


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
    flat0 = w0.reshape(-1, dim)
    flat1 = w1.reshape(-1, dim)
    embed = patch0 @ flat0 + patch1 @ flat1 + bias

    # One encoder block on `tokens` patch rows (position embeddings omitted here).
    ln1w, ln1b = tensor(reader, 'v.blk.0.ln1.weight'), tensor(reader, 'v.blk.0.ln1.bias')
    ln2w, ln2b = tensor(reader, 'v.blk.0.ln2.weight'), tensor(reader, 'v.blk.0.ln2.bias')
    qkv_w, qkv_b = tensor(reader, 'v.blk.0.attn_qkv.weight'), tensor(reader, 'v.blk.0.attn_qkv.bias')
    out_w, out_b = tensor(reader, 'v.blk.0.attn_out.weight'), tensor(reader, 'v.blk.0.attn_out.bias')
    up_w, up_b = tensor(reader, 'v.blk.0.ffn_up.weight'), tensor(reader, 'v.blk.0.ffn_up.bias')
    down_w, down_b = tensor(reader, 'v.blk.0.ffn_down.weight'), tensor(reader, 'v.blk.0.ffn_down.bias')

    h = layer_norm(x, ln1w, ln1b, eps)
    qkv = h @ qkv_w + qkv_b                      # [tokens, 3*dim]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attn = (weights @ v).transpose(1, 0, 2).reshape(tokens, dim)
    x = x + attn @ out_w + out_b

    h = layer_norm(x, ln2w, ln2b, eps)
    x = x + gelu(h @ up_w + up_b) @ down_w + down_b

    with open(args.out, 'wb') as handle:
        handle.write(struct.pack('<iiiiii', tokens, dim, heads, head_dim, ffn, 1))
        handle.write(patch0.astype(np.float32).tobytes())
        handle.write(patch1.astype(np.float32).tobytes())
        handle.write(embed.astype(np.float32).tobytes())
        handle.write(x0.astype(np.float32).tobytes())
        handle.write(x.astype(np.float32).tobytes())

    print(f'wrote {args.out}: {tokens} tokens, dim {dim}, {heads} heads x {head_dim}, '
          f'ffn {ffn}; block-0 output rms {np.sqrt((x ** 2).mean()):.4f}')


if __name__ == '__main__':
    main()
