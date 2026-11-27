#!/usr/bin/env python3
"""Bytes read per decode token by this engine, computed from the GGUF tensor table.

The roofline model assumes 4.75 GB/token (237 GB/s / 49.9 tok/s). If the engine's actual
per-token traffic is larger, the gate is unreachable by tuning and the model is wrong.
"""
import glob
import struct

GGUF = '/models/bartowski/Ling-3.2-air-GGUF/Ling-3.2-air-Q4_K_M/Ling-3.2-air-Q4_K_M-0000*-of-00003.gguf'
paths = sorted(glob.glob(GGUF))

# bits per weight for the types this quantization uses
BITS = {0: 32.0, 1: 16.0, 2: 4.5, 3: 5.0, 6: 5.5, 7: 6.5, 8: 8.5, 12: 4.5, 13: 5.5, 14: 6.5625}


def parse(fn):
    f = open(fn, 'rb')
    f.read(8)  # magic + version
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    def rs():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def skip(t):
        size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rs()
        elif t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            for _ in range(n):
                skip(et)
        else:
            f.read(size[t])

    for _ in range(n_kv):
        rs()
        t = struct.unpack('<I', f.read(4))[0]
        skip(t)
    out = {}
    for _ in range(n_tensors):
        name = rs()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        ttype = struct.unpack('<I', f.read(4))[0]
        f.read(8)
        n = 1
        for d in dims:
            n *= d
        out[name] = (n * BITS.get(ttype, 32.0) / 8.0, ttype, dims)
    return out


tensors = {}
for p in paths:
    tensors.update(parse(p))

N_LAYER = 42
TOP_K = 8
total = 0.0
per_block = {'kda': 0.0, 'mla': 0.0, 'dense': 0.0, 'moe': 0.0, 'head': 0.0, 'other': 0.0}

for il in range(N_LAYER):
    prefix = f'blk.{il}.'
    for name, (b, ttype, dims) in tensors.items():
        if not name.startswith(prefix):
            continue
        leaf = name[len(prefix):]
        if 'exps' in leaf:
            # routed experts: only TOP_K of the 512 are read per token
            per_block['moe'] += b * TOP_K / dims[2]
        elif any(k in leaf for k in ('kda_', 'ssm_')):
            per_block['kda'] += b
        elif 'mla' in leaf or 'attn' in leaf:
            per_block['mla'] += b
        else:
            per_block['dense'] += b  # shared expert and norms

for name, (b, ttype, dims) in tensors.items():
    if name.startswith('blk.'):
        continue
    if name == 'output.weight':
        per_block['head'] += b
    elif name == 'token_embd.weight':
        # one row per token
        per_block['other'] += b / dims[1]

total = sum(per_block.values())
print('bytes read per decode token, by group:')
for key, value in per_block.items():
    print(f'  {key:6s} {value / 1e9:7.3f} GB')
print(f'  TOTAL  {total / 1e9:7.3f} GB per token')
print()
BW = 237.0e9
print(f'bandwidth-bound ceiling at {BW / 1e9:.0f} GB/s: {BW / total:.1f} tok/s')
print(f'roofline.py assumes 4.753 GB/token -> 49.9 tok/s')
print(f'measured decode 16.9 tok/s implies {BW / 16.9 / 1e9:.2f} GB/token if purely bandwidth-bound')
