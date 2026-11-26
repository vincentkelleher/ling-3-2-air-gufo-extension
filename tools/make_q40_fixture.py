#!/usr/bin/env python3
"""Write a Q4_0 dot fixture from real block-42 weights.

The MTP layer is Q4_0 and neither Gufo's reference nor any existing test decodes
that type, so the kernel is validated against this independent reference instead:
real rows of blk.42.attn_q, dequantized by gguf.quants, dotted with a fixed input.

    tools/make_q40_fixture.py --out /work/q40_fixture.bin

Layout: int32 rows, cols, row_bytes; then the raw Q4_0 rows, then `cols` floats of
input, then `rows` floats of expected dots.
"""
import argparse
import glob
import struct

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT
from gguf.quants import dequantize

DEFAULT = ('/home/ai-box/Models/bartowski/Ling-3.2-air-GGUF/Ling-3.2-air-Q4_K_M/'
           'Ling-3.2-air-Q4_K_M-0000%d-of-00003.gguf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-glob', default=DEFAULT)
    parser.add_argument('--tensor', default='blk.42.attn_q.weight')
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--out', default='/work/q40_fixture.bin')
    args = parser.parse_args()

    tensor = reader = None
    for path in sorted(glob.glob(args.model_glob % 1) + glob.glob(args.model_glob % 2)
                       + glob.glob(args.model_glob % 3)):
        reader = GGUFReader(path)
        tensor = next((t for t in reader.tensors if t.name == args.tensor), None)
        if tensor is not None:
            break
    if tensor is None:
        raise SystemExit(f'{args.tensor} not found')
    if tensor.tensor_type != QT.Q4_0:
        raise SystemExit(f'{args.tensor} is {tensor.tensor_type.name}, not Q4_0')

    raw = np.asarray(tensor.data).view(np.uint8).reshape(-1)
    cols = int(tensor.shape[0])
    row_bytes = cols // 32 * 18
    rows = args.rows
    weights = np.concatenate([raw[i * row_bytes:(i + 1) * row_bytes] for i in range(rows)])
    matrix = np.stack([dequantize(weights[i * row_bytes:(i + 1) * row_bytes], QT.Q4_0)
                       .astype(np.float32) for i in range(rows)])
    x = (np.sin(np.arange(cols) * 0.01) * 0.5).astype(np.float32)
    expected = (matrix @ x).astype(np.float32)

    with open(args.out, 'wb') as handle:
        handle.write(struct.pack('<iii', rows, cols, row_bytes))
        handle.write(weights.tobytes())
        handle.write(x.tobytes())
        handle.write(expected.tobytes())
    print(f'wrote {args.out}: {rows} rows x {cols} cols, expected {expected}')


if __name__ == '__main__':
    main()
