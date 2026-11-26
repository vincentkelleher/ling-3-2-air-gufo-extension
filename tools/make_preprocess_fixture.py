#!/usr/bin/env python3
"""Fixture for the image preprocessing (resize + CLIP normalisation).

Writes, little endian:
    int32 width, height, target
    uint8 rgb[height*width*3]          (packed RGB8, the tower's input order)
    float32 numpy_reference[3*target*target]   (same double-precision formula)
    float32 pil_reference[3*target*target]     (PIL BILINEAR on uint8, then normalise)

The numpy reference checks the arithmetic; the PIL one checks that the sampling
convention matches the library the HF processors use, which is why it is compared with
a looser tolerance (PIL rounds to 8 bits while it resizes).
"""

import argparse
import struct

import numpy as np
from PIL import Image

MEAN = np.array([0.48145467042922974, 0.45782750844955444, 0.40821072459220886], np.float64)
STD = np.array([0.2686295509338379, 0.2613025903701782, 0.27577710151672363], np.float64)


def resize_numpy(image, target):
    """Centre-aligned bilinear resize, in double precision."""
    height, width, _ = image.shape
    scale_x = width / target
    scale_y = height / target
    out = np.zeros((target, target, 3), np.float64)
    for oy in range(target):
        cy = (oy + 0.5) * scale_y - 0.5
        y0 = int(np.floor(cy))
        fy = cy - y0
        if y0 < 0:
            y0, fy = 0, 0.0
        elif y0 >= height - 1:
            y0, fy = height - 1, 0.0
        y1 = min(y0 + 1, height - 1)
        for ox in range(target):
            cx = (ox + 0.5) * scale_x - 0.5
            x0 = int(np.floor(cx))
            fx = cx - x0
            if x0 < 0:
                x0, fx = 0, 0.0
            elif x0 >= width - 1:
                x0, fx = width - 1, 0.0
            x1 = min(x0 + 1, width - 1)
            top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
            bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
            out[oy, ox] = top * (1 - fy) + bottom * fy
    return out


def to_chw(values):
    chw = values.transpose(2, 0, 1)
    return ((chw / 255.0 - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=37)
    parser.add_argument('--height', type=int, default=53)
    parser.add_argument('--target', type=int, default=768)
    parser.add_argument('--seed', type=int, default=20261128)
    parser.add_argument('--out', default='preprocess_fixture.bin')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rgb = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)

    reference = to_chw(resize_numpy(rgb.astype(np.float64), args.target))
    pillowed = np.asarray(
        Image.fromarray(rgb).resize((args.target, args.target), Image.BILINEAR),
        dtype=np.float64,
    )
    pil = to_chw(pillowed)

    with open(args.out, 'wb') as handle:
        handle.write(struct.pack('<iii', args.width, args.height, args.target))
        handle.write(rgb.astype(np.uint8).tobytes())
        handle.write(reference.astype(np.float32).tobytes())
        handle.write(pil.astype(np.float32).tobytes())

    delta = np.abs(reference - pil)
    print(f'wrote {args.out}: {args.width}x{args.height} -> {args.target}, '
          f'numpy vs PIL max {delta.max():.4f}, mean {delta.mean():.5f}')


if __name__ == '__main__':
    main()
