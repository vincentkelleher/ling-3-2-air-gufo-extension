#!/usr/bin/env python3
"""Build a derived artifact directory: every Q8_0 tensor re-quantised to Q6_K.

Q8_0 is 70.7 % of a decode token's 4.753 GB of traffic, so shrinking those tensors is the only way
past the 49.9 tok/s memory ceiling (4.753 GB / 237 GB/s). Each tensor is written as
`<dir>/<tensor>.bin` by `tools/requantise.cpp`, which is what `LING3_DERIVED=<dir>` picks up.

    python3 tools/prepare_derived.py MODEL.gguf OUTDIR REQUANTISE_BINARY
"""
import argparse
import glob
import os
import struct
import subprocess
import sys

Q8_0 = 8


def tensor_table(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    f.read(4)
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rd_str():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def skip(kind):
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind == 8:
            rd_str()
        elif kind == 9:
            element = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(count):
                skip(element)
        else:
            f.read(fixed[kind])

    for _ in range(n_kv):
        rd_str()
        skip(struct.unpack("<I", f.read(4))[0])
    out = {}
    for _ in range(n_tensors):
        name = rd_str()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        f.read(8)
        count = 1
        for d in dims:
            count *= d
        out[name] = (ttype, count, dims)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="first shard of the pinned model")
    parser.add_argument("outdir")
    parser.add_argument("requantise", help="the built tools/requantise binary")
    parser.add_argument("--min-values", type=int, default=1 << 20)
    args = parser.parse_args()

    directory = os.path.dirname(args.model)
    first = os.path.basename(args.model)
    prefix, _, rest = first.partition("-00001-of-")
    total = rest.split(".")[0] if rest else "00001"
    shards = [args.model]
    if rest:
        for index in range(2, int(total) + 1):
            shards.append(os.path.join(directory, f"{prefix}-{index:05d}-of-{total}.gguf"))

    table = {}
    for shard in shards:
        table.update(tensor_table(shard))
    os.makedirs(args.outdir, exist_ok=True)

    selected = [(name, count) for name, (ttype, count, _) in table.items()
                if ttype == Q8_0 and count >= args.min_values]
    selected.sort(key=lambda item: -item[1])
    print(f"{len(selected)} Q8_0 tensors, {sum(c for _, c in selected) / 1e6:.0f}M weights")
    saved = 0
    for name, _ in selected:
        target = None
        for shard in shards:
            result = subprocess.run([args.requantise, shard, os.path.join(args.outdir, name + ".bin"),
                                     name, "q6_K"], capture_output=True, text=True)
            if result.returncode == 0:
                target = result.stdout.strip()
                break
        if target is None:
            print(f"  FAILED {name}")
            continue
        before = os.path.getsize(os.path.join(args.outdir, name + ".bin"))
        print(f"  {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
