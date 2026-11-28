#!/usr/bin/env python3
"""Re-quantise one tensor of the pinned GGUF to Q4_0, as a derived artifact.

A decode token reads 4.753 GB and this box sustains 237 GB/s, so the memory ceiling is 49.9 tok/s.
Reading fewer bytes is the only way past it, and Q8_0 tensors cost 8.5 bits per weight against Q4_0's
4.5. This tool produces the derived tensor and reports the reconstruction error it costs, so the
trade-off is measured before anything loads it.

    python3 tools/requantise_q4_0.py MODEL.gguf OUTPUT.bin TENSOR_NAME [--report]

The output is the tensor's Q4_0 blocks in ggml layout (block = fp16 scale + 16 packed nibbles, 18
bytes per 32 values) plus a sidecar .json with the name, shape and error figures.
"""
import argparse
import json
import struct

GGUF_BLOCK = {0: (4, 1), 1: (2, 1), 8: (34, 32), 12: (144, 256), 14: (210, 256)}


def read_gguf(path):
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
    tensors = {}
    for _ in range(n_tensors):
        name = rd_str()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        tensors[name] = {"dims": dims, "type": ttype, "offset": offset}
    alignment = 32
    data_start = (f.tell() + alignment - 1) // alignment * alignment
    return f, tensors, data_start


def fp16(raw):
    """IEEE half -> float, without numpy."""
    return struct.unpack("<e", raw)[0]


def dequant_q8_0(data, count):
    out = []
    blocks = count // 32
    for b in range(blocks):
        base = b * 34
        d = fp16(data[base:base + 2])
        out.extend(d * struct.unpack("<b", data[base + 2 + i:base + 3 + i])[0] for i in range(32))
    return out


def quantise_q4_0(values):
    """ggml Q4_0: one fp16 scale per 32 values, d = max / -8, nibbles in [0, 15]."""
    out = bytearray()
    for base in range(0, len(values), 32):
        block = values[base:base + 32]
        amax = max(abs(v) for v in block)
        d = amax / -8.0 if amax else 0.0
        if d == 0.0:
            out += struct.pack("<e", 0.0)
            out += b"\x00" * 16
            continue
        nibbles = []
        for v in block:
            q = int(round(v / d)) + 8
            nibbles.append(min(15, max(0, q)))
        out += struct.pack("<e", d)
        for i in range(0, 32, 2):
            out.append(nibbles[i] | (nibbles[i + 1] << 4))
    return bytes(out)


def dequant_q4_0(data, count):
    out = []
    for b in range(count // 32):
        base = b * 18
        d = fp16(data[base:base + 2])
        for i in range(16):
            packed = data[base + 2 + i]
            out.append(d * ((packed & 0x0F) - 8))
            out.append(d * ((packed >> 4) - 8))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("tensor")
    args = parser.parse_args()

    f, tensors, data_start = read_gguf(args.model)
    if args.tensor not in tensors:
        raise SystemExit(f"tensor {args.tensor} not found")
    info = tensors[args.tensor]
    count = 1
    for d in info["dims"]:
        count *= d
    per_block, elements = GGUF_BLOCK.get(info["type"], (None, None))
    if per_block is None:
        raise SystemExit(f"tensor type {info['type']} is not supported by this tool")
    f.seek(data_start + info["offset"])
    raw = f.read(count // elements * per_block)

    if info["type"] == 8:
        values = dequant_q8_0(raw, count)
    elif info["type"] == 0:
        values = list(struct.unpack(f"<{count}f", raw))
    else:
        raise SystemExit("only Q8_0 and F32 sources are supported")

    encoded = quantise_q4_0(values)
    restored = dequant_q4_0(encoded, count)
    error = sum((a - b) ** 2 for a, b in zip(values, restored))
    magnitude = sum(a * a for a in values)
    rms = (error / count) ** 0.5
    relative = (error / magnitude) ** 0.5 if magnitude else 0.0

    with open(args.output, "wb") as out:
        out.write(encoded)
    sidecar = {
        "tensor": args.tensor,
        "source_type": info["type"],
        "dims": info["dims"],
        "values": count,
        "source_bytes": len(raw),
        "derived_bytes": len(encoded),
        "rms_error": rms,
        "relative_error": relative,
    }
    with open(args.output + ".json", "w") as out:
        json.dump(sidecar, out, indent=1)
    print(f"{args.tensor}: {len(raw)} -> {len(encoded)} bytes "
          f"({100.0 * (1 - len(encoded) / len(raw)):.1f}% smaller), "
          f"rms {rms:.5f}, relative {relative:.4f}")


if __name__ == "__main__":
    main()
