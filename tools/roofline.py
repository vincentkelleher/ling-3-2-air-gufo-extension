#!/usr/bin/env python3
"""Memory-bandwidth roofline for Ling 3.2 Air Q4_K_M on this machine.

Reads the GGUF tensor table, models the bytes one decode step actually reads
(routed experts included), and turns a measured bandwidth and a measured
tokens/s into "% of roofline". Also prints the prefill bound.

Usage:
  tools/roofline.py --bandwidth 237 --tok-s 45
  tools/roofline.py --bandwidth 237 --prefill-tok-s 900 --prefill-tokens 2048
  tools/roofline.py --json

The bandwidth default is the median measured by tools/bandwidth_probe.hip on
this box (read 237 GB/s, copy 204 GB/s, write 198 GB/s, 8 GiB, 3 runs,
2026-11-25).
"""

import argparse
import os
import struct
import sys

DEFAULT_GGUF = os.path.expanduser(
    "~/Models/bartowski/Ling-3.2-air-GGUF/Ling-3.2-air-Q4_K_M/"
    "Ling-3.2-air-Q4_K_M-00001-of-00003.gguf")
DEFAULT_BANDWIDTH = 237.0

# Bytes per weight for the storage types this quantization uses.
BYTES_PER_WEIGHT = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0, "Q8_0": 34 / 32,
    "Q4_0": 18 / 32, "Q4_K": 144 / 256, "Q5_K": 176 / 256, "Q6_K": 210 / 256,
}


# Bytes per block and elements per block for the ggml types this model uses; the parser below
# reads the tensor table directly so the bench needs nothing beyond the standard library.
BLOCK = {
    0: (4, 1),      # F32
    1: (2, 1),      # F16
    2: (18, 32),    # Q4_0
    3: (20, 32),    # Q4_1
    6: (22, 32),    # Q5_0
    7: (24, 32),    # Q5_1
    8: (34, 32),    # Q8_0
    10: (84, 256),  # Q2_K
    11: (110, 256), # Q3_K
    12: (144, 256), # Q4_K
    13: (176, 256), # Q5_K
    14: (210, 256), # Q6_K
    30: (2, 1),     # BF16
}


class Field:
    """Minimal stand-in for gguf's ReaderField: only `contents()` is used."""

    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class Reader:
    def __init__(self, path):
        f = open(path, "rb")
        self._f = f
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        self.fields = {}
        for _ in range(n_kv):
            key = self._string()
            self.fields[key] = Field(self._value(struct.unpack("<I", f.read(4))[0]))
        self.tensors = _Tensors()
        for _ in range(n_tensors):
            name = self._string()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            tensor_type = struct.unpack("<I", f.read(4))[0]
            self._f.read(8)  # offset
            count = 1
            for d in dims:
                count *= d
            per_block, elements = BLOCK.get(tensor_type, (4, 1))
            n_bytes = count // elements * per_block
            self.tensors.append(name, n_bytes, tensor_type)

    def _string(self):
        n = struct.unpack("<Q", self._f.read(8))[0]
        return self._f.read(n).decode("utf-8", "replace")

    def _value(self, kind):
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind == 8:
            return self._string()
        if kind == 9:
            element = struct.unpack("<I", self._f.read(4))[0]
            count = struct.unpack("<Q", self._f.read(8))[0]
            return [self._value(element) for _ in range(count)]
        if kind in fixed:
            raw = self._f.read(fixed[kind])
            if kind == 6:
                return struct.unpack("<f", raw)[0]
            if kind == 5:
                return struct.unpack("<i", raw)[0]
            if kind == 4:
                return struct.unpack("<I", raw)[0]
            if kind == 10:
                return struct.unpack("<d", raw)[0]
            if kind == 11:
                return struct.unpack("<q", raw)[0]
            if kind == 12:
                return struct.unpack("<Q", raw)[0]
            if kind == 7:
                return bool(raw[0])
            return int.from_bytes(raw, "little")
        raise ValueError(f"unsupported GGUF value type {kind}")


class _Tensors(list):
    def append(self, name, n_bytes, tensor_type):
        class T:
            pass

        tensor = T()
        tensor.name = name
        tensor.n_bytes = n_bytes
        tensor.tensor_type = type("T", (), {"name": str(tensor_type)})()
        super().append(tensor)


def tensor_bytes(reader):
    """{name: (bytes, type)} for every tensor of the model."""
    out = {}
    for tensor in reader.tensors:
        out[tensor.name] = (int(tensor.n_bytes), tensor.tensor_type.name)
    return out


def load_all_shards(gguf_path):
    """Every shard of a split model, merged by name, using only the standard library."""
    directory = os.path.dirname(gguf_path)
    first = os.path.basename(gguf_path)
    prefix, _, rest = first.partition("-00001-of-")
    total = rest.split(".")[0] if rest else "00001"
    paths = [gguf_path]
    if rest:
        for index in range(2, int(total) + 1):
            paths.append(os.path.join(directory, f"{prefix}-{index:05d}-of-{total}.gguf"))
    tensors = {}
    fields = {}
    for path in paths:
        reader = Reader(path)
        tensors.update(tensor_bytes(reader))
        fields.update(reader.fields)
    return tensors, fields


def field_value(fields, key, default=None):
    entry = fields.get(key)
    if entry is None:
        return default
    try:
        return entry.contents()
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH,
                        help="measured read bandwidth in GB/s")
    parser.add_argument("--tok-s", type=float, default=None,
                        help="measured decode tokens/s (short context)")
    parser.add_argument("--prefill-tok-s", type=float, default=None)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=0,
                        help="context already in the cache, in tokens")
    parser.add_argument("--routing", type=float, default=None,
                        help="measured distinct experts hit per MoE block per "
                             "token (default: the 8 the router selects)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.gguf):
        sys.exit(f"missing {args.gguf}")
    tensors, fields = load_all_shards(args.gguf)

    n_layer = int(field_value(fields, "bailingmoe3.block_count", 43))
    n_nextn = int(field_value(fields, "bailingmoe3.nextn_predict_layers", 1))
    n_trunk = n_layer - n_nextn
    n_embd = int(field_value(fields, "bailingmoe3.embedding_length", 2560))
    n_ff_exp = int(field_value(fields, "bailingmoe3.expert_feed_forward_length", 768))
    n_expert = int(field_value(fields, "bailingmoe3.expert_count", 512))
    top_k = int(field_value(fields, "bailingmoe3.expert_used_count", 8))
    kv_lora = int(field_value(fields, "bailingmoe3.attention.kv_lora_rank", 512))
    n_rot = int(field_value(fields, "bailingmoe3.rope.dimension_count", 64))
    n_dense = int(field_value(fields, "bailingmoe3.leading_dense_block_count", 2))
    head_kv = field_value(fields, "bailingmoe3.attention.head_count_kv", [])
    if not isinstance(head_kv, list) or len(head_kv) < n_layer:
        sys.exit("cannot read attention.head_count_kv")
    is_kda = [int(value) == 0 for value in head_kv[:n_layer]]

    distinct = args.routing if args.routing else float(top_k)
    expert_fraction = distinct / n_expert

    def bytes_of(name):
        entry = tensors.get(name)
        if entry is None:
            return 0
        return entry[0]

    def dense_weight(name):
        return bytes_of(name)

    def expert_weight(name, count):
        return int(bytes_of(name) * count / n_expert)

    buckets = {"experts": 0, "shared": 0, "attention": 0, "router": 0,
               "norms": 0, "embedding": 0, "head": 0, "kv_cache": 0}

    for layer in range(n_trunk):
        base = f"blk.{layer}."
        buckets["norms"] += dense_weight(base + "attn_norm.weight")
        buckets["norms"] += dense_weight(base + "ffn_norm.weight")
        if is_kda[layer]:
            for name in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                         "ssm_f_a.weight", "ssm_g_a.weight", "attn_output.weight",
                         "ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                         "ssm_conv1d_v.weight", "ssm_beta.weight", "ssm_a",
                         "ssm_dt.bias", "ssm_norm.weight"):
                buckets["attention"] += dense_weight(base + name)
        else:
            for name in ("attn_q.weight", "attn_kv_a_mqa.weight",
                         "attn_kv_a_norm.weight", "attn_k_b.weight",
                         "attn_v_b.weight", "attn_gate.weight",
                         "attn_output.weight"):
                buckets["attention"] += dense_weight(base + name)
            # The MLA latent cache grows with the context.
            buckets["kv_cache"] += (kv_lora + n_rot) * 4 * args.depth
        if layer < n_dense:
            for name in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                buckets["experts"] += dense_weight(base + name)
        else:
            for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                         "ffn_down_exps.weight"):
                buckets["experts"] += expert_weight(base + name, distinct)
            for name in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                         "ffn_down_shexp.weight"):
                buckets["shared"] += dense_weight(base + name)
            buckets["router"] += dense_weight(base + "ffn_gate_inp.weight")
            buckets["router"] += dense_weight(base + "exp_probs_b.bias")

    buckets["embedding"] += n_embd * BYTES_PER_WEIGHT["Q8_0"]
    buckets["head"] += bytes_of("output.weight")
    buckets["norms"] += bytes_of("output_norm.weight")

    total = sum(buckets.values())
    roofline = args.bandwidth * 1e9 / total

    # Prefill: one pass over the prompt reads the blocks' weights once for the
    # expert set the whole prompt touches; at 2048 tokens that is every expert.
    prefill_bytes = 0
    for layer in range(n_trunk):
        base = f"blk.{layer}."
        prefill_bytes += bytes_of(base + "attn_norm.weight")
        prefill_bytes += bytes_of(base + "ffn_norm.weight")
        if is_kda[layer]:
            for name in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                         "ssm_f_a.weight", "ssm_g_a.weight", "attn_output.weight"):
                prefill_bytes += bytes_of(base + name)
        else:
            for name in ("attn_q.weight", "attn_kv_a_mqa.weight",
                         "attn_k_b.weight", "attn_v_b.weight",
                         "attn_gate.weight", "attn_output.weight"):
                prefill_bytes += bytes_of(base + name)
        if layer < n_dense:
            for name in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                prefill_bytes += bytes_of(base + name)
        else:
            for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                         "ffn_down_exps.weight"):
                prefill_bytes += bytes_of(base + name)
    prefill_roofline = None
    if args.prefill_tokens > 0:
        prefill_roofline = args.bandwidth * 1e9 * args.prefill_tokens / prefill_bytes

    if args.json:
        import json
        print(json.dumps({
            "bandwidth_gbs": args.bandwidth,
            "decode_bytes_per_token": total,
            "roofline_tok_s": roofline,
            "buckets": buckets,
            "measured_tok_s": args.tok_s,
            "percent_of_roofline": (100.0 * args.tok_s / roofline) if args.tok_s else None,
            "prefill_roofline_tok_s": prefill_roofline,
            "prefill_percent": (100.0 * args.prefill_tok_s / prefill_roofline)
                               if (args.prefill_tok_s and prefill_roofline) else None,
        }, indent=1))
        return

    print(f"bandwidth {args.bandwidth:.0f} GB/s, decode reads {total/1e6:.0f} MB/token"
          f" (depth {args.depth}, {distinct:g} experts/block)")
    for name, value in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {value/1e6:8.1f} MB  {100.0*value/total:5.1f}%")
    print(f"roofline: {roofline:.1f} tok/s")
    if args.tok_s:
        print(f"measured: {args.tok_s:.1f} tok/s = {100.0*args.tok_s/roofline:.1f}% of roofline")
    if prefill_roofline:
        print(f"prefill roofline ({args.prefill_tokens} tokens): {prefill_roofline:.0f} tok/s")
        if args.prefill_tok_s:
            print(f"measured: {args.prefill_tok_s:.0f} tok/s = "
                  f"{100.0*args.prefill_tok_s/prefill_roofline:.1f}% of roofline")


if __name__ == "__main__":
    main()
