#!/usr/bin/env python3
"""Run the engine and report measured speed as a percentage of the roofline.

Wraps `tests/forward_test.cpp` (built by the extension) at several depths and
feeds the numbers through the same model as `tools/roofline.py`, so the table in
the README is reproducible with one command.

    tools/bench_report.py --engine build/ling3_air_forward_test \
                          --model /models/Ling-3.2-air-Q4_K_M-00001-of-00003.gguf \
                          --depths 0 512 2048 --steps 16 --runs 3

`--engine` is whatever the harness calls to run one measurement: the binary
itself, or a wrapper script when the engine runs inside a container. In this
setup the wrapper is `~/ling32-work/engine_wrap.sh`, and `--model` uses the
container path while `--roofline-model` uses the host path (the roofline model is
computed on the host from the GGUF).

The bandwidth default is the measured read bandwidth of this box; pass
`--bandwidth` to use another. Prefill is reported as unavailable until the
engine has a batched pass (notes/prefill-design.md).
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BANDWIDTH = 237.0  # GB/s, tools/bandwidth_probe.hip, medians of 3


def measure(engine, model, prompt, steps, depth, context):
    env = dict(os.environ, LING3_DEPTH=str(depth), LING3_CONTEXT=str(context))
    command = [engine, model, str(steps), prompt]
    output = subprocess.run(command, env=env, capture_output=True, text=True)
    if output.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{output.stderr[-800:]}")
    match = re.search(r"([0-9.]+) tok/s", output.stdout)
    if match is None:
        raise SystemExit(f"no rate in the engine output:\n{output.stdout[-400:]}")
    text = ""
    for line in output.stdout.splitlines():
        if line.startswith("text:"):
            text = line[len("text:"):].strip()
    return float(match.group(1)), text


def roofline_tokens_per_second(model, bandwidth, depth):
    """Decode roofline from the GGUF tensor table (see tools/roofline.py)."""
    import json
    import subprocess as sp

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "roofline.py")
    output = sp.run([sys.executable, script, "--gguf", model, "--bandwidth",
                     str(bandwidth), "--depth", str(depth), "--json"],
                    capture_output=True, text=True)
    if output.returncode != 0:
        raise SystemExit(f"roofline.py failed:\n{output.stderr[-800:]}")
    return json.loads(output.stdout)["roofline_tok_s"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True,
                        help="path to ling3_air_forward_test")
    parser.add_argument("--model", required=True,
                        help="first shard of the Q4_K_M GGUF")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 512, 2048])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--roofline-model", default=None,
                        help="host path to the GGUF for the roofline model "
                             "(defaults to --model, which may be a container path)")
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()

    print(f"bandwidth {args.bandwidth:.1f} GB/s (measured), "
          f"{args.runs} runs x {args.steps} steps, medians")
    print(f"{'depth':>6} {'tok/s':>8} {'roofline':>9} {'% of roof':>10}  text")
    for depth in args.depths:
        rates = []
        text = ""
        for _ in range(args.runs):
            rate, text = measure(args.engine, args.model, args.prompt, args.steps,
                                 depth, args.context)
            rates.append(rate)
        median = statistics.median(rates)
        roofline = roofline_tokens_per_second(
            args.roofline_model or args.model, args.bandwidth, depth)
        print(f"{depth:>6} {median:>8.2f} {roofline:>9.1f} "
              f"{100.0 * median / roofline:>9.1f}%  {text[:40]}")

    print("\nprefill: not measured - the engine has no batched pass yet "
          "(notes/prefill-design.md)")
    print("reference on the same box: llama.cpp CPU decode 12.52 tok/s, "
          "prompt eval 24.20 tok/s")


if __name__ == "__main__":
    main()
