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
`--bandwidth` to use another. `tools/roofline.py` imports the `gguf` Python
package (`pip install gguf`) to derive bytes/token from the shards; without it
the roofline half fails while the measured bandwidth half still runs. Prefill is
reported as unavailable until the
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


def measure(engine, model, prompt, steps, depth, context, chunked_prefill):
    env = dict(os.environ, LING3_DEPTH=str(depth), LING3_CONTEXT=str(context))
    if chunked_prefill:
        env["LING3_PREFILL_CHUNK"] = "1"
        env["LING3_CHUNK_SIZE"] = str(chunked_prefill)
    command = [engine, model, str(steps), prompt]
    output = subprocess.run(command, env=env, capture_output=True, text=True)
    if output.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{output.stderr[-800:]}")
    # Each rate comes from its own line: the engine prints "prefill: N tokens in ...
    # = X tok/s" before "X tok/s over M steps". Matching the first "tok/s" in the
    # output reported the prefill rate as the decode rate.
    match = re.search(r"([0-9.]+) tok/s over [0-9]+ steps", output.stdout)
    if match is None:
        raise SystemExit(f"no decode rate in the engine output:\n{output.stdout[-400:]}")
    prefill = 0.0
    prefill_match = re.search(r"prefill: .* = ([0-9.]+) tok/s", output.stdout)
    if prefill_match is not None:
        prefill = float(prefill_match.group(1))
    text = ""
    for line in output.stdout.splitlines():
        if line.startswith("text:"):
            text = line[len("text:"):].strip()
    return float(match.group(1)), prefill, text


def roofline(model, bandwidth, depth, prefill_tokens):
    """Decode and prefill rooflines from the GGUF tensor table (tools/roofline.py).

    The prefill roofline is the whole weight set read once, spread over the batch:
    bytes/bandwidth gives steps/s, times the token count.
    """
    import json
    import subprocess as sp

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "roofline.py")
    output = sp.run([sys.executable, script, "--gguf", model, "--bandwidth",
                     str(bandwidth), "--depth", str(depth),
                     "--prefill-tokens", str(prefill_tokens), "--json"],
                    capture_output=True, text=True)
    if output.returncode != 0:
        raise SystemExit(f"roofline.py failed:\n{output.stderr[-800:]}")
    return json.loads(output.stdout)


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
    parser.add_argument("--prefill-tokens", type=int, default=2048,
                        help="token count for the prefill roofline")
    parser.add_argument("--chunked-prefill", type=int, default=0,
                        help="ingest the prompt through PrefillChunk with this "
                             "chunk size (0 keeps the sequential path)")
    args = parser.parse_args()

    print(f"bandwidth {args.bandwidth:.1f} GB/s (measured), "
          f"{args.runs} runs x {args.steps} steps, medians")
    print(f"{'depth':>6} {'tok/s':>8} {'roofline':>9} {'% of roof':>10}  text")
    for depth in args.depths:
        rates = []
        prefills = []
        text = ""
        for _ in range(args.runs):
            rate, prefill, text = measure(args.engine, args.model, args.prompt,
                                          args.steps, depth, args.context,
                                          args.chunked_prefill)
            rates.append(rate)
            prefills.append(prefill)
        median = statistics.median(rates)
        model_roofline = roofline(args.roofline_model or args.model, args.bandwidth,
                                  depth, args.prefill_tokens)
        decode_roofline = model_roofline["roofline_tok_s"]
        print(f"{depth:>6} {median:>8.2f} {decode_roofline:>9.1f} "
              f"{100.0 * median / decode_roofline:>9.1f}%  {text[:40]}")
        if prefills and prefills[0] > 0:
            prefill = statistics.median(prefills)
            mode = (f"chunked, {args.chunked_prefill}-token chunks"
                    if args.chunked_prefill else "token by token")
            prefill_roofline = model_roofline["prefill_roofline_tok_s"]
            print(f"       prefill {prefill:>6.2f} tok/s ({mode}) = "
                  f"{100.0 * prefill / prefill_roofline:>7.2f}% of the "
                  f"{prefill_roofline:,.0f} tok/s prefill roofline "
                  f"({args.prefill_tokens} tokens)")
    # tg128: the decode figure the contract names, measured separately from the
    # short-context rows above.
    tg_rates = []
    for _ in range(args.runs):
        rate, _, _ = measure(args.engine, args.model, args.prompt, 128, 0,
                             args.context, args.chunked_prefill)
        tg_rates.append(rate)
    tg = statistics.median(tg_rates)
    tg_roofline = roofline(args.roofline_model or args.model, args.bandwidth, 0,
                           args.prefill_tokens)["roofline_tok_s"]
    print(f"tg128  {tg:>8.2f} {tg_roofline:>9.1f} {100.0 * tg / tg_roofline:>9.1f}%  "
          "128 generated tokens, depth 0")
    # Batch: the runner serialises requests (one engine, one token at a time), so a
    # batched measurement would report a queueing time rather than a batch rate.
    print("batch  not reported: this runner serialises requests, so a batch size above "
          "one measures queueing, not throughput")
    print("reference on the same box: llama.cpp CPU decode 12.52 tok/s, "
          "prompt eval 24.20 tok/s")


if __name__ == "__main__":
    main()
