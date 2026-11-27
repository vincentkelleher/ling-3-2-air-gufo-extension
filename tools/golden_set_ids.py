#!/usr/bin/env python3
"""Golden set compared token-for-token from an *identical* context.

The text-based harness starts the two sides from different framings: `llama-completion -f`
applies its own BOS and whitespace handling, so its context is not the engine's. Here the
engine tokenises the rendered prompt, and both sides are then fed that exact id list and
decoded greedily, so a divergence is a divergence in the model, not in the harness.
"""
import argparse
import os
import subprocess
import sys

ASSISTANT = "<role>ASSISTANT</role>\n<think" + ">"

CASES = {
    "plain": ("The capital of France is", True),
    "arithmetic": ("What is 17 times 23?", True),
    "tool-call": ("What is the price of AAPL? Use the get_stock_price tool.", True, True),
    "thinking-on": ("Explain why the sky is blue in one sentence.", False),
}


def render(prompt, thinking_off, tools=False):
    text = "<role>SYSTEM</role>"
    text += "detailed thinking off" if thinking_off else "detailed thinking on"
    text += "<|role_end|>"
    if tools:
        text += ("<role>SYSTEM</role>You may call tools. "
                 "get_stock_price(symbol: str) -> float<|role_end|>")
    text += f"<role>HUMAN</role>{prompt}<|role_end|>" + ASSISTANT
    return text


def run(cmd, env=None):
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return result.stdout


def ids_of(engine, model, prompt_file):
    out = run([engine, model, "0"], {"LING3_PROMPT_FILE": prompt_file,
                                     "LING3_PRINT_PROMPT_IDS": "1", "LING3_CONTEXT": "4096"})
    for line in out.splitlines():
        if line.startswith("prompt ids:"):
            return line.split(":", 1)[1].split()
    raise SystemExit("engine did not print prompt ids")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--llama-greedy", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--workdir", default="/tmp")
    args = parser.parse_args()

    print(f"{'case':12} {'steps':>6} {'match':>6}  first divergence")
    failures = 0
    for name, spec in CASES.items():
        prompt, thinking_off = spec[0], spec[1]
        tools = len(spec) > 2
        prompt_file = os.path.join(args.workdir, f"golden_ids_{name.replace('-', '_')}.txt")
        with open(prompt_file, "w", encoding="utf-8") as handle:
            handle.write(render(prompt, thinking_off, tools))
        ids = ids_of(args.engine, args.model, prompt_file)
        joined = ",".join(ids)
        def id_line(text):
            for line in text.splitlines():
                if line.startswith("ids "):
                    return line.split()[1:]
            return []

        reference = id_line(run([args.llama_greedy, args.model, joined, str(args.steps)]))
        ours = id_line(run([args.engine, args.model, str(args.steps)],
                           {"LING3_GREEDY_FROM_IDS": joined, "LING3_CONTEXT": "4096"}))
        index = None
        for i, (a, b) in enumerate(zip(ours, reference)):
            if a != b:
                index = i
                break
        if index is None and len(ours) != len(reference):
            index = min(len(ours), len(reference))
        if index is None:
            print(f"{name:12} {len(ours):>6} {'yes':>6}")
        else:
            failures += 1
            print(f"{name:12} {len(ours):>6} {'no':>6}  token {index}: engine {ours[index]} vs llama {reference[index]}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases token-identical for {args.steps} generated tokens")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
