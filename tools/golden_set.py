#!/usr/bin/env python3
"""Golden-set comparison against llama.cpp.

Runs this engine and a llama.cpp build on the same pinned prompts, greedily, and
reports where the two streams diverge. `engine-prefill`'s contract asks for
token-for-token equality with deviations documented by divergence index, so this
is the harness that produces those indices.

Each engine is invoked through a wrapper script, because in this setup both run
inside a container:

    engine wrapper: $WRAP_ENGINE <prompt-file> <steps>      -> prints "text: ..."
    llama  wrapper: $WRAP_LLAMA  <prompt-file> <steps>      -> prints the text

    tools/golden_set.py --engine-wrapper run-engine.sh \
                        --llama-wrapper run-llama.sh [--steps 32]

Output: one row per case with the common prefix length and the first divergence.
"""

import argparse
import os
import subprocess
import sys

# The model's own framing: <role>SYSTEM</role> + thinking instruction + role_end,
# then the human turn, then the assistant marker. The marker is the seven
# characters "<think" + ">" (bytes 3c 74 68 69 6e 6b 3e), not the eight-character
# spelling other Qwen-family models use.
ASSISTANT = "<role>ASSISTANT</role>\n<think" + ">"

CASES = {
    "plain": ("The capital of France is", True),
    "arithmetic": ("What is 17 times 23?", True),
    "tool-call": ("What is the price of AAPL? Use the get_stock_price tool.", True),
    "thinking-on": ("Explain why the sky is blue in one sentence.", False),
}


def render(prompt, thinking_off, tools=False):
    instruction = "detailed thinking off" if thinking_off else "detailed thinking on"
    system = "<role>SYSTEM</role>"
    if tools:
        system += (
            "# Tools\n\nYou may call one or more functions to assist with the "
            "user query.\n\nYou are provided with function signatures within "
            "<tools></tools> XML tags:\n<tools>\n"
            '{"type": "function", "function": {"name": "get_stock_price", '
            '"description": "Get the stock price", "parameters": {"type": '
            '"object", "properties": {"symbol": {"type": "string"}}, "required": '
            '["symbol"]}}}\n</tools>\n\nIf none of the functions can be used, '
            "point it out.\n"
            + instruction
        )
    else:
        system += instruction
    return (system + "<|role_end|><role>HUMAN</role>" + prompt + "<|role_end|>"
            + ASSISTANT)


def run(wrapper, prompt_file, steps):
    output = subprocess.run([wrapper, prompt_file, str(steps)],
                            capture_output=True, text=True)
    if output.returncode not in (0,):
        raise SystemExit(f"{wrapper} failed:\n{output.stderr[-500:]}")
    text = ""
    for line in output.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("text:"):
            text = stripped[len("text:"):].strip()
        elif stripped and not stripped.startswith("step") and "tok/s" not in stripped:
            text = text or stripped
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-wrapper", required=True)
    parser.add_argument("--llama-wrapper", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--workdir", default="/tmp")
    args = parser.parse_args()

    print(f"{'case':12} {'prefix':>7} {'engine chars':>13} {'llama chars':>12}  first divergence")
    failures = 0
    for name, (prompt, thinking_off) in CASES.items():
        tools = name == "tool-call"
        text = render(prompt, thinking_off, tools)
        prompt_file = os.path.join(args.workdir, f"golden_{name.replace('-', '_')}.txt")
        with open(prompt_file, "w", encoding="utf-8") as handle:
            handle.write(text)

        ours = run(args.engine_wrapper, prompt_file, args.steps)
        reference = run(args.llama_wrapper, prompt_file, args.steps)

        prefix = 0
        while (prefix < len(ours) and prefix < len(reference)
               and ours[prefix] == reference[prefix]):
            prefix += 1
        divergence = ""
        if prefix < max(len(ours), len(reference)):
            divergence = f"engine {ours[prefix:prefix + 24]!r} vs llama {reference[prefix:prefix + 24]!r}"
            failures += 1
        print(f"{name:12} {prefix:>7} {len(ours):>13} {len(reference):>12}  {divergence}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} cases match in full; "
          f"prefix is in characters of generated text")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
