#!/usr/bin/env python3
"""Render reference chat prompts with the model's own chat template.

Runs chat_template.jinja through jinja2 exactly as transformers' 
apply_chat_template does for this template (a `tojson` filter with
ensure_ascii=False), and writes fixtures/chat_cases.json for the C++ renderer
in src/text/chat_template.cpp to reproduce byte for byte.

Usage: tools/ref_chat.py [--out fixtures/chat_cases.json]
"""
import argparse, json, os, pathlib

import jinja2

MODEL_DIR = pathlib.Path(os.path.expanduser("~/Models/inclusionAI/Ling-3.2-air"))

TOOL_DEFS = [
    {"type": "function", "function": {"name": "get_stock_price", "description": "Get the stock price",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "Ticker"}},
                    "required": ["symbol"]}}},
]


def tojson(value, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(value, **kwargs)


def render(env, messages, tools, enable_thinking, preserve_thinking, add_vision_id):
    return env.get_template("chat_template.jinja").render(
        messages=messages, tools=tools, enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking, add_vision_id=add_vision_id,
        add_generation_prompt=True)


def m(parts, thought="", tool_calls=None):
    msg = {"parts": parts, "thought": thought, "tool_calls": tool_calls or []}
    return msg


def to_jinja(msg):
    parts = msg["parts"]
    if all("text" in p for p in parts) and len(parts) <= 1:
        content = parts[0]["text"] if parts else ""
    else:
        content = []
        for p in parts:
            if "image" in p:
                content.append({"type": "image"})
            else:
                content.append({"type": "text", "text": p["text"]})
    out = {"role": msg["role"], "content": content}
    if msg["thought"]:
        out["reasoning_content"] = msg["thought"]
    if msg["tool_calls"]:
        out["tool_calls"] = [{"id": "call", "function": {"name": c["name"], "arguments": c["arguments"][0][1] if False else c["arguments_obj"]}} for c in msg["tool_calls"]]
    return out


CASES = []


def case(name, turns, tools=None, enable_thinking=True, preserve_thinking=True, add_vision_id=False):
    CASES.append({"name": name, "turns": turns, "tools": TOOL_DEFS if tools else [],
                  "enable_thinking": enable_thinking, "preserve_thinking": preserve_thinking,
                  "add_vision_id": add_vision_id})


def user(text):
    return {"role": "user", "parts": [{"text": text}], "thought": "", "tool_calls": []}


def system(text):
    return {"role": "system", "parts": [{"text": text}], "thought": "", "tool_calls": []}


def assistant(text, thought=""):
    return {"role": "assistant", "parts": [{"text": text}], "thought": thought, "tool_calls": []}


def tool_response(text):
    return {"role": "tool", "parts": [{"text": text}], "thought": "", "tool_calls": []}


case("single user", [user("What is 2+2?")])
case("system and user", [system("You are terse."), user("Hi")])
case("system empty", [system(""), user("Hi")])
case("thinking off", [user("Hi")], enable_thinking=False)
case("system with thinking instruction", [system("detailed thinking off"), user("Hi")])
case("tools", [user("Price of AAPL?")], tools=True)
case("tools with system", [system("Be brief."), user("Price of AAPL?")], tools=True)
case("assistant thought", [user("Hi"), assistant("Hello!", thought="The user greeted me.")])
case("assistant empty thought", [user("Hi"), assistant("Hello!")])
case("assistant inline think", [user("Hi"), assistant("Thinking</think>Hello!")])
case("preserve thinking off", [user("Hi"), assistant("Hello!", thought="Because."), user("Again")],
     preserve_thinking=False)
case("tool call", [user("Price?"), {"role": "assistant", "parts": [{"text": ""}], "thought": "",
    "tool_calls": [{"name": "get_stock_price", "arguments_obj": {"symbol": "AAPL"},
                    "arguments": [["symbol", "AAPL", True]]}]}])
case("tool response", [user("Price?"), {"role": "assistant", "parts": [{"text": ""}], "thought": "",
    "tool_calls": [{"name": "get_stock_price", "arguments_obj": {"symbol": "AAPL"},
                    "arguments": [["symbol", "AAPL", True]]}]}, tool_response("{\"price\": 101.2}")])
case("multi tool responses", [user("Price?"), {"role": "assistant", "parts": [{"text": ""}], "thought": "",
    "tool_calls": [{"name": "get_stock_price", "arguments_obj": {"symbol": "AAPL"},
                    "arguments": [["symbol", "AAPL", True]]}]},
    tool_response("a"), tool_response("b")])
case("image", [{"role": "user", "parts": [{"text": "What is this? "}, {"image": True}],
                "thought": "", "tool_calls": []}])
case("image with id", [{"role": "user", "parts": [{"text": ""}, {"image": True}, {"text": " describe"}],
                        "thought": "", "tool_calls": []}], add_vision_id=True)
case("two images", [{"role": "user", "parts": [{"image": True}, {"text": " and "}, {"image": True}],
                     "thought": "", "tool_calls": []}])
case("other role", [{"role": "developer", "parts": [{"text": "note"}], "thought": "", "tool_calls": []},
                    user("Hi")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fixtures/chat_cases.json")
    parser.add_argument("--template", default=str(MODEL_DIR / "chat_template.jinja"))
    args = parser.parse_args()

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(pathlib.Path(args.template).parent)))
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))

    out_cases = []
    for c in CASES:
        expected = render(env, [to_jinja(t) for t in c["turns"]],
                          c["tools"], c["enable_thinking"], c["preserve_thinking"], c["add_vision_id"])
        out_cases.append({"name": c["name"], "turns": c["turns"], "tools": c["tools"],
                          "enable_thinking": c["enable_thinking"],
                          "preserve_thinking": c["preserve_thinking"],
                          "add_vision_id": c["add_vision_id"], "expected": expected})

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"cases": out_cases}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(out_cases)} cases")


if __name__ == "__main__":
    main()
