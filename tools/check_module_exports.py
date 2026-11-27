#!/usr/bin/env python3
"""Check a Linux replacement module: python3 tools/check_module_exports.py MODULE.so."""
import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("module")
args = parser.parse_args()
lines = subprocess.check_output(
    ["nm", "-D", "-C", "--defined-only", args.module], text=True
).splitlines()
names = {line.split(maxsplit=2)[2] for line in lines if len(line.split(maxsplit=2)) == 3}
required = {f"gufo_ext_module_{name}_v1" for name in ("build_id", "manifest", "register")}
leaks = sorted(name for name in names if "gufo::ext::ling3_air::" in name or
               name.startswith("ling3_air_provider_register("))
if required - names:
    raise SystemExit(f"missing SDK entry points: {sorted(required - names)}")
if leaks:
    raise SystemExit(f"preemptible Ling implementation symbols: {len(leaks)} (e.g. {leaks[0]})")
print("OK: SDK entry points exported; Ling implementation is local")
