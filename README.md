# ling32-air-gufo

A Gufo extension for **Ling 3.2 Air** — a 128B-parameter hybrid MoE
(`bailing_moe_v3_vl`) served from the Q4_K_M GGUF, with text, image input and
the model's own MTP draft layer.

Status: **scaffold**. The manifest, probe and module build exist; the engine
does not. Nothing is served yet, and the probe is not wired into a text
provider until the runner lands. See [notes/design.md](notes/design.md) for the
model shape and the plan.

## What the model is

| Property | Value |
| --- | --- |
| Trunk | 42 blocks, hidden 2560, dense blocks 0–1 then MoE |
| Attention | hybrid: KDA (gated linear attention with short conv) and MLA blocks |
| Experts | 512 routed, top-8, group-limited (8 groups, top-4), + shared expert |
| Vocab / context | 157 184 tokens / 262 144 positions |
| Extra heads | MTP next-token layer (in the same GGUF), `qwen3_moe_vit` vision tower (mmproj) |
| Weights target | `bartowski/Ling-3.2-air-GGUF`, `Q4_K_M` (80.39 GB, 3 shards) |

## Install

Not yet installable: the extension registers metadata only.

```sh
gufo extension install ./ling32-air-gufo --yes   # once the runner exists
gufo model install ling-3.2-air                  # pinned files from models.lock.json
```

The extension targets **Gufo v0.36.0** (`4a7d8c6`, extension API 0.3.x) and
builds against that release's SDK. A Gufo built without the SDK cannot build
modules; see [Gufo's extension docs](https://github.com/gufo-org/gufo/blob/main/docs/extensions.md#what-a-module-build-needs).

## Repository layout

```text
extension.json      manifest: identity, provider, keys, pinned model files
extension.cmake     CMake targets (host library now, HIP library with the engine)
src/probe.*         host-only GGUF probe (architecture + Air shape)
src/extension.cpp   provider registration
notes/design.md     model/tensor map, cache plan, measurement plan
tools/              build and benchmark harness (added with the engine)
```

## License

Code: MIT ([LICENSE](LICENSE)). Weights: the model publisher's terms — see
[THIRD_PARTY.md](THIRD_PARTY.md). No weights are distributed here.
