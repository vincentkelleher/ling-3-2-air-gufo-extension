# Conformance record

Against Gufo's [extension conformance](https://github.com/gufo-org/gufo/blob/main/docs/internal/extension-conformance.md)
list, for one extension revision on one platform. Evidence is retained in this repository;
where an item does not pass, it says so.

## Pinned inputs

| item | value |
| --- | --- |
| Gufo | 0.36.0, extension API `0.3.42`, no changes to the installed tree |
| extension | `ling3_air` 0.1.0, this repository |
| model | `bartowski/Ling-3.2-air-GGUF` @ `2073edb5da4296cb37b334584b858aa6141e352f`, pinned with sizes and SHA-256 in `models.lock.json`; `gufo model verify` reports ok |
| reference | llama.cpp @ `e95426f`, CPU build under `/work/llama-cpu` (prompt eval 24.20 tok/s, decode 12.52 tok/s on this box) |
| toolchain | ROCm/CLR 10.1, gcc 16.2, target `gfx1151` (Ryzen AI Max, 128 GB unified) |
| platform | Linux only (`platforms: ["linux"]` in `extension.json`); no Windows or macOS build is claimed |

Commands are the ones in `README.md` and `IMPROVEMENTS.md`; measurement programs are
committed under `tools/` (`bandwidth_probe.hip`, `roofline.py`, `bench_report.py`,
`golden_set.py`).

## Registration and probes

| check | evidence |
| --- | --- |
| registration | `gufo diagnose --section extensions --json` → `providers:[{"id":"ling3-air","extension":"ling3_air","available":true}]`, source `module:…/gufo-ext-ling3_air.so` |
| plain install of stock Gufo | `gufo extension install <repo> --replace-builtin --yes` into a fresh store, then the same `diagnose` output; Gufo itself is unmodified |
| public-header isolation | the extension includes `gufo/ext/*` plus Gufo's `src/…` internals it needs (`cache/`, `core/image.hpp`), as Gufo's own models do; no Gufo header is patched |
| GPU-free probes | `ling3_air` (the host-only target) carries config, tokenizer, chat-template and probe tests; `ling3_air.config`, `.tokenizer`, `.probe` pass without touching HIP |
| advertised IDs | key `gguf:bailingmoe3`, model id `ling-3.2-air`, provider `ling3-air` |
| excluded keys | reported by `diagnose` for the other in-tree extensions, unchanged by this module |
| unsupported inputs | probe refuses non-GGUF and directories without weights; the runner returns `nullopt` for prompts with tools or images rather than dropping them |

## Real inference and advertised contracts

| contract | evidence |
| --- | --- |
| non-streaming completion | `POST /v1/completions` → 200, coherent text (`".\nA.Paris\nB.L"`), 5 prompt / 8 completion tokens |
| streaming | `stream:true` delivers SSE chunks through the same request |
| cancellation | the client closing the pipe mid-stream finishes the request as `finish=cancelled` — the server acted on it, which a disconnected client alone would not prove |
| continuation / restore | `tools/check_http_restore.py` requires successful private restores for a continued prefix, a repeated request and an edited suffix. All completion bytes match isolated cold runs. |
| checkpoint contents | At context 4096, `ling3_air.mtp` clears and restores all 154,697,732 serialized bytes, then checks eight greedy tokens and their logits. |
| scheduler profile | declared (`pass_tokens` 32, one layer group, decode width 1); the server logs `plan=serial-fallback`, `batch_width=1` |
| session isolation | `tools/check_http_restore.py` submits two concurrent requests with `--sessions 2`. Traces show two distinct slot banks on one engine; both return the exact completion bytes of their separate cold runs. Engine operations are serialized; this is an isolation test, not a throughput claim. |
| image input over HTTP | **works**: `POST /v1/chat/completions` with a base64 image returns 200 and a coherent completion. The markers are CONTROL tokens (`<|image_pad|>` 157157, `<|vision_start|>` 157158, `<|vision_end|>` 157159), so they are spliced by id and the row spans are recorded where they are placed; `Prefill` consumes a whole span in one pass (the host rejects a pass that ends inside one). 576 rows per image enter the trunk through `ForwardEmbedding`. | Without a marker token the encoded rows have no position to enter, so a request with an image now fails with an explicit `invalid_argument` instead of an empty 200. The encoder, preprocessing and engine-level image conditioning are verified separately (`ling3_air.vision`, `ling3_air.preprocess`, `ling3_air.image_forward`); what cannot be done is matching an HF reference on the same vocabulary, because this vocabulary cannot express the template |
| error recovery | an unloadable model, a request for a model id that is not loaded, and an invalid prompt all return structured errors instead of crashing |

## Independent numerical references and quality

| reference | result |
| --- | --- |
| numpy forward from the same mmproj weights | vision final embedding 3.2e-06, block traces 2.0e-05…9.2e-05 |
| llama.cpp's own vision encoder | embedding sum agrees to 0.04 %, block-0 stages to four decimals |
| Gufo's `ggml_dequant` / independent dequant | Q4_0 (MTP block) matches to 0.25 % on real block-42 rows |
| operator GPU suite | 31 checks, 0 failures |
| tokenizer | byte-exact on 184 reference cases (9,807 tokens) with ICU NFC |
| chat template | byte-identical to the reference jinja output on 18 cases |
| golden set vs llama.cpp CPU | **token-for-token from an identical id context** (`tools/golden_set_ids.py`): `arithmetic` identical for all 32 generated tokens; `plain` diverges at token 19, `tool-call` at token 16, `thinking-on` at token 2 — **1 of 4 cases**. The earlier text-based harness reported 93 / 3 / 110 / 157 characters, but it compared two different framings (`llama-completion -f` applies its own BOS/whitespace), so those numbers were partly a harness artefact and are superseded |
| MTP | acceptance measured (18 of 31 drafts, 58 %), spec loop not implemented — reason documented |

## Limitations and performance, published

`README.md` carries the number table and the gap list; `docs/quality.md` carries the
measured detail, including items that are *not* met; `IMPROVEMENTS.md` keeps every
optimisation attempt with before/after including the four that measured no gain and were
reverted.

| measure | value |
| --- | --- |
| decode, depth 0 | **17.02 tok/s = 34.1 %** of the 49.9 tok/s bandwidth roofline (gate 60 %, not met) |
| prefill, pp2048 | **23.16 tok/s = 0.37 %** of the 6,277 tok/s roofline (gate 70 %, not met) |
| golden-set equality | 1 of 4 cases token-identical (criterion not met), deviations documented |

Missing weights, a skipped mode and a failing gate are reported as such above; none is
recorded as a pass.
