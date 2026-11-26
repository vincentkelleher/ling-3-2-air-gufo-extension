# Runner plan: `TextModelRunner` for HTTP serving

The engine works and is measured; the extension does not serve yet. This is the
missing piece for success criterion 3 and the last large item before vision. It is
written to be executed directly.

## What must be implemented

`TextModelRunner` (`gufo/ext/text.hpp`) requires these pure virtuals, plus the
text-specific ones:

| method | what it does here |
| --- | --- |
| `Descriptor()` | identity, limits, scheduling profile, sampling defaults, speculative defaults |
| `ResourceClaim()` | what the runner asks the server to reserve (see "memory" below) |
| `Tokenize` / `TokenizeControl` | existing `text::Tokenizer` |
| `RenderAndTokenize` | existing `text::RenderChat` + tokenizer |
| `Decode` | existing `Tokenizer::Decode` |
| `Prefill(state, tokens, ...)` | `Engine::PrefillChunk` over the prompt, chunk by chunk |
| `Advance(state, token)` | one `Engine::ForwardToken` step |
| `CreateComponentCacheResources` | `cache::Adapter` over the KDA state and the MLA latent cache |
| `ReconcileComponentState`, `state_account` | bookkeeping for restored slots |
| `Descriptor().sampling_defaults` | temperature 1.0, top_k 20, top_p 0.95 (already parsed in `Ling3Config`) |
| `ToolDialectId()` | a new dialect for `<tool_call>` / `<arg_key>` / `<arg_value>` |

The complete pure-virtual set, read from `src/ext/text.hpp` (anything not listed
here has a default):

| # | method |
| --- | --- |
| 1 | `Descriptor()` |
| 2 | `ResourceClaim()` |
| 3 | `SupportedPlans()` |
| 4 | `Tokenize(...)` |
| 5 | `RenderAndTokenize(ChatRequest)` |
| 6 | `Decode(span<TextRunnerToken>)` |
| 7 | `CreateComponentCacheResources(ResourceLedger&, size_t)` |
| 8 | `BindComponentState(cache::Adapter&, cache::Slot&)` |
| 9 | `ReconcileComponentState(TextRunnerState&, size_t)` |
| 10 | `Prefill(state, prompt, offset, max_input_tokens)` |
| 11 | `SelectNext(state, sampler)` |
| 12 | `Advance(state, token)` |
| 13 | `CheckpointPosition(state)` |

Plus a `TextRunnerState` subclass that declares its buffers in a
`mem::StateDeclaration`, returns a `mem::StateAccount` from `state_account()`, and
calls `Ensure` before each write. `DecodeStep`/`DecodeBatch`/`AdvanceBatch` have
defaults built on `SelectNext`/`Advance`, so a first version needs only the
single-session path.

`SupportedPlans()` is the field to keep honest: it declares which inputs, batching
and state features the runner implements, and the server trusts it.

The tool dialect is the one piece with no code today: the chat template writes the
syntax, but the server needs the dialect to constrain and parse calls.

## Memory, which sets the default context

Per session, per layer:

- KDA (35 blocks): conv `9 x 4096` floats + delta `32 x 128 x 128` floats
  = **2.1 MiB per block, ~74 MiB per session**.
- MLA (7 trunk blocks + block 42): `(512 + 64)` floats per token
  = **2.25 KiB/token/block, 18 KiB per token per session**.

So the latent cache dominates: 32k context = 576 MiB, 128k = 2.3 GiB, 256k = 4.6 GiB
*per session*. A two-session server at 262,144 context would need 9.2 GiB before
weights, which fits in 128 GiB but must be declared honestly in `ResourceClaim`.
Recommended defaults: context 32,768 and two sessions (1.2 GiB), with the full
262,144 available by flag.

## Cache model

- **Append rows**: the MLA latent cache (one component per MLA block), position
  = token position, rows are `576` floats.
- **Private state**: each KDA block's conv and delta buffers (copied whole).
- **Identity** must include: model file digests (from `models.lock.json`), context
  size, and every setting that changes state (rope base, KDA gate bound, the
  per-block quantization layout). Anything that changes kept bytes invalidates.
- Bricked restores are the risk this exposes: the KDA state is a running product,
  so a restore at position `p` must equal an uninterrupted run — that is exactly
  what the cache tests check, and what the single-token verification already showed
  for one step.

## Scheduling profile

`TextSchedulingProfile` must have no zero fields. A 512-token prefill pass costs
~32 ms of work at 16 tok/s, so `pass_tokens` = the chunk size (32) and
`max_layer_groups` = 16 (a full 512-token prompt then yields every ~2 layers) keeps
decoders waiting under ~250 ms, matching the guidance in `docs/extension-api.md`.

## Order of work

1. **Runner skeleton**: `Descriptor` + `Tokenize`/`RenderAndTokenize`/`Decode` +
   `Prefill`/`Advance` with a single session and no cache. `gufo prompt` must
   produce the same text as `tests/forward_test.cpp` on the pinned prompt.
2. **Sampling**: apply the descriptor defaults; `gufo chat` with temperature 0 must
   match the golden set again.
3. **Cache adapter**: append rows for MLA, private state for KDA, identity from the
   model digest; then `gufo serve llm` with continuation and the functional cache
   suites.
4. **Tool dialect**: register the `<tool_call>` syntax so constrained calls work.
5. **MTP**: wire `ForwardMtp` once verification is cheap enough to pay (see
   `notes/prefill-design.md`; today it is 119 ms/token against 80, so it must stay
   off until the batched verifier is fast).

## Validation

- `gufo prompt` / `gufo chat` versus `tests/forward_test.cpp` on the pinned prompts
  (byte-identical text).
- Gufo's server functional suites: discovery, responses, stops, conversation,
  state edges, prefill scheduling, cache.
- The golden set and `tools/bench_report.py` rerun through the server, so the
  README numbers come from the served path rather than the test harness.
