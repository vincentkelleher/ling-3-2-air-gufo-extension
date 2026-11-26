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

## The cache adapter, read from the header (2026-11-27)

`CreateComponentCacheResources` must hand back a `cache::Adapter`
(`src/cache/adapter.hpp`), and that is the deep part of serving. Its contract, as
the four in-tree extensions implement it (DeepSeek's `ContinuationAdapter` is 669
lines):

| method | responsibility here |
| --- | --- |
| `GetCapabilities()` | which component kinds are supported |
| `Components()` | stable descriptors: one append-row component per MLA latent cache (row = 576 floats), one private-state component per KDA block (conv + delta) |
| `CompatibilityIdentity()` | model digests (from `models.lock.json`) + layout + context + settings that change state |
| `CreateSlot(guard)` | a slot owning one engine session |
| `Positions(slot)` | per-component row/position |
| `PlanPrefill(slot, first, end)` | chunk boundaries (our 32-token chunks) |
| `BeginRestore(slot, positions)` | validate capacity, guard the ranges being replaced |
| `CapturePrivate` / `LoadPrivate` (+ `…Piece`) | whole-buffer copies for KDA state, piece-wise required for arbitrary offsets |
| `CopyRowsOut` / `CopyRowsIn` | append rows of the MLA latent cache |
| `BorrowRowsView` | optional; empty means "copy instead of share" |
| `Validate(slot, positions)` | complete loads, one position per component; a failure latches |
| `Invalidate(slot)` | release borrowers and leave an empty, cold-prefillable slot |

Two facts that make a first version cheaper:

- **A null adapter is legal in principle**: `Components()` may return an empty
  span, `PlanPrefill` may return `{end}`, and then the cache simply never reuses a
  prefix. Serving would work with full prefill every request — enough to satisfy
  "HTTP serving" while reuse is added later. Worth trying first, because it removes
  `CopyRows*`, `BorrowRowsView` and the piece-wise paths entirely.
- The one unknown left in the runner itself is `sampling::SamplerState`: `SelectNext`
  must go through it rather than sample on the host, or temperature/top-k/top-p and
  the reasoning budget will diverge from the server's own accounting. Read
  `include/gufo/ext/sampling.hpp` and the LFM3 runner before writing it.

## Scale, measured against the installed API (2026-11-27)

The extension compiles against `/work/gufo-src` (the installed Gufo), not the 0.36.0 tree,
and the two differ: `include/gufo/ext/component_cache_resources.hpp` exists in 0.36.0 but
not in the installed tree, so the runner contract must be read from the installed
`src/ext/text.hpp`.

`TextModelRunner` has **13 pure virtuals** there: Descriptor, ResourceClaim,
SupportedPlans, Tokenize, RenderAndTokenize, Decode, CreateComponentCacheResources,
BindComponentState, ReconcileComponentState, Prefill, SelectNext, Advance,
CheckpointPosition. `TextRunnerState` adds only three virtuals
(`state_account`, `SetCancellationCheck`, `MeasuredResources`), which is the cheap part.

The expensive part is the cache side. The smallest in-tree implementation:

| file | lines |
| --- | --- |
| `src/models/deepseek_v4_flash/continuation_adapter.hpp` | 87 |
| `src/models/deepseek_v4_flash/continuation_adapter.cpp` | 669 |
| `src/models/deepseek_v4_flash/runner.cpp` | 931 |

i.e. ~1,700 lines of contract code even for the smallest model, of which the adapter
(components, positions, prefill planning, restore/validate/invalidate latches, private
state capture, row copy in/out, borrow views) is 756. That is the real cost of
`gufo serve llm`, and it is why the provider still registers `create = nullptr`.

A cheaper first milestone, if serving is attempted later: implement the runner with an
adapter that reports **no components** (empty `Components()`, `PlanPrefill` returning
`{end}`), which satisfies the interface without prefix reuse. That skips the row copies,
private-state capture and borrow views entirely and is worth trying before the full
adapter: it would make `gufo serve llm` work with full prefill per request.

## The no-component adapter: exact type surface (2026-11-27)

Read from the installed tree (`/work/gufo-src/src/cache/*.hpp`, `src/ext/text.hpp`). With
`Components()` empty, no row or private-state transfer ever happens, so four adapter
methods are unreachable in practice — but they still have to compile.

Types:

| name | shape |
| --- | --- |
| `cache::Rows` | `using Rows = std::uint64_t;` |
| `cache::ComponentId` | `{ std::uint32_t value; }` |
| `cache::ComponentDescriptor` | id, layout_version, kind, row_bytes, rows_per_chunk, state_bytes, window_rows |
| `cache::MutationGuard` | abstract, non-copyable; `CreateSlot(MutationGuard&)` ignores it |
| `Completion` | `explicit Completion(std::unique_ptr<CompletionSignal>)` |
| `CompletionSignal` | abstract (`Ready()` const noexcept, …) — needs a local always-ready subclass |
| `Stream` | abstract (`Synchronize()` …) — the signatures the stubs accept |
| `ComponentCacheResources` | **struct**: owner, adapter, stream/restore_stream/background_stream leases, allocate/share/device_only hooks |

Adapter methods that must compile but are unreachable with no components:
`BeginRestore`, `CapturePrivate(+Piece)`, `LoadPrivate(+Piece)`, `CopyRowsOut/In`,
`BorrowRowsView`. Stub strategy: an always-ready `CompletionSignal`, `Positions()` →
empty, `PlanPrefill(first, end)` → one `Rows` range `{first, end}`, `Validate()` → true,
`Invalidate()` → true, `GetCapabilities()`/`CompatibilityIdentity()` minimal but real
(model digest from `models.lock.json`, layout constant, context).

Order of work, so each step is independently testable:
1. `NullAdapter` + `Slot` + ready signal (compiles standalone against `adapter.hpp`).
2. `Ling3RunnerState : TextRunnerState` holding the `Engine`, position, and prompt.
3. `Ling3Runner : TextModelRunner`: Descriptor/ResourceClaim/SupportedPlans/Tokenize/
   RenderAndTokenize/Decode delegate to the existing tokenizer and chat template;
   Prefill → `PrefillChunk`; SelectNext → sampler; Advance → `ForwardToken`;
   CheckpointPosition → position; `CreateComponentCacheResources` → owner + NullAdapter;
   `BindComponentState` → state; `ReconcileComponentState` → no-op.
4. Register `create` in the provider and run `gufo serve llm` with one request.

Known unknowns left: the `sampling::SamplerState` API for `SelectNext`, and the fields
`TextRunnerDescriptor` requires (the installed API has no `TextSchedulingProfile`).
