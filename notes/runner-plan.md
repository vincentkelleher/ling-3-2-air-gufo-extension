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

## The no-component adapter is rejected by the host (2026-11-27)

Implemented it: `src/runner.cpp` + `src/runner.hpp` provide a complete
`TextModelRunner` (13 virtuals) with a `NullAdapter` that reports no components,
`PlanPrefill` returning `{first, end}`, and always-ready completions, plus a state that
owns an `Engine`. It compiles, installs, and `gufo serve llm` **loads it** — the provider
factory is called, the GGUF is opened, sessions are created — and then the loader stops
with:

    Error loading model '…Q4_K_M-00001-of-00003.gguf': prefix index has no required
    components

So the host requires at least one component to build a prefix index, and the cheap
milestone written above (empty `Components()`) is **not viable**. Recorded here so nobody
spends the same effort twice.

What that leaves, in order of size:

1. **One append-rows component that is never read back.** Report a single component with
   the MLA latent cache's row width (`latent * 4` = 2304 bytes) and implement
   `CopyRowsOut`/`CopyRowsIn` over a contiguous staging copy of the latent rows. Serving
   then works with full prefill per request; reuse stays off. This is the smallest change
   that satisfies "has components" and keeps every transfer trivially correct.
2. **The real adapter**: one append-rows component per MLA layer plus private state for
   the KDA conv/delta buffers, with `Validate`/`Invalidate` latches and
   `ReconcileComponentState` restoring each slot's frontier. That is the 756-line
   adaptation the size table above measures, and it is what prefix reuse and rollback
   actually need.

The runner's sampling is greedy (`SelectNext` ignores the sampler), which is exact at
temperature 0; honouring temperature/top-k/top-p is a separate small step.

## Where the adapter stops (2026-11-27)

The runner now registers a **private-state component** instead of an empty component set:

- the component is the engine's snapshot buffer (KDA conv/delta, MLA latent caches and
  positions), sized exactly as `Engine::Create` allocates it —
  `n_layer_total x (9 * kda_inner + heads * kda_dim^2 + context * kv_dim)` floats, exposed
  through `Engine::SnapshotData()`/`SnapshotFloats()`;
- `CapturePrivate`/`LoadPrivate` copy that buffer with `hipMemcpy` (synchronous, on the
  default stream — correct but serialised; the asynchronous form needs the device stream
  behind `cache::Stream`);
- `ComponentCacheResources` now supplies `allocate` (host storage handed back through
  `Payload::Committed`), `device_only` (false) and `share` (nullopt).

`gufo serve llm` gets through "prefix index has no required components" and now fails at
`phase=sessions` with `bad_function_call`, i.e. one more empty hook is being called. The
next value to fill is almost certainly `ComponentCacheResources::stream` (a `StreamLease`,
declared outside `src/cache/*.hpp`, so its shape still has to be read) — an empty lease
called during session creation is exactly a `bad_function_call`. After that come the
device-stream downcast for asynchronous copies and the server's streaming/cancellation
tests.

So the ladder is now: empty components (rejected) → one private-state component (accepted,
`allocate`/`device_only`/`share` needed) → stream lease + async copies → server test suite.

## Serving: what works, and the one crash left (2026-11-27)

Verified through `gufo serve llm` (installed module, no Gufo changes):

| case | result |
| --- | --- |
| one completion request | 200, `text=".\nA.Paris\nB.L"`, 5 prompt / 8 completion tokens, prefill 15.2 tok/s, decode 12.6 tok/s |
| 406-token prompt, `stream=true` | SSE chunks delivered; the client's `head -c` closed the pipe and the server finished it as `finish=cancelled` — cancellation works |
| two sequential requests | both complete, `cache_reuse=none` |
| second request's frontier capture | **crash** |

The crash is precise: after the first request the cache tries a frontier capture and
refuses with `frontier_capture_refused reason=empty ledger handle`, then the process takes
SIGSEGV. The cause is in our `ComponentCacheResources::allocate`: it hands back host
storage but builds the `Payload` with a default-constructed `ResourceCharge`, so the
capture has no ledger handle to charge. The fix is to allocate against the ledger the host
passes to `CreateComponentCacheResources`:

    ResourceLedger::Reserve(ResourceCategory, bytes) -> ResourceReservation
    // then turn the reservation into a ResourceCharge for Payload::Committed

`ResourceCharge` is only constructible through the ledger (`ResourceCharge() = default`
gives exactly the empty handle that crashes), so `Reserve(...)`'s completion step is the
one API left to read. Everything else in the serving path is in place:
positions reported per slot, prefill clamped to the engine's chunk width (a 330-token pass
used to overflow the 256-row chunk buffer and produced "invalid interrupted prefill
frontier"), engines owned for the process so shutdown cannot free them after the HIP
context, and a stream lease whose `Synchronize` does nothing because every copy is already
synchronous.

## Serving state after the ledger work (2026-11-27)

| case | result |
| --- | --- |
| one completion request | 200, coherent text, prefill 15.2 / decode 12.5 tok/s |
| 406-token streaming prompt | SSE chunks; client disconnect finishes it as `finish=cancelled` (cancellation works) |
| frontier capture | **succeeds** — `cache_captures=1`, 502 MB captured |
| next request after a capture | **SIGSEGV inside the host**, no refusal and no log line from this extension |

Fixes that landed on the way, each from a concrete host error:

- `PlanPrefill` returns the ascending boundaries each 32-row pass ends on (returning
  `{first, end}` was "invalid adapter prefill plan");
- prefill is clamped to the engine's chunk width — a 330-token pass overflowed the
  256-row chunk buffer and surfaced as "invalid interrupted prefill frontier";
- `Positions()` reports the slot's frontier as a private component position (empty was
  "incomplete capture positions");
- capture storage is charged through `ResourceLedger::Reserve(category, bytes).Convert()`
  (a default `ResourceCharge` was "empty ledger handle");
- the snapshot claim is sized from the **serving** context, not the model's 262,144-token
  trained context — sizing from the latter claimed 26 GB per session;
- engines live for the process, and the stream lease's `Synchronize` does nothing because
  every copy is already synchronous (both removed earlier SIGTERM/SIGSEGV paths).

What remains is not in this extension as far as the logs show: the crash follows a
successful capture, on the next request, before any adapter call we can observe. Chasing
it needs a stack trace from the host process (gdb or a coredump backtrace), which this
environment does not provide, so serving is **single-request verified and multi-request
unverified**, and is reported that way rather than claimed as working.

## The null reservation was ours (2026-11-27)

The SIGSEGV in `ComponentTextCache::Acquire` with `state=0x0` was not a host defect. The
pool binds a state per request; our `BindComponentState` created a **new engine every
time**, and a second engine re-registers the same 80 GiB mapping — `hipHostRegister`
refuses an already-registered range, engine creation returned null, and the pool was left
with no state, which is exactly the null reservation the cache then dereferenced.

Fix: the adapter keeps one engine per slot and `BindComponentState` reuses it
(`Ling3ComponentAdapter::Existing`). After that, repeated requests and streaming run with
no crash at all (`fatal_signal` count 0 in the server log).

Lesson worth keeping: when the host reports a null pointer that "cannot be ours", check
whether an earlier call of ours silently returned null. The `bad_function_call` chain had
the same shape — each one was a hook we had left empty rather than a host bug.

## Concurrent sessions hang: the MoE scratch is process-wide (2026-11-27)

`gufo serve llm --sessions 2` with two concurrent requests **hangs** (no completion after an
hour; the test had to be killed). The cause is in this extension, not the host:
`src/hip/moe.hip`'s `ScratchFor` is a function-local `static` holding the quantised
activation for the MoE dot. One engine is fine, but two sessions run two engines on two
threads and share that buffer, so one session quantises into it while the other reads — a
data race that also re-allocates the buffer mid-flight (`ScratchFor` resizes when a wider
call arrives).

So the supported mode is **one session**: requests queue and run one at a time, which is
what all the verified numbers above were measured with (`sessions=1`, `batch_width=1`).

**Update**: that fix is now in — the buffers live in `Engine` (`moe_q8/moe_scale/moe_sums`,
allocated with the other scratch) and `LaunchMoeGather` takes them, so `moe.hip` has no
`static` and no resize path. Greedy tokens are unchanged (16.87 tok/s, same token list).

Two concurrent sessions **still hang** (bounded run killed at 420 s), so the static scratch
was not the whole story. The next suspect is the adapter itself: `Ling3ComponentAdapter`
keeps `engines_` and `positions_` in `std::map`s that the scheduler threads touch without a
mutex (`EngineFor`, `OnPosition`, `Existing`, `Adopt`), which is a data race on the very
path a second session takes. The mutex is now in (all four map accesses lock it) and two sessions **still hang**
(killed at 420 s), so the adapter's maps were not the cause either. What is left is the
engine itself: two `Engine` instances on two threads against one HIP device — each owns its
own stream, but the weight mapping, the host-registered ranges and the shared `gpus` are
process-wide. That needs either host-level serialisation of sessions or a single engine
shared by all sessions, which is a different design from the one-session-per-slot contract.

## Vision through the server: wired, then reverted (2026-11-27)

Image input was implemented on the runner and **reverted**, because it crashes:

- `PreparePrompt` accepted `ChatMessage::images`, decoded and encoded each image with the
  extension's `VisionEncoder`, built the `<|vision_start|>…<|image_pad|>…<|vision_end|>`
  block, and returned the projected rows through a `Ling3ImageContext : TextPromptContext`
  (the host hands it back to `SetPromptContext`, which is the designed handoff).
  `Prefill` then fed each marker position with `Engine::ForwardEmbedding` and every other
  position with `ForwardToken`.
- With the mmproj supplied, the host reported the model as image-capable
  (`supports_vision=1`, "serving vision path … -> loaded"), decoded the request, and then
  **segfaulted** — before calling `PreparePrompt` at all in the first attempt, and on the
  second with this backtrace:

        hip::Device::WaitActiveStreams(hip::Stream*, bool)
        hip::getStream(ihipStream_t*, bool)
        hip::hipMemcpyAsync(...)
        Engine::ForwardFromEmbedded                     engine.cpp:769  (D2D copy of m.x)
        Engine::ForwardEmbedding                        engine.cpp:466
        Ling3Runner::Prefill                            runner.cpp:586

  i.e. a HIP-internal crash inside a plain device-to-device copy, on our path, with the
  engine's stream. The same copy runs for every ordinary token, so what differs is the
  per-token `ForwardEmbedding` drive and whatever the host did with the device when the
  request carried an image.

Since the provider must not advertise a path that crashes, `supports_vision` is back to
false and image input stays an **engine-level** capability (verified: numpy 3.2e-06,
llama.cpp encoder 0.04 %, correct description through `tests/image_forward_test.cpp`), not
a server feature. Next owner should reproduce with the trace at
`Engine::ForwardEmbedding` and check whether the host swaps or tears down the HIP context
when an image request is admitted.
