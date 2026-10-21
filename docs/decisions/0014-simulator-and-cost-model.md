# ADR 0014: The tier-0 simulator, its GPU-time model, and the trace format

Status: accepted

## Context

The balancers and the autoscaler are performance policy. Determinism oracles cannot judge them, and
tier 1 is too slow and too unlike SM100 to measure them ([roadmap](../roadmap.md#test-tiers)). They
need a simulator that replays request traces against a model of GPU time (roadmap S1). The same
policy code must also run in the live Entrypoint.

## Decision

**A discrete-event simulator** (`control/sim.py`), with time in nanoseconds.

- **Instances.** An instance is one DP group of `gpus` ranks. Every rank of an instance steps
  together, because the MoE layers synchronize them every step. A step lasts as long as its slowest
  rank. This lockstep is what makes balance inside a group matter.
- **Prefill.** Each rank takes its queue FIFO. Each step it fills a `quota_limit_ns` budget with
  chunks whose size comes from `cost(q, c)` (the LB's level 4). Only the head of the queue may be
  split; an empty rank always makes progress. When a prefill completes, it emits the first token and
  hands off to decode after a transfer delay proportional to the context.
- **Decode.** Each running request gets one token per step. A request admitted mid-step joins the
  next step.
- **Vision.** Each vision-encoder GPU encodes one image at a time, FIFO, in time proportional to its
  patch count.
- **Failures and start-up.**
  - A failed instance loses its in-flight step.
  - Its prefill jobs restart from the hit.
  - Its decode jobs re-prefill the prompt plus the tokens they have streamed, as ADR 0012 does.
    The token this produces is not a first token.
  - Instances start in `startup_s`, or in `warm_startup_s` when they use the warm reserve.
- **Same-instant events are scheduled together.** Routing and step starts run once, after every
  event at the current instant. Without this, requests that arrive together would start one step
  each, an artefact of event order.

**The GPU-time model** (`control/cost.py`) works per rank.

- **Prefill:** `cost(q, c) = a·q + b·q·(c + (q + 1)/2)`, plus a fixed cost per step. The first term
  is linear work. The second is attention over the prefix and, causally, within the chunk. A chunk is
  the largest `q ≤ min(remaining, 32768)` that fits the budget (the quadratic, solved).
- **Decode:** a step costs `base + per_request·n + per_kv·Σ context`.

The defaults are **placeholders [ours]** that reproduce DeepSeek's reported production figures [DS]:

- prefill: 60K uncached tok/s per GPU at 74K context;
- decode: 14K tok/s per GPU at 230 tok/s per user, with 61 requests of 77K context.

`test_cost_model_placeholders_reproduce_the_reported_figures` checks that. The split between the
linear and attention terms, the step overheads, the 100 ms budget and the 8M-token KV capacity are
guesses. Policy comparisons in the simulator are relative to this model until milestone 5 fits it to
measured profiles.

**Traces are JSONL**, one request per line: `id`, `arrival_s`, `session`, `input_len`, `hit_len`,
`output_len`, `patches`.

- **Hit length is data.** With a shared, equal-cost cache the hit does not depend on which instance
  serves the request (ADR 0001), so a replay under any balancer keeps it.
- **From tier 1:** `trace_from_records` builds a trace from the Entrypoint's records, using their
  submit time, prompt length, applied hit and output length.
- **Synthetic traces:** `synthetic_trace` generates multi-turn sessions with log-normal lengths.
  Turn k + 1 hits the block-aligned prefix of turn k's context, capped as the hit rule caps it.
  Sessions are open-loop: a turn's arrival does not wait for the previous turn's reply.

## Consequences

- **Analytical checks:** M/D/1 queueing within 3% (Pollaczek–Khinchine), uniform load with no
  queueing, linear growth under overload, chunking, decode step times, and exact timing of re-routes
  after a lost prefill or decode (`tests/test_sim.py`).
- **Python speed:** the simulator simulates every token, so a few thousand requests take seconds.
  Reference traces are sized for tests, not for production-scale studies.
- **Not modelled:**
  - the store: hit length is given;
  - EPLB and MoE imbalance;
  - DSpark (one token per step);
  - network contention;
  - decode KV growth beyond admission;
  - the all-reduce MIN of the chunk budget, since every rank already has the same budget.
- The cost model is unverified. Re-fitting it is part of milestone 5.
