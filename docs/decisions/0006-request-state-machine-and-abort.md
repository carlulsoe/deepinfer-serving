# ADR 0006: Request state machine, token streaming, and abort propagation

Status: accepted

## Context

A PD request lives in two engines and crosses a transport. Tokens come from both: the PROMPT engine
samples the first token, and the COMPLETION engine produces the rest. An abort can arrive at any
point: queued, mid-prefill, in transit, or decoding. Every request must terminate exactly once, and
every block and slot must be freed.

## Decision

**One owner of the client stream.** The coordinator (the Entrypoint's role; a test-side
`PDCoordinator` in milestone 2) owns each request's state and the client-facing token stream.
Engines report events; they never talk to clients.

**States** (from [architecture.md](../architecture.md), plus `failed`):

```
queued ─► prefilling ─► handed_off ─► decoding ─► finished
   │           │             │            │
   └───────────┴─────────────┴────────────┴─────► aborted | failed
prefilling ─► finished            (first token ends the request)
```

`forwarded` (decode drain) is reserved for milestone 4. The coordinator enforces the transitions
and records any invalid one as an error; the tests require zero errors.

**Tokens carry their index.** Each token event has `index = num_output_tokens - 1`. The handoff
goes from prefill to decode before the prefill instance reports the first token, and they use
different links. The coordinator therefore orders tokens by index, not by arrival. It reports a
request complete when it has a terminal event and every token up to the final index.

**Abort travels the request's own path.** The coordinator sends `abort` to the prefill instance
only:

- If the request is still in the prefill engine (queued or mid-prefill), the prefill instance
  aborts it there and reports `aborted`.
- Otherwise it forwards `abort` over the prefill→decode link. That link is FIFO, so the decode
  instance sees the abort after any handoff for the same request. If the decode instance holds the
  request, it aborts it and reports `aborted`.
- If the handoff is in the outbox awaiting an answer, the prefill instance marks it. A later `nack`
  makes the prefill instance report `aborted` and drop the buffer. A later `ack` means the decode
  instance has the request and will handle the forwarded abort.
- An abort for a request that already finished is ignored everywhere.

Each instance reports at most one terminal event per request. The coordinator still counts
duplicates, and the tests require zero.

## Consequences

- No tombstones are needed on the decode side; FIFO ordering on the prefill→decode link does that
  work. A future transport must keep per-request ordering between handoff and abort on that path.
- When the router can send a request to several decode instances (milestone 4), the prefill
  instance must forward the abort to the one it sent the handoff to. It already knows which.
- Abort latency in transit is bounded by the decode instance's loop, not by the coordinator.
