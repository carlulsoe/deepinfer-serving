# ADR 0013: Decode drain by forwarding sequences

Status: accepted

## Context

The autoscaler drains decode instances (architecture piece 5). A draining instance should not
throw away the KV of its running requests. It should forward them, with their KV, to other decode
instances, and the forwarded requests must finish with the same tokens as uninterrupted runs.

## Decision

**A forward is a handoff taken mid-decode.** Between engine steps, every unfinished sequence has
`num_cached_tokens = len(token_ids) − 1`, the same invariant a prefill handoff has. `Handoff` is
generalized to that invariant (`0 < prompt_len ≤ num_cached_tokens = len(token_ids) − 1`).
`ServingLLMEngine.take_for_forward(rid)` builds the handoff, removes the sequence from the
scheduler's `running` or `prompted` queue, and frees its blocks and slot. The receiver uses the
unchanged `receive_handoff`, which sets `swa_valid_start = num_cached_tokens − (W + B)`, so it
never stores a record whose window predates the handoff.

**Same protocol as prefill handoffs.** The draining instance sends forwards from its outbox over
the peer link, with ack/nack, retry on `no_capacity`, and the same wire format (ADR 0005, 0007). It
reports `forwarded {to}`, and the target reports `admitted`.

**Drain sequence.**

1. `Entrypoint.drain(name)` stops routing to the instance and sends it the list of other live,
   non-draining decode instances.
2. The instance forwards round-robin to them on every loop iteration. That includes handoffs that
   arrive later from prefill instances routed there before the drain: it admits them, then forwards
   them.
3. `Entrypoint.retire(name)` waits until no request depends on the instance and its engine, inbox
   and outbox are empty, then shuts it down. Nothing is re-routed.

**States.** `decoding → forwarded → decoding`. An abort while forwarded goes to both the source and
the target (ADR 0012). Losing either one while forwarded re-routes the request.

## Consequences

- The forwarded request keeps its attempt: no re-prefill, no repeated tokens.
- The target never has the source's last decode snapshot, so a record the source would have stored
  at the next finish may be missing. That makes a later hit shorter, never wrong.
- No grace period yet: drain forwards immediately. A grace period ("let short requests finish
  first") is the autoscaler's policy, built on these two calls.
- A drain with no other live decode instance forwards nothing and keeps decoding.
