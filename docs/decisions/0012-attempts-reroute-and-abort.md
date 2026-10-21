# ADR 0012: Attempts, re-routing after instance loss, and abort ownership

Status: accepted. Replaces the abort path of [ADR 0006](0006-request-state-machine-and-abort.md);
its states, token indices and exactly-once rule stay.

## Context

With several instances, any of them can die mid-request. The request must still finish with the
right tokens, and the survivors must not leak. ADR 0006 routed aborts along the request's own path
(prefill, then FIFO to decode). That needs every hop alive, and it needs forwarding state on every
hop once a request can move between decode instances (ADR 0013).

## Decision

**A request runs as a sequence of attempts.** Instances see attempt ids, `<request_id>/<n>`, and
treat them as opaque. The Entrypoint maps events back to the request and to the attempt that sent
them. Events from an attempt that is no longer current are ignored, with one exception below.

**Events are ordered by progress, not by arrival.** One attempt's events come from several instances
over separate links, so the Entrypoint can read decode's `admitted` before prefill's `handed_off`, or
a forward target's `admitted` before the source's `forwarded`. Every handoff carries a hop number (1
from prefill, +1 per forward). Each event maps to a point `(hop, arrived)`: prefilling `(0, 1)`, sent
`(h, 0)`, admitted `(h, 1)`. An attempt's progress only moves forward. A jump fills in the skipped
states, so histories stay canonical, and an event behind the current progress is ignored. Tokens
from the current attempt are accepted even after it finished, because the first token (from
prefill) can arrive after the last (from decode). A prefilling event read late still records the hit.

**Instance loss.** The Entrypoint learns of a death from EOF or a reset on the control link, from a
`fatal` report (an engine exception, whose traceback it records in `errors`), or from its own `kill`,
used for fault injection. It then:

1. drops the instance from routing and from the SWA directory, and tells every survivor
   (`peer_dead`). A survivor drops outbox entries addressed to the dead instance and falls back on
   any pull it was waiting for. Survivors also notice on their own, as EOF or a reset on the peer
   link.
2. for every non-terminal request whose current attempt depends on the dead instance, aborts the
   attempt on the live instances in its path and starts a new attempt.
   - Before decode, an attempt depends on both its prefill and its decode instance.
   - While decoding, it depends on the decode instance.
   - While forwarded, it depends on the source and the target decode instance.
3. re-runs every request that finished but is still missing tokens, if the dead instance is
   on its attempt's path. Those tokens were sent by that instance and died unread.

An attempt's path records every instance that has held it, from `admitted` and `forwarded` events,
whatever order they are read in, even after the request finished. A request missing tokens while an
instance on its path is dead is re-run as soon as both facts are known: when the instance dies, when
the request finishes, or when a late event adds a dead instance to the path. If a `forwarded` event
names a target whose death was already processed, the request is re-routed at once. If a `forwarded` event names a target whose death was
already processed, the request is re-routed at once.

**A new attempt re-prefills the prompt plus the tokens already streamed.** It has
`max_tokens` reduced by that count, and its token indices are offset by it. If nothing remains (the
budget is used up, or EOS was streamed), the request finishes without a new attempt. Re-prefill
uses the cache like any request, so the prompt part usually hits what its first prefill stored. In mini this is
exact, because prefilling a token does the same arithmetic as decoding it (contract §8), and the
failure tests check it. On real DeepInfer we expect it to hold under the deterministic mode the
oracles already require; that is unverified until milestone 5. A token index that
arrives twice must carry the same token; a mismatch is recorded as an error.

**The SWA directory forgets an entry with its last holder** (ADR 0011), so it is bounded by the SWA
tiers' capacity.

**The Entrypoint owns aborts.** `abort` makes the request terminal (`aborted`) at once, then sends
`abort <attempt>` to every live instance in the attempt's path. Each instance releases whatever it
holds:

- the sequence in its engine;
- a pending pull;
- a queued handoff in its inbox, which it nacks with `aborted`;
- an outbox entry. If the entry is awaiting an answer, it is marked, and a nack drops it.

**A stale attempt admitted anywhere is aborted.** An abort can overtake its handoff: the decode
instance reads the control link first, finds nothing, then reads the handoff and admits it. Any
`admitted` event for an attempt that is terminal or not current makes the Entrypoint send that
instance an abort. This one rule covers aborts that overtake handoffs, re-routes, and forwards.

## Consequences

- No hop keeps forwarding or tombstone state, so aborts do not depend on any hop staying alive.
- Instances no longer report `aborted` terminals. "Exactly once" holds because the Entrypoint sets
  the terminal state, and it still counts a second terminal from the same attempt as an error.
  Freeing everything is checked, as before, by allocator counts returning to baseline on every
  live instance.
- A history can now contain `queued` more than once (one per attempt), and an abort in transit ends
  at `handed_off → aborted` without a `decoding` step.
- Re-prefill repeats work the lost decode instance had done. Moving that KV instead needs the
  source alive, which is the drain case (ADR 0013).
- Instances never block on a write. Each link has a writer thread with an ordered queue. Otherwise two
  instances writing large blobs to each other (a handoff one way, an SWA pull reply the other) could
  both block with nobody reading.
- Losing the Entrypoint loses the requests; making it fault tolerant is out of scope.
