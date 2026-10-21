# ADR 0015: Load balancers as pure policy functions

Status: accepted. Replaces the least-loaded routing of
[ADR 0011](0011-entrypoint-check-and-swa-directory.md); CHECK, the SWA directory and pulls stay.

## Context

DeepSeek describes four levels of prefill balancing, a decode rule borrowed from DualPath, and a
vision-encoder rule [DS] ([background](../background/deepseek-serving-system.md#the-four-load-balancers)).
The descriptions leave details open. The policies must run unchanged in the simulator (ADR 0014) and
in the live Entrypoint (architecture piece 4).

## Decision

**Pure functions over a snapshot** (`control/balancers.py`). The callers keep all state; the
functions only decide.

- **Level 1, `assign_buckets`.** Prefill groups are spread over the context-length buckets
  (< 16K, 16–64K, 64–256K, ≥ 256K) in proportion to each bucket's share of last round's
  compute-quota demand. Shares use the largest remainder. Every bucket with demand gets at least one
  group while groups last, and a group moves only when its bucket loses a slot.
- **Levels 2 and 3, `route_prefill`.**
  - Requests wait at the LB, and groups pull them.
  - A group's remaining quota is counted **per rank**, as `Σ max(0, pull_limit − queued)`.
  - Groups with requests waiting in their own bucket pull first, the most remaining quota first.
  - Once none of those has quota left, the others **steal** the oldest request of the lowest
    non-empty bucket.
  - The request goes to the group's least-loaded rank.
- **Level 4** is `CostModel.solve_chunk` in each rank's scheduler (ADR 0014).
- **Decode, `route_decode`.** This is DualPath's intra-group rule, applied over every decode rank of
  every instance.
  - Compute `Z = 1.05 × (pending + assigned tokens) / #ranks`.
  - Among ranks with room that stay at or under Z, take the one with the fewest requests; otherwise
    the one with the fewest tokens.
  - A rank without room for the request's context is skipped. When no rank has room, the request
    waits.
- **Vision, `route_vision`.** Pick a random machine, then its GPU with the fewest patches in flight.
- **Baselines** (simulator only):
  - round-robin over ranks;
  - prefix affinity, which hashes the session to a rank and falls back to the next rank with room
    for decode.

  Neither one uses the pull queue.

**Ambiguities resolved here [ours].**

- **Remaining quota is per rank, not per group.** With a group-level sum, one rank stuck on a
  700K-token prompt (35 s of quota) stopped its sibling ranks from pulling, and its bucket's queue
  waited behind it. Measured before the change: p99 TTFT of 9.9 s against 2.1 s after it.
- **Stealing.** DeepSeek does not say what an idle group does when its own bucket is empty. Strict
  isolation starved a bucket whose only group was busy (p99 TTFT of 31 s, with idle groups
  elsewhere). Stealing only after own-bucket groups are out of quota keeps most of the isolation.
- **Per-bucket batch caps** (Q&A only) are not modelled; every bucket uses the same budget.
- **The all-reduce MIN of the budget** across a DP group has no effect while every rank has the same
  budget. It is not modelled.
- **Decode is chosen when the request is routed, not after its prefill.** The `add` message
  carries the decode instance (ADR 0011). DeepSeek has prefill ask the Decode LB after prefilling.
  The rule is the same; only the snapshot is earlier.

**In the live Entrypoint**, each mini instance is one group of one rank, all in one bucket, so levels
1 and 4 do not apply.

- Every request is CHECKed first. Its predicted cost, `cost(len − num_cached, num_cached)`, is its
  quota.
- `route_prefill` then picks the prefill instance, and `route_decode` picks the decode instance.
  For decode, a rank's tokens are the prompt plus streamed tokens of the requests routed to it, and
  its room is `num_gpu_blocks × block_size`.
- The default pull limit is infinite, which means: route on arrival to the instance with the least
  queued quota.
- With a finite limit, requests wait in `Entrypoint.pending`. Being queued there is the `queued`
  state with no attempt. Waiting requests are routed again whenever events arrive, and an abort
  while waiting sends nothing.
- A re-route goes through the same queue.

## Consequences

Measured on the reference traces of `tests/test_balancers.py` (cost model of ADR 0014, two prefill
groups of two ranks, two decode instances of two ranks):

| Trace | Policy | Prefill balance | Decode balance | TTFT p50 / p99 | TBT p50 / p99 |
|---|---|---|---|---|---|
| mixed, seed 1 | ours | **0.62** | **0.84** | **0.066 / 1.5 s** | **1.10 / 1.36 ms** |
| | round-robin | 0.60 | 0.79 | 0.083 / 35 s | 1.11 / 1.51 ms |
| | affinity | 0.59 | 0.73 | 0.092 / 33 s | 1.13 / 3.13 ms |
| mixed, seed 2 | ours | **0.72** | **0.85** | **0.064 / 3.1 s** | **1.11 / 1.34 ms** |
| | round-robin | 0.65 | 0.79 | 0.133 / 21 s | 1.13 / 1.45 ms |
| | affinity | 0.66 | 0.74 | 0.141 / 18 s | 1.15 / 2.70 ms |
| decode-heavy | ours | — | **0.90** | — | **1.26 / 1.52 ms** |
| | round-robin | — | 0.87 | — | 1.29 / 1.62 ms |
| | affinity | — | 0.84 | — | 1.31 / 1.76 ms |

How to read the table:

- **Balance** is Σ mean rank time / Σ max rank time over steps, where 1.0 is a perfectly balanced
  bound. DeepSeek reports about 0.8 in production [DS].
- **Ours beats both baselines on every metric.** The biggest gain is TTFT p99: pulling late keeps
  short requests out from behind long ones.
- **Vision:** wait p50 / p90 / p99 is 491 / 930 / 1472 ms for ours, 495 / 960 / 1584 ms for
  round-robin, and 621 / 1482 / 2656 ms for affinity. That beats affinity everywhere, but beats
  round-robin only in the tail. At the median they tie, because round-robin is already even when
  image sizes are similar.
- **Bucketing (level 1)** raised prefill balance by about 0.03 but raised median TTFT (0.066 against
  0.045 s on seed 1) in an ablation. It earns its place only if balance matters more than median
  latency, which is a judgment for milestone 5, with real costs.
- **Tier 1** checks correctness only. The routing, failure and drain oracles all pass with these
  balancers in the path, and two tests cover the pull queue
  (`tests/test_entrypoint_balancers.py`).
