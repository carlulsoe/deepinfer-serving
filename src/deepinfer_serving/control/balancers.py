"""Load balancers as pure policy functions over a telemetry snapshot (architecture piece 4, ADR 0015).

The simulator and the Entrypoint call the same functions. Level 4 of the prefill LB (FIFO with a solved chunk size)
is ``CostModel.solve_chunk``; it runs in each rank's scheduler, not here.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

BUCKETS = (16_384, 65_536, 262_144)
"""Upper bounds of the prefill LB's context-length buckets: < 16K, 16-64K, 64-256K, >= 256K [DS]."""


def bucket_of(context_len: int, bounds: Sequence[int] = BUCKETS) -> int:
    return next((i for i, b in enumerate(bounds) if context_len < b), len(bounds))


def assign_buckets(current: Mapping[str, int], demand: Sequence[float]) -> dict[str, int]:
    """Level 1: give each bucket a share of the prefill groups proportional to its share of last round's quota
    demand (largest remainder), at least one per bucket with demand while groups last. Groups stay where they are
    when their bucket keeps its count, so a round moves as few groups as possible."""
    groups = sorted(current)
    total = sum(demand)
    if not groups or total <= 0:
        return dict(current)
    ideal = [len(groups) * d / total for d in demand]
    want = [math.floor(x) for x in ideal]
    for i in sorted(range(len(demand)), key=lambda i: ideal[i] - want[i], reverse=True)[: len(groups) - sum(want)]:
        want[i] += 1
    for i in range(len(demand)):  # a bucket with demand but no group takes one from the largest
        if demand[i] > 0 and want[i] == 0:
            j = max(range(len(demand)), key=lambda j: want[j])
            if want[j] > 1:
                want[j] -= 1
                want[i] += 1
    out: dict[str, int] = {}
    spare = []
    for g in groups:
        b = current[g]
        if 0 <= b < len(want) and want[b] > 0:
            out[g] = b
            want[b] -= 1
        else:
            spare.append(g)
    for g in spare:
        b = next(i for i, w in enumerate(want) if w > 0)
        out[g] = b
        want[b] -= 1
    return out


@dataclass(frozen=True)
class PrefillGroup:
    """One prefill instance (a DP group) as the LB sees it."""

    name: str
    bucket: int
    rank_quota_ns: tuple[float, ...]
    """Predicted prefill time of the work already queued on each DP rank."""


def route_prefill(
    queues: Mapping[int, Sequence[tuple[str, float]]], groups: Sequence[PrefillGroup], pull_limit_ns: float
) -> list[tuple[str, str, int]]:
    """Levels 2 and 3. ``queues`` maps a bucket to its waiting ``(request_id, cost_ns)``, oldest first.

    A group pulls while it has remaining quota, and puts the request on its least-loaded rank. Remaining quota is
    counted per rank, ``max(0, pull_limit_ns - queued)``, so one rank stuck on a long request does not stop its
    siblings from taking work. Groups with requests waiting in their own bucket pull first, the one with
    the most remaining quota first. Once no such group has quota left, the others steal the oldest request of the
    lowest non-empty bucket, so no request waits while some group has quota. Returns ``(request_id, group, rank)``
    in pull order.
    """
    heads = {b: 0 for b in queues}
    loads = {g.name: list(g.rank_quota_ns) for g in groups}
    out = []
    while True:
        avail = [b for b in sorted(queues) if heads[b] < len(queues[b])]
        best = None
        for g in sorted(groups, key=lambda g: g.name):
            load = sum(loads[g.name])
            remaining = sum(max(0.0, pull_limit_ns - x) for x in loads[g.name])
            if not avail or remaining <= 0:
                continue
            own = g.bucket in avail
            key = (own, remaining, -load / len(loads[g.name]))  # with no limit, the least loaded per rank
            if best is None or key > best[0]:
                best = (key, g.name, g.bucket if own else avail[0])
        if best is None:
            return out
        _, name, b = best
        rid, cost = queues[b][heads[b]]
        heads[b] += 1
        rank = min(range(len(loads[name])), key=lambda r: (loads[name][r], r))
        loads[name][rank] += cost
        out.append((rid, name, rank))


@dataclass(frozen=True)
class DecodeRank:
    """One decode DP rank as the LB sees it."""

    name: str
    requests: int
    tokens: int
    free_tokens: int


def route_decode(ranks: Sequence[DecodeRank], requests: Sequence[int], headroom: float = 1.05) -> list[int | None]:
    """DualPath's intra-group rule [DS] over every decode rank of every instance. ``requests`` are the KV token
    counts of the requests to place, in order. Returns the chosen rank index for each, or None when no rank has
    room for it.

    Threshold ``Z = headroom * (pending + assigned tokens) / #ranks``. Among ranks with room that stay at or under
    Z, pick the one with fewest requests; otherwise the one with fewest tokens.
    """
    reqs = [r.requests for r in ranks]
    toks = [r.tokens for r in ranks]
    free = [r.free_tokens for r in ranks]
    pending = sum(requests)
    out: list[int | None] = []
    for need in requests:
        z = headroom * (pending + sum(toks)) / max(1, len(ranks))
        fits = [i for i in range(len(ranks)) if free[i] >= need]
        under = [i for i in fits if toks[i] + need <= z]
        if under:
            i = min(under, key=lambda i: (reqs[i], toks[i], i))
        elif fits:
            i = min(fits, key=lambda i: (toks[i], reqs[i], i))
        else:
            out.append(None)
            continue
        reqs[i] += 1
        toks[i] += need
        free[i] -= need
        pending -= need
        out.append(i)
    return out


def route_vision(machines: Sequence[Sequence[int]], rng: random.Random) -> tuple[int, int]:
    """A random machine, then its GPU with the fewest image patches in flight [DS]. ``machines[m][g]`` is GPU g's
    patches in flight."""
    m = rng.randrange(len(machines))
    return m, min(range(len(machines[m])), key=lambda g: (machines[m][g], g))


def affinity(key: str, n: int) -> int:
    """Baseline: a stable hash of the session, as a prefix-affinity router would route."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % n
