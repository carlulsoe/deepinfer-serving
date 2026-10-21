"""Load balancers (roadmap S2, ADR 0015): the policy rules, invariants on every trace, and quality on reference
traces against round-robin and prefix-affinity baselines.

Quality numbers are relative to the uncalibrated cost model (ADR 0014); they say which policy is better on the
modelled hardware, not by how much on real hardware.
"""

import functools
import random

import pytest

from deepinfer_serving.control.balancers import (
    DecodeRank,
    PrefillGroup,
    assign_buckets,
    bucket_of,
    route_decode,
    route_prefill,
    route_vision,
)
from deepinfer_serving.control.sim import POLICIES, Failure, SimConfig, SimResult, percentile, simulate
from deepinfer_serving.control.trace import Request, synthetic_trace

# ---- rules ----


def test_buckets_follow_deepseek_boundaries() -> None:
    assert [bucket_of(n) for n in (1, 16_383, 16_384, 65_535, 65_536, 262_143, 262_144, 10**6)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_bucket_assignment_is_proportional_and_moves_few_groups() -> None:
    current = {f"g{i}": 0 for i in range(8)}
    out = assign_buckets(current, [6.0, 2.0, 0.0, 0.0])
    assert sorted(out.values()) == [0] * 6 + [1] * 2
    assert sum(1 for g in current if out[g] != current[g]) == 2
    out = assign_buckets(current, [100.0, 0.0, 0.0, 1.0])  # a small bucket with demand still gets a group
    assert sorted(out.values()) == [0] * 7 + [3]
    assert assign_buckets(current, [0.0] * 4) == current


def test_prefill_pull_prefers_own_bucket_then_steals() -> None:
    groups = [PrefillGroup("a", 0, (0.0, 0.0)), PrefillGroup("b", 1, (0.0, 0.0))]
    queues = {0: [("s1", 10.0), ("s2", 10.0), ("s3", 10.0)], 1: []}
    out = route_prefill(queues, groups, pull_limit_ns=10.0)
    # a takes one per rank; then b, whose own bucket is empty, steals the third
    assert out == [("s1", "a", 0), ("s2", "a", 1), ("s3", "b", 0)]


def test_prefill_pull_counts_quota_per_rank() -> None:
    """A rank stuck on a long request does not stop its sibling from pulling."""
    groups = [PrefillGroup("a", 0, (1e9, 0.0)), PrefillGroup("b", 0, (6.0, 6.0))]
    out = route_prefill({0: [("x", 1.0)]}, groups, pull_limit_ns=10.0)
    assert out == [("x", "a", 1)]  # a has 10 left on rank 1; b has 4 + 4 left
    assert route_prefill({0: [("x", 1.0)]}, [PrefillGroup("c", 0, (10.0,))], 10.0) == []


def test_prefill_without_limit_routes_to_least_loaded() -> None:
    groups = [PrefillGroup("a", 0, (3.0,)), PrefillGroup("b", 0, (1.0,))]
    assert route_prefill({0: [("x", 5.0), ("y", 1.0)]}, groups, float("inf")) == [("x", "b", 0), ("y", "a", 0)]


def test_decode_follows_dualpath_rule() -> None:
    ranks = [DecodeRank("a", 2, 1_000, 10_000), DecodeRank("b", 1, 5_000, 10_000), DecodeRank("c", 5, 900, 10_000)]
    # Z = 1.05 * (100 + 6900) / 3 = 2450: a and c stay under it, a has fewer requests (c has fewer tokens)
    assert route_decode(ranks, [100]) == [0]
    # a long request fits under Z nowhere: fewest tokens wins
    assert route_decode(ranks, [9_000]) == [2]
    full = [DecodeRank("a", 0, 0, 50), DecodeRank("b", 0, 0, 10)]
    assert route_decode(full, [60, 40, 40]) == [None, 0, None]


def test_vision_picks_the_least_loaded_gpu_of_a_random_machine() -> None:
    machines = [[5, 1, 3], [0, 0]]
    picks = {route_vision(machines, random.Random(seed)) for seed in range(20)}
    assert picks == {(0, 1), (1, 0)}


# ---- invariants and quality ----

MIXED = {seed: synthetic_trace(seed, sessions=600, duration_s=150, image_fraction=0.3) for seed in (1, 2)}
DECODE_HEAVY = synthetic_trace(1, sessions=300, duration_s=100, output=2_000)
VISION_HEAVY = synthetic_trace(1, sessions=300, duration_s=100, image_fraction=1.0, first_input=2_000, user_tokens=200)
FAILURES = [Failure(30, "p0"), Failure(60, "d1")]


@functools.cache
def _run(trace: str, policy: str) -> SimResult:
    if trace.startswith("mixed"):
        return simulate(SimConfig([2, 2], [2, 2], vision=[2, 2], policy=policy), MIXED[int(trace[-1])])
    if trace == "failures":
        return simulate(SimConfig([2, 2], [2, 2], vision=[2, 2], policy=policy, failures=FAILURES), MIXED[1])
    if trace == "decode":
        return simulate(SimConfig([4, 4, 4], [2, 2], policy=policy), DECODE_HEAVY)
    return simulate(SimConfig([4], [4], vision=[4, 4], vision_ns_per_patch=480_000, policy=policy), VISION_HEAVY)


TRACES = {
    "mixed-1": MIXED[1],
    "mixed-2": MIXED[2],
    "failures": MIXED[1],
    "decode": DECODE_HEAVY,
    "vision": VISION_HEAVY,
}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("trace", list(TRACES))
def test_invariants_hold(trace: str, policy: str) -> None:
    """Nothing lost or duplicated, every token accounted for, quotas and KV capacity respected."""
    result = _run(trace, policy)
    requests: list[Request] = TRACES[trace]
    assert set(result.requests) == {r.id for r in requests}
    for r in requests:
        stats = result.requests[r.id]
        assert stats.completions == 1
        assert len(stats.gaps_ns) == r.output_len - 1
    assert result.budget_overruns == 0
    assert result.kv_overruns == 0
    if trace == "failures":
        assert sum(s.reroutes for s in result.requests.values()) > 0


def _metrics(result: SimResult) -> dict[str, float]:
    ttft, tbt = result.ttft(), result.tbt()
    return {
        "prefill_balance": -result.prefill_balance,  # negated: lower is better for every entry
        "decode_balance": -result.decode_balance,
        "ttft_p50": percentile(ttft, 50),
        "ttft_p99": percentile(ttft, 99),
        "tbt_p50": percentile(tbt, 50),
        "tbt_p99": percentile(tbt, 99),
    }


@pytest.mark.parametrize("trace", ["mixed-1", "mixed-2"])
def test_balancers_beat_baselines_on_mixed_traces(trace: str) -> None:
    ours = _metrics(_run(trace, "deepseek"))
    for baseline in ("round_robin", "affinity"):
        theirs = _metrics(_run(trace, baseline))
        worse = {k: (ours[k], theirs[k]) for k in ours if ours[k] >= theirs[k]}
        assert worse == {}, f"not better than {baseline}"


def test_decode_balancer_beats_baselines_on_a_decode_heavy_trace() -> None:
    ours = _run("decode", "deepseek")
    for baseline in ("round_robin", "affinity"):
        theirs = _run("decode", baseline)
        assert ours.decode_balance > theirs.decode_balance
        for p in (50, 90, 99):
            assert percentile(ours.tbt(), p) < percentile(theirs.tbt(), p)


def _vision_wait(result: SimResult, p: float) -> float:
    return percentile(sorted(s.vision_ns for s in result.requests.values() if s.vision_ns), p)


def test_vision_balancer_beats_baselines_in_the_tail() -> None:
    """Random machine, then fewest patches in flight, beats affinity everywhere and round-robin in the tail.
    At the median it ties with round-robin (within 2%): with similar image sizes, round-robin is already even."""
    ours = _run("vision", "deepseek")
    for baseline in ("round_robin", "affinity"):
        theirs = _run("vision", baseline)
        for p in (90, 99):
            assert _vision_wait(ours, p) < _vision_wait(theirs, p)
        assert _vision_wait(ours, 50) <= 1.02 * _vision_wait(theirs, 50)
