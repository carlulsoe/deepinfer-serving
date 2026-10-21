"""Autoscaler (roadmap S3, ADR 0016): reconcile rules, invariants on bursty and failure traces, and SLO attainment
against GPU-hours compared with static allocations."""

import dataclasses
import functools

import pytest

from deepinfer_serving.control import autoscaler as asc
from deepinfer_serving.control.sim import Failure, SimConfig, SimResult, simulate
from deepinfer_serving.control.trace import Request, synthetic_trace

# ---- reconcile rules ----

SPEC = asc.AutoscalerSpec(
    prefill=asc.RoleSpec(2, 1, 4, 1_000.0, max_step=2, cooldown_s=30),
    decode=asc.RoleSpec(2, 1, 4, 10.0, max_step=1, cooldown_s=30),
    warm_gpus=2,
)


def _inst(
    name: str, role: str, state: str = "serving", since_s: float = 0.0, active: int = 0, healthy: int = 0, gpus: int = 2
) -> asc.InstanceView:
    return asc.InstanceView(name, role, gpus, state, since_s, active, healthy)


def _observed(*instances: asc.InstanceView, now_s: float = 100.0, warm: int = 2, last: float = 0.0) -> asc.Observed:
    return asc.Observed(now_s, instances, warm, {asc.PREFILL: last, asc.DECODE: last})


def test_scale_up_is_bounded_by_step_and_maximum() -> None:
    one = (_inst("p0", asc.PREFILL), _inst("d0", asc.DECODE))
    busy = asc.Telemetry(uncached_tok_s=100_000, queued_uncached=0, running_requests=1_000)
    assert asc.reconcile(SPEC, _observed(*one), busy) == [asc.ScaleUp(asc.PREFILL)] * 2 + [asc.ScaleUp(asc.DECODE)]
    four = tuple(_inst(f"p{i}", asc.PREFILL) for i in range(4)) + (_inst("d0", asc.DECODE),) * 4
    assert asc.reconcile(SPEC, _observed(*four), busy) == []
    assert asc.reconcile(SPEC, _observed(*one, last=80.0), busy) == []  # cooling down


def test_backlog_counts_as_prefill_demand() -> None:
    idle_rate = asc.Telemetry(uncached_tok_s=0, queued_uncached=3_000 * 30, running_requests=0)
    actions = asc.reconcile(SPEC, _observed(_inst("p0", asc.PREFILL), _inst("d0", asc.DECODE)), idle_rate)
    assert actions == [asc.ScaleUp(asc.PREFILL)] * 2  # 3000 tok/s needs ceil(3000 / 1400) = 3 instances


def test_no_scale_down_above_the_utilization_limit() -> None:
    """Ten instances at target 0.9: desired is nine below 0.81 utilization, but draining also needs <= 0.8."""
    spec = dataclasses.replace(SPEC, prefill=asc.RoleSpec(1, 1, 10, 1_000.0, target_utilization=0.9))
    fleet = (*(_inst(f"p{i}", asc.PREFILL, active=i, gpus=1) for i in range(10)), _inst("d0", asc.DECODE))
    gpus = sum(i.gpus for i in fleet if i.role == asc.PREFILL)

    def at(util: float) -> list[asc.Action]:
        t = asc.Telemetry(util * gpus * 1_000.0, 0, 0)
        return [a for a in asc.reconcile(spec, _observed(*fleet), t) if not isinstance(a, asc.ScaleUp)]

    assert at(0.805) == []
    assert at(0.79) == [asc.Drain("p0")]  # the instance with the fewest requests


def test_no_scale_down_while_an_instance_starts_or_awaits_replacement() -> None:
    idle = asc.Telemetry(0, 0, 0)
    starting = (_inst("p0", asc.PREFILL), _inst("p1", asc.PREFILL, "starting"), _inst("d0", asc.DECODE))
    assert asc.reconcile(SPEC, _observed(*starting), idle) == []
    dead = (_inst("p0", asc.PREFILL), _inst("p1", asc.PREFILL, "dead", healthy=1), _inst("d0", asc.DECODE))
    assert asc.reconcile(SPEC, _observed(*dead), idle) == [asc.Replace("p1", 1)]  # EP re-formed at 1 GPU


def test_dead_instance_without_healthy_gpus_is_not_replaced() -> None:
    fleet = (_inst("p0", asc.PREFILL, "dead"), _inst("d0", asc.DECODE))
    assert asc.reconcile(SPEC, _observed(*fleet), asc.Telemetry(500, 0, 0)) == [asc.ScaleUp(asc.PREFILL)]


def test_draining_decode_is_forwarded_after_the_grace_period() -> None:
    def draining(since_s: float, active: int) -> list[asc.Action]:
        d1 = _inst("d1", asc.DECODE, "draining", since_s, active)
        fleet = (_inst("p0", asc.PREFILL), _inst("d0", asc.DECODE), d1)
        return asc.reconcile(SPEC, _observed(*fleet), asc.Telemetry(0, 0, 0))

    assert draining(since_s=50.0, active=3) == []
    assert draining(since_s=40.0, active=3) == [asc.Forward("d1")]
    assert draining(since_s=40.0, active=0) == []


def test_warm_pool_is_kept_at_its_target() -> None:
    fleet = (_inst("p0", asc.PREFILL), _inst("d0", asc.DECODE))
    assert asc.reconcile(SPEC, _observed(*fleet, warm=0), asc.Telemetry(0, 0, 0)) == [asc.Provision(2)]
    assert asc.reconcile(SPEC, _observed(*fleet, warm=5), asc.Telemetry(0, 0, 0)) == [asc.Release(3)]


# ---- simulated invariants and quality ----


def _phase(seed: int, sessions: int, start: float, duration: float, tag: str) -> list[Request]:
    trace = synthetic_trace(seed, sessions=sessions, duration_s=duration, turns=3, first_input=40_000, output=1_000)
    return [
        dataclasses.replace(r, id=tag + r.id, session=tag + r.session, arrival_s=r.arrival_s + start) for r in trace
    ]


BURSTY = sorted(
    _phase(1, 100, 0, 600, "a") + _phase(2, 1_000, 600, 400, "b") + _phase(3, 100, 1_000, 600, "c"),
    key=lambda r: (r.arrival_s, r.id),
)
"""Quiet for ten minutes, about 15x the uncached tokens for nearly seven, then quiet again."""

AUTO = asc.AutoscalerSpec(
    prefill=asc.RoleSpec(2, 1, 6, 30_000, max_step=2, cooldown_s=30),
    decode=asc.RoleSpec(2, 1, 6, 30, max_step=2, cooldown_s=30),
    warm_gpus=2,
)
SLO = (2.0, 0.010)
"""TTFT at most 2 s and mean TBT at most 10 ms."""


@functools.cache
def _run(name: str) -> SimResult:
    if name == "auto":
        return simulate(SimConfig([2], [2], autoscaler=AUTO, startup_s=120, warm_startup_s=15), BURSTY)
    if name == "auto-failures":
        failures = [Failure(700, asc.PREFILL, healthy_gpus=1), Failure(800, asc.DECODE, healthy_gpus=2)]
        config = SimConfig([2], [2], autoscaler=AUTO, failures=failures, startup_s=120, warm_startup_s=15)
        return simulate(config, BURSTY)
    k = int(name.removeprefix("static-"))
    return simulate(SimConfig([2] * k, [2] * k), BURSTY)


@pytest.mark.parametrize("name", ["auto", "auto-failures"])
def test_invariants_hold_under_bursts_and_failures(name: str) -> None:
    result = _run(name)
    assert all(result.requests[r.id].completions == 1 for r in BURSTY)  # nothing lost, nothing twice
    for role in (asc.PREFILL, asc.DECODE):
        rs = AUTO.role(role)
        scaling = [
            r.time_s
            for r in result.reconciles
            for a in r.actions
            if (isinstance(a, asc.ScaleUp) and a.role == role)
            or (isinstance(a, asc.Drain) and a.name.startswith(role[0]))
        ]
        for t in scaling:  # rate limit
            assert sum(1 for u in scaling if t <= u < t + rs.cooldown_s) <= rs.max_step
        for r in result.reconciles:  # bounds, counting replacements as they start
            assert rs.min_instances <= r.active[role] <= rs.max_instances
            if any(isinstance(a, asc.Drain) and a.name.startswith(role[0]) for a in r.actions):
                assert r.utilization[role] <= AUTO.no_scale_down_above
    drained = {a.name for r in result.reconciles for a in r.actions if isinstance(a, asc.Drain)}
    assert drained and all(result.instances[n][0] == "retired" for n in drained)
    assert sum(1 for r in result.reconciles for a in r.actions if isinstance(a, asc.ScaleUp)) > 0
    if name == "auto-failures":
        replaced = sorted(a.gpus for r in result.reconciles for a in r.actions if isinstance(a, asc.Replace))
        assert replaced == [1, 2]
        assert sum(1 for gpus in (g for _, g in result.instances.values()) if gpus == 1) == 1  # re-formed EP
        assert sum(s.reroutes for s in result.requests.values()) > 0


def test_drain_forwards_long_requests_after_the_grace_period() -> None:
    """Long outputs outlive a 5 s grace: they are forwarded with their KV, not re-prefilled."""
    trace = synthetic_trace(4, sessions=40, duration_s=20, turns=1, first_input=5_000, output=4_000)
    spec = dataclasses.replace(
        AUTO,
        decode=asc.RoleSpec(2, 1, 3, 1_000, cooldown_s=10),
        prefill=asc.RoleSpec(2, 1, 1, 1e9),
        drain_grace_s=5,
    )
    result = simulate(SimConfig([2], [2, 2, 2], autoscaler=spec, startup_s=10, warm_startup_s=5), trace)
    forwards = {a.name for r in result.reconciles for a in r.actions if isinstance(a, asc.Forward)}
    assert forwards and all(result.instances[n][0] == "retired" for n in forwards)
    assert all(s.completions == 1 and s.reroutes == 0 for s in result.requests.values())
    assert all(len(s.gaps_ns) == r.output_len - 1 for r, s in ((r, result.requests[r.id]) for r in trace))


def test_slo_attainment_against_gpu_hours_beats_static_allocations() -> None:
    """No static allocation is both cheaper and better, and the one nearest in GPU-hours is clearly worse.

    Measured on this trace and cost model: autoscaled 3.97 GPU-h at 0.91 attainment; static 2+2 3.90 GPU-h at
    0.79; static 3+3 5.84 GPU-h at 0.96."""
    auto = _run("auto")
    ours = (auto.gpu_hours, auto.slo_attainment(*SLO))
    statics = {k: _run(f"static-{k}") for k in (2, 3)}
    points = {k: (r.gpu_hours, r.slo_attainment(*SLO)) for k, r in statics.items()}
    for gpuh, att in points.values():
        assert not (gpuh <= ours[0] and att >= ours[1])
    nearest = min(points.values(), key=lambda p: abs(p[0] - ours[0]))
    assert abs(nearest[0] - ours[0]) < 0.1 * ours[0] and ours[1] > nearest[1] + 0.05
    assert ours[0] < 0.75 * points[3][0]  # a quarter fewer GPU-hours than the smallest good static fleet
