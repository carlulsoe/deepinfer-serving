"""The tier-0 simulator against analytical cases, and the trace format (roadmap S1, ADR 0014)."""

import itertools
import random
from dataclasses import replace
from pathlib import Path

import pytest

from deepinfer_serving.control.cost import MAX_CHUNK, CostModel
from deepinfer_serving.control.sim import Failure, S, SimConfig, simulate
from deepinfer_serving.control.trace import Request, read_trace, synthetic_trace, write_trace

MS = 1e6
LINEAR = CostModel(prefill_ns_per_token=MS, prefill_ns_per_pair=0, prefill_step_ns=0, transfer_ns_per_token=0)
"""1 ms per prefill token, nothing else, so a request's prefill time is exact."""


def _req(i: int, t: float, input_len: int = 100, output_len: int = 1, hit: int = 0, session: str = "") -> Request:
    return Request(f"r{i}", t, session or f"s{i}", input_len, hit, output_len)


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_single_queue_matches_m_d_1(rho: float) -> None:
    """Poisson arrivals, one rank, fixed service time S and budget S: the mean wait is rho * S / (2 * (1 - rho))
    (Pollaczek-Khinchine)."""
    service = 0.1
    cost = replace(LINEAR, quota_limit_ns=service * S)
    rng = random.Random(1)
    t, trace = 0.0, []
    for i in range(40_000):
        t += rng.expovariate(rho / service)
        trace.append(_req(i, t))
    result = simulate(SimConfig([1], [1], cost=cost), trace)
    mean_ttft = sum(result.ttft()) / len(trace)
    assert mean_ttft == pytest.approx(service + rho * service / (2 * (1 - rho)), rel=0.03)


@pytest.mark.parametrize("policy", ["deepseek", "round_robin"])
def test_uniform_load_is_served_without_queueing(policy: str) -> None:
    """Four ranks in two lockstep groups, four identical requests every 2S: each finds an idle rank."""
    cost = replace(LINEAR, quota_limit_ns=100 * MS)
    trace = [_req(4 * k + j, 0.2 * k) for k in range(50) for j in range(4)]
    result = simulate(SimConfig([2, 2], [1], cost=cost, policy=policy), trace)
    assert result.ttft() == pytest.approx([0.1] * len(trace))
    assert result.prefill_balance == pytest.approx(1.0)


def test_overloaded_queue_grows_linearly() -> None:
    """D/D/1 with arrivals every 0.08 s and service 0.1 s: request k finishes at (k + 1) * 0.1."""
    cost = replace(LINEAR, quota_limit_ns=100 * MS)
    trace = [_req(k, 0.08 * k) for k in range(100)]
    result = simulate(SimConfig([1], [1], cost=cost), trace)
    for k in range(100):
        assert result.requests[f"r{k}"].ttft_s == pytest.approx((k + 1) * 0.1 - 0.08 * k)


@pytest.mark.parametrize("prefill", [[1], [1] * 5], ids=["one-rank", "five-instances"])
def test_decode_step_times_follow_the_cost_model(prefill: list[int]) -> None:
    """n requests decode together on one rank; each step lasts decode_step(n, total context). With five prefill
    instances their handoffs arrive at the same instant and still join one decode step."""
    n, m, input_len = 5, 20, 100
    cost = CostModel(prefill_ns_per_token=1, prefill_ns_per_pair=0, prefill_step_ns=0, transfer_ns_per_token=0)
    trace = [_req(i, 0.0, input_len, m) for i in range(n)]
    result = simulate(SimConfig(prefill, [1], cost=cost), trace)
    prefill_end = n * input_len * 1.0 / len(prefill)  # one step packs all n, or each instance takes one
    expected = [cost.decode_step(n, n * (input_len + k)) for k in range(1, m)]
    for r in result.requests.values():
        assert r.first_token_ns == pytest.approx(prefill_end)
        assert r.gaps_ns == pytest.approx(expected)
        assert r.finish_ns == pytest.approx(prefill_end + sum(expected))
    assert result.decode_balance == pytest.approx(1.0)


def test_chunk_is_the_largest_that_fits_the_budget() -> None:
    cost = CostModel()
    for budget, c, remaining in [(1e8, 0, 50_000), (1e8, 74_000, 50_000), (3e7, 10_000, 100), (1e3, 0, 10)]:
        q = cost.solve_chunk(budget, c, remaining)
        assert q <= min(remaining, MAX_CHUNK)
        assert q == 0 or cost.cost(q, c) <= budget
        assert q == min(remaining, MAX_CHUNK) or cost.cost(q + 1, c) > budget


def test_long_prompt_is_chunked_across_steps() -> None:
    """A 300-token prompt at 1 ms/token with a 100 ms budget takes three steps."""
    cost = replace(LINEAR, quota_limit_ns=100 * MS)
    result = simulate(SimConfig([1], [1], cost=cost), [_req(0, 0.0, 300)])
    assert result.requests["r0"].ttft_s == pytest.approx(0.3)


def test_cost_model_placeholders_reproduce_the_reported_figures() -> None:
    """ADR 0014: 60K uncached tok/s per GPU at 74K context, 14K decode tok/s per GPU at 230 tok/s per user [DS]."""
    cost = CostModel()
    assert 1e9 / (cost.cost(3_000, 74_000) / 3_000) == pytest.approx(60_000, rel=0.03)
    step = cost.decode_step(61, 61 * 77_000)
    assert 61 / (step / 1e9) == pytest.approx(14_000, rel=0.03)
    assert 1 / (step / 1e9) == pytest.approx(230, rel=0.03)


def test_trace_round_trips(tmp_path: Path) -> None:
    trace = synthetic_trace(3, sessions=20, duration_s=60, image_fraction=0.3)
    write_trace(tmp_path / "t.jsonl", trace)
    assert read_trace(tmp_path / "t.jsonl") == trace
    assert any(r.patches for r in trace) and any(r.hit_len for r in trace)
    by_session: dict[str, list[Request]] = {}
    for r in trace:
        by_session.setdefault(r.session, []).append(r)
    for turns in by_session.values():
        for prev, nxt in itertools.pairwise(turns):
            context = prev.input_len + prev.output_len
            assert nxt.input_len > context and nxt.hit_len <= context


def test_lost_prefill_restarts_from_the_hit_on_another_instance() -> None:
    """300 tokens in three 100 ms steps on p0, which dies during the second; p1 prefills all 300 again."""
    cost = replace(LINEAR, quota_limit_ns=100 * MS)
    config = SimConfig([1, 1], [1], cost=cost, failures=[Failure(0.15, "p0")])
    stats = simulate(config, [_req(0, 0.0, 300)]).requests["r0"]
    assert stats.ttft_s == pytest.approx(0.15 + 0.3)
    assert stats.reroutes == 1


def test_lost_decode_reprefills_prompt_and_streamed_tokens() -> None:
    """ADR 0012: d0 dies after four 10 ms decode steps; prompt + 5 streamed tokens are prefilled again (105 ms),
    which emits token 6, and d1 decodes the last four."""
    cost = replace(
        LINEAR, quota_limit_ns=1e9, decode_step_ns=10 * MS, decode_ns_per_request=0, decode_ns_per_kv_token=0
    )
    config = SimConfig([1], [1, 1], cost=cost, failures=[Failure(0.145, "d0")])
    stats = simulate(config, [_req(0, 0.0, 100, output_len=10)]).requests["r0"]
    assert stats.reroutes == 1
    assert stats.ttft_s == pytest.approx(0.1)
    assert stats.gaps_ns == pytest.approx([10 * MS] * 4 + [0.25 * S - 0.14 * S] + [10 * MS] * 4)
    assert stats.finish_ns == pytest.approx(0.29 * S)
