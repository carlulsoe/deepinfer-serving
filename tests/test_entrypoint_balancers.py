"""The tier-0 balancers in the live Entrypoint (ADR 0015), and a tier-1 run turned into a simulator trace (ADR 0014).

Tier 1 checks correctness only: routing through the balancers never changes outputs, and nothing is lost.
"""

from pathlib import Path

from helpers import SMALL, UNBOUNDED, assert_baseline, cluster, greedy, run_normal, tokens

from deepinfer_serving.control.sim import SimConfig, simulate
from deepinfer_serving.control.trace import read_trace, trace_from_records, write_trace
from deepinfer_serving.pd.entrypoint import RequestState
from deepinfer_serving.pd.instance import StoreSpec

G = SMALL.grain


def test_requests_wait_for_prefill_quota_and_match_cold() -> None:
    """A pull limit below one request's cost: each prefill instance takes one request at a time."""
    prompts = {f"q{i}": tokens(G + 3 * i, 400 + i) for i in range(6)}
    with cluster(SMALL, 2, 2, pull_limit_ns=1.0) as co:
        for rid, prompt in prompts.items():
            co.submit(rid, prompt, greedy(5))
        assert [r.request_id for r, _, _ in co.pending] == ["q2", "q3", "q4", "q5"]
        assert {co.requests[r].attempt.prefill for r in ("q0", "q1")} == {"prefill-0", "prefill-1"}
        records = co.run_until_done()
        assert {rid: records[rid].tokens for rid in prompts} == run_normal(SMALL, "cpu", prompts, 5)
        assert all(len(records[rid].attempts) == 1 for rid in prompts)
        assert co.pending == []
        assert_baseline(co)


def test_abort_while_waiting_for_a_route_sends_nothing() -> None:
    with cluster(SMALL, 1, 1, pull_limit_ns=1.0) as co:
        co.pause("prefill-0")
        co.submit("first", tokens(G, 410), greedy(3))
        co.submit("waiting", tokens(G, 411), greedy(3))
        co.abort("waiting")
        co.resume("prefill-0")
        records = co.run_until_done()
        assert records["waiting"].history == [RequestState.QUEUED, RequestState.ABORTED]
        assert records["waiting"].attempts == []
        assert records["first"].state == RequestState.FINISHED
        assert_baseline(co)


def test_request_too_big_for_any_decode_instance_fails_without_blocking_others() -> None:
    """The decode LB finds no room for it; the request still goes out and fails, and later requests are served."""
    with cluster(SMALL, 1, 1) as co:
        co.submit("huge", tokens(SMALL.num_gpu_blocks * SMALL.block_size + 1, 430), greedy(3))
        co.submit("after", tokens(G, 431), greedy(3))
        records = co.run_until_done()
        assert records["huge"].state == RequestState.FAILED
        assert records["after"].tokens == run_normal(SMALL, "cpu", {"x": tokens(G, 431)}, 3)["x"]
        assert_baseline(co)


def test_tier1_run_replays_in_the_simulator(tmp_path: Path) -> None:
    """Two turns through the Entrypoint become a trace; the simulator reads it back and serves every request."""
    with cluster(SMALL, 2, 2, StoreSpec(str(tmp_path / "kv"), UNBOUNDED, UNBOUNDED)) as co:
        prompt = tokens(3 * G + 2, 420)
        co.submit("t1", prompt, greedy(G + 4), user_id="alice")
        turn2 = prompt + co.run_until_done()["t1"].tokens + tokens(SMALL.window + SMALL.backtrack + 3, 421)
        co.submit("t2", turn2, greedy(G + 3), user_id="alice")
        co.submit("other", tokens(2 * G, 422), greedy(4), user_id="bob")
        records = co.run_until_done()
        assert_baseline(co)
    write_trace(tmp_path / "tier1.jsonl", trace_from_records(records.values()))
    trace = read_trace(tmp_path / "tier1.jsonl")
    by_id = {r.id: r for r in trace}
    assert set(by_id) == {"t1", "t2", "other"}
    hit = records["t2"].hit
    assert hit is not None and by_id["t2"].hit_len == hit[0] > 0
    assert (by_id["t2"].input_len, by_id["t2"].output_len, by_id["t2"].session) == (len(turn2), G + 3, "alice")
    assert by_id["t1"].arrival_s == 0.0 < by_id["t2"].arrival_s
    result = simulate(SimConfig([1, 1], [1, 1]), trace)
    assert all(s.completions == 1 for s in result.requests.values())
