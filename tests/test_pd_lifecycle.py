"""Request lifecycle across processes: abort at every stage, corruption in transit (milestone 2)."""

from collections.abc import Iterator

import numpy as np
import pytest
from helpers import SMALL, assert_baseline, chunked, greedy, pd_coordinator, run_normal, tokens

from deepinfer_serving.pd.entrypoint import Entrypoint, RequestState

S = RequestState
G = SMALL.block_size
P_CONFIG = chunked(SMALL, G)  # a 3G+2 prompt prefills in four chunks


@pytest.fixture(scope="module")
def co() -> Iterator[Entrypoint]:
    with pd_coordinator(P_CONFIG, SMALL) as coordinator:
        yield coordinator


@pytest.fixture(autouse=True)
def _check_invariants(co: Entrypoint) -> Iterator[None]:
    yield
    for role in co.live:
        co.resume(role)
    co.set_faults("prefill")
    co.set_faults("decode")
    assert_baseline(co)


def test_abort_while_queued(co: Entrypoint) -> None:
    co.pause("prefill")
    co.submit("q", tokens(3 * G + 2, 0), greedy(10))
    co.abort("q")
    co.resume("prefill")
    record = co.run_until_done()["q"]
    assert record.history == [S.QUEUED, S.ABORTED]
    assert record.tokens == []


def test_abort_while_prefilling(co: Entrypoint) -> None:
    co.pause("prefill")
    co.submit("p", tokens(3 * G + 2, 1), greedy(10))
    co.step("prefill")  # first chunk only
    co.wait_for(lambda: co.requests["p"].state == S.PREFILLING)
    co.abort("p")
    co.resume("prefill")
    assert co.run_until_done()["p"].history == [S.QUEUED, S.PREFILLING, S.ABORTED]


def test_abort_in_transit(co: Entrypoint) -> None:
    co.pause("decode")
    co.submit("t", tokens(2 * G, 2), greedy(10))
    co.wait_for(lambda: co.requests["t"].state == S.HANDED_OFF)
    co.abort("t")  # the decode instance drops the queued handoff and nacks it (ADR 0012)
    co.resume("decode")
    record = co.run_until_done()["t"]
    assert record.history == [S.QUEUED, S.PREFILLING, S.HANDED_OFF, S.ABORTED]


def test_abort_before_decode_reads_the_handoff(co: Entrypoint) -> None:
    """The abort overtakes the handoff on the way to decode; the late admission is aborted too (ADR 0012)."""
    co.set_faults("decode", deaf=True)  # the handoff stays unread in the link
    aborts = co.stats("decode")["aborts"]
    co.submit("late", tokens(2 * G, 7), greedy(10))
    co.wait_for(lambda: co.requests["late"].state == S.HANDED_OFF)
    co.abort("late")
    co.stats("decode")  # the abort has been handled: nothing to release yet
    co.set_faults("decode")  # now it reads and admits the handoff
    assert co.run_until_done()["late"].history == [S.QUEUED, S.PREFILLING, S.HANDED_OFF, S.ABORTED]
    co.wait_for(lambda: co.stats("decode")["aborts"] == aborts + 1)


def test_abort_while_decoding(co: Entrypoint) -> None:
    co.pause("decode")
    co.submit("d", tokens(2 * G, 3), greedy(50))
    co.wait_for(lambda: co.requests["d"].state == S.HANDED_OFF)
    co.step("decode", 3)  # admit, then decode
    co.wait_for(lambda: len(co.requests["d"].tokens_by_index) >= 3)
    co.abort("d")
    co.resume("decode")
    record = co.run_until_done()["d"]
    assert record.history[-2:] == [S.DECODING, S.ABORTED]
    assert 3 <= len(record.tokens) < 50


def test_abort_after_finish_is_a_no_op(co: Entrypoint) -> None:
    co.submit("f", tokens(G + 3, 4), greedy(5))
    co.run_until_done()
    co.abort("f")
    assert co.requests["f"].state == S.FINISHED


def test_first_token_final_finishes_on_prefill(co: Entrypoint) -> None:
    co.submit("one", tokens(G + 3, 5), greedy(1))
    record = co.run_until_done()["one"]
    assert record.history == [S.QUEUED, S.PREFILLING, S.FINISHED]
    assert len(record.tokens) == 1
    co.abort("one")


def test_random_aborts_terminate_exactly_once(co: Entrypoint) -> None:
    rng = np.random.default_rng(0)
    prompts = {f"r{i}": tokens(int(rng.integers(5, 4 * G)), 100 + i) for i in range(12)}
    expected = run_normal(SMALL, "cpu", prompts, 30)
    to_abort = set(list(prompts)[::2])
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(30))
        co.poll(float(rng.uniform(0, 0.02)))
        if rid in to_abort:
            co.abort(rid)
    records = co.run_until_done()
    for rid in prompts:
        if rid in to_abort:
            assert records[rid].state in (S.ABORTED, S.FINISHED)  # an abort may lose the race to completion
        else:
            assert records[rid].state == S.FINISHED
        if records[rid].state == S.FINISHED:
            assert records[rid].tokens == expected[rid]


@pytest.mark.parametrize("where", ["payload", "header"])
def test_corruption_in_transit_is_detected(co: Entrypoint, where: str) -> None:
    co.set_faults("prefill", corrupt=where)
    co.submit(f"c-{where}", tokens(2 * G, 6), greedy(10))
    record = co.run_until_done()[f"c-{where}"]
    assert record.state == S.FAILED
    assert record.reason.startswith("invalid:") and "checksum" in record.reason
