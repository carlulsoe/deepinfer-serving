"""Decode drain: a draining decode instance forwards its sequences, with their KV, to another (milestone 4).

Forwarded requests must finish with the same tokens as uninterrupted runs (ADR 0013).
"""

from collections.abc import Iterator

import pytest
from helpers import SMALL, assert_baseline, cluster, decode_some, greedy, run_normal, tokens

from deepinfer_serving.pd.entrypoint import Entrypoint, RequestState

S = RequestState
G = SMALL.grain


@pytest.fixture
def co() -> Iterator[Entrypoint]:
    with cluster(SMALL, 1, 2) as entrypoint:
        yield entrypoint
        assert_baseline(entrypoint)


def test_drain_forwards_in_flight_requests(co: Entrypoint) -> None:
    prompts = {f"r{i}": tokens(G + 5 * i, 110 + i) for i in range(4)}
    co.pause("decode-0")
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(2 * G), decode="decode-0")
    decode_some(co, "decode-0", list(prompts))
    co.drain("decode-0")
    co.resume("decode-0")  # forwards before its next engine step
    records = co.run_until_done()
    assert {rid: records[rid].tokens for rid in prompts} == run_normal(SMALL, "cpu", prompts, 2 * G)
    for rid in prompts:
        assert S.FORWARDED in records[rid].history
        assert records[rid].attempt.decode == "decode-1"
        assert len(records[rid].attempts) == 1  # forwarded, not re-run
    co.retire("decode-0")
    assert co.live == ["prefill-0", "decode-1"]
    co.submit("after", tokens(G, 120), greedy(4))  # new work avoids the retired instance
    assert co.run_until_done()["after"].attempt.decode == "decode-1"


def test_abort_while_forwarded(co: Entrypoint) -> None:
    co.pause("decode-0")
    co.submit("a", tokens(G + 2, 130), greedy(3 * G), decode="decode-0")
    decode_some(co, "decode-0", ["a"])
    co.pause("decode-1")  # the forwarded handoff waits in decode-1's inbox
    co.drain("decode-0")
    co.resume("decode-0")
    co.wait_for(lambda: co.requests["a"].state == S.FORWARDED)
    co.abort("a")
    co.resume("decode-1")
    assert co.run_until_done()["a"].state == S.ABORTED
    co.retire("decode-0")
