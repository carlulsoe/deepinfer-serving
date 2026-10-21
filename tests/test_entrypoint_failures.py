"""Killing an instance mid-request re-routes what depended on it; outputs still equal cold runs (milestone 4).

A re-route starts a new attempt that re-prefills the prompt plus the tokens already streamed (ADR 0012).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from helpers import SMALL, UNBOUNDED, assert_baseline, cluster, decode_some, greedy, run_normal, tokens

from deepinfer_serving.pd.entrypoint import Entrypoint, RequestState
from deepinfer_serving.pd.instance import StoreSpec

S = RequestState
G = SMALL.grain


@pytest.fixture
def co(tmp_path: Path) -> Iterator[Entrypoint]:
    with cluster(SMALL, 2, 2, StoreSpec(str(tmp_path), UNBOUNDED, UNBOUNDED)) as entrypoint:
        yield entrypoint
        assert_baseline(entrypoint)  # survivors only


def _check(co: Entrypoint, prompts: dict[str, list[int]], max_tokens: int) -> None:
    records = co.run_until_done()
    assert {rid: records[rid].tokens for rid in prompts} == run_normal(SMALL, "cpu", prompts, max_tokens)
    for rid in prompts:
        assert records[rid].state == S.FINISHED
        assert len(records[rid].attempts) == 2


def test_kill_prefill_with_queued_and_prefilling_requests(co: Entrypoint) -> None:
    co.pause("prefill-0")
    prompts = {f"p{i}": tokens(2 * G + 3 * i, 70 + i) for i in range(3)}
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(G), prefill="prefill-0")
    co.step("prefill-0")  # one of them starts prefilling
    co.wait_for(lambda: any(co.requests[r].state == S.PREFILLING for r in prompts))
    co.kill("prefill-0")
    _check(co, prompts, G)
    assert all(co.requests[r].attempt.prefill == "prefill-1" for r in prompts)


def test_kill_prefill_while_handoff_is_in_transit(co: Entrypoint) -> None:
    """The decode instance may still admit the stale attempt; the Entrypoint aborts it there."""
    co.pause("decode-0")
    prompt = tokens(2 * G, 80)
    co.submit("t", prompt, greedy(G), prefill="prefill-0", decode="decode-0")
    co.wait_for(lambda: co.requests["t"].state == S.HANDED_OFF)
    co.kill("prefill-0")
    co.resume("decode-0")
    _check(co, {"t": prompt}, G)


def test_kill_decode_mid_decode_resumes_from_streamed_tokens(co: Entrypoint) -> None:
    prompts = {f"d{i}": tokens(G + 4 * i, 90 + i) for i in range(3)}
    co.pause("decode-0")  # stepped by hand, so no request can finish before the kill
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(3 * G), decode="decode-0")
    decode_some(co, "decode-0", list(prompts), 5)
    co.kill("decode-0")
    _check(co, prompts, 3 * G)
    for rid in prompts:
        record = co.requests[rid]
        assert record.attempt.base >= 5 and record.attempt.decode == "decode-1"
        assert S.DECODING in record.history and record.history.count(S.QUEUED) == 2


def test_kill_decode_with_handoff_waiting_in_prefill_outbox(co: Entrypoint) -> None:
    co.pause("decode-0")
    prompt = tokens(2 * G + 1, 95)
    co.submit("o", prompt, greedy(G), prefill="prefill-0", decode="decode-0")
    co.wait_for(lambda: co.requests["o"].state == S.HANDED_OFF)
    co.kill("decode-0")
    _check(co, {"o": prompt}, G)
