"""A full decode instance rejects handoffs; the prefill instance keeps the bytes and retries (ADR 0005)."""

from collections.abc import Iterator
from dataclasses import replace

import pytest
from helpers import SMALL, assert_baseline, greedy, pd_coordinator, run_normal, tokens

from deepinfer_serving.pd.coordinator import PDCoordinator, RequestState

D_CONFIG = replace(SMALL, num_state_slots=1)  # one sequence at a time on the decode side


@pytest.fixture(scope="module")
def co() -> Iterator[PDCoordinator]:
    with pd_coordinator(SMALL, D_CONFIG) as coordinator:
        yield coordinator


def test_full_decode_nacks_and_prefill_retries(co: PDCoordinator) -> None:
    prompts = {f"b{i}": tokens(20 + 7 * i, 200 + i) for i in range(4)}
    expected = run_normal(SMALL, "cpu", prompts, 25)
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(25))
    records = co.run_until_done()
    assert {rid: records[rid].tokens for rid in prompts} == expected
    assert co.stats("prefill")["nacks_received"] > 0
    assert_baseline(co)


def test_abort_while_waiting_for_capacity(co: PDCoordinator) -> None:
    co.submit("hog", tokens(20, 300), greedy(200))
    co.wait_for(lambda: co.requests["hog"].state == RequestState.DECODING)
    nacks_before = co.stats("prefill")["nacks_received"]
    co.submit("waiter", tokens(20, 301), greedy(5))
    co.wait_for(lambda: co.stats("prefill")["nacks_received"] > nacks_before)
    co.abort("waiter")
    co.wait_for(lambda: co.requests["waiter"].state.terminal)
    assert co.requests["waiter"].state == RequestState.ABORTED
    records = co.run_until_done()
    assert records["hog"].state == RequestState.FINISHED
    assert records["hog"].tokens == run_normal(SMALL, "cpu", {"hog": tokens(20, 300)}, 200)["hog"]
    assert_baseline(co)
