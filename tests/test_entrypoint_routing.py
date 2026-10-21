"""Entrypoint with two prefill and two decode instances sharing one store (milestone 4).

Turn 1 (prompt 3G+2, reply G+4) leaves blocks 0-3 on disk, the prefill instance's record at 3G, and the decode
instance's record at 4G. Turn 2 caps L1 at 4G.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from helpers import SMALL, UNBOUNDED, assert_baseline, cluster, greedy, run_normal, tokens

from deepinfer_serving.kvstore import block_hashes, namespace
from deepinfer_serving.pd.entrypoint import Entrypoint, RequestState
from deepinfer_serving.pd.instance import StoreSpec

G = SMALL.grain
NEW_USER = SMALL.window + SMALL.backtrack + 3


@pytest.fixture
def co(tmp_path: Path) -> Iterator[Entrypoint]:
    with cluster(SMALL, 2, 2, StoreSpec(str(tmp_path), UNBOUNDED, UNBOUNDED)) as entrypoint:
        yield entrypoint
        assert_baseline(entrypoint)


def _turn1(co: Entrypoint, rid: str, seed: int, **route: str) -> list[int]:
    """Run turn 1 and wait until its decode instance's 4G record is in the directory. Returns turn 2."""
    prompt = tokens(3 * G + 2, seed)
    co.submit(rid, prompt, greedy(G + 4), **route)
    turn2 = prompt + co.run_until_done()[rid].tokens + tokens(NEW_USER, seed + 1)
    key = block_hashes(turn2, G)[3]
    co.wait_for(lambda: bool(co.directory.get((key, 4 * G))))
    return turn2


def test_multi_turn_conversation_alternating_instances_equals_cold(co: Entrypoint) -> None:
    """Four turns, each on a different prefill and decode instance than the last; every turn equals cold."""
    conversation = tokens(3 * G + 2, 20)
    for turn in range(4):
        rid = f"turn{turn}"
        co.submit(rid, conversation, greedy(G + 4), prefill=f"prefill-{turn % 2}", decode=f"decode-{(turn + 1) % 2}")
        reply = co.run_until_done()[rid].tokens
        assert reply == run_normal(SMALL, "cpu", {"x": conversation}, G + 4)["x"]
        hit = co.requests[rid].hit
        assert hit is not None and (turn == 0) == (hit[0] == 0)
        conversation = conversation + reply + tokens(NEW_USER, 21 + turn)


def test_balancers_spread_concurrent_requests(co: Entrypoint) -> None:
    prompts = {f"c{i}": tokens(G + 5 * i, 60 + i) for i in range(4)}
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(G))
    records = co.run_until_done()
    assert {rid: records[rid].tokens for rid in prompts} == run_normal(SMALL, "cpu", prompts, G)
    assert {records[rid].attempt.prefill for rid in prompts} == {"prefill-0", "prefill-1"}
    assert {records[rid].attempt.decode for rid in prompts} == {"decode-0", "decode-1"}


def test_hit_length_does_not_depend_on_which_prefill_served_earlier_turns(co: Entrypoint) -> None:
    """No affinity: turn 2 gets the same hit on the prefill instance that ran turn 1 and on the other one."""
    turn2 = _turn1(co, "t1", 30, prefill="prefill-0", decode="decode-0")
    cold = run_normal(SMALL, "cpu", {"x": turn2}, G + 3)["x"]
    co.submit("same", turn2, greedy(G + 3), prefill="prefill-0", decode="decode-1")
    co.submit("other", turn2, greedy(G + 3), prefill="prefill-1", decode="decode-1")
    records = co.run_until_done()
    for rid in ("same", "other"):
        assert records[rid].hit == (4 * G, 4 * G, 0)  # record at L1, pulled from decode-0 (ADR 0011)
        assert records[rid].tokens == cold
    assert co.stats("decode-0")["swa_pulls_served"] == 2


def test_record_lost_at_the_holder_backs_off(co: Entrypoint) -> None:
    """The directory names a holder that no longer has the record: the hit shortens, the output does not change."""
    turn2 = _turn1(co, "t1", 40, prefill="prefill-0", decode="decode-0")
    co.set_faults("decode-0", lose_swa=True)
    co.submit("t2", turn2, greedy(G + 3), prefill="prefill-1", decode="decode-1")
    record = co.run_until_done()["t2"]
    assert record.hit == (3 * G, 3 * G, 3 * G)  # prefill-1 holds no record: back off from blocks alone
    assert record.tokens == run_normal(SMALL, "cpu", {"x": turn2}, G + 3)["x"]
    co.set_faults("decode-0")


def test_block_lost_after_check_is_served_cold(co: Entrypoint, tmp_path: Path) -> None:
    """CHECK saw the block; the prefill instance cannot read it, so it serves cold, and the output is the same."""
    turn2 = _turn1(co, "t1", 50, prefill="prefill-0", decode="decode-0")
    co.pause("prefill-1")
    co.submit("t2", turn2, greedy(G + 3), prefill="prefill-1", decode="decode-1")
    block = tmp_path / namespace(SMALL.fingerprint()) / block_hashes(turn2, G)[1].hex()
    block.unlink()
    co.resume("prefill-1")
    record = co.run_until_done()["t2"]
    assert record.hit == (0, 0, 0)
    assert record.tokens == run_normal(SMALL, "cpu", {"x": turn2}, G + 3)["x"]
    assert record.history[-1] == RequestState.FINISHED


def test_directory_forgets_records_with_no_holder_left(co: Entrypoint) -> None:
    turn2 = _turn1(co, "t1", 60, prefill="prefill-0", decode="decode-0")
    entry = (block_hashes(turn2, G)[3], 4 * G)
    assert co.directory[entry] == {"decode-0"}
    co.kill("decode-0")
    assert entry not in co.directory  # no empty holder sets left behind
    assert all(co.directory.values())
