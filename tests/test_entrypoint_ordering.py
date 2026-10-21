"""Events of one request come from several instances over separate links and can be read in any order.

Each test holds back one instance's events so another's overtake them (ADR 0012), plus the two failure modes
that ordering does not cover: a full link and an engine crash.
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


def test_decode_events_overtake_prefill_events(co: Entrypoint) -> None:
    prompt = tokens(2 * G + 3, 140)
    co.hold_events("prefill-0")  # prefilling, handed_off and the first token stay unread
    co.submit("o", prompt, greedy(G), decode="decode-0")
    co.wait_for(lambda: co.requests["o"].state == S.FINISHED)
    record = co.requests["o"]
    assert not record.complete and 0 not in record.tokens_by_index
    co.release_events("prefill-0")
    co.run_until_done(timeout=30)
    assert record.tokens == run_normal(SMALL, "cpu", {"o": prompt}, G)["o"]
    assert record.history == [S.QUEUED, S.PREFILLING, S.HANDED_OFF, S.DECODING, S.FINISHED]
    assert record.hit == (0, 0, 0)  # read late, still recorded


def test_forward_target_events_overtake_the_source(co: Entrypoint) -> None:
    prompt = tokens(G + 2, 150)
    co.pause("decode-0")
    co.submit("f", prompt, greedy(3 * G), decode="decode-0")
    decode_some(co, "decode-0", ["f"])
    co.hold_events("decode-0")  # its "forwarded" event and its last tokens stay unread
    co.drain("decode-0")
    co.resume("decode-0")
    co.wait_for(lambda: co.requests["f"].state == S.FINISHED)
    co.release_events("decode-0")
    record = co.run_until_done(timeout=30)["f"]
    assert record.tokens == run_normal(SMALL, "cpu", {"f": prompt}, 3 * G)["f"]
    assert record.history[-4:] == [S.DECODING, S.FORWARDED, S.DECODING, S.FINISHED]
    co.retire("decode-0")


def test_a_full_link_does_not_block_the_sender(co: Entrypoint) -> None:
    """A handoff larger than the socket buffer, to an instance that is not reading: the prefill loop goes on."""
    prompt = tokens(90 * G, 160)  # about 1.5 MB of block KV
    co.set_faults("decode-0", deaf=True)
    co.submit("big", prompt, greedy(4), decode="decode-0")
    co.wait_for(lambda: co.requests["big"].state == S.HANDED_OFF, timeout=120)
    assert co.stats("prefill-0", timeout=5)["outbox"] == 1  # answers while its write is still pending
    co.set_faults("decode-0")
    assert co.run_until_done()["big"].tokens == run_normal(SMALL, "cpu", {"big": prompt}, 4)["big"]


def test_engine_crash_is_recovered_like_a_kill(co: Entrypoint) -> None:
    prompt = tokens(G + 7, 170)
    co.pause("decode-0")
    co.submit("c", prompt, greedy(3 * G), decode="decode-0")
    decode_some(co, "decode-0", ["c"])
    co.set_faults("decode-0", crash=True)  # raised at the top of its next loop, paused or not
    record = co.run_until_done()["c"]
    assert record.tokens == run_normal(SMALL, "cpu", {"c": prompt}, 3 * G)["c"]
    assert record.attempt.decode == "decode-1" and len(record.attempts) == 2
    assert len(co.errors) == 1 and "injected crash" in co.errors[0]  # the crash is reported, not hidden
    co.errors.clear()


@pytest.fixture
def wide() -> Iterator[Entrypoint]:
    with cluster(SMALL, 2, 3) as entrypoint:
        yield entrypoint
        assert_baseline(entrypoint)


def test_finished_request_missing_tokens_from_a_dead_instance_is_rerun(wide: Entrypoint) -> None:
    """Decode finished, but the first token died unread with prefill: re-run from the gap, and agree."""
    prompt = tokens(2 * G + 1, 180)
    wide.hold_events("prefill-0")
    wide.submit("gap", prompt, greedy(G), prefill="prefill-0", decode="decode-0")
    wide.wait_for(lambda: wide.requests["gap"].state == S.FINISHED)
    wide.kill("prefill-0")  # token 0 is lost with its unread events
    record = wide.run_until_done(timeout=60)["gap"]
    assert record.tokens == run_normal(SMALL, "cpu", {"gap": prompt}, G)["gap"]
    assert len(record.attempts) == 2 and record.attempt.base == 0
    assert record.history.count(S.FINISHED) == 2 and record.history.count(S.QUEUED) == 2  # finished, re-run


def test_forward_to_a_target_whose_death_was_read_first(wide: Entrypoint) -> None:
    prompt = tokens(G + 3, 190)
    wide.pause("decode-0")
    wide.submit("fw", prompt, greedy(3 * G), decode="decode-0")
    decode_some(wide, "decode-0", ["fw"])
    wide.pause("decode-1")  # the forward arrives but is not admitted
    wide.hold_events("decode-0")  # its "forwarded" event stays unread
    wide.drain("decode-0")  # forwards round-robin, starting with decode-1
    wide.resume("decode-0")
    wide.wait_for(lambda: wide.stats("decode-1")["inbox"] == 1)
    wide.kill("decode-1")  # nothing depends on decode-1 yet, as far as the Entrypoint knows
    wide.release_events("decode-0")
    record = wide.run_until_done(timeout=60)["fw"]
    assert record.tokens == run_normal(SMALL, "cpu", {"fw": prompt}, 3 * G)["fw"]
    assert record.attempt.decode == "decode-2" and len(record.attempts) == 2
    wide.retire("decode-0")


def test_abort_reaches_a_forward_target_whose_admission_overtook_the_forward(co: Entrypoint) -> None:
    co.pause("decode-0")
    co.submit("ab", tokens(G + 4, 200), greedy(8 * G), decode="decode-0")
    decode_some(co, "decode-0", ["ab"])
    co.hold_events("decode-0")  # "forwarded" stays unread; decode-1's "admitted" is read first
    co.drain("decode-0")
    co.resume("decode-0")
    co.wait_for(lambda: co.requests["ab"].attempt.decode == "decode-1")
    aborts = co.stats("decode-1")["aborts"]
    co.abort("ab")
    co.wait_for(lambda: co.stats("decode-1")["aborts"] == aborts + 1, timeout=10)
    co.release_events("decode-0")
    co.retire("decode-0")


def test_a_dead_peer_leaves_no_writer_thread_behind(wide: Entrypoint) -> None:
    before = wide.stats("prefill-0")["threads"]
    wide.kill("decode-2")
    wide.wait_for(lambda: wide.stats("prefill-0")["threads"] == before - 1, timeout=10)


def test_request_finishing_after_its_token_holder_died_is_rerun(wide: Entrypoint) -> None:
    """Prefill dies holding token 0 unread while decode is still running; the gap shows only at the finish."""
    prompt = tokens(2 * G + 5, 210)
    wide.hold_events("prefill-0")
    wide.submit("late", prompt, greedy(2 * G), prefill="prefill-0", decode="decode-0")
    wide.wait_for(lambda: wide.requests["late"].state == S.DECODING)
    wide.pause("decode-0")
    wide.kill("prefill-0")  # the request depends only on decode-0 now: nothing to re-route yet
    assert len(wide.requests["late"].attempts) == 1
    wide.resume("decode-0")
    record = wide.run_until_done(timeout=60)["late"]
    assert record.tokens == run_normal(SMALL, "cpu", {"late": prompt}, 2 * G)["late"]
    assert len(record.attempts) == 2


def test_dead_intermediate_holder_revealed_by_a_late_forward(wide: Entrypoint) -> None:
    """decode-0 -> decode-1 -> decode-2, with decode-0 and decode-1 unread. decode-1 dies holding unread tokens;
    only decode-0's late "forwarded" names it, after the request already finished on decode-2."""
    prompt = tokens(G + 6, 220)
    wide.pause("decode-0")
    wide.pause("decode-1")
    wide.submit("hop", prompt, greedy(3 * G), decode="decode-0")
    decode_some(wide, "decode-0", ["hop"])
    wide.hold_events("decode-0")
    wide.hold_events("decode-1")
    wide.drain("decode-0")  # targets decode-1, then decode-2: the first forward goes to decode-1
    wide.resume("decode-0")
    wide.wait_for(lambda: wide.stats("decode-1")["inbox"] == 1)
    for _ in range(4):  # admit, then three tokens nobody reads
        wide.step("decode-1")
        wide.stats("decode-1")
    wide.drain("decode-1")  # its only target is decode-2
    wide.resume("decode-1")
    wide.wait_for(lambda: wide.requests["hop"].state == S.FINISHED)
    assert not wide.requests["hop"].complete
    wide.kill("decode-1")  # nothing on the recorded path is dead yet
    assert len(wide.requests["hop"].attempts) == 1
    wide.release_events("decode-0")  # its "forwarded" names decode-1
    record = wide.run_until_done(timeout=60)["hop"]
    assert record.tokens == run_normal(SMALL, "cpu", {"hop": prompt}, 3 * G)["hop"]
    assert len(record.attempts) == 2
    wide.retire("decode-0")
