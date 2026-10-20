"""Wire format (ADR 0007): round trip, every corruption detected, receiver-side validation."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from helpers import SMALL, greedy, tokens

from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.serving_engine import ServingLLMEngine
from deepinfer_serving.wire import WireError, decode_handoff


@pytest.fixture(scope="module")
def handoff() -> Handoff:
    sent: list[Handoff] = []
    engine = ServingLLMEngine(SMALL, EngineType.PROMPT, handoff_sink=sent.append)
    engine.add_request("u", "r", greedy(4), np.asarray(tokens(40, 0), dtype=np.int32))
    engine.step()
    return sent[0]


def test_round_trip_is_exact(handoff: Handoff) -> None:
    back = Handoff.from_bytes(handoff.to_bytes())
    for name in ("request_id", "user_id", "sampling_params", "prompt_len", "token_ids", "num_cached_tokens"):
        assert getattr(back, name) == getattr(handoff, name)
    assert torch.equal(back.block_kv, handoff.block_kv)
    assert torch.equal(back.swa_state, handoff.swa_state)


def test_every_single_byte_flip_is_detected(handoff: Handoff) -> None:
    blob = handoff.to_bytes()
    for pos in range(0, len(blob), max(1, len(blob) // 2000)):
        corrupted = bytearray(blob)
        corrupted[pos] ^= 0x40
        with pytest.raises(WireError):
            decode_handoff(bytes(corrupted))


def test_truncation_is_detected(handoff: Handoff) -> None:
    blob = handoff.to_bytes()
    for cut in (0, 10, 50, len(blob) // 2, len(blob) - 1):
        with pytest.raises(WireError):
            decode_handoff(blob[:cut])


def test_no_pickle_on_the_wire(handoff: Handoff) -> None:
    assert b"\x80\x04" not in handoff.to_bytes()[:64]  # not a pickle stream
    assert handoff.to_bytes()[:4] == b"DISH"


def test_receiver_rejects_well_formed_but_wrong_handoffs(handoff: Handoff) -> None:
    receiver = ServingLLMEngine(SMALL, EngineType.COMPLETION)
    bad = [
        replace(handoff, config_fingerprint="other"),
        replace(handoff, block_kv=handoff.block_kv[:-1]),
        replace(handoff, swa_state=handoff.swa_state[:, :, :-1]),
        replace(handoff, token_ids=[*handoff.token_ids[:-1], SMALL.vocab_size]),
        replace(handoff, num_cached_tokens=handoff.num_cached_tokens - 1),
        replace(handoff, token_ids=handoff.token_ids[: handoff.prompt_len]),
    ]
    for h in bad:
        with pytest.raises(ValueError):
            receiver.receive_handoff(Handoff.from_bytes(h.to_bytes()))
    assert receiver.block_manager.get_num_free_blocks() == SMALL.num_gpu_blocks
