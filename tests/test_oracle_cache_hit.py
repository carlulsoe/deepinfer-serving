"""Cache-hit oracle: turn 2 served from the store == cold turn 2, token for token. Runs on every backend."""

from pathlib import Path

import numpy as np
import pytest
from helpers import (
    BACKENDS,
    CONFIGS,
    DEVICES,
    SMALL,
    chunked,
    drop_swa,
    greedy,
    noise_like,
    open_store,
    run,
    run_normal,
    turn1,
)

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.serving_engine import ServingLLMEngine

# case -> (records to delete, cached_hit_tokens, num_cached_tokens, swa_valid_start), in units of G
CASES = {
    "record_at_l1": (set(), 4, 4, 0),
    "record_at_l1_minus_g": ({4}, 4, 3, 0),
    "no_record_backoff": ({3, 4}, 3, 3, 3),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", list(CASES))
def test_cache_hit_equals_cold(config: MiniConfig, device: str, case: str, backend: str, tmp_path: Path) -> None:
    g = config.grain
    store = open_store(backend, tmp_path)
    turn2 = turn1(config, device, store)
    drop, hit, cached, valid_start = CASES[case]
    drop_swa(store, {p * g for p in drop})
    max_tokens = g + 3

    cold = run_normal(config, device, {"t2": turn2}, max_tokens)

    warm_engine = ServingLLMEngine(config, EngineType.NORMAL, device, store=store)  # a different "machine"
    seq = warm_engine.add_request("u", "t2", greedy(max_tokens), np.asarray(turn2, dtype=np.int32))
    assert seq is not None
    assert (seq.cached_hit_tokens, seq.num_cached_tokens, seq.swa_valid_start) == (hit * g, cached * g, valid_start * g)
    assert run(warm_engine) == cold


def test_cache_hit_with_chunked_prefill_and_backoff(tmp_path: Path) -> None:
    store = open_store("file", tmp_path)
    turn2 = turn1(SMALL, "cpu", store)
    drop_swa(store, {3 * SMALL.grain, 4 * SMALL.grain})
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 10)
    warm = ServingLLMEngine(chunked(SMALL, SMALL.block_size + 1), EngineType.NORMAL, store=store)
    warm.add_request("u", "t2", greedy(10), np.asarray(turn2, dtype=np.int32))
    assert run(warm) == cold


def test_cache_hit_through_pd(tmp_path: Path) -> None:
    """Hit applied on the PROMPT engine, then handed off: still equals cold NORMAL."""
    store = open_store("file", tmp_path)
    turn2 = turn1(SMALL, "cpu", store)
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 12)
    wire = []
    prefill = ServingLLMEngine(SMALL, EngineType.PROMPT, store=store, handoff_sink=lambda h: wire.append(h.to_bytes()))
    seq = prefill.add_request("u", "t2", greedy(12), np.asarray(turn2, dtype=np.int32))
    assert seq is not None and seq.cached_hit_tokens == 4 * SMALL.grain
    out = run(prefill)
    decode = ServingLLMEngine(SMALL, EngineType.COMPLETION)
    assert decode.receive_handoff(Handoff.from_bytes(wire[0])) is not None
    out["t2"].extend(run(decode)["t2"])
    assert out == cold


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_stored_block_changes_output(backend: str, tmp_path: Path) -> None:
    """Well-formed but wrong contents (written through the API, so no checksum can catch it) reach the output."""
    store = open_store(backend, tmp_path)
    turn2 = turn1(SMALL, "cpu", store)
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 10)
    for key in store.block_keys():
        kv = store.get_block(key)
        assert kv is not None
        store.put_block(key, noise_like(kv))
    warm = ServingLLMEngine(SMALL, EngineType.NORMAL, store=store)
    warm.add_request("u", "t2", greedy(10), np.asarray(turn2, dtype=np.int32))
    assert run(warm) != cold


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_stored_swa_record_changes_output(backend: str, tmp_path: Path) -> None:
    """The record only feeds the first `window` resumed positions; it must still reach the output via block KV."""
    store = open_store(backend, tmp_path)
    turn2 = turn1(SMALL, "cpu", store)
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 30)
    for key, position in store.swa_entries():
        state = store.get_swa(key, position)
        assert state is not None
        store.put_swa(key, position, 10 * noise_like(state))
    warm = ServingLLMEngine(SMALL, EngineType.NORMAL, store=store)
    seq = warm.add_request("u", "t2", greedy(30), np.asarray(turn2, dtype=np.int32))
    assert seq is not None and seq.num_cached_tokens == 4 * SMALL.grain  # resumes from the record
    assert run(warm) != cold
