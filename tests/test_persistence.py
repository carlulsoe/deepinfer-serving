"""Persistence-position tests: which records and blocks reach the store, and when."""

from dataclasses import replace

import numpy as np
import pytest
from helpers import CONFIGS, greedy, run, tokens

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.kvstore import InMemoryKVStore
from deepinfer_serving.serving_engine import ServingLLMEngine


def _engine(config: MiniConfig) -> tuple[ServingLLMEngine, InMemoryKVStore]:
    store = InMemoryKVStore()
    return ServingLLMEngine(config, EngineType.NORMAL, store=store), store


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize(
    ("length_in_g", "extra", "record_in_g"),
    [(1, 0, None), (1, 1, 1), (3, 0, 2), (3, 2, 3)],
    ids=["G", "G+1", "3G", "3G+2"],
)
def test_after_prefill_record_position(
    config: MiniConfig, length_in_g: int, extra: int, record_in_g: int | None
) -> None:
    g = config.grain
    length = length_in_g * g + extra
    engine, store = _engine(config)
    engine.add_request("u", "r", greedy(4), np.asarray(tokens(length, 0), dtype=np.int32))
    engine.step()  # the prefill step
    assert store.swa_positions() == (set() if record_in_g is None else {record_in_g * g})
    assert len(store.block_keys()) == length // g


@pytest.mark.parametrize("config", CONFIGS)
def test_nothing_stored_mid_chunk(config: MiniConfig) -> None:
    g = config.grain
    engine, store = _engine(replace(config, max_num_batched_tokens=g))
    engine.add_request("u", "r", greedy(4), np.asarray(tokens(3 * g + 2, 0), dtype=np.int32))
    for _ in range(3):  # chunks of G, G, G
        engine.step()
        assert store.swa_positions() == set() and not store.block_keys()
    engine.step()  # last chunk (2 tokens)
    assert store.swa_positions() == {3 * g}
    assert len(store.block_keys()) == 3


@pytest.mark.parametrize("config", CONFIGS)
def test_decode_snapshots_at_grain_crossings(config: MiniConfig) -> None:
    g, b = config.grain, config.backtrack
    prompt_len = g + b  # first decode step overshoots G by B: too far, no snapshot at G
    final_len = 3 * g + 5
    engine, store = _engine(config)
    seq = engine.add_request(
        "u", "r", greedy(final_len - prompt_len), np.asarray(tokens(prompt_len, 0), dtype=np.int32)
    )
    assert seq is not None
    snapshots: list[int] = []
    while engine.has_unfinished_requests():
        engine.step()
        if seq.last_copy_state_token_idx and (not snapshots or snapshots[-1] != seq.last_copy_state_token_idx):
            snapshots.append(seq.last_copy_state_token_idx)
    assert snapshots == [2 * g, 3 * g]
    # prefill record at floor((G+B-1)/G)*G = G, finish record from the last snapshot (3G); 2G is not stored
    assert store.swa_positions() == {g, 3 * g}
    assert len(store.block_keys()) == 3  # full blocks of [0, 3G + 4)


@pytest.mark.parametrize("config", CONFIGS)
def test_decode_snapshot_tolerates_small_overshoot(config: MiniConfig) -> None:
    g = config.grain
    engine, store = _engine(config)
    seq = engine.add_request("u", "r", greedy(3), np.asarray(tokens(2 * g + 1, 0), dtype=np.int32))
    assert seq is not None
    run(engine)
    assert seq.last_copy_state_token_idx == 2 * g + 1  # overshoot 1 < backtrack
    assert store.swa_positions() == {2 * g}
