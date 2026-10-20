"""Cache-hit oracle: turn 2 served from the store == cold turn 2, token for token."""

import numpy as np
import pytest
from helpers import CONFIGS, DEVICES, SMALL, chunked, greedy, noise_like, run, run_normal, tokens

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.kvstore import InMemoryKVStore
from deepinfer_serving.serving_engine import ServingLLMEngine


def _turn1(config: MiniConfig, device: str) -> tuple[InMemoryKVStore, list[int]]:
    """Prompt 3G+2, reply G+4. Persists the prefill record at 3G and the decode record at 4G."""
    g = config.grain
    prompt = tokens(3 * g + 2, seed=10)
    store = InMemoryKVStore()
    engine = ServingLLMEngine(config, EngineType.NORMAL, device, store=store)
    engine.add_request("u", "t1", greedy(g + 4), np.asarray(prompt, dtype=np.int32))
    reply = run(engine)["t1"]
    assert store.swa_positions() == {3 * g, 4 * g}
    assert len(store.blocks) == 4  # full blocks of [0, 4G + 5)
    new_user = tokens(config.window + config.backtrack + 3, seed=11)
    return store, prompt + reply + new_user


def _drop_swa(store: InMemoryKVStore, positions: set[int]) -> None:
    for key in [k for k in store.swa if k[1] in positions]:
        del store.swa[key]


# case -> (records to delete, cached_hit_tokens, num_cached_tokens, swa_valid_start), in units of G
CASES = {
    "record_at_l1": (set(), 4, 4, 0),
    "record_at_l1_minus_g": ({4}, 4, 3, 0),
    "no_record_backoff": ({3, 4}, 3, 3, 3),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", list(CASES))
def test_cache_hit_equals_cold(config: MiniConfig, device: str, case: str) -> None:
    g = config.grain
    store, turn2 = _turn1(config, device)
    drop, hit, cached, valid_start = CASES[case]
    _drop_swa(store, {p * g for p in drop})
    max_tokens = g + 3

    cold = run_normal(config, device, {"t2": turn2}, max_tokens)

    warm_engine = ServingLLMEngine(config, EngineType.NORMAL, device, store=store)  # a different "machine"
    seq = warm_engine.add_request("u", "t2", greedy(max_tokens), np.asarray(turn2, dtype=np.int32))
    assert seq is not None
    assert (seq.cached_hit_tokens, seq.num_cached_tokens, seq.swa_valid_start) == (hit * g, cached * g, valid_start * g)
    assert run(warm_engine) == cold


def test_cache_hit_with_chunked_prefill_and_backoff() -> None:
    store, turn2 = _turn1(SMALL, "cpu")
    _drop_swa(store, {3 * SMALL.grain, 4 * SMALL.grain})
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 10)
    warm = ServingLLMEngine(chunked(SMALL, SMALL.block_size + 1), EngineType.NORMAL, store=store)
    warm.add_request("u", "t2", greedy(10), np.asarray(turn2, dtype=np.int32))
    assert run(warm) == cold


def test_cache_hit_through_pd() -> None:
    """Hit applied on the PROMPT engine, then handed off: still equals cold NORMAL."""
    store, turn2 = _turn1(SMALL, "cpu")
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 12)
    wire = []
    prefill = ServingLLMEngine(SMALL, EngineType.PROMPT, store=store, handoff_sink=lambda h: wire.append(h.to_bytes()))
    seq = prefill.add_request("u", "t2", greedy(12), np.asarray(turn2, dtype=np.int32))
    assert seq is not None and seq.cached_hit_tokens == 4 * SMALL.grain
    out = run(prefill)
    decode = ServingLLMEngine(SMALL, EngineType.COMPLETION)
    decode.receive_handoff(Handoff.from_bytes(wire[0]))
    out["t2"].extend(run(decode)["t2"])
    assert out == cold


def test_corrupt_stored_block_changes_output() -> None:
    store, turn2 = _turn1(SMALL, "cpu")
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 10)
    for key, kv in store.blocks.items():
        store.blocks[key] = noise_like(kv)
    warm = ServingLLMEngine(SMALL, EngineType.NORMAL, store=store)
    warm.add_request("u", "t2", greedy(10), np.asarray(turn2, dtype=np.int32))
    assert run(warm) != cold


def test_corrupt_stored_swa_record_changes_output() -> None:
    """The record only feeds the first `window` resumed positions; it must still reach the output via block KV."""
    store, turn2 = _turn1(SMALL, "cpu")
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 30)
    for key, state in store.swa.items():
        store.swa[key] = 10 * noise_like(state)
    warm = ServingLLMEngine(SMALL, EngineType.NORMAL, store=store)
    seq = warm.add_request("u", "t2", greedy(30), np.asarray(turn2, dtype=np.int32))
    assert seq is not None and seq.num_cached_tokens == 4 * SMALL.grain  # resumes from the record
    assert run(warm) != cold
