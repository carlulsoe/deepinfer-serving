"""Loss oracle: eviction, deletion and damage make hits shorter, never outputs wrong (roadmap, milestone 3)."""

from pathlib import Path

import numpy as np
import pytest
from helpers import BACKENDS, CONFIGS, SMALL, drop_swa, greedy, open_store, run, run_normal, turn1

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.file_store import FileKVStore
from deepinfer_serving.kvstore import block_hashes
from deepinfer_serving.serving_engine import ServingLLMEngine

# Turn 1 leaves blocks 0-3 and records at 3G and 4G; turn 2 caps L1 at 4G.
# case -> (blocks to delete, records to delete in G, cached_hit_tokens, num_cached_tokens, swa_valid_start) in G
CASES = {
    "first_block": ({0}, set(), 0, 0, 0),
    "middle_block": ({2}, set(), 1, 1, 1),  # L1 = 2G, no record at 2G or G: back off to G
    "last_block": ({3}, set(), 3, 3, 0),  # L1 = 3G, record at 3G
    "last_block_and_record": ({3}, {3}, 2, 2, 2),
    "all_blocks": ({0, 1, 2, 3}, set(), 0, 0, 0),
}


def _warm(config: MiniConfig, store, turn2: list[int], max_tokens: int) -> tuple[tuple[int, int, int], dict]:
    engine = ServingLLMEngine(config, EngineType.NORMAL, store=store)
    seq = engine.add_request("u", "t2", greedy(max_tokens), np.asarray(turn2, dtype=np.int32))
    assert seq is not None
    return (seq.cached_hit_tokens, seq.num_cached_tokens, seq.swa_valid_start), run(engine)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", list(CASES))
def test_deleted_entries_shorten_the_hit(config: MiniConfig, backend: str, case: str, tmp_path: Path) -> None:
    g = config.grain
    store = open_store(backend, tmp_path)
    turn2 = turn1(config, "cpu", store)
    blocks, records, hit, cached, valid_start = CASES[case]
    keys = block_hashes(turn2, config.block_size)
    for i in blocks:
        store.discard_block(keys[i])
    drop_swa(store, {p * g for p in records})
    cold = run_normal(config, "cpu", {"t2": turn2}, g + 3)
    got_hit, warm = _warm(config, store, turn2, g + 3)
    assert got_hit == (hit * g, cached * g, valid_start * g)
    assert warm == cold


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("blocks_kept", [1, 2, 3])
def test_eviction_under_capacity_still_equals_cold(backend: str, blocks_kept: int, tmp_path: Path) -> None:
    """A block tier too small for turn 1 evicts during it; whatever survives, turn 2 equals cold."""
    probe = open_store(backend, tmp_path / "probe")
    turn2 = turn1(SMALL, "cpu", probe)
    block_bytes = probe.block_bytes_used() // len(probe.block_keys())
    store = open_store(backend, tmp_path / "small", block_capacity_bytes=int((blocks_kept + 0.5) * block_bytes))
    engine = ServingLLMEngine(SMALL, EngineType.NORMAL, store=store)
    engine.add_request("u", "t1", greedy(SMALL.grain + 4), np.asarray(turn2[: 3 * SMALL.grain + 2], dtype=np.int32))
    run(engine)
    assert 0 < len(store.block_keys()) <= blocks_kept
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 12)
    _, warm = _warm(SMALL, store, turn2, 12)
    assert warm == cold


def test_damaged_block_files_are_served_cold(tmp_path: Path) -> None:
    """Presence says hit; the fetch finds a bad checksum, so the request is served cold and the block re-stored."""
    store = open_store("file", tmp_path)
    assert isinstance(store, FileKVStore)
    turn2 = turn1(SMALL, "cpu", store)
    key = block_hashes(turn2, SMALL.block_size)[1]
    victim = store.path / key.hex()
    good = victim.read_bytes()
    data = bytearray(good)
    data[-5] ^= 0x01
    victim.write_bytes(bytes(data))
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 12)
    got_hit, warm = _warm(SMALL, store, turn2, 12)
    assert got_hit == (0, 0, 0)
    assert warm == cold
    assert victim.read_bytes() == good  # the damaged file was dropped, then turn 2's prefill stored it again


class _EvictOnFetch(FileKVStore):
    """Loses an entry between the metadata match and the fetch, as concurrent eviction would."""

    def __init__(self, *args, what: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.what = what

    def get_block(self, key: bytes):
        if self.what == "block":
            self.what = ""
            self.discard_block(key)
        return super().get_block(key)

    def get_swa(self, key: bytes, position: int):
        if self.what == "swa":
            self.what = ""
            self.discard_swa(key, position)
        return super().get_swa(key, position)


@pytest.mark.parametrize("what", ["block", "swa"])
def test_entry_evicted_between_match_and_fetch_is_served_cold(what: str, tmp_path: Path) -> None:
    store = FileKVStore(tmp_path, 1 << 40, 1 << 40)
    turn2 = turn1(SMALL, "cpu", store)
    racing = _EvictOnFetch(tmp_path, 1 << 40, 1 << 40, what=what)
    racing.swa_tier = store.swa_tier  # same process: same host-RAM tier
    cold = run_normal(SMALL, "cpu", {"t2": turn2}, 12)
    got_hit, warm = _warm(SMALL, racing, turn2, 12)
    assert got_hit == (0, 0, 0)
    assert warm == cold
