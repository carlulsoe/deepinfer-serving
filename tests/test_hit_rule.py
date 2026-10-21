"""Hit-rule unit tests against an in-memory store (metadata only)."""

import pytest
import torch
from helpers import CONFIGS, tokens

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.kvstore import Hit, InMemoryKVStore, block_hashes, compute_hit, swa_key

_DUMMY = torch.zeros(1)


def _store(config: MiniConfig, toks: list[int], blocks: int, swa: set[int]) -> InMemoryKVStore:
    keys = block_hashes(toks, config.block_size)
    store = InMemoryKVStore()
    for key in keys[:blocks]:
        store.put_block(key, _DUMMY)
    for pos in swa:
        store.put_swa(swa_key(keys, pos, config.grain), pos, _DUMMY)
    return store


def _hit(config: MiniConfig, store: InMemoryKVStore, toks: list[int]) -> Hit:
    return compute_hit(store, toks, block_size=config.block_size, window=config.window, backtrack=config.backtrack)


@pytest.mark.parametrize("config", CONFIGS)
def test_cap_is_len_minus_window_plus_backtrack(config: MiniConfig) -> None:
    g, wb = config.grain, config.swa_window_len
    toks = tokens(7 * g, seed=0)
    store = _store(config, toks, blocks=7, swa={g * i for i in range(1, 8)})
    # cap = 5G + wb - 1 - wb = 5G - 1 -> L1 = 4G, even though more blocks are stored
    assert _hit(config, store, toks[: 5 * g + wb - 1]).cached_hit_tokens == 4 * g
    # cap = 5G exactly -> L1 = 5G
    assert _hit(config, store, toks[: 5 * g + wb]).cached_hit_tokens == 5 * g


@pytest.mark.parametrize("config", CONFIGS)
def test_hit_is_grain_aligned(config: MiniConfig) -> None:
    g = config.grain
    toks = tokens(6 * g, seed=1)
    store = _store(config, toks, blocks=6, swa={g * i for i in range(1, 7)})
    for length in range(2 * g, 6 * g, 7):
        hit = _hit(config, store, toks[:length])
        assert hit.cached_hit_tokens % g == 0
        assert hit.num_cached_tokens % g == 0
        assert hit.cached_hit_tokens <= length - config.swa_window_len


@pytest.mark.parametrize("config", CONFIGS)
def test_swa_present_at_l1(config: MiniConfig) -> None:
    g = config.grain
    toks = tokens(4 * g + config.swa_window_len, seed=2)
    store = _store(config, toks, blocks=4, swa={4 * g})
    hit = _hit(config, store, toks)
    assert (hit.cached_hit_tokens, hit.num_cached_tokens, hit.swa_valid_start, hit.swa_position) == (
        4 * g,
        4 * g,
        0,
        4 * g,
    )
    assert len(hit.block_keys) == 4


@pytest.mark.parametrize("config", CONFIGS)
def test_swa_present_only_one_block_back(config: MiniConfig) -> None:
    g = config.grain
    toks = tokens(4 * g + config.swa_window_len, seed=3)
    store = _store(config, toks, blocks=4, swa={3 * g})
    hit = _hit(config, store, toks)
    assert (hit.cached_hit_tokens, hit.num_cached_tokens, hit.swa_valid_start, hit.swa_position) == (
        4 * g,
        3 * g,
        0,
        3 * g,
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_swa_missing_backs_off_one_block(config: MiniConfig) -> None:
    g = config.grain
    toks = tokens(4 * g + config.swa_window_len, seed=4)
    store = _store(config, toks, blocks=4, swa={2 * g})  # records further back do not count
    hit = _hit(config, store, toks)
    assert (hit.cached_hit_tokens, hit.num_cached_tokens, hit.swa_valid_start, hit.swa_position) == (
        3 * g,
        3 * g,
        3 * g,
        None,
    )
    assert len(hit.block_keys) == 4


@pytest.mark.parametrize("config", CONFIGS)
def test_short_prompts_never_hit(config: MiniConfig) -> None:
    g = config.grain
    toks = tokens(4 * g, seed=5)
    store = _store(config, toks, blocks=4, swa={g, 2 * g, 3 * g, 4 * g})
    assert _hit(config, store, toks[: g // 2]) == Hit.none()  # below one block
    assert _hit(config, store, toks[: g + config.swa_window_len - 1]) == Hit.none()  # cap below one block
    assert _hit(config, store, toks[: g + config.swa_window_len]).cached_hit_tokens == g


@pytest.mark.parametrize("config", CONFIGS)
def test_one_block_without_record_is_no_hit(config: MiniConfig) -> None:
    """Contract A3: no record exists at position 0, so L1 = G without a record at G backs off to 0."""
    g = config.grain
    toks = tokens(g + config.swa_window_len, seed=6)
    assert _hit(config, _store(config, toks, blocks=1, swa=set()), toks) == Hit.none()


@pytest.mark.parametrize("config", CONFIGS)
def test_block_hits_must_be_contiguous(config: MiniConfig) -> None:
    g = config.grain
    toks = tokens(5 * g + config.swa_window_len, seed=7)
    store = _store(config, toks, blocks=5, swa={g * i for i in range(1, 6)})
    store.discard_block(block_hashes(toks, g)[2])
    assert _hit(config, store, toks).cached_hit_tokens == 2 * g


def test_hash_chain_identifies_prefix() -> None:
    a, b = tokens(48, seed=8), tokens(48, seed=9)
    mixed = b[:16] + a[16:]
    ka, km = block_hashes(a, 16), block_hashes(mixed, 16)
    assert ka[1] != km[1]  # same block-1 tokens, different prefix
    assert block_hashes(a[:40], 16) == ka[:2]  # partial blocks are not hashed
