"""KV store interface, in-memory implementation, hash chain, and the V4.1 hit rule (contract 10.2-10.3)."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

BlockKey = bytes
"""Hash-chain key of a full block. Identifies the whole prefix ending at that block."""


def block_hashes(token_ids: Sequence[int], block_size: int) -> list[BlockKey]:
    """``h_i = sha256(h_{i-1} || int32 bytes of tokens[i*bs:(i+1)*bs])`` over full blocks only."""
    keys: list[BlockKey] = []
    prev = b""
    for i in range(len(token_ids) // block_size):
        chunk = torch.tensor(token_ids[i * block_size : (i + 1) * block_size], dtype=torch.int32)
        prev = hashlib.sha256(prev + chunk.numpy().tobytes()).digest()
        keys.append(prev)
    return keys


def swa_key(keys: list[BlockKey], position: int, grain: int) -> BlockKey:
    """SWA records are keyed by the chain hash of the block ending at ``position``."""
    if position < grain or position % grain:
        raise ValueError(f"SWA position {position} is not a positive multiple of {grain}")
    return keys[position // grain - 1]


class KVStore(Protocol):
    def has_block(self, key: BlockKey) -> bool: ...
    def put_block(self, key: BlockKey, kv: torch.Tensor) -> None: ...
    def get_block(self, key: BlockKey) -> torch.Tensor | None: ...
    def has_swa(self, key: BlockKey, position: int) -> bool: ...
    def put_swa(self, key: BlockKey, position: int, state: torch.Tensor) -> None: ...
    def get_swa(self, key: BlockKey, position: int) -> torch.Tensor | None: ...


LOW_WATERMARK = 0.9
"""When a tier exceeds its capacity, LRU eviction brings it down to this fraction of the capacity."""


def namespace(config_fingerprint: str) -> str:
    """Store sub-directory for one model/cache config. Block keys hash tokens only, so configs must not share."""
    return hashlib.sha256(config_fingerprint.encode()).hexdigest()[:16]


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class _LruTier[K]:
    """Tensors in host RAM, least recently used first, evicted under a byte capacity (``None``: unbounded)."""

    def __init__(self, capacity_bytes: int | None) -> None:
        self.capacity_bytes = capacity_bytes
        self.entries: OrderedDict[K, torch.Tensor] = OrderedDict()
        self.used_bytes = 0

    def __contains__(self, key: K) -> bool:
        return key in self.entries

    def put(self, key: K, value: torch.Tensor) -> None:
        self.discard(key)
        value = value.detach().cpu().clone()
        self.entries[key] = value
        self.used_bytes += _nbytes(value)
        if self.capacity_bytes is not None and self.used_bytes > self.capacity_bytes:
            while self.entries and self.used_bytes > LOW_WATERMARK * self.capacity_bytes:
                self.discard(next(iter(self.entries)))

    def get(self, key: K) -> torch.Tensor | None:
        value = self.entries.get(key)
        if value is None:
            return None
        self.entries.move_to_end(key)
        return value.clone()

    def discard(self, key: K) -> None:
        value = self.entries.pop(key, None)
        if value is not None:
            self.used_bytes -= _nbytes(value)


class InMemoryKVStore:
    """Both tiers in host RAM. The SWA tier of every backend is one of these (ADR 0008, 0010)."""

    def __init__(self, block_capacity_bytes: int | None = None, swa_capacity_bytes: int | None = None) -> None:
        self._blocks: _LruTier[BlockKey] = _LruTier(block_capacity_bytes)
        self._swa: _LruTier[tuple[BlockKey, int]] = _LruTier(swa_capacity_bytes)

    def has_block(self, key: BlockKey) -> bool:
        return key in self._blocks

    def put_block(self, key: BlockKey, kv: torch.Tensor) -> None:
        self._blocks.put(key, kv)

    def get_block(self, key: BlockKey) -> torch.Tensor | None:
        return self._blocks.get(key)

    def has_swa(self, key: BlockKey, position: int) -> bool:
        return (key, position) in self._swa

    def put_swa(self, key: BlockKey, position: int, state: torch.Tensor) -> None:
        self._swa.put((key, position), state)

    def get_swa(self, key: BlockKey, position: int) -> torch.Tensor | None:
        return self._swa.get((key, position))

    # ---- administration (eviction, tests); not part of the KVStore protocol ----

    def block_keys(self) -> list[BlockKey]:
        return list(self._blocks.entries)

    def discard_block(self, key: BlockKey) -> None:
        self._blocks.discard(key)

    def swa_entries(self) -> list[tuple[BlockKey, int]]:
        return list(self._swa.entries)

    def discard_swa(self, key: BlockKey, position: int) -> None:
        self._swa.discard((key, position))

    def swa_positions(self) -> set[int]:
        return {pos for _, pos in self._swa.entries}

    def block_bytes_used(self) -> int:
        return self._blocks.used_bytes

    def swa_bytes_used(self) -> int:
        return self._swa.used_bytes


@dataclass(frozen=True)
class Hit:
    cached_hit_tokens: int
    """Reported hit length (upstream ``Sequence.cached_hit_tokens``)."""
    num_cached_tokens: int
    """Where compute resumes. ``[num_cached_tokens, cached_hit_tokens)`` is recomputed when they differ."""
    swa_valid_start: int
    """Lowest valid SWA position. Nonzero only when resuming from a zeroed slot."""
    block_keys: list[BlockKey]
    """Blocks to load, covering ``[0, len(block_keys) * block_size)``."""
    swa_position: int | None
    """Position of the SWA record to load, or ``None`` for a zeroed slot."""

    @classmethod
    def none(cls) -> Hit:
        return cls(0, 0, 0, [], None)


def compute_hit(store: KVStore, token_ids: Sequence[int], *, block_size: int, window: int, backtrack: int) -> Hit:
    """Hit rule. Metadata-only: checks presence, does not fetch."""
    grain = block_size
    cap = len(token_ids) - (window + backtrack)
    if cap < grain:
        return Hit.none()
    keys = block_hashes(token_ids, block_size)
    num_blocks = 0
    while num_blocks < cap // grain and store.has_block(keys[num_blocks]):
        num_blocks += 1
    l1 = num_blocks * grain
    if l1 == 0:
        return Hit.none()
    loaded = keys[:num_blocks]
    if store.has_swa(swa_key(keys, l1, grain), l1):
        return Hit(l1, l1, 0, loaded, l1)
    back = l1 - grain
    if back == 0:  # contract A3: no record exists at position 0
        return Hit.none()
    if store.has_swa(swa_key(keys, back, grain), back):
        return Hit(l1, back, 0, loaded, back)
    return Hit(back, back, back, loaded, None)
