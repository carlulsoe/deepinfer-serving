"""KV store interface, in-memory implementation, hash chain, and the V4.1 hit rule (contract 10.2-10.3)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
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


@dataclass
class InMemoryKVStore:
    blocks: dict[BlockKey, torch.Tensor] = field(default_factory=dict)
    swa: dict[tuple[BlockKey, int], torch.Tensor] = field(default_factory=dict)

    def has_block(self, key: BlockKey) -> bool:
        return key in self.blocks

    def put_block(self, key: BlockKey, kv: torch.Tensor) -> None:
        self.blocks[key] = kv.detach().cpu().clone()

    def get_block(self, key: BlockKey) -> torch.Tensor | None:
        return self.blocks.get(key)

    def has_swa(self, key: BlockKey, position: int) -> bool:
        return (key, position) in self.swa

    def put_swa(self, key: BlockKey, position: int, state: torch.Tensor) -> None:
        self.swa[(key, position)] = state.detach().cpu().clone()

    def get_swa(self, key: BlockKey, position: int) -> torch.Tensor | None:
        return self.swa.get((key, position))

    def swa_positions(self) -> set[int]:
        return {pos for _, pos in self.swa}


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
