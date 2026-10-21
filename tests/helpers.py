"""Shared configs and drivers for the oracle tests."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.llm_engine import LLMEngine
from deepinfer_serving.file_store import FileKVStore
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.kvstore import InMemoryKVStore, block_hashes
from deepinfer_serving.pd.entrypoint import Entrypoint, RequestState
from deepinfer_serving.pd.instance import InstanceSpec, StoreSpec
from deepinfer_serving.serving_engine import ServingLLMEngine

SMALL = MiniConfig(block_size=16, window=16, backtrack=3, store_extra=16, num_gpu_blocks=128)
DEFAULT = MiniConfig()

CONFIGS = [pytest.param(SMALL, id="small"), pytest.param(DEFAULT, id="default")]
DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)]


BACKENDS = ["memory", "file"]
UNBOUNDED = 1 << 40

Store = InMemoryKVStore | FileKVStore


def open_store(
    backend: str, path: Path, block_capacity_bytes: int = UNBOUNDED, swa_capacity_bytes: int = UNBOUNDED
) -> Store:
    """A store for ``backend``; ``path`` is used only by file-backed ones."""
    if backend == "memory":
        return InMemoryKVStore(block_capacity_bytes, swa_capacity_bytes)
    return FileKVStore(path, block_capacity_bytes, swa_capacity_bytes)


def store_keys(n: int) -> list[bytes]:
    return block_hashes(list(range(n * 4)), 4)


def store_tensor(seed: int) -> torch.Tensor:
    return torch.randn((2, 2, 4, 2, 8), generator=torch.Generator().manual_seed(seed))


def store_writer(path: str, block_capacity_bytes: int, seed: int) -> None:
    """Spawned by the concurrency test: write 40 shared keys in a random order, checking every read."""
    store = FileKVStore(path, block_capacity_bytes, UNBOUNDED)
    keys = store_keys(40)
    order = np.random.default_rng(seed).permutation(len(keys))
    for i in order:
        store.put_block(keys[i], store_tensor(int(i)))
        for j in order[:5]:
            got = store.get_block(keys[j])
            assert got is None or torch.equal(got, store_tensor(int(j)))


def chunked(config: MiniConfig, max_num_batched_tokens: int) -> MiniConfig:
    return replace(config, max_num_batched_tokens=max_num_batched_tokens)


def tokens(n: int, seed: int) -> list[int]:
    # Avoid EOS in prompts so tests read clearly; generation uses ignore_eos.
    rng = np.random.default_rng(seed)
    return [int(t) for t in rng.integers(2, SMALL.vocab_size, size=n)]


def greedy(max_tokens: int) -> dict:
    return {"temperature": 0.0, "max_tokens": max_tokens, "ignore_eos": True}


def run(engine: LLMEngine) -> dict[str, list[int]]:
    """Step until idle; return generated token ids per request id."""
    out: dict[str, list[int]] = {}
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.token_id >= 0:
                out.setdefault(o.request_id, []).append(o.token_id)
    return out


def run_normal(config: MiniConfig, device: str, prompts: dict[str, list[int]], max_tokens: int) -> dict[str, list[int]]:
    engine = LLMEngine(config, EngineType.NORMAL, device)
    for rid, p in prompts.items():
        engine.add_request("u", rid, greedy(max_tokens), np.asarray(p, dtype=np.int32))
    return run(engine)


def run_pd(
    p_config: MiniConfig,
    c_config: MiniConfig,
    device: str,
    prompts: dict[str, list[int]],
    max_tokens: int,
    corrupt=None,
) -> dict[str, list[int]]:
    """PROMPT engine -> Handoff bytes -> COMPLETION engine. Returns first token + decode tokens."""
    wire: list[bytes] = []
    prefill = ServingLLMEngine(p_config, EngineType.PROMPT, device, handoff_sink=lambda h: wire.append(h.to_bytes()))
    for rid, p in prompts.items():
        prefill.add_request("u", rid, greedy(max_tokens), np.asarray(p, dtype=np.int32))
    out = run(prefill)

    decode = ServingLLMEngine(c_config, EngineType.COMPLETION, device)
    for data in wire:
        handoff = Handoff.from_bytes(data)
        if corrupt is not None:
            handoff = corrupt(handoff)
        assert decode.receive_handoff(handoff) is not None
    for rid, toks in run(decode).items():
        out[rid].extend(toks)
    return out


def noise_like(t: torch.Tensor) -> torch.Tensor:
    return torch.randn(t.shape, generator=torch.Generator().manual_seed(123))


# ---- store (milestone 3) ----


def turn1(config: MiniConfig, device: str, store: Store) -> list[int]:
    """Prompt 3G+2, reply G+4. Persists the prefill record at 3G and the decode record at 4G. Returns turn 2."""
    g = config.grain
    prompt = tokens(3 * g + 2, seed=10)
    engine = ServingLLMEngine(config, EngineType.NORMAL, device, store=store)
    engine.add_request("u", "t1", greedy(g + 4), np.asarray(prompt, dtype=np.int32))
    reply = run(engine)["t1"]
    assert store.swa_positions() == {3 * g, 4 * g}
    assert len(store.block_keys()) == 4  # full blocks of [0, 4G + 5)
    new_user = tokens(config.window + config.backtrack + 3, seed=11)
    return prompt + reply + new_user


def drop_swa(store: Store, positions: set[int]) -> None:
    for key, position in store.swa_entries():
        if position in positions:
            store.discard_swa(key, position)


# ---- cross-process PD (milestone 2) ----


def pd_coordinator(
    p_config: MiniConfig, d_config: MiniConfig, p_device: str = "cpu", d_device: str = "cpu"
) -> Entrypoint:
    """One prefill instance named ``prefill`` and one decode instance named ``decode``."""
    return Entrypoint(
        {
            "prefill": InstanceSpec(EngineType.PROMPT, p_config, p_device),
            "decode": InstanceSpec(EngineType.COMPLETION, d_config, d_device),
        }
    )


def cluster(
    config: MiniConfig,
    n_prefill: int,
    n_decode: int,
    store: StoreSpec | None = None,
    device: str = "cpu",
    d_config: MiniConfig | None = None,
) -> Entrypoint:
    """Instances named ``prefill-<i>`` and ``decode-<j>``."""
    specs = {f"prefill-{i}": InstanceSpec(EngineType.PROMPT, config, device, store) for i in range(n_prefill)}
    specs |= {
        f"decode-{j}": InstanceSpec(EngineType.COMPLETION, d_config or config, device, store) for j in range(n_decode)
    }
    return Entrypoint(specs)


def decode_some(co: Entrypoint, decode: str, rids: list[str], n: int = 3, timeout: float = 30.0) -> None:
    """``decode`` must be paused before the requests were submitted. Step it until every request has at least
    ``n`` tokens, and leave it paused, so no request can finish before the test acts."""
    deadline = time.monotonic() + timeout
    co.wait_for(lambda: all(co.requests[r].state == RequestState.HANDED_OFF for r in rids), timeout)
    for done in (
        lambda: all(co.requests[r].state == RequestState.DECODING for r in rids),  # admit them all first
        lambda: all(len(co.requests[r].tokens_by_index) >= n for r in rids),
    ):
        while not done():
            assert time.monotonic() < deadline, "requests did not reach the token count"
            co.step(decode)
            co.stats(decode)  # a round trip, so steps are not queued faster than they run


def assert_baseline(co: Entrypoint, timeout: float = 30.0) -> None:
    """Every request terminated exactly once, no protocol errors, and every live instance back to baseline."""
    co.run_until_done(timeout)
    for role in co.live:
        deadline = time.monotonic() + timeout
        while True:
            s = co.stats(role)
            at_baseline = (
                s["free_blocks"] == s["num_blocks"]
                and s["free_slots"] == s["num_slots"]
                and s["unfinished"] == 0
                and s["outbox"] == 0
                and s["inbox"] == 0
                and s["pending_pulls"] == 0
            )
            if at_baseline:
                break
            assert time.monotonic() < deadline, f"{role} not at baseline: {s}"
            co.poll(0.01)
    # Checked last: a stats reply at baseline means every earlier event from that instance has been handled.
    assert co.errors == []
    assert co.duplicate_terminals == 0
