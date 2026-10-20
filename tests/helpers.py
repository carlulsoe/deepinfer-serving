"""Shared configs and drivers for the oracle tests."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.llm_engine import LLMEngine
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.pd.coordinator import PDCoordinator
from deepinfer_serving.pd.instance import InstanceSpec
from deepinfer_serving.serving_engine import ServingLLMEngine

SMALL = MiniConfig(block_size=16, window=16, backtrack=3, store_extra=16, num_gpu_blocks=128)
DEFAULT = MiniConfig()

CONFIGS = [pytest.param(SMALL, id="small"), pytest.param(DEFAULT, id="default")]
DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)]


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


# ---- cross-process PD (milestone 2) ----


def pd_coordinator(
    p_config: MiniConfig, d_config: MiniConfig, p_device: str = "cpu", d_device: str = "cpu"
) -> PDCoordinator:
    return PDCoordinator(
        InstanceSpec(EngineType.PROMPT, p_config, p_device), InstanceSpec(EngineType.COMPLETION, d_config, d_device)
    )


def assert_baseline(co: PDCoordinator, timeout: float = 30.0) -> None:
    """Every request terminated exactly once, no protocol errors, and allocator counts back to baseline."""
    co.run_until_done(timeout)
    for role in co.ROLES:
        deadline = time.monotonic() + timeout
        while True:
            s = co.stats(role)
            at_baseline = (
                s["free_blocks"] == s["num_blocks"]
                and s["free_slots"] == s["num_slots"]
                and s["unfinished"] == 0
                and s["outbox"] == 0
                and s["inbox"] == 0
            )
            if at_baseline:
                break
            assert time.monotonic() < deadline, f"{role} not at baseline: {s}"
            co.poll(0.01)
    # Checked last: a stats reply at baseline means every earlier event from that instance has been handled.
    assert co.errors == []
    assert co.duplicate_terminals == 0
