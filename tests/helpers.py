"""Shared configs and drivers for the oracle tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.llm_engine import LLMEngine
from deepinfer_serving.handoff import Handoff
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
        decode.receive_handoff(handoff)
    for rid, toks in run(decode).items():
        out[rid].extend(toks)
    return out


def noise_like(t: torch.Tensor) -> torch.Tensor:
    return torch.randn(t.shape, generator=torch.Generator().manual_seed(123))
