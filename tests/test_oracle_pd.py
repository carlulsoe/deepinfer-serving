"""PD oracle: NORMAL output == PROMPT -> Handoff bytes -> COMPLETION output, token for token."""

from dataclasses import replace

import numpy as np
import pytest
from helpers import CONFIGS, DEVICES, SMALL, chunked, greedy, noise_like, run_normal, run_pd, tokens

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.serving_engine import ServingLLMEngine


def _prompts(config: MiniConfig) -> dict[str, list[int]]:
    g = config.block_size
    lengths = {"short": g // 2 + 1, "one_block": g, "several": 2 * g + 7, "long": 3 * g + 1}
    return {name: tokens(n, seed) for seed, (name, n) in enumerate(lengths.items())}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("prefill_chunk", [None, "chunked"])
def test_pd_equals_normal(config: MiniConfig, device: str, prefill_chunk: str | None) -> None:
    prompts = _prompts(config)
    max_tokens = config.block_size + 5  # crosses at least one block boundary during decode
    expected = run_normal(config, device, prompts, max_tokens)
    p_config = chunked(config, config.block_size + 3) if prefill_chunk else config
    assert run_pd(p_config, config, device, prompts, max_tokens) == expected
    assert all(len(v) == max_tokens for v in expected.values())


def test_pd_first_token_final_sends_no_handoff() -> None:
    sent: list[Handoff] = []
    prefill = ServingLLMEngine(SMALL, EngineType.PROMPT, handoff_sink=sent.append)
    prefill.add_request("u", "r", greedy(1), np.asarray(tokens(20, 0), dtype=np.int32))
    outputs = prefill.step()
    assert [o.finished for o in outputs] == [True]
    assert sent == []
    assert not prefill.has_unfinished_requests()


def test_pd_corrupt_swa_state_changes_output() -> None:
    prompts = _prompts(SMALL)
    expected = run_normal(SMALL, "cpu", prompts, 20)
    corrupted = run_pd(SMALL, SMALL, "cpu", prompts, 20, lambda h: replace(h, swa_state=noise_like(h.swa_state)))
    assert corrupted != expected


def test_pd_corrupt_block_kv_changes_output() -> None:
    prompts = _prompts(SMALL)
    expected = run_normal(SMALL, "cpu", prompts, 20)
    corrupted = run_pd(SMALL, SMALL, "cpu", prompts, 20, lambda h: replace(h, block_kv=noise_like(h.block_kv)))
    assert corrupted != expected


def test_handoff_rejects_config_mismatch() -> None:
    sent: list[Handoff] = []
    prefill = ServingLLMEngine(SMALL, EngineType.PROMPT, handoff_sink=sent.append)
    prefill.add_request("u", "r", greedy(4), np.asarray(tokens(20, 0), dtype=np.int32))
    prefill.step()
    decode = ServingLLMEngine(replace(SMALL, seed=1), EngineType.COMPLETION)
    with pytest.raises(ValueError, match="different model/cache config"):
        decode.receive_handoff(sent[0])
