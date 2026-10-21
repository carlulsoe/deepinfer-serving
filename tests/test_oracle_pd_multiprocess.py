"""PD oracle across processes: NORMAL (in-process) == PROMPT process -> wire -> COMPLETION process."""

from collections.abc import Iterator

import pytest
from helpers import CONFIGS, DEVICES, assert_baseline, chunked, greedy, pd_coordinator, run_normal, tokens

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.pd.entrypoint import Entrypoint, RequestState

_PARAMS = [
    pytest.param((c.values[0], d.values[0]), id=f"{c.id}-{d.id}", marks=d.marks) for c in CONFIGS for d in DEVICES
]


@pytest.fixture(scope="module", params=_PARAMS)
def cluster(request: pytest.FixtureRequest) -> Iterator[tuple[MiniConfig, str, Entrypoint]]:
    config, device = request.param
    # Chunked prefill on the PROMPT side; prompts below cover chunked and unchunked.
    with pd_coordinator(chunked(config, config.block_size + 3), config, device, device) as co:
        yield config, device, co


def test_cross_process_pd_equals_normal(cluster: tuple[MiniConfig, str, Entrypoint]) -> None:
    config, device, co = cluster
    g = config.block_size
    prompts = {
        f"{device}-{name}": tokens(n, seed)
        for seed, (name, n) in enumerate({"short": g // 2 + 1, "one_block": g, "several": 2 * g + 7}.items())
    }
    max_tokens = g + 5
    expected = run_normal(config, device, prompts, max_tokens)
    for rid, prompt in prompts.items():
        co.submit(rid, prompt, greedy(max_tokens))
    records = co.run_until_done()
    assert {rid: records[rid].tokens for rid in prompts} == expected
    for rid in prompts:
        assert records[rid].history == [
            RequestState.QUEUED,
            RequestState.PREFILLING,
            RequestState.HANDED_OFF,
            RequestState.DECODING,
            RequestState.FINISHED,
        ]
    assert_baseline(co)
