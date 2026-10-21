"""ADR 0009: a CPU prefill can hand off to a CUDA decode. Completion and cleanup only; no token equality."""

import pytest
from helpers import SMALL, assert_baseline, greedy, pd_coordinator, tokens

from deepinfer_serving.pd.entrypoint import RequestState


@pytest.mark.gpu
def test_cpu_prefill_to_cuda_decode_completes() -> None:
    with pd_coordinator(SMALL, SMALL, "cpu", "cuda") as co:
        co.submit("x", tokens(2 * SMALL.block_size + 3, 0), greedy(20))
        record = co.run_until_done()["x"]
        assert record.state == RequestState.FINISHED
        assert len(record.tokens) == 20
        assert_baseline(co)
