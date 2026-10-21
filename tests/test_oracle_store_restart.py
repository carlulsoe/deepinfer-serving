"""Cache-hit oracle across processes and restarts: PD instances share the block tier through a directory.

Turn 1 runs through a prefill and a decode process. The prefill process stores blocks 0-2 and its record at
3G; the decode process stores block 3 (and its record at 4G in its own RAM, ADR 0008).

- Same processes, turn 2: L1 = 4G needs the decode process's block from disk; the prefill process holds
  the record at 3G, not 4G, so it resumes at 3G.
- After restarting both processes, turn 2: blocks survive on disk, the SWA tier does not, so the hit backs
  off to 3G with a zeroed slot.

Both equal a cold in-process NORMAL run.
"""

from pathlib import Path

import pytest
from helpers import CONFIGS, DEVICES, UNBOUNDED, assert_baseline, greedy, run_normal, tokens

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.pd.coordinator import PDCoordinator
from deepinfer_serving.pd.instance import InstanceSpec, StoreSpec

_PARAMS = [
    pytest.param((c.values[0], d.values[0]), id=f"{c.id}-{d.id}", marks=d.marks) for c in CONFIGS for d in DEVICES
]


def _cluster(config: MiniConfig, device: str, store: StoreSpec) -> PDCoordinator:
    return PDCoordinator(
        InstanceSpec(EngineType.PROMPT, config, device, store),
        InstanceSpec(EngineType.COMPLETION, config, device, store),
    )


@pytest.mark.parametrize("param", _PARAMS)
def test_cache_hit_across_processes_and_restart(param: tuple[MiniConfig, str], tmp_path: Path) -> None:
    config, device = param
    g = config.grain
    store = StoreSpec(str(tmp_path), UNBOUNDED, UNBOUNDED)
    prompt = tokens(3 * g + 2, seed=10)

    with _cluster(config, device, store) as co:
        co.submit("t1", prompt, greedy(g + 4))
        turn2 = prompt + co.run_until_done()["t1"].tokens + tokens(config.window + config.backtrack + 3, seed=11)
        cold = run_normal(config, device, {"t2": turn2}, g + 3)["t2"]

        co.submit("t2", turn2, greedy(g + 3))
        record = co.run_until_done()["t2"]
        assert record.hit == (4 * g, 3 * g, 0)
        assert record.tokens == cold
        assert_baseline(co)

    with _cluster(config, device, store) as co:
        co.submit("t2-after-restart", turn2, greedy(g + 3))
        record = co.run_until_done()["t2-after-restart"]
        assert record.hit == (3 * g, 3 * g, 3 * g)
        assert record.tokens == cold
        assert_baseline(co)
