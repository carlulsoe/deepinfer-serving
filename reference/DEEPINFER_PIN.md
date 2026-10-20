# DeepInfer pin

- Upstream: https://github.com/deepseek-ai/DeepInfer (Apache-2.0)
- Pinned commit: `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (2026-10-20, "update Q&A EN doc")

DeepInfer is not vendored or added as a submodule. Its submodules pull in SM100-only kernel
repos (FlashMLA, DeepGEMM, DeepEP) that cannot be built here. To read the source:

```sh
git init deepinfer && cd deepinfer
git remote add origin https://github.com/deepseek-ai/DeepInfer
git fetch --depth 1 origin 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
git checkout FETCH_HEAD
```

## Files mini-DeepInfer mirrors

| Upstream file | What we mirror |
|---|---|
| `deepinfer/engine/engine_type.py` | `EngineType` (NORMAL / PROMPT / COMPLETION) |
| `deepinfer/engine/llm_engine.py` | `LLMEngine.add_request`, `step`, `pre_sampler_sync`, `post_process`, `flush_inflight_reqs`, `do_copy_state`, and the virtual hooks |
| `deepinfer/worker/worker.py` | Order of forward → sample → `pre_sampler_sync` → post-process inside one step |
| `csrc/engine/scheduler.hpp` | `Scheduler` queues (`waiting`, `prompted`, `running`), `_schedule_prefill`, `_schedule_decode`, `process_outputs`, `SchedulerOutputs` fields |
| `csrc/engine/sequence.hpp`, `csrc/engine/pybind_sequence.cpp` | `Sequence` fields as seen from Python (`cached_hit_tokens`, `num_cached_tokens`, `state_id`, `swa_valid_start`, `last_copy_state_token_idx`, ...) |
| `csrc/engine/block_manager.hpp` | `allocate_seq` / `try_allocate_seq`: one state slot per sequence plus paged blocks |
| `deepinfer/worker/cache_engine.py` | `KVSTORE_GRAIN`, `StateBlock` (stored vs. dropped region, inference vs. kvstore ring size) |
| `doc/inference_deployment_qa.md` | SWA persistence positions, hit rule, bounded replay |

`docs/interface-contract.md` describes what we mirror and every deviation.
