# deepinfer-serving

Serving pieces that DeepSeek left out of DeepInfer: KV-cache storage, prefill/decode (PD)
disaggregation, load balancing, autoscaling. Everything is developed and tested against
mini-DeepInfer (`src/deepinfer_serving/engine/`), a small stand-in engine that mirrors DeepInfer's
interface and cache layout.

## Commands

```sh
uv sync                                  # install
uv run ruff check . && uv run ruff format --check .   # lint + format check
uv run pyright                           # typecheck
uv run pytest                            # tests (GPU variants skip without CUDA)
uv run pytest -m gpu                     # GPU variants only
```

## Rules

- mini-DeepInfer mirrors DeepInfer at the commit pinned in `reference/DEEPINFER_PIN.md`. Any
  deviation from upstream semantics must be documented in `docs/interface-contract.md`.
  Do not change mirrored names, hook call order, or field meanings without updating the contract.
- Correctness claims require the oracle tests (`tests/test_oracle_*.py`): token-for-token equality
  between NORMAL, PROMPT → handoff → COMPLETION, and cold vs. cache-hit runs. Helper-level unit
  tests are not evidence that serving behavior is correct.
- Serving code plugs into the engine through the `LLMEngine` hooks, the handoff object, and the
  KV store interface. Do not reach into engine internals from serving code when a hook exists.
- Do not vendor or submodule DeepInfer.
- Cross-process code uses the `spawn` start method (CUDA-safe). A script that starts a `PDCoordinator`
  must be a file with an `if __name__ == "__main__":` guard, not stdin or `python -c`.
- No pickle on any link between processes: control messages are JSON, handoffs use `wire.py`.
