# ADR 0004: Determinism oracles are the definition of done for the data plane

Status: accepted

## Context

KV transfer and storage fail quietly. An off-by-one block, a stale SWA ring position, or a
mismatched hash produces fluent but wrong text, not a crash. Helper-level unit tests (the hash is
stable, a tensor round-trips) cannot catch errors in how the pieces compose with the engine's
scheduler and cache. Eval scores catch them only statistically and late.

The data-plane pieces have a property we can test exactly. Disaggregation and caching must not
change what the model computes. A request run normally, run split across a PROMPT and a COMPLETION
engine, or run with a cache hit must give the same tokens, provided the forward pass is
deterministic.

## Decision

A data-plane piece (transfer, store, Entrypoint and index, PD orchestration) is done only when its
oracles pass:

- **PD oracle**: NORMAL output equals PROMPT → handoff → COMPLETION output, token for token.
- **Cache-hit oracle**: output with a cache hit equals a cold run, token for token.
- **Negative tests**: corrupting a stored block or the handoff's SWA state changes the output. This
  shows the cache is load-bearing and the oracle is not passing vacuously.

These run on mini-DeepInfer (CPU and the 3060 Ti), where they are exact by construction
([interface-contract.md §8, §12](../interface-contract.md)).

On real DeepInfer (tier 2), they run in DeepInfer's batch-invariant, deterministic mode:
`DEEPINFER_BATCH_INVARIANT=1`, `DEEPINFER_ENABLE_MTP_VARLEN=0`, and a fixed `seq_seed`, which is
NVIDIA only [DS, README]. Upstream's SWA bounded replay is approximate by design [DS, Q&A]. Cache
hits that go through replay (SWA state missing, and decoder-side replay) are therefore checked with
eval scores against a cold baseline, within a tolerance fixed before the run, not token equality.

Load balancers and the autoscaler are not judged by these oracles. They are judged by invariants
and by simulator metrics ([architecture.md](../architecture.md)). The PD and cache-hit oracles must
still pass with them in the request path.

## Consequences

- A correctness claim for serving behaviour must cite an oracle test (`tests/test_oracle_*.py`),
  not a helper test.
- mini-DeepInfer pays for exactness with a slow, token-serial forward pass
  ([ADR 0002](0002-mini-deepinfer-not-vllm.md)).
- On SM100 the oracles run in a mode that disables variable-length speculative verification. That
  mode is not the production fast path, so production-mode correctness relies on the oracle
  results plus eval scores.
- When an oracle cannot be exact (bounded replay), the docs and tests must say so explicitly
  rather than loosening equality silently.
