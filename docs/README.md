# docs

Three kinds of document, kept separate:

**How DeepSeek does it** (explanation of public sources; claims tagged [code] / [DS] / [3P] / [ours])

- [background/deepseek-serving-system.md](background/deepseek-serving-system.md): V4.1-Flash's production serving architecture
- [background/shared-kv-cache.md](background/shared-kv-cache.md): why a shared, equal-cost KV cache removes prefix-affinity routing, and what it costs

**What we build and why**

- [architecture.md](architecture.md): the five pieces, their hooks, interfaces and oracles; the deployment spec
- [interface-contract.md](interface-contract.md): the mini-DeepInfer interface and every deviation from upstream
- [decisions/](decisions/): ADRs
  - [0001](decisions/0001-shared-kv-cache-no-prefix-affinity.md) shared KV cache, no prefix affinity
  - [0002](decisions/0002-mini-deepinfer-not-vllm.md) mini-DeepInfer instead of vLLM
  - [0003](decisions/0003-mirror-not-vendor-deepinfer.md) mirror DeepInfer at a pinned commit; do not vendor it
  - [0004](decisions/0004-determinism-oracles.md) determinism oracles as the definition of done
  - [0005](decisions/0005-handoff-ownership-and-admission.md) handoff ownership, release, and decode admission
  - [0006](decisions/0006-request-state-machine-and-abort.md) request state machine, token order, abort propagation
  - [0007](decisions/0007-handoff-wire-format.md) pickle-free, checksummed handoff wire format
  - [0008](decisions/0008-store-across-processes.md) the store across processes: shared filesystem, producer-local SWA
  - [0009](decisions/0009-cross-device-handoff.md) cross-device handoffs allowed, not oracle-checked
  - [0010](decisions/0010-store-backends-and-eviction.md) store backends, on-disk format, LRU eviction under a watermark
  - [0011](decisions/0011-entrypoint-check-and-swa-directory.md) Entrypoint CHECK, SWA directory, SWA pulls, least-loaded routing
  - [0012](decisions/0012-attempts-reroute-and-abort.md) attempts, re-routing after instance loss, Entrypoint-owned aborts
  - [0013](decisions/0013-decode-drain.md) decode drain by forwarding sequences
  - [0014](decisions/0014-simulator-and-cost-model.md) the tier-0 simulator, its GPU-time model, and the trace format
  - [0015](decisions/0015-load-balancers.md) load balancers as pure policy functions, in the simulator and the Entrypoint
  - [0016](decisions/0016-autoscaler.md) the autoscaler as a pure reconcile function

**How we get there**

- [roadmap.md](roadmap.md): test tiers and milestones

[sources.md](sources.md) lists every source and how reliable it is.
