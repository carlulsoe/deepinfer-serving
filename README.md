# deepinfer-serving

The serving pieces that DeepSeek left out of [DeepInfer](https://github.com/deepseek-ai/DeepInfer):
KV-cache storage, prefill/decode (PD) disaggregation, load balancing and autoscaling. They are
designed for DeepSeek's production scale and tested small, against mini-DeepInfer
(`src/deepinfer_serving/engine/`), a stand-in engine that mirrors DeepInfer's interface and cache
layout.

## What DeepInfer provides and what it omits

DeepInfer (Apache-2.0, released 2026-10-20) is DeepSeek's in-house engine for
DeepSeek-V4.1-Flash on NVIDIA SM100 and Huawei Ascend 950. It ships the engine: kernels, attention
DP / balanced CP / MoE EP with EPLB, DSpark speculative decoding, SWA bounded replay, asynchronous
scheduling, batch-invariant inference and structured output. Its `LLM` class runs offline
inference only.

It omits what turns an engine into a service. Its README lists a KV cache system, VE/prefill/decode
disaggregation and load balancing as "not included". The engine exposes no-op hooks
(`init_pd_split`, `send_processed_prompts`, `pe_do_save_kvcache`, ...) and an
`EngineType` of NORMAL / PROMPT / COMPLETION for that missing code to fill. Details and sources:
[docs/background/deepseek-serving-system.md](docs/background/deepseek-serving-system.md).

## What we build

1. **KV transfer and storage client**: prefill→decode handoff, store writes, hit loads.
2. **KV index, Entrypoint and store**: hash-chain prefix match, the hit rule, the SWA-state directory.
3. **PD orchestration**: an `LLMEngine` subclass implementing the upstream hooks, plus request
   lifecycle, drain and failover.
4. **Load balancers**: prefill, decode and vision encoder. No prefix affinity.
5. **Autoscaler**: a reconcile loop over a declarative deployment spec.

The central design choice, copied from DeepSeek: a shared KV cache that every node reaches at equal
cost, so routing never needs prefix affinity
([why](docs/background/shared-kv-cache.md)).

## Status

**Milestones 1–4 (done):** mini-DeepInfer and the determinism oracles; cross-process handoff
with the request state machine and abort propagation; the KV store with a shared local-NVMe block
tier, LRU eviction, and cache hits that survive engine restarts; the Entrypoint with metadata-only
hit computation, an SWA directory with cross-engine pulls, re-routing after instance loss, and
decode drain. Next is milestone 5, the real DeepInfer subclass on rented SM100 hardware, which needs
approval for each rental; the simulator, balancers and autoscaler can proceed on CPU meanwhile. See
[docs/roadmap.md](docs/roadmap.md).

## Docs

- [docs/background/deepseek-serving-system.md](docs/background/deepseek-serving-system.md): how DeepSeek's production system works, from public sources
- [docs/background/shared-kv-cache.md](docs/background/shared-kv-cache.md): the shared-cache idea, its bandwidth cost, and DualPath
- [docs/architecture.md](docs/architecture.md): the five pieces and the deployment spec
- [docs/roadmap.md](docs/roadmap.md): test tiers and build order
- [docs/interface-contract.md](docs/interface-contract.md): the mini-DeepInfer interface and deviations from upstream
- [docs/decisions/](docs/decisions/): architecture decision records
- [docs/sources.md](docs/sources.md): sources and how far to trust each
- [reference/DEEPINFER_PIN.md](reference/DEEPINFER_PIN.md): the pinned DeepInfer commit

Commands and rules for contributors: [AGENTS.md](AGENTS.md).
