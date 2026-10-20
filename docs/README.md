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

**How we get there**

- [roadmap.md](roadmap.md): test tiers and milestones

[sources.md](sources.md) lists every source and how reliable it is.
