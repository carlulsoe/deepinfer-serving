# ADR 0002: mini-DeepInfer stand-in instead of vLLM for local development

Status: accepted

## Context

The serving pieces must be developed and tested locally on an RTX 3060 Ti (SM86, 8 GB). DeepInfer
itself cannot run there: it needs NVIDIA SM100 or Huawei Ascend 950, and FlashMLA's current release
supports only SM100/SM103 [DS, README; FlashMLA README]. DeepSeek-V4.1-Flash also does not fit.

The obvious local engine is vLLM. It runs on consumer GPUs, and DeepInfer was originally built
mostly on it [DS]. But the serving code we write must plug into DeepInfer's `LLMEngine` hooks,
`EngineType`, `Sequence` fields and cache layout: 128-token paged blocks plus a per-sequence SWA
state slot persisted at 128-token grains [code]. vLLM's engine, KV connector API and cache layout
differ. Code developed against vLLM would be tested against an interface it will never ship
against, and the port to DeepInfer would be the first time the real seams were exercised.

## Decision

We write mini-DeepInfer (`src/deepinfer_serving/engine/`): a tiny fp32 transformer that exposes
DeepInfer's interface at the pinned commit (`EngineType`, the hooks in upstream call order, the
`Sequence` fields, the scheduler queues) and its cache shape (paged block KV, an SWA ring slot, a
replica region). The model is built for exact determinism: token-serial forward, and K/V projected
from the embedding stream. It does not aim at realism or speed. The full specification and every
deviation are in [interface-contract.md](../interface-contract.md).

## Consequences

- Serving code is written once against DeepInfer's interface. At milestone 5 the same
  `ServingLLMEngine` logic subclasses real DeepInfer.
- Oracles are exact on CPU and on the 3060 Ti, so correctness bugs show up as token mismatches,
  not as small drifts in eval scores.
- mini-DeepInfer says nothing about performance; tier-1 numbers do not transfer
  ([roadmap](../roadmap.md#test-tiers)).
- We maintain a second engine. Its fidelity to upstream is only as good as the contract, so every
  deviation must be written down there, and an upstream change can require a contract change.
- Features mini-DeepInfer leaves out (MTP, zero-overhead scheduling, forks, compressed attention,
  multimodal) are first exercised on tier 2.
