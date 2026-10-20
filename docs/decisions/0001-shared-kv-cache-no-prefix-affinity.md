# ADR 0001: Shared KV cache with no prefix-affinity routing

Status: accepted

## Context

Multi-turn and agentic traffic resends almost the whole conversation every turn. DeepSeek reports
a 96% input cache hit rate in production [DS]. The common way to exploit this is to keep KV near
the GPUs that produced it and route each request to where its prefix lives. That is
prefix-affinity routing, used by the SGLang router, llm-d and NVIDIA Dynamo. It ties cache hit
rate to routing, and it fights load balancing, scaling and failure recovery.

DeepSeek's V4.1 system goes the other way. Block KV lives in a dedicated 3FS cluster that every
prefill machine reads at equal cost. The Entrypoint computes hit length from metadata before
routing, and the load balancers do not route for affinity [DS, Q&A]. Only the short-lived SWA
state stays local, behind a directory, and a miss costs a bounded one-block replay. The cost of
this design is storage bandwidth: every hit is a remote read, every turn. It is affordable for
V4.1 because its KV is heavily compressed (about 112 KiB per 128-token block [DS]). Full reasoning
and numbers: [background/shared-kv-cache.md](../background/shared-kv-cache.md).

## Decision

We copy DeepSeek's design:

- The block-KV store is shared and location-independent. Every prefill engine can load any stored
  prefix at the same cost.
- Hit length is computed before routing, from metadata only, and it enters the load balancers only
  as an input to the cost model (`c` in `cost(q, c)`). No balancer considers where a prefix is
  stored.
- SWA state is a separate tier: kept in the producer's host RAM, located through a directory,
  pulled on a remote hit. A miss falls back to the hit rule's one-block back-off.
- Retention is a storage policy (LRU under a watermark in the backend), independent of compute
  topology.

## Consequences

- Load balancers and the autoscaler optimize compute alone. Scaling or losing prefill nodes does
  not lose cache.
- We cannot reuse off-the-shelf affinity routers. We build our own balancers (roadmap S2).
- Storage and transfer must be designed for bandwidth from the start: block-granular I/O,
  asynchronous fetch, layer-wise swap-in overlapped with compute. The design must leave room for
  a DualPath-style second load path. The deployment spec has to state network and storage budgets
  ([architecture.md](../architecture.md#constraints-the-spec-must-express)).
- On models with large KV per token this design may not fit a NIC budget. We accept that: we
  target V4.1-class KV sizes.
- The no-affinity property is testable. At milestone 4, a turn's hit length must not depend on
  which prefill engine served earlier turns.
