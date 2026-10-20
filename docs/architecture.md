# Architecture

This is the system we are building: the serving pieces DeepInfer's open release leaves out. It is
designed for DeepSeek's production scale (EP64+ decode, a shared 3FS cache) and tested small.

Everything on this page is **our design** unless marked otherwise. Facts about DeepSeek's system
are in [background/deepseek-serving-system.md](background/deepseek-serving-system.md). The
mini-DeepInfer engine interface and the milestone-1 seams are specified exactly in
[interface-contract.md](interface-contract.md); this page does not repeat that detail.

## Shape

We copy DeepSeek's topology (E/P/D disaggregation plus a dedicated KV store) and its central
decision: a shared KV cache with no prefix-affinity routing
([ADR 0001](decisions/0001-shared-kv-cache-no-prefix-affinity.md),
[background](background/shared-kv-cache.md)).

```
 tokens ──► Entrypoint ──► Prefill LB ──► prefill engine (PROMPT) ──handoff──► decode engine (COMPLETION) ──► Entrypoint
               │  ▲                          │   ▲        │                        │    │
               │  └─ hit length (CHECK)      │   │        └─► Vision LB ─► VE      │    └─ writes block KV + SWA state
               ▼                             │   └─ reads hit block KV             │
          KV index + SWA directory           ▼                                     ▼
                                   ┌──────────────────── KV store ─────────────────────┐
                                   │ block tier: shared, equal cost (local NVMe │ 3FS) │
                                   │ SWA tier: host RAM of the producer, via directory │
                                   └───────────────────────────────────────────────────┘
 Autoscaler: reconciles a deployment spec against telemetry; scales, drains, replaces instances.
```

Serving code reaches the engine only through three seams: the `LLMEngine` hooks, the handoff
object, and the KV store interface. It never reaches into engine internals where a hook exists.
The serving subclass of `LLMEngine` is our reconstruction of the code DeepSeek withheld. Hook
names, order and arguments are fixed by upstream at the pinned commit (see
[interface-contract.md §4](interface-contract.md)).

## The five pieces

Each piece lists its responsibility, the upstream hooks it uses, its main interfaces, and its
**oracle**: the test that defines "correct" for it
([ADR 0004](decisions/0004-determinism-oracles.md)).

### 1. KV transfer and storage client

**Responsibility.** Move KV bytes without changing them:

- prefill → decode handoff of block KV and SWA state;
- engine → store writes after prefill, at decode snapshots, and at finish;
- store → engine loads on a hit (fetch to host memory, then swap-in to device);
- decode → decode forwarding of in-flight requests when an instance drains.

This is the hardest piece. A bug here does not crash anything; it produces fluent, wrong text.

**Upstream hooks.**

| Hook | Use |
|---|---|
| `pe_prepare_save_kvcache`, `pe_do_save_kvcache` | choose and write blocks and the SWA record after a sequence's last prefill chunk |
| `should_copy_state` → `do_copy_state` | snapshot the SWA window into the replica region at each 128-token crossing during decode |
| `do_save_kvcache` | write new blocks and the last snapshot when a decode sequence finishes |
| `send_processed_prompts` | build and send handoffs for prefilled sequences, then free them |
| `add_request(enqueue=False)`, `try_allocate_seq`, `add_prompted` | ingest a handoff or a cache hit on the receiving engine (`add_prompted` is our one scheduler addition; contract §11 A4) |

**Interfaces.**

- `Handoff`: a frozen record with identity, sampling params, tokens, `num_cached_tokens`, block KV,
  SWA state, and a config fingerprint; `to_bytes()` / `from_bytes()` (contract §10.1). Transports
  change only how the bytes move: in-process, then cross-process, later RDMA.
- `KVStore` protocol: `put_block` / `get_block` keyed by hash-chain `BlockKey`, `put_swa` /
  `get_swa` keyed by block key plus a grain-aligned position (contract §10.2). Backends: in-memory
  (tests), local NVMe (development and one-node deployments), 3FS (optional, production).
- Later: an async fetcher (store → host buffer) and a layer-wise swap-in thread, kept behind the
  same protocol. The layout is block-granular so a DualPath-style second load path can be added
  later without changing callers.

**Oracle.**

- *PD*: NORMAL output equals PROMPT → `Handoff` bytes → COMPLETION output, token for token.
- *Cache hit*: a turn served from the store equals the same turn run cold.
- *Negative*: corrupting one stored block or the handoff's SWA state changes the output. This
  proves the cache is actually used.
- *Drain* (later): a request forwarded mid-decode finishes with the same tokens as an
  uninterrupted run.

### 2. KV index, Entrypoint and store service

**Responsibility.**

- Compute the hit length for each request before routing, from metadata only (DeepSeek's CHECK
  role).
- Maintain the SWA-state directory: which machine holds which state, partitioned by key across
  Entrypoints.
- Run the store service and its eviction. Retention is enforced by the backend (LRU under a
  watermark), not by TTL logic in the index, as in DeepSeek's `kvcache-store`.

**Upstream hooks.** None directly. It hands the engine a `Hit` that the serving subclass applies
through `add_request(enqueue=False)` → `try_allocate_seq` → load → set `num_cached_tokens`,
`cached_hit_tokens`, `swa_valid_start` → `scheduler.add_sequence` (contract §10.3).

**Interfaces.**

- `block_hashes(token_ids, block_size) -> list[BlockKey]`: SHA-256 hash chain, one key per full
  block, each identifying its whole prefix.
- `match(store, token_ids, cfg) -> Hit`: the hit rule. Cap at L − (window + backtrack); find
  the longest contiguous block prefix L1; use SWA state at L1 or L1 − 128; otherwise back off one
  block.
- SWA directory: `locate(key, position) -> holder`, `register(...)`, `evict(...)`. Not yet
  specified.
- Entrypoint request API: tokens + sampling params in, token stream out. Not yet specified.
  deepseek-recipe sits in front of it and is not ours to build.

**Oracle.**

- Hit-rule unit tests: the cap, grain alignment, SWA present or absent, non-contiguous blocks,
  prompts shorter than a block.
- The cache-hit oracle run through the Entrypoint rather than a direct store call.
- Failure cases: a lost block or a missing SWA state must make the hit shorter, never make the
  output wrong.

### 3. PD orchestration

**Responsibility.** Run a request's lifecycle across vision encoder, prefill and decode instances:

- register and route requests;
- pair a prefill result with a decode instance and deliver the handoff;
- propagate aborts;
- free resources on every path;
- drain a decode instance by forwarding its requests and KV;
- re-route requests when an instance fails.

**Upstream hooks.**

| Hook | Use |
|---|---|
| `init_pd_split` | allocate the replica region (`copy_state`) and the transport endpoints on non-NORMAL engines |
| `post_schedule` | per-step bookkeeping after scheduling (in-flight tracking, telemetry) |
| `finalize_seq` | release a finished sequence and trigger its final save (COMPLETION / NORMAL) |
| `process_output`, `post_process_model_outputs` | stream tokens out and emit per-step telemetry |
| `send_processed_prompts` | hand prefilled sequences to decode (shared with piece 1) |

**Interfaces.** A `ServingLLMEngine(LLMEngine)` subclass, the same subclass on mini-DeepInfer and
on real DeepInfer. An instance controller that owns one engine (or one EP group) and talks to the
Entrypoint and the load balancers. The request state machine (`queued → prefilling → handed_off →
decoding → finished | aborted | forwarded`) is to be specified at milestone 2.

**Oracle.**

- The PD oracle across processes (milestone 2).
- Abort and failure tests: every request terminates exactly once; no block or state slot leaks
  (allocator counts return to baseline).
- The drain oracle above.

### 4. Load balancers

**Responsibility.** Route requests so DP ranks finish their attention work together. We build the
Prefill LB, the Decode LB and the Vision Encoder LB. EPLB already ships inside DeepInfer
(`deepinfer/model_executor/eplb/`) and stays upstream's.

- **Prefill LB.** Four levels following DeepSeek: length buckets resized from last round's demand,
  pull-based assignment across DP groups by remaining quota, least-loaded rank within a group, and
  a per-GPU chunk size solved from a fitted cost model `cost(q, c)`, capped at 32,768, with an
  all-reduce MIN of the budget across the DP group. No prefix affinity.
- **Decode LB.** Balance request count and total KV length across all decode GPUs. Start from
  DualPath's intra-group rule.
- **Vision Encoder LB.** Pick a random machine, then the GPU with the fewest patches in flight.

The balancers are easy to write and hard to make good. They need telemetry, a cost model fitted
per hardware, and tuning against traces. Off-the-shelf routers (Dynamo, llm-d, the SGLang router)
route for prefix affinity, the opposite of this design, so we do not reuse them.

**Upstream hooks.** The per-GPU chunk budget lands in the engine's scheduler. How it plugs in
(through `post_schedule`, a scheduler config field, or `chunk_checkpoints`) is open until we check
the upstream scheduler's quota path. Telemetry comes from `post_process_model_outputs`. The
balancers themselves sit outside the engine.

**Interfaces.**

- Pure policy functions over a telemetry snapshot, for example
  `route_prefill(requests, engine_states, cost_model) -> assignments`, so the same code runs in the
  simulator and in the live system.
- A `CostModel` with `cost(q, c) -> ns`, fitted from profiling runs.

**Oracle.** Not determinism; routing is performance.

- Invariants: no request is lost or duplicated, quotas are respected, and routing never changes
  outputs. The last is checked by running the PD and cache-hit oracles with the balancer in the
  path.
- Quality: on replayed traces in the simulator, the ratio of achieved throughput to a perfectly
  balanced bound (DeepSeek reports about 80% [DS]). Also max/mean attention time per DP group,
  and TTFT and TBT percentiles.

### 5. Autoscaler

**Responsibility.** A reconcile loop. Read the deployment spec and telemetry, compute the desired
instance counts, and act:

- scale prefill on uncached tokens and decode on running requests, rate-limited and bounded;
- keep warm spare nodes, and never scale down while quota utilization is above 0.8;
- drain decode instances with a grace period, then forward what remains;
- replace a failed instance, re-forming EP at the number of healthy GPUs.

**Upstream hooks.** None. It drives instances through the orchestration layer (piece 3).

**Interfaces.**

- `reconcile(spec, observed_state, telemetry) -> list[Action]`, a pure function. Actions include
  scale up, scale down, drain and replace. The same code runs against the simulator and, later, a
  real cluster backend (a Kubernetes operator using LeaderWorkerSet-style multi-node instances).
  The cluster backend is out of scope for now.

**Oracle.**

- Invariants on simulator traces: rate limits and bounds hold; no scale-down above 0.8
  utilization; every draining request finishes or is forwarded; failure re-routing loses nothing.
- Quality: SLO attainment (TTFT, TBT) against GPU-hours on replayed traces.

## Deployment spec

A declarative spec describes a deployment; the autoscaler and a future cluster backend realise it.
This is a **sketch**. Field names and units will change once a controller consumes them.

```yaml
deployment: v41-flash
model: deepseek-ai/DeepSeek-V4.1-Flash-1017
prefill:   { node: hgx-8gpu,  ep: 4,  scale_on: uncached_tokens }
decode:    { domain: nvl72,   ep: 64, scale_on: running_requests, drain_grace: 60s }
vision:    { gpu: cheapest,   per_instance: 1 }
kvstore:   { backend: 3fs, ec: "10+2", ttl: { agent: 3d, chat: 7d }, gb_per_gpu: 14400 }
network:   { pd_gbps_per_8gpu: 400, swa_vlan_gbps: 400 }
reserve:   { warm_nodes: 2, no_scale_down_above: 0.8 }
```

The values mirror DeepSeek's reported settings. `prefill.ep: 4` is one point in their EP2–8 range.
`ttl` is a target for sizing the store; eviction itself is the backend's LRU, not a TTL timer.

### Constraints the spec must express

A controller should reject a spec that violates these, before it schedules anything.

| Constraint | How the spec expresses it | Check |
|---|---|---|
| A decode instance fits in one scale-up (NVLink) domain | `decode.domain` names the domain type, which carries a GPU count; `decode.ep` is the instance size | `decode.ep ≤ gpus(domain)`. Wide-EP all-to-all must not cross the scale-out network. |
| Prefill fits one ordinary node | `prefill.node` carries GPUs per node and HBM | `prefill.ep ≤ gpus(node)`, and the model's weights at that EP fit in HBM |
| An instance is the unit of failure | instances are counted and scheduled whole; `reserve.warm_nodes` is spare capacity | one GPU failure takes the whole instance down. On replacement, EP is re-formed at the healthy GPU count, which must still satisfy the rows above. |
| Network bandwidth budgets | `network.pd_gbps_per_8gpu`, `network.swa_vlan_gbps` | derived P→D traffic (prefill tok/s × persisted KV bytes/token, plus SWA state per request) ≤ budget; the same for SWA pulls |
| Storage sizing from traffic | `kvstore.gb_per_gpu` or the inputs to derive it | capacity ≥ written tokens/day × bytes/token × retention × EC overhead (12/10 for 10+2). Read bandwidth ≥ hit tok/s × bytes/token. Report both against `gb_per_gpu`. |

The derived checks need per-model constants (persisted KV bytes per token, SWA state size, weight
size) and traffic assumptions (tokens/day, hit rate, input length). Those belong in a model
profile and a traffic profile next to the spec, not in it.

### One-node profile

The same controllers must run on one machine in development. P and D share the box, the store is
local NVMe, and EP is small:

```yaml
deployment: dev-one-node
model: mini-deepinfer            # or the real checkpoint on an 8×B200 node (tier 2)
prefill:   { node: local, ep: 1, scale_on: uncached_tokens }
decode:    { domain: local, ep: 1, scale_on: running_requests, drain_grace: 5s }
vision:    { disabled: true }
kvstore:   { backend: local-nvme, path: /var/tmp/deepinfer-kv, ec: none, capacity_gb: 50 }
network:   { pd_gbps_per_8gpu: null, swa_vlan_gbps: null }   # same host; budgets not enforced
reserve:   { warm_nodes: 0, no_scale_down_above: 0.8 }
```

`domain: local` and `node: local` resolve to "this machine". The constraint checks still run; they
are just trivially satisfied. Scaling in this profile means starting and stopping engine processes
on the same box, which is enough to exercise drain, replace and re-route.

## Module layout

Milestone 1 fixes the first modules (contract §1):

```
src/deepinfer_serving/
  engine/            mini-DeepInfer (mirrors upstream; no serving imports)
  kvstore.py         KV store protocol, in-memory store, hash chain, hit rule
  handoff.py         PROMPT → COMPLETION handoff
  serving_engine.py  LLMEngine subclass implementing the hooks
```

Later pieces get their own modules as they arrive: transport, Entrypoint and index, load
balancers, simulator, autoscaler. We decide their names when we write them, not here.
