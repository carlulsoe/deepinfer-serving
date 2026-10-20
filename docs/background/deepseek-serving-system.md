# DeepSeek's V4.1-Flash serving system

This page explains how DeepSeek serves DeepSeek-V4.1-Flash in production, as far as public sources
describe it. It is background for this repo, not a description of what we build; for that, see
[architecture.md](../architecture.md).

Almost everything here comes from DeepSeek's own documents. Claims are tagged:
**[code]** checked in DeepInfer source at the pinned commit, **[DS]** DeepSeek's statement (not
independently verified), **[3P]** third-party reporting, **[ours]** our inference or arithmetic.
See [sources.md](../sources.md) for the sources and the tag definitions.

Short names used below:

- **Q&A**: [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md) at the pinned commit.
- **Zhihu**: DeepSeek's post ["DeepSeek V4.1 Flash 大规模推理部署实践"](https://zhuanlan.zhihu.com/p/2095560295930508953), the Chinese original of the Q&A.
- **README**: [DeepInfer's README](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md) at the pinned commit.

## What was released and what was not

DeepInfer is DeepSeek's in-house inference framework for DeepSeek-V4.1-Flash, released under
Apache-2.0. The pinned commit is dated 2026-10-20 [code]. The README credits vLLM as the main base,
with ideas from SGLang and TensorRT-LLM [DS].

| Area | What the release contains |
|---|---|
| Hardware | NVIDIA SM100 (CUDA 13) and Huawei Ascend 950 (CANN 9.2); Linux, GCC ≥ 13, Python 3.12 [DS, README] |
| Default checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` [DS, README] |
| Parallelism | attention data parallelism (DP), balanced context parallelism (CP), MoE expert parallelism (EP) with EPLB in `deepinfer/model_executor/eplb/` [code] |
| Speculative decoding | DSpark with confidence-scheduled, variable-length verification, up to 5 draft tokens (`DEEPINFER_USE_NEXTN_DECODE`, default 5) [DS, README] |
| Other engine features | SWA bounded replay; asynchronous "zero-overhead" scheduling; batch-invariant inference and deterministic sampling (NVIDIA only); structured output (regex, JSON schema, EBNF) [DS, README] |
| Kernels | git submodules under `vendor/`: FlashMLA, DeepGEMM, DeepGEMM-Ascend, DeepEP, DeepEP-Ascend, DeepSelect [code]. Also TileKernels and in-tree TileLang ops [DS, README]. FlashMLA dropped Hopper in its 2026-09-30 release and now needs SM100/SM103; DeepGEMM needs SM90 or SM100; DeepEP needs Hopper or newer [DS, kernel READMEs] |
| Entry point | `deepinfer.entrypoints.LLM`, which constructs the engine with `offline_inference=True`. There is no online server [code] |

The README's list of what is "required for a complete inference service (not included in this
release)" has three items: a KV cache system, VE/prefill/decode disaggregation ("most required
engine functionality is already implemented"), and load balancing across DP replicas for prefill
and decode [DS]. The Q&A says the KV-cache management and load-balancing components are "tightly
coupled to our internal infrastructure" and so not yet fully open-sourced [DS]. Autoscaling and an
online server are not in the release either [code: absent], though DeepSeek does not list them as
withheld.

The disaggregation support that *is* shipped consists of these engine extension points [code]:

- `EngineType` in `deepinfer/engine/engine_type.py`: `NORMAL` (prompts and completions), `PROMPT`
  (prefill only), `COMPLETION` (decode only).
- `LLMEngine` in `deepinfer/engine/llm_engine.py` has a block of methods marked "virtual
  functions" (虚函数), with no-op or minimal bodies: `init_pd_split`, `finalize_seq` (default:
  free the sequence), `post_schedule`, `should_copy_state` (default: `False`),
  `do_save_kvcache`, `pe_prepare_save_kvcache`, `pe_do_save_kvcache`, `process_output`,
  `send_processed_prompts`, `post_process_model_outputs`. The withheld production code fills them in.
- `Sequence.cached_hit_tokens` is exposed to Python and settable (`csrc/engine/pybind_sequence.cpp`).
  The scheduler includes a `FlopsEstimator` (`csrc/engine/flops_estimator.hpp`).
- `deepinfer/worker/cache_engine.py` defines `KVSTORE_GRAIN = 128` (SWA state is persisted only at
  multiples of 128 tokens) and `StateBlock`, the per-sequence state slot layout.

How mini-DeepInfer mirrors these is in [interface-contract.md](../interface-contract.md).

DeepSeek also released [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) (MIT, Rust
with Python bindings). It converts Chat Completions, Responses and Messages API requests to tokens
and parses output back [DS]. It splits prompts at special-token boundaries it inserted itself and
encodes the text segments in parallel. Special-token strings in user text are encoded as ordinary
tokens, which blocks control-token injection. Streaming output is parsed by a state machine with
incremental KMP matching, and tool-call arguments stream as JSON fragments [DS, Q&A].

## Design goals

DeepSeek states three goals [DS, Q&A]:

1. Run on accelerators from several vendors.
2. Run prefill on ordinary 8-GPU machines and decode on ordinary 64–128-GPU supernodes.
3. Keep KV cache for more than 72 hours.

## Request flow: E/P/D disaggregation

[![DeepSeek V4.1 Flash inference system architecture](https://raw.githubusercontent.com/deepseek-ai/DeepInfer/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/images/architecture_overview.png)](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/images/architecture_overview.png)

*Figure from the DeepInfer repo at the pinned commit (`doc/images/architecture_overview.png`), linked rather than copied. [DS]*

The system disaggregates vision encoding (E), prefill (P) and decode (D), and adds a dedicated
storage cluster [DS, Q&A]. Following the diagram:

1. **deepseek-recipe** turns the user's JSON request into a token list and sampling parameters.
2. The **Entrypoint** receives the tokens. It computes the prefix-cache hit length (see
   [the hit path](#the-hit-path)) and hands the request to the Prefill LB.
3. The **Prefill LB** routes the request to a prefill service instance.
4. **Prefill** runs the uncached part of the prompt. For images it calls the **Vision Encoder LB**,
   which picks a vision-encoder GPU that returns embeddings. Prefill reads hit KV from 3FS.
5. Prefill sends the request to the **Decode LB** and the KV directly to the chosen **decode**
   instance (blue arrow in the diagram).
6. Decode generates tokens and returns them to the Entrypoint, which hands them to deepseek-recipe
   to build the JSON response.
7. Decode writes the KV it produces to 3FS, so the next turn of the conversation can hit it. The
   Zhihu post says block KV is written by decode instances, so both prefill's and decode's output
   are reusable [DS, Zhihu].

How KV moves from P to D is not described beyond "block KV and SWA KV share the same P-to-D
transfer path" [DS, Q&A]. DualPath, which DeepSeek co-authored, moves KV between engines by RDMA
on a low-priority traffic class [DS, DualPath]. That V4.1 production does the same is our
inference [ours].

**Cross-datacenter.** Because V4.1 compresses KV heavily, P and D can sit in different data
centers if there is about 400 Gbps of interconnect per 8 GPUs. DeepSeek runs its inference nodes in
different but adjacent data centers. The cost is reliability: inter-DC fiber gets cut by
construction work [DS, Q&A].

## Parallelism

| Role | Configuration | Rule |
|---|---|---|
| Prefill | EP2–8 / DP2–8 | the smallest EP that fits in HBM |
| Decode | EP64–128 / DP64–128 ("Wide-EP") | the largest EP the available GPUs allow |
| Vision encoder | one GPU per instance | use the cheapest GPU available |

All [DS, Q&A]. Attention runs data-parallel with #DP = #EP, as in V3/R1. Tensor parallelism and
context parallelism are not used in production "for now", although the engine supports CP [DS, Q&A; code].
The README's offline benchmarks used 4 SM100 GPUs for prefill and 64 for decode, each within one
scale-up domain [DS, README]. The Q&A says Wide-EP cuts per-step MoE memory traffic and, with
good EPLB, raises throughput; it shows an EP8 vs. EP64 benchmark chart [DS].

**Deployment units.** 1P + 1D works. In practice the P:D GPU ratio is about 1:1, and since decode
should run at EP64 or above, the smallest *efficient* unit is 64 prefill GPUs plus 64 decode GPUs
[DS, Q&A].

**Earlier generation, for comparison.** V3/R1 (Feb 2025, H800) ran prefill at EP32 across 4 nodes
and decode at EP144 across 18 nodes [DS, V3/R1 overview]. V4.1 shrinks prefill to a single
8-GPU machine and keeps decode inside one supernode.

## KV cache: tiers, blocks and 3FS

V4.1 keeps two kinds of KV. **Block KV** is paged in 128-token blocks. **SWA KV** is the
sliding-window state, one fixed-size slot per sequence. The two are stored in different places
[DS, Q&A and Zhihu]:

| Content | Medium | Replication | Managed by |
|---|---|---|---|
| In-flight block KV + state KV | GPU HBM | — | prefill / decode instances |
| SWA KV | host RAM of the machine that produced it | single copy; nothing on 3FS | prefill / decode instances; cross-machine hits pulled over a 400G VLAN |
| Paged block KV | dedicated 3FS storage cluster | none at application level; relies on 3FS erasure coding | `kvcache-store` |

### Block KV

| Parameter | Value | Tag |
|---|---|---|
| Page size | 128 tokens; prefill also uses paged KV | [DS] |
| Contents | main KV + indexer K for **4 layers** (Zhihu: "根据模型架构设计，存储 4 层") | [DS, Zhihu] |
| Block size on disk | about 112 KiB | [DS, Zhihu] |
| Implied bytes per token | 112 KiB / 128 ≈ 896 B, consistent with the V4.1 report's 890 B/token global KV footprint | [ours] |
| KV formats | main KV per-16 e4m3 (like NVFP4 without global scale); indexer K MXFP4 | [DS, Q&A] |

Why only 4 layers: V4.1's CED architecture reuses KV across layers, and decoder-layer block KV is
projected from the last encoder layer's hidden state [DS, V4.1 report abstract; Q&A]. We have not
read the tech report beyond the abstract and do not explain the mechanism further.

Decode does not offload KV and does not use Hi-Sparse, because decode is not capacity-bound. It
still writes its KV to disk to raise the hit rate [DS, Q&A].

### Why a dedicated storage cluster

DeepSeek gives three reasons [DS, Q&A]:

1. Every prefill machine can hit any prefix. Routing for cache affinity and routing for load
   balance therefore never conflict.
2. Storage and compute scale independently.
3. The cache survives compute failures. Compute nodes fail more than an order of magnitude more
   often than storage nodes.

This is the design idea we copy. [shared-kv-cache.md](shared-kv-cache.md) explains it in depth.

### 3FS

| Parameter | Value | Source |
|---|---|---|
| Capacity for KV | about 100 PiB SSD | [DS, Q&A, Zhihu] |
| Redundancy | 10+2 erasure coding; each EC group tolerates 2 simultaneous node failures | [DS, Q&A] |
| Read | requests go to all 11 storage nodes holding the group's data; any 10 responses decode | [DS, Q&A] |
| Peak aggregate read / write | 10 TiB/s / about 850 GiB/s; tuned for random I/O at block size (~112 KiB) | [DS, Zhihu] |
| Eviction | one GC worker per storage node evicts LRU to keep disk use under a watermark. `kvcache-store` has no expiry logic of its own | [DS, Q&A] |
| Effective TTL | 3 days for agent traffic, 7 days for chatbot traffic. A ~3-day TTL costs chatbot traffic 0.5–1% of hit rate, because chat has a heavy long tail of prefixes read many times | [DS, Q&A] |
| Logical cache per online GPU | about 14.4 TB; physical capacity needed is larger | [DS, Q&A] |
| Recommended storage server | 20 × 30.73 TB SSD; ≥ 40 GB/s read and ≥ 5 GB/s write per server | [DS, Q&A] |

**Inconsistency in the Q&A.** The storage-server answer says to provide "enough total capacity to
retain 24 hours of data". The design goal is more than 72 hours, the stated TTLs are 3 and 7
days, and the production table reports 72 hours of retention after a session goes inactive. We
cannot tell which figure the 24 hours refers to. One possibility is the minimum for a new
deployment rather than DeepSeek's own sizing [ours, unresolved].

**A rough consistency check [ours].** 100 PiB divided by 614.6 TB of raw SSD per recommended
server gives about 180 servers. At the recommended 5 GB/s write each, that is about 900 GB/s
(≈ 840 GiB/s), close to the reported 850 GiB/s peak write. At 40 GB/s read each it is about
7.3 TB/s, the same order as the 10 TiB/s peak read. These figures hang together, but this is our
arithmetic on DeepSeek's numbers, not a reported cluster size.

### The hit path

Only the first of three stages is on the critical path [DS, Q&A and Zhihu]:

1. **Match.** The Entrypoint does a hash-chain prefix match against a local `kvcache-store`
   running in the CHECK role. It looks at metadata only. It cannot be hidden, because the hit
   length must be known before routing (the Prefill LB's cost model uses it).
2. **Fetch.** A separate process copies the KV asynchronously from 3FS into host shared memory.
3. **Swap-in.** A thread in the engine process copies shared memory into HBM layer by layer,
   overlapped with the forward pass. The CUDA copy engine handles the many fine-grained block
   copies poorly, so swap-in uses GPUDirect RDMA: the RDMA NIC saturates PCIe without using SMs.

DualPath reports the same mechanism with numbers. A `cudaMemcpyAsync` submission costs about
5–7 µs and an RDMA write work request about 1 µs. Routing all GPU traffic through the compute NIC
also lets network QoS keep KV traffic from disturbing EP all-to-all [DS, DualPath §5].

### SWA state

| Parameter | Value | Tag |
|---|---|---|
| Slot | one `StateBlock` per sequence: `[stored region: SWA][dropped region: YOCO-consumer SWA layers + ca_state/ica_state compression accumulators]` | [DS, Q&A]; `StateBlock` in `cache_engine.py` [code] |
| Persisted layers | the first 20 | [DS] |
| Ring | 134 tokens = 128 window + 6 speculative backtrack | [DS] |
| Size per sequence | 20 × 134 × 528 B = 1,415,040 B (MXFP8) | [DS, Q&A] |
| Where | host RAM of the producing machine, never 3FS | [DS] |
| Directory | in memory; logically one, physically partitioned by key across the Entrypoint machines; a machine without the state pulls it from the holder | [DS, Q&A] |
| Lifetime / hit rate | TTL about 180 s; hit rate above 98% | [DS, Zhihu only] |

Why it is cheap to keep SWA state only briefly: it is reused only within an active session, on a
scale of minutes. Block KV is reused over days. A miss is recovered by bounded replay (below), not
by reading disk [DS, Zhihu].

**Persistence positions** [DS, Q&A]. SWA state is stored only at multiples of 128 tokens
(`KVSTORE_GRAIN = 128` [code]).

- *After prefill*: reserve one token at the end and round down, so the position is
  ⌊(L − 1) / 128⌋ × 128. If that is less than one full 128-token unit, nothing is stored. Only the
  last chunk of a chunked prefill stores.
- *During decode*: a replica region in device memory receives an on-device copy of the window each
  time the request crosses a 128 boundary. When the request finishes, the most recent copy is
  stored, attached at the boundary rounded down. A copy can overshoot the boundary by at most
  5 tokens; the 6 extra ring positions absorb it.

### The hit rule

Block KV decides the hit length. SWA state decides only whether the last block must be recomputed
[DS, Q&A]:

1. For an input of length L, the hit is at most **L − 134** (YOCO truncation requires the last
   128 + 6 tokens to be computed).
2. Under that cap, query block KV: hit length **L1**, a multiple of 128.
3. If SWA state exists anywhere in **[L1 − 128, L1]**, the hit is **L1**. Otherwise it is
   **L1 − 128**, and the last block is replayed.

mini-DeepInfer implements this rule; the exact reading we chose for its ambiguous corners is in
[interface-contract.md §10.3 and §11](../interface-contract.md).

### SWA bounded replay

Bounded replay recomputes only the last 128 tokens and truncates attention to the replayed
segment. Replaying from position s, the query at i attends only to keys in
[max(s, i − 127), i]. The result is an **approximate** SWA state, at a constant cost independent
of context length and layer count [DS, Q&A]. It is used in two places:

- **Encoder side.** When SWA state is missing, the last 128 tokens of the hit prefix are
  recomputed together with the uncached suffix. The replay rebuilds only SWA state; block KV is
  reused as-is, neither recomputed nor overwritten [DS].
- **Decoder side.** Prefill runs only the encoder layers in full. Under CED, decoder-layer block
  KV is projected from the last encoder layer's hidden state, so only decoder SWA state needs
  per-layer computation. Rebuilding it exactly would take the last 2,560 tokens through 20 decoder
  layers. Instead, each prefill recomputes only the last 128 prompt tokens in the decoder layers.
  The result is used for decode only and never enters the prefix cache. DeepSeek says this nearly
  halves total prefill compute [DS]. The README configures it as
  `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` with `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`,
  "the retained suffix must cover the SWA window plus the maximum backtracking length" [DS, README].
  That is 140 kept tokens against the Q&A's "128"; we have not checked how the two relate in code.

**History** [DS, Zhihu]. V4 persisted SWA state to disk and recovered misses by exact
recomputation. SWA state took nearly half of persistent KV capacity without a matching hit-rate
benefit. The V4 report's "Zero SWA Caching" (no SWA persistence, exact recompute only) was too
expensive and was not used in production. Bounded replay is the main change in V4.1's KV
management. The V4.1 report says it brings persistent KV to about 1/8 of V4-Flash's [DS].

## The four load balancers

DeepSeek has four balancers. The Prefill, Decode and Vision Encoder LBs balance attention load
across DP ranks by routing. The Expert-Parallel LB balances the MoE layers [DS, Zhihu].

### Prefill LB

Every level uses one quantity, **compute quota**: the predicted attention time from a pre-fitted
quadratic cost model `cost(q, c)` [DS]. The Zhihu post defines q as the sequence's uncached tokens.
The Q&A defines q as the number of tokens of the sequence in this step, which at the per-GPU level
is the chunk. In both, c is the prefix length already computed.

| Level | Mechanism [DS, Q&A and Zhihu] |
|---|---|
| 1. Bucketing | By total context length: < 16K / 16–64K / 64–256K / ≥ 256K. Each round, engines are reassigned among buckets in proportion to each bucket's share of last round's compute-quota demand. Each bucket has its own per-batch quota cap (Q&A only). |
| 2. Across DP groups | Engines *pull* work according to remaining quota. There is no prefix affinity, because any engine can hit any prefix. |
| 3. Within a DP group | Each arriving request goes to the DP rank with the lowest current quota load. |
| 4. Per GPU | Requests dequeue FIFO. Chunk size is solved, not fixed: each step has a time budget (`compute_quota_limit`, ns), and the scheduler solves the quadratic for the largest q that fits the remaining budget, capped at 32,768 tokens. The budget is reduced with an all-reduce MIN across the DP group so ranks finish attention together (cap and MIN: Q&A only). |

The engine already contains a `FlopsEstimator` in the scheduler [code]. We have not checked
whether it is the cost model the LB uses.

### Decode LB

The Decode LB routes uniformly across all GPUs of all decode instances. It balances two things at
once: request count, which sets the per-step batch size, and total KV length, which sets attention
cost. Short requests go to GPUs with low concurrency but high total KV; long requests go to GPUs
with low total KV [DS, Q&A]. The Q&A points to DualPath's intra-group scheduler for details. That
scheduler computes a high-token threshold Z = 1.05 × (pending + assigned tokens) / #engines. It
prefers engines that stay under Z, picking the one with the fewest requests, and otherwise picks
the one with the fewest tokens. It skips any engine without enough free HBM [DS, DualPath §6.1].

### Vision Encoder LB

A prefill machine picks a vision-encoder machine (one encoder per GPU) at random. On that machine,
the image goes to the GPU with the fewest image patches in flight, where an image's patch count is
(width / patch_size) × (height / patch_size) [DS, Q&A].

### Expert-Parallel LB (EPLB)

Each GPU holds a few redundant expert slots for replicas of hot experts. Per-expert load is
collected over the last N steps. Every M steps experts are reassigned to GPUs, and weights move
within the group over NVLink `symm_mem` [DS, Q&A]. The policy and weight shuffle are in
`deepinfer/model_executor/eplb/` (`policy.py`, `shuffle.py`; `symm_mem` imported in `__init__.py`) [code].

| | Redundant experts per GPU | Window N | Reassign every M | Measured imbalance `moe_lb` (slowest rank / mean) |
|---|---|---|---|---|
| Prefill | 4, rounded up so the total is a multiple of `ep_size` | 100 steps | 1,000 steps | **Zhihu:** 1.0320 (EP2/DP2), 1.0618 (EP4/DP4). **Q&A:** 1.045 (EP2DP2), 1.074 (EP4DP4) |
| Decode | 1, rounded up; at EP64 this gives 7 experts per GPU, excluding the shared expert | 1,000 steps | 10,000 steps | 1.090 (both sources) |

The two sources disagree on the prefill figures. They agree that EP4 is less balanced than EP2. We
do not know which is current; the Q&A is the newer, maintained document [DS, conflict unresolved].

**Overall** [DS, Q&A]. Load balancing reaches about 80% of a perfectly balanced offline benchmark.

## Autoscaling and fault tolerance

All [DS, Q&A and Zhihu].

- **Signals.** Prefill scales on uncached tokens. Decode scales on running requests. Adjustments are
  rate-limited and bounded above and below. Zhihu explains why: with PD disaggregation the right
  P:D ratio moves with traffic, and a static ratio causes queueing on one side and idling on the
  other.
- **Spare capacity.** Ready machines sit unoccupied in the pool as scale-out inventory, and more
  instances stay online than demand requires: scale-in is forbidden while quota utilization is
  above 0.8.
- **Draining.** A decode instance that gets a shutdown notice has a 1-minute grace period. Requests
  still running after that are forwarded, with their KV, to other decode instances in one shot.
  The point is to avoid holding a large-EP instance for a few long-tail requests.
- **Failures.** Restarts take minutes. Fault tolerance comes from quota redundancy plus
  request-level re-routing, not from faster restarts. In wide-EP, all GPUs of an instance are one
  communication domain, so one GPU failure takes down the instance. On restart, the platform sets
  the EP size to the number of healthy GPUs the instance actually gets.

## Other engine techniques (brief)

- **Asynchronous scheduling.** While the GPU runs forward(N) and sample(N), the CPU runs
  schedule(N+1), launch_forward(N+1), post-process(N−1) and launch_sample(N+1), so the GPU never
  idles. This follows SGLang and NanoFlow [DS]. `EngineType.default_zero_overhead_scheduling` is
  true for `NORMAL` and `COMPLETION` and false for `PROMPT` [code].
- **DSpark verification.** The draft model proposes 5 tokens. A confidence head predicts
  per-token acceptance, and the scheduler verifies the prefix subset that maximizes expected
  throughput [DS]. The reported gain over fixed-5 verification differs by source: **Zhihu**
  says +6% in production; the **Q&A** says +4–7% at the production operating point (~60
  sequences per rank), and its table runs from −1.5% at 16 to +11.0% at 96 sequences per rank.
- **Structured output.** Grammars compile asynchronously on dedicated machines, and the request
  fetches the compiled automaton before it enters the engine [DS].
- **Determinism.** Batch-invariant mode is `DEEPINFER_BATCH_INVARIANT=1`. Deterministic sampling
  additionally needs `DEEPINFER_ENABLE_MTP_VARLEN=0` and a fixed `seq_seed` in `SamplingParams`.
  Both are NVIDIA only [DS, README]. Our SM100 oracles depend on this (see
  [ADR 0004](../decisions/0004-determinism-oracles.md)).

## Production numbers

All figures in this section are **DeepSeek-reported and unverified**. The production table appears
only as an image in the Zhihu post. Its caption says machines keep some redundancy for traffic
swings.

| Metric | Value |
|---|---|
| Cluster throughput | 155 trillion tokens/day, single cluster [Q&A, Zhihu] |
| Revenue | US$38 per GPU-hour, average, at off-peak ("valley") pricing [Q&A, Zhihu] |
| P:D ratio | close to 1:1 [Zhihu table note] |
| Prefill throughput, uncached tokens only | 60K tok/s per GPU |
| Prefill throughput, including cache hits | 1.5M tok/s per GPU |
| Average input cache hit rate | 96% |
| TTFT | 400 ms average, 1,000 ms p99 |
| Average input length | 77K tokens |
| Decode throughput | 14K tok/s per GPU |
| Per-user decode speed | 230 tok/s (300 tok/s for coding) |
| KV retention after a session goes inactive | 72 h |
| SWA state cache | ~180 s lifetime, > 98% hit rate |
| Load balancing vs. ideal | ~80% of a perfectly balanced offline benchmark |

The prefill figures are consistent with each other: 60K / 1.5M = 4% uncached, matching 96% hit
[ours].

**For comparison, V3/R1 in February 2025** [DS, V3/R1 overview]: an average of 226.75 nodes of
8 × H800 (peak 278) over 24 hours; 608B input tokens (56.3% on-disk cache hits) and 168B output
tokens, about 776B in total; about 73.7K input tok/s per node in prefill and 14.8K output tok/s per
node in decode. DeepSeek computed a 545% cost-profit margin on the assumption that every token was
billed at R1 prices. It was explicit that actual revenue is lower.
