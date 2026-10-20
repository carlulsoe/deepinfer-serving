# A shared KV cache instead of prefix-affinity routing

This page explains the one design idea from DeepSeek's serving system that shapes this repo most:
a prefix cache that every prefill machine reaches **at the same cost**, so routing never has to
care where a prefix is cached. It covers why that removes prefix-affinity routing, what it costs
in storage bandwidth, and how DeepSeek's DualPath work attacks that cost.

Tags as in [sources.md](../sources.md): **[code]**, **[DS]** (DeepSeek's statement, unverified),
**[3P]**, **[ours]**. Arithmetic on DeepSeek's numbers is always [ours] and only as good as the
inputs.

## The conflict prefix affinity creates

Multi-turn chat and, more so, agents resend almost the whole conversation every turn. DeepSeek
reports a 96% average input hit rate at an average input of 77K tokens [DS]. DualPath measured
98.7% on agentic coding traces [DS, DualPath §3]. With hit rates like these, the cheapest prefill
is the one that reuses cached KV, and where that KV lives decides how the system is built.

The common design keeps KV close to the GPUs that made it: in HBM, then host DRAM, then local SSD,
sometimes pooled across a few machines. A turn can only hit if it lands on a machine, or pool,
that holds its prefix. The router therefore has to send each request to where its prefix is. That
is **prefix-affinity routing**, and it is what most open routers do: the SGLang router, llm-d,
NVIDIA Dynamo, and Mooncake's scheduler are all cache-aware in this sense [ours; DualPath §1
describes Mooncake as affinity-aware].

Affinity fights the other jobs the router has:

- **Load balance.** A hot conversation or a burst of agents sharing a prefix pins load onto the
  machines holding it. The router chooses between overloading them and recomputing elsewhere,
  which throws away the hit.
- **Scaling.** Adding prefill machines does not help until caches warm on them. Removing machines
  deletes cache.
- **Failures.** A crashed node takes its cache with it, and every session pinned to it pays a cold
  prefill. DeepSeek says compute nodes fail more than an order of magnitude more often than
  storage nodes [DS, Q&A].
- **Coupled metrics.** Hit rate becomes a function of routing quality, so cache policy and routing
  policy have to be tuned together.

## DeepSeek's answer: make every hit cost the same everywhere

V4.1 puts paged block KV in a dedicated 3FS cluster that every prefill machine reads over the
storage network [DS, Q&A]. A prefix stored by any machine is a hit for every machine, at the same
latency and bandwidth cost. DeepSeek states the consequence directly: "a dedicated storage cluster
lets every prefill machine hit any prefix, avoiding a conflict between prefix affinity and the
load-balancing routing policy", and "since any engine can hit any prefix, there is no
prefix-affinity routing" [DS, Q&A].

Three things follow:

1. **Hit length becomes an input, not a routing goal.** The Entrypoint computes the hit length once,
   by a metadata-only hash-chain match against a local `kvcache-store` in the CHECK role, before
   routing [DS]. The Prefill LB then feeds it into its cost model `cost(q, c)`, where c is the
   cached prefix length. The cache tells the router how much work a request is; it never tells the
   router *where* to send it.
2. **The balancers only balance compute.** All four Prefill LB levels (length buckets, pull by
   remaining quota, least-loaded DP rank, budgeted chunk size) optimize predicted attention time.
   None of them looks at cache location [DS]. See
   [deepseek-serving-system.md](deepseek-serving-system.md#prefill-lb).
3. **Cache lifetime is a storage decision.** Retention (3 days for agents, 7 for chat) is set by
   3FS capacity and LRU garbage collection, independent of how many prefill machines are up or
   which ones failed [DS].

The equal-cost property is helped by how 3FS reads: with 10+2 erasure coding, every read goes to
the 11 nodes of an EC group and completes on the first 10 replies [DS]. A block's cost does not
depend on one particular storage node being healthy or idle.

### The one exception: SWA state stays local

V4.1 does *not* put everything in the shared tier. The per-sequence sliding-window (SWA) state,
about 1.4 MB per sequence, stays in host RAM of the machine that produced it. An in-memory
directory, partitioned by key across the Entrypoints, records which machine holds it, and a
machine without the state pulls it over a 400G VLAN [DS, Q&A].

This does not bring affinity back, for two reasons:

- The router still ignores location. A non-local hit is a pull, not a reason to route differently.
- A miss is cheap and bounded. The hit rule backs off one 128-token block and recomputes it with
  SWA bounded replay, at constant cost regardless of context length [DS]. Block KV still decides
  the hit length.

The split follows lifetimes. SWA state is reused within an active session, over minutes; DeepSeek
reports about 180 s lifetime and > 98% hit rate. Block KV is reused over days [DS, Zhihu]. V4
persisted SWA state too, and it took nearly half of persistent capacity for little hit-rate
benefit; V4.1 dropped that [DS, Zhihu].

## What it costs: storage bandwidth

The price of equal-cost hits is that **every hit token's KV crosses the network every turn**.
Local caching pays this only on a miss; a shared tier pays it on every hit. At 96% hit rate the
system reads about 24 tokens of KV for every token it computes (1.44M hit tok/s against 60K
computed tok/s per prefill GPU) [ours, from DS numbers].

The bandwidth a prefill GPU needs from storage is roughly

    read bandwidth ≈ (hit tokens per second) × (persisted KV bytes per token)

DualPath calls the related ratio the *cache-compute ratio*, KV bytes loaded per unit of compute.
It reports about 22 GB/PFLOP for DeepSeek-V3.2 on its agentic traces and calls it "a significant
bottleneck on storage bandwidth" [DS, DualPath §3]. It also notes that GPU FLOPS have grown much
faster than NIC bandwidth and HBM capacity from Ampere to Blackwell, so the ratio gets worse on
newer hardware [DS, DualPath Fig. 3].

### Worked numbers for V4.1 [ours]

Inputs, all [DS]: 1.5M tok/s per prefill GPU including hits, 60K uncached, 77K average input,
~112 KiB per 128-token block (≈ 896 B/token), 400 Gbps of P↔D bandwidth per 8 GPUs, 1.4 MB SWA
state per sequence.

| Flow | Per prefill GPU | Per 8-GPU prefill node |
|---|---|---|
| Hit block KV read from 3FS: 1.44M tok/s × 896 B | ≈ 1.3 GB/s | ≈ 10 GB/s ≈ 83 Gbps |
| Block KV sent to decode (whole context): 1.5M tok/s × 896 B | ≈ 1.3 GB/s | ≈ 11 GB/s ≈ 86 Gbps |
| SWA state sent to decode: 1.5M / 77K ≈ 19.5 req/s × 1.4 MB | ≈ 28 MB/s | ≈ 0.2 GB/s |

Against a 400 Gbps budget per 8 GPUs, each flow uses about a fifth. For contrast, take an
MLA-era model such as V3 at FP8: about 35 KB/token (61 layers × 576 B). The same hit rate
and token rate would need about 50 GB/s per GPU from storage, roughly 40× more. No ordinary
storage NIC provides that.

So the shared-cache design and **KV compression** depend on each other. The V4.1 tech report puts
the global KV footprint at 890 B/token and the persistent footprint at about 1/8 of V4-Flash's
[DS]. That is what keeps "read every hit from remote storage" inside a NIC budget.

These numbers are averages, and they skip several things:

- **Bursts.** Peak aggregate 3FS read is 10 TiB/s [DS]. We do not know the GPU count, so we cannot
  compare it to the per-GPU demand.
- **Latency.** Swap-in must overlap the forward pass layer by layer, or a hit adds latency instead
  of saving it.
- **Write traffic.** Decode writes its KV back to 3FS [DS]. Reported peak aggregate write is about
  850 GiB/s [DS].

### What DeepSeek does to make the reads cheap

From the Q&A and Zhihu post [DS]:

- **Dedicated storage cluster**, tuned for random I/O at the block size (~112 KiB), about 100 PiB
  of SSD, 10 TiB/s peak read.
- **Off the critical path.** Only the metadata match blocks routing. Fetch from 3FS into host
  shared memory runs in a separate process. Swap-in to HBM runs in an engine thread, per layer,
  overlapped with compute.
- **GPUDirect RDMA for swap-in**, not the CUDA copy engine, which handles many small block copies
  poorly. The NIC saturates PCIe without using SMs.

## How DualPath addresses the bandwidth

[DualPath](https://arxiv.org/abs/2602.21548) is a DeepSeek-co-authored paper on this exact problem.
It used V3.2-era models on Hopper, before V4.1 [DS].

**The problem it names.** In a PD-disaggregated system with external KV storage, only prefill
engines read hit KV from storage. On their nodes the storage NIC saturates, while decode nodes'
storage NICs sit mostly idle. Total storage bandwidth is wasted, and prefill GPUs wait on I/O
[DS, DualPath §1, §3].

**The idea: two load paths.**

- *PE read path* (the usual one): storage → prefill host DRAM → prefill HBM, layer by layer; then
  the full prompt KV goes to the decode engine's buffer.
- *DE read path* (new): storage → **decode** host DRAM. During prefill, each layer's hit KV moves
  from the decode buffer to prefill HBM over RDMA on the compute network. Only the newly computed
  (miss) KV goes back to decode.

For each request the scheduler reads on whichever side has the shorter disk-read queue. Storage
bandwidth becomes a pooled resource across all nodes, prefill and decode [DS, DualPath §4, §6].

**Why it does not hurt inference.** The extra traffic shares the compute network and PCIe with
latency-critical collectives such as EP all-to-all. DualPath routes *all* GPU data movement,
including local H2D/D2H, through the GPU's paired compute NIC with GPUDirect RDMA. The network's
QoS then separates the classes: on InfiniBand, model traffic gets a high-priority virtual lane with
about 99% of bandwidth, and KV traffic uses a low-priority lane on spare capacity; RoCE would use
traffic classes and DSCP [DS, DualPath §5]. The paper derives a P:D range in which no link
saturates: for 8 GPUs and one storage NIC per node, roughly 1/7 ≤ P/D ≤ 7/2 [DS, DualPath §4.2].

**Scheduling.** Engines are grouped; groups pull work. Prefill assignment skips engines over a
token limit and prefers nodes with short disk-read queues. Decode assignment balances token count
across groups, then within a group prefers engines below a high-token threshold and picks the one
with the fewest requests. That intra-group rule is the one the Q&A points to for the V4.1 Decode
LB. Prefill batches are formed FIFO under a *compute quota* from a fitted attention-time model,
and the last request is chunked to fit. That is the same idea as the V4.1 Prefill LB's per-GPU
level [DS, DualPath §6].

**Reported results** [DS]: up to 1.87× offline throughput and 1.96× average online serving
throughput over their own baseline; storage-NIC imbalance (max/avg) down from 1.53 to 1.18; scaling
to 1,152 GPUs.

**What we do not know.** DeepSeek does not say whether V4.1 production uses the DE read path. The
Q&A cites DualPath only for the decode scheduler. Given the per-node figures above, V4.1's KV
compression may have moved the bottleneck enough that prefill-side reads suffice. That is an
inference, not a statement from DeepSeek [ours].

## What we take from this

This is the design decision recorded in
[ADR 0001](../decisions/0001-shared-kv-cache-no-prefix-affinity.md). In short [ours]:

- Our KV store is shared and location-independent. The Entrypoint computes hit length from
  metadata; the load balancers treat it as an input to a cost model and never route for affinity.
- SWA state is a separate, short-lived, locality-tracked tier behind a directory. A miss falls back
  to the hit rule's one-block back-off, never to routing.
- Transfer and storage are designed for bandwidth from the start: block-granular I/O, layer-wise
  overlap, and a transfer layer that can later carry a DualPath-style second load path and
  traffic-class separation. We do not build dual-path loading until measurements on real hardware
  say we need it.
- The trade-off we accept: we cannot use off-the-shelf affinity routers, and our design relies on a
  storage tier with enough bandwidth.
