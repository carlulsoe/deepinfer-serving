# Roadmap

How we get from an empty engine stand-in to the five serving pieces running on real DeepInfer.
This page covers where each kind of claim gets tested and the order of the build. Everything here
is our plan; see [architecture.md](architecture.md) for what is being built.

## Test tiers

We design for production scale and test small. Each tier answers different questions, and a result
from one tier is not evidence for another.

| Tier | Hardware | Runs | Can validate | Cannot validate |
|---|---|---|---|---|
| **0: simulator** | any CPU | load balancers and autoscaler against replayed request traces and a GPU-time model | routing and scaling logic; invariants (nothing lost, bounds and rate limits respected); relative policy quality on the modelled hardware; most of the tuning ("hillclimbing") | that the GPU-time model is right; anything about KV bytes, numerics or engine behaviour |
| **1: RTX 3060 Ti** (SM86, 8 GB) | local desktop | mini-DeepInfer, CPU and CUDA | correctness: PD handoff, transfer, store, Entrypoint, the hit rule, persistence positions, abort and failure handling, drain forwarding, multi-process orchestration on one box | performance of anything; real DeepInfer (it needs SM100 or Ascend 950); FP8/FP4 formats; EP, all-to-all, NVLink |
| **2: rented 8 × B200** (SM100) | one rented node | the real DeepInfer at the pinned commit with our `ServingLLMEngine` subclass | that the subclass works against the real engine; the PD oracle in DeepInfer's deterministic mode; real KV sizes and transfer and swap-in bandwidth; cost-model fitting; simulator calibration | wide-EP decode (EP64+), multi-node behaviour, 3FS at scale, production traffic |

**Performance tuned on the 3060 Ti does not transfer.** Do not tune chunk sizes, batch sizes,
transfer granularity, buffer counts or timeouts on tier 1 and carry the numbers forward. Three
reasons:

- mini-DeepInfer is token-serial fp32 PyTorch, chosen for exact determinism, not speed
  ([interface-contract.md §8](interface-contract.md)).
- The card has none of SM100's kernels or KV formats.
- It is on one PCIe host with no RDMA.

Tier 1 numbers are useful only for spotting gross regressions in our own code paths.

**The simulator's GPU-time model starts uncalibrated.** Until tier 2 exists, it uses placeholders
derived from DeepSeek-reported figures and the DeepInfer README's Pareto charts, which are [DS] and
unverified. Policy comparisons on tier 0 are relative to that model. They become meaningful in
absolute terms only after the model is fitted to tier-2 profiles (DeepInfer ships
`examples/benchmark.py` for this).

**Tier 2 fit is unverified.** The plan is 1 prefill instance at EP2 plus 1 decode instance on the
remaining GPUs of one 8 × B200 node. DeepSeek says prefill uses "the smallest EP/DP that HBM
allows" within EP2–8 [DS], and DeepInfer's examples include `--ep 2` [code]. We have not
confirmed two things: that V4.1-Flash weights plus useful KV fit in two B200s, and which EP size
the decode side can use on the remaining six GPUs (expert counts are rounded up to a multiple of
`ep_size`). Check both before renting. Tier 2 is paid and needs explicit approval for each rental.

## Build order

The data-plane track (milestones 1–5) runs in order; each milestone's oracle must pass before the
next starts. The control-plane track (simulator, load balancers, autoscaler) runs in parallel on
tier 0 and joins the data plane at milestone 4.

### Milestone 1: mini-DeepInfer and the oracles (done)

Build the stand-in engine that mirrors DeepInfer's interface and cache layout, plus in-process
seams: `Handoff`, the in-memory `KVStore`, the hash chain, the hit rule, and the persistence hooks
in `ServingLLMEngine`.

**Done when:**

- Every oracle in [interface-contract.md §12](interface-contract.md) passes: PD, cache hit (record
  at L1, at L1 − G, none), hit-rule units, persistence positions, and the negative corruption tests.
  They pass with the small config and with the upstream defaults (128 / 128 / 6 / 128).
- The PD and cache-hit oracles also pass under `pytest -m gpu` on the 3060 Ti.
- `ruff`, `pyright` and `pytest` are clean.
- The interface contract loses its "draft" status, with ambiguities A1–A7 either confirmed or
  rewritten.

**Status:** done (2026-10-20). All oracles pass on CPU and on the 3060 Ti; the contract is approved,
with ambiguities A1–A7 recorded as the readings we chose.

### Milestone 2: cross-process handoff (done)

PROMPT and COMPLETION engines run in separate processes, and handoffs travel over a real transport
on one host. Introduce the request state machine and abort propagation.

**Done when:**

- The PD oracle passes across processes, through the transport rather than a function call.
- Abort at every stage (queued, prefilling, in transit, decoding) terminates the request exactly
  once, and block and state-slot counts return to baseline.
- Corrupting bytes in transit is detected by the receiver or changes the output. It must never
  pass silently.

**Status:** done (2026-10-21). Decisions in ADRs 0005–0009. Tests: `test_oracle_pd_multiprocess.py`
(small and default configs, CPU and CUDA), `test_pd_lifecycle.py` (abort at each stage, random
aborts, corruption of header and payload), `test_pd_backpressure.py` (a full decode instance nacks and
the prefill instance retries), `test_wire.py`, and a CPU → CUDA smoke test (ADR 0009). The transport
is a `multiprocessing` socket pair carrying raw frames; no pickle on any link.

### Milestone 3: the store, with a local-NVMe backend

A persistent block tier on local NVMe and an SWA tier in host RAM, behind the existing `KVStore`
protocol, with LRU eviction under a watermark. 3FS is an optional backend behind the same protocol,
not a dependency.

**Done when:**

- A shared conformance suite passes for every backend: in-memory, NVMe, and 3FS if present.
- The cache-hit oracle passes with the NVMe backend, including across a restart of every engine
  process.
- Eviction and lost blocks make hits shorter, never outputs wrong. This is checked by deleting
  blocks and SWA records between turns and re-running the oracle.

### Milestone 4: Entrypoint and index

The Entrypoint computes hit length from metadata (CHECK role), owns the SWA directory, and routes
requests. It first uses trivial balancers, then the tier-0 ones. Run at least two prefill and two
decode engines on one box.

**Done when:**

- Multi-turn conversations through the Entrypoint match cold runs token for token.
- The hit length for a turn is the same whichever prefill engine served earlier turns. This is the
  no-affinity property.
- A cross-engine SWA hit is served by a pull through the directory.
- Killing an engine mid-request re-routes its requests; outputs still match the oracle and
  nothing leaks.
- A decode drain forwards in-flight requests and KV, and the forwarded requests match
  uninterrupted runs.

### Milestone 5: the real DeepInfer subclass on SM100

The same `ServingLLMEngine` logic runs as a subclass of DeepInfer's `LLMEngine` at the pinned
commit, on a rented 8 × B200 node (tier 2).

**Done when:**

- The PD oracle passes token for token with DeepInfer in deterministic mode
  (`DEEPINFER_BATCH_INVARIANT=1`, `DEEPINFER_ENABLE_MTP_VARLEN=0`, fixed `seq_seed`).
- The cache-hit oracle passes exactly when SWA state is present at L1.
- The bounded-replay paths (SWA missing, and decoder-side replay) are approximate upstream, so
  they are checked with eval scores against a cold baseline, within a tolerance fixed before the
  run.
- Measured KV bytes per block, transfer bandwidth and swap-in overlap are recorded and fed back
  into the simulator's model.
- Every deviation from upstream found on the way is recorded in `interface-contract.md`.

### Control-plane track (parallel, tier 0)

**S1: simulator.** Trace replay, a GPU-time model per engine role parameterized by `cost(q, c)` for
prefill and by batch size and total KV for decode, plus instance failure and start-up times.
*Done when* it reproduces simple analytical cases (single queue, uniform load) and reads a trace
format we can also generate from tier-1 runs.

**S2: load balancers.** Prefill LB (four levels), Decode LB, Vision LB as pure policy functions.
*Done when* the invariants hold on all traces, and on the reference traces each balancer beats
round-robin and a prefix-affinity baseline on the balance metric and on TTFT and TBT percentiles.
Results are reported as a ratio to a perfectly balanced bound.

**S3: autoscaler.** `reconcile()` with scaling signals, rate limits, bounds, warm reserve, the
0.8 no-scale-down rule, drain and replace. *Done when* the invariants hold under bursty and failure
traces, and SLO attainment against GPU-hours is reported against a static allocation.

The balancers and the autoscaler move onto real engines at milestone 4 (one box, tier 1) and
milestone 5 (tier 2). They stay correctness-only on tier 1.

## Not in scope yet

CI, containers, Kubernetes, cloud provisioning, and anything paid. Tier-2 rentals are the one
planned paid step, and each needs explicit approval. The deployment spec in
[architecture.md](architecture.md#deployment-spec) is designed so a Kubernetes operator can realise
it later. We are not building that operator now.
