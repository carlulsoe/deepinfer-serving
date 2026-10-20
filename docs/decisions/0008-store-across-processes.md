# ADR 0008: Where the KV store lives once engines are separate processes

Status: accepted

## Context

`InMemoryKVStore` lives inside one process. Milestone 2 splits PROMPT and COMPLETION engines into
processes, so they no longer share it. DeepSeek's design has two tiers: the block tier is shared
storage (3FS) that any machine can hit at equal cost, and the SWA tier stays in the producer's host
RAM behind an in-memory directory.

## Decision

- **Milestone 2 runs the instances without a store.** Its oracles (cross-process PD, abort, wire
  corruption) do not need one. Cache hits stay covered by the in-process milestone 1 oracles.
- **Milestone 3 shares the block tier through the filesystem, not a store server.** A local-NVMe
  backend writes one file per block key under a common directory, with write-to-temp-then-rename for
  atomicity. Every process on the host opens the same directory. This mirrors 3FS's role (shared,
  equal-cost, no owner process), and a 3FS backend implements the same protocol.
- **The SWA tier stays producer-local.** Each instance keeps its SWA records in its own host RAM. At
  milestone 4 the Entrypoint's directory records the holder, and a non-holder pulls the record over
  the same transport as handoffs. Until then an SWA record can only be hit by the instance that
  produced it.

## Consequences

- No store server process to run or keep available. Eviction is per backend (LRU under a
  watermark), as already planned.
- Until milestone 4, a cross-instance hit finds blocks but no SWA record, so the hit rule backs off
  one block. That is correct, just slower, and the cache-hit oracle covers it.
- The `KVStore` protocol does not change. Its block and SWA methods may route to different backends.
