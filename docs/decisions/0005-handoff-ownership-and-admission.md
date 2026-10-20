# ADR 0005: Handoff ownership, release, and decode admission

Status: accepted

## Context

In milestone 1 the PROMPT engine frees a sequence's blocks and state slot as soon as the in-process
sink returns. Across processes, the handoff may be rejected or lost, and the decode engine may be
full. We need to say who owns the KV at each moment, when the prefill side may free it, and what a
full decode engine does.

Upstream defers releases through pins and `DeferredRelease` (`llm_engine.py`). The withheld PD code
presumably keeps prefill blocks pinned until the transfer completes, because RDMA reads them in
place.

## Decision

**Ownership follows bytes.**

- In milestone 2 the handoff is serialized into an owned host buffer when it is built
  (`send_processed_prompts`). The PROMPT engine frees device blocks and the state slot right after
  serialization. The buffer then holds the only copy of the request's KV.
- The prefill instance keeps the buffer in an **outbox** until the decode instance answers. On
  `ack`, the buffer is dropped. On `nack`, what happens depends on the reason.
- When a zero-copy transport arrives (RDMA reading device memory), the rule becomes "free on ack":
  blocks stay pinned until the ack, mirroring upstream's pin/unpin. The outbox protocol stays the
  same; only the release point moves.

**Admission is decided on receipt, never queued.**

- The decode instance allocates blocks and a slot when it receives a handoff
  (`receive_handoff` → `try_allocate_seq`). If it cannot, it answers `nack(no_capacity)` and keeps
  nothing.
- On `nack(no_capacity)`, the prefill instance keeps the buffer and resends after a short backoff.
  At milestone 4 the Decode LB can pick another decode instance instead. Nothing is queued on the
  decode side, so a full decode instance never holds memory for requests it cannot run.
- On `nack(invalid)` (failed checksum or validation; [ADR 0007](0007-handoff-wire-format.md)),
  the request fails. We do not retry a corrupt handoff: the buffer itself is the suspect.

## Consequences

- A handed-off request costs host memory, not device memory, while it waits. The outbox is not
  bounded in milestone 2; a bound and the policy for hitting it come with the Decode LB.
- Backpressure is visible: retries and nacks can be counted per instance, and the autoscaler can
  use them.
- Free-on-serialize is correct only because the bytes are an owned copy. Any transport that reads
  device memory must switch to free-on-ack, and its tests must show that blocks are not reused
  before the ack.
