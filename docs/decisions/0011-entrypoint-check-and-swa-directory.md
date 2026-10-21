# ADR 0011: Entrypoint CHECK, the SWA directory, and SWA pulls

Status: accepted. Its least-loaded routing is replaced by the balancers of
[ADR 0015](0015-load-balancers.md); CHECK, the directory and pulls stay.

## Context

Milestone 4 runs several prefill and decode instances behind one Entrypoint. DeepSeek's design
computes each request's hit length from metadata before routing (the CHECK role), keeps the SWA
tier in the producer's host RAM, and finds it through a directory [DS]. Up to milestone 3 each
prefill instance computed its own hit against its own RAM, so a turn could only use SWA records
held by the instance that served the previous turn. That is prefix affinity, which ADR 0001 rules
out.

## Decision

**The Entrypoint computes the hit.** It runs the unchanged `compute_hit` over a view with presence
checks only (`KVIndex`: `has_block`, `has_swa`):

- `has_block` reads the shared block directory (a `FileKVStore` opened by the Entrypoint, used only
  for presence).
- `has_swa` asks the SWA directory whether any live instance holds the record.

It sends the prefill instance the result as a plan (`cached_hit_tokens`, `num_cached_tokens`,
`swa_valid_start`, number of blocks to load, record position) plus the record's holder. The prefill
instance applies it through `ServingLLMEngine.add_request(..., hit=...)`, the same
`_apply_hit` path as before (contract §10.3).

**The SWA directory is built from instance reports.** After every loop iteration, each instance
compares its SWA tier with what it last reported and sends `swa_index {added, removed}`. The
Entrypoint keeps `(key, position) → {holders}` and drops an instance's entries when it dies or
retires. It prefers the routed prefill instance as holder, then the lowest-named other holder.

**A non-holder pulls the record over the peer link.** The prefill instance sends `swa_pull` to
the holder. The holder replies with the record, using the block file's checksummed encoding
(`encode_tensor`, ADR 0010), or with `found: false`. The puller stores the record in its own SWA
tier, which then shows up in its next report, and applies the plan. If the record is missing,
damaged, or the holder dies before replying, the puller falls back to computing the hit from its
own store. That backs off one block or more: shorter, never wrong.

**Routing: least-loaded.** Prefill and decode are each chosen by the fewest non-terminal requests
that depend on the instance, ties broken by name. The tier-0 balancers replace this later. The
routed decode instance travels in the `add` message, and the prefill instance hands off to it.

## Consequences

- Hit length no longer depends on which prefill instance served earlier turns. In the tests the
  turn after a decode finishes hits the record at L1 that the decode instance stored, pulled across
  processes.
- The directory lags the instances by up to one report. A stale entry costs a pull that comes back
  empty, then a back-off. A missing entry costs a shorter hit. Neither changes output.
- The decode instance stores its final record one engine step after it reports the last token, so a
  turn submitted immediately can miss it. Tests that need the record wait for the directory.
- One Entrypoint holds the whole directory. DeepSeek partitions it by key across Entrypoints [DS];
  that is a later change behind the same messages.
- Reports send the full diff each iteration. That is fine at this scale, and bounded by the SWA
  tier's capacity.
