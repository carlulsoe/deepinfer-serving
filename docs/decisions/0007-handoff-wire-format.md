# ADR 0007: Handoff wire format

Status: accepted

## Context

Milestone 1 serialized `Handoff` with `torch.save` / `torch.load(weights_only=True)`. That was
fine in-process. Across processes, and later across machines, two things matter:

- **Untrusted input.** The receiver must not execute or deserialize objects from the wire. Even
  `weights_only` pickling is a large attack surface, and its rules change between torch versions.
- **Corruption.** A flipped byte in KV produces fluent, wrong text ([ADR 0004](0004-determinism-oracles.md)).
  The milestone 2 oracle requires that corruption in transit is detected or changes the output,
  never passes silently.

RDMA will eventually move the tensor payload separately from the metadata, so the format should
already keep them apart.

## Decision

A handoff on the wire is:

```
magic "DISH" | version u16 | header_len u32 | sha256(header) [32 B] | header (UTF-8 JSON) | payload
```

- The header holds the scalar fields (`request_id`, `user_id`, `sampling_params`, `prompt_len`,
  `token_ids`, `num_cached_tokens`, `config_fingerprint`), a tensor table
  (`name, dtype, shape, offset, nbytes`), and `sha256(payload)`.
- The payload is the raw little-endian bytes of `block_kv` then `swa_state`.
- Decoding uses `json` and `numpy.frombuffer` only. It checks the magic, version, lengths, both
  digests, the dtype allow-list (`float32`), and that the table exactly tiles the payload. Any
  failure raises `WireError`.
- The receiver then checks `config_fingerprint` and validates shapes against its own config
  (`receive_handoff`) before touching device memory.
- The transport envelope carries `request_id` outside the checksummed blob, so a receiver can name
  the request in its `nack` even when the blob is unreadable.

## Consequences

- No pickle on the data path. `Handoff.to_bytes` / `from_bytes` keep their names and use this
  format.
- SHA-256 over every handoff costs CPU time. That is negligible at our scale. At production scale
  the payload digest can move to the transport (RDMA with ICRC) or to a cheaper CRC, behind the same
  version field.
- A later RDMA transport sends the header as a message and the payload by one-sided reads, using the
  same tensor table.
