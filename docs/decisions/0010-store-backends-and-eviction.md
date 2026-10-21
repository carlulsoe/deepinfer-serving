# ADR 0010: Store backends, on-disk format, and eviction

Status: accepted

## Context

Milestone 3 puts the block tier on local NVMe, shared by every engine process on a host through a
directory ([ADR 0008](0008-store-across-processes.md)), with LRU eviction under a watermark. Several
processes write and evict the same directory with no server to arbitrate. A block that is lost,
evicted or damaged must make a hit shorter, never make an output wrong.

## Decision

**Backends.** Two classes implement the unchanged `KVStore` protocol:

- `InMemoryKVStore`: both tiers in host RAM. Used by in-process tests, and as the SWA tier of every
  other backend.
- `FileKVStore(path, block_capacity_bytes, swa_capacity_bytes)`: block tier as files under `path`,
  SWA tier in an `InMemoryKVStore` in the process's own RAM.
- 3FS is `FileKVStore` pointed at a 3FS mount. This is an inference, not verified here: 3FS ships a
  FUSE client, and the backend uses only open, read, write, rename, utime, unlink and scandir. A
  native USRBIO backend is a performance change for tier 2, behind the same protocol.

**Namespace per config.** Block keys hash tokens only. Each instance opens
`path / namespace(config.fingerprint())`, so engines with different model or cache configs never
read each other's blocks.

**One file per block key, written atomically.** The writer writes a temp file in the same directory
and `os.replace`s it onto the key's name. A reader sees the old file, the new one, or none. The file
carries the key and a SHA-256 over key, header and payload:
`"DISB" | version | key | sha256 | header_len | JSON {dtype, shape} | float32 payload`. No pickle.

**Damage is a miss.** `get_block` returns `None` and deletes the file if it is truncated, fails the
checksum, or holds another key. The engine already serves a request cold when a block or record
that matched is gone at fetch time ([contract §10.3](../interface-contract.md)); milestone 3 extends
that to SWA records. The next prefill of that prefix stores the block again.

**Eviction: LRU under a watermark, per tier, by whoever writes.**

- Each tier has a byte capacity. When a put takes it over capacity, the least recently used entries
  are removed until it is at or below `LOW_WATERMARK` (0.9) of capacity.
- On disk, recency is the file's mtime. `put` and `get` set it from `time.time_ns()`, forced to be
  strictly increasing within a process, because kernel timestamps are too coarse to order
  back-to-back operations.
- Every `put_block` scans the directory, sums file sizes, and evicts. There is no shared counter to
  keep consistent. Two processes evicting at once may both delete; that over-evicts a little and is
  harmless for a cache.
- Temp files older than 10 minutes belong to a dead writer and are removed by the scan.

**No fsync.** It is a cache. After a crash, a damaged file fails its checksum and becomes a miss.

**The SWA tier does not survive a restart.** It is host RAM (ADR 0008). After a restart, hits back
off one block until the prefix is prefilled again. The restart oracle checks exactly this.

## Consequences

- No server, lock file or index to keep consistent. The directory listing is the index.
- `put_block` costs a directory scan, O(files). Fine for development and one-node use. At scale the
  block tier is 3FS, whose own capacity management may replace this scan; decide at tier 2.
- One flat directory per namespace. Sharding into sub-directories is a later change if file counts
  need it.
- LRU across processes is approximate (mtime races, concurrent evictors). Correctness does not
  depend on it: the loss oracle checks that any surviving subset of blocks and records gives the
  cold output.
- Capacity counts file bytes on disk for `FileKVStore` and tensor bytes for `InMemoryKVStore`.
