"""Block tier on a local filesystem (local NVMe, or a 3FS FUSE mount); SWA tier in host RAM (ADR 0008, 0010).

Every process on a host opens the same directory; there is no store server. One file per block key:

``magic "DISB" | version u16 | key 32B | sha256(key | header | payload) 32B | header_len u32 | header JSON | payload``

A file that fails any check is a miss and is deleted, so damage on disk shortens hits instead of changing outputs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import struct
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from deepinfer_serving.kvstore import LOW_WATERMARK, BlockKey, InMemoryKVStore

MAGIC = b"DISB"
VERSION = 1
_PREFIX = struct.Struct("<4sH32s32sI")
_TMP_PREFIX = ".tmp-"
_STALE_TMP_S = 600.0
"""A temp file this old belongs to a writer that died; eviction removes it."""


class FileKVStore:
    def __init__(self, path: str | os.PathLike[str], block_capacity_bytes: int, swa_capacity_bytes: int) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.block_capacity_bytes = block_capacity_bytes
        self.swa_tier = InMemoryKVStore(block_capacity_bytes=0, swa_capacity_bytes=swa_capacity_bytes)
        self._last_ns = 0

    # ---- block tier: files ----

    def _file(self, key: BlockKey) -> Path:
        return self.path / key.hex()

    def _touch(self, path: Path) -> None:
        """LRU order is file mtime. Set it from a fine clock so back-to-back operations stay ordered."""
        self._last_ns = max(time.time_ns(), self._last_ns + 1)
        with contextlib.suppress(FileNotFoundError):  # another process evicted it meanwhile
            os.utime(path, ns=(self._last_ns, self._last_ns))

    def has_block(self, key: BlockKey) -> bool:
        return self._file(key).exists()

    def put_block(self, key: BlockKey, kv: torch.Tensor) -> None:
        data = _encode(key, kv)
        tmp = self.path / f"{_TMP_PREFIX}{os.getpid()}-{uuid.uuid4().hex}"
        tmp.write_bytes(data)
        os.replace(tmp, self._file(key))  # atomic: readers see the old file, the new one, or none
        self._touch(self._file(key))
        self._evict()

    def get_block(self, key: BlockKey) -> torch.Tensor | None:
        path = self._file(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        kv = _decode(key, data)
        if kv is None:
            path.unlink(missing_ok=True)
            return None
        self._touch(path)
        return kv

    def _evict(self) -> None:
        """Delete least recently used files until the tier is under the low watermark. Scans the directory."""
        entries = []
        total = 0
        now = time.time()
        for entry in os.scandir(self.path):
            try:
                st = entry.stat()
            except FileNotFoundError:
                continue
            if entry.name.startswith(_TMP_PREFIX):
                if now - st.st_mtime > _STALE_TMP_S:
                    Path(entry.path).unlink(missing_ok=True)
                continue
            entries.append((st.st_mtime_ns, entry.name, st.st_size))
            total += st.st_size
        if total <= self.block_capacity_bytes:
            return
        entries.sort()
        for _, name, size in entries:
            if total <= LOW_WATERMARK * self.block_capacity_bytes:
                break
            (self.path / name).unlink(missing_ok=True)
            total -= size

    # ---- SWA tier: host RAM of this process ----

    def has_swa(self, key: BlockKey, position: int) -> bool:
        return self.swa_tier.has_swa(key, position)

    def put_swa(self, key: BlockKey, position: int, state: torch.Tensor) -> None:
        self.swa_tier.put_swa(key, position, state)

    def get_swa(self, key: BlockKey, position: int) -> torch.Tensor | None:
        return self.swa_tier.get_swa(key, position)

    # ---- administration (eviction, tests); not part of the KVStore protocol ----

    def block_keys(self) -> list[BlockKey]:
        return [bytes.fromhex(p.name) for p in self.path.iterdir() if not p.name.startswith(_TMP_PREFIX)]

    def discard_block(self, key: BlockKey) -> None:
        self._file(key).unlink(missing_ok=True)

    def swa_entries(self) -> list[tuple[BlockKey, int]]:
        return self.swa_tier.swa_entries()

    def discard_swa(self, key: BlockKey, position: int) -> None:
        self.swa_tier.discard_swa(key, position)

    def swa_positions(self) -> set[int]:
        return self.swa_tier.swa_positions()

    def block_bytes_used(self) -> int:
        return sum(p.stat().st_size for p in self.path.iterdir() if not p.name.startswith(_TMP_PREFIX))

    def swa_bytes_used(self) -> int:
        return self.swa_tier.swa_bytes_used()


def _encode(key: BlockKey, kv: torch.Tensor) -> bytes:
    tensor = kv.detach().cpu().contiguous()
    if tensor.dtype != torch.float32:
        raise ValueError(f"unsupported dtype {tensor.dtype}")
    payload = tensor.numpy().astype("<f4", copy=False).tobytes()
    header = json.dumps({"dtype": "float32", "shape": list(tensor.shape)}, separators=(",", ":")).encode()
    digest = hashlib.sha256(key + header + payload).digest()
    return _PREFIX.pack(MAGIC, VERSION, key, digest, len(header)) + header + payload


def _decode(key: BlockKey, data: bytes) -> torch.Tensor | None:
    """The tensor, or ``None`` if the file is truncated, damaged, or belongs to another key."""
    if len(data) < _PREFIX.size:
        return None
    magic, version, file_key, digest, header_len = _PREFIX.unpack_from(data)
    if magic != MAGIC or version != VERSION or file_key != key:
        return None
    header_end = _PREFIX.size + header_len
    header, payload = data[_PREFIX.size : header_end], data[header_end:]
    if hashlib.sha256(key + header + payload).digest() != digest:
        return None
    meta = json.loads(header)
    shape = meta["shape"]
    if meta["dtype"] != "float32" or int(np.prod(shape, dtype=np.int64)) * 4 != len(payload):
        return None
    return torch.from_numpy(np.frombuffer(payload, dtype="<f4").reshape(shape).copy())
