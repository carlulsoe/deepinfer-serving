"""KVStore conformance: every backend behaves the same through the protocol (ADR 0010).

The 3FS case is the file backend on a 3FS mount; it runs only when ``DEEPINFER_SERVING_3FS_PATH`` is set.
"""

import multiprocessing as mp
import os
import struct
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from helpers import DEFAULT, SMALL, UNBOUNDED, Store, open_store, store_keys, store_tensor, store_writer

from deepinfer_serving.file_store import _PREFIX, FileKVStore
from deepinfer_serving.pd.instance import StoreSpec

_3FS = os.environ.get("DEEPINFER_SERVING_3FS_PATH")
_FILE_BACKENDS = [
    pytest.param("file", id="file"),
    pytest.param("3fs", id="3fs", marks=pytest.mark.skipif(not _3FS, reason="DEEPINFER_SERVING_3FS_PATH not set")),
]

Factory = Callable[..., Store]


def _factory(backend: str, tmp_path: Path) -> Factory:
    root = Path(_3FS or "") / f"conformance-{uuid.uuid4().hex}" if backend == "3fs" else tmp_path

    def factory(block_capacity_bytes: int = UNBOUNDED, swa_capacity_bytes: int = UNBOUNDED, name: str = "s") -> Store:
        kind = "memory" if backend == "memory" else "file"
        return open_store(kind, root / name, block_capacity_bytes, swa_capacity_bytes)

    return factory


@pytest.fixture(params=[pytest.param("memory", id="memory"), *_FILE_BACKENDS])
def make(request: pytest.FixtureRequest, tmp_path: Path) -> Factory:
    return _factory(request.param, tmp_path)


@pytest.fixture(params=_FILE_BACKENDS)
def file_make(request: pytest.FixtureRequest, tmp_path: Path) -> Factory:
    return _factory(request.param, tmp_path)


def test_block_round_trip_is_bit_exact(make: Factory) -> None:
    store = make()
    (key,) = store_keys(1)
    kv = store_tensor(0)
    assert not store.has_block(key) and store.get_block(key) is None
    store.put_block(key, kv)
    assert store.has_block(key)
    got = store.get_block(key)
    assert got is not None and got.dtype == torch.float32 and torch.equal(got, kv)


def test_swa_records_are_keyed_by_position(make: Factory) -> None:
    store = make()
    (key,) = store_keys(1)
    store.put_swa(key, 16, store_tensor(1))
    assert store.has_swa(key, 16) and not store.has_swa(key, 32)
    assert store.get_swa(key, 32) is None
    got = store.get_swa(key, 16)
    assert got is not None and torch.equal(got, store_tensor(1))


def test_store_holds_copies(make: Factory) -> None:
    store = make()
    (key,) = store_keys(1)
    kv = store_tensor(2)
    store.put_block(key, kv)
    kv.zero_()
    got = store.get_block(key)
    assert got is not None and torch.equal(got, store_tensor(2))
    got.zero_()
    again = store.get_block(key)
    assert again is not None and torch.equal(again, store_tensor(2))


def test_put_overwrites(make: Factory) -> None:
    store = make()
    (key,) = store_keys(1)
    store.put_block(key, store_tensor(3))
    store.put_block(key, store_tensor(4))
    got = store.get_block(key)
    assert got is not None and torch.equal(got, store_tensor(4))
    assert len(store.block_keys()) == 1


def _entry_bytes(make: Factory) -> tuple[int, int]:
    probe = make(name="probe")
    (key,) = store_keys(1)
    probe.put_block(key, store_tensor(0))
    probe.put_swa(key, 16, store_tensor(0))
    return probe.block_bytes_used(), probe.swa_bytes_used()


def test_block_eviction_is_lru_under_watermark(make: Factory) -> None:
    block_bytes, _ = _entry_bytes(make)
    store = make(block_capacity_bytes=int(3.5 * block_bytes))
    a, b, c, d = store_keys(4)
    for i, key in enumerate((a, b, c)):
        store.put_block(key, store_tensor(i))
    assert store.get_block(a) is not None  # a is now the most recently used
    store.put_block(d, store_tensor(3))  # 4 entries > 3.5: evict LRU (b) down to <= 0.9 * 3.5
    assert [store.has_block(k) for k in (a, b, c, d)] == [True, False, True, True]
    assert store.block_bytes_used() <= int(3.5 * block_bytes)


def test_swa_eviction_is_lru_under_watermark(make: Factory) -> None:
    _, swa_bytes = _entry_bytes(make)
    store = make(swa_capacity_bytes=int(3.5 * swa_bytes))
    (key,) = store_keys(1)
    for p in (16, 32, 48):
        store.put_swa(key, p, store_tensor(p))
    assert store.get_swa(key, 16) is not None
    store.put_swa(key, 64, store_tensor(64))
    assert [store.has_swa(key, p) for p in (16, 32, 48, 64)] == [True, False, True, True]


def test_discard(make: Factory) -> None:
    store = make()
    (key,) = store_keys(1)
    store.put_block(key, store_tensor(0))
    store.put_swa(key, 16, store_tensor(0))
    store.discard_block(key)
    store.discard_swa(key, 16)
    assert not store.has_block(key) and not store.has_swa(key, 16)
    assert store.block_bytes_used() == 0 and store.swa_bytes_used() == 0


# ---- file-backed only: sharing, restart, damage, concurrency ----


def test_block_tier_is_shared_and_survives_restart(file_make: Factory) -> None:
    writer = file_make(name="shared")
    (key,) = store_keys(1)
    writer.put_block(key, store_tensor(5))
    writer.put_swa(key, 16, store_tensor(5))
    reader = file_make(name="shared")  # another process, or the same one after a restart
    got = reader.get_block(key)
    assert got is not None and torch.equal(got, store_tensor(5))
    assert not reader.has_swa(key, 16)  # the SWA tier is host RAM of the producer (ADR 0008)


def _damage(path: Path, how: str) -> None:
    data = bytearray(path.read_bytes())
    if how == "flip_prefix":
        data[10] ^= 0x10
    elif how in ("header_len_up", "header_len_down"):
        offset = _PREFIX.size - 4  # header_len: moves the header/payload boundary, keeping the bytes
        (n,) = struct.unpack_from("<I", data, offset)
        struct.pack_into("<I", data, offset, n + (1 if how == "header_len_up" else -1))
    elif how == "flip_header":
        data[_PREFIX.size + 2] ^= 0x10
    elif how == "flip_payload":
        data[-3] ^= 0x10
    elif how == "truncate":
        data = data[: len(data) // 2]
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    "how", ["flip_prefix", "header_len_up", "header_len_down", "flip_header", "flip_payload", "truncate", "wrong_key"]
)
def test_damaged_file_is_a_miss_and_is_removed(file_make: Factory, how: str) -> None:
    store = file_make()
    assert isinstance(store, FileKVStore)
    a, b = store_keys(2)
    store.put_block(a, store_tensor(6))
    store.put_block(b, store_tensor(7))
    if how == "wrong_key":
        os.replace(store.path / b.hex(), store.path / a.hex())
    else:
        _damage(store.path / a.hex(), how)
    assert store.has_block(a)  # presence is metadata only
    assert store.get_block(a) is None
    assert not store.has_block(a)


def test_stale_temp_files_are_ignored_and_reaped(file_make: Factory) -> None:
    store = file_make()
    assert isinstance(store, FileKVStore)
    stale = store.path / ".tmp-1-dead"
    stale.write_bytes(b"x" * 100)
    os.utime(stale, (0, 0))
    store.put_block(store_keys(1)[0], store_tensor(0))
    assert store.block_keys() == store_keys(1)
    store = file_make(block_capacity_bytes=1)
    store.put_block(store_keys(2)[1], store_tensor(0))  # any put runs eviction, which reaps stale temp files
    assert not stale.exists()


def test_concurrent_writers_never_expose_partial_blocks(file_make: Factory) -> None:
    """Several processes write overlapping keys under eviction pressure; every readable block is exact."""
    store = file_make(name="race")
    assert isinstance(store, FileKVStore)
    block_bytes = _entry_bytes(file_make)[0]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=store_writer, args=(str(store.path), 20 * block_bytes, seed)) for seed in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    keys = store_keys(40)
    present = 0
    for i, key in enumerate(keys):
        got = store.get_block(key)
        if got is not None:
            present += 1
            assert torch.equal(got, store_tensor(i))
    assert 0 < present <= 20
    assert store.block_bytes_used() <= 20 * block_bytes


def test_configs_do_not_share_a_namespace(tmp_path: Path) -> None:
    """Block keys hash tokens only; a different model or cache config must not see the same blocks."""
    spec = StoreSpec(str(tmp_path), UNBOUNDED, UNBOUNDED)
    assert spec.open(SMALL).path != spec.open(DEFAULT).path
    assert spec.open(SMALL).path == spec.open(replace(SMALL, num_gpu_blocks=7)).path  # capacity is not identity
