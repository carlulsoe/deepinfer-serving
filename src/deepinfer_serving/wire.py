"""Handoff wire format (ADR 0007). No pickle: JSON header plus raw tensor bytes, both SHA-256 checked.

``magic "DISH" | version u16 | header_len u32 | sha256(header) | header JSON | payload``
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from deepinfer_serving.handoff import Handoff

MAGIC = b"DISH"
VERSION = 1
_PREFIX = struct.Struct("<4sHI32s")
_SCALARS = (
    "request_id",
    "user_id",
    "sampling_params",
    "prompt_len",
    "token_ids",
    "num_cached_tokens",
    "config_fingerprint",
)
_TENSORS = ("block_kv", "swa_state")
_DTYPES = {"float32": (torch.float32, np.dtype("<f4"))}


class WireError(ValueError):
    """The bytes are not a valid handoff."""


def encode_handoff(handoff: Handoff) -> bytes:
    table = []
    chunks = []
    offset = 0
    for name in _TENSORS:
        tensor = getattr(handoff, name).detach().cpu().contiguous()
        if tensor.dtype != torch.float32:
            raise WireError(f"{name}: unsupported dtype {tensor.dtype}")
        data = tensor.numpy().astype("<f4", copy=False).tobytes()
        table.append(
            {"name": name, "dtype": "float32", "shape": list(tensor.shape), "offset": offset, "nbytes": len(data)}
        )
        chunks.append(data)
        offset += len(data)
    payload = b"".join(chunks)
    header = {name: getattr(handoff, name) for name in _SCALARS}
    header["tensors"] = table
    header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    prefix = _PREFIX.pack(MAGIC, VERSION, len(header_bytes), hashlib.sha256(header_bytes).digest())
    return prefix + header_bytes + payload


def decode_handoff(data: bytes) -> Handoff:
    from deepinfer_serving.handoff import Handoff

    if len(data) < _PREFIX.size:
        raise WireError("truncated prefix")
    magic, version, header_len, header_digest = _PREFIX.unpack_from(data)
    if magic != MAGIC:
        raise WireError("bad magic")
    if version != VERSION:
        raise WireError(f"unsupported version {version}")
    header_end = _PREFIX.size + header_len
    if header_end > len(data):
        raise WireError("truncated header")
    header_bytes = data[_PREFIX.size : header_end]
    if hashlib.sha256(header_bytes).digest() != header_digest:
        raise WireError("header checksum mismatch")
    try:
        header = json.loads(header_bytes)
    except ValueError as e:
        raise WireError(f"header is not JSON: {e}") from e

    payload = data[header_end:]
    if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
        raise WireError("payload checksum mismatch")

    tensors: dict[str, torch.Tensor] = {}
    expected_offset = 0
    table = header.get("tensors")
    if not isinstance(table, list) or [t.get("name") for t in table] != list(_TENSORS):
        raise WireError("bad tensor table")
    for entry in table:
        if entry.get("dtype") not in _DTYPES or entry.get("offset") != expected_offset:
            raise WireError(f"bad tensor entry {entry.get('name')}")
        torch_dtype, np_dtype = _DTYPES[entry["dtype"]]
        shape = entry.get("shape")
        nbytes = entry.get("nbytes")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise WireError(f"bad shape for {entry['name']}")
        if nbytes != int(np.prod(shape, dtype=np.int64)) * np_dtype.itemsize:
            raise WireError(f"size mismatch for {entry['name']}")
        raw = np.frombuffer(payload, dtype=np_dtype, count=nbytes // np_dtype.itemsize, offset=expected_offset)
        tensors[entry["name"]] = torch.from_numpy(raw.reshape(shape).copy()).to(torch_dtype)
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise WireError("payload has trailing bytes")

    try:
        return Handoff(
            request_id=str(header["request_id"]),
            user_id=str(header["user_id"]),
            sampling_params=dict(header["sampling_params"]),
            prompt_len=int(header["prompt_len"]),
            token_ids=[int(t) for t in header["token_ids"]],
            num_cached_tokens=int(header["num_cached_tokens"]),
            block_kv=tensors["block_kv"],
            swa_state=tensors["swa_state"],
            config_fingerprint=str(header["config_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise WireError(f"bad header field: {e}") from e
