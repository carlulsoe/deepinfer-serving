"""Framing over a ``multiprocessing`` connection: JSON control messages and raw blobs. No pickle on the wire."""

from __future__ import annotations

import json
from multiprocessing.connection import Connection
from typing import Any

_JSON = b"J"
_BLOB = b"B"


def send_msg(conn: Connection, msg: dict[str, Any]) -> None:
    conn.send_bytes(_JSON + json.dumps(msg, separators=(",", ":")).encode())


def send_blob(conn: Connection, data: bytes) -> None:
    conn.send_bytes(_BLOB + data)


def recv_msg(conn: Connection) -> dict[str, Any]:
    frame = conn.recv_bytes()
    if frame[:1] != _JSON:
        raise ValueError("expected a control message")
    return json.loads(frame[1:])


def recv_blob(conn: Connection) -> bytes:
    frame = conn.recv_bytes()
    if frame[:1] != _BLOB:
        raise ValueError("expected a blob")
    return frame[1:]
