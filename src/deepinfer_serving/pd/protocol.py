"""Framing over a ``multiprocessing`` connection: JSON control messages and raw blobs. No pickle on the wire."""

from __future__ import annotations

import json
import queue
import threading
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


class Writer:
    """Sends on its own thread, in order, so the caller's loop never blocks on a full socket.

    Two instances that each write a large blob to the other would otherwise both block with nobody reading.
    ``failed`` turns true when the other end is gone; the caller treats that like EOF.
    """

    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.failed = False
        self._close_conn = False
        self._queue: queue.SimpleQueue[tuple[dict[str, Any], bytes | None] | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, msg: dict[str, Any], blob: bytes | None = None) -> None:
        """A message and its blob go out back to back."""
        if not self.failed:
            self._queue.put((msg, blob))

    def close(self, timeout: float = 5.0) -> None:
        """Flush what is queued, then stop the thread."""
        self._queue.put(None)
        self._thread.join(timeout)

    def stop(self) -> None:
        """For a peer that is gone: end the thread and close the link without waiting for it."""
        self._close_conn = True
        self._queue.put(None)

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            msg, blob = item
            try:
                send_msg(self.conn, msg)
                if blob is not None:
                    send_blob(self.conn, blob)
            except OSError:
                self.failed = True
                break
        if self._close_conn:  # only after stop(): the reader no longer waits on this connection
            self.conn.close()
