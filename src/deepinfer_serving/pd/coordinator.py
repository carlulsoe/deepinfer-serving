"""Request coordinator: owns each request's state and token stream (ADR 0006).

In milestone 2 it runs in the caller's process and drives one prefill and one decode instance. At milestone 4 the
Entrypoint takes this role.
"""

from __future__ import annotations

import contextlib
import enum
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any

from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.pd.instance import InstanceSpec, instance_main
from deepinfer_serving.pd.protocol import recv_msg, send_msg


class RequestState(enum.Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    HANDED_OFF = "handed_off"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (RequestState.FINISHED, RequestState.ABORTED, RequestState.FAILED)


_S = RequestState
_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    _S.QUEUED: {_S.PREFILLING, _S.ABORTED, _S.FAILED},
    _S.PREFILLING: {_S.HANDED_OFF, _S.FINISHED, _S.ABORTED, _S.FAILED},
    _S.HANDED_OFF: {_S.DECODING, _S.ABORTED, _S.FAILED},
    _S.DECODING: {_S.FINISHED, _S.ABORTED, _S.FAILED},
}
_EVENT_STATE = {"prefilling": _S.PREFILLING, "handed_off": _S.HANDED_OFF, "admitted": _S.DECODING}


@dataclass
class RequestRecord:
    request_id: str
    state: RequestState = RequestState.QUEUED
    history: list[RequestState] = field(default_factory=lambda: [RequestState.QUEUED])
    tokens_by_index: dict[int, int] = field(default_factory=dict)
    num_output_tokens: int | None = None
    reason: str = ""
    hit: tuple[int, int, int] | None = None
    """``(cached_hit_tokens, num_cached_tokens, swa_valid_start)`` on the prefill instance, once prefill starts."""

    @property
    def tokens(self) -> list[int]:
        return [self.tokens_by_index[i] for i in sorted(self.tokens_by_index)]

    @property
    def complete(self) -> bool:
        if not self.state.terminal:
            return False
        if self.state == RequestState.FINISHED:
            return len(self.tokens_by_index) == self.num_output_tokens
        return True


class InstanceFailed(RuntimeError):
    pass


class PDCoordinator:
    """Spawns a prefill and a decode instance and tracks requests across them."""

    ROLES = ("prefill", "decode")

    def __init__(self, prefill: InstanceSpec, decode: InstanceSpec, *, startup_timeout: float = 120.0) -> None:
        if prefill.role != EngineType.PROMPT or decode.role != EngineType.COMPLETION:
            raise ValueError("prefill must be a PROMPT instance and decode a COMPLETION instance")
        self.specs = {"prefill": prefill, "decode": decode}
        self.startup_timeout = startup_timeout
        self.requests: dict[str, RequestRecord] = {}
        self.errors: list[str] = []
        self.duplicate_terminals = 0
        self._ctrl: dict[str, Connection] = {}
        self._procs: dict[str, Any] = {}
        self._stats: dict[tuple[str, int], dict[str, int]] = {}
        self._stats_seq = 0

    # ---- lifecycle ----

    def start(self) -> PDCoordinator:
        ctx = mp.get_context("spawn")  # CUDA-safe
        data_p, data_d = ctx.Pipe(duplex=True)
        links = {"prefill": data_p, "decode": data_d}
        for role in self.ROLES:
            mine, theirs = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=instance_main, args=(self.specs[role], theirs, links[role]), name=f"pd-{role}", daemon=True
            )
            proc.start()
            theirs.close()
            self._ctrl[role] = mine
            self._procs[role] = proc
        data_p.close()
        data_d.close()
        deadline = time.monotonic() + self.startup_timeout
        for role in self.ROLES:
            remaining = deadline - time.monotonic()
            try:
                if not self._ctrl[role].poll(max(remaining, 0)):
                    raise InstanceFailed(f"{role} instance did not start")
                self._handle(role, recv_msg(self._ctrl[role]))
            except EOFError as e:
                raise InstanceFailed(f"{role} instance exited during startup") from e
        return self

    def close(self) -> None:
        for role, conn in self._ctrl.items():
            if self._procs[role].is_alive():
                with contextlib.suppress(OSError):
                    send_msg(conn, {"type": "shutdown"})
        for proc in self._procs.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for conn in self._ctrl.values():
            conn.close()

    def __enter__(self) -> PDCoordinator:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- requests ----

    def submit(self, request_id: str, prompt: list[int], sampling_params: dict, user_id: str = "u") -> None:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id {request_id}")
        self.requests[request_id] = RequestRecord(request_id)
        msg = {
            "type": "add",
            "rid": request_id,
            "user_id": user_id,
            "prompt": prompt,
            "sampling_params": sampling_params,
        }
        send_msg(self._ctrl["prefill"], msg)

    def abort(self, request_id: str) -> None:
        """Abort enters at the prefill instance and follows the request's path (ADR 0006)."""
        if not self.requests[request_id].state.terminal:
            send_msg(self._ctrl["prefill"], {"type": "abort", "rid": request_id})

    # ---- instance control ----

    def pause(self, role: str) -> None:
        send_msg(self._ctrl[role], {"type": "pause"})

    def resume(self, role: str) -> None:
        send_msg(self._ctrl[role], {"type": "resume"})

    def step(self, role: str, n: int = 1) -> None:
        """While paused, let the instance run ``n`` loop iterations (inbox, retries, one engine step each)."""
        send_msg(self._ctrl[role], {"type": "step", "n": n})

    def set_faults(self, role: str, **faults: Any) -> None:
        send_msg(self._ctrl[role], {"type": "faults", "faults": faults})

    def stats(self, role: str, timeout: float = 30.0) -> dict[str, int]:
        self._stats_seq += 1
        key = (role, self._stats_seq)
        send_msg(self._ctrl[role], {"type": "stats", "seq": self._stats_seq})
        self.wait_for(lambda: key in self._stats, timeout)
        return self._stats.pop(key)

    # ---- event loop ----

    def poll(self, timeout: float = 0.0) -> None:
        by_conn = {conn: role for role, conn in self._ctrl.items()}
        for conn in wait(list(by_conn), timeout=timeout):
            assert isinstance(conn, Connection)
            role = by_conn[conn]
            try:
                while conn.poll():
                    self._handle(role, recv_msg(conn))
            except EOFError as e:
                raise InstanceFailed(f"{role} instance exited") from e

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("condition not reached")
            self.poll(min(remaining, 0.05))

    def run_until_done(self, timeout: float = 300.0) -> dict[str, RequestRecord]:
        self.wait_for(lambda: all(r.complete for r in self.requests.values()), timeout)
        return self.requests

    def _handle(self, role: str, msg: dict[str, Any]) -> None:
        kind = msg["type"]
        if kind == "ready":
            return
        if kind == "fatal":
            raise InstanceFailed(f"{role} instance crashed:\n{msg['error']}")
        if kind == "stats":
            self._stats[(role, msg["seq"])] = {k: v for k, v in msg.items() if k not in ("type", "seq")}
            return
        record = self.requests.get(msg["rid"])
        if record is None:
            self.errors.append(f"{role}: event {kind} for unknown request {msg['rid']}")
            return
        if kind == "token":
            if msg["index"] in record.tokens_by_index:
                self.errors.append(f"{role}: duplicate token index {msg['index']} for {record.request_id}")
            record.tokens_by_index[msg["index"]] = msg["token"]
        elif kind in _EVENT_STATE:
            if kind == "prefilling":
                record.hit = tuple(msg["hit"])
            self._transition(record, _EVENT_STATE[kind], role)
        elif kind == "terminal":
            if record.state.terminal:
                self.duplicate_terminals += 1
                self.errors.append(f"{role}: second terminal {msg['state']} for {record.request_id}")
                return
            self._transition(record, RequestState(msg["state"]), role)
            record.reason = msg["reason"]
            record.num_output_tokens = msg["num_output_tokens"]
        else:
            self.errors.append(f"{role}: unknown event {kind}")

    def _transition(self, record: RequestRecord, new: RequestState, role: str) -> None:
        if new not in _TRANSITIONS.get(record.state, set()):
            self.errors.append(
                f"{role}: invalid transition {record.state.value} -> {new.value} for {record.request_id}"
            )
            return
        record.state = new
        record.history.append(new)
