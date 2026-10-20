"""Instance controller: owns one engine in its own process (architecture.md, piece 3).

Links:

- ``ctrl``: to the coordinator. Commands in (add, abort, pause, resume, step, faults, stats, shutdown); events out
  (ready, prefilling, handed_off, admitted, token, terminal, stats, fatal).
- ``data``: prefill <-> decode. Handoffs and forwarded aborts flow prefill -> decode, FIFO (ADR 0006); ack/nack
  flow back (ADR 0005).
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any

import numpy as np

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.scheduler import SchedulerOutputs
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.pd.protocol import recv_blob, recv_msg, send_blob, send_msg
from deepinfer_serving.serving_engine import ServingLLMEngine
from deepinfer_serving.wire import WireError

RETRY_BACKOFF_S = 0.005
_IDLE_WAIT_S = 0.005


@dataclass(frozen=True)
class InstanceSpec:
    role: EngineType
    config: MiniConfig
    device: str = "cpu"


@dataclass
class _OutboxEntry:
    blob: bytes
    awaiting_answer: bool = True
    abort_pending: bool = False
    retry_at: float = 0.0


class _Engine(ServingLLMEngine):
    """Uses ``post_schedule`` to report when a request's prefill starts."""

    def __init__(self, spec: InstanceSpec, controller: InstanceController) -> None:
        self._controller = controller
        sink = controller.on_handoff if spec.role == EngineType.PROMPT else None
        super().__init__(spec.config, spec.role, spec.device, handoff_sink=sink)

    def post_schedule(self, scheduler_outputs: SchedulerOutputs) -> None:
        if scheduler_outputs.prompt_run:
            for seq in scheduler_outputs.scheduled_seqs:
                self._controller.on_prefill_scheduled(seq.request_id)


class InstanceController:
    def __init__(self, spec: InstanceSpec, ctrl: Connection, data: Connection) -> None:
        self.spec = spec
        self.ctrl = ctrl
        self.data = data
        self.is_prefill = spec.role == EngineType.PROMPT
        self.paused = False
        self.step_budget = 0
        self.faults: dict[str, Any] = {}
        self.outbox: dict[str, _OutboxEntry] = {}
        self.inbox: deque[tuple[str, str, bytes | None]] = deque()
        self.stopping = False
        self._prefill_reported: set[str] = set()
        self._aborted_awaiting_output: set[str] = set()
        self.nacks_received = 0
        self.handoffs_sent = 0
        self.engine = _Engine(spec, self)

    # ---- events to the coordinator ----

    def _emit(self, msg: dict[str, Any]) -> None:
        send_msg(self.ctrl, msg)

    def _terminal(self, rid: str, state: str, reason: str, num_output_tokens: int = 0) -> None:
        self._prefill_reported.discard(rid)
        self._emit(
            {"type": "terminal", "rid": rid, "state": state, "reason": reason, "num_output_tokens": num_output_tokens}
        )

    def on_prefill_scheduled(self, rid: str) -> None:
        if rid not in self._prefill_reported:
            self._prefill_reported.add(rid)
            self._emit({"type": "prefilling", "rid": rid})

    def on_handoff(self, handoff: Handoff) -> None:
        """PROMPT engine sink. The bytes are an owned copy, so the engine frees device memory next (ADR 0005)."""
        rid = handoff.request_id
        self._prefill_reported.discard(rid)
        self.outbox[rid] = _OutboxEntry(handoff.to_bytes())
        self._send_handoff(rid)
        self._emit({"type": "handed_off", "rid": rid})

    # ---- data link ----

    def _send_handoff(self, rid: str) -> None:
        entry = self.outbox[rid]
        entry.awaiting_answer = True
        self.handoffs_sent += 1
        send_msg(self.data, {"type": "handoff", "rid": rid})
        send_blob(self.data, _apply_wire_fault(entry.blob, self.faults.get("corrupt")))

    def _on_data(self, msg: dict[str, Any]) -> None:
        kind, rid = msg["type"], msg["rid"]
        if kind == "handoff":  # decode side; queued so FIFO order with forwarded aborts is kept
            self.inbox.append(("handoff", rid, recv_blob(self.data)))
        elif kind == "abort":
            self.inbox.append(("abort", rid, None))
        elif kind == "ack":
            self.outbox.pop(rid, None)
        elif kind == "nack":
            self._on_nack(rid, msg["reason"])
        else:
            raise ValueError(f"unknown data message {kind}")

    def _on_nack(self, rid: str, reason: str) -> None:
        self.nacks_received += 1
        entry = self.outbox.get(rid)
        if entry is None:
            return
        if entry.abort_pending:
            del self.outbox[rid]
            self._terminal(rid, "aborted", "abort")
        elif reason == "no_capacity":
            entry.awaiting_answer = False
            entry.retry_at = time.monotonic() + RETRY_BACKOFF_S
        else:
            del self.outbox[rid]
            self._terminal(rid, "failed", reason)

    def _process_inbox(self) -> None:
        while self.inbox:
            kind, rid, blob = self.inbox.popleft()
            if kind == "abort":
                if self.engine.abort_request(rid)[0]:
                    self._terminal(rid, "aborted", "abort")
                continue
            assert blob is not None
            try:
                handoff = Handoff.from_bytes(blob)
                if handoff.request_id != rid:
                    raise WireError("request id in blob does not match envelope")
                seq = self.engine.receive_handoff(handoff)
            except (WireError, ValueError) as e:
                send_msg(self.data, {"type": "nack", "rid": rid, "reason": f"invalid: {e}"})
                continue
            if seq is None:
                send_msg(self.data, {"type": "nack", "rid": rid, "reason": "no_capacity"})
                continue
            send_msg(self.data, {"type": "ack", "rid": rid})
            self._emit({"type": "admitted", "rid": rid})

    def _retry_outbox(self) -> None:
        now = time.monotonic()
        for rid, entry in self.outbox.items():
            if not entry.awaiting_answer and entry.retry_at <= now:
                self._send_handoff(rid)

    # ---- control link ----

    def _on_ctrl(self, msg: dict[str, Any]) -> None:
        kind = msg["type"]
        if kind == "add":
            self.engine.add_request(
                msg["user_id"], msg["rid"], msg["sampling_params"], np.asarray(msg["prompt"], dtype=np.int32)
            )
        elif kind == "abort":
            self._abort(msg["rid"])
        elif kind == "pause":
            self.paused, self.step_budget = True, 0
        elif kind == "resume":
            self.paused, self.step_budget = False, 0
        elif kind == "step":
            self.step_budget += int(msg["n"])
        elif kind == "faults":
            self.faults = dict(msg["faults"])
        elif kind == "stats":
            self._emit({"type": "stats", "seq": msg["seq"], **self._stats()})
        elif kind == "shutdown":
            self.stopping = True
        else:
            raise ValueError(f"unknown control message {kind}")

    def _abort(self, rid: str) -> None:
        """ADR 0006: abort where the request is; otherwise forward it down the request's own path."""
        if self.engine.abort_request(rid)[0]:
            self._aborted_awaiting_output.add(rid)
            self._terminal(rid, "aborted", "abort")
            return
        entry = self.outbox.get(rid)
        if entry is not None and not entry.awaiting_answer:  # nacked, waiting to retry: we hold it
            del self.outbox[rid]
            self._terminal(rid, "aborted", "abort")
            return
        if entry is not None:
            entry.abort_pending = True
        if self.is_prefill:
            send_msg(self.data, {"type": "abort", "rid": rid})

    def _stats(self) -> dict[str, int]:
        e = self.engine
        return {
            "free_blocks": e.block_manager.get_num_free_blocks(),
            "num_blocks": e.block_manager.num_blocks,
            "free_slots": e.state_manager.get_num_free_blocks(),
            "num_slots": e.state_manager.num_blocks,
            "unfinished": e.get_num_unfinished_requests(),
            "outbox": len(self.outbox),
            "inbox": len(self.inbox),
            "handoffs_sent": self.handoffs_sent,
            "nacks_received": self.nacks_received,
        }

    # ---- loop ----

    def _report(self, outputs: list) -> None:
        for o in outputs:
            if o.request_id in self._aborted_awaiting_output:
                self._aborted_awaiting_output.discard(o.request_id)
                continue
            if o.token_id >= 0:
                self._emit(
                    {"type": "token", "rid": o.request_id, "index": o.num_output_tokens - 1, "token": o.token_id}
                )
            if o.finished:
                state = "finished" if o.token_id >= 0 else "failed"
                self._terminal(o.request_id, state, o.finish_reason, o.num_output_tokens)

    def _can_work(self) -> bool:
        return not self.paused or self.step_budget > 0

    def _has_work(self) -> bool:
        return (
            self.engine.has_unfinished_requests()
            or bool(self.inbox)
            or any(not e.awaiting_answer for e in self.outbox.values())
        )

    def run(self) -> None:
        self._emit({"type": "ready"})
        while not self.stopping:
            busy = self._can_work() and self._has_work()
            for conn in wait([self.ctrl, self.data], timeout=0 if busy else _IDLE_WAIT_S):
                assert isinstance(conn, Connection)
                while conn.poll():
                    msg = recv_msg(conn)
                    (self._on_ctrl if conn is self.ctrl else self._on_data)(msg)
            if not self._can_work():
                continue
            if self.paused:
                self.step_budget -= 1
            self._process_inbox()
            self._retry_outbox()
            if self.engine.has_unfinished_requests():
                self._report(self.engine.step())


def _apply_wire_fault(blob: bytes, where: str | None) -> bytes:
    """Test fault injection: flip one byte in the header or the payload of a handoff blob."""
    if where is None:
        return blob
    from deepinfer_serving.wire import _PREFIX

    header_len = _PREFIX.unpack_from(blob)[2]
    if where == "header":
        pos = _PREFIX.size + header_len // 2
    elif where == "payload":
        pos = _PREFIX.size + header_len + (len(blob) - _PREFIX.size - header_len) // 2
    else:
        raise ValueError(f"unknown fault {where}")
    corrupted = bytearray(blob)
    corrupted[pos] ^= 0x01
    return bytes(corrupted)


def instance_main(spec: InstanceSpec, ctrl: Connection, data: Connection) -> None:
    try:
        InstanceController(spec, ctrl, data).run()
    except Exception:
        send_msg(ctrl, {"type": "fatal", "error": traceback.format_exc()})
        raise
