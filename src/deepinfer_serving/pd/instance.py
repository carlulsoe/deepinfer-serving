"""Instance controller: owns one engine in its own process (architecture.md, piece 3).

Links:

- ``ctrl``: to the Entrypoint. Commands in: add, abort, drain, peer_dead, pause, resume, step, faults, stats,
  shutdown. Events out: ready, prefilling, handed_off, admitted, forwarded, token, terminal, swa_index, stats,
  fatal.
- ``peers``: one link to every other instance. Handoffs (prefill -> decode, and decode -> decode when draining),
  their ack/nack (ADR 0005), and SWA record pulls (ADR 0011).

Request ids here are the Entrypoint's attempt ids; the instance treats them as opaque.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

import numpy as np

from deepinfer_serving.engine.config import MiniConfig
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.scheduler import SchedulerOutputs
from deepinfer_serving.engine.sequence import Sequence
from deepinfer_serving.file_store import FileKVStore, decode_tensor, encode_tensor
from deepinfer_serving.handoff import Handoff
from deepinfer_serving.kvstore import Hit, block_hashes, namespace
from deepinfer_serving.pd.protocol import Writer, recv_blob, recv_msg, send_msg
from deepinfer_serving.serving_engine import ServingLLMEngine
from deepinfer_serving.wire import WireError

RETRY_BACKOFF_S = 0.005
_IDLE_WAIT_S = 0.005


@dataclass(frozen=True)
class StoreSpec:
    """A ``FileKVStore`` under ``path``, one sub-directory per config (ADR 0010)."""

    path: str
    block_capacity_bytes: int
    swa_capacity_bytes: int

    def open(self, config: MiniConfig) -> FileKVStore:
        return FileKVStore(
            Path(self.path) / namespace(config.fingerprint()), self.block_capacity_bytes, self.swa_capacity_bytes
        )


@dataclass(frozen=True)
class InstanceSpec:
    role: EngineType
    config: MiniConfig
    device: str = "cpu"
    store: StoreSpec | None = None


@dataclass
class _OutboxEntry:
    blob: bytes
    target: str
    hop: int
    """1 for a prefill handoff, +1 for every drain forward: lets the Entrypoint order events (ADR 0012)."""
    awaiting_answer: bool = True
    abort_pending: bool = False
    retry_at: float = 0.0


@dataclass
class _PendingPull:
    add: dict[str, Any]
    holder: str


class _Engine(ServingLLMEngine):
    """Uses ``post_schedule`` to report when a request's prefill starts."""

    def __init__(self, spec: InstanceSpec, controller: InstanceController) -> None:
        self._controller = controller
        sink = controller.on_handoff if spec.role == EngineType.PROMPT else None
        self.file_store = spec.store.open(spec.config) if spec.store is not None else None
        super().__init__(spec.config, spec.role, spec.device, store=self.file_store, handoff_sink=sink)

    def post_schedule(self, scheduler_outputs: SchedulerOutputs) -> None:
        if scheduler_outputs.prompt_run:
            for seq in scheduler_outputs.scheduled_seqs:
                self._controller.on_prefill_scheduled(seq)


class InstanceController:
    def __init__(self, name: str, spec: InstanceSpec, ctrl: Connection, peers: dict[str, Connection]) -> None:
        self.name = name
        self.spec = spec
        self.ctrl = ctrl
        self.ctrl_out = Writer(ctrl)
        self.peers = dict(peers)
        self.peer_out = {name: Writer(conn) for name, conn in peers.items()}
        self.paused = False
        self.step_budget = 0
        self.faults: dict[str, Any] = {}
        self.outbox: dict[str, _OutboxEntry] = {}
        self.inbox: deque[tuple[str, str, int, bytes]] = deque()  # (source, rid, hop, handoff blob)
        self.hops: dict[str, int] = {}  # hop of each sequence admitted here
        self.pending_pulls: dict[str, _PendingPull] = {}
        self.decode_target: dict[str, str] = {}
        self.drain_targets: list[str] | None = None
        self._drain_rr = 0
        self.stopping = False
        self._prefill_reported: set[str] = set()
        self._aborted_awaiting_output: set[str] = set()
        self._swa_reported: set[tuple[bytes, int]] = set()
        self.counters = {"handoffs_sent": 0, "nacks_received": 0, "swa_pulls_served": 0, "forwarded": 0, "aborts": 0}
        self.engine = _Engine(spec, self)

    # ---- events to the Entrypoint ----

    def _emit(self, msg: dict[str, Any]) -> None:
        self.ctrl_out.send(msg)

    def _terminal(self, rid: str, state: str, reason: str, num_output_tokens: int = 0) -> None:
        self._prefill_reported.discard(rid)
        self._emit(
            {"type": "terminal", "rid": rid, "state": state, "reason": reason, "num_output_tokens": num_output_tokens}
        )

    def on_prefill_scheduled(self, seq: Sequence) -> None:
        """Reported once, at the first chunk, so ``num_cached_tokens`` is still where the cache hit resumes."""
        rid = seq.request_id
        if rid not in self._prefill_reported:
            self._prefill_reported.add(rid)
            hit = [seq.cached_hit_tokens, seq.num_cached_tokens, seq.swa_valid_start]
            self._emit({"type": "prefilling", "rid": rid, "hit": hit})

    def on_handoff(self, handoff: Handoff) -> None:
        """PROMPT engine sink. The bytes are an owned copy, so the engine frees device memory next (ADR 0005)."""
        rid = handoff.request_id
        self._prefill_reported.discard(rid)
        target = self.decode_target.pop(rid)
        self._emit({"type": "handed_off", "rid": rid})
        self._queue_handoff(rid, handoff, target, hop=1)

    def _report_swa_index(self) -> None:
        """Tell the Entrypoint's SWA directory what this instance's RAM tier now holds (ADR 0011)."""
        store = self.engine.file_store
        if store is None:
            return
        current = set(store.swa_entries())
        if current != self._swa_reported:
            added = [[k.hex(), p] for k, p in current - self._swa_reported]
            removed = [[k.hex(), p] for k, p in self._swa_reported - current]
            self._swa_reported = current
            self._emit({"type": "swa_index", "added": added, "removed": removed})

    # ---- peer links ----

    def _send_peer(self, peer: str, msg: dict[str, Any], blob: bytes | None = None) -> bool:
        writer = self.peer_out.get(peer)
        if writer is None or writer.failed:
            self._on_peer_dead(peer)
            return False
        writer.send(msg, blob)
        return True

    def _queue_handoff(self, rid: str, handoff: Handoff, target: str, hop: int) -> None:
        self.outbox[rid] = _OutboxEntry(handoff.to_bytes(), target, hop)
        self._send_handoff(rid)

    def _send_handoff(self, rid: str) -> None:
        entry = self.outbox[rid]
        entry.awaiting_answer = True
        self.counters["handoffs_sent"] += 1
        blob = _apply_wire_fault(entry.blob, self.faults.get("corrupt"))
        self._send_peer(entry.target, {"type": "handoff", "rid": rid, "hop": entry.hop}, blob)

    def _on_peer(self, source: str, msg: dict[str, Any]) -> None:
        conn = self.peers[source]
        kind, rid = msg["type"], msg["rid"]
        if kind == "handoff":  # queued so a paused instance does not admit it
            self.inbox.append((source, rid, msg["hop"], recv_blob(conn)))
        elif kind == "ack":
            self.outbox.pop(rid, None)
        elif kind == "nack":
            self._on_nack(rid, msg["reason"])
        elif kind == "swa_pull":
            self._serve_pull(source, rid, bytes.fromhex(msg["key"]), msg["position"])
        elif kind == "swa_record":
            blob = recv_blob(conn) if msg["found"] else None
            self._on_pulled(rid, bytes.fromhex(msg["key"]), msg["position"], blob)
        else:
            raise ValueError(f"unknown peer message {kind}")

    def _on_nack(self, rid: str, reason: str) -> None:
        self.counters["nacks_received"] += 1
        entry = self.outbox.get(rid)
        if entry is None:
            return
        if entry.abort_pending or reason == "aborted":
            del self.outbox[rid]
        elif reason == "no_capacity":
            entry.awaiting_answer = False
            entry.retry_at = time.monotonic() + RETRY_BACKOFF_S
        else:
            del self.outbox[rid]
            self._terminal(rid, "failed", reason)

    def _on_peer_dead(self, peer: str) -> None:
        """The Entrypoint re-routes what depended on ``peer`` (ADR 0012); here we only drop our side of it."""
        self.peers.pop(peer, None)  # the writer thread owns the connection and closes it
        writer = self.peer_out.pop(peer, None)
        if writer is not None:
            writer.stop()
        for rid in [rid for rid, e in self.outbox.items() if e.target == peer]:
            del self.outbox[rid]
        for rid in [rid for rid, p in self.pending_pulls.items() if p.holder == peer]:
            self._admit_new(self.pending_pulls.pop(rid).add, None)
        if self.drain_targets is not None and peer in self.drain_targets:
            self.drain_targets.remove(peer)

    # ---- SWA pulls (ADR 0011) ----

    def _serve_pull(self, source: str, rid: str, key: bytes, position: int) -> None:
        store = self.engine.file_store
        record = store.get_swa(key, position) if store is not None else None
        if self.faults.get("lose_swa"):  # test fault: evicted since the directory last heard
            record = None
        msg = {"type": "swa_record", "rid": rid, "key": key.hex(), "position": position, "found": record is not None}
        if record is not None:
            self.counters["swa_pulls_served"] += 1
        self._send_peer(source, msg, encode_tensor(key, record) if record is not None else None)

    def _on_pulled(self, rid: str, key: bytes, position: int, blob: bytes | None) -> None:
        pending = self.pending_pulls.pop(rid, None)
        if pending is None:
            return  # aborted while the pull was in flight
        record = decode_tensor(key, blob) if blob is not None else None
        store = self.engine.file_store
        if record is None or store is None:
            self._admit_new(pending.add, None)  # evicted or damaged at the holder: the local hit rule backs off
            return
        store.put_swa(key, position, record)
        self._admit_new(pending.add, _hit_from_plan(pending.add, self.spec.config))

    # ---- control link ----

    def _on_ctrl(self, msg: dict[str, Any]) -> None:
        kind = msg["type"]
        if kind == "add":
            self._on_add(msg)
        elif kind == "abort":
            self._abort(msg["rid"])
        elif kind == "drain":
            self.drain_targets = [t for t in msg["targets"] if t in self.peers]
        elif kind == "peer_dead":
            self._on_peer_dead(msg["name"])
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

    def _on_add(self, msg: dict[str, Any]) -> None:
        rid, plan, holder = msg["rid"], msg["hit"], msg["swa_holder"]
        self.decode_target[rid] = msg["decode"]
        store = self.engine.file_store
        if plan is not None and plan["swa_position"] is not None and holder != self.name and store is not None:
            position = plan["swa_position"]
            key = block_hashes(msg["prompt"], self.spec.config.block_size)[position // self.spec.config.grain - 1]
            if not store.has_swa(key, position):
                pull = {"type": "swa_pull", "rid": rid, "key": key.hex(), "position": position}
                if self._send_peer(holder, pull):
                    self.pending_pulls[rid] = _PendingPull(msg, holder)
                else:
                    self._admit_new(msg, None)
                return
        self._admit_new(msg, _hit_from_plan(msg, self.spec.config) if plan is not None else None)

    def _admit_new(self, msg: dict[str, Any], hit: Hit | None) -> None:
        """``hit=None``: the engine computes the hit from what this instance can reach itself."""
        prompt = np.asarray(msg["prompt"], dtype=np.int32)
        self.engine.add_request(msg["user_id"], msg["rid"], msg["sampling_params"], prompt, hit=hit)

    def _abort(self, rid: str) -> None:
        """ADR 0012: the Entrypoint has already ended the request; release whatever this instance holds of it."""
        if self.engine.abort_request(rid)[0]:
            self.counters["aborts"] += 1
            self._aborted_awaiting_output.add(rid)
        self._prefill_reported.discard(rid)
        self.pending_pulls.pop(rid, None)
        self.decode_target.pop(rid, None)
        self.hops.pop(rid, None)
        entry = self.outbox.get(rid)
        if entry is not None:
            if entry.awaiting_answer:
                entry.abort_pending = True  # a nack drops it; an ack means the receiver has it and gets its own abort
            else:
                del self.outbox[rid]
        for item in [i for i in self.inbox if i[1] == rid]:  # (source, rid, hop, blob)
            self.inbox.remove(item)
            self._send_peer(item[0], {"type": "nack", "rid": rid, "reason": "aborted"})

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
            "pending_pulls": len(self.pending_pulls),
            "threads": threading.active_count(),
            **self.counters,
        }

    # ---- loop ----

    def _process_inbox(self) -> None:
        while self.inbox:
            source, rid, hop, blob = self.inbox.popleft()
            try:
                handoff = Handoff.from_bytes(blob)
                if handoff.request_id != rid:
                    raise WireError("request id in blob does not match envelope")
                seq = self.engine.receive_handoff(handoff)
            except (WireError, ValueError) as e:
                self._send_peer(source, {"type": "nack", "rid": rid, "reason": f"invalid: {e}"})
                continue
            if seq is None:
                self._send_peer(source, {"type": "nack", "rid": rid, "reason": "no_capacity"})
                continue
            self._send_peer(source, {"type": "ack", "rid": rid})
            self.hops[rid] = hop
            self._emit({"type": "admitted", "rid": rid, "hop": hop})

    def _retry_outbox(self) -> None:
        now = time.monotonic()
        for rid, entry in list(self.outbox.items()):
            if not entry.awaiting_answer and entry.retry_at <= now:
                self._send_handoff(rid)

    def _forward_for_drain(self) -> None:
        """ADR 0013: hand every admitted, unfinished sequence to another decode instance, between steps."""
        if not self.drain_targets:
            return
        for rid in self.engine.forwardable():
            handoff = self.engine.take_for_forward(rid)
            assert handoff is not None
            target = self.drain_targets[self._drain_rr % len(self.drain_targets)]
            self._drain_rr += 1
            self.counters["forwarded"] += 1
            hop = self.hops.pop(rid) + 1
            self._emit({"type": "forwarded", "rid": rid, "to": target, "hop": hop})
            self._queue_handoff(rid, handoff, target, hop)

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
                self.decode_target.pop(o.request_id, None)  # the first token ended it: no handoff
                self.hops.pop(o.request_id, None)
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

    def _read(self, conn: Connection) -> None:
        if conn is self.ctrl:
            while not self.stopping and conn.poll():
                self._on_ctrl(recv_msg(conn))
            return
        source = next((name for name, c in self.peers.items() if c is conn), None)
        if source is None:
            return
        try:
            while source in self.peers and conn.poll():
                self._on_peer(source, recv_msg(conn))
        except (EOFError, OSError):  # a killed peer shows up as EOF or a reset, depending on unread data
            self._on_peer_dead(source)

    def run(self) -> None:
        self._emit({"type": "ready"})
        while not self.stopping and not self.ctrl_out.failed:
            for peer in [n for n, w in self.peer_out.items() if w.failed]:
                self._on_peer_dead(peer)
            if self.faults.get("crash"):
                raise RuntimeError("injected crash")  # test fault: an engine bug
            busy = self._can_work() and self._has_work()
            peers = [] if self.faults.get("deaf") else list(self.peers.values())  # test fault: leave peer data unread
            for conn in wait([self.ctrl, *peers], timeout=0 if busy else _IDLE_WAIT_S):
                assert isinstance(conn, Connection)
                self._read(conn)
            if self.stopping or not self._can_work():
                continue
            if self.paused:
                self.step_budget -= 1
            self._process_inbox()
            self._retry_outbox()
            if self.drain_targets is not None:
                self._forward_for_drain()
            if self.engine.has_unfinished_requests():
                self._report(self.engine.step())
            self._report_swa_index()


def _hit_from_plan(msg: dict[str, Any], config: MiniConfig) -> Hit:
    plan = msg["hit"]
    keys = block_hashes(msg["prompt"], config.block_size)[: plan["num_blocks"]]
    return Hit(
        plan["cached_hit_tokens"], plan["num_cached_tokens"], plan["swa_valid_start"], keys, plan["swa_position"]
    )


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


def instance_main(name: str, spec: InstanceSpec, ctrl: Connection, peers: dict[str, Connection]) -> None:
    controller = None
    try:
        controller = InstanceController(name, spec, ctrl, peers)
        controller.run()
    except Exception:
        fatal = {"type": "fatal", "error": traceback.format_exc()}
        if controller is None:
            send_msg(ctrl, fatal)
        else:  # behind any events already queued on the control link
            controller.ctrl_out.send(fatal)
            controller.ctrl_out.close()
        raise
    controller.ctrl_out.close()
