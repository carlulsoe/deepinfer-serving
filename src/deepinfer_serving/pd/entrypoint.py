"""Entrypoint: owns each request's state and token stream, computes hits (CHECK), keeps the SWA directory, routes
through the load balancers, and recovers from instance loss and decode drains (ADR 0006, 0011-0013, 0015).

It runs in the caller's process and drives any number of prefill (PROMPT) and decode (COMPLETION) instances.
"""

from __future__ import annotations

import contextlib
import enum
import math
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any

from deepinfer_serving.control.balancers import DecodeRank, PrefillGroup, route_decode, route_prefill
from deepinfer_serving.control.cost import CostModel
from deepinfer_serving.engine.engine_type import EngineType
from deepinfer_serving.engine.sampling_params import SamplingParams
from deepinfer_serving.file_store import FileKVStore
from deepinfer_serving.kvstore import BlockKey, Hit, compute_hit
from deepinfer_serving.pd.instance import InstanceSpec, instance_main
from deepinfer_serving.pd.protocol import recv_msg, send_msg


class RequestState(enum.Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    HANDED_OFF = "handed_off"
    DECODING = "decoding"
    FORWARDED = "forwarded"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (RequestState.FINISHED, RequestState.ABORTED, RequestState.FAILED)


_S = RequestState
_END = {_S.ABORTED, _S.FAILED}
# Terminal states and re-routes. Progress within an attempt goes through ``_advance`` instead.
_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    _S.QUEUED: {_S.FINISHED, *_END},  # QUEUED -> FINISHED: a re-route found nothing left to do
    _S.FINISHED: {_S.QUEUED},  # finished, but tokens were lost with the instance that sent them (ADR 0012)
    _S.PREFILLING: {_S.FINISHED, _S.QUEUED, *_END},
    _S.HANDED_OFF: {_S.QUEUED, *_END},
    _S.DECODING: {_S.FINISHED, _S.QUEUED, *_END},
    _S.FORWARDED: {_S.QUEUED, *_END},
}

Progress = tuple[int, int]
"""``(hop, arrived)``: ``(0, 1)`` prefilling; ``(h, 0)`` hop ``h`` sent (handed off for h = 1, forwarded after);
``(h, 1)`` hop ``h`` admitted, decoding. Events of one attempt come from several instances over separate links,
so they can be read out of order; progress only moves forward (ADR 0012)."""


def _next(p: Progress) -> Progress:
    return (p[0], 1) if p[1] == 0 else (p[0] + 1, 0)


def _state_at(p: Progress) -> RequestState:
    if p == (0, 1):
        return _S.PREFILLING
    if p[1] == 1:
        return _S.DECODING
    return _S.HANDED_OFF if p[0] == 1 else _S.FORWARDED


@dataclass
class Attempt:
    """One try at a request. A re-route starts a new attempt that re-prefills prompt + tokens so far (ADR 0012)."""

    rid: str
    """Id the instances see: ``<request_id>/<n>``."""
    base: int
    """Output index of this attempt's first token."""
    prefill: str
    decode: str
    """Decode instance that holds, or is about to receive, the sequence."""
    source: str | None = None
    """While forwarded: the draining decode instance that still holds the bytes."""
    cost_ns: float = 0.0
    """Predicted prefill time (the prefill LB's quota) of what this attempt computes."""
    progress: Progress = (0, 0)
    path: set[str] = field(default_factory=set)
    """Every instance that has held this attempt: where its events, and an abort, can still go."""
    aborted: bool = False
    terminal_reported: bool = False


@dataclass
class RequestRecord:
    request_id: str
    prompt: list[int]
    sampling_params: dict
    user_id: str
    submitted_at: float = field(default_factory=time.monotonic)
    state: RequestState = RequestState.QUEUED
    history: list[RequestState] = field(default_factory=lambda: [RequestState.QUEUED])
    tokens_by_index: dict[int, int] = field(default_factory=dict)
    num_output_tokens: int | None = None
    reason: str = ""
    hit: tuple[int, int, int] | None = None
    """``(cached_hit_tokens, num_cached_tokens, swa_valid_start)`` applied by the latest attempt's prefill."""
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def attempt(self) -> Attempt:
        return self.attempts[-1]

    @property
    def tokens(self) -> list[int]:
        return [self.tokens_by_index[i] for i in sorted(self.tokens_by_index)]

    def contiguous_tokens(self) -> list[int]:
        out = []
        while len(out) in self.tokens_by_index:
            out.append(self.tokens_by_index[len(out)])
        return out

    @property
    def complete(self) -> bool:
        if self.state == RequestState.FINISHED:
            return len(self.tokens_by_index) == self.num_output_tokens
        return self.state.terminal

    def depends_on(self) -> set[str]:
        """Instances whose loss would strand the current attempt. None while it waits for a route: the request has
        no attempt yet, or its latest was aborted by a re-route."""
        if not self.attempts or self.attempt.aborted:
            return set()
        a = self.attempt
        if self.state in (_S.QUEUED, _S.PREFILLING, _S.HANDED_OFF):
            return {a.prefill, a.decode}
        if self.state == _S.FORWARDED:
            return {a.decode, *([a.source] if a.source else [])}
        if self.state == _S.DECODING:
            return {a.decode}
        return set()


class InstanceFailed(RuntimeError):
    pass


class _DirectoryView:
    """What CHECK sees: the shared block tier on disk plus the SWA directory."""

    def __init__(self, blocks: FileKVStore, directory: dict[tuple[BlockKey, int], set[str]]) -> None:
        self.blocks = blocks
        self.directory = directory

    def has_block(self, key: BlockKey) -> bool:
        return self.blocks.has_block(key)

    def has_swa(self, key: BlockKey, position: int) -> bool:
        return bool(self.directory.get((key, position)))


class Entrypoint:
    def __init__(
        self,
        instances: dict[str, InstanceSpec],
        *,
        startup_timeout: float = 120.0,
        cost: CostModel | None = None,
        pull_limit_ns: float = math.inf,
    ) -> None:
        """``cost`` and ``pull_limit_ns`` configure the prefill LB (ADR 0015). Each instance is one DP group of one
        rank. With no pull limit every request is routed on arrival; with one, requests wait here until a prefill
        instance has quota left."""
        self.specs = dict(instances)
        self.prefills = [n for n, s in self.specs.items() if s.role == EngineType.PROMPT]
        self.decodes = [n for n, s in self.specs.items() if s.role == EngineType.COMPLETION]
        if not self.prefills or not self.decodes or len(self.prefills) + len(self.decodes) != len(self.specs):
            raise ValueError("need at least one PROMPT and one COMPLETION instance, and nothing else")
        configs = {s.config.fingerprint() for s in self.specs.values()}
        stores = {s.store for s in self.specs.values()}
        if len(configs) != 1 or len(stores) != 1:
            raise ValueError("all instances must share one model/cache config and one store")
        first = self.specs[self.prefills[0]]
        self.config = first.config
        self.index = first.store.open(self.config) if first.store is not None else None
        self.startup_timeout = startup_timeout
        self.cost = cost or CostModel()
        self.pull_limit_ns = pull_limit_ns
        self.pending: list[tuple[RequestRecord, str | None, str | None]] = []
        """Requests waiting for a route, oldest first, with the instances a test pinned them to."""
        self.requests: dict[str, RequestRecord] = {}
        self.directory: dict[tuple[BlockKey, int], set[str]] = {}
        self.errors: list[str] = []
        self.duplicate_terminals = 0
        self.live: list[str] = []
        self.dead: set[str] = set()
        self.draining: set[str] = set()
        self.held: dict[str, list[dict[str, Any]]] = {}
        self._started = False
        self._ctrl: dict[str, Connection] = {}
        self._procs: dict[str, Any] = {}
        self._stats: dict[tuple[str, int], dict[str, int]] = {}
        self._stats_seq = 0

    # ---- lifecycle ----

    def start(self) -> Entrypoint:
        ctx = mp.get_context("spawn")  # CUDA-safe
        names = list(self.specs)
        peers: dict[str, dict[str, Connection]] = {n: {} for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                peers[a][b], peers[b][a] = ctx.Pipe(duplex=True)
        for name in names:
            mine, theirs = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=instance_main, args=(name, self.specs[name], theirs, peers[name]), name=name, daemon=True
            )
            proc.start()
            theirs.close()
            self._ctrl[name] = mine
            self._procs[name] = proc
            self.live.append(name)
        for conns in peers.values():
            for conn in conns.values():
                conn.close()
        deadline = time.monotonic() + self.startup_timeout
        for name in names:
            try:
                if not self._ctrl[name].poll(max(deadline - time.monotonic(), 0)):
                    raise InstanceFailed(f"{name} did not start")
                self._handle(name, recv_msg(self._ctrl[name]))
            except EOFError as e:
                raise InstanceFailed(f"{name} exited during startup") from e
        self._started = True
        return self

    def close(self) -> None:
        for name in list(self.live):
            self._send(name, {"type": "shutdown"})
        for proc in self._procs.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for conn in self._ctrl.values():
            conn.close()
        self.live = []

    def __enter__(self) -> Entrypoint:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- requests ----

    def submit(
        self,
        request_id: str,
        prompt: list[int],
        sampling_params: dict,
        user_id: str = "u",
        *,
        prefill: str | None = None,
        decode: str | None = None,
    ) -> None:
        """``prefill`` / ``decode`` pin the route (tests); otherwise the balancers choose."""
        if request_id in self.requests:
            raise ValueError(f"duplicate request id {request_id}")
        record = RequestRecord(request_id, list(prompt), dict(sampling_params), user_id)
        self.requests[request_id] = record
        self._start_attempt(record, prefill, decode)

    def abort(self, request_id: str) -> None:
        """ADR 0012: the request ends here, now; every instance on the attempt's path releases what it holds."""
        record = self.requests[request_id]
        if record.state.terminal:
            return
        self._transition(record, _S.ABORTED, "entrypoint")
        record.reason = "abort"
        self.pending = [e for e in self.pending if e[0] is not record]
        if record.attempts and not record.attempt.aborted:
            self._abort_attempt(record.attempt)

    def _abort_attempt(self, attempt: Attempt) -> None:
        attempt.aborted = True
        for name in attempt.path:
            self._send(name, {"type": "abort", "rid": attempt.rid})

    def _start_attempt(self, record: RequestRecord, prefill: str | None = None, decode: str | None = None) -> None:
        done = record.contiguous_tokens()
        params = SamplingParams(**record.sampling_params)
        params.eos_id = self.config.eos_token_id
        stopped = bool(done) and not params.ignore_eos and done[-1] == self.config.eos_token_id
        if len(done) >= params.max_tokens or stopped:
            record.num_output_tokens = len(done)
            self._transition(record, _S.FINISHED, "entrypoint")
            return
        self.pending.append((record, prefill, decode))
        self._dispatch()

    def _dispatch(self) -> None:
        """Route waiting requests: CHECK each one, then the prefill LB (levels 2 and 3) and the decode LB (ADR 0015).

        A request pinned to a prefill instance skips the prefill LB. Requests stay waiting, in order, while no
        prefill instance has quota left."""
        prefills = self._routable(self.prefills)
        decodes = self._routable(self.decodes)
        if not self.pending or not prefills or not decodes:
            return
        checked = {}
        for record, _, _ in self.pending:
            prompt = record.prompt + record.contiguous_tokens()
            hit = self._hit(prompt)
            c = hit.num_cached_tokens if hit else 0
            checked[record.request_id] = (prompt, hit, self.cost.cost(len(prompt) - c, c))
        groups = [PrefillGroup(n, 0, (self._prefill_quota(n),)) for n in prefills]
        queue = [(r.request_id, checked[r.request_id][2]) for r, p, _ in self.pending if p is None]
        routes = {rid: name for rid, name, _ in route_prefill({0: queue}, groups, self.pull_limit_ns)}
        launched = set()
        for record, pin_p, pin_d in self.pending:
            prefill = pin_p or routes.get(record.request_id)
            if prefill is None:
                break
            prompt, hit, cost = checked[record.request_id]
            decode = pin_d or self._route_decode(decodes, len(prompt))
            self._launch(record, prompt, hit, cost, prefill, decode)
            launched.add(record.request_id)
        self.pending = [e for e in self.pending if e[0].request_id not in launched]

    def _route_decode(self, decodes: list[str], need: int) -> str:
        """The decode LB. When no instance has room it picks the one with the fewest tokens, and the instance's own
        admission decides (nack and retry, ADR 0005; or failure, for a request that can never fit)."""
        capacity = self.config.num_gpu_blocks * self.config.block_size
        ranks = []
        for name in decodes:
            mine = [r for r in self.requests.values() if not r.state.terminal and r.attempts]
            mine = [r for r in mine if not r.attempt.aborted and r.attempt.decode == name]
            used = sum(len(r.prompt) + len(r.tokens_by_index) for r in mine)
            ranks.append(DecodeRank(name, len(mine), used, capacity - used))
        (choice,) = route_decode(ranks, [need])
        if choice is None:
            choice = min(range(len(ranks)), key=lambda i: (ranks[i].tokens, i))
        return decodes[choice]

    def _prefill_quota(self, name: str) -> float:
        """Predicted prefill time queued on ``name``: attempts routed there that have not handed off."""
        return sum(
            r.attempt.cost_ns
            for r in self.requests.values()
            if r.state in (_S.QUEUED, _S.PREFILLING)
            and r.attempts
            and not r.attempt.aborted
            and r.attempt.prefill == name
        )

    def _launch(
        self, record: RequestRecord, prompt: list[int], hit: Hit | None, cost: float, prefill: str, decode: str
    ) -> None:
        done = record.contiguous_tokens()
        params = SamplingParams(**record.sampling_params)
        attempt = Attempt(f"{record.request_id}/{len(record.attempts)}", len(done), prefill, decode, cost_ns=cost)
        attempt.path = {prefill, decode}
        record.attempts.append(attempt)
        plan, holder = self._plan(hit, prefill)
        msg = {
            "type": "add",
            "rid": attempt.rid,
            "user_id": record.user_id,
            "prompt": prompt,
            "sampling_params": {**record.sampling_params, "max_tokens": params.max_tokens - len(done)},
            "decode": decode,
            "hit": plan,
            "swa_holder": holder,
        }
        self._send(prefill, msg)

    def _routable(self, names: list[str]) -> list[str]:
        return [n for n in names if n in self.live and n not in self.draining]

    def _hit(self, prompt: list[int]) -> Hit | None:
        """CHECK (architecture piece 2): the hit from metadata only."""
        if self.index is None:
            return None
        c = self.config
        view = _DirectoryView(self.index, self.directory)
        return compute_hit(view, prompt, block_size=c.block_size, window=c.window, backtrack=c.backtrack)

    def _plan(self, hit: Hit | None, prefill: str) -> tuple[dict[str, Any] | None, str | None]:
        """The hit as sent to ``prefill``, and who holds the SWA record it needs, preferring ``prefill``."""
        if hit is None:
            return None, None
        c = self.config
        holder = None
        if hit.swa_position is not None:
            key = hit.block_keys[hit.swa_position // c.grain - 1]
            holders = self.directory[(key, hit.swa_position)]
            holder = prefill if prefill in holders else min(holders)
        plan = {
            "cached_hit_tokens": hit.cached_hit_tokens,
            "num_cached_tokens": hit.num_cached_tokens,
            "swa_valid_start": hit.swa_valid_start,
            "num_blocks": len(hit.block_keys),
            "swa_position": hit.swa_position,
        }
        return plan, holder

    # ---- instance control ----

    def _send(self, name: str, msg: dict[str, Any]) -> None:
        if name not in self.live:
            return
        try:
            send_msg(self._ctrl[name], msg)
        except OSError:
            self._on_death(name)

    def pause(self, name: str) -> None:
        self._send(name, {"type": "pause"})

    def resume(self, name: str) -> None:
        self._send(name, {"type": "resume"})

    def step(self, name: str, n: int = 1) -> None:
        """While paused, let the instance run ``n`` loop iterations (inbox, retries, one engine step each)."""
        self._send(name, {"type": "step", "n": n})

    def set_faults(self, name: str, **faults: Any) -> None:
        self._send(name, {"type": "faults", "faults": faults})

    def stats(self, name: str, timeout: float = 30.0) -> dict[str, int]:
        self._stats_seq += 1
        key = (name, self._stats_seq)
        self._send(name, {"type": "stats", "seq": self._stats_seq})
        self.wait_for(lambda: key in self._stats, timeout)
        return self._stats.pop(key)

    def hold_events(self, name: str) -> None:
        """Fault injection: defer ``name``'s request events, so other instances' events overtake them.

        Stats replies and deaths still get through. Deferred events are lost if the instance dies."""
        self.held.setdefault(name, [])

    def release_events(self, name: str) -> None:
        for msg in self.held.pop(name, []):
            self._handle(name, msg)

    def kill(self, name: str) -> None:
        """Fault injection: kill the instance's process outright, then recover as for any instance loss."""
        self._procs[name].kill()
        self._procs[name].join(timeout=10)
        self._on_death(name)

    def drain(self, name: str) -> None:
        """ADR 0013: route nothing new to ``name`` and forward its sequences to the other decode instances."""
        assert name in self.decodes and name in self.live
        self.draining.add(name)
        targets = [n for n in self.decodes if n in self.live and n not in self.draining]
        self._send(name, {"type": "drain", "targets": targets})

    def drained(self, name: str) -> bool:
        if any(name in r.depends_on() for r in self.requests.values() if not r.state.terminal):
            return False
        s = self.stats(name)
        return s["unfinished"] == 0 and s["outbox"] == 0 and s["inbox"] == 0

    def retire(self, name: str, timeout: float = 60.0) -> None:
        """Wait for a draining instance to empty, then stop it. Nothing is re-routed."""
        assert name in self.draining
        self.wait_for(lambda: self.drained(name), timeout)
        self._send(name, {"type": "shutdown"})
        self._procs[name].join(timeout=10)
        self.live.remove(name)
        self._forget_instance(name)

    # ---- event loop ----

    def poll(self, timeout: float = 0.0) -> None:
        by_conn = {self._ctrl[n]: n for n in self.live}
        for conn in wait(list(by_conn), timeout=timeout):
            assert isinstance(conn, Connection)
            name = by_conn[conn]
            try:
                while name in self.live and conn.poll():
                    self._handle(name, recv_msg(conn))
            except (EOFError, OSError):
                self._on_death(name)
        self._dispatch()  # progress frees quota and decode room

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

    def _on_death(self, name: str) -> None:
        """ADR 0012: drop the instance everywhere, then re-route every request that depended on it."""
        if name not in self.live:
            return
        self.live.remove(name)
        self.dead.add(name)
        self.held.pop(name, None)  # never read: lost with the instance
        self.draining.discard(name)
        self._forget_instance(name)
        for other in list(self.live):
            self._send(other, {"type": "peer_dead", "name": name})
        for record in self.requests.values():
            if self._lost_tokens(record) or (not record.state.terminal and name in record.depends_on()):
                self._reroute(record)

    def _lost_tokens(self, record: RequestRecord) -> bool:
        """Finished, missing tokens, and an instance that could have sent them is dead: they will never arrive."""
        return record.state == _S.FINISHED and not record.complete and bool(record.attempt.path & self.dead)

    def _reroute(self, record: RequestRecord) -> None:
        """New attempt from the first missing token; tokens it repeats must match (ADR 0012)."""
        self._abort_attempt(record.attempt)
        self._transition(record, _S.QUEUED, "entrypoint")
        self._start_attempt(record)

    def _forget_instance(self, name: str) -> None:
        with contextlib.suppress(OSError):
            self._ctrl[name].close()
        for entry in list(self.directory):
            self._drop_holder(entry, name)

    def _drop_holder(self, entry: tuple[BlockKey, int], name: str) -> None:
        holders = self.directory.get(entry)
        if holders is not None:
            holders.discard(name)
            if not holders:
                del self.directory[entry]

    def _handle(self, name: str, msg: dict[str, Any]) -> None:
        kind = msg["type"]
        if kind == "ready":
            return
        if kind == "fatal":
            if not self._started:
                raise InstanceFailed(f"{name} crashed during startup:\n{msg['error']}")
            self.errors.append(f"{name} crashed:\n{msg['error']}")  # kept, so tests still fail on it
            self._on_death(name)  # its requests are recovered like any instance loss
            return
        if kind == "stats":
            self._stats[(name, msg["seq"])] = {k: v for k, v in msg.items() if k not in ("type", "seq")}
            return
        if name in self.held:
            self.held[name].append(msg)
            return
        if kind == "swa_index":
            for key, position in msg["added"]:
                self.directory.setdefault((bytes.fromhex(key), position), set()).add(name)
            for key, position in msg["removed"]:
                self._drop_holder((bytes.fromhex(key), position), name)
            return
        request_id, _, n = msg["rid"].rpartition("/")
        record = self.requests.get(request_id)
        if record is None or not n.isdigit() or int(n) >= len(record.attempts):
            self.errors.append(f"{name}: event {kind} for unknown attempt {msg['rid']}")
            return
        attempt = record.attempts[int(n)]
        stale = attempt.aborted or attempt is not record.attempt or record.state in _END
        if kind == "admitted" and (stale or record.state.terminal):
            self._send(name, {"type": "abort", "rid": attempt.rid})  # it took resources somewhere: release them
            return
        if stale:
            return
        if kind == "prefilling":  # its data counts even if other instances' events overtook it
            record.hit = (msg["hit"][0], msg["hit"][1], msg["hit"][2])
        if kind in ("admitted", "forwarded"):  # who held it, whatever order the events are read in
            attempt.path.add(name if kind == "admitted" else msg["to"])
            if self._lost_tokens(record):
                self._reroute(record)
                return
        if record.state.terminal and kind != "token":
            return  # a finished attempt's tokens can still arrive late from another instance's link
        if kind == "token":
            index = attempt.base + msg["index"]
            if record.tokens_by_index.get(index, msg["token"]) != msg["token"]:
                self.errors.append(f"{name}: token {index} of {request_id} differs from an earlier attempt")
            record.tokens_by_index[index] = msg["token"]
        elif kind == "prefilling":
            self._advance(record, (0, 1))
        elif kind == "handed_off":
            self._advance(record, (1, 0))
        elif kind == "admitted":
            if self._advance(record, (msg["hop"], 1)):
                attempt.decode, attempt.source = name, None
        elif kind == "forwarded":
            if self._advance(record, (msg["hop"], 0)):
                attempt.source, attempt.decode = name, msg["to"]
                if msg["to"] not in self.live:  # the target's death was read first
                    self._reroute(record)
        elif kind == "terminal":
            if attempt.terminal_reported:
                self.duplicate_terminals += 1
                self.errors.append(f"{name}: second terminal {msg['state']} for {attempt.rid}")
                return
            attempt.terminal_reported = True
            record.reason = msg["reason"]
            record.num_output_tokens = attempt.base + msg["num_output_tokens"]
            self._transition(record, RequestState(msg["state"]), name)
            if self._lost_tokens(record):  # a holder died earlier with tokens still unread
                self._reroute(record)
        else:
            self.errors.append(f"{name}: unknown event {kind}")

    def _advance(self, record: RequestRecord, to: Progress) -> bool:
        """Move the current attempt forward to ``to``, filling in states whose events have not been read yet."""
        attempt = record.attempt
        if to <= attempt.progress:
            return False  # already past it: a late event
        while attempt.progress < to:
            attempt.progress = _next(attempt.progress)
            record.state = _state_at(attempt.progress)
            record.history.append(record.state)
        return True

    def _transition(self, record: RequestRecord, new: RequestState, by: str) -> None:
        if new not in _TRANSITIONS.get(record.state, set()):
            self.errors.append(f"{by}: invalid transition {record.state.value} -> {new.value} for {record.request_id}")
            return
        record.state = new
        record.history.append(new)
