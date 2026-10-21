"""Tier-0 discrete-event simulator (roadmap S1, ADR 0014).

It replays a trace through vision, prefill and decode instances whose step times come from a ``CostModel``, with
the balancers of ``balancers.py`` (or a baseline) routing, and optionally the autoscaler reconciling. Time is in ns.

Model, per instance:

- An instance is one DP group: ``gpus`` ranks that step together, because MoE layers synchronise them every step.
  A step lasts as long as its slowest rank.
- Prefill: each rank takes its queue FIFO and fills a ``quota_limit_ns`` budget with solved chunks (level 4).
- Decode: each running request gets one token per step.
- A vision GPU encodes one image at a time, FIFO.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from deepinfer_serving.control import autoscaler as asc
from deepinfer_serving.control.balancers import (
    BUCKETS,
    DecodeRank,
    PrefillGroup,
    affinity,
    assign_buckets,
    bucket_of,
    route_decode,
    route_prefill,
    route_vision,
)
from deepinfer_serving.control.cost import CostModel
from deepinfer_serving.control.trace import Request

S = 1e9
POLICIES = ("deepseek", "round_robin", "affinity")


@dataclass(frozen=True)
class Failure:
    at_s: float
    instance: str
    """An instance name, or a role (``prefill`` / ``decode``) for that role's busiest serving instance."""
    healthy_gpus: int = 0
    """GPUs the instance gets back on restart (for ``Replace``)."""


@dataclass
class SimConfig:
    prefill: list[int]
    """DP ranks (GPUs) of each initial prefill instance."""
    decode: list[int]
    cost: CostModel = field(default_factory=CostModel)
    vision: list[int] = field(default_factory=list)
    """GPUs of each vision-encoder machine."""
    vision_ns_per_patch: float = 20_000.0
    policy: str = "deepseek"
    """``deepseek`` (balancers.py), or a baseline: ``round_robin`` or ``affinity`` (hash of the session)."""
    buckets: tuple[int, ...] = BUCKETS
    bucket_round_s: float = 10.0
    pull_limit_ns: float | None = None
    """Prefill pull threshold per rank; default two step budgets."""
    failures: list[Failure] = field(default_factory=list)
    autoscaler: asc.AutoscalerSpec | None = None
    reconcile_s: float = 10.0
    telemetry_window_s: float = 60.0
    """The autoscaler sees the uncached-token arrival rate averaged over this long."""
    startup_s: float = 300.0
    warm_startup_s: float = 30.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")


@dataclass(eq=False)
class _Job:
    req: Request
    computed: int
    """Tokens whose KV the current prefill has: the hit, then each chunk."""
    generated: int = 0
    """Output tokens emitted so far."""

    @property
    def context(self) -> int:
        """Tokens the current prefill must cover: the prompt plus what was emitted before a re-route."""
        return self.req.input_len + self.generated

    @property
    def remaining(self) -> int:
        return self.context - self.computed


@dataclass
class RequestStats:
    arrival_ns: float
    first_token_ns: float | None = None
    finish_ns: float | None = None
    last_token_ns: float | None = None
    gaps_ns: list[float] = field(default_factory=list)
    completions: int = 0
    reroutes: int = 0
    vision_ns: float = 0.0
    """Time from arrival until the vision encoder returned (0 for text)."""

    @property
    def ttft_s(self) -> float:
        assert self.first_token_ns is not None
        return (self.first_token_ns - self.arrival_ns) / S

    @property
    def mean_tbt_s(self) -> float:
        return sum(self.gaps_ns) / len(self.gaps_ns) / S if self.gaps_ns else 0.0


@dataclass(eq=False)
class _Instance:
    name: str
    role: str
    gpus: int
    state: str
    since_ns: float
    ranks: list[Any]
    bucket: int = 0
    busy: bool = False
    epoch: int = 0
    replaced: bool = False
    healthy_gpus: int = 0
    forward: bool = False
    step: list[tuple[int, _Job, int]] = field(default_factory=list)
    """In-flight step: ``(rank, job, chunk)`` for prefill; ``(rank, job, 0)`` for decode."""

    def jobs(self) -> Iterable[_Job]:
        for r in self.ranks:
            yield from r


@dataclass(frozen=True)
class ReconcileRecord:
    time_s: float
    utilization: dict[str, float]
    """Quota utilization per role, as the autoscaler saw it."""
    actions: list[asc.Action]
    active: dict[str, int]
    """Starting plus serving instances per role, after the actions."""


@dataclass
class SimResult:
    requests: dict[str, RequestStats]
    prefill_balance: float
    """Σ over prefill steps of mean rank time / Σ of max rank time: 1.0 is perfectly balanced."""
    decode_balance: float
    gpu_hours: float
    end_s: float
    budget_overruns: int
    """Prefill rank-steps over the quota budget with more than one chunk (an invariant violation)."""
    kv_overruns: int
    """Decode admissions into a rank without room (an invariant violation)."""
    reconciles: list[ReconcileRecord]
    instances: dict[str, tuple[str, int]]
    """Every instance that existed: ``(final state, GPUs)``."""

    def ttft(self) -> list[float]:
        return sorted(r.ttft_s for r in self.requests.values())

    def tbt(self) -> list[float]:
        return sorted(g / S for r in self.requests.values() for g in r.gaps_ns)

    def slo_attainment(self, ttft_s: float, tbt_s: float) -> float:
        ok = sum(1 for r in self.requests.values() if r.ttft_s <= ttft_s and r.mean_tbt_s <= tbt_s)
        return ok / len(self.requests)


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile of sorted ``xs``."""
    return xs[min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))]


class Simulator:
    def __init__(self, config: SimConfig, trace: list[Request]) -> None:
        self.cfg = config
        self.cost = config.cost
        self.trace = sorted(trace, key=lambda r: (r.arrival_s, r.id))
        if len({r.id for r in trace}) != len(trace):
            raise ValueError("duplicate request ids")
        if any(r.patches for r in trace) and not config.vision:
            raise ValueError("trace has images but no vision machines")
        self.rng = random.Random(config.seed)
        self.now = 0.0
        self._events: list[tuple[float, int, int, str, Any]] = []
        self._kicked = False
        self._dirty: set[_Instance] = set()
        self._prefill_dirty = False
        self._seq = itertools.count()
        self.instances: dict[str, _Instance] = {}
        self._names = {asc.PREFILL: itertools.count(), asc.DECODE: itertools.count()}
        self.stats: dict[str, RequestStats] = {}
        self.unfinished = len(trace)
        self.bucket_queues: dict[int, list[_Job]] = {b: [] for b in range(len(config.buckets) + 1)}
        self.unrouted: list[_Job] = []
        """Baselines: jobs waiting for any serving prefill rank."""
        self.decode_pending: list[_Job] = []
        self._rr = {asc.PREFILL: 0, asc.DECODE: 0, "vision": 0}
        self.vision = [[deque() for _ in range(g)] for g in config.vision]
        self.vision_load = [[0] * g for g in config.vision]
        self.vision_busy = [[False] * g for g in config.vision]
        self.round_demand = [0.0] * (len(config.buckets) + 1)
        self.arrivals: deque[tuple[float, int]] = deque()
        """``(time, uncached tokens)`` of arrivals inside the telemetry window."""
        self.balance = {asc.PREFILL: [0.0, 0.0], asc.DECODE: [0.0, 0.0]}
        self.gpu_ns = 0.0
        self.warm_ready = config.autoscaler.warm_gpus if config.autoscaler else 0
        self.warm_provisioning = 0
        self.last_scale_s: dict[str, float] = {}
        self.budget_overruns = 0
        self.kv_overruns = 0
        self.reconciles: list[ReconcileRecord] = []
        for gpus in config.prefill:
            self._new_instance(asc.PREFILL, gpus, "serving")
        for gpus in config.decode:
            self._new_instance(asc.DECODE, gpus, "serving")

    # ---- event loop ----

    def _at(self, t: float, kind: str, data: Any = None, *, last: bool = False) -> None:
        heapq.heappush(self._events, (t, int(last), next(self._seq), kind, data))

    def _kick(self, *instances: _Instance, prefill: bool = False) -> None:
        """Route and start steps once every event at this instant has been handled, so requests that arrive
        together are scheduled together. ``instances`` may be able to start a step; ``prefill`` means prefill
        routing may have changed. Decode routing runs whenever handoffs wait."""
        self._dirty.update(instances)
        self._prefill_dirty |= prefill
        if not self._kicked:
            self._kicked = True
            self._at(self.now, "kick", last=True)

    def _on_kick(self, _: Any) -> None:
        self._kicked = False
        if self._prefill_dirty:
            self._prefill_dirty = False
            self._dirty.update(self._serving(asc.PREFILL))
            self._dispatch_prefill()
        if self.decode_pending:
            self._dirty.update(self._serving(asc.DECODE))
            self._dispatch_decode()
        for inst in sorted(self._dirty, key=lambda i: i.name):
            if inst.state in ("serving", "draining"):
                if inst.role == asc.PREFILL:
                    self._start_prefill_step(inst)
                else:
                    self._start_decode_step(inst)
        self._dirty.clear()

    def run(self, max_s: float | None = None) -> SimResult:
        """Run until every request finishes; fail if that takes past ``max_s`` (default: a day after the last
        arrival)."""
        if max_s is None:
            max_s = (self.trace[-1].arrival_s if self.trace else 0.0) + 86_400
        for r in self.trace:
            self._at(r.arrival_s * S, "arrive", r)
        for f in self.cfg.failures:
            self._at(f.at_s * S, "fail", f)
        if self.cfg.policy == "deepseek":
            self._at(self.cfg.bucket_round_s * S, "round")
        if self.cfg.autoscaler:
            self._at(self.cfg.reconcile_s * S, "reconcile")
        while self._events and self.unfinished and self._events[0][0] <= max_s * S:
            t, _, _, kind, data = heapq.heappop(self._events)
            self.gpu_ns += self._held_gpus() * (t - self.now)
            self.now = t
            getattr(self, f"_on_{kind}")(data)
        if self.unfinished:
            raise RuntimeError(f"{self.unfinished} requests unfinished at {self.now / S:.1f}s")
        pb, db = self.balance[asc.PREFILL], self.balance[asc.DECODE]
        return SimResult(
            self.stats,
            pb[0] / pb[1] if pb[1] else 1.0,
            db[0] / db[1] if db[1] else 1.0,
            self.gpu_ns / S / 3600,
            self.now / S,
            self.budget_overruns,
            self.kv_overruns,
            self.reconciles,
            {i.name: (i.state, i.gpus) for i in self.instances.values()},
        )

    def _held_gpus(self) -> int:
        held = sum(i.gpus for i in self.instances.values() if i.state in ("starting", "serving", "draining"))
        return held + self.warm_ready + self.warm_provisioning

    # ---- instances ----

    def _new_instance(self, role: str, gpus: int, state: str) -> _Instance:
        name = f"{role[0]}{next(self._names[role])}"
        ranks: list[Any] = [deque() for _ in range(gpus)] if role == asc.PREFILL else [[] for _ in range(gpus)]
        inst = _Instance(name, role, gpus, state, self.now, ranks)
        self.instances[name] = inst
        return inst

    def _serving(self, role: str) -> list[_Instance]:
        return [i for i in self.instances.values() if i.role == role and i.state == "serving"]

    def _set_state(self, inst: _Instance, state: str) -> None:
        inst.state = state
        inst.since_ns = self.now

    def _on_ready(self, inst: _Instance) -> None:
        if inst.state != "starting":
            return
        self._set_state(inst, "serving")
        self._kick(inst, prefill=True)

    def _on_fail(self, f: Failure) -> None:
        if f.instance in (asc.PREFILL, asc.DECODE):
            serving = sorted(self._serving(f.instance), key=lambda i: (-sum(1 for _ in i.jobs()), i.name))
            if not serving:
                return
            inst = serving[0]
        else:
            inst = self.instances[f.instance]
        if inst.state in ("dead", "retired"):
            return
        self._set_state(inst, "dead")
        inst.healthy_gpus = f.healthy_gpus
        inst.epoch += 1
        inst.busy = False
        inst.step = []
        lost = list(inst.jobs())
        for r in inst.ranks:
            r.clear()
        for job in lost:  # re-route: re-prefill the prompt plus what was emitted (ADR 0012)
            self.stats[job.req.id].reroutes += 1
            job.computed = job.req.hit_len
            self._enqueue_prefill(job)
        self._kick(prefill=True)

    def _retire_if_drained(self, inst: _Instance) -> None:
        if inst.state == "draining" and not inst.busy and not any(inst.jobs()):
            self._set_state(inst, "retired")

    # ---- arrival and vision ----

    def _on_arrive(self, req: Request) -> None:
        self.stats[req.id] = RequestStats(req.arrival_s * S)
        job = _Job(req, req.hit_len)
        self.arrivals.append((self.now, req.uncached))
        self.round_demand[bucket_of(req.input_len, self.cfg.buckets)] += self.cost.cost(req.uncached, req.hit_len)
        if not req.patches:
            self._enqueue_prefill(job)
            self._kick(prefill=True)
            return
        if self.cfg.policy == "deepseek":
            m, g = route_vision(self.vision_load, self.rng)
        else:
            flat = [(m, g) for m, gs in enumerate(self.vision) for g in range(len(gs))]
            if self.cfg.policy == "round_robin":
                m, g = flat[self._rr["vision"] % len(flat)]
                self._rr["vision"] += 1
            else:
                m, g = flat[affinity(req.session, len(flat))]
        self.vision[m][g].append(job)
        self.vision_load[m][g] += req.patches
        self._start_vision(m, g)

    def _start_vision(self, m: int, g: int) -> None:
        if self.vision_busy[m][g] or not self.vision[m][g]:
            return
        job = self.vision[m][g][0]
        self.vision_busy[m][g] = True
        self._at(self.now + job.req.patches * self.cfg.vision_ns_per_patch, "vision_done", (m, g))

    def _on_vision_done(self, mg: tuple[int, int]) -> None:
        m, g = mg
        job = self.vision[m][g].popleft()
        self.vision_load[m][g] -= job.req.patches
        self.vision_busy[m][g] = False
        self.stats[job.req.id].vision_ns = self.now - job.req.arrival_s * S
        self._start_vision(m, g)
        self._enqueue_prefill(job)
        self._kick(prefill=True)

    # ---- prefill ----

    def _enqueue_prefill(self, job: _Job) -> None:
        if self.cfg.policy == "deepseek":
            self.bucket_queues[bucket_of(job.context, self.cfg.buckets)].append(job)
        else:
            self.unrouted.append(job)

    def _job_cost(self, job: _Job) -> float:
        return self.cost.cost(job.remaining, job.computed)

    def _dispatch_prefill(self) -> None:
        serving = self._serving(asc.PREFILL)
        if not serving or not (self.unrouted or any(self.bucket_queues.values())):
            return
        if self.cfg.policy == "deepseek":
            groups = [
                PrefillGroup(i.name, i.bucket, tuple(sum(self._job_cost(j) for j in r) for r in i.ranks))
                for i in serving
            ]
            queues = {b: [(j.req.id, self._job_cost(j)) for j in q] for b, q in self.bucket_queues.items() if q}
            limit = self.cfg.pull_limit_ns or 2 * self.cost.quota_limit_ns
            waiting = {j.req.id: j for q in self.bucket_queues.values() for j in q}
            taken: set[str] = set()
            for rid, name, rank in route_prefill(queues, groups, limit):
                self.instances[name].ranks[rank].append(waiting[rid])
                taken.add(rid)
            for b, q in self.bucket_queues.items():
                self.bucket_queues[b] = [j for j in q if j.req.id not in taken]
        else:
            flat = [(i, r) for i in sorted(serving, key=lambda i: i.name) for r in range(i.gpus)]
            for job in self.unrouted:
                if self.cfg.policy == "round_robin":
                    i, r = flat[self._rr[asc.PREFILL] % len(flat)]
                    self._rr[asc.PREFILL] += 1
                else:
                    i, r = flat[affinity(job.req.session, len(flat))]
                i.ranks[r].append(job)
            self.unrouted = []

    def _start_prefill_step(self, inst: _Instance) -> None:
        if inst.busy or not any(inst.ranks):
            return
        plan: list[tuple[int, _Job, int]] = []
        times = []
        for r, queue in enumerate(inst.ranks):
            budget, spent, chunks = self.cost.quota_limit_ns, 0.0, 0
            for job in queue:
                q = self.cost.solve_chunk(budget - spent, job.computed, job.remaining)
                if q == 0 and chunks == 0:
                    q = 1  # an empty rank always makes progress
                if q == 0:
                    break
                plan.append((r, job, q))
                spent += self.cost.cost(q, job.computed)
                chunks += 1
                if q < job.remaining:
                    break  # FIFO: the head must finish before the next request starts
            if spent > self.cost.quota_limit_ns and chunks > 1:
                self.budget_overruns += 1
            times.append(spent)
        self._account(asc.PREFILL, times)
        inst.busy = True
        inst.step = plan
        self._at(self.now + self.cost.prefill_step_ns + max(times), "prefill_step", (inst, inst.epoch))

    def _on_prefill_step(self, data: tuple[_Instance, int]) -> None:
        inst, epoch = data
        if epoch != inst.epoch:
            return
        inst.busy = False
        for r, job, q in inst.step:
            job.computed += q
            if job.remaining == 0:
                inst.ranks[r].remove(job)
                self._emit(job)
                if job.generated < job.req.output_len:
                    self._at(self.now + self.cost.transfer_ns_per_token * job.context, "handoff", job)
        inst.step = []
        self._retire_if_drained(inst)
        self._kick(inst, prefill=True)

    def _emit(self, job: _Job) -> None:
        st = self.stats[job.req.id]
        job.generated += 1
        if st.first_token_ns is None:
            st.first_token_ns = self.now
        else:
            assert st.last_token_ns is not None
            st.gaps_ns.append(self.now - st.last_token_ns)
        st.last_token_ns = self.now
        if job.generated == job.req.output_len:
            st.finish_ns = self.now
            st.completions += 1
            self.unfinished -= 1

    # ---- decode ----

    def _on_handoff(self, job: _Job) -> None:
        self.decode_pending.append(job)
        self._kick()

    def _rank_tokens(self, rank: list[_Job]) -> int:
        return sum(j.context for j in rank)

    def _dispatch_decode(self) -> None:
        targets = [(i, r) for i in sorted(self._serving(asc.DECODE), key=lambda i: i.name) for r in range(i.gpus)]
        if not targets or not self.decode_pending:
            return
        cap = self.cost.decode_kv_capacity
        if self.cfg.policy == "deepseek":
            ranks = [
                DecodeRank(i.name, len(i.ranks[r]), t := self._rank_tokens(i.ranks[r]), cap - t) for i, r in targets
            ]
            choice = route_decode(ranks, [j.context for j in self.decode_pending])
        else:
            choice = []
            free = [cap - self._rank_tokens(i.ranks[r]) for i, r in targets]
            for job in self.decode_pending:
                if self.cfg.policy == "round_robin":
                    start = self._rr[asc.DECODE]
                    self._rr[asc.DECODE] += 1
                else:
                    start = affinity(job.req.session, len(targets))
                order = [(start + k) % len(targets) for k in range(len(targets))]
                pick = next((k for k in order if free[k] >= job.context), None)
                if pick is not None:
                    free[pick] -= job.context
                choice.append(pick)
        waiting = []
        for job, k in zip(self.decode_pending, choice, strict=True):
            if k is None:
                waiting.append(job)
                continue
            inst, r = targets[k]
            if self._rank_tokens(inst.ranks[r]) + job.context > cap:
                self.kv_overruns += 1
            inst.ranks[r].append(job)
        self.decode_pending = waiting

    def _start_decode_step(self, inst: _Instance) -> None:
        if inst.busy or not any(inst.ranks):
            return
        times = [self.cost.decode_step(len(r), self._rank_tokens(r)) for r in inst.ranks]
        self._account(asc.DECODE, times)
        inst.busy = True
        inst.step = [(r, job, 0) for r, rank in enumerate(inst.ranks) for job in rank]
        self._at(self.now + max(times), "decode_step", (inst, inst.epoch))

    def _on_decode_step(self, data: tuple[_Instance, int]) -> None:
        inst, epoch = data
        if epoch != inst.epoch:
            return
        inst.busy = False
        for r, job, _ in inst.step:
            self._emit(job)
            if job.generated == job.req.output_len:
                inst.ranks[r].remove(job)
        inst.step = []
        if inst.forward:
            self._forward(inst)
        self._retire_if_drained(inst)
        self._kick(inst)

    def _forward(self, inst: _Instance) -> None:
        """ADR 0013: hand every remaining sequence, with its KV, to the other decode instances."""
        inst.forward = False
        for r in inst.ranks:
            for job in r:
                self._at(self.now + self.cost.transfer_ns_per_token * job.context, "handoff", job)
            r.clear()

    def _account(self, role: str, times: list[float]) -> None:
        self.balance[role][0] += sum(times) / len(times)
        self.balance[role][1] += max(times)

    # ---- control ----

    def _on_round(self, _: Any) -> None:
        """Level 1: reassign prefill groups to buckets by last round's demand."""
        serving = self._serving(asc.PREFILL)
        assignment = assign_buckets({i.name: i.bucket for i in serving}, self.round_demand)
        for i in serving:
            i.bucket = assignment[i.name]
        self.round_demand = [0.0] * len(self.round_demand)
        self._at(self.now + self.cfg.bucket_round_s * S, "round")
        self._kick(prefill=True)

    def _on_reconcile(self, _: Any) -> None:
        spec = self.cfg.autoscaler
        assert spec is not None
        now_s = self.now / S
        views = tuple(
            asc.InstanceView(
                i.name,
                i.role,
                i.gpus,
                i.state,
                i.since_ns / S,
                sum(1 for _ in i.jobs()),
                i.healthy_gpus,
            )
            for i in self.instances.values()
            if i.state in ("starting", "serving", "draining") or (i.state == "dead" and not i.replaced)
        )
        running = sum(1 for i in self._serving_or_draining(asc.DECODE) for _ in i.jobs()) + len(self.decode_pending)
        window = min(self.cfg.telemetry_window_s, now_s) or self.cfg.reconcile_s
        while self.arrivals and self.arrivals[0][0] < self.now - window * S:
            self.arrivals.popleft()
        queued = sum(j.remaining for j in self._prefill_jobs())
        telemetry = asc.Telemetry(sum(n for _, n in self.arrivals) / window, queued, running)
        observed = asc.Observed(now_s, views, self.warm_ready + self.warm_provisioning, dict(self.last_scale_s))
        actions = asc.reconcile(spec, observed, telemetry)
        for action in actions:
            self._apply(spec, action)
        roles = (asc.PREFILL, asc.DECODE)
        self.reconciles.append(
            ReconcileRecord(
                now_s,
                {r: asc.utilization(spec, observed, telemetry, r) for r in roles},
                actions,
                {
                    r: sum(1 for i in self.instances.values() if i.role == r and i.state in ("starting", "serving"))
                    for r in roles
                },
            )
        )
        self._at(self.now + self.cfg.reconcile_s * S, "reconcile")

    def _prefill_jobs(self) -> Iterable[_Job]:
        yield from self.unrouted
        for q in self.bucket_queues.values():
            yield from q
        for i in self._serving_or_draining(asc.PREFILL):
            yield from i.jobs()
        for q in itertools.chain.from_iterable(self.vision):
            yield from q

    def _serving_or_draining(self, role: str) -> list[_Instance]:
        return [i for i in self.instances.values() if i.role == role and i.state in ("serving", "draining")]

    def _apply(self, spec: asc.AutoscalerSpec, action: asc.Action) -> None:
        match action:
            case asc.ScaleUp(role):
                self.last_scale_s[role] = self.now / S
                self._start_instance(role, spec.role(role).instance_gpus, allow_warm=True)
            case asc.Drain(name):
                inst = self.instances[name]
                self.last_scale_s[inst.role] = self.now / S
                self._set_state(inst, "draining")
                self._retire_if_drained(inst)
            case asc.Forward(name):
                inst = self.instances[name]
                inst.forward = True
                if not inst.busy:
                    self._forward(inst)
                    self._retire_if_drained(inst)
                    self._kick()
            case asc.Replace(name, gpus):
                old = self.instances[name]
                old.replaced = True
                self._start_instance(old.role, gpus, allow_warm=False)
            case asc.Provision(gpus):
                self.warm_provisioning += gpus
                self._at(self.now + self.cfg.startup_s * S, "warm_ready", gpus)
            case asc.Release(gpus):
                self.warm_ready -= min(gpus, self.warm_ready)

    def _start_instance(self, role: str, gpus: int, *, allow_warm: bool) -> None:
        warm = allow_warm and self.warm_ready >= gpus
        if warm:
            self.warm_ready -= gpus
        inst = self._new_instance(role, gpus, "starting")
        self._at(self.now + (self.cfg.warm_startup_s if warm else self.cfg.startup_s) * S, "ready", inst)

    def _on_warm_ready(self, gpus: int) -> None:
        self.warm_provisioning -= gpus
        self.warm_ready += gpus


def simulate(config: SimConfig, trace: list[Request], max_s: float | None = None) -> SimResult:
    return Simulator(config, trace).run(max_s)
