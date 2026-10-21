"""Autoscaler: ``reconcile(spec, observed, telemetry) -> actions``, a pure function (architecture piece 5,
ADR 0016). The simulator applies the actions today; a cluster backend can apply the same ones later.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

PREFILL, DECODE = "prefill", "decode"


@dataclass(frozen=True)
class RoleSpec:
    instance_gpus: int
    min_instances: int
    max_instances: int
    capacity_per_gpu: float
    """What one GPU sustains on the role's scaling signal: uncached tokens/s for prefill, running requests for
    decode."""
    target_utilization: float = 0.7
    max_step: int = 1
    """At most this many scale-ups or drains per role per cooldown."""
    cooldown_s: float = 60.0


@dataclass(frozen=True)
class AutoscalerSpec:
    prefill: RoleSpec
    decode: RoleSpec
    warm_gpus: int = 0
    """Spare GPUs kept ready, so a scale-up starts in ``warm_startup_s`` instead of a cold start."""
    no_scale_down_above: float = 0.8
    drain_grace_s: float = 60.0
    backlog_s: float = 30.0
    """Prefill sizes for clearing its queued uncached tokens within this long, on top of the arrival rate."""

    def role(self, role: str) -> RoleSpec:
        return self.prefill if role == PREFILL else self.decode


@dataclass(frozen=True)
class InstanceView:
    name: str
    role: str
    gpus: int
    state: str
    """``starting``, ``serving``, ``draining`` or ``dead``."""
    since_s: float
    """When the instance entered ``state``."""
    active: int = 0
    """Requests on the instance."""
    healthy_gpus: int = 0
    """For a dead instance: the GPUs it gets back on restart."""


@dataclass(frozen=True)
class Observed:
    now_s: float
    instances: tuple[InstanceView, ...]
    warm_gpus: int = 0
    """Ready plus provisioning."""
    last_scale_s: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Telemetry:
    uncached_tok_s: float
    """Arrival rate of uncached prompt tokens, over a window."""
    queued_uncached: int
    """Uncached tokens waiting for prefill or being prefilled."""
    running_requests: int
    """Requests in decode or waiting for a decode slot."""


def demand(spec: AutoscalerSpec, telemetry: Telemetry, role: str) -> float:
    """The role's scaling signal: uncached tokens/s for prefill, running requests for decode [DS]."""
    if role == PREFILL:
        return telemetry.uncached_tok_s + telemetry.queued_uncached / spec.backlog_s
    return float(telemetry.running_requests)


@dataclass(frozen=True)
class ScaleUp:
    role: str


@dataclass(frozen=True)
class Drain:
    name: str


@dataclass(frozen=True)
class Forward:
    """Move a draining decode instance's remaining requests, with their KV, to other decode instances."""

    name: str


@dataclass(frozen=True)
class Replace:
    """Restart a dead instance with EP re-formed at ``gpus``."""

    name: str
    gpus: int


@dataclass(frozen=True)
class Provision:
    gpus: int


@dataclass(frozen=True)
class Release:
    gpus: int


Action = ScaleUp | Drain | Forward | Replace | Provision | Release


def utilization(spec: AutoscalerSpec, observed: Observed, telemetry: Telemetry, role: str) -> float:
    """Quota utilization: the role's demand over what its serving instances sustain."""
    d = demand(spec, telemetry, role)
    gpus = sum(i.gpus for i in observed.instances if i.role == role and i.state == "serving")
    if gpus == 0:
        return math.inf if d > 0 else 0.0
    return d / (gpus * spec.role(role).capacity_per_gpu)


def reconcile(spec: AutoscalerSpec, observed: Observed, telemetry: Telemetry) -> list[Action]:
    actions: list[Action] = []
    for role in (PREFILL, DECODE):
        rs = spec.role(role)
        mine = [i for i in observed.instances if i.role == role]
        dead = [i for i in mine if i.state == "dead" and i.healthy_gpus > 0]
        actions += [Replace(i.name, i.healthy_gpus) for i in dead]
        n = sum(1 for i in mine if i.state in ("starting", "serving")) + len(dead)
        per_instance = rs.capacity_per_gpu * rs.instance_gpus * rs.target_utilization
        desired = min(rs.max_instances, max(rs.min_instances, math.ceil(demand(spec, telemetry, role) / per_instance)))
        cooled = observed.now_s - observed.last_scale_s.get(role, -math.inf) >= rs.cooldown_s
        if cooled and desired > n:
            actions += [ScaleUp(role)] * min(rs.max_step, desired - n)
        elif (
            cooled
            and desired < n
            and not any(i.state == "starting" for i in mine)  # their capacity is not measured yet
            and not dead
            and utilization(spec, observed, telemetry, role) <= spec.no_scale_down_above
        ):
            serving = sorted((i for i in mine if i.state == "serving"), key=lambda i: (i.active, i.name))
            actions += [Drain(i.name) for i in serving[: min(rs.max_step, n - desired)]]
    for i in observed.instances:
        if (
            i.role == DECODE
            and i.state == "draining"
            and i.active > 0
            and observed.now_s - i.since_s >= spec.drain_grace_s
        ):
            actions.append(Forward(i.name))
    if observed.warm_gpus < spec.warm_gpus:
        actions.append(Provision(spec.warm_gpus - observed.warm_gpus))
    elif observed.warm_gpus > spec.warm_gpus:
        actions.append(Release(observed.warm_gpus - spec.warm_gpus))
    return actions
